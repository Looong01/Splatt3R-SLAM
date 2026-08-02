---
name: splatt3r-gaussian-map
description: Diagnosis and fix for the Splatt3R-SLAM Gaussian map going "cluttered/duplicated" after a loop closure or revisit — read before touching Gaussian accumulation, SharedKeyframes, or visualization.py's GS rendering path. Contains the implemented fix (Plan A) and a documented, never-implemented fallback (Plan B) with a rollback procedure.
metadata:
  type: reference
---

# Splatt3R-SLAM: Gaussian map duplication/ghosting on revisit

## The bug this fixes

Symptom (reported 2026-07-22): with default parameters, when the camera
returns to a previously-visited viewpoint, the 3DGS map gets visibly
cluttered — the old, previously-clean reconstruction is overlaid with a
second, misaligned copy of the same geometry, and it never goes away.

### Root cause (pre-fix architecture)

MASt3R-SLAM's point-cloud map is drift-proof by construction: each
keyframe stores its point cloud in **camera space**
(`SharedKeyframes.X_canon`), and the visualizer transforms it to world
space **live, every frame**, using that keyframe's *current* `T_WC`
(`visualization.py`'s `render_pointmap()`). When the backend factor graph
(`global_opt.py: FactorGraph.solve_GN_rays/calib`) corrects a keyframe's
pose after a loop closure, the point cloud moves with it automatically.

The original Splatt3R integration did **not** follow this pattern for
Gaussians. `gaussians_to_world()` (old, now removed) baked each frame's
Gaussians into **world-space coordinates once, at append time**, using
whatever `T_WC` that frame had *at that instant* — and wrote the result
into a flat, append-only `SharedGaussians` ring buffer
(`splatt3r_slam/frame.py`, now removed) that had no way to move or delete
anything afterward. Two compounding bugs made it worse:

1. **Every tracked frame appended, not just keyframes** (`main.py`, old
   `should_append_gaussians()`). Most appended Gaussians belonged to
   non-keyframe frames whose poses are *never* part of the pose graph at
   all — even a correct re-bake mechanism could not have fixed those.
2. The `kf_id` field on the old buffer was documented as "for pruning"
   but was **never read** anywhere, and the index it stored
   (`kf_idx=len(keyframes)` in `main.py`, i.e. "the keyframe slot this
   frame *might* become") did not correspond to a real keyframe row for
   the common case where the tracked frame was not actually the one that
   became keyframe N.

Net effect: revisiting a mapped area appends a second, pose-drifted copy
of the Gaussians on top of the first, and even a perfect loop-closure
correction can never retroactively fix either copy, because nothing ever
looks at a baked Gaussian's world position again.

## Plan A — implemented (current state of this repo)

**Decision (2026-07-22): go with Plan A.** GPU has 48GB (RTX A6000);
budget analysis below shows Plan A's storage cost is ~6GB at full
resolution, so there was no reason to take Plan B's leaner-but-lossier
approach.

### Design

Mirror the point-cloud map exactly: store Gaussians in **camera space**,
per keyframe, and transform to world space **live**, using the
keyframe's current `T_WC`, every time they need to be drawn. Nothing is
ever permanently baked to world space.

- `splatt3r_slam/frame.py` — `SharedKeyframes` gained per-keyframe
  camera-space Gaussian storage, written once (never cleared) alongside
  `X_canon`/`C` in `__setitem__`:
  - `gs_means`, `gs_scales`, `gs_rotations`, `gs_sh`, `gs_opacities`,
    `gs_conf`: shape `(buffer, h*w, C)`, `share_memory_()` CUDA tensors.
  - `gs_valid`: `(buffer,)` bool, set once a keyframe's Gaussian data is
    written.
  - `get_gaussians_local(idx)`: returns the flat local-space dict, or
    `None` if not yet populated.
  - **Critical invariant**: `__setitem__` only writes/overwrites the
    `gs_*` fields when `value.gaussian_pred is not None`. `tracker.py`
    re-writes a keyframe's `X_canon` on almost every tracking step (via
    `self.keyframes[len(self.keyframes)-1] = keyframe`) to fuse pointmap
    confidence, but that `Frame` object never carries `gaussian_pred` —
    if the guard were removed, every such write would silently wipe the
    keyframe's Gaussians. This was caught and fixed during
    implementation; if you touch `__setitem__` again, keep the guard.
  - The old `SharedGaussians` class (flat, cross-process, FIFO-evicting
    accumulator) was deleted entirely — nothing needs it now.

- `splatt3r_slam/splatt3r_utils.py` — `gaussians_to_world()` was replaced
  by `bake_gaussians_world(local, img_tensor, h, w, T_WC, spatial_stride,
  depth_max_percentile, max_scale, min_confidence)`. Pure function: local
  camera-space Gaussians in, world-space Gaussians out, using whatever
  `T_WC` you pass it. Same splash-artifact filtering (depth percentile /
  max scale / min confidence) as before, just operating on stored local
  data instead of a transient `frame.gaussian_pred`.

- `splatt3r_slam/visualization.py` — `Window` owns a **live bake cache**:
  - `self._gs_world_cache`: `{kf_idx: (means_w, cov_triu, colors, opacities)}`.
  - `self._gs_cache_key`: `{kf_idx: (T_WC_snapshot, stride_used)}`.
  - `_get_world_gaussians()` (called every render frame, from
    `_render_gs_interactive`): for every live keyframe, compares its
    *current* `T_WC` (and the active `spatial_stride` GUI slider) against
    the cache key; unchanged keyframes are served from cache, changed
    ones are re-baked via `bake_gaussians_world` from their stored local
    data. **This is the actual fix** — a pose correction is picked up the
    very next render frame because the comparison naturally detects it,
    with no explicit "dirty" signal needed from the backend.
  - The `max_gaussians` GUI slider changed semantics: it used to be a
    hard cap enforced by dropping the oldest half of a flat buffer
    (FIFO). Now it's a *render budget* — `_get_world_gaussians` computes
    a uniform `budget_stride` so the total baked count across all
    keyframes stays near the budget as the map grows, floored by
    whatever `spatial_stride` the user explicitly set.
  - Reading `keyframes.T_WC[kf_idx, 0]` for the cache-key comparison, and
    the subsequent re-bake, happens inside `with self.keyframes.lock:`
    for the whole scan, to avoid tearing against the backend's concurrent
    `update_T_WCs()` writes.

- `main.py` — drastically simplified. The whole per-frame Gaussian
  accumulation dance (`should_append_gaussians`, `gaussians_to_world`
  calls, `shared_gaussians.append`, FIFO/throttle bookkeeping) is gone.
  Storing a keyframe's Gaussians is now just a side effect of the
  existing `keyframes.append(frame)` / `keyframes[idx] = frame` calls,
  because `frame.gaussian_pred` is already populated by
  `splatt3r_inference_mono` (INIT) or inside `tracker.track()` via
  `splatt3r_match_asymmetric` (TRACKING) *before* those calls run. Model
  loading was moved before `SharedKeyframes` construction because the
  buffer needs `gs_sh_dim = model.encoder.sh_degree` at construction
  time. The per-frame novel-view PNG preview (`--render-gaussians`,
  `splatt3r_render(...)`) is untouched — it never went through the map
  accumulator.

### Memory budget (why 48GB is comfortable)

Per-keyframe local storage at full resolution (h≈384, w≈512,
`sh_degree=1` → 15 floats/point: means3+scales3+rot4+sh(3·1)+opac1+conf1
= 60 bytes/point):

```
512 keyframe buffer × 196608 px/keyframe × 60 bytes ≈ 6.0 GB
```

That's on top of the existing `SharedKeyframes` buffers (X/C/feat/pos/img
≈ a few GB) and the model itself. The live world-space bake cache in the
viz process is much smaller since it's built with the GUI's
`spatial_stride`/`max_gaussians` budget applied (typically hundreds of
MB, not GB), and only re-baked for keyframes whose pose actually changed
this frame — steady state cost is near zero.

**If you're re-verifying this budget**: `sh_degree` comes from the
loaded checkpoint (`model.encoder.sh_degree`, currently 1 for
`checkpoints/epoch=19-step=1200.ckpt`). A different checkpoint with a
higher SH degree scales the `gs_sh` tensor's memory linearly — recompute
before assuming headroom.

### Density defaults, and a real crash found while raising them (2026-07-22)

The map initially looked "sparse/blurry" at the shipped defaults
(`spatial_stride=4`, `max_gaussians=4M`, `gs_resolution_scale=0.5`).
Raising density is safe in principle (48GB headroom), but pushing
`spatial_stride` all the way to `1` (full per-pixel density, no
subsampling) reproducibly **crashed the interactive renderer** with:

```
torch.AcceleratorError: CUDA error: an illegal memory access was encountered
```

surfacing asynchronously at some unrelated later CUDA sync point (per
PyTorch's own warning, the real faulting kernel launch is earlier and
un-synchronized) — in both observed crashes it happened inside/soon after
`_render_gs_interactive()`'s call into the vendored, "modified"
`diff_gaussian_rasterization` CUDA extension
(`thirdparty/diff-gaussian-rasterization-modified`). Two distinct issues
were found and only one is fixed:

1. **Fixed, in `bake_gaussians_world()`** (`splatt3r_slam/splatt3r_utils.py`):
   `build_covariance()`'s quaternion→rotation-matrix conversion
   (`splatt3r_core/utils/geometry.py: quaternion_to_matrix`) divides by
   `‖q‖² + eps` rather than truly normalizing `q` first. A near-zero-norm
   raw quaternion from the network (rare, but far more likely to be
   sampled at all once `spatial_stride` drops from 4 to 1 — 16x more
   pixels sampled per keyframe) produces a wildly non-orthonormal
   "rotation" with huge-but-finite entries, and therefore a
   huge-but-finite world-space covariance that `isfinite()` checks do
   *not* catch. Fixed by explicitly normalizing `rotations` before
   `build_covariance()`, plus an `isfinite()` safety filter on the final
   `means_world`/`cov_triu` as defense-in-depth. **This did not fully
   fix the crash** — it's a real bug worth keeping fixed regardless, but
   not the (or not the only) cause of what was observed.
2. **Not fixed — a scaling ceiling in the vendored rasterizer itself.**
   Even after (1), `spatial_stride=1` crashed again, this time almost
   immediately (before the *second* keyframe was even created — i.e. a
   single keyframe's ~196608 raw per-pixel Gaussians, filtered down by
   depth/scale/confidence/opacity, was already enough). This points to
   an internal fixed-size buffer or indexing assumption in the "modified"
   CUDA rasterizer that isn't sized for that many Gaussians in one
   `rasterizer(...)` call. Bisected empirically:
   `spatial_stride=2` (~49152 raw points/keyframe, 4x the original
   default's density) ran stable for 4.5+ minutes and well past keyframe
   10 (where the unfixed crash first appeared) with no issue.
   `spatial_stride=1` was not re-tried after finding (1) didn't fully
   resolve it — not worth the GPU time to re-confirm; treat it as
   **known-unsafe** until someone either patches the vendored rasterizer
   or adds an explicit chunking/batching step (split `means3D` etc. into
   sub-batches under some safe N and composite the results) before
   calling it with more than a few hundred thousand Gaussians in one
   shot.

   **IMPORTANT — this is a mitigation, not a fix.** Re-tested later
   (2026-07-22, see the splatt3r-color-consistency skill) on
   `freiburg1_floor` at the shipped `spatial_stride=2` default: the exact
   same `illegal memory access` crash reappeared, this time by
   frame ~120-160 — well short of the 4.5-minute/frame-880+ stability
   seen on `freiburg1_desk`. So `spatial_stride=2` reduces how often the
   vendored rasterizer chokes, it does not guarantee it won't. Notably,
   the crash happened on a run with `dataset.normalize_exposure: False`
   (the pre-fix baseline for the color-consistency A/B test); the
   equivalent run with normalization *on* survived past frame 880 on the
   same sequence. That's one data point each way, not proof, but it's
   consistent with a plausible mechanism: an unnormalized, very
   over/under-exposed input frame is more out-of-distribution for the
   network, making it more likely to emit a degenerate/extreme Gaussian
   (huge scale, near-zero-norm quaternion) that trips the rasterizer even
   after the normalization fix in (1). If you're chasing this crash
   further, treat `dataset.normalize_exposure` as a variable worth
   controlling for, not just a color fix.

**UPDATE — the shipped `spatial_stride` default is now `1`, not `2`.**
Changed at the user's explicit request ("把这个默认的参数调回1,我可以在
gui上面拉进度条调成或大或小的值") -- they accept the crash risk at
stride=1 because the GUI slider lets them raise it live if a scene
misbehaves. `main.py`'s `--spatial-stride` help text carries the warning
inline. The rest of this section's reasoning about *why* higher strides
are safer still stands; only the default changed. Note also that the
crash's actual root cause was later traced (in the LoRA-training work --
see the splatt3r-lora-finetuning skill) to `scale_invariant=True`
combined with a hardcoded `near=0.1` applying an unconditional 100x
covariance magnification, which is a much better explanation than raw
Gaussian count: reducing count via stride only ever delayed the crash,
never eliminated it.

**Other shipped defaults as of this change**:
`max_gaussians=16*1024*1024` (was `4*1024*1024`, not implicated in the
crash — it only matters once total keyframes×points approaches this
budget, which didn't happen before the stride=1 crashes did),
`gs_resolution_scale=1.0` (was `0.5`, not implicated either — it affects
output framebuffer size, not rasterizer input count). Also restored a
`min_opacity=0.3` filter in `bake_gaussians_world()` that existed in the
pre-Plan-A `SharedGaussians.append()` but was accidentally dropped during
the Plan A rewrite (opacity wasn't filtered at all for a while — another
contributor to the "blurry" look, since near-transparent splash
Gaussians were being composited).

`spatial_stride=1` is already the maximum density this pipeline can
produce (every pixel sampled), so "more density than stride=2" is no
longer the live question — the risk now runs the other way: lowering
the GUI slider back toward 1 raises the per-call Gaussian count toward
the rasterizer's apparent ceiling. Don't treat stride=1 as safe just
because it's the shipped default. If you want it to be reliably safe,
either (a) patch/rebuild the vendored rasterizer to confirm
and fix its actual limit, or (b) add batching in
`_get_world_gaussians()`/`_render_gs_interactive()` so no single
`rasterizer(...)` call ever exceeds whatever N is proven safe, and
composite multiple batches' output. Re-run the bisection in a background
process with `setsid` (so you can reliably kill the whole process tree
afterward — `timeout` alone does **not** kill a Python multiprocessing
app's spawned children, they'll leak GPU memory as orphans) and watch
for `Traceback`/`illegal memory` in the log before declaring a stride
safe.

**Second-order bug this uncovered: gaps between subsampled Gaussians.**
After landing `spatial_stride=2` + `gs_resolution_scale=1.0`, the user
reported the map looked "messy" — dense stippling/dot pattern on every
surface (visible in a screenshot they shared, using the point-cloud-style
"gs_render" panel next to the fullscreen interactive view). Root cause:
`bake_gaussians_world()` subsamples every s-th pixel but never grew each
kept Gaussian's footprint to match, so at `stride=2` neighbouring kept
splats are twice as far apart as their (unchanged) radius — they stop
overlapping and the surface reads as a halftone dot pattern instead of
continuous coverage. This was always present (visible faintly in the
very first screenshot too) but was hidden by rendering at half
resolution before; going to `gs_resolution_scale=1.0` made it sharp and
obvious. **Fixed** in `bake_gaussians_world()`: `scales = scales * s`
right before `build_covariance()`, applied *after* the `max_scale`
splash-artifact filter (which is about the network's raw per-pixel
prediction, not this display-only compensation). Verified visually (see
below) — the dot pattern is gone; what remains is normal Gaussian-splat
softness/blur plus some SH-color blending artifacts across keyframes,
which look like limitations of this particular checkpoint
(`epoch=19-step=1200.ckpt`, an early/lightly-trained Splatt3R model) —
present even in single-keyframe `splatt3r_render()` output that never
goes through stride/scale-inflation at all, so not a regression from
this pipeline.

**Third-order finding: the flat `scales * s` fix above disproportionately
blurs distant objects.** User-reported 2026-07-22: "far objects have
blurry edges." Diagnosis: the network's own predicted per-pixel scale
already grows with depth (a pixel covers more physical area at range,
purely from perspective; two-view depth estimation is also genuinely
less confident/constrained at range) — so far-field Gaussians already
start out larger/blurrier than near-field ones *before* any stride
compensation. Multiplying everything by a flat `s` compounds that
existing gap instead of correcting a uniform problem, making distant
surfaces disproportionately softer relative to near ones. Directly
visible in `SPLATT3R_GS_DUMP_DIR` captures on `freiburg1_room`: a
foreground laptop keyboard renders crisp while the background monitor/
wall in the same frame is noticeably soft. **Partially addressed**:
`bake_gaussians_world()` now clamps the inflated result —
`scales = torch.clamp(scales * s, max=max_scale)` — so the compensation
can't push an already-large far-field splat past the same ceiling the
splash filter already treats as "as big as legitimate gets." Near-field
small splats (well under the ceiling) still get the full inflation they
need to close gaps; far-field ones stop growing once clamped. This
curbs the *compounding* specifically, but the underlying softness is
still substantially the network's own depth-uncertainty behavior —
confirmed present (though less pronounced) even in single-keyframe
`splatt3r_render()` output that never goes through stride/scale
inflation at all. Don't expect this clamp to make far objects sharp;
it only stops this pipeline's own compensation from making them worse.
Fully fixing the underlying softness is a model-capability question —
see the splatt3r-lora-finetuning skill for what that would take (the
splatt3r-color-consistency skill's old training/LoRA section is
superseded).

**Crash-frequency reassessment (2026-07-22) — read this before trusting
"spatial_stride=2 is safe."** Chasing the blur fix above surfaced the
`illegal memory access` crash *again*, twice, on both `freiburg1_room`
and `freiburg1_floor`, both times within the first ~120-160 frames —
far short of the 4.5-minute/frame-880+ stability seen earlier on
`freiburg1_desk`. **This crash is a live, recurring risk, not a rare
edge case fully mitigated by `stride=2` — and the shipped default has
since been changed to the even more aggressive `stride=1` (see the
UPDATE above), so on sequences that trigger it, treat it as
when-not-if.** It seems
data/sequence-dependent (some sequences trigger it fast, some don't in
the timeframes tested) rather than purely a function of accumulated
Gaussian count, which weakens the original "just a per-call count
ceiling" theory somewhat — there may be more than one contributing
factor. `CUDA_LAUNCH_BLOCKING=1` (which PyTorch's own error message
suggests for a clearer trace) was tried once to get a synchronous
traceback and made things *worse* for debugging: the viz process died
with a silent native crash (no Python traceback at all, likely a
hardware-level trap from an actual out-of-bounds write inside the
vendored CUDA extension) rather than a catchable exception. **This
environment does have `compute-sanitizer` and `cuda-gdb` installed**
(`/usr/local/cuda/bin/`) — that's the right next tool (e.g.
`compute-sanitizer --tool memcheck --target-processes all python3
main.py ...`, expect very heavy slowdown and check it actually
instruments the spawned viz subprocess, not just the top-level
process) to get an exact array/kernel/index for the invalid access, but
that's a deeper, slower debugging session than anything tried so far —
flag to the user before spending a lot of time/GPU on it, don't just
launch into it.

**How to actually SEE the interactive map without a screenshot tool**:
this desktop runs Wayland (GNOME/Mutter). X11 tools (`scrot`, `xdotool`)
cannot see native GLFW/Wayland surfaces — they either capture solid
black or the wrong window (e.g. matched a VS Code window whose title
happened to contain the repo name). GNOME's own D-Bus screenshot API
(`org.gnome.Shell.Screenshot`) also refuses non-interactive/unauthorized
callers. The actual solution (the user's suggestion): `Window.render()`
in `visualization.py` already computes `gs_img_np`, the exact array
drawn as the fullscreen GS background quad — `_maybe_dump_gs_debug_frame()`
(called right after) dumps it to PNG every `SPLATT3R_GS_DUMP_EVERY`
(default 30) frames when `SPLATT3R_GS_DUMP_DIR` is set, e.g.:
```
SPLATT3R_GS_DUMP_DIR=/tmp/gsdump SPLATT3R_GS_DUMP_EVERY=15 \
  python3 main.py --dataset <seq>
```
then read the PNGs directly (an image-capable agent can `Read` them; a
human can just open the folder). Off by default, zero cost when unset.
This is the general pattern for verifying *any* future change to the
interactive render — don't fight Wayland, dump frames to disk.

### How to verify the fix is working

1. Run interactively (`python main.py --dataset <tum-seq>`), let the
   camera map an area, move away, then return to the same viewpoint.
   Before the fix: a visible second/offset copy of the geometry appears
   and never resolves. After the fix: revisiting may still add Gaussians
   from the new pass (that's expected — no deduplication was added,
   see the future ideas below), but once the backend logs a loop-closure
   correction (`"Database retrieval ..."` / `"RELOCALIZING against kf
   ..."` in the console), the earlier keyframes' Gaussians should visibly
   snap to align with the corrected trajectory within a frame or two,
   not stay permanently offset.
2. Unit-level sanity check (no GPU display needed) — write a synthetic
   keyframe with a known `gaussian_pred`, call
   `SharedKeyframes.get_gaussians_local()` + `bake_gaussians_world()`
   once at the original `T_WC`, then call `SharedKeyframes.update_T_WCs()`
   with a different pose and re-bake: the world-space output should move
   by exactly the pose delta while `get_gaussians_local()`'s output stays
   bit-identical (camera-space data is never touched by pose changes).
   This was verified during implementation; re-run it after any change
   to `frame.py`/`splatt3r_utils.py`'s Gaussian code.
3. Watch VRAM (`nvidia-smi` / `nvidia-smi -l 1`) over a long sequence
   with many keyframes (freiburg1_room, freiburg1_360 are good stress
   tests — lots of revisits). If usage climbs toward the 48GB ceiling,
   see "If Plan A doesn't fit" below.

### Known limitations / possible follow-ups (not implemented)

- No deduplication of overlapping geometry from independent passes over
  the same area — Plan A fixes *misalignment after correction*, not
  *redundant density*. If clutter from sheer over-density (not
  misalignment) becomes a problem, consider voxel-hash deduplication or
  opacity-based culling of overlapping Gaussians as a separate follow-up.
- The render-budget `budget_stride` is a single uniform value across all
  keyframes; it doesn't prioritize keyframes currently in view over
  off-screen ones. Fine at current scale; revisit if `max_gaussians`
  needs to go much higher.

## Plan B — documented fallback (NOT implemented)

Kept here in full so a future session can implement it from scratch
without re-deriving the diagnosis, if Plan A turns out to exceed the
VRAM budget or otherwise misbehaves.

### When to switch to Plan B

- VRAM from the `gs_*` `SharedKeyframes` buffers (see budget above)
  becomes a problem — e.g. a much higher-`sh_degree` checkpoint, a much
  larger keyframe `buffer`, or running alongside other GPU-heavy
  processes on the same card.
- The live re-bake-on-pose-change cost in `_get_world_gaussians()`
  becomes a frame-rate problem with very large keyframe counts (unlikely
  at `buffer=512`, but possible if that cap is raised a lot).

### Design

Don't store per-pixel local-space Gaussian params at all. Keep baking to
world space once, like the original code — but fix the two bugs that
made that unrecoverable:

1. **Only bake at real keyframe insertion**, tagged with the keyframe's
   *actual* index (`len(keyframes) - 1` right after `keyframes.append()`
   returns), never a speculative "next" index. This alone fixes the
   `kf_id` mislabeling bug and ensures every baked Gaussian is
   associated with a keyframe whose pose *can* later be corrected.
2. **Make the `kf_id` field on the flat world buffer load-bearing.**
   Alongside each keyframe's baked Gaussians, store the `T_WC` that was
   used to bake them (one small Sim3 per keyframe, not per Gaussian —
   e.g. a `(buffer, Sim3.embedded_dim)` tensor parallel to
   `SharedKeyframes.T_WC`).
3. **On every backend pose correction**, instead of (or in addition to)
   `SharedKeyframes.update_T_WCs(T_WCs, idx)` writing the new poses,
   apply a **rigid delta-correction** to the flat world buffer's
   Gaussians whose `kf_id` matches each corrected keyframe:
   ```
   ΔT = T_new · T_old⁻¹        (as 4×4: ΔR, Δt)
   mean'  = ΔR · mean + Δt
   cov'   = ΔR · cov · ΔRᵀ
   ```
   `T_old` is the per-keyframe baked-pose snapshot from (2); after
   applying the delta, overwrite that snapshot with `T_new`. This is a
   cheap batched tensor op (rotate/translate only the rows whose `kf_id`
   matches), not a re-render — no local-space data needs to be kept
   around, so the memory cost is back to ~the size of the old
   `SharedGaussians` buffer (hundreds of MB), not the ~6GB Plan A local
   store.
4. This delta-correction needs a call site: the natural place is right
   after `FactorGraph.solve_GN_rays()` / `solve_GN_calib()` call
   `self.frames.update_T_WCs(...)` in `global_opt.py` — but that runs in
   the **backend process**, and the Gaussian world buffer lives in the
   **viz process** (or wherever Plan A currently keeps it) — so either:
   (a) move the world buffer back to a genuinely cross-process
   `SharedGaussians`-style structure (with a `kf_id` column, `share_memory_()`
   tensors) so the backend can apply the delta directly, or
   (b) have the viz process detect the pose change itself (same
   comparison Plan A already does in `_get_world_gaussians`) and apply
   the delta client-side instead of the backend doing it. (b) is
   probably less invasive to retrofit from Plan A's cache-comparison
   logic.
5. Keep the "only real keyframes append" fix from (1) regardless of
   which of 4(a)/4(b) you pick — it's required either way.

### Rollback procedure (Plan A → Plan B)

Plan A's changes are confined to exactly four files: `main.py`,
`splatt3r_slam/frame.py`, `splatt3r_slam/splatt3r_utils.py`,
`splatt3r_slam/visualization.py`. To roll back:

1. If Plan A's commit(s) haven't been pushed/built on top of: `git log
   --oneline -- main.py splatt3r_slam/frame.py splatt3r_slam/splatt3r_utils.py
   splatt3r_slam/visualization.py` to find the commit(s), then `git
   revert <sha>` (or `git checkout <pre-plan-A-sha> -- <those 4 files>`
   if you want a silent restore instead of a revert commit). Confirm with
   the user before doing either — these are destructive/history-editing
   with respect to Plan A's work.
2. Re-read this skill's "Plan B — documented fallback" section above and
   implement it fresh against the restored (pre-Plan-A) code — Plan B was
   never coded, only designed, so there is no Plan B commit to revert
   *to*. Steps 1–5 above are the full spec.
3. Update this file's "Decision" line at the top of the Plan A section
   once the switch is made, so the next session doesn't re-read a stale
   decision.
