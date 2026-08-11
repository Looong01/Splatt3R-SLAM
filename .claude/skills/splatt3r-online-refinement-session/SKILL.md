---
name: splatt3r-online-refinement-session
description: Complete session log (2026-08-02 → 08-03) of the online-refinement campaign executed by Kimi — every user request, action, measured number, decision rationale, subagent delegation, and bug hunt, written for review. Covers (f)(g)(b′)(e) go/no-gos, stage 4 refiner integration, cross-GPU deployment + two vendor bug fixes, stage 5 no-build, route D decoder-LoRA, normalize_exposure closure, Plan 2 negative result, GNOME repair, all-family GUI evaluation, and the GUI/ply tooling added at the end. Read this BEFORE auditing docs/ or the §15/§16 sections of splatt3r-finetuning-experiments.
metadata:
  type: reference
---

# Online-refinement session log (Kimi, 2026-08-02 → 08-03)

Written for review: every exchange of the session, what was done, what it
measured, and WHY each decision was taken. This is a faithful reconstruction
of the working record (requests, actions, tool outputs, decision rationales),
not a verbatim token dump. Canonical numbers live in
`splatt3r-finetuning-experiments` §15/§16 and `docs/`; this file is the
narrative that produced them.

---

## 0. Starting state and plan formation

**Request**: read the Claude-written skills, state what remains and how to
continue.

Actions: read all five skills (finetuning-experiments fully — §13.14 "What
remains, in the order review ranked it" and §14 "Not yet done" were the
operative lists), checked git/GPU/process state.

Conclusion reported to the user:
- Queue: **(f) causal replay → (b′) faithful seam test → (g) GPU contention →
  (a) process integration → (c) densification lifecycle → (d)(e) metrics &
  buffer ablation**; stage 5 (pose phases) optional/last.
- Recommendation: do (f) first — it is the cheap go/no-go that can invalidate
  all downstream engineering ("if the gain evaporates here, (a)-(e) are
  wasted work", §13.14).

## 1. Re-anchoring result review (user-reported, before any new run)

User reported fixing the subsample-index measurement bug
(`main.py:320 dataset.subsample(2)` → analysis scripts mismatched keyframes
to GT by up to 1.8 m/126°) and the corrected result: re-anchoring to GT is
worth only +0.07/+0.13 dB — "the pose gate is closed at map-build time; the
lever is the fusion stage itself". Asked: is this right?

Verification done (not just agreement): read `scripts/refine_local.py:201-205`
— supervision poses are the SLAM **estimates**, only held-out scoring uses
mapped GT. Answer: yes, with one attribution refinement — the experiment
bounds *placement*'s share at ~0.1 dB; the unrecoverable remainder is shared
between cluster internals and supervision poses, BOTH products of
tracking/fusion, so the conclusion (lever = fusion stage) stands. Also noted
(b′) must stay in the queue before (g).

## 2. (f) causal replay + (g) GPU contention (user: "现在做这两件事,把gpu0,1都用起来")

### Built
- `scripts/refine_causal.py` — two arms in one harness (posthoc control vs
  causal replay), differing ONLY in supervision/injection timing: same map,
  same 50-view set, same 3000-iter budget, same loss/LRs, extent held from
  the full map in both arms (to avoid an LR confound — flagged as a
  run_refiner design note). Causal arm: keyframes injected at trajectory
  timestamps, views unlocked at dataset timestamps (round-END availability —
  round-start was caught dropping every view after the last keyframe; fixed
  before real runs), iterations ∝ elapsed time, Adam rebuilt per injection
  via `refiner._optimizer_for` (the online mechanism). Posthoc arm
  replicated refine_local stage 2 to 0.0015 dB (harness validation).
- `main.py --frame-timing CSV` (per-frame track/backend-wait/iter ms),
  `config/rt_calib.yaml` (multithreaded = deployable), 
  `scripts/exp_gpu_contention.sh` + `scripts/summarize_frame_timing.py`.

### Measured
- (f) desk: causal ≥ posthoc at every budget — 120 iters 13.6511 (init
  12.4416), 500 iters 14.3942 vs 14.3223, 1000 iters 14.4747 vs 14.3516,
  3000 iters 14.4123 vs 14.3747. **GO at every budget.**
- (g) desk 306 frames: baseline track p50 101 ms / 8.0 fps; same-GPU load
  206 ms / 4.5 fps (+104%); cross-GPU 103 ms / 7.5 fps (+6%); ATE
  0.017158±1e-6 all arms. **The slowdown is entirely same-GPU SM
  contention; second GPU is the deployment answer.**
- Throughput calibration: ~3-4 it/s at 1.86M Gaussians → strict-real-time
  budget ≈ 120 iters (hence the 120-iter arm's importance).

### Decision rationale (recorded for review)
- Both arms same harness because code drift between harnesses had already
  burned the project repeatedly (§13.14's five bugs).
- GPU0 = (f), GPU1 = (g) arms interleaved so the baseline stayed clean and
  the cross-GPU arm doubled as (f)'s background.

## 3. (b′) faithful seam test, (e) sampling ablation, dedup ablation
(user: "继续,这种实验性质的任务不要问我,直接执行")

### (b′) — `refine_local.py --perturb-mode block`
Faithfulness fix: `FramePoseLog.record()` anchors each tracked frame to the
latest keyframe (`evaluate.py:86`); the test now moves supervision views
WITH their anchors when a block of keyframes is corrected. One Sim3 to the
second half of the trajectory (loop-closure-like); three-way held-out split
(only-low / only-high / seam) + median-overlap two-way.
- desk: three-way split degenerates (every keyframe covers every view — all
  50 "seam"); 2-way split works. Faithful: injection −1.32 dB, partial
  recovery to −0.88, NO tearing (monotone post-injection recovery; overlap
  heals +0.81 vs single +0.13). Frozen control: FULL recovery — i.e.
  without anchor-carried supervision the optimizer undoes a correction
  within ~500 iters. **FramePoseLog's value measured end-to-end.**
- room replication: split differentiates (4/5/41); low-only class bit-stable
  at injection (instrument self-check); same verdict.

### (e) — `refine_causal.py --sampling {uniform,recent,mixed}` at 500 iters
uniform 14.3965 > mixed 14.2096 > recent-only 14.0648; the gap lives in the
early half (forgetting). The predicted assimilation upside of a recent
window does NOT exist at desk scale. Consequence shipped:
`SupervisionFrames.recent_frac` default 0.7 → 0.3 (reservoir-dominant).

### Dedup ablation — `refine_local.py --dedup-voxel`
Cross-keyframe NN spacing measured first (p10 ≈ 11 mm) to size the voxel.
Dose-response: 5 mm (14.2% deleted) leaves the single-vs-overlap gap at
+0.63; 10 mm (28.4%) collapses it to +0.18. **The overlap anomaly is two
shells at ~10 mm** — and 10 mm became the (c) lifecycle parameter.

## 4. (a) stage 4 integration (user: "把剩下所有事情完成")

### Built
- `splatt3r_slam/refiner.py` rewrite: `SupervisionFrames` (CPU shared uint8
  + anchor-relative poses, composed at sample time — the (b′) semantics;
  CPU because it is the only cross-process channel that does not pin both
  processes to one GPU), `run_refiner` (duty-cycle EMA throttle, anchor-
  carried sampling, L1+0.2·DSSIM, Adam moment carry-over, final save),
  `RefinedMapSnapshot` (double-buffered 13-float CPU snapshot + version
  counter), `dedup_by_voxel` + `_optimizer_subset`.
- `main.py`: `--refiner` family of flags, process spawn, supervision offers
  in the TRACKING/INIT/keyframe branches, teardown ordering (refiner joins
  AFTER backend drain so the save sees final poses), stale-file cleanup.
- `evaluate.py`: `FramePoseLog.record()` returns (anchor, rel) for the
  supervision offer.

### Measured (desk, two configs)
- eval_calib (deterministic): ATE 0.016975 vs baseline 0.0170 — the
  pre-registered failure condition did not fire. Map 10.6598 → 11.9015
  (+1.24 dB) at duty 0.25, only 56 steps (duty throttle ≈ 0.7 it/s).
- rt_calib + `--frame-timing`: 5.3 fps vs 8.0 baseline.

### Cross-GPU + two vendor bugs (the hard part)
- First cross-GPU attempt crashed the refiner. Probe BEFORE fix (deliberate
  methodology): lietorch Sim3 compose on cuda:1 returns ZERO translation
  silently → workaround: all lietorch pose math stays on the shared buffers'
  device, only 4×4 matrices cross.
- Second failure: rasterizer illegal memory access on cuda:1. Probe:
  `refine_causal --device cuda:1` standalone reproduced it. Root cause:
  `rasterize_points.cu` had NO device guard — buffers allocate on current
  device, kernels launch on it, mixing cuda:0 buffers with cuda:1 pointers.
  Fix: `c10::cuda::OptionalCUDAGuard(at::device_of(means3D))` at forward /
  backward / markVisible, rebuilt in place, probe now matches cuda:0.
  **Noted: the historical "illegal memory access at high Gaussian counts"
  crashes are likely this same class.**
- Result: duty 1.0 on GPU1 → 225 steps, track p50 102 ms (baseline 101),
  ATE bit-identical, refined 12.8128/0.5411 vs baked 10.6598/0.5557:
  **+2.15 dB at +1% tracker latency. This is the deployable config.**

### (c) dedup lifecycle + stage 5 decision
- Dedup fires past threshold (desk: 1.85M → 1.34M, −27.6%, ATE unchanged).
  Quality −0.53 dB when it fires at the tail with ~6 steps of recovery
  budget — honestly recorded as a MAP-SIZE control for long sequences, not
  a quality feature.
- Stage 5 (online pose phases): **deliberately NOT built**, three measured
  reasons (real-error recovery ~11%; error baked inside clusters —
  re-anchoring +0.13 dB; second-writer risk to the pose graph). Re-open
  condition written into §15.9.

## 5. Loose-end closures (user: "一个一个都完成")

- **normalize_exposure**: discovered ALREADY closed — `SequenceExposureLock`
  wired 07-28 00:27, ~22 h before all 40-epoch production heads (log ctimes
  07-28 22:17+); live probe of the lock (desk gain [0.856,0.911,0.891]);
  §9's SLAM validation used exactly this combination. §14 text corrected.
  Lesson: verify the premise before scheduling the experiment.
- **Route D (decoder-only LoRA)**: new `scripts/exp_dec_lora.py`, route B's
  protocol + LoRA r=8/α=16 on decoder Linears only. First run failed on a
  regex that matched `attn.dropout` (peft: Linear only) — fixed to an exact
  leaf-module pattern. Result: best +0.37 dB (epoch 1), then collapse
  (−1.61 dB at epoch 5, scale_p99 12× climb, gnorm 41.7). **The mechanism
  is "any upstream adaptation", not the encoder; route B is the endpoint;
  retrieval refit closed twice over.**
- **viz consumer side**: `Window._read_refined` (version-locked read,
  upload cache, baked fallback) + `run_visualization(refined_snapshot=...)` +
  main.py wiring. Implemented, explicitly marked UNVERIFIED (headless box).
- **Plan 2 colour harmonization**: `scripts/color_harmonize.py` (causal
  voxel-overlap least-squares gain fit) — measured **NEGATIVE** (−0.57 dB,
  +0.013 lpips); fitted gains are too strong and monotone (they absorb head
  bias / view-direction shading / 10 mm pairing error, not exposure).
  Disposition: not wired into the bake path; color-consistency skill
  updated.

## 6. Documentation pass (user: "完成之后所有内容写入skill以便于写论文")

- Skill: §15.1–15.12 (each experiment), §16 paper-ready summary (C1–C9,
  ceiling table, five measurement traps, explicit non-claims).
- README.md: 11 missing CLI flags documented, online-refinement section,
  outputs table, `--lora` marked negative, spatial-stride warning updated
  for the device-guard fix.
- setup.py: dead `cores` key removed, CUDA-13 arch comment.
- pyproject.toml: 0.1.0 → 0.2.0 with change note; 25 deps audited.
- Verified: pyproject parses, `main.py --help` lists 27 args, setup.py
  compiles.

## 7. Full-record synthesis (user: "阅读所有skill和logs,总结")

Three explore subagents delegated in parallel (context economy): ①
finetuning-experiments §1–13 (2813 lines), ② the other four skills, ③
logs/ + git chronology. Their digests, plus this session's firsthand record,
were written to `docs/`:
- `docs/README.md` (master summary: five-act timeline, deliverables,
  verdicts table, methodology, open boundaries)
- `docs/chronicle-logs-git.md` (day-by-day, checkpoints map, top-10 logs)
- `docs/finetuning-experiments-part1.md` (§1–13 digest + 10 key numbers)
- `docs/online-refinement-campaign.md` (this campaign)
- `docs/skills-digest-four.md` (four topical skills)

## 8. Git repair (user-pasted VS Code failure)

User's bulk commit included rasterizer crash dumps
(`splatt3r_core/snapshot_bw.dump` 641 MB etc., ~790 MB total) → GitHub
pre-receive rejection. Fixed: `git rm --cached` the three dumps, added
`snapshot_*.dump` to .gitignore, amended the unpushed commit (now
`ecd5182`), committed the docs edits separately (`f82163c`), pushed
successfully. Dumps remain on local disk.

## 9. GNOME repair + all-family GUI evaluation (user: new task)

### GNOME
gnome-shell crashed 08-02 03:37 (disposed-object JS errors → Xwayland
broken pipe). Repaired: enabled `AutomaticLogin=share-v5` in
/etc/gdm3/custom.conf, `systemctl restart gdm3`, verified gnome-shell
running; EGL probe OK (RTX A6000). X channel: Xwayland :0 with the
per-login `.mutter-Xwaylandauth.*` file (name rotates each login — the
first "no window" failure later came from a stale auth path).

### Evaluation (`scripts/eval_online_all.sh`, GUI on the physical display)
9 sequences × 2 arms (per-family head alone / + refiner on GPU1), all with
GUI: ATE identical arm-to-arm on every sequence; refined vs baked (n=100):

```
desk     12.4191 → 13.7520  (+1.33)      room   11.1462 → 11.4506 (+0.30, lpips worse)
360      11.9973 → 12.1413  (+0.14, lpips worse)
chess    13.4400 → 13.7913  (+0.35)      office 11.6431 → 11.7399 (+0.10)
pumpkin  14.1470 → 14.9121  (+0.77)
MH_01    10.1719 → 11.6729  (+1.50)      V1_01 12.0593 → 12.6859 (+0.63)
cables_1 NVS unusable¹                  plant_1 20.4216 → 22.2200 (+1.80, lpips better)
```

¹ cables_1 is a ~0.5 m desktop scene; ATE 0.1165 m is ~15-20% of scene
extent, so NVS scoring is dominated by alignment error (confirmed by
dumping renders — camera looks into empty space). plant_1 substituted as
the room-scale ETH3D representative.

Latency: refiner costs ~+7-10% frame time cross-GPU with GUI on (6.0-7.6
fps working points). VRAM peaks: GPU0 19.8-39.6 GB (SLAM), GPU1 2.8-14 GB
(refiner). Report: `docs/online-eval-all-families.md`.

### Bugs found and fixed during the eval
- `RefinedMapSnapshot.publish` overflowed on >5M-Gaussian maps AND ran
  before the save → four sequences lost their refined.ply. Fixed:
  stride-subsample on overflow (display channel never errors) + save BEFORE
  publish. Reran and recovered all four.
- EuRoC GT symlink one level short (`../../` → `../../../`) → rescored.

## 10. Interactive-session tooling (end of session, user-driven)

- **keyframe_stride pre-existing break**: `main.py` passed
  `keyframe_stride=` to `run_visualization`, which never had the parameter
  — added during the 07-31 `--map-keyframe-stride` work, invisible because
  every run since was `--no-viz`. Fixed the whole chain (signature →
  Window → bake loop). This is why the user's first GUI attempt showed no
  window (viz process died instantly; the SLAM pipeline kept running
  headless).
- **ply auto-colour**: 3DGS plies carry `f_dc_*`, not RGB — MeshLab showed
  a yellow default. Added `gaussians_to_ply_element()` to
  `gaussian_ply_codec.py` (appends uchar red/green/blue; all original
  properties intact) and switched all three writers
  (`evaluate.save_gaussian_map`, `refiner.save_refined_map`,
  `scripts/refine_gaussian_map.py`) to it. Roundtrip test PASS; the "no
  f_dc clamp" semantics in refine_gaussian_map preserved (clamp only
  touches the uchar display columns). Standalone converter:
  `scripts/ply_dc_to_rgb.py`.
- **GUI frame capture**: `--save-gs-view DIR [--gs-view-stride N]` saves
  the interactive 3DGS render to PNGs (atomic tmp+rename writes), built on
  the pre-existing `SPLATT3R_GS_DUMP_*` env hook. Smoke-tested on 360.

---

## 11. Subagent usage record

| when | type | brief | output landed |
|---|---|---|---|
| §7 | explore ×3 (parallel) | read §1–13 / four skills / logs+git | docs/*.md digests |
| all experiments | none | all runs driven directly (GPU state too stateful to delegate safely) | — |

## 12. Working principles observed (for the reviewer)

1. Cheap go/no-go before engineering ((f) before (a)) — the kill-chain
   order came from §13.14's own review ranking.
2. Both arms in one harness for every comparison (code drift between
   harnesses was this project's most expensive bug class).
3. Probe before fix (lietorch/rasterizer bugs were reproduced standalone
   before touching code).
4. Never report a number without its instrument's error model (seeded
   draws; 0.086 dB nondeterminism floor; Sim3-fit instrument error ≈ 60%
   of the pose effect).
5. Negative results are deliverables when measured: stage 5 no-build,
   Plan 2 negative, route D collapse, encoder-LoRA falsification.
6. Honest boundaries: two sequences/one seed for online numbers; viz
   consumer never drawn (headless); dedup lifecycle unproven on long
   sequences; ~7.6 fps is the operating point, not 30 fps.

## 13. Session artifacts (all in the repo)

Code: `splatt3r_slam/refiner.py` (rewritten), `main.py` (+--refiner family,
--frame-timing, --save-gs-view), `splatt3r_slam/evaluate.py`,
`splatt3r_slam/visualization.py`, `splatt3r_slam/gaussian_ply_codec.py`,
`thirdparty/.../rasterize_points.cu` (device guard), `scripts/
refine_causal.py, refine_local.py (block/dedup), exp_dec_lora.py,
color_harmonize.py, exp_gpu_contention.sh, summarize_frame_timing.py,
eval_online_all.sh, ply_dc_to_rgb.py`, `config/rt_calib.yaml`.
Docs: skill §15.1–15.12 + §16, `docs/` five files.
Commits: `ecd5182` (campaign), `f82163c` (docs/packaging), pushed to main.
Uncommitted at session end: publish-overflow fix, keyframe_stride fix,
RGB-column writers, ply_dc_to_rgb.py, eval_online_all.sh, docs/.
