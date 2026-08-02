---
name: splatt3r-color-consistency
description: Diagnosis and fix for the same physical surface (e.g. floor) getting visibly different-coloured Gaussians when revisited from a different pass/angle in Splatt3R-SLAM's map. Also covers what training/fine-tuning (including LoRA) the underlying Splatt3R checkpoint would take if a code-level fix isn't enough. Read before touching Gaussian colour computation or considering a retrain.
metadata:
  type: reference
---

# Splatt3R-SLAM: inconsistent colour on revisited surfaces (e.g. floor)

## The bug

Reported 2026-07-22: revisiting the same physical floor area from a
different pass produces a *visibly different shade* of the same colour
(e.g. "both yellow, but the newer patch is a deeper yellow"), so the
floor reads as a patchwork of mismatched tiles rather than one uniform
surface.

### Root cause

Splatt3R computes each Gaussian's colour as `network_residual +
RGB2SH(that_frame's_raw_pixel_colour)` (see `splatt3r_core/main.py`'s
`learn_residual` logic, mirrored in
`splatt3r_slam/splatt3r_utils.py: bake_gaussians_world()`:
`sh[..., 0] = sh[..., 0] + RGB2SH(img_hwc)`). The DC/base colour of every
Gaussian is, up to a small learned residual, **literally the raw pixel
colour of the source video frame it came from**. Nowhere in the pipeline
does anything compare or reconcile colour between two independent
observations of the same physical surface — each keyframe's Gaussians
are baked once and unioned into the world-space map (see the
splatt3r-gaussian-map skill for how that union/live-rebake works). If
the source video's auto-exposure or auto-white-balance drifts between
the first and second pass over an area (very plausible for a
Kinect-recorded set like TUM RGB-D), the two observations' raw pixel
colours for the *same* physical point genuinely differ, and that
difference gets baked in permanently, side by side.

This is a different failure mode from the map-duplication/ghosting bug
(splatt3r-gaussian-map skill) and from the dot-pattern/gap bug
(also splatt3r-gaussian-map skill) — geometry can be perfectly aligned
and gap-free and this will still show up, because it's purely about
per-frame colour, not position.

## Plan 1 — implemented: causal per-frame exposure normalization

`splatt3r_slam/image.py: normalize_exposure()` rescales each incoming
frame's per-channel mean to match the *first* frame of the sequence
(locked in module-level state on first call — causal, no look-ahead, safe
inside the online SLAM loop). Gain is clamped to `[0.4, 2.5]` so a
frame that's transiently very dark/bright (e.g. pointed into a shadow)
doesn't get wildly over-corrected. Wired into `splatt3r_slam/frame.py:
create_frame()`, applied to the raw `[0,1]` numpy frame *before*
`resize_img()` — so it affects the network's input, the SH colour base,
and the on-screen keyframe texture consistently, not just one of them.

Gated by `config/base.yaml: dataset.normalize_exposure` (default `True`).
Set `False` (or pass a config that overrides it — see
`SPLATT3R_GS_DUMP_DIR` pattern in the splatt3r-gaussian-map skill for how
to A/B test via a throwaway `inherit: config/base.yaml` override config)
to disable for comparison.

**Measured effect (2026-07-22, `freiburg1_floor`, `SPLATT3R_GS_DUMP_DIR`
debug dumps — see splatt3r-gaussian-map skill for that mechanism)**:
- With the fix on: map render survived past frame 880 without crashing;
  a faint tonal seam between two keyframes' floor patches is still
  visible in the accumulated map, but it's a soft gradient, not the hard
  jump originally reported. **Partial fix, not a full solve.**
- With the fix off (`normalize_exposure: False`): the interactive
  renderer crashed with the same `illegal memory access` bug covered in
  the splatt3r-gaussian-map skill, by frame ~120-160 — much earlier than
  any run with the fix on. See that skill's crash section for the
  (unproven but plausible) hypothesis that unnormalized, more
  out-of-distribution frames make the network more likely to emit a
  degenerate Gaussian that trips the vendored rasterizer. Only one run
  each side — treat as a lead, not a proven correlation, if you revisit
  this.

**Why it's not a full fix**: gray-world channel-mean matching only
corrects a *global, per-frame, roughly-uniform* exposure/white-balance
shift. It cannot fix local/directional lighting differences (e.g. a
shadow that's only over part of the floor in one pass, specular
highlights that move with viewing angle, a light source that's brighter
on one side of the room) — those need actual spatial reasoning about
*which* Gaussians correspond to the same surface, which is Plan 2.

## Plan 2 — overlap-based colour harmonization: MEASURED NEGATIVE on desk (2026-08-03)

Measured offline on the desk keyframe dump (`scripts/color_harmonize.py`,
causal order — keyframe k fits a least-squares per-channel gain against the
already-placed map inside shared 10 mm voxels, applied to its whole
contribution, clamped [0.6, 1.67]):

```
composed raw map:   psnr 12.4416  lpips 0.5027
harmonized:         psnr 11.8669  lpips 0.5158      -> -0.57 dB, +0.013 lpips
```

Why it fails here, and the caveat that bounds the claim:

- The fitted gains are strong and monotone (kf 10: 0.64, kf 12: 0.65-0.70) —
  far too aggressive for physical exposure drift. The fit is absorbing
  per-view colour bias of the head, view-direction shading, and 10 mm
  voxel-pairing error (the two-shell structure of §15.7: it pairs colours
  from slightly DIFFERENT surfaces). Applied globally, that distorts more
  than it fixes.
- Desk is a MILD-drift sequence, and the deployment path already removes
  most of what drift there is upstream (`normalize_exposure`, Plan 1). The
  claim is therefore bounded: Plan 2 is measured negative exactly where Plan
  1 already operates. A strong-drift sequence without Plan 1 could still
  benefit — but with Plan 1 in the pipeline, the remaining signal Plan 2
  would fit is mostly not exposure.
- The refiner (splatt3r-finetuning-experiments §15) is the better
  harmonizer where it matters: it optimizes colours photometrically with
  per-view supervision instead of fitting one gain per keyframe against a
  voxel-paired proxy. Measured: +2.15 dB online (§15.11), which includes
  whatever colour reconciliation the map needs.

**Disposition: do not wire Plan 2 into the bake path.** If colour seams ever
become the visible bottleneck again, the evidence says the lever is the
refiner's optimization (or Plan 3's training-side fix), not a per-keyframe
gain fit.

## Plan 3 — documented, NOT implemented: stop baking raw pixel colour at all

The deepest fix: retrain (or fine-tune) Splatt3R so the colour head
predicts an actual view-consistent colour estimate instead of
`residual + raw_pixel`. This removes the root cause instead of
compensating for it, but it's an ML training change, not a SLAM-pipeline
change — see the training section below for what that requires. Even
then, "view-consistent" colour would still need multi-view *training*
supervision that penalizes exactly this kind of cross-view
inconsistency (e.g. a photometric consistency loss across nearby views
of the same scene, or a global appearance regularizer) — the current
loss terms (`mse`, `lpips`, `mast3r` conf loss, per `main.py`'s
`training_step`) are all single-view reconstruction losses; nothing in
the visible loss config already does this, so it'd need a new loss term,
not just more training on the existing objective.

## Training the underlying model, and the LoRA question

> **⚠ SUPERSEDED (2026-07-26).** This whole section is out of date. The
> repo is no longer inference-only: a full LoRA fine-tuning pipeline was
> built for four dataset families (TUM, 7-Scenes, EuRoC, ETH3D), with its
> own data adapters that do **not** depend on ScanNet++ at all. See the
> **splatt3r-lora-finetuning** skill — treat it as authoritative for
> anything training-related. What remains valid below: the *reasoning*
> about why cross-frame colour consistency is not addressed by the
> mse/lpips objective (both are single-view losses), which is still true
> of the LoRA training that now exists.

**Historical state (no longer accurate):** `splatt3r_core/main.py`
tries `import data.scannetpp.scannetpp as scannetpp` and prints
`"Warning: scannetpp data module not available. Training functionality
disabled."` then raises if you actually call `run_experiment()`. Two
things are missing from this checkout, neither of which this session
can obtain on its own:
1. **The ScanNet++ dataset** (the original Splatt3R paper's training
   data) — needs a separate access request/license from the ScanNet++
   maintainers.
2. **The `data/scannetpp/scannetpp.py` dataset module** — not vendored
   here; it's part of the original `brandonsmart/splatt3r` repo (the
   same source as the `brandonsmart/splatt3r_v1.0` checkpoint this repo
   downloads via `splatt3r_slam/splatt3r_utils.py: load_splatt3r()`).

There's also no example training config YAML anywhere in this repo —
`workspace.load_config(sys.argv[1], ...)` needs one built from scratch,
matching the fields read throughout `main.py`: `config.data.{root,
resolution, batch_size, num_workers, epochs_per_train_epoch}`,
`config.opt.{lr, epochs, gradient_clip_val}`, `config.loss.{
mse_loss_weight, lpips_loss_weight, mast3r_loss_weight, apply_mask,
average_over_mask}`, `config.sh_degree`, `config.use_offsets`,
`config.use_pretrained` + `config.pretrained_mast3r_path`,
`config.devices`, `config.save_dir`, `config.name`, `config.seed`.

Entry point once those exist: `python splatt3r_core/main.py <config.yaml>
[dotlist.overrides=value ...]` (OmegaConf dotlist merge, per
`workspace.load_config`).

### Is LoRA applicable here?

Yes, architecturally — checked the actual module structure
(`splatt3r_core/src/mast3r_src/dust3r/croco/models/blocks.py`): the
encoder (`AsymmetricMASt3R`, `enc_depth=24`) and decoder (`dec_depth=12`)
are a standard pre-norm ViT built from plain `nn.Module`/`nn.Linear`
blocks — `Attention.qkv`, `Attention.proj`, `Mlp.fc1`, `Mlp.fc2`
(repeated per block, plus a cross-attention variant with its own
`proj`). Nothing here is a HuggingFace `PreTrainedModel`, but `peft`'s
`LoraConfig`/`get_peft_model` doesn't actually require that — it just
needs addressable `nn.Linear` submodules, which these are. In principle:
```python
from peft import LoraConfig, get_peft_model
lora_cfg = LoraConfig(target_modules=["qkv", "proj", "fc1", "fc2"], r=8, lora_alpha=16)
model.encoder = get_peft_model(model.encoder, lora_cfg)
```
(names/rank illustrative — would need a real smoke test against this
exact `mast3r_model.AsymmetricMASt3R` instance before trusting it, and
`target_modules` as bare names will match *every* block's `qkv`/`proj`/
`fc1`/`fc2` since `peft` matches by suffix — confirm that's the intent,
not e.g. accidentally also matching a differently-purposed `proj` inside
the patch embed `PatchEmbed.proj`, which is a `Conv2d` not a `Linear`
and would need excluding or would simply fail to match since LoRA
targets `Linear`/`Conv1d`/`Embedding` by default, not `Conv2d`).

**Whether LoRA is actually the right lever depends on what you're trying
to fix.** The existing training setup (`main.py`'s `MAST3RGaussians.__init__`)
already freezes the whole encoder backbone and only trains the small
Gaussian DPT head (`self.encoder.downstream_head{1,2}.gaussian_dpt.dpt`) —
i.e. it's already a parameter-efficient, head-only fine-tune, no LoRA
needed for that. LoRA would matter specifically if the *frozen backbone
features* themselves are the bottleneck — e.g. the encoder's learned
priors don't transfer well from ScanNet++'s scenes to your target domain
(different lighting, camera, clutter style) — and you want to nudge the
backbone cheaply without the cost/forgetting-risk of unfreezing it
fully. Either way, the same data blocker above applies: LoRA changes
*how* you fine-tune, not *what data* you fine-tune on. You still need a
posed multi-view dataset (ScanNet++, or your own captured/posed data
with an equivalent PyTorch `Dataset` wired in place of `scannetpp`)
before any of this — training or LoRA — is actually runnable.
