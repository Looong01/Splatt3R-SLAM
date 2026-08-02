---
name: splatt3r-lora-finetuning
description: SUPERSEDED (route measured NEGATIVE, -49% vs base -- read splatt3r-finetuning-experiments first). Per-dataset-family LoRA fine-tuning of the Splatt3R checkpoint (TUM, 7-Scenes, EuRoC, ETH3D), to push past the base checkpoint's blur/colour-consistency ceiling (see splatt3r-gaussian-map and splatt3r-color-consistency skills). Covers what's vendored, the hot-swap (unmerged) adapter loading design, per-family data adapters including self-predicted pseudo-depth for EuRoC/ETH3D, what's built and verified, what's not yet done, and every gotcha found. Read before touching splatt3r_core/lora.py, splatt3r_core/data/, splatt3r_core/pseudo_depth.py, or training/loading anything LoRA-related against these checkpoints.
metadata:
  type: reference
---

# Per-scene LoRA fine-tuning for Splatt3R

> ## ⚠️ SUPERSEDED — the approach documented here was measured and FAILED
>
> This document describes encoder-LoRA fine-tuning (LoRA on the ViT
> backbone's `qkv`/`proj`/`fc1`/`fc2`, encoder unfrozen) as the active plan.
> **It was subsequently measured against the base checkpoint under a
> controlled protocol and lost on every metric — psnr 9.50 → 7.41,
> lpips 0.3414 → 0.4545, roughly −49% overall.** Root cause: it unfreezes an
> encoder that upstream deliberately freezes, feeding out-of-distribution
> features to ~158M frozen matching/pointmap head parameters, which drives a
> Gaussian scale explosion (p99 scale 85× base). That explosion is also the
> most likely source of this project's long history of OOMs and
> `illegal memory access` crashes.
>
> **The route that works** is upstream's own recipe — freeze the encoder,
> train only the Gaussian head — measured at **+1.00 dB psnr / −0.017 lpips
> over base** on an identical seeded sample draw, at 6.3 GiB peak memory
> instead of 42 GiB.
>
> **Read `splatt3r-finetuning-experiments` first.** It carries the
> measurement protocol, both verdicts, and the root-cause analysis linking
> them.
>
> What remains valid below, and is worth keeping: the vendored data
> infrastructure (`data/data.py`, `cropping.py`), the per-family data
> adapters and coverage-matrix computation, pseudo-depth for EuRoC/ETH3D,
> the bf16 compatibility work, and the crash/gotcha catalogue. What is
> falsified: the LoRA target-module choices, the training hyperparameters
> tuned for that route, the scale-penalty patch (a symptom treatment for the
> explosion above), and the hot-swap adapter loading design at inference.

## Why this exists

`splatt3r-gaussian-map` and `splatt3r-color-consistency` both concluded the
base checkpoint (`epoch=19-step=1200.ckpt`, `brandonsmart/splatt3r_v1.0`)
has a real quality ceiling (blur at range, per-revisit colour drift) that
SLAM-pipeline code can only mitigate, not fix. **Correction to something
said earlier in that investigation**: this checkpoint is *not* an
undertrained/early checkpoint — the upstream `configs/main.yaml` sets
`opt.epochs: 20`, so epoch 19 (0-indexed) is the last epoch of the
officially recommended full training run. The blur/consistency issues are
closer to this architecture+recipe's actual ceiling than to "just needed
more training." Fixing it needs an adapted model, not more SLAM-side
compensation — the user asked specifically about per-scene LoRA fine-tuning
against the 9 locally-downloaded `datasets/tum/rgbd_dataset_freiburg1_*`
sequences.

Full design rationale and the plan this was built from:
`/home/share-v5/.claude/plans/replicated-weaving-dawn.md` (also copy it
somewhere more permanent if it matters — plan files are not guaranteed to
survive indefinitely).

## Key finding: local data is sufficient for TUM + 7-Scenes, ScanNet++ is not required

Earlier sessions concluded LoRA/training work needed ScanNet++ (external,
license-gated, not obtainable in an agent session). **That's wrong for
TUM and 7-Scenes.** Both are real RGB-D data (per-frame depth sensor
images) with ground-truth poses — exactly the data contract the training
pipeline needs, no external dataset or license required.

**EuRoC and ETH3D (this repo's downloaded "SLAM" subset) have no real
depth** — see "Four dataset families" below for how that's handled
(self-predicted pseudo-depth) and its real limitations.

## Scope: this now covers all four locally-downloaded dataset families

Originally scoped to TUM only (per-scene LoRAs); the user then asked for
one LoRA **per dataset family** (`datasets/tum/`, `datasets/7-scenes/`,
`datasets/euroc/`, `datasets/eth3d/`), each pooling **all** locally
available sequences/scenes within that family, not just a handful. That's
what's built and described below. The original single-scene
(`freiburg1_desk`-only) proof of concept is kept below as history/
grounding for the design, but is superseded by the per-family scripts.

## Hot-swap (unmerged) adapter loading — the chosen inference design

Tried and explicitly rejected: pre-merging a trained LoRA into the base
weights and saving a new standalone checkpoint per scene
(`peft`'s `merge_and_unload()` + `trainer.save_checkpoint()`). It works
(verified: loads cleanly through the ordinary `load_splatt3r()` /
`MAST3RGaussians.load_from_checkpoint` path, zero SLAM-pipeline code
changes needed) but wastes disk — each merged checkpoint duplicates the
~2.9GB base weights. **Deleted after building it** (was at
`checkpoints/lora-merged/`, since removed) in favour of:

- `splatt3r_slam/splatt3r_utils.py: load_splatt3r(path, device, lora_path=None)`
  — if `lora_path` is given, loads the base checkpoint once as always,
  then `model.encoder = PeftModel.from_pretrained(model.encoder, lora_path)`
  — **not merged**. `peft` is imported lazily inside this branch only, so
  normal (no-LoRA) runs never need it installed.
- `main.py --lora <path>` — new CLI flag wiring straight into the above,
  e.g. `python3 main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --lora checkpoints/lora/tum`.
- The base checkpoint's weights are loaded exactly once regardless of how
  many different family adapters you want to try in the same process —
  see `scripts/eval_lora_scenes.py` for the pattern: keep a reference to
  the pristine `base_encoder`, and for each family do
  `model.encoder = PeftModel.from_pretrained(base_encoder, lora_dir)` —
  a few-MB load, not a multi-GB one.
- Runtime cost: an unmerged adapter adds one small extra matmul per
  adapted `Linear`/`Conv2d` per forward pass. Not benchmarked against the
  SLAM pipeline's real-time budget — if this ever matters in practice,
  `model.encoder.merge_and_unload()` right after loading collapses back
  to zero-overhead (the same operation the rejected pre-merge approach
  used, just done once at process start instead of persisted to disk).

## What's vendored from upstream (github.com/btsmart/splatt3r, `main` branch)

This repo's `splatt3r_core/` is a partial vendoring of the upstream repo
that dropped the entire training-data infrastructure. Pulled back in
(unmodified, verified to import and run):
- `splatt3r_core/data/data.py` — `DUST3RSplattingDataset` /
  `DUST3RSplattingTestDataset`: the dataset-agnostic context/target
  sampling logic. **Do not reimplement this** — it's what plugs any
  per-dataset `data` source (with `.sequences`, `.color_paths[seq]`,
  `.get_view(seq, idx, resolution)`) into the training pipeline via a
  precomputed `coverage[seq][i][j]` overlap matrix.
- `splatt3r_core/src/mast3r_src/dust3r/dust3r/datasets/utils/cropping.py`
  and `.../transforms.py` — needed by `data/data.py`'s
  `crop_resize_if_necessary`; these two specific files were the only
  missing pieces from `dust3r/datasets/utils/` (the rest of
  `dust3r/datasets/` — the actual dataset classes like `scannetpp.py` —
  is still absent and not needed for this).

If something under `dust3r`/`mast3r`/`croco` is missing an import, check
`https://raw.githubusercontent.com/btsmart/splatt3r/main/<same path>`
before writing a replacement — it's very likely already implemented
upstream and just wasn't carried into this repo's vendored subset.

**Import path gotcha**: `data/data.py` uses `from src.mast3r_src.dust3r...`
style absolute imports (assumes `splatt3r_core/` itself is the import
root). Any new script needs the same sys.path setup `splatt3r_core/main.py`
uses:
```python
sys.path.insert(0, CORE)  # splatt3r_core/
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))
os.chdir(CORE)  # utils/loss_mask.py etc. do bare `from utils.geometry import ...`
```

## New code (this session, verified working)

- `splatt3r_core/data/tum/tum.py` — `TUMData` (mirrors `ScanNetPPData`'s
  interface exactly: same attributes, same `get_view()` return schema —
  see that file's docstrings for the TUM-specific bits: depth scale is
  `5000.0`, not ScanNet++'s `1000.0`; intrinsics come from
  `config/intrinsics.yaml`'s freiburg1 calibration, shared by all 9
  sequences). Also `compute_coverage()`: TUM has no precomputed overlap
  matrix like ScanNet++'s `data/scannetpp/coverage/*.json`, so this
  computes one using the *already-vendored*
  `splatt3r_core/utils/loss_mask.py: calculate_in_frustum_mask` (the
  same function the training loss itself uses for masking) — coarse
  camera-position-distance prefilter (default 0.5-1.0m) before running
  the real overlap check on survivors, since exact O(n²) is not
  tractable at TUM's frame counts (hundreds to ~2300 for the largest
  sequence). **Verified on freiburg1_desk**: 488 train frames → 53,902
  candidate pairs in 15.3s, overlap fractions sanely distributed
  (mean 0.44, 20k+ pairs above 0.5). Returns a *dense* `{i: {j: frac}}`
  dict (0.0 for filtered-out pairs, not missing keys) because
  `DUST3RSplattingDataset.sample()` does dense lookups over every frame
  index, not just nearby ones — a sparse dict KeyErrors.
- `splatt3r_core/lora.py` — `attach_lora(model, r, alpha, target_modules)`
  and `MAST3RGaussiansLoRA(MAST3RGaussians)`.

## Two things that were *assumed* in the plan and turned out to need fixing (verified, not just theorized)

1. **`peft.get_peft_model()` on this non-HuggingFace encoder works**,
   confirmed via a standalone smoke test (forward+backward, checked
   `.grad` on injected LoRA params). `target_modules=["qkv","proj"]`
   matched by name suffix across all 24 encoder blocks — including
   `PatchEmbed.proj`, which is a `Conv2d`, not `Linear`. **Correction to
   the original plan**: it assumed peft's LoRA only targets
   `Linear`/`Conv1d`/`Embedding` and would silently skip a `Conv2d` —
   that's wrong, `peft` does support `Conv2d` LoRA and applied it there
   too. Harmless, just don't be surprised by it showing up in
   `print_trainable_parameters()`.
2. **`peft.get_peft_model()` clobbers pre-existing `requires_grad_(True)`
   on the Gaussian head.** `MAST3RGaussians.__init__` unfreezes
   `downstream_head{1,2}.gaussian_dpt.dpt` before any LoRA touches the
   model; wrapping the encoder with `get_peft_model()` afterwards
   re-freezes *everything* except the newly-injected LoRA weights,
   silently undoing that. `attach_lora()` re-enables it explicitly after
   wrapping — this is required, not defensive-only: verified by trainable
   param counts (LoRA-only ≈ 2.37M; after the head re-enable, ≈ 42.8M,
   which is what a real training run reported).

## Verified: one full single-scene training run (freiburg1_desk)

`MAST3RGaussiansLoRA.load_from_checkpoint(ckpt) → attach_lora(...) →`
build `TUMData`+`compute_coverage` for train/val → wrap in the vendored
`DUST3RSplattingDataset` → short `L.Trainer.fit(...)` (3 epochs ×
`num_epochs_per_epoch=20` × `batch_size=2` ≈ 30 steps — a proof-of-concept
scale run, not a real convergence run) → `peft`'s `save_pretrained()` to
`checkpoints/lora/rgbd_dataset_freiburg1_desk/`.

Ran clean: no NaN, no crash, final logged `train/loss: 0.153,
train/mse: 0.078, train/psnr: 11.08, train/lpips: 0.302`. LoRA adapter
saved successfully (`adapter_config.json` + `adapter_model.safetensors`).
This validates the *entire* pipeline end to end — every piece above is
proven to actually work together, not just individually smoke-tested.

**Real hyperparameters, cross-checked against upstream's
`configs/main.yaml`** (use these as the informed starting point, not
guesses): `resolution=[512,512]` (square — note this differs from the
SLAM inference path's `resize_img`, which does long-side-512 with an
aspect-preserving crop; this session used the training-upstream
convention since it's the "proven" one, but this is a real open question
flagged in the plan, not a settled one), `opt.lr: 0.00001`,
`weight_decay: 0.05`, `gradient_clip_val: 0.5`,
`loss: {mse_loss_weight: 1.0, lpips_loss_weight: 0.25,
mast3r_loss_weight: Null, apply_mask: True, average_over_mask: True}`.
This session's proof-of-concept run used `lr=1e-4` (higher than the
official `1e-5`, deliberately — LoRA has far fewer trainable params, so
a higher LR is the usual starting adjustment) — not yet validated against
the lower official LR, worth comparing.

## Four dataset families: loaders, and how EuRoC/ETH3D get depth

`splatt3r_core/data/common.py` — shared low-level utilities used by all
four family loaders: `read_file_list`/`associate` (nearest-timestamp
matching, TUM-style; delimiter-configurable for EuRoC's CSVs),
`quat_xyzw_to_rotmat` / `quat_wxyz_to_rotmat` (TUM/ETH3D vs. EuRoC use
**different quaternion component orders** — mixing them up silently gives
a wrong-but-plausible-looking rotation, verified against both by checking
rotation-matrix orthonormality on real data, not just unit-testing the
math in isolation), `split_train_val` (contiguous, not shuffled — TUM-
style video has near-duplicate adjacent frames, a random split would leak
across train/val), and `compute_coverage` (moved here from
`data/tum/tum.py`, fully generic — only touches `.color_paths`/`.c2ws`/
`.get_view()` — reused verbatim by all four families).

Every family loader mirrors `ScanNetPPData`'s interface (`.sequences`,
`.color_paths[seq]`, `.c2ws[seq]`, `.intrinsics[seq]`,
`.get_view(seq, idx, resolution)`), and **pools every sequence/scene it
can find** under the family root into that one interface — one
`DUST3RSplattingDataset`/training run per family covers all of it, not
one call per sequence.

- `data/tum/tum.py: TUMData` — globs `rgbd_dataset_freiburg*` under
  `datasets/tum/`. Verified: **all 9** locally-downloaded sequences pool
  correctly (360, desk, desk2, floor, plant, room, rpy, teddy, xyz).
- `data/sevenscenes/sevenscenes.py: SevenScenesData` — globs
  `<scene>/seq-*/` (extracted directories only, not `.zip`s) under
  `datasets/7-scenes/`. Real depth (`.depth.png`, uint16 mm, `65535` =
  invalid — different convention from TUM's, don't reuse TUM's depth
  scale here). `pose.txt` is already a plain camera-to-world 4x4 matrix
  (no quaternion decoding, unlike the other three families). Intrinsics
  are NOT shipped per-scene; reused the fixed Kinect-v1 constants
  `splatt3r_slam/dataloader.py: SevenScenesDataset` already hardcodes
  (`fx=fy=585, cx=320, cy=240`). **As of writing, only `seq-01` is
  extracted for each of the 7 scenes** (~1000 frames each, 7 total
  sequences, ~7000 frames pooled) — the other `seq-02.zip` etc. exist on
  disk but aren't unzipped; unzip more for a bigger pool, nothing else
  needs to change (the loader re-globs on every construction).
- `data/euroc/euroc.py: EuRoCData` — globs sequence dirs under
  `datasets/euroc/` (11 found: MH_01-05, V1_01-03, V2_01-03). Only
  **cam0** is used (mono) — no stereo matching needed since depth is
  self-predicted, not computed geometrically. Images are grayscale,
  heavily distorted — undistorted here via OpenCV (`getOptimalNewCameraMatrix`
  + `initUndistortRectifyMap`, same approach `dataloader.py: EurocDataset`
  already uses for SLAM tracking on this same data, comment there: *"the
  distortion is too much to handle for MASt3R"*). Ground truth is
  **body**-frame pose (`state_groundtruth_estimate0/data.csv`); camera
  pose is `T_WC = T_WB @ T_BS` using `cam0/sensor.yaml`'s `T_BS`
  (cam0-to-body extrinsics). Timestamps are nanosecond integers, converted
  to seconds so `max_time_diff` means the same thing across all four
  families.
- `data/eth3d/eth3d.py: ETH3DData` — globs scene dirs under
  `datasets/eth3d/train/` (61 available; `max_scenes` param caps this —
  precompute/training scripts default to 15 for a tractable first pass,
  see below). Format is TUM-shaped (`rgb.txt`, `groundtruth.txt` with
  xyzw quaternions) except `calibration.txt` is one line `fx fy cx cy`,
  no distortion.

**EuRoC and ETH3D have no real depth in what's downloaded here**
(confirmed by inspecting the actual directory contents, not assumed):
EuRoC is stereo+IMU with no depth sensor; this ETH3D download is the
*monocular SLAM-benchmark subset* (`rgb.txt`/`calibration.txt`/
`groundtruth.txt`/`rgb/`), not the fuller ETH3D release that has
LiDAR-scanned ground truth for some scenes. **User's explicit choice**
(asked directly, given the alternatives — TUM+7-Scenes only, or EuRoC via
real stereo block-matching): use the **base model's own self-predicted
depth** for both. `pseudo_depth.py: predict_pseudo_depth(model, img, device)`
runs one self-view (`view1==view2`) forward pass through the *base,
non-LoRA* encoder, extracts the Z-channel of the predicted `means` as
depth and thresholds `conf >= 1.5` for validity, resizes back to the
input's native resolution via `cv2.resize`. **Important limitation, not
hidden**: this makes training push the model to be more *self-consistent*
with its own existing geometry estimate on these two families, not more
*accurate* against independent ground truth the way TUM/7-Scenes' real
sensor depth does. Not circular in the strict sense (pseudo-depth is
precomputed once from the frozen base checkpoint before any LoRA training
starts, and never updated mid-training), but it's a real, deliberate
trade-off — flag this prominently if reporting EuRoC/ETH3D LoRA results
to anyone, don't let it read as "trained against ground truth" like the
other two families.

`scripts/precompute_pseudo_depth.py` computes and caches this to
`datasets/<family>/<seq>/pseudo_depth/<frame>.npy` (idempotent — already-
cached frames are skipped, safe to Ctrl-C and resume). **Must be run
before training euroc/eth3d** — `EuRoCData`/`ETH3DData.get_view()` raise
a clear `FileNotFoundError` pointing back at this script otherwise, they
don't silently fall back to anything.

## Dual-GPU (2x RTX A6000, 96GB total VRAM) + long-run hyperparameters

Updated after the user asked explicitly for this: `DEVICES=2` +
`strategy="ddp_find_unused_parameters_true"` (the `find_unused_parameters`
variant is required, not optional -- LoRA + frozen backbone means most
params get no gradient most steps, plain DDP errors on that). Current
`scripts/train_lora_per_scene.py` constants (all editable at the top of
that file):

**⚠ This table drifted badly from the code and was corrected 2026-07-26.
Treat `scripts/train_lora_per_scene.py` as the source of truth and
re-check before relying on any number here.** Values below verified
against the file on that date:

| constant | value | notes |
|---|---|---|
| `DEVICES` | `2` | both A6000s |
| `BATCH_SIZE` | `2` | **per GPU** — global batch = 4. Was `12` here (and in this table) during the OOM phase; lowered by the user and never raised back. At the per-family resolutions below a run sits at ~23-27GiB of 49GiB, so there is headroom to raise it — untested |
| `NUM_WORKERS` | `16` | per DDP process (2 ranks × 16 = 32 workers, machine has 32 cores). Was `8`. Note an earlier `32` exhausted `/dev/shm` — do not raise further without checking `df -h /dev/shm` |
| `NUM_EPOCHS_PER_EPOCH` | `1000` | samples/Lightning-"epoch" per sequence |
| `MAX_EPOCHS` | `100` | was `200` |
| `CHECKPOINT_EVERY_N_EPOCHS` | `10` | see below |
| `LORA_TARGET_MODULES` | `("qkv","proj","fc1","fc2")` | widened from attention-only `("qkv","proj")` after val/psnr plateaued ~9; `fc1`/`fc2` match 100 further Linear layers (mlp in every enc/dec block). Trainable encoder params 42.8M → **46.6M** (was briefly reported as 87.0M: `attach_lora` was also re-enabling grad on peft's frozen `original_module` copies of the Gaussian head -- 40.4M params that are never serialized and cannot affect the output; removed) |
| `LR` / `RESUME_LR_FACTOR` | `2e-5` / `0.1` | resumed runs use `LR × RESUME_LR_FACTOR` — see the resume section below for the measured reason |
| `GRADIENT_CLIP_VAL` | `0.1` | global-norm clip. Note: with trainable params now doubled, this clips *more* aggressively per-parameter, so under-fitting is a likelier failure than divergence — loosen this before touching `LR` |
| **resolution** | **per-family, in `FAMILIES`** | no longer one global `RESOLUTION`. `tum`/`7-scenes` `(512,384)`, `euroc` `(512,320)`, `eth3d` `(512,304)`, order `(W,H)`. Each equals what `resize_img(size=512)` actually feeds that dataset at SLAM inference, so training matches deployment. All are cheaper than the old `512×512` (768/640/608 vs 1024 tokens) |
| `GAUSSIAN_SPATIAL_STRIDE` | `1` | was `2` as a crash mitigation; reverted once `scale_invariant=False` fixed the real cause, since stride=2 cost real reconstruction quality |

**Per-rank device placement**: `train_one_family()` now reads
`LOCAL_RANK` from the environment (`device = f"cuda:{local_rank}"`)
instead of a hardcoded `"cuda"` — required under DDP, since Lightning
spawns one subprocess per GPU and a hardcoded `"cuda"` would put every
rank's model on GPU 0. Coverage computation and model loading both use
this per-rank device.

**Periodic checkpointing** (`SaveLoRAAdapterCallback` in
`train_lora_per_scene.py`): a run this long needs intermediate saves, not
just an all-or-nothing one at the end. Saves the LoRA adapter (not a full
Lightning checkpoint — `enable_checkpointing` stays `False`) every
`CHECKPOINT_EVERY_N_EPOCHS` epochs to `checkpoints/lora/<family>/epoch_<N>/`,
rank-0-only (guarded by `trainer.is_global_zero`, both for this callback
and the final save, to avoid every DDP process racing to write the same
files). The script does **not** auto-resume from one of these on restart
— that's a manual `ckpt_path=` addition to `trainer.fit()` if/when needed.

**Reliability concern flagged, not resolved**: running `L.Trainer(...).fit()`
for 4 families sequentially *inside one process* means 4 rounds of
DDP process-group init/teardown. Lightning is supposed to support this
(a fairly standard pattern, e.g. k-fold CV loops), but it was never
actually exercised end-to-end here. **`scripts/train_lora_all_families.sh`**
sidesteps the question entirely by giving each family a fresh `python3`
process (one full DDP spawn/teardown cycle per family, no repeats) — use
this instead of calling `train_lora_per_scene.py` with multiple family
names directly for the real dual-GPU run. The cost is reloading the base
checkpoint once per family instead of reusing an already-loaded one —
a few seconds, irrelevant next to a multi-hour/day training run.

**Bug found and fixed (2026-07-23): DDP rank 1+ never actually started,
whole job crashed the instant it reached `trainer.fit()`.** The user ran
the real dual-GPU job (`train_lora_all_families.sh`) and observed exactly
the symptom you'd expect from a broken DDP setup: only GPU 0 in use, only
~8GB VRAM (not the 12-batch-size full training footprint), single process
in `ps`. That state is actually *normal* for the pre-`trainer.fit()` setup
phase (model load + `build_pooled_coverage()` — both inherently
single-process since Lightning only spawns extra ranks inside
`trainer.fit()`), so it wasn't proof of a bug by itself — but ~12 minutes
in, `full_run.log` showed the real failure once TUM's setup finished and
training actually tried to start:

```
Initializing distributed: GLOBAL_RANK: 0, MEMBER: 1/2
python3: can't open file '.../splatt3r_core/scripts/train_lora_per_scene.py': No such file or directory
[rank: 1] Child process with PID ... terminated with code 2. Forcefully terminating all other processes
```

Root cause: Lightning's DDP subprocess launcher re-execs this script for
every extra GPU rank via
`[sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]`
(`lightning/fabric/strategies/launchers/subprocess_script.py:_basic_subprocess_cmd`),
and that `os.path.abspath()` call happens lazily *inside* `trainer.fit()`
— i.e. after `train_one_family()`'s `os.chdir(CORE)` has already changed
the working directory. Since the script is always invoked as a
cwd-relative path (`python3 scripts/train_lora_per_scene.py tum`, run
from the repo root), resolving `sys.argv[0]` against the *new*
(post-chdir) cwd silently points at a nonexistent path
(`splatt3r_core/scripts/...` instead of `scripts/...`), so rank 1 fails
before it even imports anything, and Lightning tears down rank 0 along
with it. Verified by reading Lightning's actual source and reproducing
the path resolution in isolation (confirmed broken without the fix,
correct with it).

**Fix**: `sys.argv[0] = os.path.abspath(sys.argv[0])` added right after
`import sys`, *before* `os.chdir(CORE)` — makes Lightning's later
`abspath()` call a no-op on an already-correct absolute path. One line,
no behavior change for single-GPU runs (`DEVICES=1` never hits this launcher
code path at all).

**Related inefficiency fixed at the same time**: even with the crash
fixed, rank 1 re-execs the *entire* script from scratch, which would
otherwise redo `build_pooled_coverage()` — the expensive step — a second
time, in parallel with nothing (rank 0 already finished it before
reaching `trainer.fit()`, which is *when* rank 1 gets spawned). Added
disk caching in `build_pooled_coverage()`/`COVERAGE_CACHE_ROOT`
(`checkpoints/lora_coverage_cache/<family>_<stage>_valfrac..._pos....pkl`)
so rank 1 loads rank 0's already-computed result instead of recomputing
it. Deterministic given the same downloaded data (no randomness anywhere
in `split_train_val`/`compute_coverage`), so sharing across ranks this way
is safe — but the cache is **not** keyed on the downloaded data itself,
only on `VAL_FRACTION`/`COVERAGE_POS_THRESHOLD`, so delete
`checkpoints/lora_coverage_cache/` if sequences are added/removed under
`datasets/<family>/`.

No orphan processes needed cleanup after this — `train_lora_all_families.sh`
runs with `set -e`, and Lightning's own subprocess observer already killed
rank 0 the moment rank 1 failed, so the whole tree (bash wrapper +
`train_lora_per_scene.py`) had already self-terminated by the time this was
investigated.

**CUDA OOM at BATCH_SIZE=2, and the precision fix (2026-07-23/24).** After
the DDP fix above, the user hit `torch.OutOfMemoryError` inside the
Gaussian rasterizer's `forward()`, with 45.18/47.4GB already allocated
(not fragmentation — only 267MB reserved-but-unallocated) before the
rasterizer's own 3.82GB ask. Lowering `BATCH_SIZE` 12→3→2 barely moved it.
Root cause: `BATCH_SIZE` isn't the dominant term here. Two batch-size-
independent(ish) factors dominate instead: (1) `MAST3RGaussiansLoRA`
removes the base model's `torch.no_grad()`, so activations through the
*entire* 731M-param backbone (not just the 42.8M trainable LoRA params)
are retained for backward; (2) `DUST3RSplattingDataset` always renders
`num_target_views=3` per sample (fixed in `data/data.py`, independent of
`BATCH_SIZE`), each a separate CUDA rasterizer call with its own
geom/binning/img buffers. Trainer also had no `precision=` set, so
training ran full fp32 by default — the biggest single lever available.

First attempt, `precision="bf16-mixed"`, crashed immediately with
`NotImplementedError: "rope_2d_cuda" not implemented for 'BFloat16'`. The
vendored custom CUDA RoPE kernel every attention block calls through
(`croco/models/curope/kernels.cu:101`) is compiled with
`AT_DISPATCH_FLOATING_TYPES_AND_HALF` — covers float32/float64/float16,
NOT bfloat16 (verified by reading the actual macro in the `.cu` source,
not guessed). **Fixed by switching to `precision="16-mixed"` instead**
(fp16 hits the kernel's supported Half path, same ~2x memory win). Tradeoff:
fp16's narrower exponent range needs loss scaling, which Lightning's
`"16-mixed"` handles automatically via `GradScaler` — no manual code
needed, but if the loss curve goes NaN/inf, that's the mechanism to
suspect first. Not run/verified past that first NotImplementedError yet —
watch for it working cleanly (or a new dtype-related error) on the next
real attempt.

**Follow-up (same session): patched and rebuilt the RoPE kernel for real
bf16 support, and pre-emptively fixed the Gaussian rasterizer too, instead
of settling for fp16.** The user asked whether the RoPE kernel could just
be recompiled with an "AND_HALF_AND_BFLOAT16" dispatch macro rather than
falling back to fp16. That exact macro name doesn't exist in this
PyTorch's `ATen/Dispatch.h` (checked directly) — the correct one is
`AT_DISPATCH_FLOATING_TYPES_AND2(SCALARTYPE1, SCALARTYPE2, ...)`, used here
as `AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half,
at::ScalarType::BFloat16, tokens.scalar_type(), "rope_2d_cuda", (...))` in
`croco/models/curope/kernels.cu`. Safe to add because the kernel already
does all its math (cos/sin, multiply-accumulate) in `float` internally
regardless of input dtype (see `rope_2d_cuda_kernel`) — bf16 only changes
the read/write precision at the tensor boundary, same as the pre-existing
Half path. Rebuilt with `python3 setup.py build_ext --inplace` in that
directory (needs `LD_LIBRARY_PATH` including the conda env's
`torch/lib` to import afterward, e.g. for a standalone smoke test outside
the training script's own sys.path setup). Verified directly (not just by
inspection) with a standalone script: called `curope.rope_2d()` on
float32/float16/bfloat16 CUDA tensors, confirmed all three produce
finite, changed (non-identity) output, with bf16 vs fp32 max-abs-diff
≈0.0135 (fp16 vs fp32 ≈0.0017) — larger deviation as expected from bf16's
shorter mantissa, not garbage.

While investigating, also checked whether the *other* custom CUDA
extension in the render path — `thirdparty/diff-gaussian-rasterization-
modified` (used by `src/pixelsplat_src/cuda_splatting.py:render_cuda()`)
— would survive bf16 tensors reaching it too. It doesn't: unlike the RoPE
kernel, `rasterize_points.cu`'s `RasterizeGaussiansCUDA()` has no
AT_DISPATCH/templating at all — every tensor argument
(`means3D`, `sh`, `colors`, `opacity`, `scales`, `rotations`,
`cov3D_precomp`, `viewmatrix`, `projmatrix`, `campos`, `background`) goes
through a hardcoded `.data<float>()`/`.data_ptr<float>()`, which throws a
clear dtype-mismatch error (not a silent memory-corruption bug — PyTorch's
`data_ptr<T>()` asserts the tensor's actual scalar type first) if fed
anything but float32. Rather than wait for this to surface as a second
crash, pre-emptively patched `render_cuda()` to `.float()` all its
Gaussian/camera tensor inputs right before they reach the rasterizer —
standard practice for wrapping a non-autocast-aware custom op. Autograd
casts gradients back to the original (bf16) dtype automatically on the
way out, so this doesn't undo the backbone's mixed-precision memory
savings, it just keeps this one call fp32 as the extension requires.

`scripts/train_lora_per_scene.py` is back to `precision="bf16-mixed"`
(from the `"16-mixed"` fallback) now that both extensions support it —
preferred over fp16 for training since bf16's exponent range matches
fp32's, avoiding fp16's GradScaler/NaN risk entirely. **Neither the
rebuilt kernel nor the `render_cuda()` patch has been exercised inside an
actual end-to-end training run yet** — only the standalone kernel smoke
test above. Watch the next real run for a clean pass through at least one
full training step (encoder → decoder → rasterizer → loss → backward)
before trusting this.

**The illegal-memory-access saga (2026-07-26): user had me run and debug
training myself, live, in the background.** Found via `debug=True` on
`GaussianRasterizationSettings` (in `cuda_splatting.py` -- makes the
extension's `CHECK_CUDA` macro sync and report `__FILE__`/`__LINE__` after
every internal kernel launch, instead of deferring the error to whatever
Python-side call happens to run next) that the crash is real, inside
`thirdparty/diff-gaussian-rasterization-modified/cuda_rasterizer/
rasterizer_impl.cu` -- but at a *different* line each time depending on
which other fix was tried, which is itself informative: it's memory
corruption whose exact "first noticed" kernel is essentially arbitrary,
not one specific buggy call.

Tried, in order, each verified against a real run:
1. Sanitizing `scales`/`rotations` (then `means`/`opacities`/`sh` too) in
   `lora.py` before `build_covariance()` -- delayed the crash (further
   into training each time) but never fixed it. Wrong axis: the crash
   isn't caused by any single bad value.
2. `spatial_stride=2` on the decoder (`DecoderSplattingCUDA`, subsamples
   the H,W Gaussian grid 4x before rendering) -- crashed anyway, at a
   *different* `rasterizer_impl.cu` line (326, forward-pass tile-range
   identification, vs. 398 backward-pass render before). Ruled out "too
   many Gaussians in total" as the cause.
3. Tightening `scales` clamp from `max=10.0` to `max=0.5` (matching
   `splatt3r_slam/splatt3r_utils.py`'s own `max_scale`) after realizing
   `render_cuda(..., scale_invariant=True)` multiplies covariance by
   `(1/near)^2 = 100` on top of whatever's clamped in `lora.py` -- survived
   past epoch 0 (val + best-checkpoint save both fired) and well into
   epoch 1, further than any previous attempt, but still crashed
   eventually at the same rasterizer_impl.cu:326.
4. Current hypothesis, applied but **not yet verified**: the vendored
   extension's `cuda_rasterizer/auxiliary.h: in_frustum()` culls any
   Gaussian with camera-space `z <= 0.2f` -- a hardcoded absolute
   threshold. Screen-space footprint scales with `focal_length / z`, so a
   Gaussian just past that cutoff can still cover the whole frame.
   `decoder_splatting_cuda.py` passes a *constant* `near=0.1` (never
   derived from real scene content) into `render_cuda`'s
   `scale_invariant=True` path, which applies an unconditional 10x
   magnification (`scale = 1/near`) to every camera position and Gaussian
   on every sample -- not adaptive to anything, since `near` never
   varies. That fixed 10x stretch shifts which Gaussians land near the
   extension's fixed 0.2 cutoff, in a way disconnected from the actual
   scene. Passed `scale_invariant=False` in `decoder_splatting_cuda.py`'s
   `render_cuda(...)` call to remove this stretch entirely -- since
   `near` is constant here (not real adaptive behavior), this is a
   simplification, not a loss of functionality. **Launched a run with
   this change and it's in progress -- outcome not yet known as of this
   writing.**

If this doesn't work either, the next things to check, in order of
suspicion: (a) add an explicit additional near-depth cull/opacity-zeroing
in Python before rendering (compute each Gaussian's true camera-space z
via `world_space_to_camera_space` and mask out anything within a safety
margin of the extension's hardcoded 0.2, rather than relying on the
extension's own frustum check); (b) whether `means`'s clamp range
(`[-1000, 1000]`, lora.py) is actually appropriate for this coordinate
system at all, now that `scale_invariant` no longer rescales it 10x; (c)
accept that this vendored extension may need actual source changes
(recompiling with a configurable near-threshold) rather than a
Python-side workaround.

**Update: `scale_invariant=False` verified as a real, large improvement,
not a full fix.** A run with it survived epoch 0 AND epoch 1 (val/loss
0.2281 → 0.2151, i.e. actually learning, not collapsing) -- previously
every attempt died mid-epoch-1 at the latest. That same run then crashed
again in epoch 3 (~8730 steps in, vs ~2680-2940 before) -- same
`rasterizer_impl.cu:398` as the very first `debug=True` catch. Net: this
fix pushed the crash roughly 3x further out, but didn't eliminate it --
whatever's left is a rarer edge case, not the dominant cause anymore.

**Given crashes are now infrequent rather than near-immediate, added
resume-on-crash instead of continuing to chase full elimination:**
- `splatt3r_core/lora.py: attach_lora()` gained a `resume_from` param --
  when given a directory from a previous `encoder.save_pretrained()`,
  uses `PeftModel.from_pretrained(model.encoder, resume_from,
  is_trainable=True)` instead of a fresh `LoraConfig`/`get_peft_model()`.
- `scripts/train_lora_per_scene.py` gained `find_resume_checkpoint()` --
  picks the highest-numbered `checkpoints/lora/<family>/epoch_N/` if any
  exist, else `best/`, else `None` (fresh start). Wired into
  `train_one_family()` automatically.
- `scripts/train_lora_all_families.sh` gained a per-family retry loop
  (`MAX_RETRIES=5`) -- a crashed family's process gets relaunched, which
  now warm-starts from the last save via the above instead of starting
  over. Still exits nonzero (loudly) if a family exhausts its retries.

**Important caveat, not a true resume**: only the LoRA adapter *weights*
carry over (`enable_checkpointing=False` -- full Lightning/optimizer/
scheduler state was never saved). Every retry restarts the optimizer's
momentum, the `MultiStepLR` scheduler's position, and the epoch counter
at 0/fresh. Good enough for "don't lose the trained weights on a crash,"
not for "continue the exact training trajectory."

**Gotcha hit while wiring this up**: `checkpoints/lora/tum/` already had
`epoch_10` through `epoch_80` on disk from a much earlier session attempt
(2026-07-25 11:00, confirmed by `adapter_config.json` mtime) -- these
predate literally every fix in this whole saga (AdamW, LR=2e-5,
`GRADIENT_CLIP_VAL=0.1`, `max_scale=0.5`, `spatial_stride=2`,
`scale_invariant=False`) and, worse, `epoch_80` almost certainly *is* the
"training collapse" checkpoint documented earlier (loss stuck ~1.2 for 86
epochs) -- `find_resume_checkpoint()` would have silently resumed from
that collapsed state as "the most recent." Moved those aside to
`checkpoints/lora/tum_stale_pre_fixes/` before relying on resume for the
first time. `best/` was fine to keep -- its mtime (02:29 same day) matched
the *current* session's `scale_invariant=False` run's epoch-1 save
(val/loss=0.2151), not the stale collapse. **Lesson for next time**: don't
trust `find_resume_checkpoint()`'s pick blindly after a long gap between
sessions or after a major hyperparameter change -- check the adapter's
mtime/config against what you'd actually expect before trusting a resume.

**Two more real bugs found in the resume mechanism itself, both fixed,
from watching an actual resumed run's numbers over several epochs:**

1. `SaveBestAdapterCallback` always started `best_val_loss = float("inf")`
   on every fresh process -- a resumed run has no memory that a better
   checkpoint already existed, so its callback happily overwrote a good
   `best/` (val/loss=0.2151) with a worse one (0.2366) just because
   0.2366 < inf. Fixed: the callback now writes a `val_loss.txt` sidecar
   next to the adapter and reads it back on init if present. (The 0.2151
   weights were already overwritten by the time this was caught --
   accepted as a minor loss, not something worth re-deriving.)

2. Bigger one: a resumed run's val/loss got steadily *worse* for 4
   straight epochs (0.2151 → 0.2366 → 0.2623 → 0.2824) -- not noise, a
   real monotonic trend, confirmed by watching the actual per-epoch
   numbers rather than assuming "should be fine." Root cause: resuming
   only carries the LoRA weights, never optimizer state
   (`enable_checkpointing=False`) -- so AdamW's momentum/variance
   estimates restart at zero every time, and Adam's bias correction
   inflates the effective step size while those estimates are still
   small. That's harmless starting from random init, actively harmful
   starting from an already-decent point (shoves the weights straight
   out of the good region). Fixed with `RESUME_LR_FACTOR = 0.1` in
   `train_lora_per_scene.py`: resumed runs use `LR * RESUME_LR_FACTOR`
   instead of plain `LR`, giving AdamW a gentler restart. **Not yet
   verified against a real run** -- watch the next resume's per-epoch
   val/loss for the same monotonic-worsening pattern before trusting this.

General lesson underlined by both of these: a "resume" that only
restores model weights and silently drops everything else (optimizer,
scheduler, best-tracking state) is a much bigger behavior change than it
looks like from the diff -- verify its actual per-epoch numbers over
several epochs, don't just check that it doesn't crash.

**A run with `RESUME_LR_FACTOR=0.1` survived 16+ epochs (vs. 1-3 before)
-- then the retry wrapper itself caused a real incident.** val/loss
recovered and stayed stable (~0.28-0.30) for 16 epochs straight, by far
the longest/steadiest run of this whole saga -- strong evidence the
reduced resume LR fixed the degradation. It then crashed again (same
rasterizer_impl.cu, now line 330 -- yet another line, consistent with the
"corruption surfaces wherever it's next noticed" pattern) and the retry
wrapper correctly caught it and retried. But rank1 from the dead attempt
didn't clean up (the known "stuck spamming Broken pipe" orphan behavior),
and the wrapper's retry loop had no delay and no orphan cleanup between
attempts -- caught only by noticing both GPUs pinned at 100% with ~100
stray `train_lora_per_scene.py` processes accumulated (a fresh DDP
process-group + DataLoader worker pool spun up on top of the still-live
orphaned one from the previous attempt, repeatedly). Fixed in
`train_lora_all_families.sh`: `cleanup_orphans()` (kills anything
matching `train_lora_per_scene.py` by pattern, not a saved PID, since a
reparented orphan's PID isn't trackable that way) runs after every failed
attempt, and `RETRY_DELAY=30` seconds separates attempts -- both a
crash-loop brake and time for the kill to actually take effect and free
GPU memory before the next attempt starts. **Verify this specifically
next time a retry fires**: confirm process count and `nvidia-smi` actually
drop back to baseline between attempts, don't assume the wrapper's own
`if python3 ...; then` check is sufficient (it only observes rank0's exit
code, never rank1's).

**A stable, non-crashing run revealed a real quality regression from the
crash-mitigation settings.** With `GAUSSIAN_SPATIAL_STRIDE=2` and
`max_scale=0.5` in place, a resumed run went 5 epochs without crashing but
val/loss plateaued right around its resume starting point (~0.30, barely
moving) and val/psnr never exceeded ~9 -- notably worse than the ~9-16
range seen in the very first healthy run of this whole session, before
any crash mitigations existed. Diagnosis: both settings were defensive
measures added *before* `scale_invariant=False` (the actual fix) was
found -- `spatial_stride=2` throws away 4x of the rendered Gaussians
(direct detail loss) and `max_scale=0.5` was specifically calibrated to
counter the x100 covariance blowup that `scale_invariant=False` already
eliminates, so at 0.5 it's now needlessly starving the model of Gaussians
large enough to cover/blend surfaces properly. Loosened both back toward
their pre-mitigation values now that the real fix is in place:
`GAUSSIAN_SPATIAL_STRIDE` 2 → **1** (train_lora_per_scene.py),
`max_scale` 0.5 → **2.0** (lora.py, still 5x tighter than the original
10.0 that caused the crash -- not a full revert). **Not yet verified**
whether this reintroduces the crash or actually fixes the quality
plateau -- watch the next run for both val/psnr actually climbing above
~10 AND for the crash not recurring. If the crash comes back, tighten
`max_scale` again before reaching for `spatial_stride` (the evidence so
far points at scale, not raw Gaussian count, as the dominant lever).

**Much bigger finding, same investigation (2026-07-26): the Gaussian head's
training was never actually being saved, this entire session.** User asked
"so we save two checkpoints?" (LoRA + head) while a run was in progress --
prompted checking, which found `attach_lora()` never told peft about the
manually-unfrozen Gaussian head. `get_peft_model()` + manual
`requires_grad_(True)` on `downstream_head{1,2}.gaussian_dpt.dpt` makes
the head genuinely trainable *within a single process*, but
`encoder.save_pretrained()` (peft's own save path, used by both
`SaveLoRAAdapterCallback` and `SaveBestAdapterCallback`) only ever
serializes parameters peft itself knows to be adapter-related --
`lora_`-prefixed by default. Verified directly, not assumed: loaded the
actual `checkpoints/lora/tum/best/adapter_model.safetensors` on disk and
checked its keys -- 2,373,632 params, 100% `lora_`-prefixed, zero
`gaussian_dpt`/`downstream_head` keys. **Every single save and resume
across this entire debugging saga silently discarded all of the Gaussian
head's accumulated training and reset it to the base checkpoint's
original weights** -- only the small backbone LoRA adapter ever actually
carried forward. This is very plausibly a real contributor (on top of
`max_scale`/`spatial_stride`) to why val/psnr kept plateauing around ~9
despite many resumed epochs: the part of the model closest to the actual
rendered output (predicts means/scales/rotations/opacities/sh directly)
was effectively re-training from scratch, within-process only, every time.

Fixed with peft's purpose-built mechanism for exactly this pattern:
`LoraConfig(modules_to_save=GAUSSIAN_HEAD_MODULES)` in
`splatt3r_core/lora.py`, where `GAUSSIAN_HEAD_MODULES =
["downstream_head1.gaussian_dpt.dpt", "downstream_head2.gaussian_dpt.dpt"]`.
This makes peft treat those submodules as fully-trainable-and-saved
alongside the LoRA weights (wraps them in a
`peft.utils.other.ModulesToSaveWrapper` -- transparent for forward(), the
manual `requires_grad_(True)` calls stay in place as a belt-and-suspenders
check but are now redundant with what `modules_to_save` already does).
**Verified with an isolated smoke test before trusting it** (not just
read the peft docs and assumed): built the model, called the fixed
`attach_lora()`, called `save_pretrained()` to a throwaway dir, reloaded
the saved `adapter_model.safetensors` and confirmed
`has gaussian head keys: True` this time, 49.4M total saved params (up
from 2.37M) -- matches LoRA (~2.4M) + head (~40-47M, `modules_to_save`
keeps both a frozen "original" copy and the trainable copy in memory,
which is why total encoder params also grew, ~731M -> ~772M -- expected
peft behavior, not a bug, just a memory cost worth knowing about).

**Caveat for any resume from a checkpoint saved before this fix** (i.e.
everything under `checkpoints/lora/tum/` as of 2026-07-26): those only
have LoRA weights, no head weights to restore. Resuming from one still
only recovers the backbone LoRA half; the head silently falls back to its
base-checkpoint state -- not worse than before this fix, just not
carrying forward the (never-saved) head progress those old runs thought
they were accumulating. Going forward from this fix, saves/resumes should
actually carry the head's progress too. **Not yet verified end-to-end**
(does val/psnr actually climb further/faster now that the head's
progress compounds across resumes, instead of restarting every time) --
watch the next several epochs' val/psnr trend specifically for this.

**That `.float()`-only patch was wrong — confirmed by an actual run.** The
user hit this immediately in `validation_step`: `RuntimeError: expected
scalar type Float but found BFloat16` right at `rasterizer(...)` inside
`render_cuda()`, i.e. after the `.float()` casts had already run. Root
cause: an explicit `.float()` cast does not survive the *next*
autocast-covered op. `full_projection = view_matrix @ projection_matrix`
in that function is a matmul, and PyTorch's autocast intercepts matmul
calls and casts their output to bf16 **regardless of the input tensors'
actual dtype**, for as long as an autocast region is active (Lightning's
`precision="bf16-mixed"` wraps the whole training/validation step in
one). So `view_matrix`/`projection_matrix` being float32 going in didn't
matter — `full_projection` (used as `projmatrix`) came out bf16 anyway.
Verified this mechanism directly and in isolation before trusting the
fix: under `torch.autocast(device_type="cuda", dtype=torch.bfloat16,
enabled=True)`, `float32 @ float32` produces a `bfloat16` output; nesting
`torch.autocast(device_type="cuda", enabled=False)` inside restores
`float32` output. **Fixed by wrapping the entire body of `render_cuda()`**
(both the `.float()` casts and everything downstream through the
`rasterizer(...)` call, including the `@`) **in
`with torch.autocast(device_type="cuda", enabled=False):`** — this is
what actually pins every op inside to fp32 regardless of the ambient
training precision, not just the ops directly touching the function's
inputs. The lesson generalizes: `.float()`/`.to(dtype)` alone is never
enough to keep a block fp32 under autocast if the block contains any
autocast-covered op (matmul/conv/linear) — only `enabled=False` (or
staying entirely outside the autocast region) does that. Still not
verified past this specific error — watch for either a clean pass or a
new dtype error on the next run.

**Past the dtype errors, straight into real OOM — BATCH_SIZE stopped
being the lever (2026-07-24).** Two consecutive real runs confirmed this:
BATCH_SIZE=1 used 45.18GiB, BATCH_SIZE=2 used 45.52GiB -- barely
different, on a 47.4GB GPU. Added `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` (set before `import torch`, top of
`train_lora_per_scene.py`) after the first OOM showed real fragmentation
(2.40GiB reserved-but-unallocated) -- free + doesn't hurt, but the second
OOM at BATCH_SIZE=2 showed only 158MB reserved-unallocated, i.e. genuinely
not enough memory, not fragmentation, so this alone isn't the fix.

Root cause: the CUDA rasterizer is pinned to fp32 (the earlier
`render_cuda()` fix), so it gets none of bf16-mixed's memory savings, and
it renders `num_target_views=3` (fixed, from `data/data.py`) separate
512x512 views every step regardless of `BATCH_SIZE`. That's the actual
dominant, batch-size-insensitive cost.

Asked the user to choose between the two real levers (resolution vs.
target-view count) via AskUserQuestion since it's a training-quality
tradeoff, not a pure engineering one -- resolution wins on both fronts
(rasterizer AND backbone activations scale with H*W; target-view count
only affects the rasterizer/loss signal). **User picked lowering
RESOLUTION 512→384** (56% of the 512 rasterizer/activation footprint).
Applied in `train_lora_per_scene.py`. **Known, unverified tradeoff**: LoRA
now trains at a resolution lower than SLAM inference-time rendering (512,
via `resize_img`'s long-edge convention, itself already a different
preprocessing path than this square-crop one — see the original design
doc's discussion of this same square-vs-long-edge question). Whether
training at 384 meaningfully hurts the adapter's usefulness at 512
inference is genuinely unverified — watch for this specifically once
real training/eval output exists, not just "does it crash."

**Training collapse found by reading a real metrics.csv, and the fix
(2026-07-24).** With everything above finally letting a run survive past
setup, the user asked for an analysis of
`logs/lora_training/tum/lightning_logs/version_16/metrics.csv` (19,669
rows, CSVLogger via `log_every_n_steps=10`) instead of the terminal log
(which showed no loss/progress at all — separate, already-diagnosed
issue: piping through `tee` makes Python's stdout non-TTY, so it's fully
buffered and Lightning's Rich progress bar doesn't redraw properly
either; the CSV is the real source of truth for loss curves, not the
terminal log).

The analysis: epoch 0 was fine the whole way through (train/loss
~0.1-0.2, psnr ~9-16, low variance). Partway through epoch 1, between
logged steps 3369 and 3379, loss jumped 0.59 → 1.28 in a single 10-step
interval and never recovered — climbed further to ~1.3-1.5 over the next
~150 steps and then sat flat at ~1.19-1.24 for the remaining 86 epochs
(matched almost exactly by val/loss). This is a real, irreversible
training collapse, not noise — confirmed by computing per-epoch mean/std
and locating the exact step of the jump directly from the CSV, not
eyeballing it. No usable checkpoint exists before the collapse:
`CHECKPOINT_EVERY_N_EPOCHS=10` means the first periodic save is at epoch
10, long after step ~3370 (mid-epoch-1).

Suspected root cause: `LR=1e-4` with `GRADIENT_CLIP_VAL=0.5` wasn't a
strong enough bound against whatever produced one outlier gradient.
Leading hypothesis for the trigger (unconfirmed — no per-batch mask-size
logging exists to prove it): a batch with a sparse-but-nonzero loss mask.
`calculate_loss()`'s division only guarded the exact `mask.sum()==0` case
(via `.clamp(min=1)`, from the earlier DDP-hang fix above) — a mask with
only a handful of valid pixels lets a few badly-fit pixels dominate the
averaged loss and produce a disproportionate gradient, and `min=1` did
nothing to stop that.

User asked specifically whether switching to AdamW would auto-adjust the
learning rate and fix this. Answered directly rather than just
implementing it uncritically: Adam-family optimizers are already
per-parameter adaptive regardless of Adam vs. AdamW — that adaptivity
comes from a running average of past gradients (bias-corrected moment
estimates), which does not cap the damage from a single outlier gradient
spike happening right now. AdamW only adds decoupled weight decay on top
of that. So AdamW alone would not have prevented this collapse.
Implemented it anyway as a real, complementary improvement (it also
matches upstream `configs/main.yaml`'s `weight_decay: 0.05`, which the
prior plain `torch.optim.Adam(...)` call was silently missing entirely)
— `splatt3r_core/lora.py: configure_optimizers()` now uses
`torch.optim.AdamW(params, lr=..., weight_decay=0.05)`. But the changes
that actually address the collapse are separate:
- `scripts/train_lora_per_scene.py`: `LR` 1e-4 → **2e-5** (matches
  upstream's 1e-5 order of magnitude instead of the 10x-higher guess this
  started at), `GRADIENT_CLIP_VAL` 0.5 → **0.1** (a much tighter hard
  bound on per-step gradient norm, independent of what produces an
  outlier gradient).
- `splatt3r_core/main.py: calculate_loss()`: mask-sum denominator clamp
  raised from `min=1` to **`min=1000`** (~0.7% of one 384×384 view) as a
  hardening measure targeting the leading hypothesis specifically, on top
  of the LR/clip changes which are the root-cause-agnostic defense.

**None of this has been run yet** — syntax-checked only. The real test is
whether a fresh run's `metrics.csv` shows loss actually decreasing past
epoch 1 without another collapse, not just "does it start."

## Training/eval scripts (written this session; training has since been run — see the crash/resume/collapse saga above — but eval has not)

- `scripts/train_lora_per_scene.py` — one LoRA per family (despite the
  filename, kept from the original single-scene version). Loops over
  `FAMILIES = {tum, 7-scenes, euroc, eth3d}` by default, or pass names as
  argv to do a subset (e.g. `python3 scripts/train_lora_per_scene.py tum 7-scenes`
  to skip the two that need pseudo-depth precomputed first) — **but see
  "Dual-GPU + long-run hyperparameters" above: prefer
  `scripts/train_lora_all_families.sh` for the real multi-GPU run**, which
  wraps this script with one fresh process per family. Per family: fresh
  `attach_lora()` on the base checkpoint, pool all sequences via
  `build_pooled_coverage()` (calls `compute_coverage` once per sequence —
  **this is the slow part for large/spatially-dense sequences**, see
  Known caveats), wrap in `DUST3RSplattingDataset`, `L.Trainer.fit`
  (DDP, 2 GPUs, ~100 epochs), save adapter to `checkpoints/lora/<family>/`
  (plus periodic intermediate saves, see above).
- `scripts/eval_lora_scenes.py` — loads the base checkpoint once, then
  for each family: picks a representative scene (`val_data.sequences[0]`
  — edit that line for a specific scene by name instead), renders 3
  held-out (context, context, target) triples through the real
  `DecoderSplattingCUDA` rasterizer (same code path
  `training_step`/`validation_step` use — `model.forward` +
  `model.decoder` + `model.calculate_loss`, not a novel/untested render
  path), hot-swaps in that family's LoRA via `PeftModel.from_pretrained`,
  renders again, reports MSE/PSNR/LPIPS for both and saves
  base/lora/ground-truth PNGs to `logs/lora_eval/<family>/<scene>/`.

### Run order

```bash
cd /home/share-v5/Codes/Splatt3R-SLAM

# 1. Only needed once, before training euroc/eth3d (tum/7-scenes need no prep):
python3 scripts/precompute_pseudo_depth.py

# 2. Train (all 4 families, dual-GPU DDP, one fresh process per family):
bash scripts/train_lora_all_families.sh

# 3. Evaluate (needs step 2 done for all families it checks):
python3 scripts/eval_lora_scenes.py

# 4. Try a trained LoRA in the actual interactive SLAM pipeline:
python3 main.py --dataset datasets/tum/rgbd_dataset_freiburg1_desk --lora checkpoints/lora/tum
```

### Verification actually performed at script-writing time (that round ran no training/inference, per explicit user instruction — training runs happened later, see above)

- All four loaders' **construction** (file parsing, timestamp
  association, quaternion→rotation, pose composition) verified against
  the real downloaded data: correct sequence counts, finite positions,
  rotation-matrix orthonormality error <1e-3 for every sequence in every
  family (7-Scenes' `pose.txt`-direct matrices had the largest error,
  ~1e-4-1e-3, still fine — likely just recorded with less precision than
  the others' quaternions).
- TUM's **full** `DUST3RSplattingDataset` path (coverage → context/target
  sampling → `depthmap_to_absolute_camera_coordinates` → `pts3d`/
  `valid_mask`) verified end-to-end on `freiburg1_desk`: 84.0% of pixels
  valid, correct `context`/`target` counts (2/3, matching upstream), all
  expected dict keys present with correct shapes.
- 7-Scenes' loader verified the same way as TUM up through `get_view()`;
  the full `DUST3RSplattingDataset` pass wasn't completed for it in this
  session (its `chess_seq-01` coverage computation didn't finish inside a
  90s test budget — see Known caveats on coverage timing; this is a
  performance characteristic of that specific sequence, not a sign of a
  bug, since it's the exact same generic code TUM already proved correct).
- EuRoC/ETH3D verified through construction only (their `get_view()`
  needs the pseudo-depth cache, i.e. real inference, out of scope for
  this no-training-or-inference verification pass) — but the trickiest
  parts (wxyz quaternion, body→camera composition, undistortion,
  multi-scene pooling with a `max_scenes` cap) all checked out
  numerically sane.
- The full pipeline including actual model training (LoRA injection,
  `no_grad` fix, real `Trainer.fit`, adapter saving) was proven working
  **earlier in this session** on the single-scene `freiburg1_desk` proof
  of concept — see below. That code is what `train_lora_per_scene.py`
  generalizes; the generalization itself (multi-sequence pooling, 4
  families, hot-swap loading) is what's newly-verified-but-not-model-run
  in this section.

## Remaining / explicitly deferred

- **The eval script (`scripts/eval_lora_scenes.py`) has not been run** —
  the training scripts have (extensively: see the DDP crash fixes, the
  resume saga, and the metrics.csv collapse analysis above), but no
  base-vs-LoRA MSE/PSNR/LPIPS comparison exists yet. Don't report
  quality improvements that don't exist.
- **Hyperparameters are still a reasoned starting point, not a tuned
  recipe.** `NUM_EPOCHS_PER_EPOCH=1000` is the number of samples drawn
  per sequence per Lightning "epoch" (not optimizer steps), and
  `MAX_EPOCHS=100` was cross-checked against upstream's recipe rather
  than empirically tuned on these datasets; the later LR/clip retuning
  came from exactly one observed collapse. Treat each fresh run's
  val-loss curve as the thing to watch before trusting any quality
  claim.
- **7-Scenes' full `DUST3RSplattingDataset` pass** wasn't completed in
  verification (coverage computation for `chess_seq-01` ran past a 90s
  test budget) — see Known caveats below on coverage timing variance.
  Not a known bug, just not personally watched to completion.
- **EuRoC/ETH3D's `get_view()` (i.e. the pseudo-depth cache actually
  being read back correctly by `crop_resize_if_necessary` etc.)** wasn't
  exercised at all — needs `scripts/precompute_pseudo_depth.py` to have
  actually run first, which needs real inference, out of scope this
  round. First real risk surface to watch when these two families are
  actually run.
- **Stereo depth for EuRoC as a more-real alternative to pseudo-depth**
  — user explicitly chose pseudo-depth for both EuRoC and ETH3D over this
  (see the AskUserQuestion exchange this session); EuRoC ships full
  stereo calibration (`cam0`+`cam1` sensor.yaml) so real block-matching
  depth is possible if pseudo-depth's self-consistency-not-accuracy
  trade-off turns out to matter in practice.

## Known caveats

- **Coverage computation time varies a lot by sequence "spatial
  density."** `freiburg1_desk` (488 frames): 15.3s. A rotating/panning
  sequence like `freiburg1_360`, or a room-scanning `7-scenes` sequence
  like `chess_seq-01` (850 frames), didn't finish in 90s in this
  session's testing — many more frames end up within the 0.5m position
  prefilter when the camera stays roughly in place while turning, versus
  a sequence that keeps translating through a room. `train_lora_per_scene.py`
  computes this once per sequence per family per stage (train+val) — for
  a family with many such sequences this could legitimately take
  minutes, not seconds, before training even starts. Not a bug, just set
  expectations accordingly; lower `COVERAGE_POS_THRESHOLD` in that script
  if it's a problem in practice.
- **Pseudo-depth is self-consistency, not ground truth** (EuRoC/ETH3D
  only) — see above, don't let LoRA results on these two families be
  read as "trained against real geometry" the way TUM/7-Scenes can be.
- Per-scene/per-family overfitting is close to the actual goal here
  (re-run these exact benchmark scenes, look sharper) but won't
  generalize past the training data's viewpoints/scenes.
- Doesn't fix the colour-consistency root cause
  (`splatt3r-color-consistency` skill, Plan 3) — `mse`/`lpips` are
  single-view losses, nothing here penalizes cross-frame colour
  inconsistency specifically.
