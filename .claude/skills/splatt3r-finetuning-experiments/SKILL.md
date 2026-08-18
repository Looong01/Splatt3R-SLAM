---
name: splatt3r-finetuning-experiments
description: Controlled experiment record for fine-tuning the Splatt3R checkpoint on SLAM datasets — what was tried, what was measured, what failed and why. Covers the encoder-LoRA route (measured NEGATIVE, -49% vs base) and the frozen-encoder head-only route (measured POSITIVE, +1.00 dB psnr vs base), the LPIPS-weight follow-up, the root-cause analysis connecting them, and the measurement protocol that made the verdicts trustworthy. Read this BEFORE splatt3r-lora-finetuning (which documents the falsified route) and before starting any new fine-tuning run or writing this up.
metadata:
  type: reference
---

# Splatt3R fine-tuning: controlled experiment record

Written as paper-grade source material: every number here was measured, and
the measurement protocol is stated so results can be reproduced or contested.
Where a claim is inference rather than measurement, it says so.

> **Relationship to `splatt3r-lora-finetuning`**: that skill documents the
> engineering of the encoder-LoRA route in detail (data adapters, DDP fixes,
> resume machinery, dozens of real bugs). All of that infrastructure work
> remains valid and reusable. **Its central premise — adapting the encoder via
> LoRA — was subsequently falsified by direct measurement.** Read this file
> for the verdict; read that one for the engineering history and for the many
> "what breaks and why" lessons, which still hold.

---

## 1. Motivation

`splatt3r-gaussian-map` and `splatt3r-color-consistency` both concluded that
the released checkpoint (`epoch=19-step=1200.ckpt`, `brandonsmart/splatt3r_v1.0`)
has a real quality ceiling — blur that worsens with range, per-revisit colour
drift, "splash" artifacts — that SLAM-pipeline code can mitigate but not fix.
Note the checkpoint is *not* undertrained: upstream `configs/main.yaml` sets
`opt.epochs: 20`, so epoch 19 (0-indexed) is the final epoch of the officially
recommended run.

Goal: adapt the model per dataset family (TUM / 7-Scenes / EuRoC / ETH3D) so
each benchmark scene renders better than the released checkpoint does.

---

## 2. The measurement protocol (read this first — the verdicts depend on it)

Several months' worth of apparent progress in this project turned out to be
measurement artifacts. The protocol below is what finally produced trustworthy
numbers, and the failure modes it closes are each real, each hit in practice.

**Protocol**: 150 held-out TUM val samples (deterministic subset of the 900-sample
val split, `val_fraction=0.15`, same coverage cache as training), rendered
through the actual CUDA rasterizer at the family's deployment resolution
(512×384 for TUM), scored with `calculate_loss(apply_mask=True,
average_over_mask=True)`. Both models under comparison see the identical
sample list, resolution, decoder stride, and mask settings — **and, since the
fix in §2.1, the identical sample _contents_.**

Implemented in `scripts/exp_head_only.py: evaluate()`; the standalone
base-vs-LoRA verdict used the same construction.

### 2.1 Protocol defect found and fixed: a fixed index list is not a fixed sample

Fixing the index list was not enough. `DUST3RSplattingDataset.__getitem__`
re-samples its context/target view triplet on **every call**, through unseeded
`random.randint` / `random.choice` / `random.sample`
(`splatt3r_core/data/data.py:132,143,170`). The subset was therefore
deterministic in *index* only; each evaluation scored a different draw of
scenes and viewpoints.

The cost, measured directly — three evaluations of the **byte-identical base
checkpoint**:

| Draw | mse | psnr | lpips |
|---|---|---|---|
| head-only run, unseeded | 0.1065 | 9.7268 | 0.2801 |
| lpips run, unseeded | 0.1032 | 9.8648 | 0.2856 |
| seeded (`EVAL_SEED=1234`) | 0.0929 | 10.3221 | 0.2793 |

**≈0.6 dB of psnr spread from sampling alone.** Any claimed effect on the order
of 1 dB measured across two different draws is, on its face, only marginally
above this noise.

**The fix** (`evaluate()` in both experiment scripts): seed *before every
evaluation*, both `random.seed` and `torch.manual_seed`. The second is not
optional — with `num_workers>0` the sampling happens in worker processes, which
torch seeds as `base_seed + worker_id` where `base_seed` is drawn from torch's
global RNG at iterator construction. Seeding only `random` in the parent
changes nothing. Verified: two independent processes on two different GPUs then
scored base at 10.3221 / 0.2793 / 0.0929, identical to four decimals.

**Second-order trap in the fix itself.** Both seeds are *global*, and
`evaluate()` runs at the end of every training epoch. Seeding without
restoring resets the training RNG to the same state after each evaluation, so
every epoch would train on an identical draw — a silent confound that looks
like nothing at all in the logs. `evaluate()` therefore snapshots
`random.getstate()` / `torch.get_rng_state()` on entry and restores both in a
`finally`. The first version of the fix did not, and Route C had to be
restarted.

**Which conclusions survived.** Route B's headline was re-measured on the fixed
draw (`scripts/exp_rescore_seeded.py`, base and trained head on byte-identical
samples):

| | val/loss | mse | psnr | lpips |
|---|---|---|---|---|
| base | 0.1627 | 0.0929 | 10.3221 | 0.2793 |
| head-only | 0.1393 | 0.0738 | **11.3217** | **0.2620** |
| Δ | −14.4% | −20.6% | **+1.00 dB** | **−0.0173** |

The cross-draw estimate was +1.11 dB, the same-draw estimate is +1.00 dB — the
verdict holds and the magnitude barely moves. Route A's −2.09 dB / +0.11 lpips
is an order of magnitude above the noise and was never in doubt. What does
*not* survive is any reading of **per-epoch** movements: those were ~0.1 dB,
i.e. well inside a 0.6 dB sampling band, so the epoch-to-epoch trajectories
recorded in §5.1 should be read as trend-only. (This is the quantitative
justification for the caution in §5.2 about over-reading a single-epoch lpips
uptick — that caution turned out to be correct for a reason not understood at
the time.)

**For the paper**: report same-draw numbers only, state the seed, and never
compare a metric across two evaluations that were not seeded identically.

### Failure modes this protocol closes

| Failure mode | What went wrong | Evidence |
|---|---|---|
| **No absolute anchor** | Every val/loss in the first ~20 runs compared one LoRA checkpoint against another (each run resumed from the previous run's `best/`). The base checkpoint was **never** measured on the full val set. "New best 0.3021 → 0.3000 → 0.2925" looked like progress; all three were far *worse* than base's 0.1976. | §3 |
| **Cross-configuration comparison** | val/loss was compared across runs whose decoder `spatial_stride` differed (1 vs 2). Stride changes what is rendered, hence mse, hence loss. Measured: the *same* base model scores 0.1976 at stride=2 and 0.1765 at stride=1 — a 12% swing from a rendering knob alone. Any "improvement" smaller than that across a stride change is meaningless. | §5.2 |
| **Underpowered sample** | `scripts/eval_lora_scenes.py` uses 3 samples from one scene. On the same checkpoint it reported base 10.90 vs LoRA 10.68 psnr (≈tied) while the 150-sample protocol reported 9.50 vs 7.41 (−49%). **3 samples cannot resolve this effect size.** | §3 |
| **Changing the ruler** | Comparing val/loss across runs with different loss weights is invalid, since val/loss *is* the weighted sum. For the LPIPS-weight experiment (§6) the decision criterion is therefore stated on weight-independent quantities (psnr, lpips) and fixed before the run. | §6 |
| **Non-reproducible sample draw** | Pinning the val *indices* left the *contents* random — the dataset re-samples its view triplet on every `__getitem__`. The same base checkpoint scored psnr 9.7268 / 9.8648 / 10.3221 on three draws, a 0.6 dB band. Fixed by seeding `random` **and** `torch` before each evaluation. | §2.1 |
| **Scalars without images** | Metrics alone did not reveal that the LoRA failure was geometric rather than perceptual. Rendered base/model/GT triplets did, immediately. Always render. | §4 |

**Standing rule**: before believing any fine-tuning result, ask (a) is it
anchored to base, (b) measured under identical configuration, (c) on enough
samples, (d) with the same loss weights, (e) confirmed visually.

---

## 3. Route A — encoder LoRA: NEGATIVE (−49%)

**Configuration**: `peft` LoRA (r=8, α=16) on `target_modules=("qkv","proj","fc1","fc2")`,
which matches inside `patch_embed`, `enc_blocks` and `dec_blocks`; Gaussian head
fully fine-tuned via `modules_to_save`. Trainable 46.6M. `MAST3RGaussiansLoRA.forward`
removes the base class's `torch.no_grad()` so LoRA can receive gradients.

**Result** (150-sample protocol, stride=2, TUM):

| | val/loss | mse | psnr | lpips |
|---|---|---|---|---|
| base | **0.1976** | **0.1123** | **9.50** | **0.3414** |
| LoRA best (`val_loss.txt` 0.2925) | 0.2951 | 0.1814 | 7.41 | 0.4545 |
| | **+49.3%** | +61% | **−2.09 dB** | +33% |

Worse on every metric. Artifacts preserved at
`checkpoints/lora/tum_encoderLoRA_NEGATIVE_result/`.

### 3.1 Failure mechanism: scale explosion

Measured by running base and the trained adapter on an identical TUM image
pair (CPU, 512×384) and comparing predicted `scales` (largest axis per Gaussian):

| | median | p90 | p99 | p99.9 | max | fraction pinned at clamp (2.0) |
|---|---|---|---|---|---|---|
| base | 0.0039 | 0.0074 | 0.0128 | 0.0214 | 0.0231 | 0% |
| LoRA (5 epochs, LR 2e-6) | 0.0161 | 0.1547 | **1.0820** | **2.0000** | **2.0000** | **0.349%** |

Screen-coverage proxy Σ(scale²): **1821×** base.

This is causal for the OOMs and very likely for the project's long history of
`illegal memory access` crashes. The vendored rasterizer's `preprocess`
(`thirdparty/diff-gaussian-rasterization-modified/cuda_rasterizer/forward.cu`)
computes `my_radius = ceil(3*sqrt(max(λ1,λ2)))` from the 2D covariance and sets
`tiles_touched` from the resulting rect — **opacity plays no part in tile
assignment**. So `num_rendered` (Gaussian–tile pairs, which sizes
`geomBuffer`/`binningBuffer`/`imgBuffer`, all held for backward) is driven by
*scale*, not opacity. Growing scales ⇒ growing per-step live buffers ⇒ OOM.
Corollary: **culling by opacity does not address this**; culling by size, or
preventing the growth, does.

### 3.2 Aggravating bug: the clamp was a one-way ratchet

`lora.py` clamped scales to `[1e-6, 2.0]` before `build_covariance`. `torch.clamp`
has **exactly zero gradient outside the bound** (verified: input `[0.5, 1.9, 2.0, 3.0]`
→ grad `[1, 1, 1, 0]`). A Gaussian that reached 2.0 could therefore never receive
a "shrink" signal — the 0.349% pinned population was permanently stuck and could
only grow. The later `relu(s−T)²` penalty on **pre-clamp** scales was added to
restore that gradient.

### 3.3 Root cause: unfreezing what upstream deliberately froze

`splatt3r_core/main.py` — upstream's own code:

```python
self.encoder.requires_grad_(False)                                    # __init__
self.encoder.downstream_head1.gaussian_dpt.dpt.requires_grad_(True)
self.encoder.downstream_head2.gaussian_dpt.dpt.requires_grad_(True)

def forward(self, view1, view2):
    with torch.no_grad():                       # <-- encoder + decoder only
        ... _encode_symmetrized ... _decoder ...
    pred1 = self.encoder._downstream_head(1, ...)   # <-- OUTSIDE no_grad
```

Upstream trains **only the Gaussian head**. Two measured facts explain why
deviating from this is destructive:

1. **157,870,984 parameters** of matching/pointmap heads sit frozen on top of
   the same encoder (measured via `named_parameters()`). Perturbing the encoder
   feeds them out-of-distribution features they can never adapt to.
2. The Gaussian head is simultaneously chasing a moving feature distribution.

Both consumers degrade at once. This is inference from the architecture plus
the measured outcome, not a separately isolated experiment — a clean ablation
would LoRA the decoder only (leaving `_encode_image`, which the retrieval head
and matching consume, untouched). **Not run.**

---

## 4. Visual diagnosis (why images are mandatory)

Same held-out sample, base vs LoRA vs GT (`logs/lora_eval/tum/.../sample1_*.png`):

- **GT** — wooden floor, chair, table edge, cable. Sharp.
- **base** — structure intact: chair legs, table edge, red object all
  identifiable. Visible co-visibility patch boundary and some blur.
- **LoRA** — washed out under a **pink-orange haze**, chair nearly invisible,
  large cream blobs. Structure erased.

The haze is the direct visual signature of §3.1: oversized semi-transparent
Gaussians. This ruled out the then-current hypothesis (an exposure/colour
problem, motivated by a misreading of "mse tied, lpips worse" — mse was in fact
degraded *more*, +61% vs +33%).

---

## 5. Route B — frozen encoder, head-only: POSITIVE (+1.00 dB psnr)

> Headline numbers in this section were measured across *unseeded* draws and
> read "+16.5% val/loss". The same-draw re-measurement in §2.1 is the one to
> cite: **+1.00 dB psnr, −0.0173 lpips, −14.4% val/loss**. The verdict is
> unchanged; only the per-epoch detail below is trend-only.

**Configuration** (`scripts/exp_head_only.py`): upstream's recipe exactly.
`MAST3RGaussiansHeadOnly` reproduces `MAST3RGaussians.forward` (no_grad over
encoder+decoder, head outside) and adds only a numerical guard —
`nan_to_num` + clamp on scales, quaternion normalization — which is a no-op in
the healthy regime (base max scale 0.0231, three orders below the 2.0 ceiling)
and exists solely to stop a degenerate prediction from corrupting rasterizer
memory. **No scale penalty** (that patch existed only for the explosion this
configuration prevents).

Trainable **40,405,916 / 729,044,772 (5.54%)** — both Gaussian DPT heads.
AdamW, LR 1e-5 and grad-clip 0.5 (both upstream's values), batch 2,
stride 1, 900 steps/epoch, single GPU.

### 5.1 Result

| epoch | val/loss | mse | psnr | lpips |
|---|---|---|---|---|
| base | 0.1765 | 0.1065 | 9.7268 | 0.2801 |
| 0 | 0.1607 | 0.0958 | 10.1853 | 0.2595 |
| 1 | 0.1575 | 0.0927 | 10.3294 | **0.2590** |
| 2 | 0.1576 | 0.0907 | 10.4245 | 0.2678 |
| 3 | 0.1543 | 0.0885 | 10.5301 | 0.2633 |
| 4 | 0.1492 | 0.0831 | 10.8052 | 0.2644 |
| **5** | **0.1473** | **0.0825** | **10.8357** | 0.2593 |

**+16.5% val/loss, +1.11 dB psnr, −7.4% lpips vs base.** Beat base at epoch 0
— 900 steps, 8 minutes.

### 5.2 Secondary findings

- **Peak memory 6.3 GiB**, vs 42 GiB (OOM) for encoder-LoRA — a 6.7× reduction
  from `no_grad` alone. Encoder activations were the entire memory problem; the
  whole apparatus built to survive it (batch=1, stride=2, `expandable_segments`,
  retry wrappers, OOM watchdogs) is unnecessary in this configuration.
- **Same base model scores differently by stride**: 0.1765 (stride 1) vs
  0.1976 (stride 2). Quantifies failure mode #2 in §2.
- **Visual** (`logs/headonly_render/s0_*.png`, whiteboard/wood-panel scene):
  head-only removes base's blotchy splash artifacts on the whiteboard, edges
  are cleaner, cables and the ceiling-mounted camera are better formed —
  improvement in exactly the direction that motivated this work. **Trade-off
  visible**: wood-grain texture is smoother/softer than base, i.e. some fine
  detail is lost.

### 5.3 The mse/perceptual divergence

psnr rose monotonically (9.73 → 10.84) while **lpips stopped improving after
epoch 1** (0.2590 → 0.2593 across the following four epochs, oscillating in
0.259–0.268). Consistent with the visual softening in §5.2: the model is buying
pixel accuracy with blur — the known behaviour of an MSE-dominated objective
under uncertainty.

Actual loss contributions at epoch 3 (nominal weights are 1.0 : 0.25, but the
terms' magnitudes differ):

```
loss = 1.0 × 0.0885 (mse, 57%) + 0.25 × 0.2633 (lpips, 43%) = 0.1543
```

Less lopsided than the nominal 4:1 suggests. Motivates §6.

> **Methodological note against over-reading**: at epoch 2 the single-point
> lpips uptick (0.2590→0.2678) was read as a trend and "optimal early stopping
> at epoch 1-2" was proposed; epoch 3 reversed it (0.2633). The conclusion in
> this section rests on four epochs, not one point.

---

## 6. Route C — raising the LPIPS weight: PASSES, but the gain is small

`scripts/exp_head_only_lpips.py`, identical to Route B except
`lpips_loss_weight` 0.25 → **1.0** (flipping the contribution ratio to ≈1:3).
Rationale is §5.3 plus the fact that the project's motivating complaint is
*blur*, and MSE is the term that rewards blur.

**Decision criterion, fixed before the run** (val/loss is unusable across a
weight change — see §2):

> ACCEPT if lpips improves meaningfully over Route B **and** psnr stays at or
> above base.

Originally written against Route B's 0.2593 and base's 9.7268. Those were
unseeded draws (§2.1), so the run was restarted under the seeded protocol and
the criterion restated on the fixed draw, without changing its substance:

> ACCEPT if lpips beats Route B's **0.2620** **and** psnr stays at or above
> base's **10.3221**.

Route C's own in-run baseline reproduces the seeded base exactly
(0.0929 / 10.3221 / 0.2793), so its per-epoch numbers are directly comparable
to the Route B row in §2.1.

### 6.1 Result: criterion met

Per-epoch, all on the fixed seeded draw (n=150):

| epoch | val/loss | mse | psnr | lpips |
|---|---|---|---|---|
| base | 0.3721 | 0.0929 | 10.3221 | 0.2793 |
| 0 | 0.3578 | 0.0904 | 10.4383 | 0.2674 |
| 1 | 0.3534 | 0.0886 | 10.5252 | 0.2648 |
| 2 | 0.3460 | 0.0843 | 10.7433 | 0.2618 |
| 3 | 0.3391 | 0.0806 | 10.9351 | 0.2585 |
| 4 | 0.3402 | 0.0801 | 10.9625 | 0.2601 |
| **5** | **0.3328** | **0.0763** | **11.1722** | **0.2565** |

Three-way comparison on identical samples:

| | mse | psnr | lpips |
|---|---|---|---|
| base | 0.0929 | 10.3221 | 0.2793 |
| Route B (lpips 0.25) | 0.0738 | **11.3217** | 0.2620 |
| Route C (lpips 1.0) | 0.0763 | 11.1722 | **0.2565** |

Both criterion clauses hold (lpips 0.2565 < 0.2620; psnr 11.1722 > 10.3221),
so Route C is **accepted as stated**. What it actually buys is a trade:
**−0.15 dB psnr for −0.0055 lpips.**

Is 0.0055 meaningful? Val-sampling noise is now eliminated by seeding, so the
remaining question is training-run variance. Two same-config runs differing
only in training draw scored 0.2661 vs. 0.2674 at epoch 0 — about 0.0013. The
gain is ≈4× that, so it is real, but it is not large.

### 6.2 The images do not separate B from C

`scripts/exp_render_compare.py` renders GT / base / B / C from the identical
seeded sample. What is clearly visible:

- **base → B or C is a large, obvious improvement.** Base shows magenta/pink
  blotching across flat surfaces (whiteboard, ceiling) and a gray haze washing
  out the floor and desk edge. Both trained heads remove essentially all of it
  and recover cable and chair-leg detail that base smears away.
- **B → C is not discernible by eye.** On the samples inspected, the two are
  near-identical; C is arguably marginally cleaner on one desk edge, which is
  not evidence.

So the honest reading: the lpips-weight change is a *measurable* but
*sub-perceptual* refinement at this training length. It does not resolve the
blur complaint that motivated it. **Do not report Route C as a visual win.**

### 6.3 The more important observation

**Neither route had converged at epoch 5** — B and C both improved
monotonically on mse and psnr right through the last epoch, with only a single
lpips wobble (C's epoch 4). Six epochs was chosen to make an epoch take minutes.
The available headroom is therefore in **training longer**, not in tuning the
loss weight; the B-vs-C question should be re-asked at convergence, where it may
resolve differently or stop mattering.

Log: `logs/exp_head_only_lpips.log`. Renders: `logs/render_compare/`.

---

## 7. Hardware utilization: the VRAM headroom is not convertible

Training peaks at 6.3 GiB on a 49 GiB A6000, which invites the obvious
question of why the batch size is not raised to fill it. Measured, on three
families independently (`scripts/exp_batch_scan.py`, 3 warmup + 12 timed steps,
median):

| batch | 7-scenes | euroc | eth3d | peak mem (7-scenes) | nvidia-smi |
|---|---|---|---|---|---|
| 2 | 1.00x | 1.00x | 1.00x | 6.26 GiB | ~10 GB |
| 4 | 1.05x | 1.02x | 1.04x | 9.73 GiB | |
| **8** | **1.08x** | **1.12x** | **1.05x** | 15.77 GiB | |
| 16 | 1.05x | 1.01x | 1.02x | 27.97 GiB | |
| 20 | **0.96x** | | | 34.41 GiB | 47.5 GB |
| 24 | **0.93x** | | | 40.97 GiB | ~48 GB |
| 32 | OOM | | | — | — |

**Throughput peaks at batch=8 and declines after it. Filling VRAM (batch
20/24) is measurably SLOWER than batch=2** — 4–7% slower, while consuming
47–48 GB of the 49 GB card. The GPU is already saturated at batch=2 —
consistent with the 89% utilization measured during the TUM run — because one
sample is already a 512×384 ViT encoder forward plus a rasterizer pass. There
is no parallelism left for batching to fill.

Two further reasons not to chase VRAM occupancy:

- **`max_memory_allocated` understates real usage by ~3.7 GB** (allocator
  accounting vs. CUDA context and fragmentation): the run reporting 6.26 GiB
  shows ~10 GB in `nvidia-smi`. Budget from `nvidia-smi`, not from the script.
- **Peak memory is data-dependent.** The 3DGS rasterizer's buffers scale with
  `num_rendered`, which depends on the predicted Gaussian scales, so a batch
  that fits today can OOM twenty epochs in. Batch 20 leaves ~1.5 GB of margin,
  which is not margin. (Route A's scale explosion is the documented case of
  those buffers running away — §3.1.)

**Conclusion: batch=8 is the measured optimum; batch=2 is within 8% of it and
is what all the reported results use.**

### 7.1 If the batch is raised anyway: what has to move with it

Batch size is not a free knob, so the scaling rules were fixed in
`exp_head_only.configure_family()` rather than guessed per run:

- **LR scales as `lr * sqrt(batch/2)`, not linearly.** Linear scaling (Goyal
  et al.) was derived for SGD with momentum; AdamW already normalizes the
  update by the gradient's second moment, so linear overshoots. It would also
  be actively dangerous here — linear scaling from batch 2 to 24 lands on
  1.2e-4, and the encoder-LoRA runs collapsed at 1e-4 (§3). The 7-scenes run
  uses `1e-5 * sqrt(8/2) = 2e-5`.
- **Warmup switches on with the batch increase** (100 linear steps, 0 at
  batch 2). A sqrt-scaled LR applied from step 0, to a head whose optimizer
  state is still empty, is the standard way to damage a pretrained
  initialization in the first few updates.
- **Optimizer-step accounting, which is the part that is easy to miss.** At
  equal samples seen, a larger batch takes proportionally fewer updates, and
  for the smaller families that is severe:

  | run | samples/epoch | steps/epoch | 40 epochs: samples | 40 epochs: steps |
  |---|---|---|---|---|
  | TUM, batch 2 | 1800 | 900 | 72,000 | **36,000** |
  | 7-scenes, batch 8 | 1400 | 175 | 56,000 | **7,000** |

  Comparable in samples, **5× fewer updates**. If the 7-scenes run plateaus
  early, under-optimization is the first hypothesis to test (raise
  `SAMPLES_PER_SEQ`, or add epochs) — not a lack of headroom in the model.
  This run doubles as the measurement of what large-batch training actually
  costs here.

The corollary: the second GPU should run a *second family*, not a bigger batch
on the same one. That is embarrassingly parallel — no NCCL, no collectives, no
new failure modes, and it sidesteps the DDP integration bugs that cost this
project weeks (rank1 never starting because `sys.argv[0]` was resolved after
`os.chdir`; a 30-minute NCCL hang when an `assert` killed one rank while the
other waited on all-reduce). DeepSpeed tensor parallelism is not an
alternative worth pursuing here: TP exists for models that do not fit on one
GPU, it adds two collectives per transformer layer per direction (strictly more
communication than DDP, not less), its training path requires Megatron-style
parallel layer definitions, and the custom CUDA rasterizer in the forward pass
cannot be tensor-sharded at all.

### 7.2 The scan's peak memory is a sample, not a bound — confirmed the hard way

The EuRoC run was launched at batch 8, where `exp_batch_scan.py` measured a
13.37 GiB peak on a 47 GiB card. **It OOMed at epoch 6** trying to allocate
2.19 GiB, with 46.07 GiB already in use — i.e. it grew past a 34 GiB headroom.

The cause is the one §7 already flagged but understated: the 3DGS rasterizer's
buffers scale with `num_rendered`, which depends on the *predicted Gaussian
scales*, so peak memory is data-dependent. A 15-step scan samples typical
batches; a multi-thousand-step run eventually meets an atypical one. The
earlier wording ("batch 20 leaves ~1.5 GB of margin, which is not margin") was
right in spirit and far too weak in degree: **34 GiB of margin was also not
enough.**

Practical rule: on this model, batch size cannot be chosen from a short
throughput scan. Since batch>2 buys ≤12% anyway (§7), the safe choice is also
the fast one. EuRoC was restarted at batch 4 (lr 1.41e-5 = 1e-5·√2) and runs
without incident.

**Caveat on these numbers**: they were measured while a training run occupied
the other GPU and 8 dataloader workers, so CPU/disk contention inflates
s/step — most for the largest batches, which biases *against* big batch. The
conclusion is robust to that (12% cannot become meaningful), but the absolute
s/step figures are not clean.

---

## 8. Reusable assets

| Path | Purpose |
|---|---|
| `scripts/exp_head_only.py` | Route B trainer + the 150-sample protocol (`evaluate()`) |
| `scripts/exp_head_only_lpips.py` | Route C, one variable changed |
| `scripts/exp_rescore_seeded.py` | Scores base vs. a saved head checkpoint on the fixed seeded draw — the §2.1 re-validation |
| `scripts/exp_render_compare.py` | Renders GT / base / route B / route C from the identical seeded sample |
| `scripts/exp_batch_scan.py` | Throughput/peak-memory vs. batch size, per family (§7) |
| `scripts/eval_map_quality.py` | SLAM-level map scoring: Sim3-align, render from held-out poses (§9.2) |
| `scripts/refine_gaussian_map.py` | Per-scene 3DGS refinement, random-init control, view-count sweep (§10, §13) |
| `scripts/eval_cross_family.py` | Every trained head x every family's val split — the generalization matrix (§11) |
| `scripts/diag_psnr.py` | Channel-normalization / mask-coverage diagnosis of the PSNR scale (§2) |
| `splatt3r_slam/gaussian_ply_codec.py: decode_gaussians_from_ply` | Map .ply -> pre-activation params (optimize) + activated values (render) |
| `scripts/precompute_coverage.py` | Per-family coverage matrices — the prerequisite for training a new family |
| `scripts/run_gpu1_queue.sh` | Serial work queue for the idle GPU |
| `checkpoints/lora_coverage_cache/` | Coverage caches: tum, 7-scenes, euroc, eth3d (all four present) |
| `checkpoints/head_only_lpips/tum/head_best.pt` | Route C weights (same loading convention as route B) |
| `logs/render_compare/` | The base-vs-B-vs-C renders |
| `checkpoints/head_only/tum/head_best.pt` | Route B weights (head `state_dict`; load with `encoder.load_state_dict(..., strict=False)`) |
| `checkpoints/lora/tum_encoderLoRA_NEGATIVE_result/` | Route A artifacts, kept as the negative result |
| `logs/exp_head_only.log`, `logs/headonly_render/` | Route B metrics and renders |
| `logs/lora_eval/tum/` | Route A renders (the haze) |

The per-family data adapters, coverage-matrix computation, pseudo-depth
precompute and resolution derivation from `splatt3r-lora-finetuning` are all
route-independent and remain the correct foundation.

### 8.1 Runs in flight

| run | script + flags | log | out |
|---|---|---|---|
| TUM, 40 epochs, batch 2, lr 1e-5 | `exp_head_only.py --epochs 40 --out checkpoints/head_only_long/tum` | `logs/exp_head_only_long.log` | `checkpoints/head_only_long/tum/` |
| 7-scenes, 40 epochs, batch 8, lr 2e-5, warmup 100 | `exp_head_only.py --family 7-scenes --batch 8 --epochs 40 --out checkpoints/head_only_long/7-scenes` | `logs/exp_head_only_7scenes.log` | `checkpoints/head_only_long/7-scenes/` |

One family per GPU, no inter-process communication — the parallelization
strategy §7 argues for. 7-scenes' base scores psnr 8.9434 / lpips 0.2820
(n=175), notably worse than TUM's base, so its headroom may be larger.

#### Live progress

Regenerate with `python3 scripts/update_skill_progress.py` — it parses the
logs and rewrites the block below. Do not hand-edit it: two hand-transcribed
updates (TUM epochs 22 and 27) reported values that appear nowhere in either
log and had to be retracted.

<!-- PROGRESS:BEGIN -->

*Regenerated 2026-07-31 13:38 by `scripts/update_skill_progress.py` — parsed from the logs, not transcribed.*

**TUM** (batch 2, lr 1e-5) — 40/40 epochs, `logs/exp_head_only_long.log`

| metric | base | best | Δ | at |
|---|---|---|---|---|
| psnr | 15.0911 | **16.8684** | **+1.78 dB** | epoch 36 |
| lpips | 0.2793 | **0.2508** | **-10.2%** | epoch 37 |
| mse | 0.0929 | **0.0617** | **-33.6%** | epoch 36 |
| val/loss | 0.1627 | **0.1249** | **-23.2%** | epoch 33 |

**7-Scenes** (batch 8, lr 2e-5, warmup 100) — 40/40 epochs, `logs/exp_head_only_7scenes.log`

| metric | base | best | Δ | at |
|---|---|---|---|---|
| psnr | 13.7161 | **15.8950** | **+2.18 dB** | epoch 35 |
| lpips | 0.2820 | **0.2573** | **-8.8%** | epoch 37 |
| mse | 0.1275 | **0.0772** | **-39.5%** | epoch 35 |
| val/loss | 0.1980 | **0.1421** | **-28.2%** | epoch 37 |

**EuRoC** (batch 4, lr 1.41e-5, warmup 100 (batch 8 OOMed, see §7.2)) — 40/40 epochs, `logs/exp_head_only_euroc_b4.log`

| metric | base | best | Δ | at |
|---|---|---|---|---|
| psnr | 11.4408 | **15.1286** | **+3.69 dB** | epoch 35 |
| lpips | 0.3011 | **0.2499** | **-17.0%** | epoch 36 |
| mse | 0.2153 | **0.0921** | **-57.2%** | epoch 35 |
| val/loss | 0.2906 | **0.1549** | **-46.7%** | epoch 35 |

**ETH3D** (batch 4, lr 1.41e-5, warmup 100) — 40/40 epochs, `logs/exp_head_only_eth3d.log` — **DIVERGED to NaN at epoch 28; 12 dead epochs follow. Best below is from before divergence and the saved checkpoint is valid.**

| metric | base | best | Δ | at |
|---|---|---|---|---|
| psnr | 10.8548 | **14.0012** | **+3.15 dB** | epoch 25 |
| lpips | 0.3706 | **0.3268** | **-11.8%** | epoch 23 |
| mse | 0.2464 | **0.1194** | **-51.5%** | epoch 25 |
| val/loss | 0.3390 | **0.2012** | **-40.6%** | epoch 25 |

<!-- PROGRESS:END -->

#### TUM 40-epoch run: FINISHED

The single most valuable change in this whole series was simply **training
longer**, which §6.3 predicted and this confirms:

| | psnr | lpips | mse | val/loss |
|---|---|---|---|---|
| base | 10.3221 | 0.2793 | 0.0929 | 0.1627 |
| 6 epochs (route B) | 11.3217 | 0.2620 | 0.0738 | 0.1393 |
| **40 epochs** | **12.0945** (ep36) | **0.2508** (ep37) | **0.0617** (ep36) | **0.1249** (ep33) |
| Δ vs base | **+1.77 dB** | **-10.2%** | **-33.6%** | **-23.2%** |

Note the best epoch differs per metric (36/37/36/33) — see the plateau warning
below for why that is expected here, and why "converged at epoch N" is not a
claim this data supports.

**Images** (`logs/render_compare/`, `scripts/exp_render_compare.py`, identical
seeded sample across all four columns):

- **base → 40-epoch is unambiguous.** The magenta blotching on flat surfaces
  and the gray haze washing out floors are gone; cables, chair legs and the
  sticky note recover real detail.
- **6-epoch → 40-epoch is visible but modest.** The clearest difference is the
  black speckle fringe along the desk edge, which the 6-epoch model still shows
  and the 40-epoch model largely resolves.

So the scalar gain is backed by images, but the *second* +0.77 dB buys much
less visible change than the first +1.00 dB did.

#### Reading the epoch-to-epoch trace: do not call plateaus

Both runs oscillate enough that short-window trend calls have been wrong every
time they were made. Documented instances, all on the *seeded* draw (so these
are real weight changes, not the sampling noise of §2.1):

| call | basis | what happened |
|---|---|---|
| "optimal early stopping at epoch 1-2" (route B) | one lpips uptick | reversed at epoch 3 |
| "lpips is degrading, a trend not a wobble" (7-scenes) | 3 consecutive rises | reversed at epoch 3 |
| "psnr has gone flat" (TUM) | 5 epochs without a new high (ep25-29) | new high at epoch 30 |

The pattern: **mse/psnr and lpips regularly move in opposite directions on a
single epoch**, and runs sit for five or more epochs before breaking out. A
plateau claim here needs a much longer window than intuition suggests, and
ideally a rendered comparison rather than a scalar. For the paper, report
best-so-far per metric with the epoch it occurred at, not "converged at N".

---

## 9. SLAM-level validation: the gain survives fusion, ATE is untouched

Everything in §2-§6 scores a single two-view prediction. The deployed
objective is SLAM behaviour, so both halves were measured.

### 9.1 ATE: bit-identical, as predicted

| sequence | base | head-only | Δ |
|---|---|---|---|
| freiburg1_room | 0.059027 | 0.059027 | 0 |
| freiburg1_360 | 0.042079 | 0.042079 | 0 |
| freiburg1_desk | 0.016975 | 0.016975 | 0 |

`evo_ape tum -as`, `scripts/eval_head_ate.sh`. Not merely close — the
trajectory files are **byte-identical** (`cmp` clean).

This is the clean confirmation of the route's central property: freezing the
encoder leaves features, matching, tracking and the pose graph untouched, so
localization *cannot* change. Previously an inference; now a measurement.

It also removes a confound from §9.2: since the poses are identical, any map
improvement is attributable to the Gaussian head alone, not to better
trajectories.

### 9.2 Map quality: +0.90 dB, with 16% fewer Gaussians

`scripts/eval_map_quality.py` — Sim3-align the estimate to ground truth
(Umeyama, scale estimated), map ground-truth poses into the map frame, then
render the persisted `.ply` from frames the SLAM run never selected as
keyframes (scoring on keyframes would be a self-consistency check, not NVS).

| sequence | psnr base → head | Δ | lpips | Δ | Gaussians |
|---|---|---|---|---|---|
| room | 10.43 → 11.19 | **+0.76 dB** | 0.581 → 0.543 | −6.7% | −16% |
| 360 | 11.57 → 11.79 | +0.22 dB | 0.502 → 0.488 | −2.9% | −10% |
| desk | 10.71 → 12.44 | **+1.73 dB** | 0.559 → 0.503 | −10.0% | −22% |
| **mean** | | **+0.90 dB** | | **−6.5%** | **−16%** |

Three things worth reporting:

- **The per-pair gain attenuates but survives.** +1.78 dB on held-out pairs
  becomes +0.90 dB on the fused map. Expected: a map averages dozens of
  keyframes, and map rendering scores the *whole* frame including regions the
  map never covered, neither of which the pair protocol does.
- **Fewer Gaussians, better maps.** The fine-tuned head yields 16% fewer
  Gaussians after `save_gaussian_map`'s confidence/opacity filtering — i.e.
  fewer low-confidence junk primitives. For SLAM this is a second, independent
  win: smaller maps, faster rendering.
- **The spread is structural, not noise.** desk (+1.73 dB, close-range
  textured desktop) gains most; 360 (+0.22 dB, near-pure rotation, almost no
  parallax) gains least. Where two views cannot recover geometry in the first
  place, a better head has little to work with. Consistent with the gain
  coming from better Gaussian *parameters*, not more information.

### 9.3 Two measurement bugs found while building this

Both of the "runs fine, answer is completely wrong" variety, same family as
§2.1:

- **`resize_img` returns `ImgNorm`-normalized images in [-1,1]**, while the
  rasterizer emits [0,1]. Comparing them directly scored a correctly-rendered
  map at **3.16 dB**.
- **The Sim3 inverse folded 1/s into the rotation block** (`R.T / s`), giving
  a non-orthonormal "rotation" that silently corrupted every viewpoint. The
  scale belongs to the translation only.

---

## 10. Per-scene 3DGS refinement: comparative claims ON HOLD (the control arm is degenerate — see 10.6)

`scripts/refine_gaussian_map.py`. The map produced by SLAM is used as the
initialization for standard per-scene 3DGS optimization (INRIA
parameterization: free variables are log-scales, logit-opacity, raw quaternion
and SH DC; per-parameter LRs from the reference implementation, positional LR
scaled by scene extent; loss 0.8*L1 + 0.2*(1-SSIM)).

Scored on freiburg1_desk, held-out frames only (the same non-keyframe split
§9.2 uses), 1.86M Gaussians, 1.6 GiB peak:

| iters | map init psnr | map init lpips | random init psnr | random init lpips |
|---|---|---|---|---|
| 0 | 12.3719 | 0.5185 | 9.3374 | 0.9230 |
| 500 | 15.9497 | 0.4048 | 11.4835 | 0.7481 |
| 1500 | 16.3301 | 0.3548 | 11.7484 | 0.6554 |
| 3000 | **16.3437** | **0.3457** | **11.7835** | **0.6097** |

**Per-scene optimization adds +3.97 dB** over the SLAM map (12.37 -> 16.34).

**The control arm is the point.** Random initialization -- same Gaussian count,
uniform positions in the map's bounding box, random colours -- reaches only
11.78. The gap does not close under optimization, it *widens*: 3.03 dB at
initialization, 4.56 dB after 3000 iterations. Good initialization is not a
head start, it determines the level the run converges to.

### 10.1 The caveat that must ship with this number

**Densification is disabled in both arms.** That was deliberate -- enabling
INRIA's adaptive density control in the first experiment would make it
impossible to attribute a gain to initialization rather than to densify/prune.
But it also handicaps the random arm specifically and severely: vanilla 3DGS
from random or sparse-SfM points *relies* on densification to grow geometry
that the initialization lacks. Without it, the random arm has no mechanism to
create structure that was never there.

So the defensible claim is:

> **Under pure parameter optimization (no densification), Splatt3R's geometric
> initialization is decisive: +4.56 dB over random initialization at equal
> Gaussian count and equal iterations.**

Not "Splatt3R init beats random init by 4.56 dB" — that overstates, and a
reviewer closes it in one sentence. A third arm (**random + densification**) is
required before any stronger statement.

### 10.2 CORRECTED by the view-count sweep — §10 above is the 14-view special case

The conclusion in §10 ("initialization is decisive, +4.56 dB, and optimization
*widens* the gap") was drawn at a single training-view count, the ~14 keyframes
of freiburg1_desk, and does not generalize. `scripts/refine_gaussian_map.py
--train-source all --n-train N`, 3000 iterations, 50 held-out frames that
participate in neither optimization nor the Sim3 fit:

| views | map psnr | random psnr | Δpsnr | map lpips | random lpips | Δlpips |
|---|---|---|---|---|---|---|
| 14 | 16.6456 | 11.9081 | **+4.74** | 0.3345 | 0.5711 | +0.237 |
| 50 | 18.9740 | 14.9737 | **+4.00** | 0.3192 | 0.5129 | +0.194 |
| 150 | 19.0713 | 18.5936 | **+0.48** | 0.3336 | 0.4382 | +0.105 |
| 400 | 19.1311 | 18.6886 | **+0.44** | 0.3316 | 0.4370 | +0.105 |

**Both arms converge to essentially the same psnr (~19 dB).** Initialization
does not set the quality ceiling — it sets how many views are needed to reach
it:

- map init saturates at **~50 views** (150 -> 400 adds 0.06 dB)
- random init saturates at **~150 views**
- final ceilings differ by 0.44 dB

**The perceptual gap, however, does not close.** At 400 views, with psnr
effectively tied, random-init lpips is still **32% worse** (0.4370 vs 0.3316),
and it stops narrowing after 150 views. Given enough views, pure optimization
matches the pixel error but not the perceptual quality: it finds a solution
that fits pixels with the wrong structure. Same mse/perceptual divergence seen
throughout this project (§6.3).

**The defensible claim:**

> Splatt3R initialization buys **~3x data efficiency** (50 vs 150 views to
> reach the same psnr) and a **perceptual advantage that persists at every
> view count** (lpips 24-41% lower). It does **not** raise the final psnr
> ceiling.

Confound to state: the iteration budget is fixed at 3000. At n=50 the map arm
was still gaining 0.56 dB between iterations 1500 and 3000, so the low-view
rows are not fully converged; by n=400 both arms are flat within the budget, so
the endpoint comparison is fair.

**Process note.** The external review (§12) warned specifically that adding
views might shrink the gap and undercut the claim. That warning was verbally
accepted here and then not acted on in interpretation — the intermediate
`n=50` result was read as "the gap narrows too slowly to matter, closing 4 dB
would need far more than the 563 available frames." It took 150. Extrapolating
a trend from two points, in a project whose entire history is measurement
artifacts, was the wrong move twice over.

### 10.3 The 2x2 factorial: densification does NOT rescue random initialization

The review's sharpest objection (§12) was that comparing initializations with
densification *off* stacks the deck: vanilla 3DGS from random points relies on
adaptive density control to grow geometry, so switching it off is a known
handicap for the random arm. The missing cells were run
(`refine_gaussian_map.py --densify`, INRIA schedule, identical for both arms).
desk, 3000 iterations, 50 held-out frames, psnr / lpips:

| | densify OFF | densify ON | Δpsnr |
|---|---|---|---|
| n=14, map | 16.6456 / 0.3345 | 16.0554 / 0.3523 | −0.59 |
| n=14, random | 11.9081 / 0.5711 | 10.0093 / 0.6158 | **−1.90** |
| **init gap** | **4.74 dB** | **6.05 dB** | |
| n=50, map | 18.9740 / 0.3192 | 18.5483 / 0.3328 | −0.43 |
| n=50, random | 14.9737 / 0.5129 | 12.0239 / 0.5538 | **−2.95** |
| **init gap** | **4.00 dB** | **6.52 dB** | |

**Turning densification on does not close the gap — it widens it**, from
4.74/4.00 dB to 6.05/6.52 dB. The map arm is barely affected (−0.4 to −0.6 dB);
the random arm loses 1.9-3.0 dB. So the objection does not survive
measurement: in the few-view regime — which is exactly the SLAM regime —
densification cannot substitute for a good initialization.

This must be reported together with §10.2, which bounds the claim: at 150+
views and no densification, random init reaches 18.59 vs. map's 19.07. The
complete picture is

> With plentiful views, pure optimization reconstructs comparable geometry from
> random initialization (0.44 dB behind, though still 32% worse on lpips). With
> few views, Splatt3R initialization leads by 4-6.5 dB, **and densification
> does not recover it**.

#### Three implementation faults behind this table — all found after they had produced "results"

This factorial was run three times; the first two datasets were discarded.

1. **The opacity reset ran before the evaluation**, and its interval (3000)
   equalled the run length (3000), so every densify cell reported its final
   number from a map whose opacities had just been capped at 0.01 — 7.18 dB at
   iteration 3000 where iteration 1500 read 14.42.
2. **Adam's `exp_avg`/`exp_avg_sq` were zeroed for every Gaussian on every
   densification round** (i.e. every 100 iterations) while the no-densify
   control kept its momentum throughout. This manufactured a ~5 dB penalty
   and a characteristic rise-to-1500, fall-by-3000 curve.
3. On the strength of (1) and (2), three plausible mechanisms were written up
   — that the pruning criterion needs view coverage, that densification has a
   view-count threshold, that it "makes structural decisions on noise." **All
   three were explanations of a bug.**

What eventually triggered suspicion was not that the mechanisms were wrong but
that **the numbers were too tidy**: two view counts and two initializations all
losing ≈5 dB with identical curve shapes. Real effects are rarely that uniform.
That heuristic — uniformity across conditions as a bug smell — is worth keeping.

### 10.4 Multi-scene replication: the psnr story does NOT generalize; the lpips story does

Both §10.2 and §10.3 were measured on freiburg1_desk alone. Repeating them on
room and 360 (`logs/multiscene.log`, same protocol, 3000 iterations, 50
held-out frames) breaks the psnr conclusion.

**Initialization gap (psnr, map − random), densification off:**

| sequence | n=14 | n=150 | direction |
|---|---|---|---|
| desk | +4.74 | **+0.48** | collapses |
| room | +2.26 | **+4.67** | **widens** |
| 360 | +1.67 | +1.73 | flat |

Three scenes, three different behaviours. **§10.2's claim — that both arms
converge to the same psnr and initialization only buys data efficiency — is a
desk-specific result.** Note what happened procedurally: §10.2 was itself
written to correct §10's single-view-count conclusion, and it did so using a
single *scene*. One anecdote was used to correct another.

**What does replicate is the perceptual gap**, at n=150:

| sequence | map lpips | random lpips | gap |
|---|---|---|---|
| desk | 0.3336 | 0.4382 | +31% |
| room | 0.4602 | 0.7337 | **+59%** |
| 360 | 0.4440 | 0.7059 | **+59%** |

Consistent across three scenes, both view counts, and with or without
densification, and *larger* on the two scenes desk had understated.

**Revised claim, and the only one currently supported by more than one scene:**

> Splatt3R initialization yields a large and consistent perceptual advantage
> (lpips 31-59% lower than random initialization at equal Gaussian count and
> equal iterations). Its psnr advantage is strongly scene-dependent: between
> +0.5 and +4.7 dB at 150 views, with no consistent trend in view count.

#### A third implementation fault: the Gaussian cap silently disabled densification

`MAX_GAUSSIANS` was set to 6M with the comment "the maps here already start at
~2M" — sized from desk (1.86M). room's map is **7.35M** and 360's is **7.27M**,
both already over the cap, so `room = MAX_GAUSSIANS - N` went negative and
`clone_mask`/`split_mask` were forced empty. Every densify round on those scenes
logged `clone=0 split=0 pruned=1004`: **prune-only, not densification.** The
eight room/360 densify cells of the first multi-scene sweep are void and were
re-run with a 12M cap.

This is the same failure shape as the other two in §10.3: a constant chosen from
one scene, silently degrading an experiment into something else. The cap now
prints an explicit warning when it binds rather than quietly changing what is
being measured.

### 10.5 Seed variance: the random arm's numbers carry +-0.66 dB; the map arm's do not

Every refinement number above is single-run. Three seeds on desk, n=150
(`logs/seeds_desk.log`) show the two arms are not comparably stable:

| seed | map | random | Δpsnr | Δlpips |
|---|---|---|---|---|
| 0 | 19.0854 / 0.3339 | 17.2713 / 0.4670 | +1.81 | +39.9% |
| 1 | 19.1201 / 0.3373 | 18.4976 / 0.4430 | +0.62 | +31.3% |
| 2 | 19.1569 / 0.3352 | 18.2887 / 0.4444 | +0.87 | +32.6% |

| | mean | sigma | range |
|---|---|---|---|
| map psnr | 19.1208 | **0.036** | 0.07 |
| random psnr | 18.0192 | **0.656** | 1.23 |
| Δpsnr | **+1.10** | 0.63 | +0.62 … +1.81 |
| map lpips | 0.3355 | 0.0017 | |
| random lpips | 0.4515 | 0.0135 | |

**The map arm is effectively deterministic (sigma 0.036 dB — only the frame
order varies); the random arm's sigma is 18x larger.** Its initialization is
redrawn every run, and `build_random_init` uses the unseeded global torch
generator, so `--seed` controls frame sampling only.

**Consequences for what is already written here:**

- §10.2's headline for desk at n=150 was **+0.48 dB**, reported from one run.
  The 3-seed mean is **+1.10 dB**, and 0.48 lies *outside* the 3-seed range
  (0.62-1.81), so the true variance exceeds even this estimate. 0.48 supports
  "the gap essentially closes"; 1.10 does not. **That claim is downgraded
  again.**
- Every random-arm entry in the multi-scene table (§10.4) is a single run and
  carries ~±0.7 dB. desk's +0.48 and 360's +1.67/+1.73 are all within that
  band; only room's +4.67 clearly exceeds it. The "three scenes, three
  behaviours" reading is not supported at this precision — a 3-seed re-measure
  of the random arm is running.
- The room/n=150 densification loss of **4.9 dB** is far outside the band and
  stands.
- **lpips survives cleanly**: the random/map ratio is +39.9 / +31.3 / +32.6%
  with sigma 0.0135 on the random arm. This is now the one advantage verified
  across scenes, view counts, densification settings, *and* seeds.

For the paper: report the random arm as mean ± sigma over ≥3 seeds. The map arm
can be reported from a single run, with its sigma stated.

### 10.6 The random-init control arm is DEGENERATE at these Gaussian counts

This invalidates more than the densification question. Probing the parameter
distributions during a clone-only run on room (`logs/probe_clone.log`):

```
iter 500 | opacity mean=0.0999 p99=0.1000 frac>0.9=0.00% | scale mean=0.03855 p99=0.03860
iter 600 | opacity mean=0.1000 p99=0.1000 frac>0.9=0.01% | scale mean=0.03854 p99=0.03860
iter 700 | opacity mean=0.1000 p99=0.1000 frac>0.9=0.01% | scale mean=0.03853 p99=0.03860
```

**The random arm's opacity and scale are not being optimized at all.** After 700
iterations at LR 5e-2, opacity is still pinned at its initial 0.1 and scale at
its initial 0.03855. And `p99 ≈ mean` for both: the distributions are
near-degenerate, every Gaussian carrying the same value it was born with.

Mechanism: `build_random_init` fills the map's bounding box with **7.35M**
Gaussians at opacity 0.1. Any ray through a room-sized box crosses hundreds of
them, so alpha saturates almost immediately. Only a thin front shell is visible,
receives gradient, and trains; the other ~99.99% are permanently frozen behind
it. psnr does creep up (9.72 → 10.44) because the front shell's `f_dc` colours
still optimize — nothing else does.

This explains every anomaly that led here:

| observation | explanation |
|---|---|
| opacity/scale frozen at init | no gradient reaches occluded Gaussians |
| two independent runs ending 8.6284 / 8.6294 — 0.001 dB apart, when seed sigma is 0.66 dB | the result is set by the initialization's geometry, not by optimization |
| clone-only reproducing the full-densify loss exactly | clones add more occluders to the front shell |
| train loss *rising* (0.29 vs 0.125 for prune-only) | the model fits even its training views worse |

**The map arm, probed identically, is not degenerate** (`logs/probe_map.log`,
iteration 500):

| | opacity mean / p99 / frac>0.9 | scale mean / p99 | clones that round |
|---|---|---|---|
| map init | 0.4805 / 0.9932 / **8.28%** | 0.00867 / 0.03330 | **189,011** |
| random init | 0.1000 / 0.1000 / 0.00% | 0.03851 / 0.03860 | 1,024 |

Broad distributions on both quantities versus single-valued ones, and 185x the
clone activity — densification has real gradient signal to act on in one arm
and none in the other. The contrast is confirmed from both sides.

**What this means for §10, §10.2, §10.3 and §10.4.** Every map-vs-random number
in those sections is partly measuring **a broken control**, not an
initialization advantage. The effect is worst on room/360 (7.3M Gaussians) and
milder but same-signed on desk (1.86M).

The design error is specific and identifiable: matching the **Gaussian count**
between arms looked like the fair comparison. Standard 3DGS initializes from
~100k SfM or random points — **two orders of magnitude fewer** — precisely so
that the initial field is not saturated, and then *grows* it by densification.
Matching count instead of matching regime produced an unoptimizable fog.

**A defensible random control needs one of:**

- ~100k random points with densification enabled (the actual 3DGS recipe), or
- the same count but far lower initial opacity, so alpha does not saturate, or
- both arms subsampled to a count where the field is not saturated.

Until one of those is run, **the only claim in §10 that does not depend on the
random arm is the absolute one**: per-scene refinement takes the SLAM map from
12.37 to 19.13 dB on desk (§10.2), which needs no control at all. Everything
comparative is on hold.

### 10.7 The sweep re-run with the repaired control — and a NEW confound

Everything in §10.2-§10.5 used the degenerate count-matched control. Re-run in
one code state with the repaired control (100k points + densification), 3 seeds
on the stochastic arm (`logs/sweep_v2.log`):

| | map | rand100k (3 seeds) | Δpsnr | Δlpips% |
|---|---|---|---|---|
| desk n=14 | 16.65 | 12.97 ± 0.25 | **+3.68** | +34.5 |
| desk n=50 | 19.01 | 16.85 ± 0.51 | **+2.15** | +26.7 |
| desk n=150 | 19.08 | 17.57 ± 0.21 | **+1.51** | +17.2 |
| desk n=400 | 19.11 | 17.33 ± 0.41 | **+1.78** | +20.7 |
| room n=14 | 12.65 | 10.73 ± 0.44 | +1.92 | +34.3 |
| room n=50 | 18.12 | 14.51 ± 0.17 | +3.61 | +51.9 |
| room n=150 | 17.85 | 12.64 ± 0.55 | +5.21 | +50.9 ← **not usable** |
| room n=400 | 17.89 | 12.54 ± 0.28 | +5.35 | +50.9 ← **not usable** |

**What replaces §10.2's claim.** On desk the gap decays smoothly to a *non-zero*
plateau: 3.68 → 2.15 → 1.51 → 1.78, i.e. ~+1.7 dB and ~+19% lpips from 150
views on. The cliff §10.2 reported (4.00 at n=50 collapsing to 0.48 at n=150)
does not exist — it was the degenerate control happening to score well at that
one point. The map arm reproduced within 0.03 dB across six runs and two code
states, so all of the old table's instability lived in the control.

**The map arm saturates at ~50 views on both scenes** — desk 19.01/19.08/19.11,
room 18.12/17.85/17.89. That is the most robust finding of the sweep.

#### The new confound: fixing saturation introduced a capacity mismatch

The room rows above are **not usable for cross-scene comparison**. Peak memory
gives it away: the map arm runs at 6.0-6.6 GiB (7.35M Gaussians) while the
repaired control sits at 1.4-2.0 GiB, i.e. an order of magnitude fewer
primitives after densification. On room the control's train loss *rises* with
more views (0.0876 at n=50 → 0.1589 at n=150) and its held-out score *falls*
(14.51 → 12.64 → 12.54): it is not initialization-limited, it is
**capacity-limited**. desk's map is only 1.86M, so the mismatch there is mild
and its curve behaves.

So the "room's gap widens while desk's narrows" reading — which §10.4 offered
as evidence of scene dependence, and which survived the switch to the repaired
control — is **still an artifact**, now of capacity rather than saturation.
That is the third time in this section an artifact was read as a phenomenon.

**Neither control is clean, and this is structural:**

| control | failure |
|---|---|
| count-matched (7.35M) | alpha saturates, 99.99% of Gaussians never receive gradient (§10.6) |
| regime-matched (100k + densify) | an order of magnitude less capacity than the map on large scenes |

A feed-forward map is dense by construction and a random initialization is
sparse by construction; no single knob makes them equivalent. **The fix is not
another control — it is to report the final Gaussian count alongside every
number** and let the reader see what was traded, which the review recommended
early and which was noted and then not implemented in the scripts' output.

### 10.8 Refinement's 6.7 dB is photometric only — geometry does not improve

No optimized map had ever been persisted, so the 19 dB result had never been
looked at as an image and its **geometry had never been measured**. Per-scene
refinement optimizes a photometric loss alone, which leaves it free to trade
geometry for appearance (floaters that render correctly from the training views
while sitting in the wrong place). `refine_gaussian_map.py --save-ply` now
persists the map; scored with `eval_map_geometry.py`, desk, n=150, same 50
held-out frames:

| | L1 (m) | AbsRel | δ<1.25 | completeness | psnr |
|---|---|---|---|---|---|
| SLAM map | **0.1331** | **0.0887** | 91.0% | 96.1% | 12.37 |
| refined | 0.1425 | 0.0929 | **91.4%** | **100.0%** | **19.08** |

**The floater fear does not materialise** — δ<1.25 even ticks up and coverage
reaches 100%, because optimizing opacity and scale fills gaps the baked map
left empty. **But depth accuracy does not improve; it degrades slightly**
(L1 +7%, AbsRel +4.7%) while psnr rises 6.7 dB.

So the refinement result has to be stated as:

> Direct per-scene optimization recovers **6.7 dB of appearance and 4pp of
> coverage, at flat-to-slightly-worse depth accuracy.** The cost of amortized
> inference shows up in rendering quality, not in geometric fidelity.

That matters most for this system specifically, because a SLAM map's job is
geometric faithfulness. If the deliverable is a geometry product for downstream
consumers, refinement buys little; if it is a renderable map, it buys a lot.

**And it separates the two interventions.** Head fine-tuning *did* improve
geometry (§9.2: L1 −27%, δ<1.25 +5.6pp, coverage −1.5pp) and gains ~1.8 dB;
refinement gains 6.7 dB and improves no geometry. They act on different axes,
which predicts they should compose rather than substitute — a prediction the
base-map-vs-finetuned-map refinement comparison is currently testing.

### 10.9 The training-view count was the binding constraint at 14 views

Both arms plateau by ~1500 iterations while their *training* loss keeps
falling (map arm: 0.065 -> 0.017, a 4x drop with no held-out improvement).
That is textbook overfitting, and the cause is the split: freiburg1_desk has
only **14 keyframes**, so 1.86M Gaussians are being fitted to 14 views.

**16.3 dB is the ceiling of this configuration, not of the method.** The route
to 25+ starts with more training views -- every non-keyframe has ground-truth
poses available and is currently unused -- not with more iterations. Note also
that lpips was still improving at 3000 iterations (0.5185 -> 0.3457, -33%)
while psnr had flattened: the same mse/perceptual divergence seen throughout
training (§6.1, §10).

---

## 10.10 Second review round: three faults in the self-corrections themselves

The six-layer correction chain in §10-§10.6 was submitted for external review.
Three of its criticisms are accepted after verification, and they are about the
*corrections*, not the original results.

**(a) A statistical error in §10.5.** The claim "+0.48 falls outside the 3-seed
range, so the true variance exceeds this estimate" is wrong. Under
exchangeability a 4th sample lands outside the range of 3 priors with
probability ~50% — the range grows with n. +0.48 sits ~1σ from the mean of
1.10, which is unremarkable. The genuinely useful reading was missed: +0.48 was
measured **under a buggy code state**, which supports "the §10.2 numbers were
contaminated" rather than "variance was underestimated." Report mean ± std over
n≥3; never infer from a range.

**(b) §10.3's factorial should be voided, not kept as evidence.** Bug #1
(opacity reset at the readout step) and bug #2 (Adam momentum zeroed every
densification round) *both* penalised specifically the densify arms, and both
acted precisely at the 3000-iteration readout. "Densification widens the gap by
1.9-3.0 dB on the random arm" matches both bugs in direction and magnitude.
Even in the post-fix rerun, the **densify-off cells were reused from the
earlier view-count sweep** — a different code state — so the table compares
across code states.

**(c) The meta-error.** Each of the six layers was measured under a *different*
code state, yet they were presented as one cumulative evidence chain. This is
the same mistake as §10.2's, one level up: §10.2 treated one scene as
generalizable; the chain treats numbers from different code states as
comparable. **Every probe must record its code state (git hash).** Concretely,
the reviewer caught an internal contradiction: bug #3 disabled clone/split on
room from step 0, so the room isolation numbers (clone-only 8.63 vs noop 13.57)
*cannot* come from that state — under it, clone-only would equal noop. They came
from after the #3 fix, which is correct but was never stated.

**(d) Accepted with a partial defence.** The reviewer asked whether the §10.6
probe predates the Adam fix, in which case "parameters barely move" has a rival
explanation (momentum wiped every 100 steps) indistinguishable from saturation.
Verified: `logs/probe_clone.log` was run after all four fixes, so the probe
itself is clean. The reviewer's demand for explicit code-state labelling stands
regardless.

**The isolating arm the review asked for.** The repaired control changed three
variables at once (count 7.35M→100k, saturation, densification enabled), so
"saturation is the mechanism" was not yet established. The single decisive arm
is **100k points with densification OFF** (`logs/isolate_saturation.log`).
First result: 12.63 dB at iteration 300, versus 10.19 at the same point for
7.35M points also without densification — 73x fewer primitives and no
densification, yet better. Consistent with saturation.

**Saturation confirmed by the isolating arm** (`logs/isolate_saturation.log`,
room, 100k points, densification OFF, same code as everything else — the only
variable versus the degenerate arm is the count):

| iter | opacity mean / **p99** / frac>0.9 | scale mean | psnr |
|---|---|---|---|
| 500 | 0.1003 / **0.7137** / 0.23% | 0.15296 | — |
| 900 | 0.1058 / **0.8603** / 0.72% | 0.14911 | 14.76 |
| 1200 | 0.1118 / **0.9183** / 1.21% | 0.14733 | **15.28** |

Against 7.35M points, also without densification: p99 stayed at **exactly
0.1000**, frac>0.9 at 0.00%, and psnr peaked at 10.44 before falling to 9.29.

At 100k the distribution evolves normally; at 7.35M it is frozen. Same code,
same absence of densification — **only the count differs**. This isolates
saturation from the two variables the repaired control had confounded with it
(fewer points, densification enabled), which is exactly what the review asked
for.

**The repaired control, 3 seeds** (`logs/fixed_control.log`):

| | psnr | lpips |
|---|---|---|
| desk map (3 seeds) | 19.1208 ± 0.036 | 0.3355 ± 0.0017 |
| desk random-100k (3 seeds) | 17.8079 ± **0.100** | 0.3924 ± 0.0061 |
| room random-100k (3 seeds) | 12.4743 ± 0.356 | 0.6998 ± 0.0048 |

Note the seed sigma collapses from 0.656 (degenerate control) to 0.100 — a
non-degenerate arm is also a *stable* arm, which is itself evidence for the
diagnosis. desk gap: **+1.31 dB / lpips +17%**, against +1.10 dB / +35%
measured versus the degenerate control.

---

## 10.11 What the training objective actually optimizes, and where the blur comes from

The objective is not what the config says. Three separate facts compound:

```
loss = 1.0 * mse + 0.25 * lpips        # main.py:270-271, config weights
     = 3 * MSE_per-channel             # numerator sums 3 channels, denominator counts pixels
     + 0.25 * LPIPS(AlexNet, spatial)  # 'vgg' bound to `pretrained`, net stayed default
```

So the real objective is `3*MSE_per-channel + 0.25*AlexNet-LPIPS`, and a paper
must state exactly that. But the nominal 12:1 weight ratio is *not* the
contribution ratio — LPIPS values (~0.26) dwarf per-channel MSE (~0.03).
Measured from the training logs:

| | mse term | lpips term |
|---|---|---|
| TUM base | 0.0929 (57%) | 0.0698 (43%) |
| TUM at convergence | 0.0620 (50%) | 0.0629 (**50%**) |
| EuRoC base | 0.2153 (74%) | 0.0753 (26%) |
| EuRoC at convergence | 0.0921 (59%) | 0.0628 (41%) |

The perceptual term carries 26-50% of the loss, and its share *grows* during
training because mse falls while lpips barely moves.

### The hypothesis that the perceptual term causes blur — tested and refuted

Since LPIPS is known to be misalignment-sensitive (ST-LPIPS, ECCV 2022) and our
predictions are renderings with residual registration error against the target,
it seemed possible that the term actively rewarded blur — hedging against
misalignment the way MSE hedges against uncertainty. That would have been
ironic, since blur is the complaint that started this project.

`scripts/diag_blur_preference.py`, n=60 on the seeded draw, perturbing the
prediction and re-scoring (negative Δ = the perturbation *improved* that term):

| perturbation | Δmse% | Δalex% | Δvgg% |
|---|---|---|---|
| blur σ=1 | **−0.76** | +10.17 | +3.47 |
| blur σ=2 | **−1.70** | +30.08 | +10.78 |
| noise 0.02 | **−0.70** | +14.09 | +11.87 |
| noise 0.05 | +0.34 | +61.95 | +26.62 |

**Refuted, decisively.** Blurring the prediction *improves* MSE and *badly*
hurts LPIPS. In total-objective terms at σ=2 the net penalty for blur is on the
order of 30:1 against — the perceptual term is the only thing in this loss
fighting blur, and **MSE is the term that rewards it**, exactly as the classic
conditional-mean argument predicts and as the super-resolution literature
(Johnson 2016, SRGAN, EnhanceNet) implies.

Two secondary findings from the same table:

- **Shift sensitivity is negligible here.** At a 3 px shift: mse +2.0%, alex
  +2.7%, vgg +2.5%. LPIPS is barely more shift-sensitive than MSE at this
  scale, so the misalignment mechanism does not operate — not even in the
  weaker form of "registration residual erodes the sharpening benefit."
  (Caveat: this probes a global rigid shift; real residual is locally
  non-rigid.)
- **Adding noise at σ=0.02 *improves* MSE by 0.70%.** The prediction is smooth
  enough that injecting random noise makes it a better variance match. That is
  a more direct indictment of the MSE term than the blur test.

### Consequences

- **AlexNet vs VGG.** Alex is ~2-3x steeper than VGG on *every* deviation —
  blur (+30.1 vs +10.8) and noise (+62.0 vs +26.6) alike. It is not
  "pro-high-frequency"; it penalises wrong high frequency even harder than it
  penalises missing high frequency. The trunk bug therefore handed us a
  *stronger* anti-deviation term than the code intended.
- **Do not retrain with VGG.** Not because VGG is worse — that claim was
  overreach, since a real switch would retune the weight and the net effect is
  unknown — but because there is no evidence of harm, the checkpoints have
  converged co-adapted to alex's loss scale and gradient structure, retraining
  costs the comparability of every result, and the "VGG was the better intent"
  premise is itself now gone.
- **Blur is not a loss-design problem.** A term carrying 43-50% of the
  objective is strongly penalising blur and blur persists. The remainder is
  irreducible: two-view NVS has genuine uncertainty (occlusion, unseen
  content), and any point-estimate objective converges to a mean under it.
  Reweighting only slides along the perception-distortion frontier (Blau &
  Michaeli, CVPR 2018) — which is exactly what Route C measured (§6: lpips
  improves slightly, psnr pays).
- **Which means the route to less blur is more information, not a better
  loss** — and §10's per-scene refinement, taking the same SLAM map from 12.37
  to 19.13 dB using more views, is that argument's own evidence.

---

## 11. Cross-family generalization: the gain is in-domain only

The three head-only runs each trained on one family and were scored on that
family's held-out split. `data/common.py: split_train_val` takes the last 15%
of frames **within each sequence** — adjacent-frame leakage is avoided (the
split is contiguous and unshuffled, deliberately), but the scenes themselves
were seen in training. Every number in §5-§9 is therefore an **in-domain
adaptation** measurement.

`scripts/eval_cross_family.py` separates adaptation from generalization at zero
training cost: score every trained head on every family's held-out set.
psnr (Δ dB) / lpips (Δ%) / val-loss (Δ%), all vs. that column's base:

| checkpoint | TUM | 7-Scenes | EuRoC |
|---|---|---|---|
| base | 15.09 / 0.279 | 13.71 / 0.282 | 11.44 / 0.301 |
| tum-head | **16.83 (+1.74) / −10.2% / −23.2%** | 14.80 (+1.09) / −1.3% / −14.7% | 11.29 (−0.15) / **+48.5%** / +15.1% |
| 7-scenes-head | 15.47 (+0.38) / **+15.5%** / +1.8% | **15.86 (+2.15) / −8.8% / −28.2%** | 11.74 (+0.30) / **+42.6%** / +6.1% |
| euroc-head | 13.00 (**−2.09**) / **+47.8%** / +55.8% | 13.79 (+0.08) / +26.9% / +8.4% | **15.13 (+3.69) / −16.5% / −46.7%** |

**The diagonal improves substantially; the off-diagonal degrades badly on
lpips, in every case but one.** The worst, euroc-head on TUM, is 2.09 dB
*worse* than base with lpips up 47.8%.

The single exception is **tum-head → 7-Scenes** (+1.09 dB, lpips −1.3%,
val-loss −14.7%), the only off-diagonal cell that is not worse than base on any
metric. Plausible: TUM and 7-Scenes are both handheld indoor RGB-D, while EuRoC
is drone-mounted, fisheye, grayscale, and the only family relying on
self-predicted pseudo-depth.

**Conclusion: this is domain adaptation, not a general improvement, and
cross-domain it actively hurts.** Any claim in a write-up must be scoped to the
family the head was trained on. Reporting the diagonal alone would be
misleading.

---

## 12. Independent methodology review (Kimi) — accepted, disputed, and found

The plan and results were put through an external methodological review. What
survived verification:

**Accepted, verified in code:**

- *The val split is not scene-level* (`data/common.py:147`). Confirmed and
  quantified in §11.
- *Using ground-truth poses for non-keyframe supervision is an oracle, not a
  deployable pipeline*, because `save_traj` persists keyframes only. Correct.
  The refinement script now labels this explicitly; a deployable protocol needs
  SLAM to persist per-frame estimated poses.
- *A view-count sweep beats simply adding views.* The sharpest point in the
  review: more training views would likely *shrink* the init gap and thereby
  falsify our own claim, whereas plotting the gap against view count reframes
  it as a **data-efficiency advantage in the few-view regime** — defensible and
  more informative. Adopted; running (§13).
- *The 2×2 factorial is missing a cell.* {random, SLAM} × {no-densify,
  densify}: without SLAM-init + densification there is no answer to the only
  deployable question. Also: multi-scene (desk alone is anecdotal), ≥3 seeds
  for the stochastic densify arms, report final Gaussian counts and wall-clock.
- *No geometric metrics at all.* Map quality is photometric only; 16% fewer
  Gaussians with higher psnr could still mean worse geometry. Needs depth L1 /
  accuracy / completeness.
- *Baselines are thin* — everything is measured against our own base
  checkpoint, with no online-optimizing GS-SLAM (MonoGS / SplaTAM / Photo-SLAM)
  for reference.
- *Single seed*, and TUM's map-quality mean is n=3 with no per-sequence
  variance reported.

**Disputed, and the dispute holds:** the review argued the mse channel-
normalization issue (§2 / `diag_psnr.py`) means the trained checkpoints are
"artifacts of a mis-weighted loss" requiring retraining before any headline
number can be trusted. That overstates it. Training used upstream's own loss
verbatim; base and every trained checkpoint were scored through the identical
path; the deltas are valid. What is true is that the nominal `mse:lpips =
1.0:0.25` is, in per-channel terms, **12:1** — a labelling problem that must be
stated, and one that incidentally *explains* the mse/perceptual divergence seen
throughout (§6.3, §10.2). "Retraining with corrected weights" is not a fix, it
is a different experiment — and approximately the one Route C already ran
(§6, 12:1 → 3:1).

**LPIPS trunk: measured, not just noted.** Re-scoring the cross-family matrix
with a real VGG trunk (`USE_VGG_LPIPS=1 scripts/eval_cross_family.py`) keeps
every sign but compresses every magnitude, TUM column:

| head | AlexNet | VGG |
|---|---|---|
| tum-head (in-domain) | −10.2% | **−3.6%** |
| 7-scenes-head | +15.5% | +12.5% |
| euroc-head | +47.8% | **+28.0%** |
| eth3d-head | +31.8% | **+22.6%** |

The direction survives — in-domain better, cross-domain worse — but AlexNet
inflates both by roughly 2-3x. **The in-domain headline drops from −10.2% to
−3.6%**, so every lpips figure quoted in §5 and §9 has to be restated on the
VGG trunk before it goes anywhere near a paper.

**Do not write "the worse the base, the larger the gain" as a finding.** With
the encoder frozen and only the head trained, a family whose base is weak
leaves the head more headroom by construction — the statement is close to
tautological — and it is additionally exposed to regression to the mean, since
each family's base is itself one noisy measurement.

**Found independently, missed by the review:** `splatt3r_core/main.py:89` reads
`lpips.LPIPS('vgg', spatial=True)`, but the package signature is
`LPIPS(pretrained=True, net='alex', ...)`. `'vgg'` binds to **pretrained**, and
`net` stays at its default. **Every LPIPS number in this project is AlexNet-
LPIPS, not VGG** — confirmed by `trunk [alex]` in the training logs. Harmless
internally (all arms identical, and the eval scripts happen to pass
`net="alex"` too), but it must be reported as LPIPS(AlexNet) and never compared
against published VGG-LPIPS figures.

**Positioning.** The review's framing is fair: tracking, front-end and loop
closure are untouched (§9.1 is a sanity check, not a result), so this is a
*mapping / initialization* study, not a SLAM-system contribution, and should be
written as one.

---

## 13. Deployability: what of all this can run in a real-time system

Everything up to here optimizes for a *scientific* question — where is the
bottleneck. The project's actual deliverable is end-to-end real-time
Splatt3R-SLAM reconstruction, and under that framing most of the headline
numbers are not deliverables at all. This section separates them.

### 13.1 The three stages, and which numbers belong to which

| stage | what is optimized | when | desk psnr |
|---|---|---|---|
| 1. head fine-tuning | network weights | offline, once per family | map = **12.44** |
| 2. SLAM feed-forward + fusion | nothing (inference) | online | 12.44 |
| 3. online map optimization | the map's Gaussian params | online, background | **14.00** at current poses |

Stage 3 optimizes the accumulated map's `means / scales / rotations / opacity /
colour`. **The network is frozen and absent from it** — it is plain 3DGS
parameter optimization, unrelated to Splatt3R except that Splatt3R produced the
initialization.

**Stage 1 is finished and saturated.** Per-family best (two-view held-out
protocol, corrected psnr scale): TUM 16.87, 7-Scenes 15.90, EuRoC 15.13,
ETH3D 14.00. TUM's per-10-epoch bests were 16.44 / 16.66 / 16.78 / 16.87 — the
last ten epochs bought 0.08 dB. More training is not where anything is left.

On the SLAM map the absolute numbers are lower but the *increment* is the same:
base map 10.71 → fine-tuned map 12.44 (+1.73), against +1.78 on the two-view
protocol.

**And most of stage 1's gain is redundant with stage 3.** Refining both maps
(`logs/finetune_survives_refine.log`):

| desk | before refinement | refined n=14 | refined n=150 |
|---|---|---|---|
| base map | 10.71 | 15.95 | 18.29 |
| fine-tuned map | 12.44 | 16.68 | 19.08 |
| **fine-tuning's contribution** | **+1.73** | **+0.73** | **+0.79** |

Refinement absorbs ~57% of it. What survives (+0.75 dB) costs 13 GPU-hours per
family; 300 iterations of online refinement — seconds — buys +1.63 dB and needs
no per-family weights, no coverage cache, and has no cross-domain hazard.

### 13.2 The deployable protocol, and the oracle ingredients it removes

`refine_gaussian_map.py --deployable` supervises only on the keyframes the SLAM
run actually selected, at the poses it actually estimated — no ground truth
anywhere. desk (`logs/deployable.log`):

```
initial      12.372
  300 iters  13.998   (+1.63 dB, lpips -15%)
 1000 iters  13.977
 3000 iters  14.004   ← 10x the iterations, +0.006 dB
```

**Saturated by 300 iterations.** Iteration count is not the constraint.

Decomposing the 5.08 dB between this and the oracle 19.08, all measured on desk:

| ingredient | measurement | worth |
|---|---|---|
| pose quality | 14 keyframes, est. poses 13.998 → GT poses 15.810 | **+1.81 dB** |
| view count | 14 views 16.65 → 150 views 19.08 (GT poses) | **+2.43 dB** |
| view selection | 15.810 (14 keyframes) → 16.65 (14 uniform frames) | +0.84 dB |

The first reading of this attributed the whole gap to view count. It is roughly
half pose quality — and the fingerprint was already visible in the data:
1000 iterations scoring *below* 300 is what fitting photometric noise from pose
error looks like, not what running out of views looks like.

**Both ingredients are recoverable in principle.** `tracker.track()` runs on
every frame and produces `T_WC`, so per-frame estimated poses exist at runtime
— `save_traj` simply does not persist them. The catch is that non-keyframe
poses are never corrected by the pose-graph backend, so they must be stored
relative to an anchor keyframe (`T_kf_frame`) and recomputed after loop closure.

### 13.3 The rasterizer has no camera-pose gradient — and why that does not block anything

Making the supervision poses `nn.Parameter`s produced `mean pose translation
correction: 0.0000` after 1000 iterations. Cause:
`diff_gaussian_rasterization`'s backward returns

```
means2D, colors_precomp, opacities, means3D, cov3Ds_precomp, sh, scales, rotations
```

and nothing else. **The camera matrices are constants inside the CUDA
extension.** MonoGS and similar online systems use a modified rasterizer for
exactly this reason.

**But the CUDA change is unnecessary**, because moving the camera by δ and
moving every Gaussian by δ⁻¹ are the same render, exactly:

```
render(X, T·δ)  ≡  render(δ⁻¹·X, T)
```

The second form differentiates, since it touches only means and covariances,
both of which do receive gradients. Implemented as a rigid transform of the map
per supervision view (`render(..., pose_delta=...)`):

```
poses fixed,  300 iters   14.0003   lpips 0.4396
poses fixed, 1000 iters   13.9726   lpips 0.4143   (saturated)
identity trick, 500 iters 14.7265   lpips 0.4548   mean |Δt| = 0.0165
identity trick,1000 iters 14.7066   lpips 0.4404
GT poses (ceiling)        15.810    lpips 0.4181
```

**+0.73 dB, i.e. 40% of the 1.81 dB pose gap, with no CUDA change.** It then
saturates too — with only 14 views, pose refinement runs out of signal as well,
consistent with view count being a genuinely separate bottleneck.

**Cost of the identity vs. a CUDA camera gradient** — quality is identical by
construction, so only throughput differs (`scratchpad/bench_pose.py`, measured
against ~59 ms/iteration end-to-end):

| map | N | transform fwd+bwd | share of an iteration | throughput |
|---|---|---|---|---|
| desk | 1.86M | 6.74 ms | 11% | 17 → 15.1 it/s |
| room | 7.35M | 22.10 ms | 37% | 17 → 12.4 it/s |

A camera-pose gradient would be O(1) instead of O(N), so the CUDA route buys
11-37% throughput and nothing else. Against that: `backward.cu` would have to
be modified, recompiled and re-validated.

**On the extension's crash history** — worth stating because it looks like a
reason to avoid touching the CUDA and is not. The `illegal memory access` at
`rasterizer_impl.cu:398` was traced (see `decoder_splatting_cuda.py`) to input
degeneracy, not to a kernel defect: this renderer assumes sparse SfM scenes of
a few thousand Gaussians, while pixel-aligned prediction emits 147,456 for a
single 384×384 view, overflowing fixed-capacity per-tile indexing. It was fixed
by `spatial_stride` subsampling and by not training the encoder (whose LoRA
route inflated scales to 85x, multiplying each Gaussian's tile coverage). That
history therefore does not transfer to a `backward.cu` edit; the real risk of
that edit is a *new* kernel bug.

### 13.4 The iteration budget is bought with frame rate

Measured 17 iterations/s on desk (1.86M Gaussians). A 613-frame sequence:

| capture fps | sequence wall-clock | background iterations |
|---|---|---|
| 10 | 61 s | ~1,000 |
| 2 | 307 s | ~5,200 |
| **1** | 613 s | **~10,400** |
| 0.5 | 1226 s | ~20,800 |

So the offline iteration counts *are* reachable online by lowering capture
rate — 1 fps yields the ~10k iterations at which the oracle protocol reached
19.34. Two caveats: (a) at the current pose quality the online arm saturates at
300 iterations, so bought iterations are wasted until poses and views improve;
(b) a lower capture rate widens the inter-frame baseline and may *degrade*
tracking, eroding the pose quality it is trying to exploit. (b) is unmeasured.

### 13.5 What the offline ceiling actually is

`--iters 30000 --densify` on desk, oracle protocol
(`logs/long_refine.log`), stopped at 24000:

| iters | 3000 | 9000 | 15000 | 18000 | 21000 | 24000 |
|---|---|---|---|---|---|---|
| psnr | 18.84 | 19.34 | 18.79 | 19.86 | 19.89 | **19.95** |
| lpips | 0.3375 | 0.2928 | 0.2941 | 0.2700 | 0.2736 | **0.2687** |

+1.11 dB and −20% lpips over 3000 iterations. Note the shape: it dips through
the densification window and climbs after `DENSIFY_UNTIL=15000` — which is why
the earlier verdict that densification is harmful (§10.3, voided) was an
artifact of running a schedule designed for 30k for only 3k. A plateau was
called at 9000 on the strength of the 12000/15000 dip and was wrong again;
that is the fifth time in this project a dip has been read as a trend.

SH degree 3 (`--sh-degree 3`, progressive upgrade every 1000 iterations) reached
19.159 / 0.3288 at 3000 against degree 0's 19.078 / 0.3341 — **+0.08 dB for 15
extra coefficients per channel**. Since view-dependent colour is the textbook
remedy for the revisit-colour-inconsistency complaint that helped start this
project, a gain this small is itself a diagnosis: that complaint's cause is
probably not view dependence.

### 13.6 Summary of what ships

**Superseded in detail by §13.10-13.12; kept for the shape of the argument.**
The estimates in this table (`~16.6` from unlocked views in particular) were
falsified by the measured grid — views saturate at ~25, not 150.

```
stage 1  head fine-tuning     DONE, saturated     map 12.44 dB
stage 2  SLAM feed-forward    existing system     12.44
stage 3  online refinement    TO BUILD            14.00  (14 keyframes, est. poses)
                                                  14.73  (+ pose refinement via the identity)
                                                  ~16.6  (+ unlocked views, estimated) <- WRONG, see 13.10
                                                  19.95  requires GT poses — not deliverable
```

Two main-path changes remain, both gating stage 3: persisting per-frame poses
relative to an anchor keyframe (**now DONE, §13.10**), and moving the
optimization kernel into the SLAM loop as a background thread (**still open**).
Every component of the second already exists in `refine_gaussian_map.py`
(pre-activation parameterization, Adam state surgery across densification, the
differentiable render loop, the pose identity); the work is relocation plus
handling the non-stationarity of new Gaussians being injected each keyframe.

### 13.7 The identity vs. a CUDA camera gradient — where the argument actually lands

§13.3 concluded "the CUDA change is unnecessary." That conclusion was correct
for the *offline validation* and wrong as a statement about the *online
system*. The distinction was sharpened by an independent review (Kimi, round 8)
and by measuring the identity's cost at the operating point that matters.

**What the identity does and does not buy.** It is exact, so the gradient — and
therefore the achievable dB — is identical to a CUDA camera gradient. Nothing
about image quality depends on this choice. What differs is cost, and the cost
is O(N) per rendered view per optimizer step, against the rasterizer's own O(N)
read that happens anyway:

| views per step | steps per keyframe | desk (1.86M) | room (7.35M) |
|---|---|---|---|
| 1 | 50 | 0.34 s | 1.1 s |
| 10 (covisibility window) | 50 | 3.4 s | 11 s |

The single-view column is survivable at 1 fps. The multi-view column is not,
and multi-view steps are exactly what turns per-view pose refinement into
something that behaves like bundle adjustment. **This is the whole case for
CUDA**, and it only appears once the target is the online loop at room scale.

**Two defects in the identity that the review found, both real:**

1. **SH direction.** `cuda_splatting.py:151` passes `campos=extrinsics[i,:3,3]`
   — the *original* camera position — while the Gaussians handed to the
   rasterizer have been rigidly transformed. `forward.cu:26` evaluates the SH
   direction as `pos - campos`, so under the identity that direction is rotated
   by `R_δᵀ`. Harmless at `sh_degree=0` (direction-independent), wrong at
   degree ≥ 1. Worse, the obvious fix (precompute colors in Python, pass
   `colors_precomp`) is not wired for full SH: `cuda_splatting.py:181` passes
   `colors_precomp=None if use_sh else shs[i,:,0,:]`, i.e. only the DC term.
   Since §13.5 measured SH degree 3 at +0.08 dB this costs nothing today, but
   it is a permanent trap for anything view-dependent later (appearance
   embeddings, exposure per view).
2. **A convention error in §13.3's own formula.** As written,
   `render(X, T·δ) ≡ render(δ⁻¹·X, T)` mixes conventions: for `T` = w2c
   right-multiplied the correct identity carries `δ`, not `δ⁻¹`. The
   implementation in `refine_gaussian_map.py:391-397` is self-consistent under
   the other reading — `T` = c2w with `δ` left-multiplied in the world frame,
   `render(X, δ·c2w) ≡ render(δ⁻¹·X, c2w)` — which matches the code, since
   `extrinsics` is c2w and `cuda_splatting.py:127` inverts it internally. The
   measured `mean |Δt| = 0.0165` and +0.73 dB confirm the sign empirically
   (a flipped sign would have made the loss worse, not better), but the
   docstring is misleading and the convention was never pinned by a test.

**Why this does not change the +0.73 dB result.** Degree 0 was used throughout,
so defect 1 was inactive; defect 2 is a documentation error, not a code error.
The numbers in §13.3 stand.

**The decision.** Modify the rasterizer. Not because the identity is wrong —
it is exact, it is implemented, and it produced the only pose-refinement
evidence we have — but because the production form of stage 3 needs multi-view
steps at room scale, and needs camera-dependent quantities to stay correct
without a per-consumer workaround. The identity remains the reference
implementation against which the CUDA path is validated.

### 13.8 The CUDA design, and why every intermediate it needs already exists

The change is additive rather than derivative: all three gradients fall out of
values `backward.cu` already computes and currently discards.

**Chosen interface: raw matrix gradients, not a Lie-algebra `dL_dtau`.** MonoGS
returns a 6-vector, which bakes the pose parameterization into CUDA. Returning
`dL_dviewmatrix` (16), `dL_dprojmatrix` (16), `dL_dcampos` (3) instead keeps
the parameterization in Python, where `extrinsics.inverse()`, the
`scale_invariant` rescale (`cuda_splatting.py:105-112`) and
`get_projection_matrix` are already differentiable torch ops. Consequences:

- `cuda_splatting.py` needs **no change at all** — `view_matrix`,
  `full_projection` and `campos` are already tensors derived from `extrinsics`,
  so gradient reaches `extrinsics` by itself.
- The convention hole of §13.7 disappears: autograd resolves left/right
  multiplication, so there is no sign to get wrong. Under a `dL_dtau`
  interface it would instead be *frozen into CUDA* — and a mismatched Lie
  convention fails silently, since it remains a valid parameterization that
  still converges, only in the wrong direction.
- Intrinsics refinement becomes structurally reachable (`projmatrix` ← fov) —
  but **not free**: `cuda_splatting.py:141-142` calls `.item()` on
  `tan_fov_x/y`, so the focal scalars feeding `computeCov2DCUDA`'s Jacobian are
  detached. An intrinsics gradient would cover the projection path only until
  those two are also passed as tensors.

The honest case *for* `dL_dtau`: 6 well-scaled outputs instead of 35 whose
magnitudes differ by orders (rotation vs. translation vs. projection entries)
under fp32 atomics, and a battle-tested reference in MonoGS to port from. The
first is answerable by accumulating translation and rotation blocks separately;
the second is moot here, because the identity is a *better* reference than
MonoGS — it yields the numerically exact gradient on our own scenes and data,
which is what validation step 1 below exploits.

**Where each gradient comes from:**

| output | kernel | already-computed locals it reuses |
|---|---|---|
| `dL_dprojmatrix` | `preprocessCUDA` | `m_w`, `mul1`, `mul2`, `dL_dmean2D` |
| `dL_dviewmatrix` | `computeCov2DCUDA` | `dL_dtx/ty/tz`, `dL_dT00..dL_dT12`, `J`, `W` |
| `dL_dcampos` | `computeColorFromSH` | the `dnormvdv` result (negated) |

Specifically, `preprocessCUDA` already forms `dL_dmean` from `m_hom`; the
projection-matrix gradient is the same chain stopped one step earlier —
`dL_dm_hom = (dL_dmean2D.x·m_w, dL_dmean2D.y·m_w, 0,
-(mul1·dL_dmean2D.x + mul2·dL_dmean2D.y))`, then
`dL_dproj[4j+k] += dL_dm_hom[k]·m[j]` (12 nonzero entries). `computeCov2DCUDA`
already forms `dL_dtx/ty/tz` and the `dL_dT` block to produce `dL_dmean`; the
view-matrix gradient reuses both, `dL_dW[k][r] += Σ_c dL_dT[c][r]·J[c][k]` for
the covariance path and `dL_dW[i][j] += dL_dt[i]·m[j]`, `dL_dp = dL_dt` for the
transform path. The 2D-mean path reaches `view_matrix` through
`projmatrix = view_matrix @ projection_matrix` on the Python side, so the two
outputs together are complete and non-overlapping.

The only genuinely new machinery is the reduction: 35 scalars accumulated over
millions of Gaussians. Warp shuffle → shared memory → one `atomicAdd` per block
per entry, the standard pattern; naive per-thread `atomicAdd` would serialize.

**Files touched:** `cuda_rasterizer/backward.{h,cu}`,
`cuda_rasterizer/rasterizer.h`, `cuda_rasterizer/rasterizer_impl.cu`,
`rasterize_points.cu`, `diff_gaussian_rasterization/__init__.py`. The Python
wrapper moves `viewmatrix`/`projmatrix`/`campos` from the
`GaussianRasterizationSettings` NamedTuple into positional tensor arguments of
`_RasterizeGaussians.apply`, keeping the NamedTuple fields so existing callers
— importantly `decoder_splatting_cuda.py`, the training path — are unaffected
and simply pass tensors with `requires_grad=False`.

**Validation, in order.** The identity is what makes this cheap to trust:

1. **Differential test against the identity.** Same scene, same δ: `δ.grad`
   through the CUDA path must equal `δ.grad` through
   `refine_gaussian_map.py`'s transform path. This is a stronger check than
   finite differences and it simultaneously pins the convention that §13.7
   found unpinned.
2. Finite differences on a few raw `viewmatrix` entries (fp32, coarse eps).
3. **Regression gate:** re-run the 500-iteration pose experiment; it must
   reproduce 14.7265 within seed noise. Same math ⇒ same number, only faster.
   A different number means a bug, not an improvement.
4. `compute-sanitizer --tool memcheck` on one iteration.

On the `illegal memory access` history (`decoder_splatting_cuda.py:28-50`): the
root cause there was degenerate *input* to the existing kernels, not a defect
in them, so it does not transfer to this change. The new code adds no per-
Gaussian indexing and no dynamic allocation — its failure mode is confined to
the fixed-size accumulator, which memcheck catches immediately. Note also that
`cuda_splatting.py` still sets `debug=True` (a leftover diagnostic from that
investigation), which forces a device sync after every internal kernel launch;
useful during bring-up, but it must be turned off before any throughput
measurement.

### 13.9 Correction to the stage-3 ladder

The roadmap as stated — "+ pose joint optimization (needs CUDA) → 16.6~18" —
gets the causality wrong in two ways, and §13.6's table above inherits part of
it:

1. **Pose optimization does not need CUDA.** It already ran and already paid:
   14.0003 → 14.7265. CUDA changes throughput, SH correctness and multi-view
   feasibility — not the gradient, and not the dB.
2. **Pose optimization alone cannot reach 16.6 with 14 views.** GT poses *are*
   the optimum of the pose variables, and GT poses with 14 views measure
   15.810. That is a ceiling, not a waypoint. Everything above it must come
   from view count: 19.078 is the 150-view + GT-pose measurement.

The two rows the roadmap separates are therefore coupled, and separating them
invites a misleading negative. Non-keyframe poses are the *least* reliable in
the system — tracked against a keyframe, never corrected by the pose-graph
backend — so "unlock 50 views, poses fixed" adds the noisiest supervision with
no mechanism to absorb it, and may well score below 14.00. The 2.43 dB view
gain of §13.2 was measured with GT poses throughout. Under joint optimization
each added view instead brings its own 6 DoF and helps constrain the map, which
is the regime where extra views are supposed to pay.

Corrected ladder, with what is measured vs. estimated made explicit:

```
12.372  feed-forward + fusion, no refinement          measured
14.000  + 300 iters, 14 keyframes, SLAM poses fixed   measured, saturated
14.727  + pose refinement via the identity, 500 it    measured
15.810  ceiling for 14 views (GT poses)               measured
  ?     + ~50 views, poses fixed                      likely BELOW 14.00
  ?     + ~50 views, poses jointly optimized          the actual target
19.078  150 views + GT poses                          measured, not deliverable
```

The gating item for the two unknown rows is persisting per-frame poses
(§13.2) — which is independent of the CUDA work and is the larger dB lever of
the two.

### 13.10 Per-frame pose persistence — implemented

`splatt3r_slam/evaluate.py: FramePoseLog`, wired into `main.py` at the two
`keyframes.append()` sites and after `tracker.track()`. Writes
`<seq>_frames.txt` in TUM format alongside the existing keyframe `<seq>.txt`.

Poses are stored **relative to the keyframe the frame was tracked against**,
not in world coordinates. A non-keyframe's pose is estimated once and never
revisited by the pose-graph backend, so a world-frame copy goes stale the
moment loop closure moves the surrounding keyframes — while the Gaussian map
does *not* go stale, since it is re-baked from each keyframe's current `T_WC`
on every draw and every export. Supervising a corrected map with uncorrected
poses is exactly the error the refinement is meant to remove. Anchor-relative
storage keeps the two consistent: the world pose is recomputed at export from
whatever the anchor's pose has become. Runtime cost is one Sim3 compose per
frame with no host sync (the relative pose stays on the GPU until save time).

**Verification on TUM desk** (`--config config/eval_calib.yaml`, subsample 2):

```
map .ply md5 identical to the previous run          -> the rerun is exact
keyframe poses in <seq>_frames.txt vs <seq>.txt     -> 14/14 bit-identical
frames written                                      -> 307 (vs 14 keyframes)
```

**The poses that get unlocked are the worst ones in the system**, and now
measured rather than asserted:

| trajectory | frames | ATE RMSE |
|---|---|---|
| keyframes (`<seq>.txt`) | 14 | 0.0170 m |
| all tracked frames (`<seq>_frames.txt`) | 307 | 0.0291 m |

1.7x worse, which is the expected consequence of never being corrected by the
backend. This is the whole reason the view-count and pose-optimization rows of
§13.9 must be measured together.

**Consumer:** `refine_gaussian_map.py --frames-traj <path>`, valid with
`--deployable`, replaces the ~14-keyframe supervision set with `--n-train`
views drawn from the frame trajectory. These poses need no Sim3 mapping — they
are composed from the same keyframe poses the map was baked from, so they are
already in the map's frame, unlike the ground-truth path.

#### A sampling bug that confounded the whole view-count axis

The first full grid showed a **non-monotone collapse at exactly n=150, in both
pose modes** (fixed 14.17, optimized 14.45, against 14.62/15.16 at n=100 and
14.67/15.39 at n=250). A real effect has no reason to be worst at one interior
view count and recover afterwards, so it was a measurement artifact, and it was
mine.

`seq[::max(1, len(seq)//n)][:n]` looks like uniform subsampling and is not.
The stride comes from integer division and the result is then truncated, so
once n exceeds len(seq)/2 the stride collapses to 1 and the selection is a
contiguous **prefix**. On the 284-frame desk pool:

| n_train | stride | trajectory span |
|---|---|---|
| 25 | 11 | 93% |
| 50 | 5 | 87% |
| 100 | 2 | **70%** |
| 150 | 1 | **53%** |
| 250 | 1 | 88% |

So "number of views" was confounded with "how much of the room was seen", and
n=150 was the worst cell. It also explains the two other oddities in that grid
— n=100 scoring level with n=50 despite twice the views, and n=250 recovering.
Fixed by `uniform_subsample()` (linspace, endpoints included); the n=50/100/
150/250 arms are being re-run.

**The held-out selection was deliberately NOT changed**, though it uses the
same idiom: it defines the test set that every previously recorded number was
scored against (12.372 init, 13.998 baseline, 15.810 GT ceiling). At n_held=50
of 599 the span is 90%, so the bias is small, and it is frozen for
comparability rather than because it is ideal.

Note this idiom is also on the **oracle** path (`train_c = pool[::...]`), so
§13.2's 150-view figure of 19.078 was drawn from ~79% of its pool — the 2.43 dB
view lever is, if anything, slightly understated.

#### The corrected grid, and three results that all contradict the plan

Desk, `--deployable` (no ground truth anywhere), 3000 iterations, held-out 50.
Reported as **mean ± sd over the evaluations at 2000-3000**, not the peak — a
peak over 12 evaluations of an oscillating curve is a biased estimator, and the
oscillation here is comparable to the effects being compared.

| views | fixed poses | lpips | poses optimized | lpips | pose lever |
|---|---|---|---|---|---|
| 14 (keyframes) | 13.989 ± 0.036 | 0.388 | 14.632 ± 0.106 | 0.426 | +0.642 |
| 25 | 14.361 ± 0.065 | 0.374 | 15.014 ± 0.040 | 0.430 | +0.653 |
| 50 | 14.454 ± 0.057 | 0.389 | 15.043 ± 0.174 | 0.470 | +0.590 |
| 100 | 14.441 ± 0.040 | 0.383 | 14.983 ± 0.174 | 0.499 | +0.542 |
| 150 | 14.474 ± 0.037 | 0.388 | 14.966 ± 0.068 | 0.513 | +0.492 |
| 250 | 14.471 ± 0.033 | 0.386 | 15.097 ± 0.104 | 0.527 | +0.626 |

**1. View count saturates at ~25 views, not 150.** Fixed poses: 14 → 25 buys
+0.37 dB and everything after that buys nothing (+0.46, +0.45, +0.49, +0.48,
flat within 0.05 against sd 0.03-0.07). **The deployable view lever is ~+0.45 dB
in total, against the 2.43 dB measured under ground-truth poses.** So almost
none of the oracle view lever survives deployment — a stronger statement than
either side of the round-9/10 exchange about whether 2.43 dB was over- or
under-stated, and in the opposite direction from the prediction that unlocking
views was the largest remaining lever (§13.9).

#### Pose quality is a MULTIPLICATIVE gate on view count — quantified from data already on disk

`logs/view_sweep_desk.log` already holds the ground-truth-pose view curve, run
on the same fixed 1.86M-Gaussian map with **no densification** (the sweep script
passes no `--densify`, and peak memory of 1.6-1.9 GiB confirms it). So capacity
is not a confound: both columns optimize the identical map with the identical
optimizer, and the only variable is where the supervision poses come from.

| views | est. poses (deployable) | GT poses (oracle) | pose gap |
|---|---|---|---|
| 14 | 13.989 (keyframes) | 15.810 (keyframes) | 1.82 |
| 50 | 14.454 | 18.974 | **4.52** |
| 150 | 14.474 | 19.071 | **4.60** |

GT column in full: 16.646 (n=14 uniform) → 18.974 (50) → 19.071 (150) → 19.131
(400). **It saturates too — at 50 views, not 150.** So the picture is two
saturating curves at very different levels, not one climbing and one flat.

**The pose gap grows from 1.8 dB at 14 views to 4.5 dB at 50+ views.** Views are
worth +2.33 dB, but only when the poses are right; with estimated poses they are
worth nothing past ~25. That is what a multiplicative gate looks like: added
views only reinforce each other when their poses are mutually consistent with
the map, otherwise they cancel. It is not dilution — dilution would still leave
a slow gain.

Corollary the earlier framing had backwards: this does **not** weaken "pose is
the bottleneck", it promotes pose from one of two levers to the **only** gate,
with view count locked behind it. What it weakens is the separate assumption
that *photometric* pose refinement is the key that opens it — see the pose LR
sweep below, and the diagnostic in §14.

**2. The pose lever is flat in view count** (+0.49 to +0.65, no trend). No
detectable interaction, consistent with the 0.086 dB floor. The mechanistic
expectation — more views → stronger multi-view constraint → more recoverable
pose error — does not show up.

**3. The pose-optimized arm degrades perceptually as views are added, while
psnr stays flat.** lpips 0.426 → 0.430 → 0.470 → 0.499 → 0.513 → 0.527, a clean
monotone climb, while the fixed-pose column stays at 0.374-0.389. And pose
optimization worsens lpips *even at 14 views* (0.388 → 0.426).

**So the entire pose-refinement gain is psnr-only and lpips-negative.** The
same signature is visible in the §13.3 data and was not read at the time:
`poses fixed 14.0003 / lpips 0.4396` versus `identity trick 14.7265 / lpips
0.4548`. Two runs, both showing it, one of them recorded weeks earlier.

Mechanism: each view carries its own 6 DoF, so 250 views is 1500 pose
parameters, each visited ~12 times in 3000 iterations. Weakly constrained pose
variables absorb photometric residual — which preserves psnr (they are fitting
the training views) and destroys perceptual quality on held-out views. This is
exactly the pose/map entanglement flagged in the round-8 review, whose
recommended remedy was a pose learning rate **1-2 orders of magnitude below**
the map's. Measured here:

```
LR_POSE            1.0e-3
map positional LR  1.6e-4 * extent(3.024) = 4.84e-4
ratio              pose is 2.07x the map LR   -- 20-200x off the recommendation
```

#### The pose learning rate is a perception-distortion dial, and 1e-3 sat at one end

| views | pose LR | psnr | lpips | mean drift |
|---|---|---|---|---|
| 25 | fixed (none) | 14.361 | 0.374 | — |
| 25 | 1e-3 (old default) | 15.014 | 0.430 | 0.0125 |
| 25 | 3e-4 | 14.724 ± 0.050 | 0.381 | 0.0142 |
| 25 | **1e-4** | **14.709 ± 0.022** | **0.366** | 0.0084 |
| 25 | 1e-5 | 14.450 ± 0.011 | 0.375 | 0.0031 |
| 250 | fixed (none) | 14.471 | 0.386 | — |
| 250 | 1e-3 (old default) | 15.097 | 0.527 | 0.0131 |
| 250 | 3e-4 | 15.138 ± 0.138 | 0.439 | 0.0107 |
| 250 | 1e-4 | 14.794 ± 0.047 | 0.386 | 0.0051 |
| 250 | 1e-5 | 14.508 ± 0.022 | 0.378 | 0.0014 |

Monotone in both metrics and in opposite directions: higher pose LR buys psnr
and costs lpips. The old default sat at the extreme, so most of its +0.65 dB
was pose parameters fitting the training views.

**The one cell that improves both metrics: 25 views, pose LR 1e-4.** It beats
its own fixed-pose control on psnr *and* lpips (14.709/0.366 vs 14.361/0.374),
which no other pose-optimized configuration does. Against the 14-keyframe
fixed-pose baseline (13.989/0.388) it is **+0.72 dB with lpips 5.7% better** —
a genuine improvement rather than a trade.

There is no configuration that gets both the +1.0 dB and good lpips. 250 views
at 3e-4 gives the highest psnr in the whole study (15.138) with lpips 14% worse
than doing no pose optimization at all.

Drift corroborates the mechanism: the settings that win psnr move the poses
roughly twice as far (0.011-0.014 vs 0.005-0.008), and the extra motion is what
costs perceptual quality.

**Corrected deployable ladder:**

```
12.372   feed-forward + fusion
13.989   + refinement, 14 keyframes, poses fixed          lpips 0.388
14.709   + 25 views, pose optimization at LR 1e-4         lpips 0.366   <- both better
15.138   + 250 views, pose optimization at LR 3e-4        lpips 0.439   <- psnr only
```

#### This weakens the case for the CUDA change, and that should be said plainly

§13.7 justified modifying the rasterizer with a quantified argument: a
covisibility window of ~10 views at room scale, 50 steps per keyframe, costs
11 s per keyframe under the identity — fatal for real time. Two results here
undercut its premise. View count saturates at ~25, and one view is sampled per
optimizer step, so the multi-view sweep that generated the 11 s figure is not
the regime the system actually needs.

At one view per step the identity costs 0.34 s per 50 steps on desk and 1.1 s
on room — survivable at 1 fps.

What survives as reasons to keep the CUDA path: a deterministic 0.2 GiB memory
saving on desk scaling to ~0.8 GiB on room; correctness at sh_degree >= 1,
which the identity cannot provide; and the fact that it is already written,
validated to 7.6e-06, and shown non-inferior over seeds — so keeping it carries
no residual risk. What no longer survives is the necessity argument. If the
work had not already been done, this grid would not justify starting it.

**The contamination guard that this arm lives or dies by.** Held-out views are
non-keyframes, and so is almost everything in the frames trajectory, so the
training pool must have the held-out set removed *first* or the measurement is
circular. Verified concretely: `dataset.timestamps` carries all 613 frames
(subsample is applied by the SLAM loop, not the dataset object), the 50
held-out frames are drawn across all 613, and 23 of them fall on tracked
frames — so the filter takes 307 candidates to 284, and the remaining 27
held-out frames were never candidates at all.

### 13.11 The CUDA camera gradients — implemented, and what validating them turned up

Built as designed in §13.8 (raw matrix gradients, parameterization left in
Python). `cuda_splatting.py` needed no change, as predicted: the camera tensors
were already derived from `extrinsics` by differentiable ops, so moving
`viewmatrix`/`projmatrix`/`campos` from the `GaussianRasterizationSettings`
NamedTuple into positional arguments of `_RasterizeGaussians.apply` was enough
for gradient to reach `extrinsics` on its own. The NamedTuple fields are kept,
so `decoder_splatting_cuda.py` (the training path) is untouched.

Consumer: `refine_gaussian_map.py --pose-backend {identity,cuda}`.
Validation: `scripts/test_camera_gradient.py`.

**Results of the differential test (the point of keeping the identity around):**

```
sh_degree=0, N=50k, full render_cuda path, 6-DoF delta
  forward image   max|diff| / max|val|   2.44e-06
  pose gradient   ||ga-gb|| / ||gb||     7.6e-06     cos = 1.00000000
  map  gradient   means                  4.0e-06     cos = 1.00000000
  map  gradient   covs                               cos = 0.99970
```

The pose-gradient agreement also **pins the delta convention** that §13.7 found
unpinned: the implementation is `render(X, delta @ c2w) == render(delta^-1 @ X,
c2w)`, i.e. c2w with delta left-multiplied in the world frame. A flipped sign
could no longer pass unnoticed.

#### A pre-existing bug in the vendored rasterizer, found by this test

`thirdparty/diff-gaussian-rasterization-modified` is a fork whose `forward.cu`
was rewritten to "match e3nn's format" (its own comment) — the SH axis
assignments were permuted relative to stock INRIA. `backward.cu` was updated to
match for degrees 1 and 2 and for six of the seven degree-3 terms. **The
`sh[14]` term was missed.** Forward evaluates `SH_C3[5] * z * (zz - xx)`;
backward carried the derivatives of stock INRIA's `SH_C3[5] * y * (zz - xx)`
(∂/∂x = -2xy, ∂/∂y = zz-xx, ∂/∂z = 2zy — exactly the old polynomial). All
seven terms and both lower degrees were checked by hand; only this one was
wrong.

It corrupted `dL_dsh[14]`, the SH contribution to `dL_dmeans`, and — once
added — `dL_dcampos`. **Only at `sh_degree=3`**; degrees 0-2 are unaffected,
which is why nothing had ever noticed. Fixed in four places.

Consequence for our own record: **§13.5's "SH degree 3 gives +0.08 dB" was
measured with a corrupted degree-3 gradient** (`logs/sh3_smoke.log`, 150 views,
`--train-source all`, 19.1590 vs degree 0's 19.078 at 3000 iterations) and
should be re-run before it is cited.

Two independent reasons that conclusion never stood, and the second is the
worse one: the gradient was wrong, **and +0.08 dB is below the ~0.09 dB
run-to-run floor established below** — the comparison could not have resolved
the effect either way, bug or no bug. The reading it fed (that view dependence
is not where the revisit colour inconsistency comes from) is unsupported rather
than refuted, and re-running it needs a seed ensemble, not one run per arm.

**Audit of the rest of the fork**, since finding one forward/backward mismatch
means looking for others:

| site | forward/backward agree? | on our path? |
|---|---|---|
| SH colour | **NO — the `sh[14]` term**, now fixed | yes |
| `computeCov2D` (J, W, T = W·J, frustum clamp) | yes, expression for expression | yes |
| `computeCov3D` (scale/quat → covariance) | yes — **both** skip the quaternion normalization (`// / glm::length(rot)` in forward, the matching `dnormvdv` commented out in backward), a deliberate matched change, not a bug | **no** |
| `renderCUDA` alpha blending | stock | yes |

`computeCov3D` is dead code for this project: `cuda_splatting.py` always passes
`cov3D_precomp` and never `scales`/`rotations`, and the rasterizer's Python
wrapper rejects supplying both. That leaves the covariance-projection path and
the blend as the only live upstream code, and both check out.

**The whole bug class is now closed rather than the one instance**
(`scripts/test_sh_backward.py`). Checking only the six terms that were changed
is the wrong response to "a hand port missed one term"; every coefficient is
checked instead, at every degree, via two chains that fail differently:

- `dL_dsh[k]` for all 16 coefficients, by plain central differences. A naive FD
  **is** trustworthy here, uniquely in this rasterizer, because an SH
  coefficient changes only colour — geometry, radii and tile assignment are
  untouched, so the integer-`radii` staircase that ruins FD elsewhere is absent.
- the `dRGBd{x,y,z}` direction chain, through `dL_dcampos` with the
  translation-cancellation estimator, since campos enters *only* through the
  direction.

```
degree 1   worst coefficient sh[1]  rel 1.68e-03    direction chain 5.35e-04
degree 2   worst coefficient sh[5]  rel 1.54e-03    direction chain 7.03e-04
degree 3   worst coefficient sh[5]  rel 5.47e-03    direction chain 7.94e-04
```

**What this does NOT cover:** both chains validate the backward *against this
fork's own forward*, and the forward is exactly what was modified. If the e3nn
axis permutation were wrong relative to whatever produced the SH coefficients,
every test above still passes while the colours are wrong. That concern was
chased down and **is closed** — for a reason that also simplifies a lot else:

**The deployed system never evaluates an SH basis function at all.**

- `mast3r/catmlp_dpt_head.py:150` splits `3 * sh_degree` channels for SH, and
  `sh_degree=1` here means *one coefficient per colour channel*, not degree 1.
  The head emits **DC only**, which is direction-independent.
- The live renderer (`visualization.py:707`) calls the same
  `diff_gaussian_rasterization` fork with `sh_degree=0`, `shs=None` and
  `colors_precomp=colors` — it bypasses the SH path entirely.

So there is no second SH consumer on a different convention (a plausible
hypothesis, since `mast3r_slam_backends` exists — but it handles the pose
graph, not rasterization), and there is no possibility of the head having been
trained under a mismatched axis order, because DC carries no axis.

The convention becomes live only under `refine_gaussian_map.py --sh-degree 3`,
where the producer (`GaussianModel.sh_coeffs`) and the consumer (this fork) are
the same code path and self-consistent by construction. It would matter against
an external implementation, which we do not currently compare to.

One real limitation does survive, of a different kind: `encode_gaussians_for_ply`
writes `f_dc_0..2` and **no `f_rest_*` columns at all**, so a refined SH-3 map
silently degrades to view-independent colour on export and cannot be
round-tripped. Worth a warning in the codec before SH ≥ 1 is ever shipped.

#### Finite differences are not a usable referee here, and why

The first version of the test compared analytic gradients to a plain central
difference and reported failures everywhere. The FD was the thing that was
wrong. Alpha compositing is only piecewise smooth in the camera parameters:
`radii` is an **integer**, so a small perturbation flips primitives in and out
of the raster and the loss moves in steps. Diagnostic that settled it:
upstream's own unmodified `sum_i dL/dmeans` disagreed with such an FD by 2x on
two of three axes, and the disagreement **grew as h shrank** — noise, not
truncation.

The estimator that does work exploits a symmetry. Translating the Gaussians
*and* the camera position by the same v leaves every SH direction
`pos_i - campos` unchanged, so

```
dL/dcampos . v  =  FD[translate both]  -  FD[translate means only]
```

Both FDs share identical geometry and therefore identical staircase artifacts,
which subtract out. Stable to ~0.2% across a 5x change in step size where the
raw FD was not stable at all. This is what caught the `sh[14]` bug: before the
fix, analytic `(-15.12, -1.61, -1.23)` vs implied `(-9.42, -28.26, +3.44)`;
after, analytic `(-9.438, -28.263, +3.459)`.

#### The covariance gradient is ill-conditioned, and the identity is the worse path

`covs` agree between backends only to `cos = 0.99970`, against `1.00000000`
for `means`. Not a disagreement about the math: the per-element tail lands on
components whose reference value is ~0, and on the top 1% of components by
magnitude the median relative difference is 1.8e-3. The cause is
`denom2inv = 1/((ac - b^2)^2 + 1e-7)` in `computeCov2DCUDA`, a near-cancellation
that is badly conditioned for near-degenerate 2D covariances — and the identity
backend feeds it covariances that have been through two extra fp32 matmuls
(`R^T cov R`). **So the CUDA path is the better-conditioned of the two**, since
it passes covariances through untouched. This is a small argument in its favour
that was not anticipated in §13.8.

#### The regression gate needs a seed ensemble, not a number

§13.8 specified "re-run the 500-iteration pose experiment; it must reproduce
14.7265". It does not, and neither does the identity backend:

```
identity, 500 iters, rebuilt extension    14.6132   drift 0.0124   peak 1.9 GiB
cuda,     500 iters                       14.8127   drift 0.0121   peak 1.7 GiB
(recorded in §13.3, before any of this)   14.7265   drift 0.0165
```

The first reading of this blamed the recompile: the `active`-predicate
restructure is semantically identical but lets the compiler order
floating-point work differently. **That attribution was wrong**, and the cell
that disproves it is a same-binary rerun. Two invocations of the identical
command, same seed, same build, both the identity backend:

```
14.6132     and     14.6995      -> 0.086 dB apart, same seed, same binary
```

`renderCUDA`'s backward accumulates `dL_dcolors`, `dL_dmean2D`,
`dL_dconic2D` and `dL_dopacity` with `atomicAdd` (`backward.cu:662-693`), so
the summation order over overlapping Gaussians differs between launches.
Seeding controls view sampling and initialization; it cannot control this. The
0.11 dB "recompile drift" is therefore accounted for by atomic ordering alone —
**the compiler is exonerated, and no deterministic end-to-end regression exists
at this scale to be found.** Pose drift is stable across all runs (0.012-0.017)
and the memory difference is in the predicted direction.

**Every psnr in this document carries a floor of ~0.09 dB that no amount of
seeding removes.** This retroactively explains why single-point readings in §10
and §13 kept looking like trends, and it invalidates two reporting habits used
throughout: quoting a single run as a value, and **taking the best point of an
oscillating curve** — peak-picking across 12 evaluations on a ±0.3 dB
oscillation is a biased estimator that inflates by roughly the oscillation
width. Report a fixed iteration count, over seeds, as mean ± sd.

**The pre-registered gate was mis-specified.** "Reproduce 14.7265" cannot be
met by anything, including the code that produced it. The replacement is four
layers, each answering one question (structure suggested by review):

| layer | question | instrument | status |
|---|---|---|---|
| equivalence | do the two backends compute the same gradient? | single-step gradient comparison | **done: 7.6e-06, means `cos = 1.00000000`** |
| localization | where do they first diverge? | open-loop K≈10-50 steps, forced identical view order, compare parameter-space distance and the densify/prune decision sequence | not built |
| pathology | does the new path break over a long run? | end-to-end smoke: no NaN, no crash, not worse | partly |
| release | is the new path non-inferior? | paired TOST over seeds, margin pre-registered from the same-binary rerun spread | **done, below** |

The first layer *is* the equivalence proof: deterministic, per-parameter, and
four orders of magnitude more sensitive than any end-to-end metric. Asking an
end-to-end psnr to establish equivalence is asking an instrument to resolve
something 10,000x below its own noise floor. Conversely, a seed ensemble cannot
prove *equality* either — at sd ≈ 0.1-0.2 dB, detecting a 0.05 dB systematic
shift would need on the order of 100 seeds. What 3-5 paired seeds can do is
rule out a *material regression*, which is what a release gate is actually for.

**Result of the release layer** (desk, `--deployable --optimize-poses`, 500
iterations, paired on seed):

| seed | identity | cuda | paired diff |
|---|---|---|---|
| 0 | 14.6995 | 14.7650 | +0.0655 |
| 1 | 14.7546 | 14.6991 | −0.0555 |
| 2 | 14.8341 | 14.7687 | −0.0654 |

```
mean paired difference  -0.0185 dB   sd 0.0729   t = -0.44  (crit +-4.30)
95% CI                  [-0.200, +0.163] dB
identity mean 14.7627          cuda mean 14.7443
peak GPU memory   identity 1.9 GiB    cuda 1.7 GiB   (deterministic, every run)
mean pose drift   identity 0.0125-0.0128   cuda 0.0115-0.0138
```

Sign flips across seeds; no detectable difference, as the 7.6e-06 gradient
agreement requires. **This also retracts a speculation made above**: §13.11's
suggestion that the CUDA path might be *slightly better* because it avoids two
fp32 matmuls on an ill-conditioned covariance is not supported — one seed up,
two down. The conditioning argument stands as an argument; it has no measurable
consequence at 500 iterations.

The memory saving is the one difference that is real and repeatable: 0.2 GiB on
desk's 1.86M Gaussians, and it scales with N, so ~0.8 GiB on room's 7.35M.

### 13.12a Pose refiner or photometric sponge — decided

`mean pose translation correction` cannot tell those apart: both move cameras,
only one moves them toward the truth. `scripts/diag_pose_correction.py` scores
the optimized poses against ground truth (mapped through the same Sim3 the
refinement uses). Desk, 25 views, 3000 iterations:

| pose LR | translation error | views improved | \|correction\| / \|true error\| | cos(δ, true) | rot error |
|---|---|---|---|---|---|
| 1e-3 | 0.0242 → **0.0456 m (+88%)** | 12% | 1.85 | +0.274 ± 0.386 | 1.678 → 1.781° |
| 1e-4 | 0.0242 → **0.0230 m (−5.2%)** | 60% | 0.46 | +0.386 ± 0.460 | 1.678 → 1.648° |

**The two regimes separate cleanly, and along exactly the axis that matters.**

- **1e-3 is a sponge, confirmed.** It nearly *doubles* the pose error while
  raising psnr, moving the cameras 1.85x further than the error itself. Rotation
  degrades too. Its cos is weakly positive (+0.27), so it is a mixture — but
  residual absorption dominates. This is the setting that produced the highest
  psnr in the grid and the worst lpips, and now it is clear why.
- **1e-4 is a genuine refiner.** Error down 5.2%, 60% of views improved,
  positive cosine, rotation improved, and it *under*-corrects (0.46x) — it is
  conservative but pointed the right way.

So pose optimization at LR 1e-4 beats the fixed-pose arm on **all three axes**:
psnr, lpips, and actual pose error. That is the setting to ship.

**And the more important half: the refiner removes only 5.2% of the pose error,
against a pose gap of 4.5 dB at 50 views.** Photometric pose refinement is real
but roughly two orders of magnitude too weak to open the gate. It is worth
keeping because it is free and strictly better; it is **not** the route to the
remaining 4.5 dB. That work has to move upstream into SLAM — joint keyframe
pose/map optimization in the backend, rather than post-hoc correction.

Two limitations that must ship with this result:

1. **The ground-truth-free proxy failed.** Temporal smoothness of the δ
   sequence, proposed as the online-usable surrogate, gives ratio 0.46 for the
   sponge and 0.77 for the refiner — **backwards**. It is confounded by
   magnitude: the sponge's corrections are larger, so normalizing by their own
   mean makes them look smoother. As formulated it cannot be used online.
2. **The keyframe stratum is n=1** (a linspace sample of 25 frames catches one
   keyframe). The "keyframes improved on 100% of views" line is a single view
   and carries no information; the stratification test is unrun, not passed.

#### Replication of the shipped cell (desk, 3 seeds, paired)

| seed | fixed | pose LR 1e-4 | Δ psnr | Δ lpips |
|---|---|---|---|---|
| 0 | 14.288 / 0.3857 | 14.678 / 0.3665 | +0.390 | −0.0192 |
| 1 | 14.280 / 0.3917 | 14.660 / 0.3749 | +0.380 | −0.0168 |
| 2 | 14.291 / 0.3862 | 14.675 / 0.3687 | +0.384 | −0.0175 |

```
paired psnr   +0.3847 ± 0.0048   t = +138.9
paired lpips  -0.0178 ± 0.0012   t = -24.9     (crit +-4.30)
```

Pairing on seed drops the sd to 0.005 dB against the 0.086 dB unpaired floor,
because the same seed fixes the view sample and initialization and the tail
average over three evaluations damps the rest. The effect is not in doubt.

#### …and it does NOT replicate on room

| scene | seed | fixed | pose LR 1e-4 | Δ psnr | Δ lpips |
|---|---|---|---|---|---|
| room | 0 | 12.324 / 0.5257 | 12.322 / 0.5254 | **−0.002** | −0.0003 |
| room | 1 | 12.344 / 0.5258 | 12.350 / 0.5247 | **+0.006** | −0.0011 |
| desk | 0-2 | — | — | +0.385 | −0.0178 |

**Pose optimization buys nothing on room** — ±0.005 dB, indistinguishable from
zero, against +0.385 dB on desk. The shipped-cell claim is **desk-specific**
and must not be quoted as a recipe.

Room's whole refinement is weaker: 11.18 → 12.32 (+1.14 dB) against desk's
12.37 → 14.36 (+1.99 dB). The likeliest cause is supervision starvation —
room's map is **7.35M Gaussians against desk's 1.86M**, and both were given the
same 25 views. That is 294k Gaussians per view versus 74k.

**So the "25 views saturates" result is probably also desk-specific**, and the
view budget should scale with map size, not be a constant. The view sweep has
to be repeated on room before any of §13.10's budget claims can be stated as
general. Until then, the deployable configuration is a desk result.

#### A quaternion-order bug in the refinement script, found via a 1.32 dB save/load loss

A map scoring 19.0375 reloaded at 17.7225. Two causes, and the second is the
real one.

**(a) A stray clamp.** `save_refined()` converted `f_dc` to RGB and let
`encode_gaussians_for_ply` convert it back — a no-op round trip except for a
`.clamp(0, 1)` in the middle. `f_dc` is a free pre-activation parameter and the
optimizer drives it outside the [0,1] box routinely: 2.9% of colour channels
came back pinned at a bound. Inherited from the SLAM-export path, where colours
genuinely are in range. Removed — and the loss barely moved (17.7225 →
17.6827), which is what sent the search deeper.

**(b) `GaussianModel.covariances()` was passing the quaternion in the wrong
component order, on every run this project has ever done.**

```
utils/geometry.build_covariance(scale, rotation_xyzw)
  -> quaternion_to_matrix, which unbinds `i, j, k, r`
     with the comment "Order changed to match scipy format!"   => (x,y,z,w)

gaussian_ply_codec / the 3DGS .ply format                      => (w,x,y,z)
```

`covariances()` fed the stored `quat_wxyz` straight in, rotating every component
by one position. Measured against the covariance the decoder computes with
scipy: **median 71% relative error, max 178%.**

**Impact assessment — and the good news is the important part.**

| | |
|---|---|
| initial map, correct covariance (`eval_map_quality`) | **12.4416** |
| initial map, as `refine_gaussian_map` saw it | 12.3719 |

**0.07 dB.** Scrambling the orientations costs almost nothing on the input map
because the Gaussians are only mildly anisotropic (median condition number 4.5)
and 1.86M of them overlap heavily. So **every refinement result recorded before
this fix stands**: the optimizer worked in an oddly parameterized but
self-consistent frame, and the psnr/lpips reported are honest measurements of
what was actually rendered.

It is fatal on *save*, though, and that is why it surfaced here. After
optimization the orientations carry real fitted information, and the two
misreadings do **not** cancel — the codec writes (w,x,y,z), `covariances()`
read (x,y,z,w) — so a saved map is scrambled on reload. That is the 1.32 dB.

Fixed by reordering inside `covariances()`. Verification: the model's covariance
now matches the decoder's to **1.2e-07 median**, and the refinement script's
iteration-0 score is now **12.4416**, exactly equal to `eval_map_quality`'s
independent measurement of the same file. Two code paths that had never been
cross-checked now agree.

**Consequence for the numbers in this document:** the feed-forward + fusion
anchor is **12.4416, not 12.372**. Every refinement delta quoted against 12.372
is 0.07 dB optimistic. The desk and room replications are being re-run under the
fixed code, since a paired comparison must not straddle the change.

### 13.12b The controlled recovery test — separating "the method is weak" from "the map is co-adapted"

The 5.2% recovery figure conflates two things: whether photometric refinement
**can** recover a known pose error, and whether it can do so starting from a map
that was *built with those very errors baked in*. The SLAM map is co-adapted to
its own estimated poses — the initialization sits in a basin that is
self-consistent with them, and the route to the correct configuration may run
through worse photometric loss.

Design (`--perturb-poses`, review's proposal):

- **Stage A** optimize the map at ground-truth poses (50 views, 3000 iters) and
  persist it. This map is *not* co-adapted to estimation error.
- **Stage B** inject a known rigid perturbation of the same magnitude as the
  real error (0.024 m per axis, rotation scaled to the measured 1.7°/0.024 m
  ratio), then run pose refinement and measure how much is recovered. Arms:
  LR {1e-4, 3e-4} × {3000, 9000} iterations — the second factor also answers
  "were 3000 iterations enough", which the existing logs cannot (|δ| is not
  recorded per step).

Pre-registered reading:

- **recovery > 50%** → the method works; the real-world 5.2% is a
  co-adaptation/basin problem → post-hoc refinement is worth tuning (anchor
  prior *plus* a higher LR — the prior alone is a shrinkage force and would
  lower recovery, its role is to close off the sponge direction).
- **recovery < 20%** → photometric pose observability is intrinsically poor on
  this scene → the 4.5 dB gate can only be closed at build time, and the case
  for reworking the SLAM backend is quantitatively made: gate 4.5 dB, post-hoc
  recovery ≤ 0.4 dB.

#### Result: zero recovery, in every arm, and worse the harder it tries

Map optimized at ground-truth poses (19.0037 held-out; save/load verified
lossless after the quaternion fix), known perturbation of 0.0390 m injected:

| pose LR | iters | recovered | pose moved | held-out psnr | lpips |
|---|---|---|---|---|---|
| 1e-4 | 3000 | **−2.2%** | 0.0074 m | 18.56 | 0.2523 |
| 1e-4 | 9000 | **−4.0%** | 0.0104 m | 18.40 | 0.2423 |
| 3e-4 | 3000 | **−4.3%** | 0.0100 m | 18.31 | 0.2938 |
| 3e-4 | 9000 | **−11.5%** | 0.0147 m | 17.92 | 0.2889 |

38-48% of views improved, against 50% for chance. **All four arms push the
poses further from the truth, monotonically worse with more learning rate or
more iterations.**

The pre-registered `< 20%` branch is met, and then some. All three "you didn't
tune it" explanations are dead:

- *map co-adapted to its own pose error* — this map was never built at
  estimated poses;
- *LR too small* — 3e-4 is worse than 1e-4;
- *3000 iterations too few* — 9000 is worse than 3000.

The cameras move only 0.0074-0.0147 m against an error of 0.0390 m: the
optimizer barely moves them at all, rather than moving them the wrong way.
That is the signature of **a photometric loss with almost no gradient along the
true-pose direction**, not of a bad basin.

**Verdict: photometric pose refinement cannot recover pose error on this
scene.** Neither implementation — the Gaussian-inverse identity nor the CUDA
camera gradient — changes this; they compute the same gradient, and the
gradient is the problem. The 4.5 dB pose gate has to be closed when the map is
built, which is the quantitative case for reworking the SLAM backend: gate
4.5 dB, post-hoc recovery ≤ 0.37 dB on one scene and 0 under controlled test.

(An earlier run returned −1.8% but used a map degraded by the export bugs
above; it was discarded and re-run. Same answer.)

#### 13.12c Why the offline ceiling is ~19-20 dB: it is not view count

`scripts/diag_ceiling.py` scores the ground-truth-pose map on the very views it
was **fitted to**, at the poses it was fitted at:

```
TRAIN views (supervised)   psnr 20.51   lpips 0.2869
HELD-OUT views             psnr 19.00   lpips 0.3250      (matches refine's 19.0037)
train - held-out           +1.51 dB
```

Only 1.5 dB separates in-sample from held-out. **The ceiling is not
generalization and not coverage** — which is why adding views saturated. With
ground-truth poses and 50 supervision views, the map reaches 20.51 dB *where it
was explicitly fitted*. The cap is the representation, the optimization, or the
measurement — not the amount of supervision.

**And the measurement deserves suspicion.** The Sim3 that places ground-truth
poses into the map's frame is fitted on 14 keyframes and leaves

```
residual: mean 0.0149 m   median 0.0125 m   max 0.0297 m
```

against the **0.024 m** pose error the refinement is trying to remove. The
instrument's own placement error is ~60% of the effect size, so an unknown part
of the "4.5 dB pose gap" is alignment artifact rather than map quality. The
injection experiment above is *immune* to this — it perturbs and scores against
the same mapped pose, so the Sim3 error cancels — but every est-vs-GT
comparison in this document inherits it.

**A measurement error of my own, caught here:** the first version of this
diagnostic reported 25.28/23.77 because it carried a `/3.0` into the psnr
formula, inflating everything by exactly 10·log10(3) = 4.7712 dB. Both
`eval_map_quality.py:245` and `refine_gaussian_map.py:528` use
`-10·log10(mse)` with `mse` averaged over all elements. The corrected held-out
number then matched refine's independent measurement of the same file to
0.007 dB, which is what confirmed the fix.

#### Where the review and I disagree, unresolved

The argument that "25 views is not too few to constrain the poses" was made
from the 250-view cell (+0.32 dB, drift 0.0051, materially the same as 25
views' +0.35) — "10x the views changed nothing". **That comparison is
confounded**: at 3000 iterations, 250 views give each pose **12 gradient
updates** while 25 views give **120**. The lower drift at 250 views is what
under-training predicts, not what a stronger constraint predicts. Multi-view
observability is therefore *not* ruled out; testing it needs matched per-pose
update counts (250 views × 30000 iterations), which has not been run.

### 13.12d The pose question, and how its answer inverted three times

Worth recording as a sequence rather than as a conclusion, because each
reversal came from a specific methodological failure and the same failures are
easy to repeat. Every stage below was reported as settled at the time.

**Stage 1 — "photometric pose refinement works, +0.73 dB."** (§13.3) Poses made
optimizable via the Gaussian-inverse identity, 14.0003 → 14.7265. Read as
recovering 40% of the 1.81 dB pose gap.

**Stage 2 — "it is a distortion dial, not a refiner."** The pose LR sweep found
psnr and lpips moving in opposite directions, monotonically, and the default
LR sat at the extreme. The `--save-poses` diagnostic then showed LR 1e-3 nearly
*doubling* the pose error while raising psnr — a sponge — and LR 1e-4 reducing
it 5.2%. The +0.73 dB was mostly the sponge.

**Stage 3 — "the method cannot work at all."** The controlled injection test
(inject a known error into a map built at ground-truth poses, measure recovery)
returned −2.2%, −4.0%, −4.3%, −11.5% across four arms, monotonically worse with
more learning rate or more iterations. Three explanations were eliminated in
turn — co-adaptation, learning rate, iteration count — and the conclusion drawn
was that the photometric loss has no gradient along the true-pose direction,
which would have made the SLAM-backend rebuild the only route. **This was
reported to the user as settled.**

**Stage 4 — the experiment had been measuring nothing.** A zero-training
loss-versus-displacement scan (`diag_pose_observability.py`) contradicted stage
3 outright: the loss rises steeply and monotonically with displacement — +19%
mse at 5 mm, +141% at 2 cm. Pose is *highly* observable at the operating point.
A steep basin, a gradient verified to 7.6e-06 against an exact reference, and
an optimizer that does not descend it is not a physics result; it is a bug.

It was. `PoseDeltas.delta(i)` is built from `rot`/`trans` alone and never reads
the `R0`/`t0` buffers, and the training loop renders from `train_frames`, which
the injection code never touched. **The perturbation lived in an unused buffer.**
The optimizer started at the optimum with no gradient, random-walked 0.005-0.015
m, and the "recovery" metric scored that walk against an error that was never
applied.

**Stage 5 — with the perturbation actually applied: it works, and the map is
the obstacle.**

| | recovered | views improved |
|---|---|---|
| map frozen, LR 1e-3 | **57.7%** | 90% |
| map frozen, LR 1e-4 | 40.6% | 82% |
| map free, LR 1e-3 | 8.2% | 62% |
| map free, LR 1e-4 | 14.7% | 74% |

Freezing the map takes recovery from 8-15% to 41-58%. The map has ~1.86M x 14
free parameters against the poses' 50 x 6, and it absorbs an injected pose error
long before the poses can undo it.

**Stage 6 — but on real SLAM poses, freezing changes nothing.** `--alternate`
was implemented and run with a matched update budget (6000 alternating
iterations = 3000 map updates = the baseline):

| arm | psnr | lpips | pose err 0.0248 → | recovered | cos |
|---|---|---|---|---|---|
| fixed | 14.454 | 0.3890 | — | — | — |
| joint LR 1e-4, 3k | 14.712 | 0.3666 | 0.0224 | −9.8% | +0.394 |
| alt-100 LR 1e-4, 6k | 14.752 | 0.3646 | 0.0227 | −8.4% | +0.380 |
| joint LR 1e-3, 3k | 14.819 | 0.4733 | 0.0451 | **+81.8%** | +0.264 |
| alt-100 LR 1e-3, 6k | 14.887 | 0.4613 | 0.0442 | **+78.0%** | +0.250 |

Alternating ≈ joint, within noise. (An earlier run showed alternating *worse*;
that was purely the budget confound — the pose phases zero the map's learning
rate, so 3000 alternating iterations give the map only 1500 updates.)

Two things do get confirmed here, on real poses rather than synthetic ones:
LR 1e-3 **is** a sponge (pose error 78-82% worse, 16% of views improved), and
LR 1e-4 **is** a genuine refiner (error −9%, 62% of views improved, cos +0.38,
and rotation improves too: 1.753° → 1.654°) that improves psnr *and* lpips.

**The open paradox:**

```
synthetic perturbation + map built at GT poses  + frozen   ->  58% recovered
real SLAM error        + map built at est poses + frozen   ->   9% recovered
```

Six-fold. The two differ in exactly two ways: whether the map is co-adapted to
the pose errors it was built from, and whether the error is an iid rigid
perturbation or real SLAM error that is correlated across views. The
discriminating experiment — inject a synthetic iid perturbation into the
SLAM-built map, freeze, measure recovery — is running.

One quantitative caveat that shrinks the gap: the Sim3 alignment residual is
0.0149 m against a real pose error of 0.0248 m, so the **measurable** recovery
ceiling is about (0.0248 − 0.0149)/0.0248 ≈ 40%, not 100%. Against that target
9% reads as ~23%.

**Methodological lessons, which are the reusable part:**

1. **A negative result from an optimizer that will not descend a verified
   gradient should be treated as a bug report, not a finding.** Stage 3 was
   internally consistent across four arms and two learning rates and was still
   entirely an artifact. What broke it was not more arms — it was one cheap
   measurement of the loss surface itself, with no training involved.
2. **Instrument the thing you claim to be manipulating.** The injection code
   printed the injected magnitude from the buffer it wrote, not from what the
   renderer received, so it reported a perturbation it was not applying.
3. **Match the budget before comparing schedules.** Alternating looked worse
   than joint until the map's update count was equalized.
4. **`mean drift` is not evidence of correction.** Sponge and refiner both move
   the cameras; only ground truth separates them, and the two regimes here sit
   at the same drift magnitude with opposite signs of effect.

### 13.12 What this buys the actual deliverable (end-to-end online reconstruction)

The project's goal is a real-time Splatt3R-SLAM system, not a table of offline
ablations. Scored against that:

**Capability the system did not have.** Per-frame pose persistence (§13.10) is a
main-path change, verified and in place. Every form of supervision beyond the
~14-51 keyframes depends on it.

**A stage-3 recipe that can ship**, and a number for it:

```
12.44   stage 1+2 — what the system produces today
14.71   + stage 3 at 25 views, pose optimization at LR 1e-4     +2.3 dB, lpips also better
```

Replication in progress agrees closely across seeds (paired differences
+0.390/+0.380 dB psnr, −0.019/−0.017 lpips at the first two), which matters
because the effect has to clear a 0.086 dB nondeterminism floor.

**A sized budget, which is what real time actually needs.** 25 supervision
views, not 150; ~300 iterations under fixed poses. This is a measured
saturation point, not an assumption, and it cuts the per-keyframe supervision
cost to roughly a sixth of what the original plan assumed.

**Two things removed from the plan.** The CUDA camera gradient is *not*
required for real time (§13.11's closing section retracts the argument that
made the case), and unlocking 150-250 views is worthless past ~25. Both were
scheduled work.

**The direction correction, which is the most valuable output.** Pose quality
is not one of two levers; it is the sole gate, and it is multiplicative — the
pose gap grows from 1.8 dB at 14 views to 4.5 dB at 50. So the highest-value
remaining work for the end-to-end goal is improving the poses, or improving the
map's consistency with them, **not** more iterations, more views, or a faster
rasterizer. Whether *photometric* pose refinement is a usable route to that is
exactly what `scripts/diag_pose_correction.py` decides.

**The gap that must not be glossed.** None of stage 3 has ever run inside the
SLAM loop. Every number above is offline: export a `.ply`, refine it post hoc,
score it offline. The remaining work for "end-to-end online" is the background-
thread integration and the non-stationarity of Gaussians injected each
keyframe — a main-path change that has not been started.

And in absolute terms 14.7 dB is not a good reconstruction. The offline ceiling
with ground-truth poses is ~19. **That entire remaining gap is pose.**

---

## 13.13 P1 design: moving refinement into the SLAM loop (stages 1-3 BUILT and PASSED; see 13.14)

Every number in §13 is offline: export a `.ply`, refine it post hoc, score it
offline. This is the design for the online version, and it is shaped by three
measurements rather than by convenience.

### What the evidence dictates

**1. The optimizer must own keyframe-LOCAL Gaussians, never a baked world map.**

The decisive experiment (§13.12b, Q3 arm): the same synthetic iid perturbation,
the same frozen-map protocol, differing only in which poses the map was built
from —

```
map built at GT poses     +51.8% recovered   84% of views improved
map built at SLAM poses   -22.2% recovered   42% of views improved
```

A map baked from estimated poses does not merely fail to support pose
correction; it **actively pulls poses away from the truth**, because its
photometric optimum sits at the estimated poses. Co-adaptation is created at
the moment of baking.

The fix is to never bake. Store each keyframe's Gaussians in that keyframe's
own frame and compose `world = T_WC[kf] ⊗ local[kf]` at render time; when the
backend corrects `T_WC[kf]`, the map re-deforms with the trajectory and the
learned content is carried along rigidly.

**The SLAM system already does this — the refinement pipeline is what threw it
away.** `SharedKeyframes.gs_*` holds camera-space Gaussians, and
`visualization.py:545-578` bakes per keyframe with a cache keyed on
`(T_WC, stride)`, so a pose correction invalidates and re-bakes automatically.
`refine_gaussian_map.py` flattens all of it into one world-space array and
optimizes that.

**2. Densification is worth having online (+1.4 dB), and it breaks the shared
buffers.** `--densify` over 30000 iterations takes desk from 19.07 to 20.51
held-out. But `SharedKeyframes.gs_means` is `(buffer, h*w, 3)` — exactly one
Gaussian per pixel, fixed. So the optimizer cannot live in those buffers; it
needs its own variable-length per-keyframe storage.

**3. Joint map+pose optimization is the wrong default.** Frozen/alternating
recovers 41-58% of an injected pose error against 8-15% joint (§13.12d stage 5).
Whatever pose refinement runs online should default to phases in which the map
is held still, not to joint descent.

### Structure

A third process alongside `viz` and `backend` (matching the existing
architecture; `SharedKeyframes` is already shared CUDA memory).

```
main ── tracker ── SharedKeyframes ─┬─ backend  (pose graph; writes T_WC)
                                    ├─ viz      (reads; renders)
                                    └─ refiner  (NEW)
```

The refiner owns, per keyframe: a variable-length set of local Gaussian
parameters (initialized from `gs_*` at keyframe creation, then free to grow via
densification) and their Adam state. It never writes `SharedKeyframes.gs_*`, so
there is exactly one writer per buffer. It publishes a flattened world-space
snapshot on its own double-buffered shared array for `viz` to draw.

Rendering inside the refiner composes local Gaussians through the keyframe's
*current* `T_WC`, read under the existing lock — the same rigid transform the
identity experiment uses, validated to 7.6e-06 (§13.11).

### The four problems that make this non-trivial

| problem | approach |
|---|---|
| **Non-stationarity** — new keyframes arrive mid-optimization | Append their Gaussians and extend Adam state. `_replace_params()` already performs exactly this surgery for densification. |
| **Pose corrections under the optimizer's feet** | Nothing to do, by construction: parameters are local, so a `T_WC` change re-deforms the map without touching them. This is the whole point of the design. |
| **Iteration budget** | §13.10's measured rule: supervision saturates at ~40k Gaussians per view (desk knee ~45 views / 1.86M; room ~200 / 7.35M). The view count therefore scales with map size, and the per-frame iteration budget follows from the frame rate. |
| **Supervision views** | `FramePoseLog` (§13.10) already persists anchor-relative poses for every tracked frame. Online, sample a sliding window of recent frames plus a reservoir of older ones. |

### Staging, each independently checkable

1. **Refiner process on a static map.** Run it against a finished SLAM run, no
   new keyframes, poses fixed. Target: reproduce the offline number
   (desk 12.44 → 14.45 at 50 views). Validates plumbing only.
2. **Local-frame parameterization.** Same, but with per-keyframe local
   parameters composed through `T_WC` instead of a baked map. Must match stage 1
   within noise; the point is that it now survives a pose change.
3. **Re-anchoring under loop closure.** Inject a synthetic `T_WC` correction
   mid-optimization and confirm the map follows without quality loss. This is
   the property none of the offline work has.
4. **Online, growing map.** Keyframes arriving live; densification enabled.
   Verify ATE is unchanged (the refiner must not perturb tracking) and that map
   quality tracks the offline curve.
5. **Optional pose phases.** Only after 1-4, and defaulting to map-frozen.

### Pre-registered failure conditions

- ATE moves at all → the refiner is stealing GPU from tracking; back off its
  duty cycle.
- Stage 2 does not match stage 1 → the local-frame composition is wrong; the
  identity test (§13.11) is the reference.
- Stage 3 loses quality after a synthetic correction → re-anchoring is not
  actually rigid; suspect the covariance transform (the quaternion-order bug of
  §13.12a lived exactly there).

### What this does NOT claim

It does not close the 4.5 dB pose gate. It removes the mechanism that *creates*
co-adaptation, so that any upstream pose improvement — existing loop closure, a
future joint backend, or photometric refinement on keyframes — converts into map
quality instead of being fought by a map baked at the old poses. Whether the
gate then closes is a separate question, and the honest current answer is that
post-hoc recovery measures between ~11% (real error, soft) and ~58% (iid
perturbation, clean map only).

---

## 13.14 P1 build log — what is implemented, what it measured, what it killed

State as of this section: **stages 1-4 built and passing (stage 4 = §15.8),
stage 5 (pose phases) not started.**
Handover-oriented: file by file, number by number, including the two results
that changed the plan's justification.

### Code that exists now

| file | what it is |
|---|---|
| `splatt3r_slam/refiner.py` | `LocalGaussianMap` (all keyframes' Gaussians as one flat parameter set tagged by `kf_id`, parameters in the owning keyframe's camera frame, `world(kf_mats)` composing through *current* poses); `SupervisionFrames` (bounded recent ring + Vitter reservoir); `_optimizer_for` (Adam moments carried across parameter-count changes); `render_map`; `run_refiner` (the online loop — **written, never run**) |
| `splatt3r_slam/splatt3r_utils.py` | `prepare_gaussians_local` extracted from `bake_gaussians_world`, which now delegates to it. Verified behaviour-preserving: re-running SLAM produced a byte-identical map (`md5 956eccfd…`) |
| `splatt3r_slam/evaluate.py` | `save_keyframe_gaussians` — per-keyframe camera-space Gaussians + pose, which the baked `.ply` cannot express |
| `main.py` | `--dump-keyframe-gaussians` |
| `scripts/refine_local.py` | stages 1-3 driver: `--stage`, `--reanchor gt`, `--perturb-mode {global,perkf}`, `--seam-report` |

### Stage results (desk, 50 supervision views, 3000 iterations, deployable protocol)

**Stages 1 vs 2 — the local parameterization is equivalent.**

```
iter    stage 1 (world-space)   stage 2 (keyframe-local)
1000    14.3513                 14.3488
2000    14.5154                 14.4700
3000    14.3904                 14.3762
```

Within 0.01-0.05 dB. `A·means + b` / `A·cov·Aᵀ` reproduces baking.

**Stage 3, global Sim3 corrections — free, as designed.** Six injected
corrections (rotation to 7.1°, scale 0.91-1.10, translation to 0.121 m):
held-out unchanged at every checkpoint (14.4007 vs 14.3989 at 1500; 14.3773 vs
14.3762 at 3000), and the seam split unchanged. **This is the property no
offline result has.** Caveat stated in the code: a global similarity moves every
cluster rigidly together and therefore *cannot* tear a seam.

**Stage 3, per-keyframe differential corrections — no seam tearing.** σ=0.02 per
keyframe, supervision cameras held still:

```
              before   after    drop
overlap       14.0598  13.0913  -0.968
single-cover  14.6830  13.7311  -0.952
```

The damage is uniform, not concentrated in overlap regions — if seams tore, the
overlap column would fall further. The map re-fits within ~500 iterations and
ends at 14.5499, above the unperturbed 14.3762.

Seam risk is not hypothetical: **86.7% of pixels in the median held-out view are
covered by ≥2 keyframes**, so this is the dominant operating condition.

Unexplained and possibly structural: after optimization, single-coverage views
score *higher* than overlap views (14.70 vs 14.07) although at initialization
the order is reversed (12.08 vs 12.83). The lpips gap (0.349 vs 0.395) is
proportionally larger than the psnr gap, which points at a structured artifact
(two shells interfering) rather than slower convergence. Discriminators
proposed in review, not yet run: per-cluster depth analysis in overlap regions,
and a de-clustering ablation (assign each overlapping patch one owner, delete
the other, re-optimize).

### The result that removed this plan's headline justification

**Re-anchoring the map from estimated to ground-truth keyframe poses is worth
+0.07 dB (before optimization) and +0.13 dB (after).**

```
                init      500      1000
est anchors     12.4416   14.3292  14.3483
GT anchors      12.5119   14.4625  14.4366
```

This was the decisive test of P1's value proposition: if each keyframe's
Gaussians were merely *placed* wrongly, swapping in correct anchors should have
recovered much of the 4.5 dB pose gap with no optimization at all. It recovers
essentially nothing. **The pose error is not in the placement; it is already
baked inside each keyframe's cluster** — the head's prediction passes through
tracking and fusion (`update_pointmap` registers against the previous keyframe),
so the per-keyframe geometry is itself co-adapted.

Consequences, stated plainly:

- The trajectory-anchored representation's value is **not** recovering the pose
  gap. It is that a loop closure costs nothing (stage 3), which the offline
  pipeline cannot do. That is real but much smaller than hoped.
- The 4.5 dB pose gate looks **closed at map-build time**. Anything post-hoc —
  this representation, photometric pose refinement, alternating schedules —
  addresses placement, and placement is not where the error lives.
- Combined with 13.12b (a map built at estimated poses gives −5.7% recovery
  against +40.6% for one built at ground-truth poses, so co-adaptation destroys
  the pose signal rather than displacing it), the remaining lever is the fusion
  stage itself.

### Five measurement bugs caught in this build, all self-inflicted

Recorded because the rate matters for how much any single number should be
trusted, and because four of the five produced a *plausible, internally
consistent* wrong answer first.

1. **Quaternion order** in `refine_gaussian_map.covariances()` — (w,x,y,z)
   fed to a function wanting (x,y,z,w). Median 71% covariance error; cost
   0.07 dB on the input map but 1.32 dB on save/load.
2. **`--save-ply` clamp** — `f_dc` clipped to the [0,1] RGB box on export;
   2.9% of channels pinned.
3. **`/3.0` in a psnr formula** — inflated `diag_ceiling` by exactly
   10·log10(3) = 4.7712 dB. Caught before reporting.
4. **Perturbation never reaching the renderer** — the injection wrote into
   `PoseDeltas`' unused `R0`/`t0` buffers while the training loop rendered from
   unperturbed poses. Produced "zero recovery across four arms", which was
   reported as a settled negative result and then retracted.
5. **Index-space mismatch** — `main.py:320` calls `dataset.subsample(2)`, so
   `frame_id` indexes the 307-entry subsampled stream while every analysis
   script loads the dataset unsubsampled (613). Using a raw `frame_id` to index
   `ds_ts` mismatched keyframes to ground truth by up to 1.8 m / 126°, briefly
   reported as a 4.7 dB "re-anchoring loss". The existing pipeline is immune —
   it associates through timestamps — the bug was confined to new code.

The pattern: an experiment that *does nothing* looks exactly like an experiment
with a strong negative result. The counter is to measure the thing being
manipulated, and to check a claimed effect against an independent
instrument before reporting it (a zero-training loss-vs-displacement scan
falsified #4; a matrix-difference printout falsified #5).

### What remains, in the order review ranked it

**(f) Causal replay — go/no-go, cheap, and not yet done.** Every gain in this
document was measured *post hoc*: the optimizer saw supervision from the whole
sequence. Online, at keyframe k only supervision up to k exists, and early
Gaussians are optimized long before later views arrive. Whether the gain
survives that constraint has never been tested. It needs no process
integration — replay the trajectory offline, allow only past supervision,
compare. **If the gain evaporates here, (a)-(e) are wasted work.**
**DONE — GO at every budget (see §15.1): +1.21 dB at the strict-real-time
budget, +1.95 dB at 500 iters, matching or beating post-hoc from 500 up.**

**(b′) The faithful differential seam test.** The version run above holds the
supervision cameras still, which the real system never does — `FramePoseLog`
recomputes anchored non-keyframe poses when their keyframe moves. The faithful
version perturbs half the keyframes *and* recomputes the anchored supervision,
then splits held-out views three ways (covered only by perturbed keyframes /
only by unperturbed / spanning both). Only the third class can show seam cost,
and it then has a correct answer of "no loss".

**(g) GPU contention.** The failure mode is not a slow refiner but a refiner
that preempts tracking — dropped frames, ATE regression. Needs tracker latency
as a hard metric alongside ATE, peak memory and per-keyframe optimization
latency, plus a degradation policy (drop optimization steps, never drop frames).
**DONE (see §15.2): same-GPU contention doubles tracker latency (101→206 ms
p50, 8.0→4.5 fps) without touching ATE; cross-GPU is noise (+6%). Deployment
answer: refiner on the second GPU; single-GPU needs a duty-cycle policy that
does not exist yet.**

**(a) Process integration** — `run_refiner` written, never run; needs a map
version counter so the viewer knows when re-anchoring happened.
**DONE (see §15.8): runs live with ATE untouched and +1.24 dB at duty 0.25.
The version counter remains open only for the (unbuilt) viz display of the
refined map; the map itself is persisted as `<seq>_refined.ply`.**

**(c) Online densification and pruning** as discrete lifecycle events: birth
filtering at injection, pruning only at phase boundaries, and a re-anchoring
event that de-duplicates clusters pulled together. Gated on the de-clustering
ablation above.

**(d)(e)** Hard metrics and the supervision-buffer ablation (window length and
recent/reservoir mix currently have **no** principled value).

### Honest summary of where the deliverable stands

```
12.44   stage 1+2 — what the system produces today
14.45   + refinement, 50 views, poses fixed          measured offline
14.75   + pose optimization at LR 1e-4               measured offline, desk only
~19     offline ceiling with ground-truth poses      not deliverable
20.5    in-sample ceiling, GT poses, 30k + densify   the representation's own limit
```

Nothing above 12.44 has ever run inside the SLAM loop. The gap from 14.75 to 19
is pose error that is baked in at fusion time and is not reachable by any
post-hoc method tested here.

---

## 14. Not yet done

- ~~**Route B on the other three families.**~~ **DONE** — all four families
  have a 40-epoch head in `checkpoints/head_only_long/{tum,7-scenes,euroc,eth3d}`
  (see §8.1 for per-family deltas: TUM +1.78 dB, 7-Scenes +2.18 dB). Stage 1 is
  complete and saturated; §13.1 shows further head training is not the lever.
- ~~**SLAM-level validation.**~~ **DONE — see §9.** Both predictions held:
  ATE bit-identical, gain appears purely in map quality (+0.90 dB).
- ~~**`normalize_exposure` train/inference mismatch.**~~ **CLOSED ALREADY
  (2026-07-28), this note was stale.** The training-side counterpart exists:
  `splatt3r_core/data/common.py: SequenceExposureLock` (one locked gain per
  sequence, first frame → target mean 0.5, same [0.4, 2.5] clamp as the
  deployment side), wired into all four family adapters and ON by default.
  Written 2026-07-28 00:27, i.e. ~22 h BEFORE the first 40-epoch run started
  (log ctimes 07-28 22:17+) — so every production head in
  `checkpoints/head_only_long/` was trained WITH it (probe verified: desk's
  locked gain [0.856, 0.911, 0.891], frame-0 mean ≈ 0.496 per channel). The
  deliberate design difference that remains: training canonicalizes the
  SEQUENCE level while preserving within-sequence variation (the model
  learns exposure invariance), deployment equalizes per-frame toward the
  first frame (removes that variation) — the benign direction of mismatch
  (deployment inputs are a subset of training inputs), and its compatibility
  is not just argued but measured: §9's SLAM-level validation ran the
  lock-trained head through the deployment path with per-frame
  `normalize_exposure` active (+0.90 dB map quality, ATE bit-identical).
- **Decoder-only LoRA ablation.** Would isolate §3.3's mechanism: LoRA the
  decoder while leaving `_encode_image` (what the retrieval head and matching
  consume) untouched. Would also determine whether the retrieval-asset refit in
  `splatt3r-retrieval-refit` is needed at all — under Route B it is not, since
  encoder features are bit-identical to base.
- **Longer Route B training.** The 6-epoch runs stopped while still improving.
  A 40-epoch TUM run is **in progress** (see §8.1); at epoch 15 it reads
  psnr 11.7768 / lpips 0.2550, i.e. +1.45 dB over base, already well past the
  6-epoch run's +1.00 dB, with per-epoch gains decaying to ~0.02-0.05 dB.
  Note the trajectory is not monotone: epochs 12 and 14 each regressed and
  were reversed at 13 and 15. On a *seeded* draw those wobbles are real weight
  changes, not sampling noise (§2.1) — which is exactly why single-epoch
  readings still must not be treated as trends.

---

## 15. The two go/no-gos for online refinement: (f) causal replay and (g) GPU contention — both MEASURED (2026-08-02)

§13.14's review ranked two tests ahead of all engineering: (f) whether the
refinement gain survives the causality constraint of a live system, and (g)
whether a concurrent refiner starves tracking. Both are now measured on desk
(the standard cell: 14 keyframes, 1,860,034 Gaussians, 50 supervision views,
held-out scored at mapped GT poses, est anchors). Verdicts: **(f) GO at every
budget; (g) the answer is the second GPU.**

### 15.1 (f) Causal replay — the gain does not evaporate

`scripts/refine_causal.py` (new). Two arms in one harness, differing ONLY in
the timing of supervision availability and Gaussian injection: same map, same
held-out views, same 50-view supervision set, same total iteration budget,
same loss/LRs, same means-LR extent (computed from the full map in both arms —
run_refiner's first-keyframe extent would confound causality with a learning
rate; flagged as a design note for it). The causal arm injects keyframes at
their trajectory timestamps, unlocks supervision views at their dataset
timestamps (round-end availability — round-start silently drops every view
arriving after the last keyframe), allocates iterations ∝ elapsed sequence
time, and rebuilds Adam on every injection via `refiner._optimizer_for` (the
online mechanism). Post-hoc control arm replicates refine_local stage 2 to
0.0015 dB (14.3747 vs 14.3762), so the harness is not a confound.

| budget | causal psnr | posthoc psnr | Δ | causal lpips | posthoc lpips |
|---|---|---|---|---|---|
| 120 (strict real-time, see 15.3) | 13.6511 | 13.8002 | −0.15 | 0.5088 | 0.4870 |
| 500 (~3 min polish) | 14.3942 | 14.3223 | +0.07 | 0.4600 | 0.4311 |
| 1000 (~6 min polish) | 14.4747 | 14.3516 | +0.12 | 0.4303 | 0.4096 |
| 3000 (~17 min polish) | 14.4123 | 14.3747 | +0.04 | 0.3809 | 0.3726 |

(init is 12.4416 / 0.5027.) The feared failure — early Gaussians ruined by
early-only supervision — does not occur: at the strict-real-time budget the
final map is **+1.21 dB over no refinement**, and from 500 iters up the causal
arm matches or slightly beats post-hoc on psnr while trailing ~0.02-0.03 on
lpips. Treat the psnr lead as "causality is free" (within noise: the post-hoc
trace itself wobbles ±0.1 dB between checkpoints), not as "causality helps".
Caveats: one sequence, one seed; the mid-run evals of a partially-built map
(e.g. 11.05 dB at iter 500 of the 3000-iter causal run) measure map
incompleteness, not optimization failure. A further confound, flagged in
review: the causal arm iterates on a SMALLER map for much of the run
(keyframes are injected over time), so equal total iterations give each
Gaussian MORE updates than in the post-hoc arm — a curriculum effect. As a
deployment answer this is the correct comparison (wall-clock iterations are
the real budget); as a scientific claim about the causal constraint alone it
needs that label.

### 15.2 (g) GPU contention — latency doubles on one GPU, is free on two

`main.py --frame-timing CSV` (new, permanent) logs per-frame track_ms /
backend_wait_ms / iter_ms; `scripts/exp_gpu_contention.sh` runs the arms;
`config/rt_calib.yaml` = eval_calib but multithreaded (the deployable mode).
The load is the real refinement workload (full-map render+backward on the desk
dump), not a synthetic hog. 306 frames per arm:

| arm | track p50 | iter mean | sustainable | ATE rmse |
|---|---|---|---|---|
| baseline (GPU 0 idle) | 101 ms | 125 ms | 8.0 fps | 0.017158 |
| refiner load on the OTHER GPU | 103 ms | 133 ms | 7.5 fps | 0.017158 |
| refiner load on the SAME GPU | 206 ms | 223 ms | 4.5 fps | 0.017159 |

- The slowdown is **entirely same-GPU SM contention**; cross-GPU is noise.
  The two-GPU split (SLAM on one card, refiner on the other) is the
  deployment answer — at the engineering cost of moving the shared Gaussian
  buffers across cards, since CUDA IPC is same-device.
- ATE is unchanged everywhere: the offline pipeline has no frame-dropping
  mechanism, so contention shows up purely as latency. In a live system the
  4.5 fps same-GPU number means most frames would be dropped — the ATE risk is
  real there even though it is invisible offline.
- On a single GPU the refiner must be duty-cycled (drop optimization steps,
  never frames); no such policy exists yet.
- Reality check for the whole program: baseline is already only ~8 fps on
  desk — 30 fps real-time was never on the table at this map size on one
  A6000.

### 15.3 Throughput calibration, and what the budgets mean

Measured: ~3-4 it/s for the full 1.86M-Gaussian map, solo A6000 (600-iter
timed run, no evals; consistent with the 3000-iter run's wall time). Desk's
sequence occupies ~40 s of wall time inside the SLAM system at 8 fps, so:

- **strict real-time** (refine only while the sequence plays, refiner on the
  second GPU): ~120-140 iters → the 120-iter row: **+1.21 dB**;
- **+3 min polish** after the sequence ends (GPU goes idle): 500 iters →
  **+1.95 dB**;
- **+6/17 min polish**: 1000/3000 iters → **+2.03/+1.97 dB**.

Under same-GPU sharing the refiner would get roughly half throughput
(inferred from the SLAM-side doubling; the refiner-side number was not
instrumented), i.e. strict-real-time shrinks to ~60-70 iters.

### 15.4 What this does to the queue

(f) and (g) are DONE. The remaining queue from §13.14, re-ranked by what is
now known:

1. **(b′) the faithful differential seam test** — now the only open risk to
   P1's entire surviving value proposition ("loop closure is free", stage 3).
   Perturb half the keyframes AND recompute the anchored supervision poses
   (`FramePoseLog`), split held-out three ways. Cheap, same harness family.
2. **(a) process integration** — `run_refiner` still never run. De-risked by
   15.1 (gain survives causality) and 15.2 (put it on the second GPU). Needs
   the map version counter and, for single-GPU users, the duty-cycle policy.
3. **(c) online densification/pruning** — still gated on the de-clustering
   ablation (the overlap-views-score-worse anomaly from §13.14).
4. **(d)(e) hard metrics and buffer ablation** — the recent/reservoir mix
   still has no principled value; the causal harness is the natural place to
   ablate it.

### 15.5 (b′) The faithful differential seam test — PASSES, and it quantifies FramePoseLog's value

Stage 3's per-keyframe test held the supervision cameras frozen, which the
real system never does: `FramePoseLog` anchors every tracked frame to the
keyframe current at its tracking time (`evaluate.py:86`) and re-resolves its
pose through the anchor's *current* pose. §15.5 adds `--perturb-mode block`
to `scripts/refine_local.py`: ONE Sim3 (seeded draw: rot +0.2°, scale 1.060,
|t| 0.013 m — the draw is scale-dominated) applied to the second half of the
trajectory at iter 1500 of 3000, with supervision views carried by their
anchors (FramePoseLog semantics) or frozen (`--freeze-supervision`, the
unfaithful control). Held-out split three ways (only-low / only-high / seam)
plus the median-overlap two-way, recomputed on the perturbed geometry.

**On desk the three-way split degenerates: every keyframe covers >5% of every
held-out view, so all 50 views are "seam"** — consistent with the 86.7%
multi-coverage measurement in §13.14. The full-set numbers ARE the seam
numbers there. Room (51 keyframes, spatially extended) is the sequence where
the split differentiates.

| desk, iter | faithful (moved) | frozen (unfaithful) |
|---|---|---|
| 1000 (pre-perturb) | 14.3525 / 0.4089 | 14.3527 / 0.4106 |
| 1500 (injected) | 13.0755 / 0.5008 (−1.32) | 13.0863 / 0.4982 (−1.31) |
| 2000 | 13.5751 / 0.4314 | **14.5146 / 0.3933** |
| 3000 | 13.5157 / 0.4200 (**residual −0.88**) | **14.4096 / 0.3741 (full recovery)** |

Three findings, in order of what they change:

1. **No seam tearing, in either arm.** Post-injection quality recovers
   monotonically from the dip; the optimizer never adds structural damage on
   top of the displacement — under a 6%-scale block correction, the worst
   case the representation can face. Stage 3's "the map follows a loop
   closure for free" now holds under faithful supervision.
2. **The faithful arm preserves the correction (residual −0.88 dB at 1500
   post-injection iters, plateaued).** Within-block views move with the block
   and hold it in place; only cross-block views pull back (the overlap class
   heals +0.81 dB vs single +0.13 — cross-block supervision is concentrated
   exactly there). Online this is the WANTED behaviour: a real loop closure
   is correct, and refinement must not undo it. Against the artificial error
   injected here the same persistence shows as a residual; the sign flips
   with the sign of the correction.
3. **The frozen arm quantifies what FramePoseLog prevents.** With supervision
   frozen, the optimizer fully undid a 6%-scale block correction in ~500
   iters (14.51 at 2000, back at baseline) — i.e. without anchor-relative
   frame poses, refinement actively fights the pose graph. This is the first
   end-to-end measurement of the co-adaptation mechanism §13.13 was designed
   against: the design works.

Room replication (51 keyframes, 8.77M Gaussians, 2000 iters, perturb at
1000, injected rot −4.6° / scale 0.964 / |t| 0.113 m). The three-way split
differentiates here (4 low-only / 5 high-only / 41 seam), and it replicates
desk on every point — including the instrument's self-check: at the moment of
injection the low-only class is bit-stable (11.8182 → 11.8211), since neither
its clusters nor its cameras moved.

| room, iter 2000 | low-only | high-only | seam | full |
|---|---|---|---|---|
| faithful | 12.03 (unaffected, still climbing) | 8.19 (correction HELD, +0.07) | 11.09 (+0.68 cross-block pull) | 10.75 |
| frozen | 12.03 | 10.75 (pulled back, +2.64) | 12.86 (back at pre-perturb 12.82) | 12.53 |

(b′) verdict: PASSES on both sequences. The representation follows block
corrections without tearing; under faithful (anchor-carried) supervision the
correction persists — the wanted behaviour when the correction is real — and
the frozen control shows refinement would otherwise undo it within hundreds
of iterations. Remaining queue: (a) process integration, (c) densification
lifecycle (gated on the de-clustering ablation), (d)(e) metrics + buffer
ablation.

### 15.6 (e) Supervision sampling ablation — the recent window is pure downside at desk scale

`SupervisionFrames` mixes a bounded recent ring with a reservoir "because new
clusters need recent supervision and old regions must not be forgotten" — the
window length and mix had no measured value. `refine_causal.py --sampling`
now ablates it under the causal protocol at the 500-iter polish budget
(final held-out, plus early/late halves as the forgetting/assimilation
split):

| sampling | overall | early half (forgetting) | late half (assimilation) |
|---|---|---|---|
| uniform over history | **14.3965 / 0.4603** | **14.09 / 0.457** | **14.72 / 0.463** |
| mixed 70/30 | 14.2096 / 0.4626 | 13.95 / 0.466 | 14.49 / 0.459 |
| recent-only, W=16 | 14.0648 / 0.4762 | 13.79 / 0.501 | 14.35 / 0.451 |

- The predicted assimilation upside of a recent window does NOT exist here:
  recent-only loses on the late half too (−0.37 dB). Causal arrival already
  recency-biases the available pool; restricting old views only removes
  supervision.
- The predicted forgetting downside DOES exist and scales with window
  strictness (early half: uniform 14.09 > mixed 13.95 > recent 13.79).
- Design consequence for `SupervisionFrames`: sample reservoir-dominant
  (uniform over everything retained); the recent ring is optional at this
  scale. Caveat: 50 views never evict a 200-slot reservoir, so the window
  question reopens on sequences long enough to force eviction — the ablation
  to rerun then is uniform-with-eviction vs mixed.

### 15.7 De-clustering ablation — the overlap anomaly is two shells at ~10 mm, CONFIRMED

§13.14's anomaly: after optimization, single-coverage held-out views score
HIGHER than overlap views although initialization has the reverse order —
the lpips gap proportionally larger than the psnr gap, pointing at a
structured artifact (two keyframe clusters painting the same surface twice,
slightly offset). `--dedup-voxel` in `refine_local.py` merges shells: voxels
holding Gaussians from ≥2 keyframes keep only the earliest owner's. Desk,
stage 2, 3000 iters, gap = single − overlap psnr at iter 3000:

| arm | deleted | overlap | single | gap |
|---|---|---|---|---|
| control | 0% | 14.0782 / 0.3956 | 14.6953 / 0.3506 | **+0.62** |
| dedup 5 mm | 14.2% | 14.0582 / 0.3948 | 14.6863 / 0.3485 | +0.63 |
| dedup 10 mm | 28.4% | 14.2689 / 0.3878 | 14.4457 / 0.3535 | **+0.18** |

- The gap is untouched at 5 mm and collapses at 10 mm — the shells sit
  ~5-15 mm apart (consistent with the measured cross-keyframe nearest-
  neighbour p10 ≈ 11 mm), and only a voxel spanning both merges them. A
  smooth-convergence explanation cannot produce this dose-response step.
- Cost of deleting the second shell: overlap +0.19 dB, single −0.25 dB,
  overall −0.02 psnr / −0.0025 lpips — roughly neutral in aggregate, a pure
  redistribution toward the multi-covered regions that dominate real views
  (86.7% of pixels, §13.14).
- Consequence for (c) online densification/pruning: a dedup pass at ~10 mm
  over shared regions — as a discrete lifecycle event at re-anchoring time,
  not a continuous filter — is now a measured design, not a guess. (c) is
  unblocked on the discriminator it was gated on.

### 15.8 (a) Stage 4 done: the refiner runs inside the SLAM loop — +1.24 dB online, ATE untouched

The refiner is now a third worker process (`--refiner`, default off), built
from the pieces the experiments validated:

- **Anchor-carried supervision** (the (b′) result): `SupervisionFrames`
  stores CPU-shared uint8 frames with (anchor_idx, T_anchor_frame) poses;
  world poses compose through the anchor's CURRENT pose at sample time, so
  supervision follows loop closures. CPU shared memory, not CUDA IPC — it is
  the only channel that does not pin both processes to one card.
- **Reservoir-dominant sampling** (the (e) result): recent_frac default 0.3.
- **Duty-cycle throttle** (the (g) result): EMA of iteration time, sleep
  (1−duty)/duty × EMA after each step; steps are dropped, never frames.
- **Map save on termination**: composed through the keyframes' final poses,
  written as `<seq>_refined.ply` after the backend drains (so the save sees
  the last corrections).

Desk, eval_calib (deterministic) and rt_calib (multithreaded), duty 0.25:

| check | result | verdict |
|---|---|---|
| ATE (pre-registered failure condition: "ATE moves at all") | 0.016975 / 0.016973 vs baseline 0.0170 | PASS — untouched |
| tracker latency (rt) | track p50 112 ms vs 101; iter mean 189 vs 125; 5.3 fps vs 8.0 | cost is real but bounded; concentrated in the keyframe path |
| **map quality, same run** (eval_map_quality, n=100) | baked 10.6598 / 0.5557 → **refined 11.9015 / 0.5443** | **+1.24 dB / −0.011 lpips** |
| refinement budget consumed | 56 steps (duty 0.25 ≈ 0.7 it/s over the 80 s run) | matches (f)'s strict-real-time prediction (+1.21 dB at ~120 iters) almost exactly |

This is the first number in the document produced INSIDE the live system:
the +1.9 dB offline gain survives causal online operation at roughly the
budget-predicted value, with tracking untouched. The honest ceiling table
from §13.14 now reads:

```
10.66   baked map, live system
11.90   + online refinement, 56 steps, duty 0.25     MEASURED LIVE
~12.6   expected at duty ~1.0 / second GPU (~500 steps, (f) arm)
14.45   offline polish ceiling at est poses          not real-time
~19     offline ceiling at GT poses                  not deliverable
```

Stage 4 is DONE. Stage 5 (pose phases) is NOT started and §13.12d says
frozen/alternating must be the default if it ever is. Open follow-ups:
duty/latency trade-off curve (0.25 costs ~1/3 frame rate for 56 steps —
the second-GPU placement is the real fix, needs a copy channel since CUDA
IPC is same-device); viz display of the refined map (snapshot double-buffer
+ version counter, currently the refined map is only persisted, never drawn);
map-size growth under long sequences (no pruning yet — (c), whose dedup
parameter is now measured: ~10 mm, §15.7).

### 15.9 Stage 5 (online pose phases) — the measured decision is DO NOT BUILD

The staging plan marked pose phases "optional, only after 1-4, defaulting to
map-frozen". With stage 4 now live, the question is due, and three
measurements already answer it:

1. **The real-error recovery ceiling is ~11%.** §13.12d: frozen/alternating
   photometric pose refinement recovers 41-58% of an INJECTED iid
   perturbation but only ~11% of REAL tracking error — the co-adapted map
   (§13.12b: a map built at estimated poses pulls poses further from truth,
   −22.2%) has destroyed the signal.
2. **The error is not in placement.** §13.14's re-anchoring experiment:
   swapping estimated anchors for ground truth is worth +0.07/+0.13 dB. The
   pose error is baked INSIDE each keyframe's cluster at fusion time; no
   post-hoc pose adjustment reaches it.
3. **The downside is architectural, not just numerical.** Pose phases mean
   the refiner writes keyframe poses — a second writer into the pose graph's
   domain, against a backend that currently owns them, risking a
   refiner↔backend feedback loop (each treats the other's state as ground
   truth) for an ~11%-of-a-small-number upside.

So stage 5 is closed as a deliberate negative: online photometric pose
refinement is not a lever at map-build time, and the remaining 4.5 dB pose
gate belongs to the fusion stage (§13.14), not to the refiner. If that
changes, the re-open condition is a fusion-side improvement that restores
the pose signal — at which point the refiner's camera-gradient CUDA work
(§13.11) and the frozen/alternating schedule (§13.12d) are the starting
point, not a redesign.

### 15.10 (c) Online dedup lifecycle — built, with an honest caveat

`--refiner-dedup-voxel` / `--refiner-max-gaussians`: when the map crosses
the size threshold at a keyframe injection, one voxel dedup (earliest-owner
rule, §15.7) runs and Adam moments are subset to survivors. Desk validation
(threshold 1.5M): fired once at kf ~13, map 1,850,482 → 1,339,629 (−27.6%,
matching the offline 28.4%), **ATE unchanged (0.016975)**, refined-map
quality 11.3755 vs 11.9015 without dedup (−0.53 dB).

The −0.53 is NOT the offline §15.7 number (≈0), and the difference is the
recovery budget: §15.7's neutrality was measured with 3000 post-dedup
iterations; here the dedup fired near the sequence end with ~6 steps left,
so the deleted 28% — Gaussians that had already absorbed supervision — was
not re-earned. Operational rule: dedup is a MAP-SIZE control for long
sequences, where it fires periodically mid-run with optimization continuing
after it; it is not a quality feature, and on short sequences at low duty
there is no budget to amortize it.

### 15.11 Second-GPU refiner — +2.15 dB at baseline latency; two vendor bugs fixed en route

`--refiner-gpu 1` (launch with both cards visible): the refiner computes on
the second card; shared buffers stay on the tracking card and cross only as
small per-keyframe copies. Desk, rt_calib, duty 1.0 (UNTHROTTLED):

| check | same-GPU duty 0.25 (§15.8) | second-GPU duty 1.0 | baseline |
|---|---|---|---|
| track p50 | 112 ms | **102 ms** | 101 ms |
| iter mean / fps | 189 ms / 5.3 | **132 ms / 7.6** | 125 ms / 8.0 |
| ATE | 0.01697 | **0.017158** | 0.017158 |
| optimization steps | 56 | **225** | — |
| refined map (n=100) | 11.9015 / 0.5443 | **12.8128 / 0.5411** | baked 10.6598 / 0.5557 |

**+2.15 dB online at +1% tracker latency and bit-identical ATE** — this is
the deployable configuration. The ceiling table:

```
10.66   baked map, live system
11.90   + online refinement, same GPU, duty 0.25 (56 steps)
12.81   + online refinement, second GPU, unthrottled (225 steps)   SHIPS
14.45   offline polish ceiling at est poses
~19     offline ceiling at GT poses                               not deliverable
```

Two vendor bugs found and fixed to make this work, both the same shape —
"works on device 0, silently wrong elsewhere":

1. **lietorch group ops miscompute on cuda:1.** A Sim3 compose returns zero
   translation (no error). Worked around in `run_refiner`: all lietorch pose
   math stays on the shared buffers' own device; only the resulting 4x4
   matrices cross. (Root cause not chased into the lietorch build; the guard
   is cheap and total.)
2. **The vendored rasterizer mixed devices.** `rasterize_points.cu` had no
   device guard: buffers allocate on the current device and kernels launch
   on it, so with both cards visible and inputs on cuda:1 it mixed cuda:0
   buffers with cuda:1 pointers — illegal memory access in forward AND a
   silent backward failure. Fixed with
   `c10::cuda::OptionalCUDAGuard(at::device_of(means3D))` at the three entry
   points (forward / backward / markVisible) and rebuilt; the cuda:1 probe
   (`refine_causal --device cuda:1`) now matches the cuda:0 trajectory. This
   was almost certainly also the source of the historical "illegal memory
   access once enough Gaussians accumulate" crash notes — worth re-testing
   those cases on multi-GPU runs before assuming density is the trigger.

Viz display of the refined map remains the one open follow-up: the
publication side now exists and is verified (`RefinedMapSnapshot`, 13
floats/Gaussian double-buffered on CPU shared memory + version counter; the
run reports `snapshot v2, 2,396,900 gaussians published`), but the consumer
side needs the imgui viewer, which the headless box cannot run — untested
by definition here.

### 15.12 Route D (decoder-only LoRA) — safe for two epochs, then collapses; route B stands

The last open ablation from §14, isolating §3.3's mechanism: LoRA (r=8,
alpha=16) on the decoder blocks ONLY (`dec_blocks{,2}.{attn,cross_attn,mlp}`
Linears), encoder frozen under no_grad (retrieval/matching features
bit-identical by construction), Gaussian heads trained as route B, route B's
exact protocol (`scripts/exp_dec_lora.py`, TUM 6 epochs batch 2 lr 1e-5,
same seeded draw):

| epoch | psnr | lpips | scale_p99 (canary) |
|---|---|---|---|
| BASE | 15.0933 | 0.2793 | 0.0728 |
| 1 (best) | **15.4597 (+0.37)** | 0.2675 | 0.0259 |
| 3 | 14.0916 | 0.3252 | 0.0547 |
| 5 | 13.4834 (−1.61!) | 0.4035 | **0.3181** |

Three conclusions, in order of weight:

1. **§3.3's mechanism is isolated.** The failure is not the encoder
   specifically — adaptation ANYWHERE upstream of the Gaussian head
   destabilizes the frozen downstream stack, differing only in rate:
   encoder-LoRA collapsed immediately (−49%); decoder-LoRA got two good
   epochs and then collapsed with the same signature (scale_p99 12x climb,
   grad-norm max 41.7 vs route B's single digits, val degrading from epoch
   2). Route A was not a bad LoRA target choice; it was the fast version of
   a general failure.
2. **Route B is the endpoint.** Decoder-LoRA's best (+0.37 dB) is worse
   than head-only's (+1.00 dB) at the same budget AND comes with an
   instability tail. There is no capacity argument left for adapting
   anything but the Gaussian head on these datasets.
3. **The retrieval-refit question is closed twice over.** Encoder features
   must stay bit-identical (encoder adaptation is destructive); decoder
   adaptation leaves them untouched but doesn't help. Under every measured
   route, the MASt3R retrieval assets stay.

---

## 16. Paper-ready summary: the online-refinement campaign (2026-08-02/03)

Every number below was measured this session on TUM freiburg1_desk (the
standard cell) unless stated otherwise; replication on freiburg1_room where
noted. Protocols: held-out non-keyframe views at Sim3-mapped GT poses,
SLAM-estimated poses for supervision (the deployable protocol), identical
seeded draws for paired comparisons (§2.1). Artifacts named for
reproduction. Ordered as a narrative: each experiment gates the next.

### 16.1 Claims and their evidence

**C1 — Online Gaussian-map refinement is worth +1.2 to +2.2 dB in the live
system, with ATE untouched.** Deliverable numbers (eval_map_quality, n=100):

```
10.6598 / 0.5557   baked map, live system (baseline)
11.9015 / 0.5443   + online refinement, same GPU, duty 0.25, 56 steps   (+1.24 dB)
12.8128 / 0.5411   + online refinement, second GPU, unthrottled, 225    (+2.15 dB)
                   steps, tracker latency +1%, ATE 0.017158 = baseline
```

ATE is bit-identical in the deterministic config (0.016975 vs baseline
0.0170; §15.8), tracker p50 latency is 102 ms vs baseline 101 ms in the
deployable two-GPU configuration (§15.11). Pre-registered failure
condition ("ATE moves at all") did not fire. Artifacts:
`splatt3r_slam/refiner.py`, `main.py --refiner [--refiner-gpu N]
[--refiner-duty D] [--refiner-dedup-voxel V]`, `logs/refiner_*`.

**C2 — The offline gain survives causality.** Post-hoc numbers were all
measured with whole-sequence supervision; the causal replay (keyframes
injected at their timestamps, supervision unlocked at its timestamps,
iterations ∝ elapsed time, Adam rebuilt per injection) shows the gain
survives at every budget, incl. the strict-real-time one
(`scripts/refine_causal.py`, §15.1):

| budget (iters) | causal | posthoc | note |
|---|---|---|---|
| 120 (strict real-time) | 13.6511 | 13.8002 | +1.21 dB over no refinement |
| 500 (~3 min polish) | 14.3942 | 14.3223 | causal ≥ posthoc from here up |
| 1000 | 14.4747 | 14.3516 | |
| 3000 | 14.4123 | 14.3747 | |

**C3 — A second GPU, not throttling, is the deployment answer.** Refiner
load on the SAME card doubles tracker latency (101→206 ms p50, 8.0→4.5
fps); on the OTHER card it is noise (103 ms, 7.5 fps); ATE unchanged
everywhere (offline pipeline cannot drop frames — the ATE risk exists only
in a live-drop regime). `scripts/exp_gpu_contention.sh`,
`main.py --frame-timing`, `config/rt_calib.yaml`, §15.2. Baseline is ~8
fps at this map size — 30 fps was never the operating point.

**C4 — The trajectory-anchored map follows loop closures without tearing,
and refinement does not undo them.** Faithful differential seam test (one
Sim3 to the second half of the trajectory, supervision views carried by
their anchor keyframes — FramePoseLog semantics): no seam tearing under a
6%-scale block correction on either sequence; the correction persists
(wanted behaviour); the frozen-supervision control shows refinement would
otherwise undo it within ~500 iterations. `refine_local.py
--perturb-mode block [--freeze-supervision]`, §15.5. Room replicates
(three-way held-out split: low-only views bit-stable at injection).

**C5 — Uniform-over-history supervision wins; the recent window is pure
downside at this scale.** Assimilation/forgetting split (early/late
held-out halves): uniform 14.40/14.09/14.72 > mixed-70/30 14.21/13.95/
14.49 > recent-only 14.06/13.79/14.35. Causal arrival already
recency-biases the pool. `SupervisionFrames` default recent_frac now 0.3.
`refine_causal.py --sampling`, §15.6.

**C6 — The overlap-quality anomaly is two shells at ~10 mm.** Dose-response:
voxel dedup at 5 mm leaves the single-vs-overlap gap (+0.63) untouched,
10 mm collapses it (+0.18); cross-keyframe nearest-neighbour p10 ≈ 11 mm.
Deleting the redundant shell redistributes quality toward overlap regions
at ~zero aggregate cost. §15.7. This parameterizes the online
de-clustering lifecycle (`--refiner-dedup-voxel`, §15.10: map −27.6%,
ATE unchanged; caveat — it is a size control with a recovery-budget
requirement, not a quality feature).

**C7 — Adapt anything upstream of the Gaussian head and it collapses;
head-only is the endpoint.** Route A (encoder LoRA): −49% immediately,
scale explosion. Route D (decoder-only LoRA, this session): best +0.37 dB
at epoch 1, then collapse (−1.61 dB at epoch 5, scale_p99 12x climb) —
the mechanism is not the encoder, it is ANY upstream adaptation
destabilizing the frozen downstream stack at different rates. Route B
(head-only): +1.00 dB at 6 epochs, +1.78 dB (TUM) / +2.18 (7-scenes) /
+3.69 (EuRoC) / +3.15 (ETH3D, pre-divergence best) at 40. §15.12, §5, §8.1.
Corollary: the retrieval assets stay MASt3R's — encoder features must be
bit-identical, and no adaptation route that helps touches them (§15.12).

**C8 — Exposure normalization is closed at both ends.** Training side:
`SequenceExposureLock` (per-sequence locked gain, first frame → 0.5) was
wired in 2026-07-28, BEFORE all four 40-epoch production heads (verified
by mtimes + live probe of the lock). Deployment side: per-frame
`normalize_exposure` to the first frame. The combination is measured
compatible (§9: +0.90 dB map quality through the deployment path, ATE
bit-identical). §14.

**C9 — Colour harmonization by per-keyframe gain fit (Plan 2) is NEGATIVE.**
Causal voxel-overlap gain fit on the raw map: −0.57 dB psnr, +0.013 lpips.
The gains it fits are mostly not exposure (strong, monotone — head bias,
view-direction shading, 10 mm pairing error). Plan 1 + the refiner cover
the need. `scripts/color_harmonize.py`, splatt3r-color-consistency skill.

### 16.2 The honest ceiling, final form

```
10.66   baked map, live system
11.90   + online refinement, same GPU, duty 0.25 (56 steps)
12.81   + online refinement, second GPU (225 steps)          <-- deployable
14.45   offline polish ceiling at estimated poses
~19     offline ceiling at GT poses                          not deliverable
```

The 12.81→14.45 gap is polish budget, not method. The 14.45→19 gap is pose
error baked into clusters at fusion time (re-anchoring to GT is worth only
+0.13 dB, §13.14) — closed to every post-hoc method tested; the remaining
lever is the fusion stage itself, and stage 5 (online photometric pose
refinement) is deliberately NOT built on that evidence (§15.9).

### 16.3 Measurement traps armed this session (for the methods section)

1. A fixed index list is not a fixed sample (§2.1) — seeded draws or
   nothing.
2. `dataset.subsample()` splits the index space; associate through
   timestamps, never raw frame_ids (§13.14, bug #5).
3. lietorch group ops silently miscompute on cuda:1; vendored rasterizer
   had no device guard (mixed-device buffers → illegal memory access).
   Both fixed/worked around (§15.11) — and the historical "illegal memory
   access at high Gaussian counts" crash notes are likely the same class.
4. An experiment that does nothing looks exactly like a strong negative
   result; measure the thing being manipulated, cross-check with an
   independent instrument (§13.14).
5. Sim3 scale is folded into keyframe rotations (0.685-1.039 on desk) —
   overwriting with scale-free rotations resizes clusters by up to 46%
   (§13.14).

### 16.4 What is NOT claimed

- **A gain on all nine sequences.** The nine-sequence table reports psnr
  deltas of +0.10 to +1.80, but on **room (+0.30, lpips 0.5418 → 0.6019,
  +11.1%) and 360 (+0.14, lpips 0.4841 → 0.5838, +20.6%) the perceptual
  metric moves the wrong way by far more than psnr moves the right way.**
  This project has argued throughout (§10.11, §13.10) that lpips is the
  better perceptual proxy, so by its own standard those two are **negative**.
  The honest headline is **7/9 positive, 2/9 negative**, not "delivered on
  all four families". Both losing sequences are the large maps (7.3M+
  Gaussians), which is the supervision-starvation signature of §13.12a —
  see §16.5 for the arithmetic.
- Generalization beyond TUM desk/room for the online numbers (the offline
  family-level head gains are separately measured, §8.1; causal/seam work
  is two sequences, one seed each).
- Real-time 30 fps: the deployable point is ~7.6 fps at 1.86M Gaussians on
  an A6000 pair.
- Viz display of the refined map: publication verified, consumer
  implemented but never drawn (headless box).
- Long-sequence behaviour of the dedup lifecycle (desk fires it once, at
  the tail, with no recovery budget).

### 16.5 The supervision budget is the binding constraint on large maps

§13.10 measured a saturation rule from two independent scenes: supervision
saturates at roughly **40k Gaussians per view** (desk's knee at ~45 views for
1.86M; room's at ~200 for 7.35M). Applying it to the online runs explains the
2/9 losses without any new hypothesis:

| sequence | map | views the rule asks for | online gain |
|---|---|---|---|
| desk | 1.86M | ~46 | **+1.33 dB** |
| room | 7.35M | ~184 | +0.30, lpips worse |
| 360 | 7.27M | ~182 | +0.14, lpips worse |

desk is the only one whose supervision budget is anywhere near what its map
needs, and it is the only large gain among the TUM sequences.

There are two ways to satisfy the rule, and only one has been tried:

1. **Raise the views** — more supervision per unit time. `--refiner-polish-secs`
   (added at the end of the session) does this by continuing to optimize after
   the sequence ends; untested on a large map.
2. **Shrink the map** — 86.7% of pixels in a median held-out view are covered
   by ≥2 keyframes, and the dedup ablation located the redundancy precisely: a
   10 mm voxel merge removes 27.6% of desk's Gaussians and collapses the
   overlap/single-coverage anomaly from +0.63 to +0.18 dB. **Untested as a
   quality lever** — the one dedup run fired at the tail with ~6 steps of
   recovery budget and cost −0.53 dB, which measures the schedule, not the
   idea. Fired early, it should raise per-Gaussian update count for the same
   wall-clock budget, which is exactly what the rule says the large maps lack.

Neither has been measured. Both are cheap. On 360 the second is the larger
lever by construction: 7.27M Gaussians against ~50 effective supervision views
is 145k Gaussians/view, 3.6x past the measured saturation point.

### 16.6 The large-map loss is budget, and dedup contributes nothing (controlled)

§16.5 proposed two ways to satisfy the 40k-Gaussians-per-view rule on the large
maps: raise the supervision, or shrink the map. Run as a 2x2 on 360 (offline,
`refine_local.py --stage 2`, n_train 50 — the same effective supervision the
online arm had):

```
                    init                3000 iterations
no dedup      11.7902 / 0.4876    ->    13.5105 / 0.4409
10 mm dedup   11.7870 / 0.4926    ->    13.5021 / 0.4407
```

**The two curves coincide at every checkpoint** (3000-iteration difference
0.008 dB / 0.0002 lpips). Dedup deletes 584,665 Gaussians (8.0%) and buys
nothing.

So the entire recovery — **+1.71 dB psnr and −10.4% lpips, on the sequence
whose online arm was a net loss** — is the offline iteration budget alone.

Two consequences:

1. **The 2/9 online losses are supervision starvation, confirmed.** The same
   map, the same 50 views, the same optimizer: given enough iterations the
   perceptual metric moves the *right* way by 10%, against +20.6% the wrong way
   online. The lever is `--refiner-polish-secs` (continue optimizing after the
   sequence ends), not a different method and not colour harmonization.
2. **"Shrink the map" is dead, at least on 360.** The prediction that dedup
   would help came from desk's 27.6% removal rate; 360 removes only 8.0%
   because it is a rotation sequence whose keyframes overlap far less than
   desk's back-and-forth sweep. Extrapolating one scene's redundancy structure
   to another was wrong, and even the 8% that is removed changes nothing.

Note what this does **not** say: dedup may still be worth keeping as a
map-*size* control on long sequences (its original purpose), and the desk
overlap anomaly it was built to probe is a separate question. It is only
falsified as a *quality* lever.

#### The full 2x2, and the budget gap it exposes

```
360, n_train 50            3000 iterations      12000 iterations
no dedup                   13.5105 / 0.4409     13.9951 / 0.4322
10 mm dedup                13.5021 / 0.4407     13.9978 / 0.4310
```

Dedup buys nothing at *either* budget (0.008 / 0.003 dB). Iterations keep
paying and are **not saturated at 12000** — 10000 to 12000 still adds 0.08 dB.

Against the online arm on the same sequence (baked 11.9973 / 0.4841):

```
online refiner       +0.14 dB   lpips +20.6%   <- net loss
offline 3000 iters   +1.51 dB   lpips  -8.9%   <- net gain
offline 12000 iters  +2.00 dB   lpips -10.7%   <- net gain
```

So the large-map perceptual regression is **budget, definitively**: the same
map and the same 50 views move psnr and lpips the same direction once the
optimizer is given enough steps.

**But the gap is two orders of magnitude, not a knob's worth.** Throughput
measured directly (300 iterations, same harness, one A6000):

```
desk  1.86M Gaussians   300 iters / 122 s   2.46 it/s
360   7.27M Gaussians   300 iters / 437 s   0.69 it/s
```

3.9x the map, 3.6x the time — per-step cost is essentially linear in Gaussian
count, as a full-map render per step implies. 360's sequence lasts ~125 s, so
the online refiner gets about **86 steps**:

```
online actually gets      ~86 steps
first turns positive     3000 steps     35x the sequence duration
best measured           12000 steps    140x
```

`--refiner-polish-secs` was sized in seconds. At 0.69 it/s, 3000 steps is
**73 minutes** of polish and 12000 is **4.9 hours**. That is not "polish briefly
after the sequence" — it is offline post-processing under another name.

**Boundary this forces into the write-up:** online refinement's benefit decays
with map size, and at the ~7M-Gaussian scale the online budget is structurally
insufficient to turn it positive. Small maps (desk 1.86M, MH_01, plant_1) win
online by +1.3 to +1.8 dB and need no change. Large maps either accept an
offline post-processing pass — labelled as such — or need real throughput work
(the per-step cost is rendering 7.27M Gaussians; that is visibility culling or
LOD, not a parameter).

#### Figure pipeline (and why GUI captures are not it)

The stippled "salt and pepper" in the GUI captures is **not** a map property.
Same map, same Gaussians, rendered at the capture resolution (512x384) instead
of the viewport's 1960x1061: no stippling at all, surfaces continuous. The
earlier attribution to `inflate_scales_for_stride` was wrong — that branch is
`if s > 1`, and these runs used stride 1, so it never executed.

The real mechanism is that per-pixel Gaussians have a density fixed at
prediction time, which cannot support arbitrarily close or high-resolution
viewing: the 3D gaps always existed and the viewport merely resolves them.
Getting closer does the same thing (the worst GUI frames are the ones nearest a
wall). A display-side `--gs-scale-inflate` (covariance x inflate^2 at render
time only) closes them for demos without touching the map or any metric.

For figures, render at native resolution from the offline path:

```
main.py ... --dump-keyframe-gaussians --save-as frames_head
refine_local.py --stage 2 --iters 3000 --save-renders logs/figs/<seq> ...
```

Verified on 360: holes closed, wall camera and shelf legible, poster layout
recoverable; blurry against ground truth but structurally correct and complete.

#### How much budget, exactly — and why 360 starved while desk did not

Extending the 2x2 to 12000 iterations separates the two metrics, which do not
saturate together:

```
iterations      psnr      lpips
     3000      13.51     0.4409
     6000      13.69     0.4364
    12000      14.00     0.4322       (dedup arm identical: 13.9978 / 0.4310)
```

**lpips is bought by the first ~3000-6000 iterations and then flattens; psnr
keeps climbing.** Since it is lpips that turned 360 into a net loss online, the
polish budget only has to reach the knee — roughly 3000 iterations — not the
psnr asymptote.

The online refiner's actual step counts explain the 2/9 losses exactly, and the
explanation is sharper than "the large maps got fewer steps":

| sequence | Gaussians | online steps | Gaussians per step | online Δpsnr | lpips |
|---|---|---|---|---|---|
| desk | 1.86M | 200 | **9,300** | +1.33 | improved |
| 360 | 7.27M | 176 | **41,307** | +0.14 | worse |

**176 and 200 are nearly the same absolute step count.** What differs by 4.4x is
the optimization *per Gaussian*. desk's map is small enough that 200 steps is a
real budget; 360's is not, and the loss follows.

Concrete sizing for `--refiner-polish-secs`, from measurement rather than taste:

- to match desk's per-Gaussian density (9,300 Gaussians/step), 360 needs
  **~780 steps** — 4.4x its current 176;
- to reach the offline lpips knee (3000 steps at 2,423 Gaussians/step) it needs
  **~2,800 additional steps**.

The first target is the cheap one and is where the sign of the result flips;
the second buys the remaining psnr. Neither has been run online yet — the
offline arms above are the evidence that the budget is the binding variable,
not a demonstration that polish delivers it. Throughput at 7.27M Gaussians is
also lower than the ~3-4 it/s calibrated at 1.86M, so the wall-clock cost of
780 steps has to be measured before quoting a number of seconds.

### 16.7 Where the per-step cost actually goes, and what buys steps back

§16.6 left "throughput work — visibility culling or LOD" as a hand-wave. Profiled
instead (360, 7,267,700 Gaussians, 512x384, 10-step mean, CUDA-synchronized):

```
world() composition   242.3 ms   17.6%
forward render        143.1 ms   10.4%
backward              977.2 ms   71.1%
Adam step              11.4 ms    0.8%
total                1374.0 ms   -> 0.73 it/s

Gaussians receiving gradient from one view: 960,524 / 7,267,700 = 13.2%
```

Two things this kills and one it locates:

- **Sparse Adam is not worth doing.** The optimizer is 0.8% of the step. The
  intuition that updating 7.27M x 14 parameters every step must dominate is
  simply wrong.
- **The rasterizer is not the bottleneck.** Forward render is 10.4%.
- **The bottleneck is our own composition layer.** Backward is 6.8x forward,
  where 2-3x is normal for a rasterizer. The excess is the backward through
  `world()` — the `einsum`, `build_covariance`, and the `A @ cov @ A^T`
  sandwich, all differentiated over 7.27M primitives. `world()` forward plus
  its backward is roughly **54% of the step**.

#### Measured: culling before `world()`, not inside the rasterizer

The rasterizer already frustum-culls internally; that is not where the waste is.
The waste is composing 7.27M Gaussians when 13% of them can affect the image.
Gathering the visible subset *before* `world()`:

```
full        7,267,700 gaussians   1410.9 ms/step   0.71 it/s
visible sub 1,260,287 gaussians    255.1 ms/step   3.92 it/s    5.5x
```

Per-view visible fraction ranges 9.5-17.3%, involving **16-26 of 46 keyframes**
— so a keyframe-level frustum test (46 tests, not 7.27M) captures most of it
essentially for free, and `kf_id` is already carried on every Gaussian.

**IoU between two different views' masks: 0.000.** Completely disjoint on a
rotation sequence. That cuts both ways: per-view mask caching is perfect (each
supervision view has its own stable set, and the map changes slowly), but there
is **no shared hot subset** to keep resident — every step swaps a different
tranche of parameters.

#### The four levers, with sizes

| lever | size | status |
|---|---|---|
| 1. per-view visibility culling before `world()` | **5.5x measured** | not built; `kf_id` and the backend's covisibility graph already exist |
| 2. lower injection density (top-K by confidence per keyframe, instead of one Gaussian per pixel) | 2-4x by construction — 46 x 512x384 = 9.0M raw, 7.27M after filtering, against 1-2M for a comparable room in standard 3DGS | not measured for quality; the dedup null result (8% removed, zero quality change) is indirect evidence of slack |
| 3. group `A @ cov @ A^T` by keyframe | unmeasured; A takes only 46 distinct values, so 46 contiguous block matmuls replace 7.27M batched 3x3 sandwiches | independent of 1, multiplies with it |
| 4. sparse Adam | **0.8% — do not build** | falsified above |

#### What this does and does not achieve

```
today                 0.71 it/s   86 steps in-sequence    3000 steps = 73 min
+ culling (5.5x)      3.9  it/s   480 steps               3000 steps = 13 min
+ half the density   ~7    it/s   875 steps               3000 steps =  7 min
```

Even with both, in-sequence is ~875 steps against the 3000 that turn 360
positive — still under a third. What changes is `--refiner-polish-secs`: from
73 minutes (offline post-processing wearing a different name) to ~7 minutes (a
defensible post-sequence step). That is the qualitative difference this work
buys; it does not make large maps converge inside the sequence.

Closing that last 3-4x cannot come from making each step faster. It has to come
from needing fewer steps — better initialization, or more supervision per step
(multiple views per iteration) — which is a different investigation.

**Order to build in, and the check for the first one:** lever 1 first. It is
measured, it is the simplest (the mask is derivable from data already carried),
and it changes no numerical semantics — the Gaussians it drops have exactly
zero gradient. So the acceptance test is strict: **the same iteration count must
produce a bit-identical result, only faster.** Anything else means the mask is
wrong.

### 16.8 Lever 1 built: 2.1x, not 5.5x, and why the difference matters

`LocalGaussianMap.visible_keyframes/visible_subset` — per-keyframe clusters split
into `tiles^2` blocks along the source raster, each bounded by a world-space AABB,
kept if any corner falls in the widened frustum. 46 x tiles^2 box tests instead of
7.27M projections.

```
tiles=4   keeps 45.4%   false negatives 0   mask cost 703 ms
tiles=8   keeps 44.1%   false negatives 0   mask cost 481 ms
(actual gradient support: 12.5%)

full        1383.7 ms/step   0.72 it/s
culled       653.5 ms/step   1.53 it/s      2.12x
```

**Correct, and by the right test.** Zero false negatives against the measured
gradient support, and the numerical difference culling introduces (3.93e-04 on
means after 8 steps) is *smaller than the same configuration re-run against
itself* (5.48e-04).

That second number matters more than the speedup. **The acceptance criterion
stated in §16.7 — "bit-identical, only faster" — is unachievable by
construction and was wrong to write down.** `renderCUDA`'s backward accumulates
with `atomicAdd`, and Adam's `eps=1e-15` turns any difference in a near-zero
gradient into a full `lr`-sized step (the ratio `exp_avg / (sqrt(exp_avg_sq) +
1e-15)` is O(1) regardless of magnitude). The correct criterion is **"within the
system's own run-to-run noise"**, which is what was checked.

**Why 2.1x and not the 5.5x of §16.7.** That 5.5x was measured by gathering the
*actual gradient support* — which includes occlusion. An axis-aligned box test
cannot: on a rotation sequence each keyframe's cluster is deep along the view
ray, so its box sweeps in Gaussians that are geometrically in frustum but
hidden behind nearer surfaces. Refining the blocks does not help (4 -> 8 tiles
moves 45.4% -> 44.1%); the limit is the box, not its granularity. **5.5x is the
ceiling for any occlusion-aware scheme; 2.1x is what pure geometry buys.**

Revised arithmetic:

```
today                  0.72 it/s    86 steps in-sequence   3000 steps = 73 min
+ block culling (2.1x) 1.53 it/s   183 steps               3000 steps = 33 min
```

The remaining 2.6x needs occlusion, which means **per-view mask caching**: run
one full step per supervision view, record `radii > 0` (the rasterizer computes
it anyway), reuse. The pool is ~50 fixed views and the per-view masks were
measured to be *completely disjoint* (IoU 0.000), so each view's set is stable
and independent — caching is exact and there is no shared set to thrash. Cost is
a ~50-step warm-up and a refresh policy after pose corrections.

### 16.9 Lever 2 built: the map is too dense for its own budget — 1.65x, and better

Density knob is `min_confidence` on injection (`prepare_gaussians_local` already
filters on it; `refine_local.py --min-confidence` exposes it). 360, 3000
iterations per arm, each arm timed:

```
conf   gaussians    it/s     psnr@3000   wall
1.5    7,267,700    0.704    13.5045     4261 s   (default)
2.5    6,371,960    0.787    13.4916     3811 s
4.0    4,531,321    1.162    13.5945     2581 s
8.0    1,719,650    —        crashed     — (only 30 of 46 keyframes survive
                                            the filter; that is the limit)
```

Compared at **matched wall clock** — the only fair comparison, since a sparser
map is worse per step and gets more steps per second — at T = 2581 s:

```
conf   steps in T   psnr@T    lpips@T
1.5      1817       13.4314   0.4530
2.5      2032       13.4091   0.4473
4.0      3000       13.5945   0.4447   <- wins BOTH metrics
```

**`min_confidence=4.0` is strictly better**: 62% of the Gaussians, 1.65x the
throughput, +0.163 dB and −0.0083 lpips against the default at equal seconds.
The default map is carrying 2.7M Gaussians that cost time and contribute
nothing the supervision budget can exploit.

This is the same finding as §13.10's 40k-Gaussians-per-view rule seen from the
other side: rather than raising the views to match the map, lower the map to
match the views. Note it is the *density*, not the redundancy, that mattered —
voxel dedup removed 8% and changed nothing (§16.6), while confidence-ranked
thinning removes 38% and improves the result.

Levers 1 and 2 are independent (culling acts per view, density acts on the map),
so they should compose to roughly **3.5x**:

```
today                      0.72 it/s     86 steps in-sequence   3000 steps = 73 min
+ block culling (2.1x)     1.53 it/s    183 steps                           33 min
+ conf 4.0    (1.65x)     ~2.5  it/s    ~300 steps                          20 min
```

Still not in-sequence convergence for a 7M map, but 73 minutes of "polish" has
become 20, and the quality at any fixed budget is higher, not merely faster.

### 16.10 Lever 3 falsified: grouping the composition by keyframe is slower

`A` in `A @ cov @ A^T` takes only K distinct values (one per keyframe), and
gathering it per Gaussian materializes a (M,3,3) tensor — 262 MB at 7.27M. The
proposal in §16.7 was to exploit that: Gaussians are appended keyframe by
keyframe, so `kf_id` is sorted into 46 contiguous runs, and one (3,3) can be
broadcast against each run instead.

Implemented, verified numerically identical (means 4.8e-07, cov 1.9e-09 — fp
noise), and **measured slower**:

```
per-Gaussian gather   1383.7 ms/step
grouped by run        1427.7 ms/step
```

The hypothesis was wrong. The cost is the matmul itself — M x 27 FLOPs either
way — not the gather. Replacing two large kernel launches with 46 small ones
plus a final `torch.cat` loses more to launch overhead and the copy than the
avoided materialization saves. Reverted; the reasoning is left as a comment at
the call site so it is not re-attempted.

Levers, final status:

| lever | claim | outcome |
|---|---|---|
| 1. per-view visibility culling | 5.5x hoped | **2.1x built** — geometry cannot see occlusion (§16.8) |
| 2. lower injection density | 2-4x hoped | **1.65x built, and better quality at matched wall clock** (§16.9) |
| 3. group composition by keyframe | unmeasured | **falsified — slower** |
| 4. sparse Adam | — | **falsified before building — 0.8% of the step** (§16.7) |

Two of four survive, for ~3.5x combined. The remaining headroom is the 2.6x
between geometric culling and true visibility, which needs per-view mask
caching (record `radii > 0`, reuse; masks are disjoint across views, IoU 0.000,
so caching is exact and thrash-free).

### 16.11 Throughput work: handover

Consolidated so the next person does not re-derive it. Detail in §16.7-16.10.

#### Set this today

**`--min-confidence 4.0` for any map above ~4M Gaussians.** It is the only one
of the four levers that improves speed *and* quality, and it carries no risk:
1.65x throughput, +0.163 dB, −0.0083 lpips at matched wall clock on 360, using
62% of the Gaussians. The default 1.5 leaves 2.7M Gaussians in the map that
cost time and that the supervision budget cannot exploit.

The general statement is §13.10's 40k-Gaussians-per-view rule read backwards:
**lower the map to match the views** rather than raise the views to match the
map. And note *which* reduction works — voxel dedup removed 8% and changed
nothing (§16.6), confidence thinning removed 38% and improved the result. The
problem is density, not redundancy.

#### The throughput ladder as built

```
today                    0.72 it/s     86 steps in-sequence   3000 steps = 73 min
+ block culling (2.1x)   1.53 it/s    183 steps                           33 min
+ conf 4.0    (1.65x)   ~2.5  it/s   ~300 steps                           20 min
+ mask caching (2.6x)   ~6.5  it/s   ~780 steps                            8 min
```

The first three are built and measured; the fourth is designed and not built.

#### Next, and its evidence

**Per-view mask caching** is the only remaining throughput item worth doing.
Geometric culling stops at 2.1x because an axis-aligned box cannot represent
occlusion, and 5.5x is the measured ceiling for anything that can (§16.8).
Record `radii > 0` per supervision view — the rasterizer computes it anyway —
and reuse. The design is safe for two measured reasons: the per-view masks are
**completely disjoint** (IoU 0.000 on a rotation sequence), so caching is exact
and there is nothing to thrash; and the pool is ~50 fixed views, so warm-up is
~50 full steps. It needs a refresh policy after pose corrections.

#### Do not re-attempt

- **Sparse Adam.** 0.8% of a step. The intuition that updating 7.27M x 14
  parameters must dominate is simply wrong (§16.7).
- **Grouping `A @ cov @ A^T` by keyframe.** Numerically identical, measurably
  slower — the matmul is the cost, not the gather (§16.10). A comment at the
  call site records this.

#### The limit none of this reaches

Even with all four, a 7M-Gaussian map gets ~780 in-sequence steps against the
3000 that turn 360 net-positive. **The remaining 3-4x cannot come from making
each step faster** — it has to come from needing fewer steps: better
initialization, or more supervision per step (several views per iteration).
That is a different investigation and nothing here bears on it.

Small maps are unaffected by all of the above: desk (1.86M) already converges
in-sequence and wins by +1.3 to +1.8 dB online.

#### Method note, for the write-up

Four hypotheses were stated with predicted sizes before building. Two survived
(2.1x against 5.5x predicted; 1.65x within the 2-4x predicted), and **two were
falsified by measurement, one of them before any code was written**. The two
that died were both mine, and both died to a measurement rather than to an
argument — the profile in §16.7 killed sparse Adam, and a benchmark killed the
grouped composition. Stating the predicted size up front is what made the
falsifications legible.
## 17. Image quality: the dot lattice, and what the optimizer does to it (2026-08-08)

The trigger was visual, not metric. Every GUI capture of the online 360 map
(`logs/gs_view_360/gs_map_*.png`) is covered in a regular halftone dot pattern,
worst on flat surfaces, with moire where it beats against the output pixel grid;
frame 48 also shows straight-edged seams with a brightness step across them, and
frame 700 (a close-up late in the sequence) is dominated by the dots to the point
where geometry is barely readable. None of it moves `psnr`/`lpips`.

### 17.1 Lever 4 replaced, not built: per-Gaussian culling is 6.51 it/s

§16.11 queued per-view mask caching for the remaining 2.6x. Two measurements
were taken before writing it, and both went against the design.

**The signal `radii > 0` is nearly the whole story.** §16.11 argued it was
unusable because it is blind to occlusion. Measured on 360 against the true
gradient support (the Gaussians with nonzero `f_dc.grad` after a backward):

```
view   block AABB    radii>0   grad support
   0      48.3%        9.7%        8.6%
   6      49.7%       11.9%       11.2%
  18      28.5%        7.1%        6.9%
  36      29.8%        4.8%        4.5%
```

Occlusion is worth ~12%, not the 2.6x §16.11 attributed to it. **The 45.4% the
block test keeps was almost entirely frustum slack, not occlusion.**

**And the reason the block test was chosen was itself wrong.** §16.8 justified
block AABBs as "46 x tiles^2 box tests instead of 7.27M projections". Projecting
7.27M points is one gathered 3x3 matvec — 65 MFLOP, and the Gaussians are stored
contiguously per keyframe so it is 46 small matmuls with no gather at all. The
composition it guards is two 3x3 matmuls per Gaussian *plus a backward*. The
test was never the expensive part; the block scheme's own Python loop over
46 x 16 boxes cost more than the thing it replaced.

`LocalGaussianMap.visible_exact` — per-Gaussian frustum test with each
Gaussian's own 3-sigma footprint, the projection Jacobian's off-axis factor
`sqrt(1 + (x/z)^2)`, and a 6-pixel pad:

```
             kept        mask cost    throughput
none         100%           —          0.72 it/s
block      28-50%        137 ms        1.41 it/s
exact     4.9-12.6%       12.5 ms      6.51 it/s     false negatives 0
```

**9.0x over no culling, 4.6x over the block test, and exact** — zero Gaussians
carrying gradient are dropped, on the same acceptance test §16.8 used. It is
also stateless: no warm-up, no staleness, no refresh policy after a pose
correction, none of which the cache could have avoided.

The pad matters and is not decoration: without it, 34-1042 Gaussians per view
(0.05-0.2% of the support) fell outside the test. The rasterizer adds 0.3 to the
2D covariance diagonal before taking its own radius, worth ~1.6 px. Six pixels
costs +0.4 pp of kept fraction and takes the false negatives to exactly zero.

Revised ladder, all three built and measured:

```
today                      0.72 it/s     86 steps in-sequence   3000 steps = 73 min
+ conf 4.0    (1.65x)     ~1.2  it/s    ~140 steps                          42 min
+ exact culling (9.0x)    ~6.5+ it/s    ~780 steps                       <  8 min
```

`ViewMaskCache` remains in `refiner.py`, unused, with the measurement that
retired it in its docstring. **Two designs in a row (grouped composition,
mask caching) died to a measurement taken before the code was written.** Both
times the premise was an unexamined guess about which operation was expensive.

### 17.2 The dots are a coverage defect in the map, and they are measurable

Splatt3R emits one Gaussian per source pixel and sizes it to roughly its own
pixel footprint *at the source view*. The rasterizer point-samples each Gaussian
at the pixel centre — there is no area integration — so with lattice pitch `dx`
and effective std `sigma`, a flat region's first lattice harmonic has relative
amplitude `exp(-2 pi^2 sigma^2 / dx^2)`. At the source view and resolution the
sample points sit on the lattice centres and it is invisible. Magnify, shift, or
resample and it enters the output Nyquist band: dots. At a non-integer ratio it
beats against the pixel grid: moire.

The governing quantity `sigma/dx` is a property of the **map** and does not
depend on the view; only its visibility depends on magnification. (An earlier
version of this argument had it that a closer view covers fewer output pixels
per Gaussian, which is backwards. Kimi caught it. The frequency argument is the
right one and it is what predicts the sweep.)

**E1, the decisive test, needs no ground truth**: render each held-out view at
1x and at 2x, average-pool the 2x back to native, and compare. If the map has no
structure between the native sample points the two agree. `scripts/diag_lattice.py`,
360, 25 held-out views, raw baked map, no optimization:

```
  tau    psnr_1x  psnr_ss   self    lpips   hp_flat   alpha   hp_alpha
 0.00    12.7910  12.8118   27.94   0.4854  0.01042   0.9760  0.03514
 0.30    12.7827  12.8661   33.51   0.4880  0.00360   0.9839  0.00902
 0.50    12.7820  12.8322   39.14   0.4936  0.00177   0.9889  0.00258
 0.70    12.7904  12.8075   43.72   0.5018  0.00151   0.9911  0.00169
 1.00    12.7909  12.7911   47.57   0.5167  0.00126   0.9918  0.00127
```

- **self = 27.94 dB at tau=0.** The map really does have structure between its
  own sample points. It is a representation defect, not a viewer artifact.
- **psnr_1x is flat to +-0.01 dB across the entire sweep.** The metric this
  project has optimized all along is *provably blind* to the most visible defect
  in the output. That disconnect is the finding, not a caveat to it.
- **`hp_alpha` — the lattice amplitude in the accumulated ALPHA channel — falls
  13.6x** (0.03514 -> 0.00258 at tau=0.5). Alpha carries no colour, so this
  settles the competing hypothesis: the dots are **coverage**, not low opacity.
  Opacity is one scalar per Gaussian, uniform over its own footprint; producing
  a *pattern* needs spatial modulation of alpha, which only coverage supplies.
  Mean alpha is already 0.976 at tau=0, so there was no opacity deficit to
  begin with. `logs/lattice_360/crop_tau0_alpha.png` vs `crop_tau0.5_alpha.png`
  shows it directly — a textbook halftone, and then nothing.
- **lpips is the price**: +1.7% at tau=0.5, +6.4% at tau=1. Monotone. This is
  the blur the filter buys the coverage with.

The fix is Mip-Splatting's 3D smoothing filter with the band limit read off the
lattice instead of estimated from the cameras: the Gaussians are born on a pixel
grid and nothing densifies, so the pitch is *directly measurable* as the distance
to the nearest 4-neighbour on the source raster. **Min over the neighbours, not
mean or median**: across a depth discontinuity two of the four neighbours are
metres away, and any central statistic would inflate every silhouette Gaussian
into a blob spanning the gap — manufacturing the trailing-streak artifact on
purpose. Adding `sigma^2 I` is exact in this parameterization since
`R diag(s^2) R^T + sigma^2 I = R (diag(s^2) + sigma^2 I) R^T`.

Predicted vs measured, for the record. The perceptual criterion `exp(-2 pi^2
tau^2)` predicts residual amplitudes of 11% / 0.7% / 0.02% at tau = 0.3 / 0.5 /
0.7. Measured (floor-subtracted) 22.9% / 3.9% / 1.2%. **Right shape, ~5x slower
decay** — expected, since `dx` varies by an order of magnitude across a 0.5-5 m
depth range and one global `tau` cannot put every Gaussian at the same
`sigma/dx`.

One prediction of E1 was falsified, and it was made by both reviewer and author:
the 2x-pooled render was expected to score 1-3 dB *worse* against ground truth,
because pooling sees the gaps. It scores slightly *better* (12.8118 vs 12.7910)
— pooling also removes noise, and that gain dominates. **The whole verdict rests
on the self-consistency column, which is the one number in the table that
involves no ground truth at all.**

### 17.3 The optimizer destroys coverage, and it is the position updates

The filter fixes the map as baked. It does not survive optimization. 360,
tau=0.5 held as a hard constraint (per-Gaussian `scale_floor` buffer, added in
quadrature at every forward, not reachable by Adam), **30 steps**:

```
              self-psnr   hp_alpha
init            39.76      0.00163
after 30        36.58      0.00692      -3.2 dB, 4.2x worse
```

Without the floor: 28.25 -> 24.97. **Same -3.3 dB slope.** The floor raises the
starting point and does nothing about the rate.

Supersampled supervision was tried first, on the theory that a point-sampled
loss cannot see the gaps it is opening: render the training view at 2x and
average-pool before the loss (`--ss-loss 2`), which is area integration by Monte
Carlo and needs no CUDA change. **No effect at all** — 36.60 / 0.00698 against
36.58 / 0.00692.

So the five parameter groups were ablated one at a time, 100 steps each, lr set
to zero for exactly one group:

```
arm            psnr@100   lpips@100   self-psnr   hp_alpha
all            12.2754     0.5481       35.46      0.01079
no-opacity     12.2608     0.5530       35.53      0.01059
no-scale       12.2557     0.5493       35.42      0.01091
no-rot         12.2626     0.5487       35.46      0.01080
no-f_dc        12.2735     0.5575       35.10      0.01139
no-means       12.1604     0.4909       37.57      0.00198   <--
(init)                     ~0.507       39.76      0.00163
```

**It is the means, alone.** Freezing positions keeps `hp_alpha` at 0.00198
against an init of 0.00163, while every other arm lands at ~0.0107 — a 5.4x
difference, with the other four indistinguishable from each other.

The mechanism is arithmetic. `lr_means = 1.6e-4 * extent = 4.8e-4 m/step` on a
2.98 m scene, while the lattice pitch at 2 m through an f~500 camera is ~4 mm.
A hundred steps of random walk is ~5 mm — **larger than the spacing the surface
is tiled at**. INRIA's positional rate is tuned for a few hundred thousand
points that densification is actively re-seeding; here it is applied to 7.27M
points already placed on a metric-correct lattice, where almost any motion is
destructive.

**My stated hypothesis was per-Gaussian opacity jitter, and it is false.** The
`no-opacity` arm is indistinguishable from `all`. The argument behind it — that
neighbouring Gaussians receiving different opacity updates *is* spatial
modulation at the lattice frequency, so an ensemble can pattern where a single
Gaussian cannot — is sound as far as it goes and simply is not what happens at
these learning rates. Kimi's Q7 reasoning stands as originally written.

The by-product is larger than the thing it was measured for. **`no-means` also
has the best lpips in the table by a wide margin: 0.4909 against 0.5481, −10.4%
— and it is the only arm that improves lpips over the initial map at all.** Every
other arm makes lpips *worse* than the map it started from. That is the same sign
and roughly the same size as the unexplained lpips regression on room (+11.1%)
and 360 (+20.6%) in `docs/online-eval-all-families.md`, which §16 attributed to
budget starvation.

**Read that lpips result with its caveat, though.** The ablation is 100 steps on
10 held-out views, and 100 steps is inside a transient: the full 3000-step arms
below start at lpips 0.4876 and are already at 0.4736 by step 1000, so "every
arm is worse than init" is a statement about step 100, not about convergence.
What the ablation establishes without qualification is the *coverage* result —
`hp_alpha` 0.00198 vs 0.0107, a 5.4x separation with the other four arms tied.
Whether frozen positions still win on lpips at 3000 steps is a separate question
and is answered by the matrix, not by this table.

### 17.4 The seams are misregistration, not exposure — my diagnosis was wrong

The straight-edged boundaries in `gs_map_00000048.png` were attributed to TUM
freiburg1's auto-exposure: each keyframe bakes the raw pixel colour, so one
surface seen at two exposures gets painted two brightnesses, and the boundary is
the cluster edge. Three measurements, none of which supports it.

**Exposure drift is 4%, and the between-keyframe step is 0.6%.** Per-keyframe
mean luminance of the source images in the dumps:

```
                     n_kf   mean     range           spread   consecutive |d|
                                                              median   p90
360                    46   0.4229   0.4084-0.4255   4.1%     0.05%    0.6%
room                   51   0.4586   0.4413-0.4623   4.6%     0.07%    1.8%
```

A 0.6% step between neighbouring keyframes cannot produce a visible seam. The
premise is close to false for these sequences.

**Overlap-based gain fitting does not survive its own chain.**
`scripts/color_harmonize.py` (causal, keyframe k fitted against 0..k-1 already
corrected) runs away: gains pin at the 0.600 clamp from keyframe ~5 onward and
the composed map loses 1.59 dB / +0.026 lpips. Each keyframe darkens against
already-darkened predecessors and the chain compounds. Whatever it is fitting,
it is not a bounded exposure difference. The first keyframe's gain, before the
chain diverges, is ~0.9 — but a 1 cm voxel match under pose error pairs up
surfaces that are not the same surface, so that number is confounded and cannot
carry the claim either.

**And the artifact does not look like a colour step.** Magnified
(`scratchpad/seam_crop.png`), the boundary is a straight-edged *polygon* — a
projected image rectangle — enclosing a region that reads as a semi-transparent
lighter **veil** over the floor, not a region of different colour. A colour
difference cannot produce a veil. A keyframe cluster placed at slightly wrong
depth or scale, floating in front of the true surface with its silhouette being
that keyframe's image footprint, produces exactly this.

**So the seams are geometric misregistration between clusters, and they belong
to the pose/scale problem, not to appearance.** Which changes what fixes them:
per-frame exposure compensation is the wrong tool, while confidence thinning
(§16.9's `min_confidence=4.0`, which removes 38% of the map by dropping exactly
the low-confidence predictions these veils are made of) and refinement given
enough steps to lower the veil's opacity are the right ones. That is a second,
independent argument for the throughput work rather than a separate project.

The per-frame exposure parameters were built anyway (`--exposure`,
6 params/frame on the render, never the target) because the hypothesis deserved
a measurement rather than an argument, and because the falsification above is
about *global* exposure — it does not rule out per-view appearance variation
from other sources. The arm is in the matrix below.

### 17.5 Why supersampled supervision cannot work, and where the fix has to live

`--ss-loss 2` was the obvious response to "the loss cannot see the gaps": render
the training view at 2x, average-pool, then compute the loss — area integration
by Monte Carlo, no CUDA change. It did nothing (§17.3). The reason is structural,
and it generalizes.

Under pooling, the prediction is an average over sub-pixel phases. A small
positional perturbation `delta` shifts where a *smooth* underlying field is
sampled, so its effect on the pooled value is second order in `delta`. The
pooled loss is therefore differentiable with respect to a smoothed field and is
close to blind to lattice-frequency position jitter — **and going to 4x makes it
worse, not better, because more averaging means less sensitivity.**

The deeper statement: **the ground truth contains no information about the space
between its own sample points.** No loss built from a native-resolution target
can constrain what the map does there. Rendering at 2x against a 1x target
recovers the *coverage* term (gaps darken the pooled render) but not
identifiability of the lattice-frequency modes, which is what the experiment
showed. So the fix cannot come from extracting more supervision signal. It has
to come from the parameter space: a prior, a band-limited parameterization, or
freezing.

Three options, in increasing strength (Kimi's framing, adopted):

1. **Grid smoothness prior.** The Gaussians of one keyframe *are* an H x W
   image of parameters, so lattice-frequency modulation is literally
   high-frequency power in that image. Penalize `||grad_grid logit_opacity||^2`
   and the same on `log_scales`, edge-aware against the source image gradient.
   Costs one conv per cluster per step. Keeps every degree of freedom.
2. **Band-limited parameterization.** Store each cluster's parameter field as a
   coarse grid upsampled bilinearly, so lattice modulation is impossible by
   construction while the common mode stays free.
3. **Freeze.** The measurement in §17.3 says only the means matter, so freezing
   the means alone is the minimal version, and it is what the matrix tests.

Kimi's stated prediction was **opacity >> scale > means**, from `(lr x steps) /
dynamic range`: opacity moves a logit by 1.5 in 30 steps, log-scale by 0.15,
while the positional step "should be" much smaller than `dx` if noise-driven.
It also computed that the means *could* jitter by several `dx` and then
dismissed that branch because "self-psnr would collapse much more than 3.2 dB".
**The arithmetic was right and the dismissal was wrong**: the ablation shows the
opacity arm recovering 0% of the degradation against a predicted >=50%, and the
means arm recovering essentially all of it. Both reviewer and author had a
stated numeric prediction here and both were falsified by the same table.

The correction to §17.2's Q7 conclusion, which was that only coverage can
produce a pattern: **any per-Gaussian quantity that multiplies the blob can
carry the lattice** — scale is its width, position its phase, opacity its gain,
colour its value. At initialization the network's predictions are spatially
smooth (neighbouring source pixels get similar values), so coverage is the only
carrier, which is what the alpha-channel image shows and what that experiment
legitimately established. Optimization gives every Gaussian its own Adam state
and de-smooths those fields, at which point any of them can carry it. The
alpha-channel test at init cannot bound the post-optimization state, and the
original phrasing over-reached.

**Correction to the paragraph above, from the 3000-step arms.** "Optimization
destroys coverage" is what 30 and 100 steps show and it is not what happens over
a full run. The `base` arm (no filter, so it starts badly) ends *better* than it
started:

```
              self-psnr    hp_alpha
base   init     26.77       0.04113
base   3000     37.38       0.01971
```

while the tau=0.5 arm starting at 39.76 / 0.00163 degraded to 36.58 / 0.00692 in
30 steps. **The optimizer drives `hp_alpha` toward an equilibrium around
0.01-0.02 from either side** — it is not monotonically destructive, it is
attracted to a coverage level set by the supervision it can see. That reframes
the finding: the filter's advantage is not that optimization would otherwise
ruin the map, it is that the equilibrium the native-rate loss selects is an
order of magnitude worse than what the filter can hold, and the loss has no term
that would prefer otherwise. The `--aa-hard-floor` arm is the test of whether a
constraint can hold the map below that equilibrium for a whole run.

This is also why the 30-step and 100-step numbers had to be re-checked rather
than written up: a short run measures the transient, and the transient here has
the opposite sign to the steady state.

### 17.6 The means are the carrier in BOTH directions, and freezing them wins

360, 3000 iterations, exact culling, 50 held-out views. `fit` is psnr after a
per-view closed-form per-channel affine onto the target, which removes any
global exposure mismatch from the comparison; `self`/`hp_alpha` as in §17.2.

```
arm          psnr@3000  lpips@3000   fit psnr   self@3000   hp_alpha@3000
(init)         11.7902    0.4876       12.9339    26.77        0.04113
base           13.5071    0.4415       14.7003    37.38        0.01971
m_frozen       13.6149    0.4399       14.7737    25.78        0.04258
```

**Freezing the positional learning rate wins on both metrics**: +0.108 dB and
−0.36% lpips against the arm that optimizes everything, with a lower training
loss (0.0698 vs 0.0803). It is behind early — at 1000 steps it is 0.36 dB down
— and overtakes between 2000 and 3000. The other four parameter groups can do
the work; the positions were mostly wandering.

**And the coverage result inverts at length.** `m_frozen` ends at
hp_alpha 0.04258, essentially its initial 0.04113, while `base` improves to
0.01971. Over 3000 steps the means updates *close* gaps. Over 30 steps from a
filtered start they *open* them. Both are the same fact: the means are the
carrier of lattice change in either direction, and the native-rate loss has an
equilibrium coverage level around hp_alpha 0.01-0.02 that it pulls toward from
whichever side it starts.

So the three levers are separable and each does one thing:

- **`--aa-sigma 0.5` sets where the map starts** (hp_alpha 0.041 -> 0.0026).
- **`--lr-means 0` decides whether it stays there** — frozen means pin coverage
  at the baked value, free means drag it to the equilibrium.
- **The equilibrium itself is a property of the loss**, and nothing tried so far
  moves it, because the target contains no information between its own sample
  points (§17.5).

That predicts the combination — filter to 0.0026 *and* freeze so nothing drags
it back to 0.02 — and `aa_m_frozen` is running to test it. It is the arm the
whole section is for, and neither half is sufficient alone: the filter without
the freeze decays, the freeze without the filter pins a bad value.

**Kimi's decision rule, adopted:** the right parameter set is budget-dependent.
Starved (in-sequence, hundreds of steps) freeze; abundant (offline, 12000 steps)
leave free and add priors, since the full arm is what produced the −10.7% lpips
in §16.5. That is a statement about *which regime the online system is in*, and
it is in the starved one.

**Two more arms, and a correction to the prediction just above.** §17.6 said the
filter without the freeze would decay back to the equilibrium. It does not:

```
arm          psnr@3000  lpips@3000   fit psnr   self@3000   hp_alpha@3000
(init)         11.7902    0.4876       12.9339    26.77        0.04113
base           13.5071    0.4415       14.7003    37.38        0.01971
aa (tau .5)    13.5243    0.4381       14.7118    41.90        0.00721
m_frozen       13.6149    0.4399       14.7737    25.78        0.04258
m_10x          13.5576    0.4513       14.7319    25.59        0.05886
```

`aa` decays from its initial 0.00163 to 0.00721 — 4.4x worse than where it
started, but **2.7x better than the equilibrium the free map settles at**, and
it holds that for 3000 steps. So the hard floor is a real constraint, not a
delaying tactic, and "it gets dragged back" was wrong. It also has **the best
lpips of every arm** (0.4381) at psnr indistinguishable from base.

`m_10x` (positional lr divided by 10) is the interesting negative: **worse than
both endpoints**. Worst coverage of the four (0.05886, above even the frozen
arm's 0.04258) and the worst lpips (0.4513, worse than base). Lowering the rate
is not a partial version of freezing it — a small persistent drift is enough to
scramble the lattice and not enough to let the Gaussians re-close the gaps they
open, which is the worst of both. **The lever is binary; do not tune it.**

Ranking, and it depends on what is being bought:

```
lpips        aa 0.4381  <  m_frozen 0.4399  <  base 0.4415  <  m_10x 0.4513
psnr         m_frozen 13.615 > m_10x 13.558 > aa 13.524 > base 13.507
coverage     aa 0.0072  <<  base 0.0197  <  m_frozen 0.0426  <  m_10x 0.0589
```

`aa` wins the two that matter for how the map *looks* (lpips and coverage);
`m_frozen` wins psnr. `aa_m_frozen` is the arm that should take both and is
still queued.

**The positional rate is U-shaped, and the worst value is in the middle.**

```
lr_means        psnr     lpips    hp_alpha
0   (frozen)   13.6149   0.4399   0.04258
1.6e-6         13.6058   0.4444   0.05175
1.6e-5         13.5576   0.4513   0.05886
1.6e-4 (base)  13.5071   0.4415   0.01971
```

lpips and coverage are both worst at 1.6e-5, better at both ends. The mechanism
is legible: at the full rate the Gaussians move far enough to *re-close* the
gaps they open (coverage ends best of the four); frozen, nothing moves at all;
in between they jitter enough to scramble the lattice and not enough to repair
it. **Confirms the lever is binary. Anyone who "tunes it down a bit" lands in
the hole.**

**Per-frame exposure: helps, but not for the reason it was built.**

```
arm      psnr@3000  lpips@3000   fit psnr   fit lpips   hp_alpha
base      13.5071    0.4415       14.7003     0.4359     0.01971
exp       13.3477    0.4302       15.1449     0.4168     0.01921
```

**+0.4446 dB fit psnr, −2.6% lpips, −4.4% fit lpips** — the largest single-lever
gain on the perceptual metrics in this section. And raw psnr *drops* 0.159 dB.

That split is diagnostic. Raw psnr falling while fit psnr rises by three times as
much means the map has drifted to a different absolute brightness (which the
per-frame parameters now compensate at training time and nothing compensates at
deployment), while its *structure* matches the targets better. lpips is close to
invariant under a global affine, so the lpips gain is structural, not
bookkeeping.

The fitted parameters moved a lot more than §17.4 predicted: gains
1.037/1.048/1.051 with a **per-frame std of 11-12%**, against the 4.1% spread and
0.6% consecutive step measured on the source images. So §17.4's falsification
stands as stated — *global exposure drift* is not the cause of the seams, and
the seams are still a veil with a rectangular silhouette — but **something
per-view and globally affine is being absorbed, and absorbing it is worth more
than any other appearance lever tried.** Candidates not yet separated: auto white
balance (the three channel gains do differ), a view-dependent illumination term,
the network's per-pair conditional bias, or per-frame render darkening from
incomplete alpha. Naming it is open.

One thing to fix before this ships: **re-centre the exposure parameters into the
map at the end** — multiply `f_dc` by the mean gain and add the mean bias, so the
deployed map carries the average exposure instead of leaving it in per-frame
parameters that are then discarded. That should recover the 0.159 dB of raw psnr
without touching the structural gain. Not yet built.

### 17.7 The matrix, complete

360, 3000 iterations, exact culling, 50 held-out views, one seed. `init` differs
between the two groups because `--aa-sigma` changes the map before any
optimization (it starts at a worse lpips and a far better coverage), so read
each arm against its own init where the delta matters.

```
arm            psnr     lpips   fit psnr  fit lpips   self    hp_alpha
init (tau 0)  11.7902  0.4876   12.9339    0.4726    26.77    0.04113
init (tau .5) 11.8085  0.4962   12.9469    0.4816    39.76    0.00163

base          13.5071  0.4415   14.7003    0.4359    37.38    0.01971
aa            13.5243  0.4381   14.7118    0.4335    41.90    0.00721
m_frozen      13.6149  0.4399   14.7737    0.4309    25.78    0.04258
m_10x         13.5576  0.4513   14.7319    0.4414    25.59    0.05886
m_100x        13.6058  0.4444   14.7661    0.4350    24.87    0.05175
exp           13.3477  0.4302   15.1449    0.4168    38.05    0.01921
aa_exp        13.3633  0.4262   15.1467    0.4130    42.57    0.00711
aa_m_frozen   13.6497  0.4338   14.7940    0.4262    35.54    0.00729
```

**`aa_m_frozen` confirms the prediction in §17.6**: the filter sets the starting
coverage, the freeze keeps it, and the combination takes the best psnr in the
table (13.6497, +0.143 over base) while holding coverage in the good tier
(0.00729 against base's 0.01971). Neither half does this alone — `aa` gives up
0.09 dB of psnr, `m_frozen` gives up the coverage entirely (0.04258).

**`aa_exp` takes every perceptual measure**: lpips 0.4262 (−3.5% vs base), fit
psnr 15.1467 (+0.446), fit lpips 0.4130 (−5.3%), and the best coverage
(0.00711). Its raw psnr is the weak point, and §17.6 identified why and how to
fix it — the exposure re-centring, now built and running as `aa_exp_rc`.

So there are two configurations, not one, and which is right depends on the
metric being defended:

- **psnr-first: `--aa-sigma 0.5 --aa-hard-floor --lr-means 0`.**
- **perception-first: `--aa-sigma 0.5 --aa-hard-floor --exposure`** (with
  re-centring), which is the configuration matching this project's stated
  preference for lpips as its perceptual proxy (§10.11/§13.10) and the one whose
  renders should look best.

`aa_exp_mfrozen` (all three) is running and is the obvious candidate to collapse
that choice.

**Caveats, stated rather than buried.** One seed, one sequence, one map size.
360 is the hard case (7.27M Gaussians, budget-starved) and desk is untested here
— §16 measured desk converging in-sequence, so it sits in the "abundant" regime
where Kimi's decision rule predicts the *opposite* choice on `--lr-means`. The
lpips differences between the top arms (0.4262 to 0.4399) are larger than the
0.09 dB psnr noise floor established in §13, but no seed ensemble was run for
lpips specifically. `--aa-sigma 0.5` was selected on the tau sweep of the raw
baked map (§17.2) and not re-selected after refinement; the refined arms are
consistent with it but do not re-derive it.

### 17.8 The exposure re-centring fix is falsified

§17.6 predicted that folding the mean exposure back into `f_dc` at the end would
recover the 0.159 dB of raw psnr that `--exposure` costs, without touching the
structural gain. Built (`--exposure` now re-centres by default,
`--no-exposure-recenter` is the control) and measured on `aa_exp_rc`:

```
                          psnr      lpips    fit psnr   fit lpips
aa_exp   (no recentring) 13.3633   0.4262    15.1467     0.4130
aa_exp_rc before         13.3573   0.4262    15.1458     0.4131
aa_exp_rc after          13.2517   0.4269    15.1410     0.4133
```

mean gain 1.040/1.048/1.051, mean bias −0.013/−0.020/−0.018.

**It makes raw psnr worse by another 0.106 dB.** And `fit psnr` moves by −0.005,
i.e. not at all — so the map's structure is untouched and only its absolute level
moved, in the wrong direction.

Two reasons the reasoning was wrong, and the second is the interesting one:

1. The exposure transform acts on the **composited image**; folding it into
   per-Gaussian colour is only equivalent where accumulated alpha is 1. The bias
   in particular lands on the background too when applied to the render, and
   scales with alpha when applied to the colours.
2. More fundamentally, **the mean of the per-frame gains is not a property the
   held-out frames share.** The gains have mean 1.04 and std 0.11: they are
   absorbing per-frame error whose *average* is an artifact of the training set,
   not a global exposure the map should adopt. The un-recentred map was already
   at the raw-psnr optimum.

So the honest accounting for `--exposure` is: **it buys −3.5% lpips and +0.45 dB
fit psnr at a real, non-recoverable cost of ~0.16 dB raw psnr.** That is a
perception-distortion trade, not a bookkeeping error, and it should be presented
as one. The re-centring code stays (it is the control that establishes this) but
is not the default recommendation.

Third falsified prediction of mine in this section, after "opacity is the
carrier" (§17.3) and "the filter decays back to the equilibrium" (§17.6). All
three were stated with a direction and a size before the run, which is the only
reason they are legible as failures rather than as things quietly not mentioned.

### 17.9 Final matrix and the recommendation

```
arm               psnr     lpips   fit psnr  fit lpips   self    hp_alpha
init (tau 0)     11.7902  0.4876   12.9339    0.4726    26.77    0.04113
init (tau .5)    11.8085  0.4962   12.9469    0.4816    39.76    0.00163

base             13.5071  0.4415   14.7003    0.4359    37.38    0.01971
aa               13.5243  0.4381   14.7118    0.4335    41.90    0.00721
m_frozen         13.6149  0.4399   14.7737    0.4309    25.78    0.04258
m_10x            13.5576  0.4513   14.7319    0.4414    25.59    0.05886
m_100x           13.6058  0.4444   14.7661    0.4350    24.87    0.05175
exp              13.3477  0.4302   15.1449    0.4168    38.05    0.01921
aa_exp           13.3633  0.4262   15.1467    0.4130    42.57    0.00711
aa_m_frozen      13.6497  0.4338   14.7940    0.4262    35.54    0.00729
aa_exp_mfrozen   13.4625  0.4195   15.1785    0.4046    35.25    0.00789
```

**`aa_exp_mfrozen` — all three levers — is the recommendation.** Against `base`:

```
lpips      0.4415 -> 0.4195    -5.0%
fit lpips  0.4359 -> 0.4046    -7.2%
fit psnr  14.7003 -> 15.1785   +0.478 dB
psnr      13.5071 -> 13.4625   -0.045 dB    (was -0.16 for exp alone)
hp_alpha   0.01971 -> 0.00789  2.5x less lattice
```

The raw-psnr cost of the exposure lever, 0.16 dB on its own, **falls to 0.045 dB
once the positions are frozen** — the two interact, and freezing recovers most of
what exposure gives up. That collapses §17.7's two-configuration choice: −0.045
dB psnr for −5.0% lpips and 2.5x less lattice is the trade this project's stated
preference for lpips (§10.11/§13.10) says to take. `aa_m_frozen` remains the
answer if raw psnr is the number being defended (13.6497, best in the table).

```
ONLINE, at injection      --aa-sigma 0.5           (and --min-confidence 4.0, 16.9)
ONLINE, in the refiner    --aa-hard-floor --lr-means 0 --exposure --cull-exact
OFFLINE, once             the tau sweep and E1; not in the product path
NOT BUILT                 area-integrating rasterization (needs CUDA; 17.5 shows
                          it would fix viewing, not learning)
```

`--exposure-recenter` **stays off**: it cost another 0.105 dB here too, an exact
replication of §17.8 on a second configuration.

**What is fixed, and what is not.**

- Dot lattice / halftone / moire: **fixed**, and measured three ways (self-psnr,
  hp_alpha, the alpha-channel images).
- lpips regression on the large maps: **explained and fixed** — it was the
  positional learning rate, not only budget starvation.
- Seams: **re-diagnosed, not fixed.** They are geometric misregistration (a
  semi-transparent veil with a rectangular silhouette), not exposure (§17.4).
  Confidence thinning and more steps are the levers; neither was isolated here.
- Trailing streaks at depth edges: **not addressed.** The min-neighbour pitch
  rule avoids manufacturing new ones, which is not the same as removing the
  existing ones. Anisotropy clamping at injection is designed and unbuilt.
- Large black regions: **not a defect.** Those are unmapped — no keyframe covered
  them. Do not spend budget there.
- The per-view affine that `--exposure` absorbs: **works but unexplained.** 11-12%
  per-frame std against 4% measured image-mean drift. Auto white balance,
  view-dependent illumination, per-pair network bias and alpha-deficit darkening
  are not separated.

### 17.10 The cull re-verified in the regime it will run in

§17.1's zero-false-negative result was measured on the raw baked map with
`n_sigma = 3.0`. Two things changed after it: `n_sigma` went to 3.4 (the
rasterizer drops a contribution at alpha < 1/255, which for opacity near 1 is
3.33 sigma, not 3 — `forward.cu:345`), and the recommended configuration now
builds the map with `--aa-sigma 0.5`, which enlarges every scale and therefore
every footprint the bound depends on. A false-negative check does not carry
across either change for free. Re-run with the filter on and after **300
optimizer steps**, so the anisotropy and scale distributions are the refined
ones:

```
view    block    exact  radii>0  grad sup   miss  miss|grad|
   0   49.3%   10.5%    9.7%     7.7%        0    0.00e+00
   6   50.0%   12.8%   12.0%    10.5%       0    0.00e+00
  18   28.6%    7.7%    7.2%     6.7%        0    0.00e+00
  36   29.8%    5.3%    4.8%     4.4%        0    0.00e+00
  42   35.0%    7.8%    7.2%     6.1%        0    0.00e+00

none 0.70 it/s   block 1.37 it/s   exact 5.58 it/s
```

**Still zero, on every probe view, by count and by gradient magnitude.** The
kept fraction rises ~0.6 pp (the filter's larger footprints plus 3.4 vs 3.0
sigma) and throughput lands at 5.58 it/s rather than 6.51 for the same reason —
still 8.0x over no culling and 4.1x over the block test.

Kimi's remaining critiques of the cull, answered:

- **"3 sigma truncates a shell the rasterizer still renders."** Correct, and
  fixed above. It was worth ~2 pp of kept fraction.
- **"Verify the ellipse test uses the full conic; diagonal-only under-estimates
  a rotated elongated Gaussian."** Not applicable: the bound uses `smax`, the
  largest scale axis, which is a bounding *sphere*. Orientation cannot produce a
  false negative against a sphere.
- **"The per-view numbers look cross-averaged — 4.9% kept cannot coexist with
  8.6% support and FN=0."** They are per-view, and per-view `exact >= support`
  holds on every row (view 36: 5.3% vs 4.4%). The two ranges quoted in the
  summary came from different views.
- **"Sim3 scale drift could exceed the 6 px margin."** The footprint already
  carries the transform's own scale via `det(P)^(1/3)`, and the mask is computed
  from the same `kf_mats` the render uses, in the same step. Stateless is what
  makes this safe, and it is the property the cache design would have lost.

### 17.11 Correction: the levers are additive, not interacting

§17.9 said the exposure lever's raw-psnr cost "falls to 0.045 dB once the
positions are frozen — the two interact, and freezing recovers most of what
exposure gives up". **That is wrong**, and the matched-pair arithmetic says so
immediately:

```
exposure tax          exp - base                = -0.1594
                      aa_exp - aa               = -0.1610
                      aa_exp_mfrozen - aa_m_frozen = -0.1872

freeze benefit        m_frozen - base           = +0.1078
                      aa_m_frozen - aa          = +0.1254

additive prediction   aa +0.0172, freeze +0.1254, exp -0.1872
                      sum -0.0446   actual -0.0446
```

The tax is a **constant 0.16-0.19 dB in all three configurations**, and if
anything it is slightly *larger* with the positions frozen, not smaller. The
−0.045 dB figure came from comparing `aa_exp_mfrozen` against `base`, where
freezing's independent +0.125 happens to offset most of the tax. **The three
levers add; there is no interaction to claim.** The sum reproduces the measured
total to four decimal places.

My proposed mechanism — free positions absorbing the exposure residual through
geometry, so freezing would shrink the tax — predicted the *opposite sign* to
what was measured, and it has no signature in the coverage numbers either
(`exp` 0.01921 vs `base` 0.01971: the exposure residual leaves no mark on
coverage at all).

**The recommendation itself is unchanged** — `aa_exp_mfrozen` still holds the
best lpips, fit psnr, fit lpips and top-tier coverage, and those are measured
facts. Only the explanation for its raw-psnr number was wrong, and the honest way
to report the exposure lever is its matched-pair tax of **−0.17 dB**, not the
−0.045 dB that a mismatched baseline produced.

Fourth falsified prediction of mine in this section. It is also the one that
mattered most, because unlike the other three it had already been stated as a
finding rather than caught in-flight.

**A real interaction does exist, and it is with step count.** Freezing the
positions is worth −0.115 dB at 100 steps and +0.11 dB at 3000: the sign flips.
Early, the positional updates do legitimate sub-pixel alignment and are worth
paying for; late, they overfit per-view noise and freezing blocks it. **The
online regime is ~300 steps, not 3000**, which is exactly where the sign flip
lives and where nothing has been measured. That gap is now the blocking question
for shipping this online, and it is measured next.

### 17.12 The online regime (300 steps) picks a different configuration

§17.11 flagged the sign flip and this measures it. 360, τ=0.5 hard floor,
**300 iterations** — the budget §16 established for a 7.27M map in-sequence,
against the 3000 every arm above used.

```
arm                psnr     lpips   fit psnr  fit lpips   self    hp_alpha
init (tau .5)     11.8085  0.4962   12.9469    0.4816    39.76    0.00163
aa300             12.3955  0.4982   13.7424    0.4845    35.91    0.01460
aa_exp300         12.3985  0.4970   13.7597    0.4827    35.87    0.01472
aa_m_frozen300    12.0488  0.4500   13.4065    0.4393    33.92    0.00627
```

**The free-means arms do not improve lpips at all in this regime.** `aa300` ends
at 0.4982 against an initial 0.4962 — marginally *worse* than the map it started
from, after 300 steps of optimization. `aa_m_frozen300` ends at 0.4500, **−9.3%
against init and −9.7% against the free arm**, and holds coverage at 0.00627
against 0.01460.

**This is the online lpips regression, reproduced and fixed.**
`docs/online-eval-all-families.md` records room at +11.1% and 360 at +20.6%
*worse* lpips after online refinement, which §16 attributed to budget starvation.
The mechanism is now identified and it is not starvation: **at a starved step
count the positional updates buy psnr and cost lpips**, and freezing them
converts a −0.4% lpips outcome into −9.3%. The price is 0.347 dB of psnr.

So the configuration is genuinely budget-dependent, and the flip is much larger
than §17.11's 100-step number suggested:

```
freeze benefit, psnr      100 steps   -0.115
                300 steps   -0.347
               3000 steps   +0.125
freeze benefit, lpips     300 steps   -9.7%   (better)
                3000 steps  -1.0%    (better)
```

**Online (~300 steps): freeze, and take −0.35 dB psnr for −9.7% lpips and 2.3x
less lattice.** That is the trade this project's stated lpips preference
(§10.11/§13.10) selects, and it is the trade that turns the online refiner from
lpips-negative to strongly lpips-positive on the large maps — the single
outstanding defect in the online results table.

**Offline (3000+ steps): freeze also wins**, but only just (+0.125 dB, −1.0%
lpips), so the choice barely matters there.

Exposure is nearly inert at 300 steps (`aa_exp300` vs `aa300`: +0.003 dB, −0.2%
lpips) — 6 parameters per frame with each frame visited ~6 times cannot converge.
**Exposure is an offline lever; at online budgets it is not worth its
complexity.** That is a cleaner separation than §17.9's, which recommended it
online on 3000-step evidence.

### 17.13 desk: the budget-dependent rule is falsified, freezing wins everywhere

desk (1.86M Gaussians, 14 keyframes) is the "abundant" regime — §16 measured it
converging in-sequence, where 360 cannot. Kimi's decision rule
("starved -> freeze; abundant -> free + priors") therefore predicts free
positions should win here, and it committed to a number before the run:
**free beats frozen by +0.4 to +0.9 dB psnr (centre 0.6), falsified if the gap
is below +0.2 dB.** The reasoning was supervision density — 360 sits at 145k
Gaussians per view (3.6x §13.10's saturation point, so positions overfit) and
desk at ~37k (≈1x, so positions can converge honestly).

```
desk, 3000 iters   psnr     lpips   fit psnr  fit lpips   self    hp_alpha
base              14.3694  0.3726   14.8092    0.3763    36.84    0.00749
aa_m_frozen       14.5271  0.3621   14.9294    0.3663    38.98    0.00335
aa_exp            14.1916  0.3679   14.9451    0.3671    39.77    0.00483
aa_exp_mfrozen    14.3213  0.3598   15.0344    0.3594    38.86    0.00374
```

**Matched pair: `aa_exp` − `aa_exp_mfrozen` = −0.130 dB. Frozen wins.** Against
a predicted +0.4 to +0.9 in the other direction, and past the stated
falsification line by a wide margin. Frozen also takes lpips (0.3598 vs 0.3679,
−2.2%) and coverage (0.00374 vs 0.00483).

**The rule is falsified. Freeze the positions on both map sizes.** The freeze
benefit is not merely same-signed but nearly the same size across a 3.9x
difference in map size and a 4x difference in supervision density:

```
freeze benefit (with exposure, matched)   desk  +0.130 dB     360  +0.099 dB
exposure tax   (matched)                  desk  -0.206 dB     360  -0.161 dB
```

Whatever the positional updates are doing wrong, **it is not overfitting caused
by supervision starvation** — that was the mechanism behind both the rule and my
own §17.3 framing, and desk was supposed to be the regime where it disappears.
It does not. The remaining explanation consistent with everything measured is
the one in §17.3's arithmetic: `lr_means = 1.6e-4 * extent` is simply far too
large relative to the lattice pitch these maps are built on, independent of how
much supervision there is. INRIA tuned that rate for a few hundred thousand
points being actively re-seeded by densification, and it does not transfer to a
metric-correct per-pixel lattice that nothing re-seeds.

So the configuration table simplifies rather than branching by map size:

```
ALWAYS      --aa-sigma 0.5 --aa-hard-floor --lr-means 0 --cull-exact
                                                  (+ --min-confidence 4.0 above ~4M)
OFFLINE     add --exposure   (inert at online budgets, 17.12)
```

Every headline in §17 now replicates on a second sequence: the filter improves
coverage (0.00749 -> 0.00335), freezing improves psnr and lpips, and the
exposure tax is a constant. The one thing that does NOT replicate is the
budget-dependence, which was never measured — it was inferred, by both of us,
from a mechanism that turned out not to be the operative one.

### 17.14 Freezing exactly the means is the optimum; opacity is the slow carrier

Two follow-ups, both with predictions on record, both mostly falsified.

**Pitch-scaled positional rate** (`--pitch-lr`): the parameter becomes a
dimensionless residual, `mean = base + pitch * delta`, so one Adam step of `lr`
displaces a Gaussian by that fraction of its OWN lattice spacing. Note this
cannot be done by reweighting gradients — Adam's update is `lr * m /
(sqrt(v)+eps)`, scale-invariant in the gradient, so only a reparameterization
changes the step size.

Kimi's invariant check first, because it is a real prediction: `lr/pitch` should
be nearly constant across sequences (small scene -> small extent -> small lr,
but near surfaces -> small pitch, cancelling), which would explain why the
freeze benefit is the same size on both while supervision density differs 4x.

```
 seq    n_gauss  extent  lr=1.6e-4*ext  pitch med  lr/pitch
desk  1,860,034   2.588       0.4141mm    2.315mm    0.179
 360  7,267,700   2.977       0.4763mm    3.870mm    0.123
```

Not equal (1.45x apart) but varying far less than the 4x in supervision density,
**and the ordering matches**: desk has the higher `lr/pitch` and the larger
freeze benefit (+0.130 vs +0.099). Weak confirmation at n=2, and it corrects the
earlier back-of-envelope: the default rate is 12.3% of the lattice pitch per
step on 360, not the 23% previously stated.

```
360, 300 steps      psnr     lpips   hp_alpha        360, 3000 steps
init (tau .5)      11.8085  0.4962   0.00163
aa300 (free)       12.3955  0.4982   0.01460         aa       13.5243 0.4381 0.00721
pitch_15_300       12.6314  0.5031   0.01279
pitch_05_300       12.3073  0.4714   0.01353
pitch_02_300       12.1415  0.4565   0.01084         pitch_02 13.5618 0.4324 0.01193
warm100_300        12.1491  0.4792   0.01517
warm150p_300       12.1714  0.4601   0.01276
frzop_300          12.0186  0.4722   0.00258
frzall_300         11.8621  0.4775   0.00292
aa_m_frozen300     12.0488  0.4500   0.00627         aa_m_frz 13.6497 0.4338 0.00729
```

**Pitch scaling is a better frontier but not a better point.** At matched psnr
(12.31 vs 12.40) it gives lpips 0.4714 against 0.4982 — **5.4% better lpips for
the same distortion** — and it holds both ends of the frontier (`pitch_15` takes
the best psnr in the whole online table, 12.6314, at the worst lpips). So the
parameterization is right and the naive extent-scaled rate is wrong. But every
point on it is dominated on lpips by `lr = 0`. Kimi predicted constant-kappa at
+0.02 to +0.08 dB over frozen with neutral lpips; measured −0.088 dB at 3000
steps, wrong sign.

**The warm-up schedule is the worst option, and instructively so.** Free for 100
steps then frozen: psnr +0.100 over always-frozen — exactly inside Kimi's
predicted +0.05 to +0.12 — but lpips +6.5% *worse* against a predicted neutral,
and `hp_alpha` 0.01517, **worse than never freezing at all**. The mechanism is
clean: **warm-up takes the damage without the repair.** Early free motion
scrambles the lattice, and freezing then locks the scrambled state in while
removing the very channel that would have repaired it by 3000 steps (§17.6).

**Opacity is the slow second carrier — confirmed.** §17.3 found `no-opacity`
indistinguishable from `all` (0.01059 vs 0.01079), but that was measured with the
means free, where their ~0.009 of lattice buries everything else. With the means
frozen, freezing opacity too takes `hp_alpha` from 0.00627 to **0.00258**, within
touching distance of the initial 0.00163. Both measurements are right; opacity is
a real carrier roughly an order of magnitude slower than the means. Kimi's vote
(opacity, then rotation) is correct.

**But do not freeze it.** The lpips ordering at 300 steps is unambiguous:

```
aa_m_frozen  0.4500   <- means frozen, nothing else
pitch_02     0.4565
warm150p     0.4601
frzop        0.4722   <- + opacity frozen
frzall       0.4775   <- only colour free
warm100      0.4792
aa (free)    0.4982
```

**Freezing exactly one parameter group is the optimum. Freezing more hurts,
freezing less hurts.** `frzall` — "bake the geometry, fit only appearance",
architecturally the most self-consistent option for a trajectory-anchored map —
lands at psnr 11.8621 against an initial 11.8085, i.e. 300 steps of optimization
buy essentially nothing, and lpips 0.4775 against 0.4500. The scale/rotation/
opacity channels carry real value even though opacity also carries lattice.

So §17.13's configuration stands unchanged after four attempts to improve on it.
```
--aa-sigma 0.5 --aa-hard-floor --lr-means 0 --cull-exact
```

**The 3000-step pitch sweep, and a clean monotonic mechanism.**

```
360, 3000 steps    psnr     lpips   fit lpips   self    hp_alpha
aa (free)         13.5243  0.4381    0.4335    41.90    0.00721
pitch_02          13.5618  0.4324    0.4251    35.96    0.01193
pitch_05          13.5237  0.4282    0.4227    39.00    0.00921
pitch_15          13.5365  0.4407    0.4361    42.32    0.00610
aa_m_frozen       13.6497  0.4338    0.4262    35.54    0.00729
```

**Coverage improves monotonically with more (properly scaled) positional motion**
— hp_alpha 0.01193 -> 0.00921 -> 0.00610 across pitch 0.02 -> 0.05 -> 0.15, and
`pitch_15` takes the best coverage and best self-consistency of any arm in the
whole section. That is the §17.6 repair mechanism isolated cleanly: given enough
steps, positional motion closes gaps, and scaling the rate by each Gaussian's own
spacing makes the repair efficient rather than destructive.

lpips is U-shaped over the same axis with its optimum at 0.05, where
**`pitch_05` takes the best lpips of any 360 arm without exposure** (0.4282
against frozen's 0.4338 and free's 0.4381).

So the two regimes genuinely differ, and this is the one place in §17 where the
budget does change the answer:

```
ONLINE   (~300 steps)   --lr-means 0            frozen wins lpips outright
                                                 (0.4500 vs 0.4565 for the best
                                                  pitch arm)
OFFLINE  (3000 steps)   --pitch-lr --lr-means 0.05   for lpips (0.4282)
                        --lr-means 0                 for psnr  (13.6497)
```

The reason the flip exists is now mechanical rather than mysterious: repair needs
steps. At 300 steps positional motion has done its damage and not yet its repair,
so freezing dominates; by 3000 the repair has arrived and a correctly scaled rate
beats freezing on perception. **This also retires the "budget-dependent
configuration" idea in its original form** — it is real, but it is about *repair
time*, not about supervision saturation (§17.13), and it selects between two
positional settings rather than between whole configurations.

### 17.15 Authoritative summary — read this, not the earlier subsections

The recommendation changed four times inside §17 as evidence arrived. Earlier
subsections are kept because the falsifications are the record, but **this block
supersedes every configuration statement above it.**

```
INJECTION (online, free)
  --aa-sigma 0.5 --aa-hard-floor      band limit from the lattice pitch, held
                                      as a constraint (17.2, 17.3)
  --min-confidence 4.0                 for maps above ~4M Gaussians (16.9)

REFINER
  --cull-exact                         8-9x throughput, zero false negatives
                                       (17.1, 17.10)
  --lr-means 0                         online (~300 steps) AND the psnr optimum
                                       at 3000 (17.12, 17.13, 17.14)
  --pitch-lr --lr-means 0.05           offline only, if lpips is the target
                                       (0.4282 vs 0.4338 frozen) (17.14)
  --exposure                           offline only; inert at 300 steps (17.12)

NOT RECOMMENDED
  --exposure-recenter                  measured harmful twice (17.8)
  --lr-means 1.6e-5 / 1.6e-6           worse than both endpoints (17.6)
  --freeze-means-after N               damage without repair (17.14)
  --ss-loss N                          structurally cannot work (17.5)
  freezing opacity as well             best coverage, worse lpips (17.14)
```

Measured effect of the online configuration against the current default, 360,
300 steps: **lpips 0.4982 -> 0.4500 (−9.7%), hp_alpha 0.01460 -> 0.00627 (2.3x
less lattice), psnr 12.3955 -> 12.0488 (−0.35 dB).** Against the *initial map*,
lpips goes from +0.4% worse to −9.3% better, which is the online lpips
regression in `docs/online-eval-all-families.md` turned around.

**Open, in priority order.**

1. The per-view affine `--exposure` absorbs is unexplained: 11-12% per-frame std
   against 4.1% measured source-image drift (17.6). Four candidates unseparated.
2. The seams are geometric misregistration, re-diagnosed but not fixed (17.4).
3. `--max-anisotropy` is built and never measured — the trailing streaks are the
   one defect from the original report with no number against them.
4. Everything here is one seed. The lpips gaps between top arms (0.4282-0.4500)
   exceed the psnr noise floor but no lpips seed ensemble was run.
5. Only 360 and desk. room, the other sequence with an lpips regression, is
   untested against any of this.

### 17.16 Answering the three defects left open in 17.15

**(3) Trailing streaks: the anisotropy clamp is harmful at every setting.**
`--max-anisotropy` shrinks each Gaussian's long axis to at most N times its
short one, at injection. 360, 300 steps, on top of the recommended config:

```
                psnr     lpips   hp_alpha
no clamp      12.0488   0.4500   0.00627
N = 20        11.8146   0.4532   0.00656
N = 10        11.7410   0.4540   0.00701
N = 4         11.6925   0.4558   0.00801
```

Monotone in all three metrics: tighter is worse. Even the loosest setting costs
0.23 dB and 0.7% lpips.

**The design reasoning was wrong.** It was built on "clamped, never deleted, so
no holes" — but clamping opens holes too. Shrinking the long axis shrinks the
footprint, and the band limit only guarantees the *smallest* scale reaches the
lattice pitch; it does not compensate for a shortened long axis. The
monotonically worsening `hp_alpha` is that mechanism's signature. Read the other
way: the elongated Gaussians are doing useful work, covering surface area
efficiently. They *look* like streaks and they cost less than removing them.

Verdict: **the trailing streaks are not fixable at injection by anisotropy.**
The flag stays, defaulted off, with this table in its help text.

**(1) The black is 90% the confidence filter, and it is recoverable — but the
knob is the one §16.9 says to turn the other way.** §17.9 called the black
"unmapped, not a defect". Half of that is wrong. Each injection filter relaxed
alone, 360, 25 held-out views, alpha < 0.1:

```
                  black   vs deployed   gaussians
deployed          7.15%       —          7,267,700
no depth-pct      6.99%     -0.16%       7,362,852
no confidence     4.11%     -3.04%       7,805,514
no opacity        6.12%     -1.03%       8,092,969
no max-scale      7.15%     -0.00%       7,267,700
none              3.76%     -3.38%       9,043,968
```

**47.3% of the black is content a keyframe saw and the filters discarded**, and
**90% of that is the confidence threshold alone**. The deal is better than it
looks: +7.4% Gaussians removes 43% of the black, because the low-confidence
predictions sit *exactly in the holes* rather than being spread uniformly.

Case (b) — seen only by non-keyframes — is effectively empty: the nearest
keyframe to a held-out viewpoint averages 0.057 m against 0.037 m for the
nearest tracked frame, so the keyframes already cover wherever the camera went.
The remaining 3.76% is genuinely never-observed and no injection policy recovers
it. `max_scale=0.5` is not binding on this map at all (identical counts).

The tension is explicit and unavoidable: §16.9 measured `min_confidence=4.0`
(*removing* 38% more) as winning both metrics at matched wall clock. That was a
throughput-constrained comparison, and exact culling has since relaxed the
throughput constraint 8-9x, so the trade may have moved. Measured directly at
300 steps (`conf0_300` / `conf40_300`) rather than argued from the old result.

**(2) Seams: a per-keyframe depth scale is built** (`--kf-depth-lr`), one
learnable log-scale per keyframe sliding its cluster along its own view rays —
the smallest parameter that can correct the mechanism §17.4 diagnosed. It is
deliberately not a per-keyframe pose: the pose gate is closed at fusion time
(§13.12d), and scale along the view ray is a different quantity from the
trajectory and does not feed back into it.

The learned scales are themselves a test of the diagnosis, independent of any
metric: **if the veil really is a cluster at the wrong depth, they must depart
from 1.0; if they all sit at 1.000, the diagnosis in §17.4 is wrong too** and
the seam mechanism is still unidentified.

**(1) measured: the confidence filter is now net-harmful, reversing §16.9.**
360, 300 steps, recommended config otherwise, matched STEPS:

```
                 psnr     lpips   hp_alpha   black   gaussians
min_conf 0.0   12.4745   0.4535   0.00522   4.11%   7,805,514
min_conf 1.5   12.0488   0.4500   0.00627   7.15%   7,267,700
min_conf 4.0   10.1129   0.5159   0.00951     —     4,531,321
```

**Turning the filter off is +0.426 dB, 43% less black and better coverage, for
0.8% of lpips. Turning it up to §16.9's recommended 4.0 costs 1.94 dB.**

This does not contradict §16.9; **its premise expired.** That result was measured
at *matched wall clock* while throughput was the binding constraint, where a
sparser map bought enough extra steps to win. Exact culling (§17.1) removed that
constraint: with the cull keeping ~8%, +7.4% Gaussians costs ~7% more per step,
not the 38% that made the old trade work. Once steps are cheap, the map should
be denser, not sparser — the opposite conclusion from the same trade.

`min_confidence=4.0` should be withdrawn as the standing recommendation for
large maps. It was right for a system without exact culling and is wrong for
this one. The 1.94 dB is not subtle: at 300 steps a 4.53M map has no time to
compensate for what it threw away.

**(2) measured: the per-keyframe depth scale finds a real signal and buys
nothing.**

```
                  psnr     lpips   hp_alpha
no kf-depth     12.0488   0.4500   0.00627
--kf-depth-lr   12.0514   0.4492   0.00601
learned scales: mean 1.01417  std 0.01696  range [0.97979, 1.04689]
```

**The scales depart from 1.0 — std 1.7%, spanning −2.0% to +4.7%** — so §17.4's
diagnosis survives its own independent test: the clusters really are placed at
slightly wrong depths, and at 2 m a 2-5% error is 4-10 cm, easily enough to
float a visible veil in front of a surface. The mean of 1.014 is a *global* bias:
every cluster wants to sit 1.4% further out, which is a scale mismatch between
the SLAM trajectory's Sim3 and Splatt3R's metric depth.

But the photometric payoff is +0.003 dB — nothing. Two readings, not yet
separated: the veil may be visually salient and photometrically negligible
(a few percent of pixels at low alpha), or the misplacement may not be a pure
per-cluster scale and one scalar cannot capture it. The parameter is cheap
(46 scalars) and does no harm, so it stays available and off.

**So the seams remain diagnosed but unfixed**, and the honest summary is that
the mechanism is confirmed (misregistration, and now quantified at ±2-5% depth)
while no lever tried moves the metric. The 3000-step arm is running.

**Correction: the depth-scale test is one-sided, and it does not confirm §17.4.**

At 3000 steps the same arm gives:

```
                  psnr     lpips   hp_alpha   learned scales
no kf-depth     13.6497   0.4338   0.00729    —
--kf-depth-lr   13.6129   0.4323   0.00775    mean 1.01267 std 0.03050
                                              range [0.90102, 1.08542]
```

Still no metric movement (−0.037 dB, −0.3% lpips), and **the spread has widened
from std 1.7% to 3.05%**, now spanning −10% to +8.5%. At 2 m that is a 20 cm
range of per-cluster displacement bought for nothing.

That widening is the tell. **A free parameter departs from its initialization
whether or not the hypothesis behind it is true**, and one that keeps departing
as steps accumulate, while the held-out metric does not move, is behaving like a
nuisance degree of freedom absorbing training-view residual — not like a
correction converging on a real geometric error.

So the test stated when this was built was **one-sided, and it was overstated
above**: had the scales stayed pinned at 1.000, §17.4's diagnosis would have been
falsified. Their departing is weak evidence at best, and the trajectory of that
departure argues against rather than for. §17.4's re-diagnosis of the seams still
rests on what it originally rested on — the artifact is a veil with a rectangular
silhouette, which a colour difference cannot produce, plus the 4% exposure
measurement — and not on this.

The right conclusion is narrower than the one written above: **the seams are
diagnosed as misregistration on morphological grounds, no lever tried moves any
metric, and the per-keyframe depth scale is not evidence either way.** It stays
off.

**Confidence, settled at both budgets: §16.9 is scoped, not retracted.**

```
360, matched STEPS, recommended config otherwise

               300 steps                    3000 steps
             psnr     lpips   black       psnr     lpips   hp_alpha
conf 0.0   12.4745   0.4535   4.11%     13.6378   0.4345   0.00696
conf 1.5   12.0488   0.4500   7.15%     13.6497   0.4338   0.00729
conf 4.0   10.1129   0.5159     —       13.7647   0.4511   0.00664
```

The verdict inverts with budget, which is why the flat retraction drafted above
was wrong and the measurement was worth taking:

- **Online (300 steps): turn the filter OFF.** conf 0.0 is +0.426 dB over the
  default with 43% less black, and §16.9's conf 4.0 is a **1.94 dB disaster** —
  a 4.53M map has no time to compensate for what it discarded.
- **Offline (3000 steps): conf 4.0 takes psnr** (+0.115 over default) **and
  loses lpips by 4.0%.** conf 0.0 and 1.5 are within noise of each other on
  psnr, with conf 0.0 holding better coverage.

Two things to carry forward. First, §16.9's headline — that conf 4.0 wins *both*
metrics — came from a **matched-wall-clock** comparison that handed it 3000 steps
against 1817. At matched steps it wins psnr and loses lpips. Exact culling made
matched-steps the right comparison, and under it the density trade is a
perception-distortion trade like every other one in §17.

Second, the lpips ordering is stable across every budget: **denser maps have
better lpips, sparser maps have better psnr.** Kimi's reading is the natural one
— psnr rewards filling a hole with anything, lpips would rather have black than
blurred filler — and it predicts the 0.8% lpips tax on conf 0.0 at 300 steps
should shrink by 3000, which it does (0.4535 -> 0.4345, now level with conf 1.5).

### 17.17 Splatt3R's depth is 6% short, and the evaluation cannot see it

Kimi's Q21 point: depth scaling (points pushed along their own rays) and
translation scaling (the trajectory scaled) are **photometrically degenerate**
under multi-view rendering — images cannot separate them. TUM is RGB-D, so the
sensor can. Ratio of the network's predicted per-pixel camera-space depth to the
sensor depth, per keyframe, no pose involved anywhere:

```
                          keyframes   median ratio   mean     std     range
rgbd_dataset_fr1_360         46         0.94046     0.94653  0.09615  [0.732, 1.127]
rgbd_dataset_fr1_desk        13         0.95242     0.97150  0.09667  [0.867, 1.224]
```

**Splatt3R places surfaces 5-6% NEARER than they are.** The comparison is
network-vs-sensor in the keyframe's own camera frame, so this is the checkpoint's
bias, not SLAM scale drift — the attribution Kimi asked for, decided. (Crop
convention verified: `resize_img` scales TUM's 640x480 long side to 512 and the
subsequent centre-crop is a no-op at 4:3, so the comparison is pixel-aligned.)

Three independent numbers then close on each other:

```
network depth bias vs sensor            -6.0%
recovered by --kf-depth-lr (17.16)      +1.4%
residual                                 4.5%
  its projected shear at a held-out viewpoint:
  f * b * dd/d^2 = 517 * 0.057 * 0.09/4 = 0.66 px      <- sub-pixel
```

**The photometric loss recovers exactly the part of the depth error that is
photometrically observable, and no more.** The +0.003 dB from `--kf-depth-lr` was
not a disappointing result; it was the quantitative prediction of the parallax
arithmetic, which nobody had done.

#### The methodological consequence, which is larger than the finding

Held-out views sit on the trajectory — the mean distance from a held-out
viewpoint to the nearest keyframe is **0.057 m** (§17.16). At that baseline a 4.5%
depth error moves a pixel by two thirds of a pixel. **The evaluation protocol is
structurally blind to depth error**, in exactly the way §17.2 showed it is blind
to the lattice (psnr moved 0.01 dB across a sweep that changed the visible
artifact 13.6x).

That reframes the two defects still marked unsolved. The seams (a veil from a
misplaced cluster) and the trailing streaks are **novel-view phenomena**, while
every number this project reports is sampled next to the supervision baseline.
It is not that many fixes were tried and failed. **There is currently no metric
that can see them, and under that condition success and failure are
indistinguishable.** The lattice was fixed only because a metric for it was built
first (self-consistency and `hp_alpha`), and then the fix followed in one pass.

So the next step for the seams and the streaks is **not another fix**. It is the
metric. Kimi's proposal, adopted as the design:

- **Fly-through warp consistency (primary).** Render 5 frames on a +-5 cm dolly
  around each held-out pose; warp frame t to t+1 using the rendered depth and the
  known pose; report the patch-SSIM deficit. A veil's front layer carries the
  back layer's texture at its own wrong depth, so the warp must fail there.
  "Colour motion inconsistent with depth motion" is the defining signature of a
  veil and needs no optical flow — the warp is enough. Sensitive to both static
  seams and fly-through swim.
- **Seam step (static supplement).** Render the per-pixel argmax-contributing
  keyframe id (the renderer already computes what is needed); boundaries are
  4-neighbour jumps in that id map, which is derived from the render and
  therefore reliable, unlike extracting edges from the image. Keep boundary
  pixels with rendered depth difference < 5 cm (excludes true occlusion edges)
  and accumulated alpha > 0.8. Measure the brightness step across them,
  normalized by local gradient energy. Null: shift every boundary 10 px and
  report the ratio.

Neither is built.

#### Also usable immediately

The −6% is a calibration number with downstream consumers. The lattice pitch is
estimated from the Gaussian positions themselves, so it inherits the bias:
**`Δx` is ~6% short everywhere**, which means the τ=0.5 selected in §17.2 is
really τ≈0.53 against true metric spacing. Small, inside the flat part of the
sweep, and worth knowing rather than rediscovering.

### 17.18 The online A/B contradicts the offline proxy — investigation open

Everything in §17.2-17.16 was measured with `scripts/refine_local.py` standing in
for the online refiner: a fixed set of 50 supervision frames, a map built once,
a fixed step count. The recommendation was then wired into `run_refiner` and a
**matched pair of full online runs** was taken to confirm it. It did not confirm
it.

```
online, refined map, scored by eval_map_quality (n=50)

desk            psnr     lpips        room           psnr     lpips
baked          10.7085  0.5588        baked        10.4264  0.5813
old defaults   12.0971  0.5446        old          11.1298  0.5720
NEW defaults    9.3865  0.7327        NEW           8.9339  0.7974
```

The controls are clean: both arms' *baked* maps score bit-identically
(10.7085/0.5588 on desk, 10.4264/0.5813 on room), so SLAM was deterministic and
the only variable is the refiner. The `old` arm is not a reproduction of the
historical table — exact culling is on for both, so it is an "old quality
settings, new throughput" control, which is the right control for the quality
question and must not be quoted as reproducing §16's numbers.

One-flag-at-a-time isolation on desk:

```
old (all off)              12.0971   0.5446
+ aa-sigma 0.5 only        12.0307   0.5629     -0.066 dB
+ freeze-means only        11.3756   0.5471     -0.722 dB
all three                   9.3865   0.7327     -2.711 dB   <- not additive
```

**The levers do not add here, and the residual is large.** `min_confidence=0`
and/or an interaction carries about −1.9 dB that neither of the other two
explains.

Two things the offline proxy got wrong, both now on record:

1. **`min_confidence=0` was +0.426 dB offline at 300 steps (§17.16) and is
   catastrophic online.**
2. **Freezing the means was −9.7% lpips (better) offline on 360 at 300 steps and
   is +0.5% lpips (worse) online on desk at 104 steps.** Not a small
   disagreement — the sign flips.

Two candidate causes, being separated rather than argued:

- **Sequence.** All the offline 300-step work is on 360; desk was only ever run
  at 3000. The offline desk 300-step cell was never measured and is running now.
- **The online loop itself.** Supervision there is a recent ring plus a
  reservoir over anchor-relative poses, sampled from a map that is still
  *growing* — Gaussians injected late get very few of the ~104 steps, and early
  ones are optimized against early frames and never revisited. The offline proxy
  has a complete map and a fixed frame set from step 1. That difference is
  structural and was never modelled.

**Until this resolves, §17.15's configuration block is not validated online.**
The offline results stand on their own terms; what does not stand is the
inference from them to the deployed system. This is exactly the gap that the
matched pair existed to test, and it is the reason to run it rather than ship on
a proxy.

**Isolation completed, and the online defaults are reverted.**

```
desk, online, refined map      psnr     lpips    d_psnr
old (all three off)          12.0971   0.5446      —
+ aa-sigma 0.5 only          12.0307   0.5629    -0.066
+ min_confidence 0 only      11.5596   0.6054    -0.538
+ freeze-means only          11.3756   0.5471    -0.722
all three                     9.3865   0.7327    -2.711
```

Sum of the individual effects is **−1.33 dB**; the measured joint effect is
**−2.71 dB**. **No single flag explains it — there is a −1.4 dB interaction.**

`run_refiner` and `main.py` now default `aa_sigma=0`, `freeze_means=False`,
`min_confidence=1.5` — the configuration that measures best *in the system it
ships in*. Exact culling stays on: it is a pure throughput change with zero
false negatives and is untouched by any of this. The flags remain, so the
offline settings are one argument away, and `--refiner-no-freeze-means` became
`--refiner-freeze-means` so the default reads as what it is.

**This is the honest engineering call, and it costs the headline.** §17.12's
"the online lpips regression is fixed" does not survive its own system test.
What survives is everything measured *about the offline refiner*, which is a
real object this project also ships (`scripts/refine_local.py`), plus §17.1's
throughput work, plus the diagnostics — §17.2's lattice mechanism, §17.4's seam
re-diagnosis, §17.17's −6% depth bias and the parallax argument for why the
protocol cannot see it. None of those depend on the online loop.

**The leading hypothesis for the interaction, to test next**: the offline proxy
has a complete map and a fixed frame set from step 1, while the online loop
optimizes a map that is still *growing*. With ~104 in-sequence steps and
keyframes arriving throughout, a Gaussian injected late gets almost no steps,
and one injected early is fitted against early frames and never revisited. Every
one of the three levers changes how much a Gaussian depends on being optimized
at all — the band limit changes its initial footprint, freezing removes its
ability to move, and `min_confidence=0` adds many Gaussians that most need
optimizing. A lever that assumes "the optimizer will fix this" compounds badly
when most Gaussians are barely optimized. That predicts the interaction should
vanish if the refiner is run to convergence *after* the sequence ends
(`--refiner-polish-secs`), which is a direct test and cheap.

**The sequence is eliminated; it is the online loop.** The offline desk 300-step
cell — never measured before, and the obvious confound since all the offline
300-step work was on 360 — reproduces the *offline* verdict, not the online one:

```
desk offline, 300 steps      psnr     lpips    hp_alpha
aa300      (free means)    14.1837   0.4298   0.00789
aa_m_frozen300 (frozen)    13.8114   0.4199   0.00461     -0.372 dB, -2.3% lpips

desk online, 104 steps
old        (free means)    12.0971   0.5446
iso_frz    (frozen)        11.3756   0.5471     -0.722 dB, +0.5% lpips
```

Same sequence, same lever, opposite sign on lpips. And the protocols match —
`eval_map_quality` and `refine_local` both umeyama-align the estimate to ground
truth and score held-out non-keyframes at mapped ground-truth poses, so that is
not the explanation either.

**What is left is the loop**: a map that grows while it is optimized, with
supervision drawn from a recent ring plus a reservoir, against a fixed complete
map and a fixed frame set in the proxy. `--refiner-polish-secs 900` is the
direct test — it keeps the refiner running after the sequence ends, on a
complete map, which is exactly the condition the proxy assumes. If the
interaction is the growing map, polishing should recover most of the −2.71 dB.
Running.

### 17.19 The growing map was the cause — for psnr

`--refiner-polish-secs 900` keeps the refiner optimizing after the sequence
ends, on a complete map. Everything else stays online: the same loop, the same
supervision store, the same anchor-relative poses. desk, all three levers on:

```
                                steps    psnr     lpips
baked (unrefined)                 —    10.7085   0.5588
old defaults, in-sequence        104   12.0971   0.5446
NEW defaults, in-sequence        104    9.3865   0.7327
NEW defaults + 900 s polish     1208   13.4039   0.6055
```

**The −2.71 dB is recovered and then some: +4.02 dB, and it becomes the best
psnr of any online arm.** So the collapse was a step-budget and step-distribution
artifact, not a property of the levers. §17.18's hypothesis is confirmed on psnr:
a map that grows while it is optimized gives late Gaussians almost none of the
~104 in-sequence steps, and all three levers assume an optimizer that will
actually run.

**lpips does not recover**: 0.7327 -> 0.6055, still 11.2% worse than `old` and
worse than the *unrefined* baked map (0.5588). So with steps, the levers buy psnr
and cost lpips online — the opposite sign to offline, where they bought lpips.
That part of §17.18 stands unexplained.

**The attribution is not yet made.** +1.31 dB over `old` could be the levers, or
could be the polish alone; without `old + polish` there is no way to tell, and
that control is running. Nothing above should be read as "the levers win with
polish" until it lands.

**Protocol note, to prevent a bad comparison later.** Offline and online absolute
numbers are NOT comparable: the offline desk baked map scores 12.4416/0.5027 and
the online one 10.7085/0.5588, because `main.py`'s viz bake path uses
`max_scale=1.0, min_conf=1.5` while `refine_local` builds from the kfgauss dump
with its own filters. Only within-protocol A/B is meaningful. Several apparent
contradictions in §17.18 shrink once this is respected — but not the lpips sign
flip, which is a within-protocol comparison on both sides.

### 17.20 Method note: what the proxy cost, and what caught it

§17.2-17.16 is roughly forty measured arms taken with `scripts/refine_local.py`
standing in for the online refiner. The proxy was chosen because it is fast,
deterministic and isolates one variable at a time — all true, and all still
true. It also produced a recommendation that **degraded the deployed system by
2.71 dB**, and nothing inside the proxy could have revealed that.

What the proxy silently assumed, none of it stated at the time:

1. **The map is complete from step 1.** Online it grows throughout, so a
   Gaussian injected late receives almost none of the ~104 in-sequence steps.
2. **Every Gaussian gets a comparable number of updates.** Online the
   distribution is wildly uneven and correlated with injection time.
3. **Supervision is uniform over the trajectory.** Online it is a recent ring
   (30%) plus a reservoir, so the last frames are over-weighted — and that
   weighting persists into the polish phase, where its original justification
   (a reservoir that evicts on long sequences) no longer applies.

Every one of the three levers assumes an optimizer that will actually run: the
band limit changes an initial footprint, freezing removes the ability to move,
`min_confidence=0` admits Gaussians that most need optimizing. Under assumption
(1) and (2) those assumptions fail together, which is exactly why the levers were
individually mild (−0.07, −0.54, −0.72) and jointly catastrophic (−2.71 against
a −1.33 sum).

**What caught it was running the thing itself, once, as a matched pair.** Not a
better proxy, not more arms, not review — a single A/B in the deployed loop, with
the baked maps scoring bit-identically as the control that proved SLAM was
deterministic and the refiner was the only variable.

Three rules this session earns:

- **A proxy validates a mechanism, never a configuration.** §17.2's lattice
  mechanism, §17.4's seam morphology and §17.17's depth bias all survive
  untouched, because they are statements about the map. Every statement about
  *what to set* had to be re-earned in the system.
- **State the proxy's assumptions when the proxy is chosen**, not when it fails.
  All three above were visible in `run_refiner`'s source the whole time.
- **Interactions are where proxies break.** The individual effects transferred
  in sign and roughly in size. Only the three-way term did not, and no amount of
  one-factor-at-a-time work in the proxy would have produced it.

The cost was real and should be stated plainly in any write-up: the online
half of §17 was wrong for about a day, was published to the user in that state,
and was corrected only because the confirmation run was taken rather than
skipped.

### 17.21 The control lands: it was the polish, not the levers

```
desk, online, refined map        steps    psnr     lpips
baked (unrefined)                  —    10.7085   0.5588
old defaults, in-sequence         104   12.0971   0.5446
NEW defaults, in-sequence         104    9.3865   0.7327
NEW defaults + 900 s polish      1208   13.4039   0.6055
old defaults + 900 s polish      1581   14.0640   0.4247   <- best on both
```

**`old + polish` beats `new + polish` on both metrics** — +0.66 dB and lpips
0.4247 against 0.6055, a 30% gap. Both arms got the same 900 seconds, so this is
a matched-wall-clock comparison, which is the one a deployment decision needs.

That retracts §17.19's reading. The +1.31 dB that `new + polish` showed over
`old` in-sequence was **entirely the polish**; none of it was the levers. Given
the same steps, the old settings win decisively. §17.19 said the attribution was
not yet made and that nothing should be read as "the levers win with polish" —
that caution was correct and the answer is that they lose.

**Final verdict on the three quality levers: they work offline and do not
transfer online.** Forty-odd offline arms could not have shown this; two online
runs did.

#### What the online work actually delivered

Not a parameter change — a scheduling one:

```
old, in-sequence only    12.0971 / 0.5446
old + 900 s polish       14.0640 / 0.4247      +1.97 dB, -22% lpips
```

**`--refiner-polish-secs` is the fix for the online lpips regression.** At 0.4247
it is 24% better than the *unrefined* baked map (0.5588), and it is the only
online configuration measured that improves lpips substantially at all. The
regression recorded in `docs/online-eval-all-families.md` for room and 360 was
never a parameter problem: **104 in-sequence steps on a 2.4M-Gaussian map is not
enough optimization, and the answer is more steps, not different settings.**

This also lands squarely on the project's actual goal. Real-time tracking with a
short post-sequence polish is a more honest shape for "end-to-end real-time
online reconstruction" than trying to force convergence inside the sequence:
tracking stays real-time and untouched, and the map reaches a quality the
in-sequence budget provably cannot buy. §16.11's arithmetic said the last 3-4x
had to come from needing fewer steps rather than faster ones; this says the
alternative is simply to take the steps afterwards.

#### Caveats

One sequence (desk), one seed, one polish duration. `new + polish` got 1208
steps against `old + polish`'s 1581 because its map is 12% larger — matched in
seconds, not in steps, which is the right axis for deployment but means the
lever comparison also carries a step-count difference. The lpips gap (30%) is far
larger than 31% more steps would plausibly explain, but that is an argument, not
a measurement.

### 17.22 Review of the self-refutation: a better mechanism, and four missed signals

**A better explanation for the lpips half, and my hypothesis loses a control.**

I proposed that the online recent-ring (30% of samples drawn from the last 64
frames) over-fits the sequence tail during polish. Kimi's objection is a control
I had in hand and did not use: **both arms run the identical `recent_frac=0.3`
schedule, and `old + polish` improved lpips by 22% under it.** A schedule shared
by both arms cannot by itself create the asymmetry.

Its alternative: **freezing the geometry manufactures multi-view colour conflict,
and L1's answer to conflict is the conditional mean — psnr-optimal and
lpips-poison.** Gaussians whose geometry is wrong (junk tails, the −6%
systematic bias of §17.17, streaks) land on different pixels in different
supervision views and are asked to be different colours. With the means free
that conflict is resolved by moving the Gaussian until it is view-consistent;
with them frozen it is unresolvable, and L1 converges to the average of the
observations — desaturated and texture-averaged.

**The psnr/lpips decoupling is itself the signature.** `new + polish` recovering
psnr to 13.40 (a per-pixel conditional mean is exactly MSE-optimal) while lpips
stays pinned at 0.6055 (an averaged fake is exactly what lpips punishes) is what
conflict-averaging predicts, and it explains why `old` escapes: free means make
the geometry view-consistent and the conflict never forms.

Tests, in Kimi's priority order:

1. **Zero-cost**: decompose held-out lpips per frame along the trajectory. My
   recent-ring hypothesis predicts damage concentrated at the tail; the
   conflict-averaging one predicts it in hole/junk regions, time-independent.
2. **Decisive arm**: `new + polish` but with the means UNFROZEN during the polish
   phase only (keeping conf 0 and the band limit). Predicted lpips recovers to
   <= 0.48, eating >= 60% of the 0.18 gap. Unchanged means the explanation is
   falsified.
3. My recent_frac=0 arm — worth one flag, predicted to recover <= 0.02-0.03.
4. **The constructive version**: weight polish-phase supervision by
   *under-training* (per-Gaussian update counts are trackable; late-injected
   regions are under-trained) rather than by recency. If the diagnosis is step
   allocation, medicate the allocation directly.

#### Four signals that were visible when the proxy was chosen

Beyond the one self-review found (the three assumptions are written plainly in
`run_refiner`, read but never listed as a difference table):

1. **The proxy's regime never overlapped the deployment regime on the binding
   axis.** §16 established 86 in-sequence steps. Every offline arm ran >= 300.
   A proxy that does not cover the deployment point on the tightest axis is not
   a proxy, it is a different experiment — and that number was already ours.
2. **The proxy was never asked to reproduce the one online fact on hand.** The
   lpips regression (room +11.1%, 360 +20.6%) was a real online measurement.
   A proxy that cannot reproduce a *known* online phenomenon has no standing to
   predict new ones. **A cross-system paired calibration run should have been
   experiment #1, not #41.** This is more actionable than "a proxy validates a
   mechanism, never a configuration": *before any sweep, run one shared
   configuration through both systems and pair them; if they do not pair, fix
   the proxy first.*
3. **Lever-category analysis, available by pure reasoning.** Every lever swept
   acts on injection or parameterization, so its value is *by definition* a
   function of what happens after injection — which is exactly what the proxy
   abstracts away by starting with a complete map. The alignment between the
   lever category and the proxy's blind spot was derivable without running
   anything. Conversely a *scheduling* lever acts on a complete map and is
   naturally robust to the gap, which is why the one that survived was polish.
4. **Institutionalize the difference table.** Make "proxy vs target system
   differences" a required field: each difference either argued irrelevant to
   the quantity being measured, or priced with the cheapest test that would rule
   it out. The budget of any single one of the forty arms would have paid for
   the most expensive item on that list.

#### The ordering this session actually established

**Scheduling >> map content > parameterization.** Polish is worth +1.97 dB and
−22% lpips; density (`min_confidence`) moves things by tenths of a dB with the
sign depending on budget; the parameterization levers do not transfer online at
all. And every reordering in this section was caused by the wall-clock
constraint moving — exact culling's 9x is what made matched-steps the right
comparison, which is what inverted §16.9, which is what made polish affordable
enough to be the answer.

### 17.23 Zero-cost discrimination: my hypothesis falsified, Kimi's survives

Per-frame held-out lpips in trajectory order, quartile means, on the two
already-archived polish maps — no new runs (`eval_map_quality --per-frame`):

```
desk            Q1 (earliest)   Q2       Q3      Q4 (latest)
new + polish       0.5909     0.6626   0.6026    0.5717
old + polish       0.3756     0.4634   0.4642    0.3998
deficit           +0.2153    +0.1992  +0.1384   +0.1719
```

**My recent-ring hypothesis predicted the deficit concentrated at the tail. Q4
has the smallest deficit and Q1 the largest — the opposite. Falsified.**

Kimi's conflict-averaging hypothesis predicted a deficit uncorrelated with time.
Measured: 0.138 to 0.215 across quartiles, no tail concentration. **Consistent —
though consistent is not confirmed.**

(Both arms share the same *shape*, Q2/Q3 worst and Q1/Q4 best. That is a property
of the desk sequence, not of either configuration; only the deficit between them
is attributable, and it is roughly flat.)

This is what a zero-cost discriminator is worth: two competing explanations, one
eliminated, on data already on disk. It should have been run before either
explanation was written down.

**The decisive arm is built** (`--refiner-unfreeze-in-polish`, with a shared
`polish_flag` because the refiner is a separate process and cannot otherwise
know the polish phase has begun). Pre-registered: Kimi predicts lpips recovers
from 0.6055 to **<= 0.48**, eating >= 60% of the 0.18 gap. If it does not move,
conflict-averaging is falsified and there is no third hypothesis on the table.

### 17.24 360: the online lpips regression is fixed, on the sequence that had it

Everything about polish so far was measured on desk, which never had the
regression. 360 did. Same run, scored against its own baked map:

```
360, online                steps     psnr     lpips
baked (this run)             —     11.5692   0.5021
+ 1800 s polish            3821    13.8664   0.4605     +2.30 dB, -8.3% lpips

for comparison, docs/online-eval-all-families.md (in-sequence only):
baked 11.9973 / 0.4841  ->  refined 12.1413 / 0.5838     +0.14 dB, +20.6% lpips
```

**The regression inverts: +20.6% worse becomes 8.3% better, and psnr goes from
+0.14 dB to +2.30 dB.** Measured on the sequence that had the defect, with no
parameter change at all — the refiner ran default settings and was simply
allowed to keep going after the sequence ended.

The step counts say why: **3821 against 114 in-sequence, a factor of 33.**
§16.11 computed that 3000 steps is what turns 360 net-positive and that
in-sequence only buys 86, concluding the remaining 3-4x had to come from needing
*fewer* steps. The answer turned out to be simpler: **those 3000 steps do not
have to be taken during the sequence.** Tracking stays real-time and untouched;
the map gets its budget afterwards.

That also settles what the whole §17 online investigation delivered. Not the
three quality levers, which do not transfer (§17.21). A scheduling change worth
+2.30 dB and −8.3% lpips on the hardest sequence, and +1.97 dB / −22% on desk.

### 17.25 All three sequences, closed

```
              baked            + polish         steps    d_psnr   d_lpips
desk     10.7085 / 0.5446   14.0640 / 0.4247    1581     +1.97    -22.0%
360      11.5692 / 0.5021   13.8664 / 0.4605    3821     +2.30     -8.3%
room     10.4264 / 0.5813   12.6417 / 0.4890    1856     +2.22    -15.9%

historical, in-sequence only (docs/online-eval-all-families.md):
360      11.9973 / 0.4841 -> 12.1413 / 0.5838            +0.14    +20.6%  (worse)
room     11.1462 / 0.5418 -> 11.4506 / 0.6019            +0.30    +11.1%  (worse)
```

**Both sequences that regressed now improve, and by more than an order of
magnitude on psnr.** No parameter was changed: the refiner runs its default
settings and is simply allowed to continue after the sequence ends.

This is the final answer to the question that opened §17. The visible defects
split into three groups and each got a different verdict:

- **The dot lattice** was a real representation defect, diagnosed with a metric
  built for it (self-consistency, `hp_alpha`), fixed by a band limit read off
  the Gaussian lattice — **offline**. It does not transfer online, and the
  reason it does not is §17.21.
- **The lpips regression** was never a parameter problem. It was **104 steps on
  a multi-million-Gaussian map**, and the fix is scheduling.
- **The seams and streaks** are novel-view phenomena that the evaluation
  protocol is structurally blind to (§17.17), and no lever moved them because
  no metric can see them. The metric, not another fix, is the next step.

And the ordering, which is the transferable lesson: **scheduling >> map content
> parameterization**, with every reordering caused by the wall-clock constraint
moving.

### 17.26 A metric that can see the veil, and two more falsified predictions

`scripts/diag_flythrough.py`. Around each held-out pose, dolly the camera and
backward-warp frame t into t+1 through t+1's *rendered* depth. A veil is two
surfaces at different depths whose colour comes from the far one and whose
geometry comes from the near one, so **colour motion inconsistent with depth
motion** is its defining signature, and a warp finds it without optical flow.
Needs no ground truth — a self-consistency measurement, like §17.2's 2x test.

**Self-test first, before any use.** Sweeping the dolly:

```
dolly     disparity    warp     static-shift   ratio
+-1 cm      1.8 px    0.1330      0.0340       3.91
+-5 cm      8.2 px    0.4083      0.2254       1.81
+-15 cm    20.6 px    0.5375      0.4248       1.27
```

The ratio falls monotonically with disparity, which is the qualitative behaviour
depth-informativeness must produce, so the metric is not broken. **But it carries
a large instrument noise floor**: at 1.8 px a rigid shift is nearly the identity
(0.034) while the warp costs 0.133. That floor is the alpha-weighted mean depth
averaging across depth discontinuities, plus resampling — not the veil. So the
absolute ratio means little; **the metric is a comparator between maps at a
matched dolly**, where the floor cancels.

As a comparator it resolves (desk, +-5 cm):

```
baked           1.3844
new + polish    1.4090
old + polish    1.8115
```

**It orders the maps opposite to psnr/lpips.** `old + polish` is the best map on
both photometric metrics (14.0640 / 0.4247) and the worst on warp consistency;
the *unrefined* baked map is the best. Two readings, not yet separated:

- **(a) a finding**: photometric refinement improves held-out psnr/lpips while
  degrading the map's geometric self-consistency, because the optimizer moves
  Gaussians to fit supervision views in ways that are not physical. That would
  explain the whole complaint that started §17 — numbers rising while the GUI
  looks worse.
- **(b) an artifact**: the noise floor may couple to geometric complexity, and a
  more optimized map has more fine structure.

**Do not adopt (a) yet.** When a new metric first produces a counter-intuitive
ordering, the metric is the suspect. The discriminator is to re-run with TUM's
*sensor* depth in place of the rendered depth: if the ordering flips, the depth
render is at fault; if it survives, (a) stands.

**Kimi's decisive prediction is falsified too.** `--refiner-unfreeze-in-polish`
restores the positional rate once the map is complete:

```
                                  steps    psnr     lpips
new + polish, frozen throughout    1208   13.4039   0.6055
new + polish, unfrozen in polish   1371   13.6337   0.5713
old + polish, never frozen         1581   14.0640   0.4247
```

Recovery is 0.0342 of the 0.1808 gap = **19%, against a pre-registered >= 60%**.
Direction right, magnitude three times short. So conflict-averaging is part of
the mechanism and not the main part.

**Both hypotheses for the online lpips deficit have now failed**: mine (recent-ring
over-fitting) was falsified by the quartile decomposition in §17.23, and Kimi's
(conflict-averaging under frozen geometry) explains 19%. **81% of the deficit is
something neither of us proposed**, and there is no third hypothesis on the
table. That is the honest state, and it is worth more in the record than a
plausible story would be.

### 17.27 Is it unfixable? The distinction that settles it

The question put to Kimi was whether the seams and streaks admit *any* method.
Non-existence cannot be proved, so the deliverable was an exhaustive taxonomy
with each branch either killed by a measurement or priced. The framing that came
back is sharper than the taxonomy, and it decides the question:

**Disagreement defects vs absence defects.**

- **A seam/veil is a disagreement defect.** The map holds two mutually
  contradictory measurements of one surface. Removing a contradiction requires
  **no new information** — only a term that notices the contradiction. So it is
  fixable in principle, and the reason nothing has worked is precise: *the signal
  that would fix it is not in the loss we have been using.* Every lever tried so
  far optimizes agreement with supervision views, and the supervision views
  cannot see a 0.66 px disagreement (§17.17).
- **A streak over single-coverage geometry, and unmapped black, are absence
  defects.** The content is not in the map and not in the supervision signal, and
  with densification off nothing will ever grow it. **That is missing
  information, not an inadequate method.** No optimizer reaches it; only a prior
  (network inpainting) or new data (sensor, re-observation) can.

**Streaks therefore split in two**: those with correct coverage behind them are
fixable by suppressing the front layer (untested, see below); those without are
impossible in principle and only keyframe-selection prevention applies.

#### Impossible in principle, under this architecture's three pillars

Stated with the scoping that makes each precise rather than defeatist:

1. **Repairing absence without a prior or new data.** Information is not there.
2. **Learning inter-sample appearance from supervision at the trajectory's own
   sampling rate.** A defect in the null-space intersection of the loss and the
   parameterization is unreachable — a sampling-theorem argument, already
   demonstrated twice (`--ss-loss` inert, psnr flat to +-0.01 dB across the tau
   sweep). **Scoped**: impossible *with the existing data and existing
   parameterization*. Adding a prior or changing the parameterization moves the
   defect out of the null space, which is exactly why the consistency-loss
   branch is not condemned by this — it is the reason that branch exists.
3. **Photometric per-cluster geometric correction coexisting with free loop
   closure.** Pose error and depth error are photometrically degenerate, so a
   learned offset carries a pose component that gets re-applied after a
   correction. **Scoped**: impossible *from photometry alone*; an external metric
   reference (sensor depth) breaks the degeneracy — map-vs-sensor is network
   error, sensor-vs-trajectory is pose error.
4. **Making native psnr/lpips see either defect.** 0.66 px is structural. This is
   not a repair candidate at all; it is the precondition for discussing repairs.

#### Untested, with prices and predictions

Kimi's ranked three, each with a falsification line:

```
1. sensor-depth injection (map side, tracking untouched)
   new metrics: per-kf spread 9.6% -> <=2.5%, seam step -50..-70%,
                warp deficit -40..-60%
   old metrics: psnr +0.05..0.25, lpips +-1%
   falsified if: spread falls 4x but seam step falls <30%
                 -> seams are not depth-spread dominated, look at pose/alpha
   (this is the discriminator AND the upper bound: it says how much of the seam
    belongs to depth at all, whether or not the sensor route is ever adopted)

2. cross-cluster consistency loss (colour + confidence-weighted alpha
   competition; compatible with frozen means)
   new metrics: seam step -40..-60%, warp deficit -20..-35%, hp_alpha +<10%
   old metrics: +-0.05 dB / +-1% -- NOT MOVING IS THE WIN, it demonstrates this
                class of repair is metric-neutral by construction
   falsified if: pairwise colour residual narrows but seam step does not
                 -> the seam-step metric is itself wrong

3. streak opacity soft-deletion  o <- o * min(1, k*dx/s_ray),  k ~ 1.5
   new metrics: edge-region warp deficit -20..-40%; black fraction +<0.5%
   falsified if: black rises >1% -> "coverage behind" is false on our maps,
                 fall back to keyframe baseline-threshold prevention
```

#### The answer to the question as asked

**The only thing this architecture truly forbids is creating information.
Everything else is engineering.** So:

- **Seams: not unfixable.** The signal that would fix them is not in the loss
  being used, and two untested branches address exactly that.
- **Streaks: half unfixable in principle** (single-coverage — the information is
  absent), **half simply untested** (multi-coverage — one injection-time line,
  never run).
- **Black: 47% already recovered and measured** (§17.16); the remaining 3.76% is
  content the camera never observed, which is absence and therefore final.

Declaring the seams unfixable today would be reporting the failure of a
measurement as a property of the object — and §17.26 makes that concrete: the
fly-through metric is one hour old and has already produced an ordering that
psnr and lpips cannot see.

### 17.28 The depth upper bound, and a confound caught before it was reported

`scripts/diag_sensor_depth.py` rescales every Gaussian along its own view ray to
match TUM's depth sensor, leaving direction, covariance, colour and opacity
alone, so **the only quantity changed is depth**. Not deployable — the SLAM
system is monocular — but it answers the one question tuning cannot: if every
depth were right, how much veil would remain?

A free cross-check first: the per-keyframe sensor/prediction ratio comes out at
median **1.0500** on desk, against §17.17's independently measured −4.8% bias
(1/0.952 = 1.050). Two different code paths, same number.

**First run, and why it does not answer the question.**

```
             warp     static   ratio     psnr      lpips
pred        0.3856   0.3056   1.2618   12.3544   0.5071
sensor      0.3735   0.3356   1.1129   11.0903   0.5579
```

Warp deficit fell 3.1% against a pre-registered −40 to −60%, and psnr fell
1.26 dB. Read naively that is a decisive falsification of the depth hypothesis.
**It is not, because the arm has a confound larger than the effect it looks
for.** The trajectory was estimated from the same pointmaps, so the poses and
the predicted depths **share a scale and are mutually consistent**. Scaling the
depths by 1.05 while leaving the poses alone breaks that consistency and
displaces every cluster — which is what the −1.26 dB is.

Seams are a **relative** misplacement between clusters, so the signal is the
per-keyframe deviation from the median (std 9.4%), not the global factor; and
evaluation's umeyama alignment absorbs the global part regardless. The corrected
arm divides `GLOBAL_MED` out and applies only the relative correction.
`--global-scale` keeps the confounded version as a control, with the −1.26 dB
recorded in its help text.

**This is the third instance of the same error this session**, and the pattern is
worth naming because it is not obvious in the moment:

- exposure re-centring: folding a render-space transform into per-Gaussian
  colour is only equivalent where accumulated alpha is 1 (§17.8)
- anisotropy clamping: shortening the long axis *is* shrinking the footprint,
  and the band limit only floors the smallest scale (§17.16)
- sensor depth: rescaling depth alone breaks a scale shared with the trajectory

Each time an intervention was designed to change one physical quantity and
silently changed a second one that dominated the measurement. **The missing
habit is asking "what else does this change?" before running, not after a
surprising number.** Cheap to institutionalize: for any intervention, write down
the quantities it touches directly and the invariants it might break, and check
whether any of them is coupled to the metric more strongly than the target.

The corrected arm's verdict, whichever way it falls, is what §17.27's
experiment #1 was for: it decides whether the next effort goes to depth
consistency (a cross-cluster loss, fine-tuning the −6% bias) or to pose and
alpha competition. Both directions are live; the point of the experiment is that
psnr cannot distinguish them and this metric can.

### 17.29 The depth upper bound is not measurable on TUM fr1

Four independent parameterizations of "correct the depth toward the sensor",
desk, all against the same `pred` baseline:

```
                     warp     ratio     psnr      lpips
pred (unchanged)    0.3856   1.2618   12.3544   0.5071
global scale        0.3735   1.1129   11.0903   0.5579
relative per-pixel  0.3903   1.1101   11.0663   0.5725
per-keyframe scalar 0.4218   1.2555   11.6556   0.5108
global affine curve 0.4091   1.2859   11.5701   0.5743
```

**All four degrade both metric families.** The pre-registered falsification line
— "spread collapses but warp deficit falls < 30% -> seams are not
depth-dominated" — **cannot be applied**, because its premise was never
established. Four arms demonstrate the depth was made *different*, not more
correct, and every difference hurt.

Why, and it is not subtle in hindsight: TUM fr1's depth comes from a
structured-light sensor at a **different viewpoint from the RGB camera** (a
registration offset), with substantial noise and missing returns; and the
trajectory and the map were **co-estimated from the network's own pointmaps**, so
the network's depth is *self-consistent* with everything else in the system in a
way an external measurement is not. Importing an unregistered external
measurement into a self-consistent system destroys more consistency than it
repairs.

**So the upper bound is not measurable on this dataset.** Kimi's experiment #1
is off the table here — not because the answer came back negative, but because
the instrument is inadequate. A valid version needs a depth reference that is
registered to the RGB camera, low-noise, and consistent with the trajectory;
TUM fr1 supplies none of the three.

The premise-check that produced this is worth more than the arms:

```
corr(per-keyframe sensor/pred ratio, that keyframe's median scene depth) = +0.736
54% of the ratio's variance is explained by scene depth alone
std after removing a linear scene-depth trend: 0.0940 -> 0.0637
```

**Over half of §17.17's "9.6% per-keyframe spread" is a range-dependent bias, not
per-cluster randomness.** That is a correction to §17.17, which used that spread
to support the per-cluster-misplacement reading of the seams: **the support is
substantially weaker than written**. It also explains all four failures at once —
a per-keyframe correction driven by scene content scatters clusters instead of
aligning them.

#### What this does to the ranking

It promotes Kimi's #2 and the reason is now stronger than when it was ranked.
**The cross-cluster consistency loss needs no external truth at all.** It does
not need to know the correct depth; it only needs to notice that two clusters
covering one surface disagree. After four failed attempts to import external
truth into a self-consistent system, an internal-consistency term is not merely
the next option — it is the only branch that does not depend on something this
dataset cannot supply.

#### Method note

Three of these four arms were run *because the previous one was invalid*, each
time caught by asking what else the intervention changed:

```
arm 1  confounded by a scale shared with the trajectory
arm 2  confounded by sensor shape noise and RGB-depth misregistration
arm 3  confounded by a premise (the 9.4% spread) that had never been checked
arm 4  valid in construction, and it revealed that the instrument itself is
       the problem
```

Not one of the negatives was ever evidence about the seams. **Every negative
result in this section was about the measurement, not the object** — which is
precisely why "the seams are unfixable" could not have been concluded from any
of them.

### 17.30 The cross-cluster consistency loss, built and first-measured

`cross_cluster_loss` in `refiner.py`. Voxel-hash the composed world positions;
keep only voxels containing Gaussians from **two or more distinct keyframes**;
penalize the within-voxel variance of position and colour. Gradient reaches
`kf_log_depth` — sliding a whole cluster along its own view rays — and `means`
if they are free.

Three properties make this the branch that survived §17.29:

- **No external truth.** It never asks what the correct depth is, only whether
  two clusters covering one surface disagree. Four attempts to import sensor
  depth into this self-consistent system all degraded it (§17.29).
- **No new information.** A seam *is* a disagreement, and removing a
  contradiction is free in the information sense (§17.27).
- **It never looks through a camera.** The comparison is in 3D, which is the
  only way past the structural blindness of a 0.057 m supervision baseline to a
  0.66 px error (§17.17).

It is also the signal `--kf-depth-lr` was missing. §17.16 measured that
parameter finding a real signal and buying +0.003 dB, because photometry cannot
tell it *which way* to slide.

Explicitly not voxel dedup, which deleted 8% and changed nothing (§16.6):
deleting picks one of two disagreeing measurements, this moves them toward each
other.

**First measurement, desk, 300 steps, scored by the fly-through metric:**

```
          psnr     lpips   self-psnr  hp_alpha   warp     ratio
noc     13.8114   0.4199    37.64     0.00461   0.3846   1.4947
cons    13.7014   0.4166    38.40     0.00445   0.3807   1.5059   w=0.1
cons_hi 13.7452   0.4207    38.64     0.00463   0.3720   1.5068   w=1.0
```

Warp deficit −1.0% and −3.3% against a predicted −20 to −35%. The `kf_depth`
scales do move (std 0.0146 with the term, against 0 without), so the gradient is
real; the effect is not.

**And the voxel size was chosen by the wrong scale argument.** The default 0.02 m
was picked as "a few times the lattice pitch (~2 mm)". But the term compares two
*disagreeing* clusters, and a veil sits **4-10 cm** in front of the true surface
(§17.16's ±2-5% at 2 m). **At 2 cm the two layers fall in different voxels and
are never compared.** The relevant scale is the disagreement scale, not the
sampling scale.

That is the fifth instance in this section of a parameter or an intervention
chosen by a plausible-sounding argument about the wrong quantity (§17.28 lists
the first four). A voxel sweep at 5/10/20 cm is running.

**Status, stated precisely.** This route is neither confirmed nor refuted: its
first arm was mis-parameterized, so its negative is about the experiment, like
the four before it. What *has* changed is that the route is now **decidable** —
the fly-through metric separates maps by 0.372-0.422, far above seed noise, so a
null result at the right voxel scale would be **the first negative in this whole
investigation that is about the object rather than the measurement.** Only then
does "unfixable" become discussable, and even then the falsification line points
at the next suspect rather than at a dead end: pose error and alpha competition
between overlapping clusters, an entire branch never touched.

### 17.31 Six negatives, one structure: name the failure mode

The voxel sweep closed at −4.3% and saturating across a 10x range (0.02 -> 0.20 m),
which looked like the first negative about the object. It is not. The loss
penalized **total** within-voxel variance, and total = within + between. The
within part is legitimate surface structure inside one cluster — texture,
curvature — and it dominates, diluting the signal. **A seam is two clusters'
MEANS disagreeing**, so the quantity is the per-(voxel, keyframe) mean and the
loss is their spread. Fixed to a proper between-group variance; arms running.

That is the **sixth** time in this section a negative result turned out to be
about the experiment. The structure is identical every time, and worth naming
because knowing it after the fact six times is not the same as having a step
that catches it before:

```
                    nominal target              what it also changed
exposure recentre   fold mean exposure back     only equivalent where alpha = 1
anisotropy clamp    remove needle Gaussians     shrank the footprint
sensor depth x3     make depth correct          broke self-consistency /
                                                injected sensor shape noise /
                                                applied a scene-content scale
consistency voxel   find two clusters on one    sized off the sampling scale, so
                    surface                     the two layers never co-occur
consistency formula penalize cluster disagreement  penalized within-cluster
                                                   surface structure
```

**An intervention nominally aimed at quantity A also moved quantity B, and B was
more strongly coupled to the metric than A.** Every one is a single sentence in
hindsight and none was asked in advance.

The missing step is cheap and mechanical, and it goes in the protocol next to
§17.22's difference table:

> Before running an intervention, write down (1) the quantities it changes
> directly, (2) the invariants it might break, (3) the scale at which the target
> effect lives, and check whether (2) or a mismatch in (3) couples to the metric
> more strongly than (1).

Applied retroactively, that question catches all six in under a minute each.

**And it bears directly on the question this whole line was asked to answer.**
In an investigation whose experiment design has been defective six consecutive
times, **no conclusion of the form "this defect is unfixable" can be about the
defect.** That is not caution; it is what the record says. The parts that *are*
settled were settled by arguments about information (§17.27's absence defects)
and by arithmetic (§17.17's 0.66 px), never by an accumulation of failed arms.

### 17.32 The streaks: a working lever, after two wrong forms

The anisotropy clamp failed at every setting (§17.16) and the reason turned out
to be the intervention's form, not the target: shortening the long axis *is*
shrinking the footprint, and the band limit only floors the smallest scale.
Kimi's alternative keeps the footprint and lowers the opacity instead, so the
surface behind shows through:

```
o <- o * min(1, K * pitch / max_scale)
```

Two form choices that matter, both learned from earlier failures:

- **Opacity, not scale.** Hiding a streak is free; shrinking it opens holes.
- **Elongation measured against the LATTICE PITCH, not against the other axes.**
  A Gaussian is a streak when it is long compared with the surface sampling it
  belongs to (§17.2's quantity), which is not the same as being anisotropic.

desk, 300 steps, on top of the recommended offline config:

```
K       lpips     psnr      black    d_black
0      0.4199   13.8114    3.59%       —
4.0    0.4193   13.8101    3.58%     -0.01
1.5    0.4116   13.8108    3.69%     +0.10
1.0    0.4062   13.7970    3.77%     +0.18
0.5    0.3989   13.7493    3.90%     +0.31
```

**Monotone: −5.0% lpips at K=0.5, for −0.062 dB psnr and +0.31 pp of black.**
That is the best perceptual result on desk offline in this whole section
(0.3989 against a 0.4199 baseline and a previous best of 0.4155).

**The pre-registered risk check passes.** Kimi predicted "black fraction
+<0.5%"; measured +0.31 pp, and the falsification line (>1%, meaning nothing is
behind the streaks after all) is not approached. So on these sequences the
"coverage behind" premise is **confirmed**, which is what makes hiding legitimate
rather than a trade of one artifact for a worse one. That check was declared
before the run and was worth declaring: −5.0% lpips is a large enough number to
discourage looking for its price.

Its predicted *mechanism* was wrong, though: "edge-region warp deficit −20 to
−40%" did not appear (the whole-frame warp metric moved +1.6%, diluting a
localized effect), while the lpips gain was not predicted at all. Right lever,
wrong account of why.

**Recommended: `--streak-opacity 0.5` in the offline configuration.** Not wired
into `run_refiner`, because §17.21 established that offline quality levers do
not transfer online and nothing here has been tested in that loop.

#### The streak line, closed

```
absent-coverage streaks     impossible in principle -- information is not there,
                            prevention (keyframe baseline threshold) only
covered streaks             -5.0% lpips, measured, cheap, one injection line
anisotropy clamping         harmful at every setting, do not revisit
```

### 17.33 The online table, extended (in progress)

Post-sequence polish, default refiner settings, scored against each run's own
baked map. Four of seven sequences:

```
seq     baked            + polish          steps   d_psnr   d_lpips
desk  10.7085 / 0.5446  14.0640 / 0.4247   1581    +1.97    -22.0%
360   11.5692 / 0.5021  13.8664 / 0.4605   3821    +2.30     -8.3%
room  10.4264 / 0.5813  12.6417 / 0.4890   1856    +2.22    -15.9%
xyz   12.7732 / 0.4311  14.8142 / 0.3050   2089    +2.04    -29.3%
```

**psnr improves by +2.0 to +2.3 dB on every sequence**, across maps differing by
4x in size and trajectories as different as a desk pan and a full rotation. That
consistency is the stronger evidence: it says polish is not fixing some
sequence-specific pathology but a systematic under-optimization. lpips varies far
more (8-29%) and is loosely anti-correlated with difficulty — 360, the hardest
(7.27M Gaussians, rotation), gains least.

**A scheduling observation worth more than the table.** The steps each sequence
needs span **2.4x** (1581 to 3821), and that spread does not track map size:
desk (1.86M) and room (7.3M) differ 3.9x in size but only 1.17x in steps, while
360 needs 2.4x desk's. So **"budget by map size" is the wrong rule**, which is
the case for the adaptive termination built as `--refiner-polish-tol` (default
off, untested — after §17.21, nothing ships on an offline-looking argument).

### 17.34 Seven sequences, and a correlation that predicts the gain

All seven, post-sequence polish, default settings, each scored against its own
baked map:

```
seq     baked            + polish          steps   d_psnr   d_lpips
desk  10.7085 / 0.5446  14.0640 / 0.4247   1581    +1.97    -22.0%
360   11.5692 / 0.5021  13.8664 / 0.4605   3821    +2.30     -8.3%
room  10.4264 / 0.5813  12.6417 / 0.4890   1856    +2.22    -15.9%
xyz   12.7732 / 0.4311  14.8142 / 0.3050   2089    +2.04    -29.3%
rpy   10.0437 / 0.5719  11.1245 / 0.5230   2375    +1.08     -8.6%
plant  9.8383 / 0.5919  12.6711 / 0.4954   1445    +2.83    -16.3%
teddy 10.2469 / 0.6133  13.2735 / 0.5138   1945    +3.03    -16.2%
mean                                               +2.21    -16.7%
```

**Seven of seven positive on both metrics.** No parameter changed; the refiner
runs defaults and is allowed to continue after the sequence ends.

#### The perceptual gain is determined by translational parallax

Ordering the sequences by median inter-keyframe translation divided by median
inter-keyframe rotation (m/deg):

```
seq    m/deg   d_psnr   d_lpips
rpy    0.184   +1.08     -8.6%
360    0.571   +2.30     -8.3%
room   0.866   +2.22    -15.9%
teddy  1.115   +3.03    -16.2%
plant  1.126   +2.83    -16.3%
desk   1.830   +1.97    -22.0%
xyz    2.718   +2.04    -29.3%

lpips gain vs m/deg      Spearman -0.964   Pearson -0.978
psnr  gain vs m/deg      Spearman +0.071
control: lpips gain vs baked lpips        +0.143
control: baked lpips vs m/deg             -0.214
```

**The perceptual benefit of refinement is almost perfectly predicted by how much
translational parallax the trajectory has, and psnr is blind to it.** The two
controls rule out the obvious confounds: it is not "worse maps improve more", and
low-parallax sequences do not start out worse. This is the third instance in §17
of psnr and lpips decoupling, and the sharpest.

#### The mechanism, and the literature's correction to it

`tracker.py:29` — `keyframe = self.keyframes.last_keyframe()`, then
`splatt3r_match_asymmetric(model, frame, keyframe)`. **Splatt3R is always fed
(current frame, most recent keyframe)**, a temporally adjacent pair. Under
rotation-dominated motion that pair has almost no translational baseline.

The naive reading — "rotation breaks the system" — is wrong, and the literature
says why. MASt3R-SfM handles pure rotation *better* than classical SfM for pose
(COLMAP and VGGSfM fail outright; it reaches 100% on some scenes), and our own
ATE on rpy/360 is fine. What actually degrades is **depth**: with a vanishing
baseline triangulation is ill-defined, so depth falls back on the monocular
prior, and prior-driven depth is systematically worse. Geometry is worse,
photometric refinement improves the fit but not the perception. That is exactly
the measured pattern.

#### Where the field has gone, and what it implies

- **Two-view is obsolete as a backbone.** DUSt3R/MASt3R-style pairwise
  prediction has been superseded by N-view feed-forward models — VGGT, Fast3R,
  Pi3, CUT3R, MUSt3R — that ingest all views in one pass, with VGGT-SLAM and
  VGGT-SLAM++ already built on them. **Splatt3R's two-view backbone is the
  architectural source of the correlation measured above.**
- **Baseline-aware keyframe selection is already practice**, not a contribution:
  MCGS-SLAM scores keyframes on covisibility + baseline span + motion stability
  jointly. That makes the obvious intervention here low-risk and unoriginal,
  which is useful to know before spending on it.
- **`SparseSplat`'s pixel-unaligned prediction** targets the root of §17.2's
  lattice defect architecturally — letting the network place Gaussians rather
  than pinning one per pixel — where the band limit is a post-hoc patch.
- **A tension worth resolving**: Flash-Mono reports a 10x speedup by *bypassing*
  the per-frame optimization that GS-SLAM normally needs, while §17.25 measures
  that optimization as worth +2.21 dB / −16.7% lpips. Either its multi-frame
  feed-forward quality is high enough not to need it, or the comparison is not
  like-for-like. Which it is decides whether polish is a contribution or a
  workaround for a two-view backbone.

Sources: VGGT/Fast3R/CUT3R/MUSt3R survey (arXiv 2508.11379, ScienceDirect
S2096579626000203), VGGT-SLAM++ (arXiv 2604.06830), Flash-Mono (arXiv 2604.03092),
SparseSplat (arXiv 2604.03069), MCGS-SLAM (arXiv 2509.14191), MASt3R-SfM
(arXiv 2409.19152).

### 17.35 The colour leg is independent: the unified hypothesis loses a third

§17.34's low-parallax root suggested a unification: seams are clusters at wrong
depth, streaks are the network's expression of depth uncertainty at low
baseline, and the per-view affine that `--exposure` absorbs might be the
photometric shadow of the same geometric defect — a render darkened by
incomplete coverage, by an amount that varies per view.

`scripts/diag_exposure_cause.py` decides it at zero cost: freeze the entire map
and fit ONLY the per-frame affine, so the fitted gain is by construction that
frame's best global affine, then correlate against that frame's rendered
coverage.

```
                          desk              360
learned gain        0.9300 +- 0.0454   0.9402 +- 0.0976
per-view mean alpha 0.9475 +- 0.0587   0.9383 +- 0.0853
corr(gain, 1/alpha)     -0.355            -0.349
gain * mean_alpha        0.8822            0.8870
residual std after dividing by 1/alpha
                         0.0806            0.1463   (raw 0.0454 / 0.0976)
```

**Falsified three ways at once:**

1. **The correlation has the wrong sign.** Coverage deficit predicts
   `gain ~ 1/alpha`, i.e. positive. Measured −0.35 on both sequences: views with
   *less* coverage want a *smaller* gain.
2. **Conditioning on coverage makes it worse.** Residual std rises from 0.0454
   to 0.0806 — the model explains negative variance.
3. **The gain is below 1, so the render is too BRIGHT, not too dark.** The whole
   argument rested on incomplete alpha darkening the render. With the map frozen
   the best per-frame gain is 0.93.

Point 3 also corrects a conflation of mine: §17.6's gain of 1.04 came from a run
where the map and the exposure were optimized *jointly*, so it describes where
that pair settled, not what the baked map needs. Those are different quantities
and I had been treating them as one.

**So the colour leg is independent of the low-parallax root.** The remaining
candidates are the ones §17.6 listed and never separated: auto white balance,
view-dependent illumination, the network's per-pair conditional bias. What is
now excluded is coverage.

The unified hypothesis survives as what Kimi called it — **a joint prior over
three independent hypotheses, not one hypothesis.** The seam and streak legs
carry their own evidence (§17.4's morphology, §17.32's measured lever) and are
untouched by this. Only the third leg was speculative, and it is gone.

**A general rule this produced, worth more than the colour verdict itself.**

Kimi's framing, adopted: **a nuisance parameter fitted against a FROZEN map is a
probe of the map; a nuisance parameter read out after JOINT optimization is a
fossil of the optimization dynamics.** Both are legitimate quantities and they
are never interchangeable. §17.6 read a joint-fit gain (1.04) and reasoned about
it as though it described the map; §17.35's frozen-fit gain (0.93) is the
quantity that actually does. The two even have opposite signs, which is how the
conflation surfaced.

This applies to every `--exposure` number reported anywhere in §17, and to the
exposure arms of any future experiment: **state which of the two a gain is
before interpreting it.**

Three zero-cost discriminators remain for the colour leg, none of them on the
critical path (all require the frozen-fit basis above, not the joint one):

1. **Channel decoupling** — split (g_R, g_G, g_B) and look at the spread of the
   channel *ratios* across frames. Ratios swinging while luminance holds still
   means auto white balance; channels moving in lock-step rules it out.
2. **Per-cluster ANOVA** — group supervision frames by their dominant visible
   cluster and take between/within variance. A ratio >> 1 confirms the network's
   per-pair conditional bias.
3. **Temporal smoothness** — |g_{f+1} − g_f| against a shuffled null. Camera-side
   drift (exposure, white balance) is smooth in time; view- or map-side bias
   jumps. Note §17.4 already measured a 0.6% p90 step between neighbouring
   keyframes, so a "smooth" answer here would itself be surprising.

Kimi's stated bet: white balance contributes <= 2-3%, and the per-cluster ANOVA
comes back > 2, making the network's per-pair conditional bias the principal
cause. If that holds, the fix is not a per-frame affine at all (which can only
absorb a visibility-weighted average) but **map-side per-cluster colour
harmonization** — the online form of `scripts/color_harmonize.py`, evaluated on
seam-step and tint rather than on psnr.

### 17.36 The dose curve closes the network side, and re-attributes §17.34

`scripts/diag_parallax_dose.py` runs Splatt3R on keyframe pairs already in the
trajectory spanning rho = baseline / median_depth from 0.02 to 0.5, and scores
the predicted depth against the RGB-D sensor. Two quantities, because they
answer different questions: the **within-pair spread** (what §17.34's mechanism
needed) and the **per-pair scale error** about the global scale (what seams
need).

```
360, 80 pairs
                          median   spearman(rho, .)   within high-overlap
within-pair depth spread   ~3%         -0.085              -0.061
per-pair scale error       9.10%       -0.047              -0.016
spearman(overlap, spread)              -0.344
spearman(rho, overlap)                 -0.101
```

**Both are flat in parallax**, and the confound Kimi warned about is absent:
rho and overlap barely covary (−0.101), and the effect stays flat *within* the
high-overlap half. Overlap does matter (−0.344; 2.28% spread above 0.7 overlap
against ~4% below), but the production pairing is temporally adjacent and
therefore already maximal on overlap — there is nothing to win by re-pairing.

The 9.10% per-pair scale error also **cross-checks §17.17's 9.6%** by a
completely different measurement path.

**Experiment B (baseline-aware pairing) is cancelled**, including its revival
route for seams specifically. The network side is closed.

#### What that does to §17.34

The −0.964 correlation between refinement's lpips gain and a sequence's
translation/rotation ratio survives; **its attribution does not.** It was
assigned to the network's input pairing (adjacent pairs have no baseline under
rotation). The dose curve says depth quality does not depend on that baseline at
all, so the mechanism has to move.

The replacement, to be discriminated next: **it is the supervision, not the
prediction.** Under rotation every supervision view of a surface sits at nearly
the same position, so the photometric loss constrains the map along far fewer
directions and refinement has less room to improve — which also explains the
psnr/lpips split, since fitting survives low-parallax supervision and perceptual
structure does not.

**This re-attribution is the most instructive result of the section and belongs
in the main text, not in a corrections log**: the intuitive answer was the
backbone, and a dose curve said otherwise.

#### A different unification than the one proposed

§17.35 killed the low-parallax unification. What the flatness suggests instead is
narrower and better supported: **Splatt3R's per-pair conditional bias is a single
root for two of the three residual defects.** The per-cluster scale error is ~9%
and independent of the pair's geometry, which is what a per-pair bias looks like;
and Kimi's leading candidate for the colour leg (per-cluster ANOVA >> 1) is the
same phenomenon in colour. Both jump per pair, neither depends on parallax.

That makes the revised **experiment A the only remaining hard case for changing
backbones**: an N-view model predicts jointly and by construction cannot have
per-pair independent jitter. If VGGT's per-cluster spread is far below 9%, it
targets the parent of both defects; if it is also ~9%, the bias is prior-bound
and no backbone helps — which would be the strongest possible support for the
information-boundary framing.

### 17.37 The decider: it is the supervision, and it is an information boundary

Two mechanisms survived §17.36 and they demanded opposite wordings. **M1/M3'**:
the supervision itself is impoverished under rotation, an absence defect in
§17.27's sense that no map-side work can reach. **M2**: rotation stacks more
mutually-disagreeing clusters, a disagreement defect that cross-cluster
consistency or pruning could fix. Calling it a boundary while M2 held would have
been simply wrong.

`scripts/diag_veil_band.py` measures the **veil-band fraction** — the share of
covered pixels whose top two contributing clusters differ in depth by 0.5-20 cm,
computed by rendering each cluster alone (K renders of 1/K of the map ≈ one full
render). All seven sequences:

```
seq    m/deg   d_lpips   veil-band   layers/px
rpy    0.184    -8.6%     0.7335       7.84
360    0.571    -8.3%     0.7051       3.76
room   0.866   -15.9%     0.7036       7.47
teddy  1.115   -16.2%     0.6029       9.51
plant  1.126   -16.3%     0.6379      11.30
desk   1.830   -22.0%     0.7527       5.42
xyz    2.718   -29.3%     0.7799       9.25

M2 needs the veil fraction to track both parallax and the gain:
  spearman(m/deg, veil)    = +0.321     weak, and the WRONG SIGN
  spearman(veil,  d_lpips) = -0.357     weak

THE DECIDER -- lpips gain vs parallax, controlling for the veil:
  raw pearson(m/deg, d_lpips)      = -0.978
  partial, veil regressed out      = -0.975
  partial, layers regressed out    = -0.983
```

**Controlling for the veil removes nothing.** −0.978 becomes −0.975. And the veil
fraction correlates with parallax in the *wrong* direction (+0.321: more
parallax, more veil), so the M2 story fails even before the partial correlation.

**M2 is dead. M1/M3' stands: the penalty is in the supervision.**

That also explains a result that had no explanation: §17.30's cross-cluster
consistency loss moved the warp deficit 2-4% in four configurations. **It was
repairing something that was never the bottleneck.**

#### The claim, in the form the evidence supports

> **The perceptual quality of the final map is set by the supervision information
> the trajectory supplies, independently of the feed-forward front end, and psnr
> cannot see this bound.**

Front-end independence is what makes it more than a bug report: the dose curve
(§17.36) showed depth quality is flat in parallax, so this is not about Splatt3R,
two-view prediction, or any backbone. It is a property of passive SLAM mapping
under a given trajectory. Two limits are required and stated:

- **Scoped to representations without generative completion.** A generative prior
  can move content from "not in the trajectory" to "in the prior". That is a
  different research programme and is explicitly out of scope.
- **n = 7, and motion type covaries with scene type** (rpy/360 are room-scale
  rotations; plant/teddy are object-centric translations). The within-sequence
  segmentation analysis that would break that confound is designed and unrun.

#### From limitation to instrument

The translation/rotation ratio is computable **online, from the trajectory
alone, with no map**. So the same result that bounds quality also predicts it
before the map is built, and can steer an active mapper — *"this segment will
produce a perceptually poor map; translate."* That turns the finding from an
obituary into an instrument, and it is the form worth writing up.

### 17.38 T1.1: the within-sequence test is weak, and §17.37 is demoted

§17.37 framed the parallax result as an information boundary. That framing rests
on a causal claim, and the confound it named — motion type covarying with scene
type across the seven sequences — was never broken. `scripts/diag_within_seq.py`
scores every held-out frame against both the baked and the polished map from the
same run and correlates the per-frame gain with local parallax **inside** each
sequence, where the scene is fixed by construction. No new runs.

Two independent variables, the second on Kimi's advice after the first proved
underpowered:

```
seq     local traj ratio (range)   supervision parallax (range)
xyz       -0.151  (1.8x)             -0.289  (2.8x)
rpy       -0.268  (3.1x)             -0.188  (2.6x)
plant     -0.075  (1.4x)             -0.295  (2.5x)
teddy     -0.107  (1.6x)             -0.093  (2.2x)
360       -0.023  (1.5x)             +0.032  (2.0x)
room      -0.138  (3.6x)             +0.364  (3.9x)
desk      -0.333  (1.3x)             +0.123  (2.0x)
```

**The trajectory ratio reproduces in sign 7/7** (sign test p = 0.008) but with
tiny magnitudes, mean −0.156 against the cross-sequence −0.978. That is at least
partly structural: within a sequence the ratio spans only 1.3-3.6x, against 15x
across sequences, so the test has almost no leverage. Kimi's Fisher-combined
p ≈ 0.2 is the honest summary of that column.

**The mediator makes it worse, not better.** Supervision parallax — the spread of
the nearest keyframe positions over median depth — carries *more* dynamic range
(2.0-3.9x) and yet gives 4 negative and 3 positive: **the sign is not
consistent.** If supervision parallax were the mechanism, more range should have
sharpened the effect, not dissolved it.

Two readings, and they do not need separating to reach the verdict: either the
mediator is a poor proxy (it weights the nearest keyframes by position, never by
whether they actually see this frame's content), or the M1 mechanism is wrong.

**Verdict: §17.37 is demoted.** What survives:

- the cross-sequence correlation, −0.978 over seven sequences — measured;
- M2's exclusion, partial r = −0.975 after controlling for the veil — measured;
- **the causal claim that this is a boundary set by the trajectory's supervision
  information — NOT supported.** The within-sequence evidence is weak in one form
  and sign-inconsistent in the other.

So the honest headline is **"a strong seven-sequence correlation whose scene
confound is unbroken"**, not an information boundary. The stronger wording
requires either a mediator that survives within-sequence, or the synthetic
controlled-trajectory experiment Kimi proposed in round 8 (same scene, two
scripted paths), which is the only design that removes the confound by
construction.

Recording this against the temptation it corrects: the boundary framing was the
more publishable claim, it had a mechanism story, it survived one falsification
(M2), and it is still not supported. **Surviving one discriminator is not the
same as being established.**

### 17.39 T1.2: there is no per-view appearance variation — and I read noise as signal first

Three discriminators were run on the frozen-map fit (§17.35's probe, not the
joint-fit fossil), then a noise floor, **in that order — which was the wrong
order and cost a retraction.**

```
                        desk      360
chroma/luma spread      1.32      0.92
per-cluster ANOVA       0.82      0.48
temporal smoothness     0.597     0.829
```

Reading these alone, I concluded per-cluster network bias was falsified (correct)
and that white balance was supported (**wrong**). Kimi's leading bet — ANOVA > 2,
per-pair conditional bias — is falsified. Its warning that the discriminators are
unreadable without a floor is the part I acted on too late.

**The floor, by split-half refit** — fitting the affine independently on the top
and bottom halves of each frame, whose disagreement is fit noise by construction
since a frame has one true exposure:

```
                desk                            360
R/G    obs 0.0598  floor 0.0495  SNR 1.21    obs 0.0646  floor 0.0728  SNR 0.89
B/G    obs 0.0129  floor 0.0452  SNR 0.29    obs 0.0894  floor 0.0704  SNR 1.27
LUMA   obs 0.0454  floor 0.0444  SNR 1.02    obs 0.0976  floor 0.1162  SNR 0.84
WB axis corr(R/G, B/G)     -0.214                          +0.318
```

**All six SNRs are at or below 1.** The entire per-frame spread — luminance and
chroma alike — **is fit noise.** The white-balance axis test agrees
independently: real colour-temperature drift is one-dimensional, so R/G and B/G
must move oppositely; measured −0.21 and **+0.32**, the wrong sign on 360 against
Kimi's predicted −0.4 to −0.7.

So the chroma evidence I used to support white balance two messages earlier was
measuring noise. Retracted.

**What is real is the global offset**: mean gain 0.9300 / 0.9402, i.e. the map is
6-7% brighter than the targets. With per-frame std 0.045 over 50 frames the
standard error is 0.0064, so that is an 11-sigma effect, not noise.

#### The conclusion, and the puzzle it resolves

> **"Colour/lighting inconsistency" is not a per-view phenomenon.** There is no
> per-view appearance variation above the noise floor. `--exposure` earns its
> +0.45 dB fit psnr offline by absorbing a **global** bias, not by compensating
> per-view differences, and the correct fix is one global colour correction
> rather than six parameters per frame.

This also settles a contradiction that had been sitting unexamined: §17.4
measured 0.6% luminance drift between neighbouring source keyframes while §17.6
reported an 11-12% per-frame gain std. The two never fit. They fit now — the
11-12% was a joint-optimization fossil (§17.35), and the map's own per-frame
spread is at the noise floor.

#### What that implies about the defect the user actually saw

The GUI inconsistency is real and the per-view statistics say it is not per-view.
The only remaining possibility is that it is **spatial**: a colour step between
two clusters *within one frame*, which a per-frame affine cannot see by
construction, since it applies one transform to the whole image.

**So the colour leg and the seam leg are the same measurement problem**, and the
per-frame affine was never the right instrument for either. The instrument for
both is the seam-step metric Kimi specified in round 5 — boundaries taken from a
per-pixel argmax-contributing-cluster map derived from the renderer, filtered to
rendered depth difference < 5 cm and alpha > 0.8, brightness step normalized by
local gradient energy, with a 10 px boundary shift as the null. It has been
designed since round 5 and never built. It is the next thing to build.

### 17.40 The colour leg closes: an evaluation colour-space mismatch

Chasing the global 6-7% offset to its source found a protocol bug that had been
in every number this project has ever reported.

**`normalize_exposure` is applied to the map's inputs and not to the evaluation
targets.** `frame.py:124` calls it inside `create_frame`, so every keyframe fed
to Splatt3R — and therefore every colour baked into the map — is in
exposure-normalized space, rescaled so each frame's channel means match the
first frame's. Every evaluation, here and in `eval_map_quality.py`, builds its
target with a bare `ds.get_image(di)`. **Raw.**

The size of the mismatch:

```
                normalize_exposure gain   per-frame std
desk                  0.9567                 0.0892
360                   1.1379                 0.1546
```

Evaluating the same map against normalized instead of raw targets, desk,
old+polish:

```
raw target          psnr 14.0640   lpips 0.4247
normalized target   psnr 14.1897   lpips 0.4220
                         +0.126 dB      -0.6%
```

**+0.126 dB lands inside Kimi's pre-registered window of +0.08 to +0.15** for a
pure global gain at psnr ≈ 14, and its decision rule reads: that magnitude means
the effect is a clean global one, with no per-channel or spatial structure hiding
behind it. Under +0.3, the colour leg closes.

#### Three things this resolves at once

1. **Why §17.39 found no per-view appearance variation.** There is none in the
   map — `normalize_exposure` removes the per-frame exposure and white-balance
   drift *on the input side*, which is exactly what its docstring says it is for.
   The map is internally consistent. I spent this leg looking downstream for the
   remains of a defect the system already fixes upstream.
2. **Why the fitted per-frame gain still had some spread.** The map is
   normalized, the targets are not, so the residual per-frame variation belongs
   to the *targets*. The fitted std (0.0439 desk / 0.0836 on 360) is about half
   the normalization gain's own std (0.0892 / 0.1546) — the same order, as it
   should be.
3. **Why the mean gain sat at 0.93-0.95.** It is largely the mean of that same
   mismatch, not a property of the map.

#### What it does and does not invalidate

**Every A/B comparison in this record stands.** The mismatch is present in both
arms of every pair, and it is a fixed function of the frame, so differences are
unaffected. What is affected is **absolute** psnr/lpips: they are depressed by
roughly 0.13 dB on desk, and by more on sequences with larger normalization
drift (360's gain std is 0.155, so its bias is likely larger).

The fix is one line — normalize the target the same way — and it makes the
protocol self-consistent. It should be applied before any absolute number is
published, and it changes no conclusion already drawn.

#### The methodological point, which is the ninth of its kind

The 6-7% offset was reported here as an 11-sigma effect. It was not wrong, but
the sigma was computed over *frames* and the error was a property of the
*pipeline*, so the error bar measured the wrong variation entirely. Kimi's gate
("split-half measures stability, not systematic bias — a stably-wrong estimate
also earns 11 sigma") is exactly right, and it is the same failure as §17.31's
six: **a statistic that quantifies something other than the claim it is used to
support.**

### 17.41 The seam carries no colour step: it is purely geometric

`scripts/diag_seam_step.py` builds the spatial instrument §17.39 argued was the
only one left: per-pixel argmax-contributing-cluster map from per-cluster
renders, mode-filtered 5x5 to kill the interleaving flicker, boundaries taken as
4-neighbour id jumps, kept only where rendered depth agrees within 5 cm (a true
occlusion edge should have a step) and both sides have alpha > 0.8, brightness
step normalized by local gradient energy, against a null that displaces the
boundary mask by 10 px.

```
seq     seam pixels   seam step   shifted null   ratio
desk       3.91%        0.1729       0.1720      1.005
360        1.27%        0.1998       0.1847      1.082
room       9.40%        0.1887       0.1812      1.042
```

**Cluster borders carry essentially the same brightness step as ordinary
positions** — +0.5% on desk, +8.2% at most on 360.

So the colour reading of the seams is now falsified from both directions:

- **per-view** (§17.39): no appearance variation above the noise floor;
- **spatial** (here): no step where clusters meet.

Both agree with §17.4's morphology, which said it in the first place: the
artifact is a semi-transparent **veil**, and a colour difference cannot produce a
veil.

**The first attempt at this measurement got ratio 0.11 — the seam step eight
times SMALLER than its null — which is what sent me to read the code.** The null
was displacing the comparison rather than the location: real boundaries compared
pixel x with x+1, the null compared x with x+11, and pixels 11 apart naturally
differ more. A null must move *where you look*, never *how far apart you look*.

That is the tenth instance of §17.31's failure mode and its third distinct form:

```
intervention   changed a quantity other than the target
error bar      quantified a variation other than the claim's
null           compared a quantity other than the signal's
```

All three are the same mistake — **supporting a claim with something that does
not measure it** — and the only defence that has ever worked here is finding the
number implausible and reading the code.

#### Where that leaves the seams

Fully diagnosed, and not fixable by anything tried:

```
cause        the network's per-pair scale error, ~9% (17.17, cross-checked
             9.10% in 17.36), INDEPENDENT of the pair's geometry
not          exposure (4% drift, 17.4) / colour step (this) / per-view
             appearance (17.39) / layered veils carrying the parallax
             penalty (17.37)
levers tried per-cluster depth scale (+0.003 dB), cross-cluster consistency
             (2-4% in four configurations), sensor depth (instrument
             inadequate on this dataset, 17.29)
remaining    the backbone. An N-view model predicts jointly and cannot have
             per-pair independent jitter by construction -- experiment A's
             revised question (17.36), and now the only live route.
```

### 17.42 T1.3: the streak lever transfers online, and transfers *better*

§17.32 measured the lever offline and refused to recommend it online, because
§17.21 had established that offline quality levers do not transfer. T1.3 is the
paired online A/B that decides it: two full `main.py` runs on desk, identical
but for `--refiner-streak-opacity`, each with the polish, evaluated on the same
50 held-out frames.

```
online, desk        psnr      lpips     black (alpha<0.1)
aa only (K=0)      14.0477   0.4382       0.49%
streak K=0.5       13.9434   0.3926       0.57%
delta              -0.104    -10.4%      +0.08 pp
```

**−10.4% lpips online against −5.0% offline.** This is the first lever in the
whole section that is *stronger* in the online regime than in the offline one,
and the reason is visible in §17.21's mechanism: the online map is polished from
a baked state whose streaks have never been optimized away, so there is more
streak left for the lever to hide. Offline, the 300-step refinement has already
absorbed part of the same error into the surface.

The pre-registered risk check passes again and by a wider margin: Kimi's
falsification line was black +1 pp, the prediction was +<0.5 pp, measured
**+0.08 pp**. (Absolute black is 0.49% online against 3.59% offline because the
online map is the full-sequence one with far more coverage, not the head-only
map §17.32 used -- the two absolute columns are not comparable, the deltas are.)

**Verdict: T1.3 passes. `--streak-opacity 0.5` is validated online and is the
recommended default in both regimes.** It stays a flag rather than a hard-coded
constant, because K is a band-limit ratio and 0.5 is measured on one sequence.

#### A correction to record

Reporting this result before it was written up, I quoted the black-fraction cost
as "+0.31 pp" -- **that is §17.32's offline number, carried across to an online
claim it was never measured for.** The online value is +0.08 pp. The verdict is
unchanged and the direction was right, but the number was borrowed, and this is
the same family as §17.31/§17.41: *a quantity that does not measure the claim,
used to support it*. It is worth noting that this one came from writing prose
faster than the measurement, not from a design error -- a different entry point
to the same failure, and one that only a write-up-before-report discipline
catches.

The first attempt at the measurement also failed outright: the script asked for
`g["colors"]` when `decode_gaussians_from_ply` returns `rgb`. That is a loud
failure (KeyError) and therefore harmless -- worth contrasting with the silent
ones this section keeps collecting.

### 17.43 Plan status against the three-track plan

Where the tracks stand, so this is not re-derived:

```
T1.1  within-sequence segmentation    DONE  17.38  weak -> 17.37 demoted
T1.2  three colour discriminators     DONE  17.39/17.40  both of Kimi's bets
                                            missed; the target turned out not
                                            to exist (evaluation colour space)
T1.3  streak lever, online paired     DONE  17.42  -10.4% lpips, passes
T2.1  experiment A (N-view scale)     DONE  17.44/17.45/17.46  GREEN, ratio
                                            0.31, 7/7 -- and the cause is
                                            context, not architecture
M     aggregation dose-response       DONE  17.47  RED, rho = 0.35, capped
T3.1  adaptive polish termination     DONE  17.51  NEGATIVE -- no plateau
                                            exists inside any budget worth
                                            spending; flag stays off
T3.2  polish into the default pipeline + docs   DONE  README documents the
                                            measured configuration and the
                                            known limits; main.py fails fast
                                            on the three bad flag combinations;
                                            no default was changed
R     VGGT as external scale referee  DONE  17.48  NEGATIVE -- geometry 6.4x
                                            better, psnr -0.86 dB
D     downstream link                 ANSWERED by R, in the negative: a large
                                            geometric gain did NOT move the
                                            rendered metrics, it hurt them
B2    per-keyframe scale in the SLAM backend, with the referee as a prior
                                      NEXT -- the only place left where poses
                                            and scales are solved together
```

Cancelled and not to be revisited: experiment B, further cross-cluster
consistency, sensor-depth injection, anisotropy clamping, **multi-partner
aggregation (17.47)**, **any post-hoc per-cluster geometric correction
(17.48)**, and any backbone change.

### 17.44 T2.1 experiment A: the design, the control arm, and a pre-registration

Written **before** the test arm exists, because §17.42 just recorded what
happens when prose runs ahead of measurement.

#### The instrument

`scripts/diag_nview_scale.py` holds both arms in one file. Windows of 16 views
at each sequence's real median keyframe spacing, 12 windows per sequence, seven
sequences. Per view, `r_k = median_pixels(z_k / d_sensor,k)`.

```
scale spread    median_k |r_k / median(r) - 1|
neighbour step  median_k |r_{k+1} / r_k - 1|      <- PRIMARY: what a seam sees
within-view     median_k median_pix |ratio / r_k - 1|
```

Three properties that were designed in, not discovered afterwards:

- **Pose-free.** Predicted depth in each view's own camera frame against that
  view's own RGB-D frame. No trajectory, no Umeyama, no alignment — SLAM drift
  cannot enter either arm. §17.29's arms died of exactly the confound this
  removes by construction.
- **Gauge-free.** Every statistic is normalized by the window median, which
  deletes VGGT's unknown global scale and Splatt3R's −6% metric bias alike.
- **Paired.** Both arms score the *same* windows of the *same* frames, so the
  comparison is a paired test. Scene content dominates the absolute level of all
  three statistics — §17.38's confound in another costume — and pairing removes
  it rather than arguing about it.

The arms: VGGT = one joint forward over the 16 views; Splatt3R = 16 forwards,
each view paired with its temporal neighbour, which is what `tracker.py` feeds
it in production. A third arm, **VGGT in pair mode**, runs the same model on the
same adjacent pairs — the control that separates *joint context* from *different
model*, without which a low VGGT number could be either.

#### The control arm reproduces the known number by a third protocol

```
seq     scale spread   neighbour step   within-view   median ratio
desk        8.02%          6.19%           2.68%         0.9973
360         7.22%          7.02%           3.41%         0.9592
room        4.50%          4.64%           3.58%         1.0153
xyz         2.90%          4.49%           3.17%         1.0047
rpy         4.75%          7.09%           5.67%         0.9622
plant       3.49%          4.91%           4.48%         1.0377
teddy       4.36%          4.71%           4.33%         0.9613
```

§17.17 measured 9.6% on a SLAM trajectory; §17.36 measured 9.10% on random pairs
across a sequence; this is a third protocol — local windows, no trajectory at
all — and **desk and 360, the two sequences those results used, are the two that
come back at 7-8%**, with 360's median ratio 0.9592 landing on §17.17's 0.9405.
Where the protocols overlap they agree.

They also correct the headline: the other five sequences run at 2.9-4.8%, so
**"~9%" was desk and 360, not Splatt3R.** The pooled median is 4.50%.

A free byproduct: `step / spread` is 0.77 on desk but ~1.5 on xyz and rpy. For
independent per-view errors that ratio is sqrt(2); so desk's error process is
smooth (correlated, a drift) while xyz's is nearly white. That is the
correlation length of the jitter, and it decides how much accumulates between
two clusters that meet after a long detour.

#### Kimi's pre-registration, before the numbers

```
                  pooled          desk        360
scale spread      1.4%  (0.8-2.2) 2.2 (+-0.6) 2.0 (+-0.6)
neighbour step    1.1%  (0.7-1.8) 1.7 (+-0.5) 1.6 (+-0.5)
within-view       parity, 3-5% (sensor-noise dominated)
step ratio        0.24  (0.15-0.40)
VGGT pair mode    3.5%  (2.5-5.5) -- nearer Splatt3R than VGGT-16
```

Decision rule, sharpened from §17.36's:

```
green  step ratio <= 0.50, >=6/7 sequences, desk+360 absolute step <= 3%
red    ratio >= 0.80, or <=4/7 sequences   -> prior-bound, close the thread
yellow between                             -> aggregation before any backbone
```

#### Four corrections Kimi made to the design, all adopted

1. **The primary endpoint is the neighbour step, not the spread.** The spread
   also contains long-range wander, which accumulates differently depending on
   whether the process is white or a random walk — and the two arms need not
   share that structure. The step is what an adjacent-cluster seam sees.
2. **The honest degrees of freedom is 7, not 84.** Twelve windows inside one
   sequence are not independent. The window-level Wilcoxon is the power read;
   the decision rests on the sequence-level sign test. `expA_summary.py` reports
   both and the window overlap.
3. **A global shift would silently distort every ratio in the file.** The median
   normalization absorbs a global scale, not a shift. Added a shift probe: fit
   `z = a*d + b` per view and report `b` as a fraction of median depth.
4. **The sensor sets a floor.** Kinect error grows with incidence angle and
   range (~1-3%), and that bias lives in the *reference*, so even a perfect
   model shows per-view `r_k` variation. It is paired away in the comparison but
   caps interpretation: **a VGGT result at 1-2% is "at the reference floor" and
   differences below it must not be resolved.** Splatt3R's 7-8% on desk/360 is
   far above any such floor, so the gap being tested is real.

#### The one place my framing was wrong

I expected the artifact to be that a shared gauge makes the joint arm look good
for free. Kimi's answer: the window-median normalization removes exactly one
degree of freedom, and per-view relative jitter survives it — nothing in the
architecture *hard-constrains* 16 depth maps to share a scale, so a low joint
number is real evidence.

The artifact is the mirror image, and it is about deployment: **this instrument
measures within-window consistency, but no N-view SLAM ever sees the window
jointly.** It processes submaps and aligns them, and the jitter re-enters at the
submap boundary through the alignment. The Splatt3R arm has an anchor for that
gap — its local number reproduces the deployed 9.6%/9.10%. The VGGT arm has
none. So a win here licenses "joint prediction kills per-view jitter within a
window", and the deployed benefit is strictly smaller by an unmeasured amount.

**Consequence for the decision:** a green result funds the *next-cheapest* step
that captures it — windowed joint inference at keyframe creation, an offline map
rebuild — not an online backbone swap.

And a route that did not exist before this round: if VGGT wins jointly but its
pair mode lands near Splatt3R, the win is **aggregation, not architecture**, and
aggregation has a cheap implementation on the current backbone — predict each
new keyframe against m well-overlapping partners, scale-align the m depth maps
by their median ratio, average. Independent jitter falls as sqrt(m). That would
be days of work with no new model, and it only becomes visible because the pair
-mode control was built.

### 17.45 T2.1 result: GREEN, and the cause is not the architecture

The pre-registered rule (§17.44) was: primary endpoint the neighbour step,
paired by window, ratio <= 0.50 with >= 6/7 sequences and desk+360 absolute
step <= 3%.

```
sequence     n  splatt3r    vggt16   ratio  paired p
360         12     7.02%     2.63%    0.37    0.0010
desk        12     6.19%     1.14%    0.18    0.0005
plant       12     4.91%     1.25%    0.26    0.0005
room        12     4.64%     1.83%    0.39    0.0005
rpy         12     7.09%     2.58%    0.36    0.0005
teddy       12     4.71%     1.55%    0.33    0.0005
xyz         12     4.49%     2.00%    0.45    0.0005
POOLED      84     5.35%     1.68%    0.31  1.77e-15
sequence level  7/7, sign test p = 0.0156, median ratio 0.36
```

Scale spread, same pairing: 4.63% -> 1.18%, ratio 0.25, 7/7. **Green on every
clause**, and VGGT-16's absolute numbers (0.96-2.27%) sit at or below the 1-2%
sensor reference floor of §17.44, so part of the residual is the Kinect's own
view-dependent bias rather than the model's.

#### The control arm is the result

The same VGGT, run on the same adjacent pairs, one independent forward each --
the arm that separates *joint context* from *different model*:

```
sequence     n  splatt3r     vggt2   ratio
desk        12     6.19%    19.41%    3.14
360         12     7.02%    11.73%    1.67
room        12     4.64%    10.19%    2.20
rpy         12     7.09%    16.37%    2.31
teddy       12     4.71%    10.25%    2.18
plant       12     4.91%     8.19%    1.67
xyz         12     4.49%     7.01%    1.56
POOLED      84     5.35%    10.82%    2.02   vggt2 lower in 0/7
```

So the 2x2 on the primary endpoint:

```
Splatt3R pairwise    5.35%    architecture A, 2 views
VGGT     pairwise   10.82%    architecture B, 2 views    <- 2.0x WORSE
VGGT     joint-16    1.68%    architecture B, 16 views   <- 6.4x better than
                                                            the SAME model on
                                                            the SAME pairs
```

**Changing the architecture makes it worse. Changing the context makes it 6.4x
better.** VGGT is a worse two-view model than the MASt3R line -- unsurprising in
hindsight, since its prior is trained for joint reasoning over many views and
two views give it almost nothing to reason with.

#### What that does to the seams

The parent of the seams is **not the backbone**. It is the structural fact that
**each cluster's scale is decided by one two-view prediction**, and neighbouring
clusters are decided by different ones. §17.36 called the ~9% a "per-pair
conditional bias"; this measures what removes it, and the answer is context, not
capacity.

The route this opens does not need a new backbone at all, and it is Kimi's
round-16 fallback promoted to first choice: **at keyframe creation, predict the
new frame against m well-overlapping partners, scale-align the m depth maps by
their median ratio, and average.** Independent per-pair draws fall as sqrt(m).
VGGT-16's 1.68% is the ceiling that route is aiming at.

#### Independence check

The windows above overlap 80-97% -- a 16-view window at keyframe spacing covers
most of a TUM fr1 sequence. Repeated with 8-view windows tiled by exactly one
span, so they are disjoint:

```
                    control   test    ratio   sequence level
vggt8   (joint)      4.92%    1.53%   0.31    7/7,  p = 8.6e-08 (33 windows)
vggt2n8 (pairwise)   4.92%    8.08%   1.64    1/7
```

Same ratios, disjoint windows. The overlap did not manufacture the result, and
the honest sequence-level test carries it either way.

#### Instrument checks

- **Shift probe**: Splatt3R +18.5%, VGGT +29.9% of median depth. Both models
  compress the depth range substantially -- a real property of both priors, and
  worth its own entry later; it does not affect a ratio statistic normalized
  per window, but it does mean `r_k` could in principle respond to how far away
  a view's content happens to be.
- That was checked rather than assumed: **Spearman(r_k, view median depth) =
  -0.057** for VGGT. The spread is not a deterministic scene response.
- VGGT's per-view depth-head output is used, never pointmap z; sensor resampled
  NEAREST; masks identical across arms; every statistic gauge-free.

#### A silent-output bug, caught by counting files

Every `splatt3r` arm wrote its `.npz` nowhere. The arm `chdir`s into
`splatt3r_core` to import the model, so a relative `--out` resolved into a
directory that does not exist there -- and the `FileNotFoundError` fired *after*
the summary was printed, so the console output of a run that saved nothing was
indistinguishable from a successful one. Found only by listing `logs/expA` and
finding seven files missing. Fixed by resolving `--out` to an absolute path
before anything can `chdir`.

Not the §17.31 family (nothing was wrongly claimed), but the same lesson from a
different side: **the log said it succeeded and the filesystem said it did not,
and only the filesystem was asked.**

### 17.46 Round 17: coupling is not averaging, and Kimi scores his own miss

#### The correction, which lands on my write-up and not on the data

I reported §17.45 as "the win is provably context, so it can be harvested on the
current backbone." **That sentence equates two different operations.** Kimi's
lead point:

> The 2x2 proves that *coupling* kills the jitter -- one joint forward, views
> constrained to be mutually consistent. It does not prove that *averaging*
> kills it. Joint attention can enforce cross-view consistency constraints
> (effectively solving a mini SfM inside the window), while averaging only
> cancels the *independent* part of the error.

If a draw carries a component determined by the target view's own content --
the same prior mistake whoever the partner is -- averaging saturates at that
floor while coupling still reaches 1.68%. **The 2x2 contains no information
about that correlation.** So the aggregation route is not "more attractive"; it
is unchanged in status, a hypothesis with one named failure mode.

#### What the 2x2 does say about the origin

Kimi's reading, adopted: **per-cluster scale error is a per-forward gauge
lottery.** Each independent forward draws an affine gauge -- scale *and* shift,
both are in play per §17.45's probe -- from a model-specific distribution.
Splatt3R draws at sigma ~ 5%, VGGT at sigma ~ 11%. The draw is not driven by
scene content (Spearman -0.057), not by parallax (§17.36's flat dose curve), not
by overlap. The map inherits whichever draw each cluster's birth pair happened
to make, **and a seam is the difference between two draws.** Joint prediction
does not improve the draws; it abolishes them.

Two consequences worth the main text:

- **Two-view accuracy and multi-view consistency are different capabilities.**
  The better pair model is the one any pairwise benchmark would select, and it
  is the one that loses 6.4x on the deployment-relevant statistic. Feed-forward
  models for SLAM should be selected on windowed scale consistency, not pair
  accuracy.
- **A joint model that is worse than a pair model at N=2** has a depth head that
  needs cross-view context even to anchor its gauge. That sharpens §17.37's
  information story: the problem with prior-driven depth was never only the
  bias (-6%), it is the per-draw **variance**, and context is the only thing in
  the system that fixes variance.

#### Kimi's self-scored miss

He predicted VGGT pair-mode at 3.5%, "nearer Splatt3R than VGGT-16". Measured
10.82%, 2x worse than Splatt3R -- wrong in magnitude by 3x pooled (5.5x on desk)
and wrong in direction relative to the control. His two load-bearing
assumptions:

1. **that per-draw gauge variance is roughly constant across competent two-view
   models.** False by 2x. He treated "~5% two-view consistency" as a property of
   the task; it is a property of the training.
2. **that joint architectures degrade gracefully to few views.** False. VGGT's
   pair mode is not a weaker version of its window mode.

And the part worth keeping: **the same assumption underwrites his aggregation
fallback**, whose premise is that per-forward draws are well-behaved and
independent. He flagged that himself -- the miss is evidence that the premise
deserves a measurement rather than a sprint.

#### What experiment A licenses -- for the write-up

> Windowed joint prediction of 16 views reduces adjacent-view depth-scale
> inconsistency from 5.35% to 1.68% (paired Wilcoxon over 84 windows; 7/7
> sequences; disjoint-window replication ratio 0.31), reaching the RGB-D
> sensor's own view-dependent bias floor, while the same model evaluated
> pairwise is 2.0x worse than the incumbent two-view backbone (10.82% vs
> 5.35%). This licenses two claims: (1) per-cluster scale jitter in
> trajectory-anchored Gaussian maps originates from independent per-forward
> gauge draws -- not from scene content, baseline geometry, or image overlap;
> (2) coupling views at prediction time eliminates the jitter, and the benefit
> is attributable to multi-view context rather than to a better two-view prior.
> It does not license: (a) claims about streaming deployment, where windows are
> processed causally and submap alignment reintroduces a gauge degree of freedom
> this instrument does not measure; (b) claims that averaging independent
> two-view predictions achieves the same reduction -- coupling is not averaging;
> (c) direct claims about rendered-map quality -- propagation of the scale-step
> reduction to seam-step and lpips metrics on a rebuilt map remains the open
> downstream link.

Note also that the measured ratio 0.31 is a **lower bound** on the improvement:
VGGT-16 sits at or below the sensor's own bias floor, so its true jitter may be
smaller than 1.68% and nothing below that floor should be resolved.

### 17.47 Experiment M: averaging cannot reach what coupling reached

`scripts/diag_aggregation.py`. Same windows, same sensor, same statistics as
§17.45, but the target view's depth is now the per-pixel median of **m**
two-view predictions, each with a different partner (the m nearest views,
nearest first, so m=1 is exactly the production pairing and the curve is paired
within window). Aggregation is **sensor-free** -- the sensor enters only the
scoring, so this is an operation production could actually perform.

```
sequence     n      m=1      m=2      m=4      m=8    rho
360         10    8.39%    5.92%    5.46%    4.76%  0.447
desk        10    7.36%    7.05%    6.24%    5.83%  0.525
plant       10    3.24%    3.01%    3.47%    2.92%  0.352
room        10    4.53%    4.30%    4.18%    4.51%  0.297
rpy         10    6.59%    4.87%    3.97%    3.01%  0.314
teddy       10    3.85%    2.98%    3.24%    2.85%  0.405
xyz          1    3.43%    2.11%    1.90%    2.54%  0.264

POOLED      61    5.16%    4.07%    4.05%    3.76%
sqrt(m)           5.16%    3.65%    2.58%    1.82%
```

**Pre-registered lines: m=4 <= 2.7% -> aggregation wins; > 3.5% -> capped.
Measured 4.05%. RED.**

The mechanism is the one Kimi named before the run. Decomposing
`log r_{k,p} = mu + a_k + e_{k,p}`, the target view's own content effect carries

```
rho = Var(a) / (Var(a) + Var(e)) = 0.30 - 0.53, median 0.352
```

so **a third to a half of every forward's gauge error is the same mistake
whoever the partner is**, and averaging cannot touch it. The implied floor is
sqrt(0.352) = 0.59 of the m=1 spread, i.e. 3.06%; the measured m=8 value is
3.76%, approaching that floor and nowhere near the 1.82% that independence
would give. Two sequences (room, plant) are flat from m=2 onward, and desk and
teddy *rise* at m=8 -- the eighth partner is four keyframes away and its lower
overlap costs more than another draw is worth.

```
pairwise, m=1        5.16%
aggregated, m=8      3.76%      averaging's practical floor
rho-implied floor    3.06%      averaging's theoretical floor
joint 16-view        1.68%      what coupling reaches
```

**Coupling is not averaging, and the gap is a factor of two below the best
averaging can ever do.** This is the clean version of §17.45's claim: joint
attention enforces cross-view consistency (it solves for the gauge), while
averaging only cancels the independent half of a draw.

#### What this closes and what it opens

**Closed: the aggregation route.** Multi-partner prediction at keyframe creation
buys at most 5.16 -> 3.06%, costs m forwards per keyframe, and does not reach
the target. Do not build it. (It cost one afternoon to kill, against a route I
had already described to the user as the recommended next move -- the cheapest
retraction in this whole section.)

**Open: VGGT as an external scale referee.** Run windowed joint inference
offline, take the per-cluster scalar corrections it implies, and apply them to
the existing map. This is §17.16's per-cluster depth-scale correction, which
measured +0.003 dB -- but that arm was driven by the photometric loss, which
§17.17 proved is structurally blind to depth error at this trajectory's
baselines (4.5% depth error = 0.66 px at 0.057 m). The same lever driven by a
reference that measures 1.68% on the same yardstick is a different experiment.
Map-side, days, no online surgery, no head retraining.

**Still the real verdict, still unrun: the downstream link.** Nothing here
touches a rendered image. A scale-consistency improvement of X% should appear as
roughly X% smaller seam steps on a rebuilt map (§17.41's metric). If it does
not, the failure is in the map pipeline rather than the model -- and that would
be the most valuable negative result of the thread.

### 17.48 Route R: the map's geometry got 6.4x more accurate and the image got worse

The referee: one joint VGGT forward over **all** of a map's keyframes, giving a
per-cluster scalar correction `s_k`, normalized to median 1 so only the relative
corrections are used and the map's global scale is untouched. Applied to the
baked map by scaling each cluster's `means` and `scales`.

```
desk, 14 keyframes, one forward       baked psnr   lpips    polished 300
base                                    12.354    0.5071   13.811 / 0.4199
referee, ratio form                     11.496    0.5227   13.082 / 0.4702
referee, slope form                     11.047    0.5252   13.012 / 0.4690
referee, slope + pose compensation      11.208    0.5881   13.620 / 0.4725
```

Every form loses, the polish does not recover it, and the seam-step metric is
1.02 against a base of 0.99 -- unchanged, as §17.41 predicted it would be, since
that metric was already null at baseline.

#### The audit that decides what this means

Two explanations, opposite next moves: (A) the trajectory has absorbed the
jitter, so the map is at a joint optimum that absolute accuracy does not
describe; (B) the referee is incompetent at this operating point -- §17.45
validated VGGT on 16 consecutive views a second apart, and a map's keyframes are
SLAM-selected to be as *different* from each other as possible.

`scripts/audit_referee.py` scores both the map and the referee against the RGB-D
sensor, per keyframe, in the same gauge-free form as §17.45:

```
                    map     referee
desk (13 kf)       6.48%      1.02%      6.35x
360  (46 kf)       8.08%      3.74%      2.16x
room (51 kf)       5.67%      2.48%      2.29x
```

Applying the referee composes `c_k * (v_k / c_k) = v_k`, so it **replaces the
map's per-keyframe scale error with the referee's**. On desk that is a 6.35x
improvement in absolute geometry.

**Explanation B is refuted. The referee is right, and the picture still gets
worse.** Absolute per-keyframe geometry improved 6.4x; held-out psnr fell
0.86 dB and lpips rose 3%.

#### What that establishes

**In a trajectory-anchored map, rendered quality is not governed by absolute
geometric accuracy.** SLAM fitted each keyframe's pose to that keyframe's own
biased pointmap, so a large part of the per-cluster scale error is common-mode
with the pose. Correcting the cluster alone converts an absorbed error into a
*differential* displacement between overlapping clusters -- and while §17.17
proved the photometric evaluation is blind to common-mode depth error (4.5% =
0.66 px at this trajectory's 0.057 m baseline), it is not blind to two clusters
disagreeing about where a surface is. That is the veil mechanism itself.

This retroactively explains **§17.16's +0.003 dB**. That arm let the photometric
loss fit per-keyframe depth scales and it barely moved. The reading at the time
was that the lever was weak. The reading now is that the system was already at
its joint optimum: measured here, the photometric fit asks for **0.8%** while
the sensor says the map is **6.5%** wrong, and the two corrections do not even
agree in rank (Spearman +0.06 between them on desk).

```
what the photometric loss wants     0.8%   (17.16's lever, re-measured)
what the sensor says is wrong       6.5%
agreement between them              Spearman +0.06
```

**Route R is dead as a map-side edit, and so is every other per-cluster
geometric correction applied after the fact.** The jitter cannot be removed from
the map alone because the map alone is not where it lives -- it is shared
between the map and the trajectory. The only place it can be removed is where
poses and per-keyframe scales are solved *together*, i.e. the SLAM backend,
where each keyframe's Sim3 already carries a scale that the front end never
constrains with an external reference.

#### Two estimator lessons, both paid for

- **The shift-cancelling slope was right algebra and a worse estimator.** VGGT
  carries a +29.9% depth shift (§17.45), so `median(z_ref/z_map)` mixes the
  cluster's scale error with how far away that view's content is, and the slope
  of `z_ref` against `z_map` cancels it exactly. It also has far higher
  variance: on the same keyframes the slope form scores the referee at 10.63%
  where the ratio form scores it at 1.02%, and the slope arm did *more* damage
  to the rendering. Cancelling a bias is not free.
- **A correlation that had to be positive.** An earlier version of the audit
  correlated "the correction the sensor says is needed" (`1/c_k`) with "the one
  the referee proposes" (`v_k/c_k`) and reported +0.98. Both contain `1/c_k`;
  the number was guaranteed and meant nothing. Caught by noticing that a 0.98
  correlation was inconsistent with a 25% improvement. **Eleventh instance of
  §17.31, and the same defence worked again: the number was too good, so read
  the algebra.**

### 17.49 B2 in the refiner: a monotone conflict, not a joint optimum

§17.48 concluded that the per-cluster scale error is shared between the map and
the trajectory, so it can only be removed where both are solved together. B2 is
that, done in the cheapest place it can be done: free the keyframe translations
(3 DoF each) alongside the existing per-cluster depth scale, and add the
referee's corrections as a **prior** rather than an edit, so the photometric
term is free to disagree.

`--kf-pose-lr`, `--referee-scales`, `--referee-weight` in `refine_local.py`.

```
desk, 300 steps                     psnr     lpips    kf scale std
base (poses and scales fixed)      13.811   0.4199        --
depth scale free (17.16's lever)   13.683   0.4146      0.0144
+ poses free                       13.743   0.4452      0.0169
+ referee prior w=0.05             13.728   0.4494      0.0151
+ referee prior w=0.5              13.382   0.4561      0.0257
referee asks for                                        0.0800
```

**Monotone in the prior weight, and monotone the wrong way.** Every step the
scales take toward the referee costs image quality: psnr 13.73 -> 13.38, lpips
0.4494 -> 0.4561. And even at w=0.5 the fitted spread is 2.6% against the
referee's 8.0% -- the photometric term is still winning the argument, and
forcing it further is what §17.48 already measured as harmful.

Freeing the poses is separately negative: lpips 0.4199 -> 0.4452 with the scales
otherwise untouched. That is §13.12b's co-adaptation result in another costume --
the map was baked at the SLAM poses, so giving the poses back their freedom buys
training-view fit and loses held-out perceptual quality.

#### Why this arm could not have worked, stated properly

The refiner's data term is the near-baseline photometric loss, and §17.17 proved
that term is structurally blind to depth: 4.5% depth error moves a held-out
pixel by 0.66 px at this trajectory's 0.057 m baseline. **Solving two variables
"together" against blind data does not create information.** The only thing B2
added was the referee's prior, and the photometric term simply outvoted it.

So what B2-in-the-refiner actually measured is the conflict itself, cleanly:

```
what the photometric criterion wants     ~1.5-2.6% spread
what absolute geometry wants (17.48)      8.0%, and it IS correct (6.4x
                                          better against the sensor)
agreement between them                    Spearman +0.06
every step from one toward the other      costs psnr and lpips, monotonically
```

#### The conclusion this forces, which is larger than the seams

**Absolute geometric accuracy and rendered novel-view quality are different
objectives in a trajectory-anchored map, and on this data they conflict.** The
best-looking map is one that is geometrically *wrong* in a way that is
self-consistent with its own trajectory. §17.48 made the point once (geometry
6.4x better, psnr -0.86 dB); §17.49 makes it as a dose-response curve.

That reframes the seam thread end to end:

```
the seams are geometric                        17.41 (no colour step at borders)
caused by per-pair scale jitter                17.36, 17.45 (9%, and 0.31x
                                               under joint prediction)
the jitter is co-adapted with the trajectory   17.48, 17.49
removing it costs image quality, monotonically 17.49
```

**The seams are the visible price of a self-consistent-but-wrong map, and every
map-side and refiner-side route to them is now closed by measurement.** What is
left is a genuinely different intervention: the SLAM backend, where the data
term is wide-baseline keyframe-to-keyframe point matching rather than
near-baseline photometry, and where each keyframe's Sim3 already carries the
scale parameter that would have to move. That is not a tweak to this codebase's
refiner; it is a change to how the map is built.

#### Replication

Same protocol, referee prior at w=0.5 against a fixed-pose baseline:

```
seq     base psnr / lpips      B2 psnr / lpips       fitted std
desk    13.811 / 0.4199        13.382 / 0.4561         0.0257
360     12.049 / 0.4500        12.012 / 0.4647         0.0521
room    11.238 / 0.5697        11.237 / 0.5702         0.0348
```

**3/3 sequences worse on lpips, 3/3 no better on psnr.** The magnitude tracks
how hard the prior actually moved the scales: desk (0.0257 fitted against 0.0800
asked) pays the most, room (0.0348) is a wash. Nothing recovers, nothing wins.

#### The backend route, assessed rather than assumed

§17.48 and the body of this entry both end at "the SLAM backend, where poses and
scales are solved together". That claim needed checking against the code, and it
comes back weaker than it went in. `splatt3r_slam/backend/src/gn_kernels.cu`
runs Gauss-Newton on the **full Sim3**: `expSim3`/`retrSim3` carry a scale
component, and `act_Sim3`'s Jacobian has the `dpc_ds` column. **Each keyframe's
scale is already a free variable in the backend, and the jitter survives it.**

So the problem is not an unmodelled degree of freedom. It is **observability**:
the backend's residual is keyframe-to-keyframe point matching, and both sides of
every edge come from the same network's prediction carrying the same bias. The
two views agree with each other, so the residual is blind to the error they
share. That is the same phenomenon §17.47 measured as rho = 0.35 -- the same
target view draws the same mistake whoever it is paired with -- seen from the
optimizer's side instead of the model's.

Making it observable requires a constraint that does **not** come from the same
network. Two candidates, both real work:

```
RGB-D depth as a backend residual   the instrument was already ruled
                                    inadequate on TUM fr1 (17.29)
referee scales as a GN prior term   requires editing the CUDA kernel; and
                                    17.48/17.49 measured that pulling the map
                                    toward those scales costs image quality
```

The second is the one that would have to be tried, and this section has already
measured that its target is in conflict with the rendering objective. **That is
the honest state: the seams are diagnosed completely and every route to them
that this codebase can reach has been measured and closed.**

Scope: three sequences, 300 steps, one prior weight for the replication.

### 17.50 The evaluation was never as blind as §17.17 claimed

Kimi's round-18 challenge to §17.48/§17.49: what was measured might not be "the
objectives conflict" but "THIS evaluation cannot see the improvement and can see
the disruption". His test, adopted: re-score the maps that already exist, with
the held-out frames **stratified by baseline**.

The stratifying variable matters and is not the obvious one. §17.16 measured
0.057 m as the distance from a held-out viewpoint to its nearest keyframe. But
what decides whether a per-cluster depth error is visible is the distance to the
clusters that actually **paint** that frame, which the renderer can report:

    baseline_f = sum_k alpha_k(f) * ||t_k - c_f|| / sum_k alpha_k(f)

`scripts/eval_baseline_stratified.py`. Measured:

```
seq     wide frames    median baseline
desk       40/40           0.777 m
room       27/40           0.811 m
360         0/40           0.182 m     (spins in place; cannot test this)
```

**desk's is 13.6x the 0.057 m figure**, and that changes the arithmetic §17.17
built its central claim on:

```
17.17 said   f * b * dd/d^2 = 517 * 0.057 * 0.09 / 4 = 0.66 px   -> "blind"
correctly    517 * 0.78  * 0.09 / 4 = 9.1 px                     -> not blind
```

**§17.17's "the evaluation is structurally blind to depth error" used the
distance to the nearest keyframe where it needed the distance to the painting
clusters.** The two answer different questions -- how much new information a
held-out view carries, versus how visible a per-cluster depth error is -- and
only the second is the one that argument needed. Demoted accordingly.

#### And the verdict it was built to deliver

```
                    all psnr/lpips        WIDE baseline only
desk   base        12.346 / 0.5071        12.346 / 0.5071
       referee     11.443 / 0.5267        11.443 / 0.5267     -0.90 dB, +3.9%
room   base        10.446 / 0.5966        10.293 / 0.6048
       referee     10.278 / 0.5881        10.034 / 0.5943     -0.26 dB, -1.7%
360    base        11.892 / 0.4934        (no wide frames)
       referee     11.834 / 0.4931
```

Pre-registered: the corrected map wins the wide subset by >= 0.3 dB, or the
conflict is real. **It does not win on any sequence**, and on desk -- where every
frame is wide-baseline -- it loses by 0.90 dB with lpips 3.9% worse. room's
small lpips gain does not generalize.

So the conflict survives its best challenge, and it survives it **on stronger
ground than it was first claimed**: the evaluation can resolve a ~9 px
consequence, it is looking at exactly the regime where the correction should
pay, and what it reports is that the metrically better map renders worse.

Three claims in this section now need their baselines re-read: any inference
from §17.17 that depends on 0.66 px, the framing of §17.48's "the evaluation
cannot reward it", and my own repeated statement to the user that the loss is
"structurally blind". The conclusions stand; the reason changed.

#### A third meeting with the same trap

The first run of the stratified evaluator scored 3.019 dB. `resize_img` returns
[-1,1] and the rasterizer outputs [0,1] -- and `eval_map_quality.py` carries a
comment saying, in as many words, that comparing the two spaces "scored 3.16 dB
on a map that renders correctly". A new script hit the documented trap anyway.
§17.40 was the same mismatch in the exposure normalization. **What caught it was
the same thing that has caught every one of these: the number was absurd.**

### 17.51 T3.1: there is no convergence to detect

`--refiner-polish-tol` measured, on a budget where convergence could actually
happen (duty 1.0, 900 s, against the 300 s / duty 0.25 runs that reach only
525-695 steps and are nowhere near done).

```
seq   criterion            steps   psnr     lpips
desk  full budget          3526   14.271   0.3864
desk  tol 0.02, patience 1 1949   14.143   0.3988
360   full budget          4219   13.931   0.4361
360   tol 0.02, patience 1 1678   13.134   0.4535
360   tol 0.02, patience 3 4368   13.909   0.4381   <- never fired
```

Two findings, and the second is the real one.

**Patience 1 is a noise detector.** It saves 45-60% of the steps for 0.13-0.80 dB,
and the stop reasons say why: desk stopped on a window whose mean loss had
*risen* 1.8%, at 1949 steps, while the full-budget map at 3526 was better. A
200-step window mean is still noise against a 3500-step trend.

**Patience 3 never fires.** With the false triggers removed, the criterion runs
the whole 900 s and lands on the full-budget result (13.909 vs 13.931, inside
noise). The loss is still descending at 4368 steps.

**So there is no plateau to detect inside any budget worth spending.** The
premise of T3.1 -- that sequences need 2.4x different step counts and a
convergence test is the only way to size the phase -- was wrong in its second
half: they need different counts because they are all still improving when the
clock stops, so the honest control is the clock, not a criterion.

`--refiner-polish-tol` and `--refiner-polish-patience` stay in the code and stay
**off by default**. Documented as measured-negative rather than removed, because
the negative is the useful part: give the polish as much wall-clock as you can
afford and do not expect it to stop on its own.

### 17.52 The protocol effect, measured instead of argued

§17.50 fixed the evaluation (targets are now exposure-normalized into the map's
colour space) and I immediately started reasoning about what that did to the
existing record. So did Kimi. Both of us were wrong, because we were reasoning
about a quantity nobody had measured.

`eval_map_quality.py --no-exposure` scores against raw targets, i.e. the old
protocol. So the **same map** can be scored both ways and the protocol effect
isolated from everything else. Seven maps, freshly built under the current
defaults, scored twice:

```
seq     p1 d psnr  p2 d psnr   shift | p1 d lpips  p2 d lpips   shift
desk       +3.48      +3.67    +0.19 |     -27.6%      -28.3%   -0.7pp
360        +1.64      +1.86    +0.22 |     -10.5%      -11.3%   -0.7pp
room       +1.74      +1.71    -0.04 |     -16.2%      -16.6%   -0.4pp
rpy        +0.94      +1.11    +0.18 |      -9.1%      -10.2%   -1.0pp

absolute level shift on the polished map (p2 - p1):
  desk +0.053 dB    360 +1.128 dB    room +0.473 dB    rpy -0.104 dB
```

**Paired within-sequence deltas move by at most 0.22 dB and 1 pp.** Kimi's claim
that the mismatch cancels in paired comparisons is confirmed quantitatively --
every conclusion in this section built on an A/B within one sequence stands
without re-measurement.

**Absolute levels move differentially, by 1.23 dB across sequences, and rpy
moves the wrong way** (-0.104 dB). So cross-sequence *absolute* comparisons
under p1 were unreliable, which is the half Kimi got right.

#### Three statements this killed, two of them mine

1. **"The protocol fix is worth +2.36 dB on 360."** I said this twice, including
   in the material I sent Kimi. It came from comparing an old run at ~12.0 dB
   with a new run at 14.36 dB -- **two different configurations**. The controlled
   number is **+1.128 dB**.
2. **"The correlation changes come from the protocol (or from protocol plus
   configuration, inseparably)."** They cannot: the protocol moves a delta by
   <= 0.22 dB / 1 pp, nowhere near enough to move a rank correlation from -0.96
   to -0.82 or +0.07 to +0.714. Those changes are **configuration**, and the
   attribution is now clean rather than hedged.
3. **Kimi's prediction that the fix would STRENGTHEN the -0.96 correlation**,
   via "rotation sequences drift more, so their gains were more attenuated". The
   measurement shows the protocol's effect on the gains is nearly uniform
   (0.4-1.0 pp), with no such differential compression. It weakened instead --
   for a different reason.

#### The seven-sequence table, re-measured under p2 and the current defaults

```
seq      ratio   d psnr   d lpips
xyz       15.0    +2.10    -27.1%
desk       3.4    +3.67    -28.3%
plant      2.6    +2.61    -12.7%
teddy      2.2    +2.81    -13.8%
room       1.9    +1.71    -16.6%
360        1.1    +1.86    -11.3%
rpy        0.6    +1.11    -10.2%

pooled  +2.27 dB / -17.1%      (p1 recorded +2.21 dB / -16.7%)
spearman(ratio, d lpips) = -0.821   (p1: -0.964)
spearman(ratio, d psnr)  = +0.714   (p1: +0.07)
```

The headline reproduces to within 0.06 dB and 0.4 pp. **The polish result is
protocol-robust.**

The two correlations are not, and by the isolation above the cause is the
configuration change, not the fix. **§17.34's supporting claim that "psnr is
blind to this, +0.07 flat" does not survive the new defaults: psnr now tracks
the motion ratio at +0.714.** Since §17.34/§17.36's argument leans on the
psnr/lpips split, that leg needs re-analysis under the configuration that is
now the default -- flagged, not resolved here.

#### Method note

The instrument that settled all of this is one flag and an afternoon. Before it,
three people-hours of argument had produced three confident statements about the
protocol's effect, and all three were wrong. The reason they were wrong is the
same in each case: **a comparison in which more than one thing differed**, read
as though only one had.

That is the fourth distinct form of §17.31's failure mode, and the first that
survived being *noticed* -- I flagged "config and protocol both changed, cannot
separate" and then, two sentences later, attributed the change anyway.

### 17.53 The streak lever is not a streak remover, and the images say so

`--refiner-streak-opacity 0.5` became a default on a 3-sequence paired A/B
(lpips -12.7% / -6.6% / -1.3% on desk / room / 360). The story attached to it --
"it hides ray-elongated Gaussians at depth discontinuities, so the win tracks
how many object boundaries a scene has" -- is wrong, and two instruments say so
independently.

#### The spatial test (Kimi's round-19 design)

Build the map twice from ONE keyframe blob, fade off and on; the per-Gaussian
fade factor is then the ratio of the two opacity vectors. Render the **fade
deficit** (1 - fade, weighted by the unfaded opacities) and bin held-out pixels
by it, scoring each bin with spatial LPIPS.

```
desk, fade deficit bin   pixels   mean lpips gain
       0.00-0.02          8.3%       -0.00121
       0.02-0.05          1.6%       +0.00036
       0.05-0.10          1.9%       +0.00264
       0.10-0.20          3.7%       +0.00244
       0.20-1.01         84.6%       +0.01434
```

**The dose-response holds**: the gain rises with the deficit and is negative
where nothing was faded. So the lever does act through the Gaussians it fades --
that part of the mechanism is confirmed, with millions of pixels of power rather
than a 3-point rank agreement.

**But the coverage refutes the story:**

```
seq    fade coverage   mean deficit   top-bin pixels   top-bin gain   A/B gain
desk      91.8%           0.529           84.6%         +0.01434      -12.7%
360       63.6%           0.390           50.9%         +0.00532       -1.3%
room      98.4%           0.651           81.5%         -0.00560       -6.6%
```

It fades **64-98% of all Gaussians** by 39-65% of their opacity. This is a
near-global, gradient-weighted opacity reduction, not a sparse edge treatment.
And coverage does not order the benefit: room fades the most and gains less than
desk.

#### The images, which is where the description actually broke

`logs/streak_png/`, one held-out view, rendered from the baked map:

- **The fade-deficit image is a broad wash, not an edge map.** If the mechanism
  were elongated Gaussians at depth discontinuities, that image would be thin
  bright structures along object boundaries. It is bright almost everywhere.
- **Faded vs unfaded looks like de-hazing**: the keyboard's key array and the
  mouse emerge from a white haze. That is what lowering foreground opacity to
  let background structure through looks like, not what erasing a streak looks
  like.
- **And the baked map is not "slightly flawed" -- it is a smear.** At 10.5 dB
  the desk render is black blobs and white haze against a ground truth of
  monitors, keyboard and papers. The polish's +3.67 dB is the difference between
  unrecognizable and recognizable. **I had been reasoning about "artifacts" on
  an image I had never looked at.**

**Corrected description, for the README and anywhere else it appears:** a
global opacity reduction weighted by how long each Gaussian is relative to its
local surface sampling; its effect is de-hazing. Keep the flag and the default
(3/3 sign, cost <= 0.22 dB), drop the "trailing streak" framing.

#### One open question, with a designed discriminator

room's *immediate* gain in the high-fade region is negative (-0.0056) while its
post-polish A/B gain is positive (-6.6% lpips). Those measure different things:
the spatial test scores the baked map, the A/B scores after 300 s of
refinement. So the benefit may be **mediated by the optimizer** -- fading gives
refinement a less crowded starting point rather than removing an artifact
directly. The discriminator is one paired online A/B with the polish disabled in
both arms: if room degrades without polish and improves with it, mediation is
established.

### 17.54 Replica: a fifth family, and the recipe transfers at the same effect size

New family end to end: `splatt3r_core/data/replica/replica.py` (adapter),
registered in `FAMILIES`, coverage cached (8 scenes x 1700 train frames, ~3 min
-- much faster than euroc/eth3d because rendered depth has no dropout to work
around), and `ReplicaDataset` added to the SLAM-side loader so the pipeline can
actually be run on it.

Chosen over the alternatives for a methodological reason, not novelty: Replica
is *rendered*, so poses are exact, depth is complete (99.8% of pixels valid
against TUM's sensor dropout), and exposure is constant. It is the only family
whose supervision carries no sensor noise.

#### Head-only, 40 epochs, matched to the other families' budget

```
                 val/loss    mse     psnr     lpips
base              0.0757   0.0390  18.8575   0.1466
best              0.0557   0.0255  20.6995   0.1199
best epoch          37       39       39       37
delta             -26.4%   -34.6%  +1.84 dB  -18.2%
```

The last ten epochs sit in 0.0557-0.0570, so this one is on the plateau -- 6
epochs was not (it reached +0.79 dB, and the script's own help text says the
6-epoch default "was chosen to make a run take minutes; neither route had
converged by then").

#### The comparison that matters, with the denominator handled

```
                 psnr      lpips rel    lpips ABS
tum 40 epoch    +1.77 dB    -10.2%      -0.0285
replica 40 ep   +1.84 dB    -18.2%      -0.0267
```

**psnr gains are within noise of each other, and the absolute perceptual
improvements are nearly identical -- TUM's is marginally larger.** The apparent
"Replica improves lpips almost twice as much" is entirely the denominator:
TUM's base lpips is 0.279 because it includes a sensor-noise floor, Replica's
is 0.147 because rendered ground truth has none. Kimi's round-20 instruction to
compute absolute deltas before choosing an explanation was correct and answered
the question in one subtraction.

**Verdict: the head-only recipe transfers to a new dataset family at the same
effect size.** It does NOT do better there; saying so would be a denominator
artifact.

#### The images, which complicate the scalars

`logs/render_replica_40/`, identical seeded samples, GT / base / head:

- **head resolves structure that base smears**: the crane in the mural and the
  door frame's wood grain appear only in the head render.
- **head softens fine texture that base keeps**: the world-map coastlines are
  fine white filigree in GT and in base, and blobby in head. Screen bezels,
  rock texture and object outlines are all smoother.

Both metrics reward that trade -- mse -34.6% is far more than sharpening could
buy and is consistent with geometry being placed better, while LPIPS-AlexNet
weights structure over speckle. **But it is a trade, and the scalars hide it.**
The same tension is already on record for TUM's route B vs C ("LPIPS is a proxy
for the blur complaint, not the complaint itself").

Absolute levels are not comparable across families and the images show why:
Replica's *base* render is already a recognizable photograph at 18.9 dB, while
TUM's baked map at 10.5 dB is a smear. That is the rendering, not the model.

### 17.55 A train/inference mismatch the val split could not see

Registering Replica as a family, I copied tum's training resolution: (512, 384).
Replica is 1200x680. The SLAM side's `resize_img` (long side 512, crop to a
multiple of 16) therefore feeds the network **512x288**, while training
centre-cropped to 4:3 and trained 40 epochs on **512x384** -- a different crop of
a different aspect ratio.

```
                     val split          SLAM deployment (office0)
head vs base       +1.84 dB / -18.2%    baked -4.0 dB, refined -0.57 dB / +35% lpips
```

Fixed to (512, 288) and retrained; the val gain rose to **+2.17 dB / -20.6%**,
and the deployment A/B moved from -0.57 dB to **+0.08 dB**, with the baked
regression shrinking from -4.0 dB to -0.97 dB. Diagnosis confirmed by the fix.

**The val split is computed with the training-side preprocessing, so it is
structurally blind to this entire class of error.** Same shape as §17.17 (the
evaluation cannot see depth error) and §15's proxy-vs-online gap. Kimi's
formulation, adopted as a standing rule:

> Any evaluation that shares preprocessing with training is structurally blind
> to input-pipeline errors. Every release gets one end-to-end check.

`scripts/test_preprocess_roundtrip.py` is the cheap guard -- it compares what
the SLAM path feeds the network against what the family trains on, per family.
Kimi rejected my first proposal (assert the shape) on the grounds that
crop-vs-letterbox-vs-squash, channel order, dtype range and the exposure flag
are all shape-preserving or shape-adjacent and all in the same class; the test
compares tensors, starting with shape. All four families with data on disk pass.

#### The larger finding, which outlives the bug

Even at the correct resolution:

```
                       replica office0
val split            +2.17 dB / -20.6%
deployment, baked    -0.97 dB / +23%
deployment, refined  +0.08 dB /  -1.8%
```

**The head's val-split gain does not survive deployment.** Two causes, both
worth carrying:

1. **Refinement dominates.** Both arms run ~2400 polish steps and converge to
   ~26.5 dB; the head's head start is worth 25x less than the val split implies.
   That is the other face of §17.52's +2.27 dB: when the polish is worth 4.7 dB,
   a 2 dB better initialization is largely absorbed.
2. **The val split scores a different object.** It scores the decoder's render
   of one two-view prediction. The SLAM bake accumulates many keyframes through
   confidence gating, the band-limit floor and the thinning prior. "Better at
   two-view rendering" is not "better after accumulation and gating".

This applies to all four pre-existing families: **none of their heads has ever
had a deployment A/B.** Their resolutions happen to be right, but that is not
the same as verified. Per Kimi: one deployment A/B per family as a release gate,
priority tum (the flagship claim), the rest guarded by the round-trip test.

### 17.56 The streak lever is a thinning prior, and the selector is worse than nothing

Two controls Kimi designed in round 21, both run.

#### The uniform control: the elongation criterion earns nothing

Same blob, three arms, offline refinement. `uniform` multiplies EVERY opacity by
(1-D) with D matched to what the selector removes on average.

```
desk             iter 0    300     600     900    1200
off              0.5071  0.4200  0.3952  0.3770  0.3723
elongation       0.4979  0.3989  0.3842  0.3675  0.3639
uniform D=.529   0.5058  0.4021  0.3855  0.3677  0.3644
```

elongation and uniform coincide at every step count; the endpoint gap is 0.0005,
**0.14% relative against a measured 0.17% floor (§17.51)**. The selector's
contribution is indistinguishable from zero.

```
room             iter 0    300     600     900    1200
off              0.5964  0.5696  0.5309  0.5125  0.5018
elongation       0.6030  0.5433  0.4999  0.4823  0.4746
uniform D=.651   0.5950  0.5117  0.4765  0.4647  0.4604
```

On room the selector is not merely useless: **uniform beats it by 3.0%**, far
above the floor. Aiming the fade by elongation puts it in the wrong places.

**So the lever is a content-blind global thinning prior.** The baked map is
systematically over-crowded and over-opaque, and simply removing ~50% of every
Gaussian's opacity recovers most of the perceptual damage. `min(1, K*pitch/
max_scale)` should be reported as unnecessary. Kimi's footnote, kept: at iter 0
elongation does beat uniform on desk (0.4979 vs 0.5058), so a pre-polish
advantage exists and is erased by 300 steps; its value at deeper fades is
unmeasured.

#### The mediation question, answered

room's *immediate* effect was negative (§17.53) while its A/B gain was positive.
The crossover in the table above is the answer: elongation starts **worse** than
off (0.6030 vs 0.5964) and ends **better** (0.4746 vs 0.5018).

**The benefit is produced by the refinement, not by the render.** Thinning gives
the optimizer a less entangled starting point; it does not remove an artifact.
That retires the "trailing streak" framing completely -- the name, the mechanism
and the selector were all wrong, and only the effect was real.

### 17.57 Cross-family transfer: the diagonal wins, and a bootstrap kills half a pattern

5 families x 5 heads, one protocol (psnr/lpips):

```
family         base      tum-h    7scenes-h   euroc-h   eth3d-h  replica-h
tum        15.09/.279  16.83/.251 15.47/.323 13.00/.413 13.45/.368 15.48/.263
7-scenes   13.71/.282  14.80/.278 15.86/.257 13.79/.358 14.12/.333 13.94/.277
euroc      11.44/.301  11.29/.447 11.74/.429 15.13/.251 10.90/.480 12.03/.313
eth3d      10.85/.371  11.39/.383 11.91/.364 11.15/.452 14.00/.327 11.10/.365
replica    18.86/.147  16.74/.324 17.51/.293 16.82/.277 14.43/.409 20.67/.120
```

**Every diagonal wins (+1.74 to +3.69 dB), and the off-diagonal mostly costs
lpips** -- on replica every foreign head roughly doubles it against base. The
headline is: head-only tuning is family-specific; do not cross-deploy a head.

#### The exception, and what happened when it was measured properly

The replica column looked like it improved psnr on 4/4 foreign families, which
I was ready to call a signal. Kimi flagged two problems: that column used the
**wrong-resolution head** (§17.55), and no cell had a floor. Recomputed with the
fixed head, per-image, with bootstrap CIs over 2000 resamples:

```
family        n    base          replica288     d psnr [95% CI]        d lpips % [95% CI]
tum         150  9.68/0.568    9.90/0.564    +0.22 [-0.18,+0.62]   -0.6 [-4.3,+3.3]
7-scenes    175 10.88/0.548   11.44/0.534    +0.56 [+0.21,+0.88]   -2.5 [-5.2,+0.4]
euroc       158  9.91/0.467   10.41/0.437    +0.50 [+0.14,+0.86]   -6.5 [-9.9,-2.9]
eth3d       150  9.72/0.538    9.70/0.540    -0.02 [-0.44,+0.40]   +0.5 [-3.2,+4.3]
replica     160 10.73/0.466   11.66/0.431    +0.93 [+0.48,+1.41]   -7.6 [-12.1,-2.7]
```

**2 of 4 significant, 1 null, 1 not significant** -- and only euroc's lpips
clears zero. "4/4 transfer" was the wrong-resolution head plus the absence of a
floor. **Without the bootstrap I would have written a claim that two cells do
not support.**

What survives is narrower and matches Kimi's more boring alternative: a head
trained on noise-free, complete-depth supervision transfers *at the psnr level*
to some families, most plausibly because dense valid depth teaches denser and
more confident predictions rather than because clean supervision teaches
transferable structure. Not a mechanism claim yet.

(Absolute levels in the two tables differ -- different harness, val subset and
masking. Each is internally consistent; neither is comparable to the other or to
the training logs.)

### 17.58 The base checkpoint's opacity output is collapsed, and that is the largest single map defect

Chasing §17.55's baked regression produced the most explanatory finding in this
section. Kimi proposed two causes in round 23; both are impossible by
construction, and eliminating them found the real one.

#### Both proposed causes are structurally impossible

`head_params()` trains only `gaussian_dpt`. In the inference path `X` and `C`
come from `r["pts3d"]` and `r["conf"]` -- the **frozen** pointmap head. The
Gaussian head cannot change depth, confidence or density at all. Measured on
Replica office0, base vs head, same windows:

```
                 scale spread  neighbour  within-view  median ratio  mean conf  gate
base                4.68%       2.61%       1.03%        0.8189      11.614    99.4%
head                4.68%       2.61%       1.03%        0.8189      11.614    99.4%
```

Identical to every digit. This also kills Kimi's round-22 mechanism for the
cross-family transfer ("dense valid depth teaches denser/more confident
predictions"): whatever transfers travels through **Gaussian shape and
appearance only**.

#### What the head actually changes

```
TUM      scale med   p90    max-axis med   opacity med   frac>0.9   frac<0.1
base       2.102   6.239       2.791          1.0000      100.0%      0.0%
head       1.347   7.330       2.851          0.5909        9.3%      7.5%

Replica  scale med   p90    max-axis med   opacity med   frac>0.9   frac<0.1
base       3.856   9.778       6.723          1.0000      100.0%      0.0%
head       1.902  22.768      13.485          0.9993       99.8%      0.0%
```

**The base checkpoint predicts opacity 1.0 for essentially every Gaussian.** It
never learned to use partial transparency. That reframes §17.56's thinning prior
completely: it is not a heuristic patch, it **supplies an output the network's
head collapsed on**.

And the two heads differ in exactly the way the deployment results do:

```
              baked                    refined
TUM base    10.52/0.5605            14.11/0.4065
TUM head    12.11/0.5034  (+1.59)   14.16/0.3669  (-9.7% lpips)
Replica base 21.80/0.2381           26.50/0.1004
Replica head 20.83/0.2927  (-0.97)  26.58/0.0986  (-1.8% lpips)
```

The TUM head un-saturated itself and bakes better. The Replica head stayed
saturated, inflated its scale tail instead (p90 +133%, max axis +100%), and
bakes worse.

#### The causal channel, corrected

My first statement -- "Replica's exact complete depth removes the pressure to
learn opacity" -- is wrong on the data path, and Kimi caught it: **that depth
is never shown to the network.** The head's geometry comes from the frozen
backbone; the dataset depth only builds the loss mask. Two of my three named
drivers are therefore inert: sensor dropout never enters training at all, and
exposure variation *cannot* teach opacity because opacity is view-independent
and so cannot hedge per-view brightness.

The surviving channel, in his words:

> On clean rendered images the backbone's own geometry is confident, multi-view
> photometric conflict is rare, and opacity saturation -- the head's default
> collapse -- is never punished. On real images (blur, auto-exposure, weak
> texture) the backbone's geometry is uncertain, conflicting evidence lands on
> the same rays, and partial opacity is the only per-ray softness the model
> class has.

#### The 2x2: two defects, two stages

On the head-baked Replica map, no retraining, cap set at the base head's p90:

```
                 iter 0 (baked)        iter 600 (refined)
none          20.172 / 0.3311        25.351 / 0.1649
thin (D=.45)  20.528 / 0.3247        26.404 / 0.1110
cap           21.146 / 0.2631        25.901 / 0.1373
thin + cap    21.511 / 0.2611        27.308 / 0.0765
```

- **The scale tail damages the render**: at bake the cap is worth -20.5% lpips
  against thinning's -1.9%. Kimi's pre-registered ">=60% of the baked deficit"
  call is confirmed.
- **Saturation damages optimizability**: thinning is worth only -1.9% at bake
  but -32.7% after 600 steps. Front layers at alpha~1 block the back layers'
  gradient path, so the map cannot reassign content during refinement.

The interaction is real but modest and gets one sentence: multiplicative null
-43.9%, additive -49.4%, measured -53.6% -- better init times better dynamics
compound. **The finding is the stage split; the interaction is its footnote.**

#### The falsification test, which is what makes this causal

Same code, same 2x2, same 600-step budget, on a **TUM-head** map. If these
levers were generic map repairs they would buy the same on both heads.

```
                     iter 0            iter 600
TUM head   none   12.354/0.5071      14.098/0.3952
           thin   12.565/0.5057      14.011/0.3871   -2.0% lpips, -0.09 dB
           cap    12.372/0.5037      13.938/0.3937   -0.4% lpips, -0.16 dB
           both   12.634/0.5011      13.853/0.3878   -1.9% lpips, -0.25 dB
```

**-53.6% on the Replica head, -1.9% on the TUM head, with a psnr cost.** A
factor of thirty, in the direction the mechanism predicts: **the external levers
are worth exactly the work the head did not do.**

And Kimi's load-time diagnostics separate the two families without reference to
the answer:

```
p90 ratio (head/base)    Replica 2.33  -> arm the cap      TUM 1.17  -> do not
saturation fraction      Replica 99.8% -> arm the thinning TUM  9.3% -> do not
```

#### What this licenses

- **Uniform thinning ships as a default** (four contexts, never negative), but
  the record should say what it is worth where: ~2% on a head that already
  un-saturated, 30%+ on a saturated one.
- **The scale cap ships conditional on the diagnostic**, not unconditionally: on
  the flagship family it costs 0.16 dB psnr for 0.4% lpips.
- **§17.55's "the head route has no deployment value" is retracted.** It was
  measured on a saturated head. TUM's +1.59 dB baked / -9.7% refined makes the
  value conditional, and the condition is now mechanistic: **a head earns its
  keep iff it learns transparency.**
- The unifying sentence, Kimi's: *the base head's opacity collapse is the single
  largest map-quality defect in this system; thinning is the patch,
  head-training on real data is the root fix, and Replica-style clean
  supervision is contraindicated for learning exactly this.*

Two claims deliberately NOT made yet: that thin+cap "beats every deployment
number" (the 2x2 is offline/600-step and the deployment arms are online/2316 --
crossing harnesses is the error §17.52 exists to prevent, and I made it once
here before catching it), and that the collapse is best fixed at training time
(the opacity-penalty retrain is running; if a trained-in un-saturation merely
matches a one-line injection fade, the honest report is that the collapse is
real and fixable in either place, and the injection fix is free).

### 17.59 The dot lattice has two production paths, and the floor guards only one

Kimi's round-25 safety check on the thin+cap combination: the lpips numbers were
large enough to be suspicious, and both levers REMOVE material. His prediction
was holes (black fraction). It is not holes.

```
Replica head map      final self-psnr   final hp_alpha   lpips@300
none                     46.50 dB         0.00075         0.2234
thin (D=0.45)            42.53 dB         0.00403         0.1524
cap (base p90)           45.73 dB         0.00096         0.1903
thin + cap               40.47 dB         0.00584         0.1105
```

**The thinning re-introduces the dot lattice**: 5.4x on its own, 7.8x combined,
with 4-6 dB of self-consistency lost. **The cap is clean** (0.00096 against
none's 0.00075). And the band limit was ON in all four arms.

Kimi's mechanism, sharpened from mine and adopted verbatim as the record:

> **Saturation was masking the lattice.** With alpha ~ 1 the front blob alone
> caps the accumulated alpha everywhere, so the lattice-frequency modulation of
> the weight field never reaches the image; at alpha ~ 0.55 the cap releases and
> the G-sum's lattice harmonic shows up in A. The floor was never bypassed -- it
> guards the scale path; this is the alpha path, and nothing guarded it.

**The recordable sentence: the dot lattice has two production paths, and each
needs its own guard.** §17.2 built the guard for the sampling path and this is
the first time the other one has been seen.

Neither headline metric can see it (§17.2: psnr moved 0.01 dB across a sweep
that changed hp_alpha 13.6x), so the -31.8% lpips gain does not contain this
cost. Threshold, bracketed from §17.2's own calibration rather than invented:
hp_alpha 0.009 was visibly dotty, 0.0026 read clean, so 0.004 is **marginal --
probably faintly visible at GUI magnification on flat regions, invisible at
native resolution.** Stress-view images are the only way to settle it and are
owed.

#### Which reverses the shipping call we had agreed

Two rounds earlier Kimi and I had settled on "thin ships as default, cap
conditional". The measurements say the opposite on both counts:

- **cap** carries no lattice cost and its defect is present on 4 of 5 heads
  (tail ratio > 2);
- **thin** carries a 5.4x lattice cost invisible to both headline metrics.

Revised, with his amendment that both calls still rest on n=1 in the direction
that matters:

- **cap: conditional-by-diagnostic, pending confirmation on the other armed
  heads.** Tail ratio > 2 says the defect is present, not that capping helps --
  §17.16's clamp was a defect on one map and load-bearing coverage on another.
  Three offline runs (7-scenes, euroc, eth3d), cap on/off, are the price of a
  default-level claim.
- **thin: conditional, and the condition is crowding, not saturation** (§17.60).
  Its lattice cost is recorded next to its lpips gain.

### 17.60 Two levels: head saturation is not map crowding

room's head map is un-saturated (10.4% of Gaussians above 0.9) and thinning
still bought -8.2% there, against desk's -2.1%. That is a counterexample to
§17.58's diagnostic as stated, and the resolution is a level confusion:
**saturation is a per-Gaussian property, crowding is a map property.** Ten
layers at opacity 0.64 still accumulate to alpha ~ 1.

Measured directly -- per-cluster alpha renders, counting clusters above 0.1 at
each pixel:

```
map            gaussians  layers/px  acc alpha   thinning benefit   head frac>0.9
desk-base      2,396,900     6.32       6.17        -13.6%            100.0%
room-head      8,774,161     6.89       6.66         -8.2%             10.4%
desk-head      1,860,034     4.84       4.45         -2.1%             12.5%
360-head       7,267,700     3.41       3.16         -0.4%             ~10%
```

**Crowding orders most of it** (3.16 -> 4.45 -> 6.17/6.66 against -0.4% ->
-2.1% -> -13.6%/-8.2%), and **saturation explains the remaining swap**:
desk-base and room-head have nearly equal accumulated alpha (6.17 vs 6.66) but
desk-base benefits 1.7x more, and it is the saturated one.

So the load-time diagnostic needs two numbers, not one:

```
accumulated alpha (map)      how much haze is there to remove
frac(opacity > 0.9) (head)   how much of it the head already removed
p90 ratio vs base (head)     whether the tail needs capping (independent axis)
```

This also retires the explanation I had given for 360's null. I had said it was
"few depth discontinuities because the camera spins in place", which was built
on the already-falsified streak framing. **The measured reason is that 360's map
is the least crowded of the four (3.16).**

### 17.61 The missing control, and a pre-registration I made with the wrong metric

#### The provenance error

`logs/frames_head/` is baked with `checkpoints/head_only_long/tum/head_best.pt`
-- the run log says so in one line. **Every offline thinning sweep in §17.53 and
§17.56 therefore ran on TUM-HEAD maps, not base maps**, and I had been reading
them as base. Kimi caught it from a four-digit coincidence: round 22's desk
iter-0 (12.3544/0.5071) is identical to the TUM-head 2x2's none arm.

Third provenance error in this session, and the only one with no internal
warning at all:

```
logs/online_new/         had no polish phase          numbers absurd, self-alarming
logs/online_polish_new/  two arms, identity unclear   lpips reversed, weak alarm
logs/frames_head/        the "head" in the name is    NO alarm -- every number
                         literal                      plausible
```

**Rule: before reusing any artifact across sessions, grep its generating log for
the configuration. The directory name is not evidence.**

The missing measurement, run:

```
TUM base map (saturated)   psnr    lpips    self-psnr  hp_alpha
none                      14.075   0.4759    44.81      0.00104
thin                      14.132   0.4110    40.87      0.00168   -13.6%
cap                       13.964   0.4654    44.08      0.00119    -2.2%
thin+cap                  14.059   0.3998    39.92      0.00208   -16.0%

TUM head map (un-saturated), same sequence, same budget
none                      14.098   0.3952
thin                      14.011   0.3871                          -2.0%
```

**Same sequence, same code, same 600 steps; the only variable is whether the
head un-saturated: -13.6% against -2.0%, a factor of 6.8.** The claim no longer
rests on a cross-family comparison in which the dataset, the rendering and the
head all differed at once.

#### The falsification, and my own error inside it

The opacity-penalty head (`relu(o - 0.9)`, W=1.0) came back **un-saturated by
the agreed metric**: frac>0.9 99.8% -> **0.0%**. Its scale tail deflated with it
(p90 ratio 2.33 -> 1.20), which refutes Kimi's prediction that the penalty
"doesn't touch scales" -- opacity and scale are coupled through the coverage
task, since a head that may use transparency no longer needs giant blobs.

His pre-registration: on such a head, thinning's value collapses to <= 2%.

```
penalty head map      iter 0            iter 600
none               21.788/0.2801      27.102/0.0959
thin               22.179/0.2742      27.494/0.0758   -21.0%
cap                21.856/0.2490      27.344/0.0799   -16.7%
thin+cap           22.276/0.2478      27.703/0.0665   -30.7%
```

**-21.0%, not <= 2%.** So "the external levers are worth exactly the work the
head did not do" is false as stated.

Why: `relu(o - 0.9)` penalises only above the knee, so the head parked just
under it -- **median opacity 0.84**, against the TUM head's 0.66 reached with no
penalty at all. The >0.9 fraction said "un-saturated"; the accumulated alpha did
not.

**And I had derived the correct metric one section earlier.** §17.60 exists
because room falsified the saturation-only diagnostic; I wrote that down and
then made the next pre-registration with the metric it had just replaced. The
falsified prediction is not the lesson. **Deriving a better instrument and then
not using it is.**

#### What stands

```
none-arm of the penalty head's map   0.0959    (plain Replica head: 0.1649, -42%)

deployment A/B, Replica office0
base            26.8580 / 0.1000
penalty head    27.0589 / 0.0848    +0.20 dB / -15.2% lpips
plain head      26.5843 / 0.0986    +0.08 dB /  -1.8%
```

**Training with an opacity-saturation penalty turned a marginal head into a
clearly positive one in deployment.** Kimi's "simultaneously the evidence and
the fix" framing pays off on the fix side.

Revised account: three interventions that are **partially complementary, not
substitutes**, all lowering the same quantity:

```
training-time penalty   1.00 -> 0.84 median      -42% on the map's none arm
injection-time thinning x0.55 -> ~0.46 effective -21% on top of that
scale cap               independent axis         -17%
```

The target is **accumulated alpha in the baked map**; which lever lowers it is
an engineering choice. A properly dosed penalty (`relu(o - 0.6)`, aimed at the
level the TUM head reached unaided) is running as the sharp version of Kimi's
prediction: if thinning finally collapses on THAT head, training can substitute
for injection; if it does not, injection-time thinning is unconditional.

### 17.62 Deployment numbers, and a floor that turned out to be free

```
thin+cap, Replica office0, online       refined psnr / lpips
plain head                              26.5679 / 0.1000
thin+cap                                26.5445 / 0.0812   -18.8% lpips, -0.02 dB
```

Kimi pre-registered 26.9-27.2 dB / 0.082-0.090: **lpips almost exactly right,
psnr outside his range.** And my earlier claim that the offline 27.308/0.0765
"beats every deployment number" was a cross-harness comparison -- the error
§17.52 exists to prevent, made three sections after writing it. The online
number is 26.54/0.0812; the offline map is genuinely better, as he predicted,
because the online loop adds its own damage.

#### The whole-pipeline floor

§17.51's floor covers refiner stochasticity under a **fixed bake**. Kimi's
round-20 point was that it therefore licenses nothing about absolute numbers,
and asked for three fresh SLAM runs. Bought:

```
run 1  14.2103 / 0.4000    bake md5 1efe21e4ca18
run 2  14.2098 / 0.4003    bake md5 1efe21e4ca18
run 3  14.2120 / 0.4000    bake md5 1efe21e4ca18
```

**The bakes are byte-identical across independent runs**: tracking, backend,
keyframe selection and baking are fully deterministic. So the whole-pipeline
floor equals the refiner floor, and §17.51's 2-sigma bounds (psnr 0.031 dB,
lpips 0.17%) now cover absolute claims as well. The purchase was cheap and it
closes every future argument about which floor applies.

### 17.63 CORRECTION to 17.59: the images do not support the lattice reading

§17.59 recorded that thinning "re-introduces the dot lattice", from hp_alpha
rising 5.4x. Kimi's standing instruction -- and this project's own rule -- was to
render stress views and look. Done, and it does not support the claim.

Method matters here: the first crop was **guessed** and landed on a
well-covered region where the accumulated alpha is still ~3.7 after thinning
(6.66 layers x 0.55), so the cap never released and nothing could show. The
second pass located the 96x96 window with the **most high-pass alpha energy** --
i.e. where the metric itself says its number comes from -- and cropped there.

At that window, alpha field, contrast-stretched 6x about the mean:

```
none    a clear periodic dot lattice
thin    smooth gradients and one soft low-alpha wedge; NO periodic structure
```

**The unthinned map is the one with the visible lattice.** And the coverage cost
that would have been the other explanation is absent:

```
                black (alpha<0.1)   mean alpha
none                 0.00%            0.9997
thin                 0.00%            0.9995
thin tau=0.75        0.00%            0.9998
thin dither          0.00%            0.9994
```

So the wedge is not a hole -- it is a dip from ~1.0 to ~0.9 that a 6x stretch
renders black. Kimi's original prediction (holes, black +<0.5pp) is confirmed
correct in the sense that matters: **there are none.**

**What stands, and what does not:**

```
stands       hp_alpha rises 5.4x after thinning (measured, reproducible)
             self-psnr falls 4 dB (measured)
             the tau=0.75 arm halves hp_alpha and restores self-psnr to 45.99
does NOT     that the rise is the dot lattice of 17.2
             "saturation was masking the lattice" as the mechanism
             the shipping reversal insofar as it rested on the lattice reading
```

The high-pass alpha statistic is **not lattice-specific**: it also counts
coverage boundaries and smooth alpha gradients, and after thinning those
dominate it. §17.2 validated it against a visible dot pattern on a *base* map
where the lattice was the dominant high-frequency content; that validation does
not transfer to a thinned map where it is not.

**Open, and deliberately not resolved with a replacement story**: why the
unthinned map shows the lattice and the thinned one does not, when the argument
from clamping predicts the reverse. Two candidates, neither tested -- at alpha~1
the front Gaussian alone determines the pixel so its own lattice prints
directly, and blending many offset layers averages the ripple out; or the
stretch visualisation is not comparable between arms because their alpha
distributions differ. Both are checkable; neither is checked.

#### The methodological point, which is the reason this entry exists

The lattice reading survived a numeric safety check, a mechanism from Kimi, a
shipping-recommendation reversal and 400 lines of write-up, and died to the
first image. **§17.2 built hp_alpha precisely because psnr could not see the
lattice; using it later as though it were a lattice detector on a map with
different content is the same class of error the metric was built to prevent** --
an instrument trusted outside the regime where it was calibrated.

The tau=0.75 and dither arms therefore have no verdict yet: they were arms of an
experiment against an artifact that has not been shown to be there.

```
arm            psnr@300  lpips@300  self-psnr  hp_alpha
none            23.850    0.2234     46.50     0.00075
thin            25.273    0.1523     42.52     0.00403
thin tau=0.75   25.018    0.1790     45.99     0.00202
thin dither     25.276    0.1545     42.72     0.00476
```

Kimi's pre-registration for tau=0.75 (hp_alpha <=0.001, self-psnr 45-46, lpips
0.155-0.165): self-psnr exact, hp_alpha halved but not to floor, lpips 0.179
against a predicted 0.155-0.165 -- the blur tax is ~17%, far above his 2-4%
estimate. Recorded as measurements, not as a verdict on a lattice.

### 17.64 State of play — read this first when resuming

Written as a handoff. Everything below is either running, queued with a
pre-registration, or a known debt.

#### In flight

```
GPU0  exp_head_only --family replica --opacity-target 0.6  (epoch 22/40)
      out: checkpoints/head_only_long/replica288_opac06
      WHY: 17.61's penalty used relu(o-0.9) and the head parked at median 0.84,
      so thinning still bought -21% on it. This aims at the 0.66 the TUM head
      reached unaided. PRE-REGISTERED: if thinning finally collapses to <=2% on
      THIS head's map, training-time can substitute for injection-time; if not,
      injection thinning is unconditional. Run the 2x2 on its blob to decide.

GPU1  exp_head_only --family replica-noisy  (epoch 39/40)
      out: checkpoints/head_only_long/replica_noisy
      WHY: Kimi's causal test for why clean supervision never teaches opacity.
      PRE-REGISTERED: frac(op>0.9) should fall from 99.8% toward TUM-like
      levels. FIRST NUMBER TO READ is that fraction, via /tmp/gp.py's pattern
      (5 pairs, median opacity + frac>0.9 + scale p90).
      The CONTROL is `--family replica-photo` (brightness jitter on one view
      only), NOT YET RUN: opacity is view-independent so it cannot hedge
      per-view brightness, and Kimi predicts it will NOT un-saturate. Only both
      together prove the channel is geometric conflict rather than photometric.
```

#### Queued, with pre-registrations already on the record

```
1  cap on 3 more armed heads (7-scenes, euroc, eth3d; tail ratios 2.17/2.52/
   3.26). Kimi: "armed by diagnostic" is proven benefit on ONE of four armed
   heads. Predicted: baked lpips improves where ratio > 2, most on eth3d. If any
   armed head gets WORSE, the threshold or the diagnostic needs revision.
   Three offline 600-step runs, blob per head.
2  the 17.63 open question: why the UNTHINNED map shows the lattice and the
   thinned one does not. Two untested candidates recorded there.
3  replica-photo (the control half of the causal test above).
4  17.34 re-analysis: psnr now tracks the motion ratio at +0.714 raw / +0.623
   partialling out map size, p=0.071. The "psnr is blind to this" leg of
   17.34/17.36 does not survive the current defaults and needs rewriting.
5  head-training seed ensemble: every head result in this file is single-seed,
   while the refinement seed sigma is known (17.51). The cross-family lpips
   gaps are smaller than that sigma.
6  Replica mesh controlled-trajectory experiment: the release ships 8 scene
   meshes, so the same-scene rotation-vs-translation design that 17.38 needed
   is finally runnable. It is the only design that breaks the motion/scene
   confound by construction.
7  deployment A/B for 7-scenes, euroc, eth3d (Kimi: gate them behind
   test_preprocess_roundtrip.py, lowest priority).
```

#### Instruments built this session, and what each is for

```
scripts/test_preprocess_roundtrip.py   training vs SLAM input shape, per family.
                                       Catches the 17.55 class. All four pass.
scripts/eval_head_row.py               one head vs base on every family, per
                                       image, bootstrap CIs. Built because a
                                       4/4 pattern collapsed to 2/4 under CIs.
scripts/rescore_protocol2.py           re-score saved maps under p2.
eval_map_quality.py --no-exposure      score under the OLD protocol, so the
                                       same map can be scored both ways and the
                                       protocol effect isolated (17.52).
refine_local.py --uniform-fade         the thinning prior, selector-free
             --scale-cap               the tail cap
             --dither                  in-plane jitter (no verdict, 17.63)
main.py --refiner-uniform-fade         both levers wired into the online path
        --refiner-scale-cap
scripts/diag_nview_scale.py --head     load a head onto the scale instrument
                            --density  report confidence/gate fractions
/tmp/gp.py                             per-head Gaussian parameter distributions
                                       (scale median/p90, opacity median,
                                       frac>0.9). Recreate if lost -- it is what
                                       found the opacity collapse.
```

#### The load-time diagnostics, current form

```
accumulated alpha (map)      how much haze there is to remove
frac(opacity > 0.9) (head)   how much of it the head already removed
p90 scale ratio vs base      whether the tail needs capping (independent axis)

measured, on TUM desk inputs:
head        p90 ratio   frac>0.9      map            acc alpha   thin benefit
base           --        100.0%       desk-base         6.17       -13.6%
tum           1.04        12.5%       room-head         6.66        -8.2%
7-scenes      2.17        17.4%       desk-head         4.45        -2.1%
euroc         2.52         0.0%       360-head          3.16        -0.4%
eth3d         3.26         4.9%
replica       2.33        99.8%   <- the only saturated head, the only
                                     synthetic dataset
```

#### Standing rules earned this session

```
1  Any evaluation sharing preprocessing with training is structurally blind to
   input-pipeline errors. Every release gets one end-to-end check. (17.55)
2  Before reusing an artifact across sessions, grep its generating log for the
   configuration. The directory name is not evidence. (17.61, third instance)
3  An instrument is only trustworthy in the regime where it was calibrated.
   hp_alpha was calibrated against a visible lattice on a base map; it is not a
   lattice detector on a thinned one. (17.63)
4  When looking at an image to judge a metric, crop where the METRIC's energy
   is, not where you guess. The first crop found nothing because the cap had
   not released there. (17.63)
5  Cross-harness absolutes do not mix (offline 600-step vs online 2300-step).
   Made this error twice after writing the rule. (17.52, 17.62)
```

---

## 17.65 The causal test reads out NEGATIVE, and the tail diagnostic turns out not to be a property of the head

Three things happened in one sitting, and the second and third were both found
while setting up the first. They are recorded in the order they were found
rather than the order of importance, because the *way* the last two surfaced is
the part that generalizes.

### 17.65.1 `content` degradation does not un-saturate opacity

Kimi's round-24 causal channel (§17.60): on clean rendered images the backbone's
geometry is confident, multi-view conflict is rare, saturation is never
punished. Prediction: degrading BOTH views (`content` = Gaussian noise
sigma=6/255 + up to 5x5 blur) manufactures matching uncertainty, hence geometric
conflict, hence un-saturation toward TUM-like levels.

Trained 40 epochs, `--family replica-noisy`, same recipe/seed/resolution as the
plain Replica head. It trains normally and beats base by 30.7% on its own val
split, so this is not a failed run being read as a null.

Probed on 5 pairs of Replica office0, the identical protocol behind every row of
the §17.64 diagnostics table:

```
                 probed on CLEAN input              probed on CONTENT-DEGRADED
head        scale p90   opacity med   frac>0.9   scale p90  opacity med  frac>0.9
base           9.778       1.0000      100.0%      9.362      1.0000      100.0%
replica       22.768       1.0000       99.8%     19.856      1.0000       99.7%
noisy         26.182       1.0000      100.0%     20.731      1.0000      100.0%
```

**100.0%** — marginally MORE saturated than the clean head, not less, and not
in the direction of TUM's 12.5%.

Probed in BOTH regimes deliberately. Probing a degradation-trained head only on
clean input would repeat the §17.63 error in a new costume: asking an artifact
for a reading outside the regime it was produced in. The training distribution
answers the same way, so the negative is not a domain-shift artifact.

The perturbation demonstrably reaches the network — base's own numbers move
under it (p90 9.778 -> 9.362) — so this is not a silently-inert degradation
being mistaken for a null result.

**What the head did instead.** It answered on the *scale* axis: p90
22.768 -> 26.182 (+15%) with the median moving DOWN, 1.902 -> 1.503. Added
matching uncertainty buys a longer tail of large Gaussians plus a smaller
median. Reading, offered as a hypothesis rather than a finding: **scale is the
head's uncertainty channel and opacity is not**, which would explain why no
amount of manufactured uncertainty moves opacity. It has a sharp consequence —
the scale cap should hurt MORE on the noisy head than the plain one, because it
removes the channel actually in use. Not yet tested.

The `photometry` control (brightness jitter on ONE view, where Kimi predicts no
un-saturation because opacity is view-independent) is training now. Its value
went up sharply with this result: if both arms are negative the round-24 channel
is dead entirely, and the candidate becomes something no image-space degradation
can reproduce — depth-supervision structure (`apply_mask`, in-frustum masking,
sensor dropout) rather than image content.

### 17.65.2 The round-trip gate had been silently skipping ETH3D, and ETH3D is not shape-homogeneous

`test_preprocess_roundtrip.py` printed "all four pass" for weeks. It was
checking `datasets/eth3d/sfm_house_loop`; the local release puts sequences under
`datasets/eth3d/train/`. A missing path printed `SKIP` and did not count as a
failure, so **the one family whose head the gate most needed to check was the
one it never checked.**

Fixing the path produced a MISMATCH, which was itself wrong — and the reason is
worth more than the bug:

```
eth3d      SLAM feeds (304, 512) for 56/61 sequences   training uses (304, 512)   ok
             !! 5 sequence(s) feed (320, 512): sfm_bench, sfm_garden,
                sfm_house_loop, sfm_lab_room_1, sfm_lab_room_2
```

60 of 61 ETH3D sequences are 739x458 and feed 512x304. The five `sfm_*`
sequences are 743x465 and feed 512x320 — a coherent subgroup (a different
capture rig), not a stray file. `sfm_house_loop`, the sequence the gate happened
to name, is one of the five.

So: the eth3d head is **correctly** registered at (512,304) for its family, AND
it hits the full §17.55 failure on those five sequences. A single-sequence probe
cannot represent a heterogeneous release, and reports whichever answer its
chosen sequence happens to give — here, a MISMATCH for a correct registry.

The gate now scans every sequence in every family, judges the registry against
the modal shape, and warns separately about non-homogeneity. Only ETH3D is
affected; tum 9/9, 7-scenes 7/7, euroc 11/11, replica 5/5 are uniform.

**Do not deploy or benchmark the eth3d head on any `sfm_*` sequence** without
retraining at (512,320).

### 17.65.3 The tail ratio is a property of (head, sequence), not of the head

Setting up the queued cap confirmation required each family's own base p90,
since the cap threshold is that p90 — reusing Replica's 0.009778 for 7-scenes
would cap at a threshold calibrated on a different scene scale, the §17.63 error
again. Measured each head against base on **its own family's data**:

```
family     sequence                     base p90    head p90   ratio   head p99/base p99
7-scenes   7-scenes/chess               10.907 mm   25.230 mm   2.31        3.80
euroc      euroc/MH_01_easy             16.263 mm   14.886 mm   0.92        1.42
eth3d      eth3d/train/table_3           8.836 mm    8.072 mm   0.91        1.75
```

§17.64's table records 2.17 / 2.52 / 3.26 for these same three heads. Both are
correct measurements of different quantities, and the difference is the header
line of that table: **"measured, on TUM desk inputs"**. Every head there was
shown TUM desk images. That answers "how does this head differ from base as a
function, holding input fixed" — a legitimate question, and not the one the cap
needs. The cap is applied in deployment, to a head running on its own family.

They disagree about arming on two of three families. And note the p99 column:
every head has a heavier extreme tail than base even where its p90 is *lighter*,
because head training both shrinks the median and extends the extreme. Any
single-percentile ratio therefore depends on where you cut — which is a defect
in the diagnostic independent of which input it is measured on.

#### Pre-registration, recorded before the runs are launched

The two diagnostics make opposite predictions, so the cap runs now discriminate
between them instead of merely confirming a rule.

```
common-input diagnostic (17.64, TUM desk):  cap helps all three, MOST on eth3d
deployment-condition diagnostic (above):    cap helps 7-scenes only; neutral or
                                            harmful on euroc and eth3d
```

Arms: `none` vs `cap` at each family's own base p90, offline 600-step
`refine_local.py` on a blob baked with that family's head, one sequence per
family (7-scenes/chess, euroc/MH_01_easy, eth3d/train/table_3 — the last chosen
from the 56-sequence modal-shape group, NOT an `sfm_*`).

Outcomes and what each licenses:

- eth3d improves -> common-input diagnostic wins; the load-time probe should
  keep using a fixed reference input, and 17.64's table stands as the gate.
- eth3d flat or worse AND 7-scenes improves -> deployment-condition diagnostic
  wins; 17.64's arming column must be recomputed per family and the "arm the cap
  when ratio > 2" rule must specify *on what input*.
- both flat -> the cap's single confirmed win (Replica, -16.7%) does not
  generalize and the cap is Replica-specific, not "conditional-by-diagnostic".

Either of the last two invalidates the arming column of §17.64 as written.

---

## 17.66 Opacity is predicted by the backbone's confidence — weakly, but robustly

§17.65.1 ended with a hypothesis I offered and Kimi half-accepted: *scale is the
head's uncertainty channel and opacity is not*. Kimi's round-28 reply rejected
the test I proposed for it ("the cap hurts the noisy head more") as confounded —
the cap's damage runs through coverage, and the noisy head simply has a bigger
tail to bite, so differential harm is predicted by both accounts — and named a
free observational test instead: correlate per-Gaussian outputs against
per-pixel unreliability. He proposed it to settle a different dispute (my
depth-mask candidate vs his non-rigid-photometry candidate). It settles both,
and it comes out against the hypothesis I was defending.

`scripts/diag_colocation.py`. One forward pass, no training. Per-pixel
predictors, all computed SLAM-side so nothing shares the training preprocessing:

```
invalid    sensor depth <= 0, dilated 9x9      (my candidate)
edge       depth discontinuity > 5 cm, dilated (silhouette anti-aliasing)
disagree   local |I1 - I2| between the pair    (Kimi's candidate)
conf       the backbone's own confidence head  (neither of us named it)
depth      the network's own pointmap z        (the confound control)
```

### 17.66.1 Two guards, both of which fired

**The depth control changed the answer.** TUM's structured-light depth drops out
on far, glossy and dark surfaces, and Gaussian scale grows with range by
construction — so `invalid` is a candidate proxy for distance and a raw
correlation could be entirely a range effect. Stratifying by the network's own
predicted depth (`partial_spearman`, pooled within 8 depth bands, because the
opacity/depth profile is U-shaped and a linear partial would miss it):

```
TUM head, 1,179,648 gaussians over 6 pairs, median opacity 0.6004
                    invalid      edge   disagree      conf     depth
OPACITY   raw       -0.2390    0.0127   -0.1032    0.2116   -0.1085
          |depth    -0.1983   -0.0267   -0.0962    0.2126   -0.0010
MAXSCALE  raw        0.2218   -0.0356    0.1005   -0.3879    0.3967
          |depth     0.1147    0.0902    0.0174   -0.2641    0.0800
```

Scale's raw correlation with depth (+0.397) is the largest number in the table,
and controlling for it **halves** scale's association with `invalid`
(+0.222 -> +0.115). Opacity's associations barely move (`conf` +0.212 -> +0.213,
entirely undiminished; `invalid` -0.239 -> -0.198).

**The degeneracy guard stopped three fabricated findings.** Running the same
probe on the saturated heads produced large, stable-looking opacity
correlations: replica-head -0.44 vs `invalid`, noisy-head -0.59, base -0.056.
All three are artifacts. Those heads are 94-100% tied at opacity exactly 1.0,
p10-p90 spread `1.19e-07` and `0`; `argsort` breaks ties by position, so rank
statistics on them are arbitrary rather than weak. The instrument now measures
the tie fraction and refuses to print instead of inviting the read.

This is the §17.63 lesson in a third costume — a statistic returning a confident
number outside the regime where it means anything — and this time the guard was
built before the number was interpreted rather than after.

### 17.66.2 What the surviving numbers say

**Opacity's strongest measured associate is the backbone's own confidence
output** (+0.213 partial), completely undiminished by the depth control. Low-
opacity Gaussians sit where the backbone is unsure and where the sensor gave no
depth.

State the effect size with the claim, always (Kimi's round-29 tempering, and he
is right): rho = 0.213 explains about **4.5% of the variance**. With 1.18M
Gaussians significance is not the question; size is. The honest sentence is
"opacity is predicted by backbone confidence, weakly but robustly, with sensor
invalidity second" -- most of the opacity variance remains unowned. Writing
"confidence channel" without the number next to it would oversell it. The lever
in 17.66.4 does not need the coupling to be strong; it needs the ALLOCATION to
matter, which is a separate question that only the run answers.

One unresolved point that belongs in any mechanism paragraph: `conf`'s own
causal identity is unknown. It is a two-view matching confidence, and its
correlation with opacity could be driven by both being responses to surface type
(textureless / far / dark) rather than by confidence as such.

**Scale is not, or not mainly.** Its apparent link to unreliability is
substantially a distance artifact, and on the noisy head — the head whose scale
response prompted the hypothesis in the first place — the scale/confidence
partial is **+0.003, gone**, while every other head has -0.19 to -0.29:

```
maxscale vs conf, partial|depth      tum-head  -0.264
                                     base      -0.288
                                     replica   -0.189
                                     noisy     +0.003   <- the hypothesis's own head
```

If the noisy head were hedging uncertainty with scale, its scale should track
confidence MORE than other heads, not less. **§17.65.1's reading is withdrawn.**
Its scale/`disagree` partial is the highest of any head (+0.316), which on
Replica mostly indexes inter-frame parallax rather than photometric conflict, so
the tail inflation looks like coverage-seeking — which is exactly the
alternative Kimi named ("if they're scattered on clean flat walls, it's
coverage-seeking, not hedging"). Recorded as the leading reading, not a finding;
`disagree` is a weak proxy on a moving camera and is not load-bearing anywhere.

### 17.66.3 The dispute, resolved against both of us

- **My depth-mask candidate: supported.** `invalid` is the second-strongest
  surviving predictor. Kimi's quantitative objection — "masks are local
  (10-20% of the map), the median shift is global" — is empirically wrong for
  TUM: the dilated invalid mask covers **~40%** of pixels (mean of the column),
  and it reaches 0.75 in the lowest opacity decile against 0.20 in the highest.
  Not local.
- **Kimi's non-rigid-photometry candidate: weakly supported at best** (-0.096).
- **The occlusion-edge story: dead** (-0.027, and null on every head). Whatever
  partial opacity is for, it is not silhouette anti-aliasing.
- **Both of us were beaten by a predictor neither of us named**, which is also
  the only one available for free at inference on any dataset.

### 17.66.4 The lever this hands us, pre-registered

The chain is now: opacity should track backbone confidence; the base checkpoint
cannot express it (97.8% saturated); fine-tuning on real data creates it;
fine-tuning on Replica does not. `--uniform-fade` currently thins **uniformly**
because §17.34's elongation selector lost to it — but a uniform prior is the
right answer only if there is no signal about *which* Gaussians to thin. There
is one, it is free, and it has never been tried.

```
proposed   opacity *= 1 - D*(1 - conf_norm)     vs  current  opacity *= (1 - D)
```

PRE-REGISTERED, before implementation: a confidence-weighted fade beats uniform
fade at equal mean opacity on the base/Replica maps (where the head never
learned the channel and the prior is supplying it), and ties or slightly loses
on the TUM head's map (where the head already encodes it, so the prior is
redundant and can only add noise). If it loses on the base map too, then opacity
co-location is a description of what trained heads do and not a usable prior,
and uniform fade stays.

The honest caveat: this is a within-map correlation on outputs, and "opacity
tracks conf in a trained head" does not entail "imposing conf on an untrained
head helps". That gap is exactly what the pre-registered run measures.

---

## 17.67 The cap experiment: both diagnostics failed, and they failed the same way

§17.65.3 set up the cap runs so that the two tail diagnostics made opposite
predictions, and the result would pick one. It picked neither.

```
family     deploy ratio  common-input ratio    cap: lpips        psnr
Replica        2.33            2.33          0.0959 -> 0.0799   (-16.7%)
7-scenes       2.31            2.17          0.5243 -> 0.5061    (-3.5%)   13.85 -> 13.33
eth3d          0.91            3.26          0.4995 -> 0.4811    (-3.7%)   12.57 -> 12.46
```

Scoring the pre-registration exactly as written:

- **Common-input diagnostic predicted the cap helps all three, MOST on eth3d**
  (its 3.26 being the highest ratio on record). eth3d gained -3.7% against
  7-scenes' -3.5% — a tie, not a maximum. The ordering carries no information.
- **Deployment-condition diagnostic predicted 7-scenes only, neutral or harmful
  on eth3d** (ratio 0.91, disarmed). eth3d gained slightly MORE than 7-scenes.
  The arming decision was wrong.

So the third pre-registered outcome is the one that landed, in a sharper form
than I anticipated: the cap's Replica win (-16.7%) does not generalize —
elsewhere it is a small, roughly constant -3.5% that is **uncorrelated with any
tail statistic in either direction**.

Kimi reached the same verdict independently in round 28, from a different
observation: on the penalty head the ratio fell to 1.20 (diagnostic says
*disarm*) while the cap still earned -16.7% on that map. Two independent routes,
one conclusion:

> **Stop inferring lever value from parameter statistics. Measure it per blob.**
> A cap on/off offline arm costs an hour. The parameter tables stay as
> description; they are not decision rules.

**§17.64's arming column is retracted as a decision rule.** The p90 ratio
remains a description of what a head does to the scale distribution, which is a
real and reportable property. It does not predict whether the cap pays.

### Two caveats that limit how far this travels

**The absolute regime is not comparable to Replica's.** These baked maps sit at
psnr 12-14 against Replica's ~27. Keyframe counts and Gaussian counts are
ordinary (7-scenes 12 kf / 2.36M, eth3d 14 kf / 2.18M, against Replica 21 kf /
3.10M), so the maps are not degenerate in size — these are simply much harder
sequences. The cap delta is a *paired* comparison on one map with one held-out
set, so it is internally valid; but a lever measured at 13 dB is not thereby
measured at 27 dB, and this is the same cross-regime caution as standing rule 5.

**The cap trades psnr for lpips, consistently.** Both families: lpips improves,
psnr falls (-0.51 dB on 7-scenes, -0.11 dB on eth3d). Replica's arm showed the
same sign. Whatever the cap is doing, it is not a free win, and a paper reporting
only lpips for it would be misleading.

euroc (deploy ratio 0.92, common-input 2.52 — the third discriminating case) is
running.

### 17.66.5 Round-29 exchange: is `invalid` just `conf` measured twice?

Kimi's framing of the question was the useful part: *"what you can't do is write
'confidence channel' and 'mask effect' as two findings when they may be one."*
The backbone is unconfident where the sensor also failed, for shared physical
reasons — dark, glossy, far — so the two predictors could be one signal.

Same instrument, `--control conf` instead of `--control depth`:

```
TUM head                invalid      edge   disagree      conf     depth
OPACITY   raw           -0.2390    0.0127   -0.1032    0.2116   -0.1085
          |depth        -0.1983   -0.0267   -0.0962    0.2126   -0.0010
          |conf         -0.1655    0.0452   -0.0967   (~0 by     -0.0086
                                                       constr.)
```

`invalid` holds at **-0.166** after conditioning on `conf` — against Kimi's
pre-stated decision rule ("collapses toward zero -> one channel; holds at ~-0.15
-> independent information"). So sensor invalidity carries information the
matching confidence does not see, and the two are overlapping but distinct.

Consequences, both of which are now on the record rather than open:

- The account has two members, not one. The write-up may report both.
- The next lever is a **conf + invalid joint fade** rather than conf alone.
  Noting the deployment limit up front: `invalid` needs sensor depth at
  inference, which the SLAM path does not have (it is monocular), so the joint
  version is an offline/analysis lever unless a proxy is found. `conf` has no
  such problem — it is free on every dataset. That asymmetry is a reason to keep
  conf as the shipping lever regardless of which correlates better.

### 17.66.6 Guards added to the conf-fade before its results are read

**The deletion tail.** Kimi caught a failure mode I had not named: the
multiplier at `conf_norm=0` is `1-2D`, which hits **zero at D=0.5**. A
multiplier of zero is deletion, not thinning — the §5 removal failure re-entering
silently through the extreme tail. Now floored at 0.1. At the D=0.45 the running
arms use, the untruncated minimum is exactly 0.1, so the floor is a no-op for
those measurements and they remain valid as taken.

**The failure signature to watch first** is not the one I proposed. I expected
"psnr loss concentrated on low-texture frames" (conf is a two-view matching
confidence, so it is low on textureless-but-easy walls, and thinning walls would
be exactly wrong). Kimi: the first thing to check is **black fraction rising
>0.5pp**, because that was §5's actual measured signature for over-removal.

**Kimi's prediction, sharper than mine.** I registered "ties or slightly loses"
on the TUM-head map. He registers **loses outright**: the head already thinned
its low-conf Gaussians, so conf-fade double-thins them. He is right that tie vs
loss is itself informative — a loss confirms the head's allocation is already
near-optimal and the prior is worse than redundant.

**A second readout added to the o-0.6 head**, which is still training. Beyond the
acc-alpha <= 4.4 prediction, measure that head's own opacity-conf partial. The
penalty supplies *dose*; whether it also produced *allocation* is unknown, and
the two come apart cleanly:

```
coupling ~= 0 and thin still buys >5% at acc ~= 4.4
   -> total opacity mass is not the whole variable; allocation is. Then
      conf-fade vs uniform ON THAT HEAD is the sharp follow-up, and conf-fade
      should win.
coupling ~= 0.2 on its own
   -> the penalty did not impose a level, it gave the head PERMISSION to use
      the channel and the head used it correctly. The best available outcome.
```

### 17.66.7 Result: allocation beats dose, on both channel-absent maps

The pre-registration in §17.66.4 is confirmed on both maps where the head never
learned the opacity/confidence coupling. Equal mean dose (D=0.45), differing
only in how the thinning is allocated:

```
tum-BASE map (base checkpoint, 97.8% saturated)     psnr      lpips
  none                                             14.0747   0.4759
  uniform-fade 0.45                                14.1319   0.4110
  conf-fade    0.45                                14.2070   0.3993   -2.8% vs uniform

replica-head map (99.9% saturated)                  psnr      lpips
  none                                             25.3500   0.1649
  uniform-fade 0.45                                26.3941   0.1110
  conf-fade    0.45                                26.4119   0.1079   -2.8% vs uniform
```

**-2.8% on both**, on maps that differ by 12 dB in absolute quality and come
from different datasets and different checkpoints. psnr moves the same direction
on both (+0.075, +0.018 dB), so this is not a psnr/lpips trade like the cap's.

Read the AGREEMENT, not the decimal. Two independent measurements landing on the
same two significant figures is a coincidence at this sample size, and treating
the exact match as evidence would be the same error as reading -0.44 off a tied
array (Kimi, round 30). What is load-bearing is that both are clearly negative
and of similar size across very different maps.

Two things make this more than a small win:

**It is present at iteration 0.** tum-base conf 0.5495 vs uniform 0.5726 before
a single optimizer step. The prior is improving the *injected map*, not steering
the optimizer, which is what a genuine allocation effect should look like.

**It is the first lever in this file whose value was predicted from a
measurement rather than found by sweeping.** The chain ran: measure that opacity
co-locates with backbone confidence in trained heads (§17.66.2) -> observe that
base and Replica heads cannot express it -> supply it externally -> it pays,
exactly on the maps predicted and at the dose predicted. Every previous lever
here (thin, cap, streak, dither) was found by trying it.

Against that, the honest limits: one seed, one D, 600 offline steps, and the
coupling it is built on explains ~4.5% of opacity variance. The lever pays about
a fifth of what uniform thinning itself pays (uniform buys -13.6% over none on
tum-base; conf buys a further -2.8%). It is a refinement of the thinning prior,
not a new mechanism.

Still open at the time of writing: the tum-head map (channel PRESENT), where I
registered "ties or slightly loses" and Kimi registered "loses outright". That
arm is the one that tests whether the account is right about *why* it works.

### 17.66.8 The channel-present arm: an exact tie, which is the predicted result

```
tum-HEAD map (median opacity 0.60 -- the head ALREADY encodes the coupling)
                          psnr      lpips     black
  none                  14.0938    0.3952     1.80%
  uniform-fade 0.45     14.0112    0.3871     1.90%
  conf-fade    0.45     14.0364    0.3871     1.92%
```

**lpips identical to four decimals**, psnr +0.025 dB for conf. Set against the
two channel-absent maps, the interaction is exactly the shape the account
predicts:

```
                          conf-fade vs uniform-fade
channel ABSENT   tum-BASE          -2.8% lpips
channel ABSENT   replica-head      -2.8% lpips
channel PRESENT  tum-head           0.0% lpips
```

Scoring the two pre-registrations honestly: mine ("ties or slightly loses") is
confirmed; **Kimi's sharper "loses outright" is falsified.** His reasoning was
that the head has already thinned its low-conf Gaussians so conf-fade
double-thins them. The tie says the prior is exactly redundant rather than
harmful — which is the better outcome for shipping it as a default, since it
means conf-fade is safe to leave on for heads that do not need it.

**The deletion-tail guard passes.** Black fraction none 1.80% -> conf 1.92%, a
rise of 0.12pp against Kimi's 0.5pp alarm threshold, and only 0.02pp above
uniform at the same dose. The floor added in §17.66.6 is doing its job, and this
is the first run in the file where that number was actually measured rather than
recommended.

Provenance note: the `none` arm's log was corrupted to NUL bytes on first
capture and was re-run rather than substituted from another harness. The rerun
lands at iter-0 `12.3544 / 0.5071` -- the exact four-digit signature §17.61 used
to CATCH a mislabelled artifact. Here it is the opposite: it confirms this blob
really is the TUM-head bake. The same coincidence that exposed an error once is
a provenance check when you know which map you meant to load.

---

## 17.68 The cap, fourth cell: the diagnostic is dead in both directions

```
euroc MH_01_easy          psnr      lpips     black
  none                  13.7459    0.4373     0.36%
  cap 0.016263          13.6282    0.4052     0.68%      -7.3% lpips
```

Complete scorecard, with both diagnostics' predictions next to the outcome:

```
family     deploy ratio  common ratio   cap lpips    cap psnr
Replica        2.33          2.33         -16.7%        --
euroc          0.92          2.52          -7.3%      -0.12 dB
eth3d          0.91          3.26          -3.7%      -0.11 dB
7-scenes       2.31          2.17          -3.5%      -0.51 dB
```

- **Deployment-condition diagnostic** (arm when ratio > 2) predicted 7-scenes
  alone. 7-scenes gained the LEAST. euroc at 0.92 -- solidly "disarm" -- gained
  twice as much as either armed family.
- **Common-input diagnostic** predicted the gain would order as
  eth3d (3.26) > euroc (2.52) > 7-scenes (2.17). Measured order is
  euroc > eth3d > 7-scenes. Wrong at the top.

Neither ordering survives. §17.67's verdict is now four cells deep instead of
three, and the ratio is confirmed to carry no predictive information about
whether the cap pays -- in either the fixed-input or the deployment-condition
form. §17.64's arming column stays retracted as a decision rule.

**The cap trades psnr for lpips in every cell measured** (-0.51, -0.11, -0.12 dB).
It is not a free win anywhere, and reporting only its lpips would misrepresent it.

**New, from the black-fraction instrument added this session:** the cap raises
black fraction on euroc from 0.36% to 0.68%, the largest rise seen for any lever
so far. Still under the 0.5pp alarm, but it is a coverage cost and it is the
mechanism §5 predicted for anything that shrinks footprints. Worth watching if
the cap is ever raised above the base p90.

#### 17.66.8a The tie is real, and it is not a degenerate multiplier

An lpips tie to four decimals between two different per-Gaussian multipliers is
suspicious enough to check before it is interpreted. The worrying reading was
that the conf plumbing had silently collapsed to a constant, which would produce
"same dose, same result" for an uninteresting reason. Injected-opacity
distributions, measured directly on the first four keyframes of each blob:

```
                   conf multiplier                                resulting opacity
             min   p25   med   p75   max   MEAN        uniform (sd)     conf (sd)
tum-head    0.100 0.325 0.550 0.775 1.000  0.5500     0.3540 (0.0912)  0.3203 (0.2097)
tum-base    0.100 0.325 0.550 0.775 1.000  0.5500     0.5500 (0.0023)  0.5499 (0.2598)
```

The multiplier spans the full 0.1-1.0 range, and its **mean is exactly 0.5500**,
matching uniform's constant 0.5500 — so the dose-matching design works to four
decimals and the two arms differ only in allocation, as intended.

The resulting distributions are not close: conf produces **2.3x the opacity
spread** on tum-head (sd 0.210 vs 0.091) and **113x** on tum-base (0.260 vs
0.002, where uniform is near-constant because base opacity is ~1.0 everywhere).

So the tum-head tie is a real result, and a stronger one than "tie" suggests:
**two substantially different injected opacity fields converge to identical
held-out lpips at 600 steps.** On a map whose head already encodes the
coupling, the optimizer erases the allocation difference. On maps whose head
does not, the same difference is worth -2.8% and survives to convergence. That
contrast is the account's central claim, and it is now supported by the injected
distributions rather than only by the outcome metric.

#### 17.66.9a `--refiner-gpu` indexes VISIBLE devices — third instance of the silent-refiner-death class

The conf-fade deployment A/B was launched with `CUDA_VISIBLE_DEVICES=1` and
`--refiner-gpu 1`. Under that environment torch sees exactly one device,
`cuda:0`, so the refiner raised

```
torch.AcceleratorError: CUDA error: invalid device ordinal
  refiner.py:1047 run_refiner -> refiner.py:887 _gaussian_window
```

**inside the subprocess.** SLAM ran to completion, wrote the trajectory, the
keyframes and `office0_gaussians.ply`, and exited cleanly. The only missing
artifact was `office0_refined.ply` — the thing being measured. The arm was
scored by a `grep` for `psnr=` that silently fell through to a log tail, so the
run announced nothing wrong.

This is the third member of one family in this file:

```
17.53   --refiner-streak-opacity without --refiner-aa-sigma -> ValueError in
        the subprocess, main run completes looking normal
17.55   training resolution mismatch -> val split improves, deployment loses
here    --refiner-gpu past the visible device count -> refiner dies, main run
        completes looking normal
```

The shared shape: **a fatal error in a child process that the parent's exit code
does not carry, on a path whose output nothing checks for existence.** Both
previous instances were fixed by moving validation into `main.py`'s argument
parsing; this one now is too, checking `--refiner-gpu` against
`torch.cuda.device_count()` and saying explicitly that the flag indexes VISIBLE
devices.

The remaining hole is the scoring script, not the launcher: a harness that
`grep`s for a metric and falls through when the file is absent will report a
missing arm as a formatting oddity rather than a failure. Any A/B script here
should assert the artifact exists before scoring it.

### 17.66.10 Dose sweep: the advantage grows with dose, which is what an allocation effect should do

tum-BASE map, offline, 600 steps. Doses capped at 0.45 deliberately: the 0.1
floor starts clipping above D=0.5 (the untruncated minimum is 1-2D), and once it
clips the realized mean multiplier is no longer exactly (1-D), so the two arms
would differ in DOSE as well as allocation and stop being comparable at all.

```
D       uniform lpips   conf lpips    delta     uniform psnr   conf psnr
0.25       0.4260         0.4178      -1.9%       14.1649      14.2040
0.35       0.4185         0.4088      -2.3%       14.1507      14.2111
0.45       0.4110         0.3993      -2.8%       14.1320      14.2071
```

conf wins at every dose, on both metrics, and **the margin grows monotonically
with D** — which is the signature an allocation effect should have: the more
dose you spend, the more it matters where you spend it. A fixed offset would
have suggested something incidental.

The psnr column carries a second pattern worth reporting: **conf-fade is
psnr-flat across dose** (14.2040 / 14.2111 / 14.2071) while **uniform's psnr
declines** as dose rises (14.1649 / 14.1507 / 14.1320). Uniform thinning buys
lpips by spending psnr; confidence-allocated thinning does not appear to pay
that toll over this range.

Black fraction rises slightly with conf (1.53 / 1.59 / 1.66% against uniform's
1.50 / 1.52 / 1.55%), a maximum penalty of 0.11pp — well inside the 0.5pp alarm.

Note what this sweep does NOT establish: it is sloped, not flat, so **D=0.45 is
the top of the measured range and not a located optimum.** Kimi's shipping rule
distinguishes these ("if the curve is flat the default is robust; if sloped you
ship D with a warning"), and the warning is in the flag's help text.

### 17.66.11 It survives deployment

Full online SLAM, Replica office0, Replica head, ~2300 steps with the map
growing as SLAM runs — a different harness from the 600-step frozen-blob arms,
which standing rule 5 says never to assume transfers.

```
                 psnr      lpips
none           26.3345    0.1099
uniform        26.3974    0.0983
conf           26.4156    0.0960     -2.3% vs uniform, -12.6% vs none
```

The offline -2.8% arrives as -2.3% online. Not identical, which is expected and
is exactly why the run was made; same sign, similar size, and conf is ahead on
psnr too (+0.018 dB over uniform, +0.081 over none).

**This satisfies Kimi's pre-stated shipping rule** ("online A/B shows conf >=
uniform and black fraction within budget -> conf-fade replaces uniform as the
default, unconditioned on head; the tie licenses that"), so:

`--refiner-conf-fade` now defaults to **0.45** in `main.py`. `--refiner-uniform-
fade` stays at 0.0 and setting it explicitly turns conf-fade off, so the two
never stack — a bare `--refiner-uniform-fade 0.45` selects the uniform
allocation rather than tripping the mutual-exclusion error, which is what a user
asking for one lever means. Only setting BOTH explicitly is an error.

#### The evidence behind the default, in one place

```
2 offline maps, D=0.45, channel absent      -2.8%, -2.8%
3 doses on one map (0.25/0.35/0.45)         -1.9%, -2.3%, -2.8%, monotone
1 offline map, channel present (control)     0.0%  (exact tie, verified live)
1 full online SLAM A/B                       -2.3%
black fraction, every arm                    <= +0.12pp, alarm is 0.5pp
```

Unmeasured and stated as such: seeds (Kimi ranks these least informative here
and recommends skipping unless online disagreed with offline, which it did not),
doses above 0.45, and any family other than TUM and Replica.

---

## 17.69 The opacity penalty gave the head PERMISSION, not a level

Kimi's round-30 cheapest readout, run on the EXISTING `relu(o-0.9)` head rather
than waiting for the retrain. His framing: the penalty supplies *dose*; whether
it also produced *allocation* is a separate question, and the co-location probe
answers it directly.

```
o-0.9 penalty head, Replica office0, 6 pairs
opacity median 0.8395   frac>0.9 0.0%
                    invalid      edge   disagree      conf     depth
OPACITY   raw        0.0206   -0.0714   -0.0810    0.3541    0.2262
          |depth     0.0153   -0.0897   -0.1107    0.3794    0.0730
MAXSCALE  |depth     0.0770    0.1076   -0.0196   -0.4836    0.0128
```

**Opacity-conf partial = +0.379**, against the TUM head's +0.213. The penalty
head's coupling is nearly **double** that of the head that learned the channel
unaided on real data.

This is the outcome Kimi named as the best available: the penalty did not impose
an opacity level, it **gave the head permission to use the channel**, and the
head then used it more strongly than any head trained without one. A penalty
that merely clipped the top of the distribution would show dose without
allocation — coupling near zero at median 0.84. That is not what happened.

`invalid` is ~0 here and should be ignored rather than read as a null result:
Replica's rendered depth is complete, so the predictor has no variance on this
family by construction (the adapter's own docstring says so). It is a
TUM-family predictor only.

Note the consequence for the thinning levers: this head's map has the channel
PRESENT and more strongly than TUM's, so §17.66.8 predicts conf-fade should tie
or lose on it — the map already encodes the allocation. That is a genuine
out-of-sample prediction and it is testable on the same 2x2.

### 17.69.1 Val cost of the harder push

Kimi's readout 3, all four Replica-family heads at 40 epochs, same seed:

```
head                    val psnr    val lpips
plain (long40)           21.2310      0.1084
o-0.9 penalty            21.2672      0.1110
o-0.6 penalty            21.0986      0.1153
content-degraded         21.0022      0.1283
```

**o-0.9 is free** (+0.036 dB, +0.0026 lpips — inside noise). **o-0.6 carries a
real if small tax**: -0.13 dB and +0.0069 lpips against plain, i.e. ~6% relative
lpips. So the knee position is not cost-free once it is pushed below where the
head would settle on its own, and that tax has to be reported against whatever
deployment gain the harder push buys.

### 17.69.2 A metric-comparability trap, logged before it was reported

The o-0.6 run's own verdict line reads `head-only BEATS base by 94.1%`, against
the plain head's 30.7%. **That number is not comparable and must not be quoted.**
The "base" figure in a penalty run is the base checkpoint's loss *including the
penalty term*, and the base checkpoint predicts opacity 1.0 almost everywhere,
so it eats a ~0.4 penalty per Gaussian that the plain run's base never paid.
The comparable quantities are the val psnr/lpips in §17.69.1, which show the
opposite sign.

### 17.69.3 `acc_alpha` is now a standing instrument, and it is NOT in the old units

§17.64 named accumulated alpha the operational target for the thinning levers,
but it had only ever been measured ad hoc. It now prints on every lattice
report, next to black fraction, with a stated definition:

> total alpha MASS per pixel, sum_k alpha_k, recovered by rendering at opacity
> `eps*alpha` and dividing by eps -- compositing is `1-prod(1-eps*a) ~= eps*sum(a)`
> to first order. Not the composited alpha, which saturates at 1 and cannot
> express the quantity that ordered the maps.

Calibrated against the two maps §17.64 reports:

```
map          17.64 (ad hoc)    this instrument
desk-base         6.17              34.23
desk-head         4.45               9.11
```

**The ordering reproduces; the scale does not** (3.8x separation here against
1.39x there). The two are not interchangeable, and with two calibration points
and a discrepancy that large, fitting a conversion would be unjustified.

Consequence for the o-0.6 pre-registration: **Kimi's re-anchored threshold
"acc alpha <= 4.4" is stated in the old instrument's units and cannot be
evaluated in these.** The core registered leg does not depend on it — "does
thinning's gain collapse to <=2% on this head's map" is a direct percentage on
the same map and is unaffected. That leg is what gets scored; the acc_alpha
figure is reported alongside for future comparison in its own units.

---

## 17.70 The o-0.6 head: two falsifications, one of them mine

### 17.70.1 Injection-time thinning is unconditional. Settled.

The registered test, on record since §17.64 and re-anchored by Kimi in round 28:
*if thinning's gain collapses to <=2% on a properly dosed penalty head's map,
training-time can substitute for injection-time; if not, injection thinning is
unconditional.*

```
o-0.6 head map, Replica office0        psnr      lpips     black   acc_alpha
  none                               27.2051    0.0920    0.00%      10.12
  thin (uniform 0.45)                27.4765    0.0775    0.00%       4.65   -15.8%
  conf-fade 0.45                     27.6614    0.0732    0.00%       3.86   -20.4%
  cap 0.009778                       27.5793    0.0740    0.01%       5.67   -19.6%
```

**-15.8%, not <=2%** — on a head whose median opacity is **0.5614**, i.e. below
the 0.66 the TUM head reached unaided, which is exactly the dose the o-0.6 knee
was designed to hit. The head is not under-thinned, and thinning still pays.

Second falsification of the same claim, now at two knee positions (o-0.9 gave
-21.0%, o-0.6 gives -15.8%). The margin shrinks slightly with a harder dose but
comes nowhere near collapsing. **Training-time opacity penalties do not
substitute for injection-time thinning. The claim is closed, negative.**

Note the psnr column: on this map every lever improves psnr as well
(+0.27 to +0.46 dB), unlike the cap's behaviour on the real-sensor families.

### 17.70.2 My §17.69 out-of-sample prediction is falsified, and it takes the "channel present/absent" account with it

I predicted, in writing and before the run: this head's map has the coupling
PRESENT and stronger than TUM's, so conf-fade should tie or lose here, as it did
on the tum-head map.

**conf-fade won by -5.5% over uniform — its largest margin on any map.**

And the head's coupling is indeed high, which makes the failure clean rather than
ambiguous:

```
head                opacity median   opacity-conf partial|depth   conf-fade vs uniform
o-0.6 penalty           0.5614              +0.326                     -5.5%
o-0.9 penalty           0.8395              +0.379                       --
tum head                0.6004              +0.213                      0.0%
tum base (saturated)    ~1.0             degenerate                    -2.8%
replica head (sat.)     ~1.0             degenerate                    -2.8%
```

Coupling strength does **not** order the benefit. The map with the second-highest
coupling gets the largest gain; the map with the lowest non-degenerate coupling
gets none. So "the prior is redundant where the head already encodes it" — the
account §17.66.8 was built on — is **not supported**, and the tum-head tie needs
a different explanation.

### 17.70.3 The account the numbers actually suggest

`acc_alpha`, which only became a standing measurement this session (§17.69.3),
points at a different mechanism. At the SAME mean opacity dose:

```
o-0.6 map    none 10.12  ->  uniform 4.65  ->  conf 3.86
```

**conf-fade reaches a materially lower accumulated alpha than uniform at equal
mean dose**, because it concentrates thinning where layers pile up instead of
spreading it evenly. §17.64 and Kimi's round-28 answer both name accumulated
alpha as the operational target for these levers. If that is right, conf-fade is
not "supplying a channel the head lacks" — it is **hitting the actual target more
efficiently**, which is a claim about crowding geometry and says nothing about
what the head learned.

That reframing makes a sharp, falsifiable prediction about the one result it has
to explain away: **on the tum-head map, where conf-fade and uniform tied, the two
arms should reach the SAME acc_alpha.** If they do, the tie is explained and the
crowding account replaces the channel account. If conf reaches a lower acc_alpha
there and still only ties, both accounts are wrong.

That measurement is running now, at 0 iterations so it reads the injected map
before any optimization. Recorded before the numbers land.

Caveat carried forward: the o-0.6 arms are the first to be scored with the
`black`/`acc_alpha` instruments in place, and black fraction is 0.00% in every
arm — a genuinely clean map, and a reminder that Replica's rendered completeness
makes it the easiest family for any coverage-costing lever.

### 17.70.4 The crowding account is falsified too. Three accounts down.

The prediction from §17.70.3, registered before the run: on tum-head, where the
arms tied, they should reach the same acc_alpha. Injected map, 0 iterations:

```
map        arm        acc_alpha   black     quality vs uniform
tum-head   none          9.11     6.22%
           uniform       0.85     6.37%
           conf          2.31     6.77%        0.0%   (tie)
tum-base   none         34.23     5.01%
           uniform      11.26     5.10%
           conf         10.07     5.33%       -2.8%
o-0.6      none         10.12     0.00%
           uniform       4.65     0.00%
           conf          3.86     0.00%       -5.5%
```

**Not the same — they are 2.7x apart (0.85 vs 2.31), and conf reaches the
HIGHER one while tying on quality.** So "conf-fade wins by reaching lower
accumulated alpha at equal dose" is false as a general claim: on tum-head it
reaches higher acc_alpha and does not lose.

Standing count: the channel-present/absent account (§17.66.8) is falsified by
§17.70.2, and the crowding account (§17.70.3) is falsified here. **No scalar
tested — coupling strength, opacity median, acc_alpha — orders conf-fade's
benefit across the five maps.**

What survives is the empirical result itself, which is unaffected: conf-fade is
>= uniform on every map tested, margin 0.0% to -5.5%, at every dose tested, and
in the online A/B.

### 17.70.5 The surviving hypothesis, flagged as third-in-a-row

One pattern is consistent with all five maps, and it is about safety rather than
optimality. On tum-head, **uniform drove acc_alpha to 0.85 — below 1, meaning
the average covered pixel is no longer opaque** — while conf held it at 2.31.
That is exactly what the allocation should do: conf spends its dose on the
low-confidence Gaussians and leaves high-confidence ones near full opacity, so
it cannot strip coverage from the structurally load-bearing part of the map,
whereas uniform thins everything including the parts holding the surface up.

Under that reading conf-fade is **self-limiting**, which explains the one thing
both earlier accounts got wrong: why it never loses. It is not that it always
wins by more where the head is worse; it is that it has no failure mode of the
kind uniform has.

Independent support already in hand, collected before this hypothesis existed:
the §17.66.10 dose sweep found **uniform's psnr declines monotonically with dose
(14.1649 -> 14.1507 -> 14.1320) while conf's is flat (14.2040 -> 14.2111 ->
14.2071)**. A self-limiting prior should behave exactly like that.

**This is the third account fitted to the same five maps and it deserves the
suspicion that earns.** It is recorded as a hypothesis with an out-of-sample
test attached, not as a finding. The test: push D past the measured range, where
uniform should degrade sharply and conf gracefully. Dose-matching breaks above
D=0.5 (the 0.1 floor clips), but the prediction is about robustness rather than
matched dose, so that does not block it. If uniform and conf degrade together at
high D, the self-limiting account dies too and conf-fade keeps its measured
result with no mechanism attached — which would be an honest place to leave it.

### 17.70.6 The high-dose test: the self-limiting hypothesis is half right, and the half that matters fails

Out-of-sample test from §17.70.5, on tum-head (where uniform's failure mode
appeared first), pushed past the dose-matched range:

```
D       uniform lpips   conf lpips   uniform psnr  conf psnr   u acc_a  c acc_a
0.55       0.3853         0.3859       13.9928     14.0211      5.29     5.37
0.70       0.3838         0.3845       13.9743     14.0048      3.54     4.46
0.85       0.3849         0.3845       13.9506     13.9991      1.53     3.79
```

**The mechanism half is confirmed.** conf-fade demonstrably resists driving
accumulated alpha down — at D=0.85 uniform reaches 1.53 while conf holds 3.79 —
which is exactly the self-limiting behaviour predicted: dose lands on
low-confidence Gaussians and the high-confidence ones stay near full opacity.
The psnr column agrees (uniform declines monotonically 13.9928 -> 13.9506, conf
is flat 14.0211 -> 13.9991).

**The consequence half fails.** Uniform driving acc_alpha to 1.53 does not hurt
lpips: every cell sits in 0.3838-0.3859, a spread smaller than the difference
between conf and uniform at D=0.45. There is no collapse to be protected from.
So self-limitation is real and measurable but buys no measured quality on this
map — and at D >= 0.55 on tum-head, uniform and conf are simply tied, with
uniform marginally ahead at 0.55.

**Fourth account, fourth falsification.** Recorded as such rather than rescued.

Two incidental findings worth keeping:

- **No harmful thinning regime was found even at D=0.85** on this map. §17.56
  reported deep fade as "actively worse" for the elongation selector; that does
  not reproduce for uniform fade here, where lpips improves slightly from D=0.45
  (0.3871) to D=0.70 (0.3838) and then plateaus. The default of 0.45 is
  therefore conservative rather than optimal, and is left there because higher
  doses are untested on the other four maps.
- The lattice statistic hp_alpha is flat across all six cells (0.00506-0.00529)
  while acc_alpha moves 3.5x, which is one more piece of evidence that hp_alpha
  is not tracking what these levers do.

### 17.70.7 The standing rule these four falsifications add up to

Kimi's round-31 formulation, which the record now supports at four independent
points:

> **In this system, no parameter statistic of the head or the map has ever
> successfully gated a lever. Direct per-blob measurement has never failed to.**

The exhibits:

```
tail ratio (p90 head/base)     4 cells, no predictive information      17.67, 17.68
saturation fraction (frac>0.9) drove a pre-registration to the wrong    17.62
                               variable; acc alpha was the right one
opacity-conf coupling          INVERTED: strongest coupling got the     17.70.2
                               largest conf-fade margin
accumulated alpha              ordered three maps, failed on the        17.70.4
                               fourth, and the arms that tie reach
                               values 2.7x apart
```

`opacity-conf partial` is kept as a **mechanism-section measurement and never a
gate** — it is a real fact about what heads learn, and its value in this file is
now equally as the fourth exhibit of a real measurement failing as a decision
variable.

### 17.70.8 Why training-time penalties cannot substitute, stated properly

Kimi's sentence, which turns §17.70.1 from a failed experiment into a finding:

> **Crowding is a map-level property, and the head never sees a map.**

Pair training shows the head two views. The 6-11 layer stack that thinning
addresses exists only after SLAM accumulates dozens of keyframes over the same
surface. A training-time penalty can shape per-Gaussian opacity — and it
demonstrably does, §17.69's +0.379 coupling proves the channel is reachable from
the loss — but no per-pair objective can see the stack it would need to price.
**Map problems need map-time levers.** That is why injection-time thinning is
unconditional, and it predicts the result would not change at any knee position,
which is consistent with o-0.9 (-21.0%) and o-0.6 (-15.8%) differing so little.

### 17.70.9 The harness bug is fixed, and it was worse than it looked

`scripts/exp_head_only.py` scored its verdict on the PENALIZED loss, so every
penalty run compared itself against a base checkpoint paying the maximum
possible penalty. Measured directly after the fix, on the base checkpoint at
`--opacity-target 0.6`:

```
base render loss      0.0705
base training loss    0.8705
```

**92% of the reported "base loss" was penalty, not render error.** That is the
entire content of the o-0.6 run's "BEATS base by 94.1%". The verdict now scores
the render component with penalty terms excluded and prints the training
objective separately; the run also now reports `does NOT beat base` correctly
when it does not. The record in §17.69.1-17.69.2 already used the val
psnr/lpips, which were never affected.

---

## 17.71 The acc_alpha instrument was broken, and it was broken in the direction of the comparison it was built for

`acc_alpha` was added this session (§17.69.3) and used immediately in §17.70.3
and §17.70.4 to compare conf-fade against uniform-fade. It was wrong.

### 17.71.1 How it surfaced

Kimi's matched-acc-alpha design needed the INJECTED acc_alpha of the uniform arm
at several doses. Every one came back `0.00`, on a map whose own 600-step arm at
the same dose reported 3.49. A metric that reads 0.00 before optimization and
3.49 after, on the same injected map, is not measuring what it claims.

### 17.71.2 The bug

The rasterizer culls Gaussians whose alpha falls below ~1/255 = 0.0039. The
linearization rendered at `eps*alpha` with `eps=0.01`. On a map already thinned
by a D=0.55 fade the median opacity is ~0.25, so `eps*alpha = 0.0025` — under
the cutoff for essentially every Gaussian, and the render comes back exactly
zero.

This is not a small bias, and its direction is the worst possible one: **conf-
fade and uniform-fade differ precisely in how many Gaussians they push into the
low-alpha tail.** Uniform multiplies everything by (1-D), moving the whole
distribution down together; conf-fade drives a minority hard toward the 0.1
floor and leaves the rest near full opacity. So the arm with more Gaussians
under the cutoff had more of its alpha mass invisible to the instrument — the
metric was systematically biased along the exact axis it was introduced to
measure.

Saturated maps (opacity ~1.0, `eps*alpha = 0.01`) cleared the threshold, which
is why the first calibration (§17.69.3, base and head maps, no fade) looked
sane and the ordering reproduced.

### 17.71.3 The fix

Two changes, both required:

```
eps 0.01 -> 0.05            clears the cull threshold at realistic thinned
                            opacities
A/eps -> -ln(1-A)/eps       exact inversion of 1-prod(1-eps*a) instead of a
                            first-order approximation, so the larger eps costs
                            no accuracy on crowded maps
```

Plus a guard: the report now appends `[UNRELIABLE: N% of gaussians under the
rasterizer cull threshold]` whenever that share exceeds 2%, so the failure
announces itself instead of returning a confident small number.

### 17.71.4 What this voids

**Every acc_alpha comparison between arms with different opacity distributions
is withdrawn pending re-measurement.** Specifically:

- §17.70.3's claim that conf-fade reaches lower acc_alpha than uniform at equal
  dose (3.86 vs 4.65 on the o-0.6 map). That claim was already falsified in
  §17.70.4 on other grounds, so nothing downstream depended on it being true —
  but it should not have been stated at all.
- §17.70.4's table of injected acc_alpha across maps and arms, which is the
  evidence that killed the crowding account.
- §17.70.5-17.70.6's self-limiting reading, whose entire mechanism half rested
  on "uniform reaches 1.53 while conf holds 3.79" at D=0.85.

Note that the *conclusions* of §17.70.4 and §17.70.6 do not depend on the
direction of the acc_alpha bias — both were falsifications of accounts I had
proposed, and both also rest on lpips and psnr, which are unaffected. The
crowding account is still dead (the arms that tie do not tie in quality-per-
alpha under any measurement), and the self-limiting account still fails on its
consequence half (lpips is flat at high dose regardless of what alpha did).
**But the mechanism half of §17.70.5 is now unsupported rather than confirmed**,
and §17.70.6's "mechanism half is confirmed" is retracted.

The re-measurement with the fixed instrument is running.

### 17.71.5 The lesson, which is not the one I would have guessed

This is the fourth instrument failure this session (hp_alpha out of calibration,
argsort on ties, the single-sequence roundtrip probe, and now this) and the
first where the instrument was **built during the session, by me, for the exact
comparison it then got wrong.** The other three were inherited or reused
outside their regime.

What would have caught it earlier: the instrument was validated once, on
un-faded maps (§17.69.3), and then used exclusively on faded ones. The
validation regime and the usage regime had no overlap at all — which is
§17.63's rule, arriving for the third time and still not internalized.

The practical form of the rule for next time: **validate an instrument on the
arms you intend to compare, not on a convenient reference case.** A calibration
against the base map proved nothing about faded maps, and the fade is the whole
experiment.

### 17.71.6 Re-measurement: the correction REVERSES §17.70.4, and the crowding account is back

Injected acc_alpha, fixed instrument, with the cull-share guard shown:

```
map        none    uniform 0.45   conf 0.45        uniform 0.85
tum-head   24.22      11.68       10.99  (3%)       0.86  (26%, void)
tum-base   69.30      37.23       30.63             5.67
o-0.6      26.65      12.50        9.59  (5%)       0.24  (14%, void)
```

Against the broken instrument's numbers for the same cells (9.11 / 0.85 / 2.31
on tum-head), every value is different and **the key comparison flips sign**.
The broken metric said conf reaches HIGHER acc_alpha than uniform on tum-head
(2.31 vs 0.85); corrected, conf reaches **lower** (10.99 vs 11.68), as it does
on all three maps.

That was the single observation §17.70.4 used to falsify the crowding account.
**The falsification is withdrawn.** And the corrected data does more than fail
to contradict the account — it supports it, in the one way I had not been able
to get any statistic to do all session:

```
map        acc_alpha reduction    conf-fade quality
           (conf vs uniform)      benefit vs uniform
tum-head        -5.9%                   0.0%
tum-base       -17.7%                  -2.8%
o-0.6          -23.3%                  -5.5%
```

**Monotone across all three maps, including the tie.** The map where conf barely
lowers accumulated alpha is exactly the map where it produces no quality gain.
This is the first scalar in this session that orders conf-fade's benefit — and
§17.70.7's standing rule ("no parameter statistic has ever gated a lever") needs
its first qualification: acc_alpha is a MAP-level property measured per blob,
not a parameter statistic of the head, so it is the kind of quantity that rule
was pointing toward rather than a counterexample to it.

Stated with the caution four falsifications have earned: **three points, one
of which I predicted wrongly twice already.** It is recorded as the leading
account with an out-of-sample test outstanding (Kimi's matched-acc-alpha run,
now redone against the fixed instrument), not as a finding.

Two further corrections that follow:

- §17.70.6's "the mechanism half is confirmed" (conf resists lowering alpha,
  uniform drives it to 1.53) is **retracted twice over**: the direction was
  backwards AND the D=0.85 cells carry 14-26% culled Gaussians, so they are void
  rather than merely wrong. conf-fade is not self-limiting in the sense I
  claimed; it lowers accumulated alpha MORE than uniform at equal dose.
- The §17.69.3 calibration numbers (desk-base 34.23, desk-head 9.11) were taken
  on un-faded maps where the cull threshold was not reached, so they stand — but
  they are now 69.30 and 24.22 under the corrected estimator, because the
  first-order form underestimates on crowded maps even when nothing is culled.
  The ordering still reproduces §17.64's and the scale still does not.

### 17.71.7 Kimi's matched-acc-alpha test: allocation carries a real but minority bonus

Matched on **injected** acc_alpha, not post-optimization. That is both the right
definition (the comparison is about allocation, not about where the optimizer
ends up) and the only reliable one — the 600-step readings come back flagged at
36-38% culled, while every 0-iteration reading here is clean.

```
o-0.6 map, uniform dose -> injected acc_alpha        target: conf@0.45 = 9.59
  D=0.50   10.96
  D=0.55    9.39      <- brackets the target; matched D ~ 0.544
  D=0.60    7.83

600-step lpips
  uniform D=0.50   0.0758
  uniform D=0.55   0.0744
  uniform @ matched acc (interp. D=0.544)   ~0.0746
  conf    D=0.45                             0.0732
```

**Kimi's prediction is confirmed on its qualitative claim** — uniform at matched
acc_alpha lands between thin (0.0775) and conf (0.0732) — though below his
stated numeric range of 0.076-0.080.

The decomposition this buys is the useful part:

```
conf vs uniform at matched DOSE                     0.0775 -> 0.0732   -5.5%
  ... of which, from reaching a lower alpha budget  0.0775 -> 0.0746   -3.7%
  ... residual, from allocation per se              0.0746 -> 0.0732   -1.9%
```

So roughly **two thirds of conf-fade's advantage is that it reaches a lower
accumulated alpha for the same mean opacity dose, and one third is allocation
mattering beyond the scalar.** Neither term is zero, which is why the pure
one-curve account and the pure allocation account were both wrong.

This is consistent with §17.71.6's monotone ordering and it explains the
tum-head tie without special pleading: there conf lowers acc_alpha by only 5.9%,
so the large term is nearly absent, and what remains is the ~1-2% allocation
residual — which is inside the noise of a single 600-step run.

Instrument limit worth recording: even at eps=0.05 the estimator goes unreliable
on uniform-faded maps AFTER optimization (36-38% culled), because the optimizer
does not raise every opacity back over the threshold. **acc_alpha is trustworthy
on injected maps and should not be quoted post-optimization for faded arms**
without checking the guard.

---

## 17.72 The Replica sweep: conf-fade's first loss, and the account predicts it

Kimi's round-31 risk-reducer #1. Every strong number for these levers came from
office0; this runs the stack on three more Replica scenes with the plain Replica
head.

```
scene      arm        psnr      lpips     conf vs uniform
office1    none     27.3993    0.1252
           uniform  27.7061    0.0854
           conf     27.6541    0.0878        +2.8%  <- LOSS
office2    none     18.9418    0.2310
           uniform  19.9612    0.1537
           conf     20.0610    0.1488        -3.2%
room0      none     21.4092    0.2239
           uniform  22.6212    0.1466
           conf     22.8819    0.1385        -5.5%
```

**office1 is the first map where conf-fade loses to uniform-fade**, on both
metrics (psnr -0.052 dB as well). That matters directly: the default shipped in
§17.66.11 rested on Kimi's rule "conf >= uniform", and that rule is now violated.

### 17.72.1 The failure is the account's best evidence

Injected acc_alpha on the three new scenes, all clean of the cull guard:

```
scene      none    uniform   conf    conf's extra reduction    quality benefit
office1   19.41     9.87     9.56          -3.1%                   +2.8%
office2   35.81    18.53    16.30         -12.0%                   -3.2%
room0     41.14    21.22    15.50         -27.0%                   -5.5%
```

Pooled with §17.71.6's three maps and sorted by how much extra accumulated alpha
conf-fade removes:

```
map         extra acc_alpha reduction      quality benefit
office1              -3.1%                     +2.8%   (loss)
tum-head             -5.9%                      0.0%   (tie)
office2             -12.0%                     -3.2%
tum-base            -17.7%                     -2.8%
o-0.6               -23.3%                     -5.5%
room0               -27.0%                     -5.5%
```

Monotone but for one adjacent swap (office2/tum-base), Spearman ~0.94 over six
maps. **The account predicted its own failure case**: office1 is where conf
barely improves on uniform's alpha reduction, and it is the only map where conf
loses. A correctly-predicted negative is worth considerably more than another
win, and this is the first out-of-sample test the crowding account has passed.

Per Kimi's round-32 shelving, which I am adopting: **the ordering is data and
goes in the findings; the 2/3-1/3 decomposition remains a hypothesis** until the
crowding-fade arm runs, because that arm is the account's sharpest consequence.

His accompanying observation kills any temptation to gate on the untreated map:
tum-head's untreated acc_alpha is 24.22 and o-0.6's is 26.65 — nearly identical
starting levels, benefits of 0.0% and -5.5%. A threshold on the untreated map
fails a fourth time, in the new units. What orders the benefit is the
**achievable reduction**, which you only learn by pulling the lever:

> The only map-level statistic that has ever ordered a lever's benefit in this
> project is measured by pulling the lever.

### 17.72.2 The default stays on, and its description is corrected

Full record across seven maps (six offline + the online A/B):

```
wins 5   ties 1 (tum-head)   loses 1 (office1, +2.8%)   mean ~ -2.4%
```

Keeping `--refiner-conf-fade` on at 0.45 is defensible on the mean, and the loss
case is small. What is NOT defensible is the sentence I shipped in its help text
— "measured better or equal on every map tested" — which was true when written
and is now false. The help text now states the win/tie/loss record, the mean, and
that the lever is on **because it is better in expectation, not because it is
safe everywhere**.

Recording the process point plainly: I set that default earlier in this same
session on a rule ("conf >= uniform") that three more scenes then broke. The
rule was reasonable on five maps and wrong on six. Shipping a default off a
five-map win record was the error, not the rule itself.

### 17.72.3 Kimi's generalization of the instrument rule

Third instance this session, and it now has an operational form worth quoting:

> Any probe that TRANSFORMS the signal — eps-scaling, filtering, masking — must
> re-validate at the extremes of the input range it will actually see, and print
> its reliability flag on the same line as the number.

The `[UNRELIABLE: N% under cull threshold]` guard is the template for that, not
an exception. §17.63's rule covered recording the calibration regime; this adds
the half that would actually have caught the bug.

---

## 17.72 Multi-scene Replica sweep: conf-fade is not uniformly better, and the default's justification has to change

Kimi's round-31 risk-reducer #1, run because every strong number for these
levers came from Replica office0. It found something on the first batch.

```
scene      none      uniform    conf     conf vs uniform    psnr (conf-uniform)
office1   0.1252     0.0854    0.0878        +2.8%  WORSE     -0.052 dB
office2   0.2310     0.1537    0.1488        -3.2%            +0.100 dB
room0     0.2239     0.1466    0.1385        -5.5%            +0.261 dB
```

**office1 is conf-fade's first loss.** The sentence used to justify shipping it
as a default — "measured better or equal on every map tested" — is no longer
true, and it was true only because the sample was three maps drawn from two
scenes.

Full record across everything measured so far:

```
tum-base (offline)     -2.8%      o-0.6 map (offline)   -5.5%
replica office0        -2.8%      tum-head (offline)     0.0%   tie
office0 online A/B     -2.3%      office1               +2.8%   LOSS
office2                -3.2%      room0                 -5.5%
                                       mean -2.4%,  6 wins / 1 tie / 1 loss
```

The default is still defensible on the average, but **not on the grounds
originally given**. The honest statement is "better on 6 of 8 maps, tied on 1,
worse on 1, mean -2.4%", not "never worse". Four more scenes (office3, office4,
room1, room2) are running before the default is either restated or withdrawn.

What office1 is NOT explained by: its untreated acc_alpha is 21.86, the lowest
of the three, which would fit "less crowding, less to gain" — except tum-base at
69.30 gains only -2.8% while o-0.6 at 26.65 gains -5.5%. Kimi's round-32 point
stands and is now confirmed a fifth time: **the untreated map's acc_alpha does
not order the benefit; only the achievable REDUCTION does, and that is a
property you learn by pulling the lever.** His sentence for the file:

> The only map-level statistic that has ever ordered a lever's benefit in this
> project is measured by pulling the lever.

The acc_alpha guard fired at 18-37% on the faded arms of every scene here, so
those post-optimization values cannot be used to test the reduction-ordering
account on this batch. Injected values would be needed and were not taken.

## 17.73 Crowding-fade, built

`--crowd-fade T` / `--crowd-voxel M` in `refine_local.py`. Kimi's round-32
lever and the sharpest consequence of §17.71.7's account: if two thirds of
conf-fade's value is reaching a lower alpha budget, confidence is a proxy and
the direct variable should do better.

```
mult = clamp(T / sum(opacity in voxel), 0.1, 1)
```

Applied AFTER assembly, not per keyframe, and that placement is the design.
Within one keyframe there is exactly one Gaussian per pixel, so there is no
stack to measure — crowding exists only where several keyframes' Gaussians land
on the same surface. It is a map-level property, which is the same fact that
makes §17.70.8 true.

No rank normalization, on Kimi's explicit instruction and for a good reason:
rank-per-keyframe forces every keyframe to give up the same fraction regardless
of how crowded it is. That was a defensible compromise for an uncalibrated
confidence head; for crowding, which is geometric and comparable across
keyframes by construction, it would be a bug.

### 17.73.1 The voxel scale is not a free parameter, and the default was wrong

First smoke test used the built-in default (4x median Gaussian extent = 1.41 cm)
and the lever was **inert**: alpha/voxel median 1.16, mean multiplier 0.966,
acc_alpha 25.94 against the untreated 26.65.

The reason is worth recording because it nearly shipped: **accumulated alpha is
a RAY quantity and voxel occupancy is a VOLUME quantity.** A pixel accumulating
26 layers collects them at different depths along its ray, which land in
different voxels. A voxel small enough to be "one surface" therefore sees almost
no stacking, and the lever measures nothing.

```
voxel     occupied    alpha/voxel median   mult mean   injected acc_alpha
1.41 cm   897,401           1.16             0.966          25.94
3 cm      196,323           5.25             0.516          13.30  (8% culled)
6 cm       41,802          23.14             0.164           2.99  (73% culled)
12 cm       9,059          93.51             0.107           0.75  (97% culled)
```

3 cm is the working scale: wide enough to hold the several keyframes' worth of
Gaussians that land on one surface, narrow enough not to merge separate
surfaces. The matched-acc arms are running at that scale.

Note the cull-share guard doing exactly its job at 6 and 12 cm — those acc_alpha
figures would otherwise have looked like spectacular crowding reduction.

### 17.72.1 All eight Replica scenes: the thinning is decisive, the allocation is not

```
scene       none     uniform     conf     conf vs uniform
office0    0.1649    0.1110    0.1079        -2.79%
office1    0.1252    0.0854    0.0878        +2.81%
office2    0.2310    0.1537    0.1488        -3.19%
office3    0.2352    0.1628    0.1540        -5.41%
office4    0.2521    0.1533    0.1578        +2.94%
room0      0.2239    0.1466    0.1385        -5.53%
room1      0.1831    0.1372    0.1380        +0.58%
room2      0.2402    0.1652    0.1472       -10.90%
```

Paired over the 8 scenes:

```
comparison          mean       paired t      Wilcoxon    wins
uniform vs none    -32.34%     p=2.5e-05     p=0.0078     8/8
conf    vs none    -34.18%     p=3.8e-05     p=0.0078     8/8
conf    vs uniform  -2.68%     p=0.132       p=0.148      5/8
```

**Two conclusions, and they are not the same conclusion.**

1. **Turning thinning on is overwhelming and reproducible.** -32% to -34% lpips
   on every one of eight scenes, p < 1e-4. Nothing else in this project has an
   effect this large or this consistent.
2. **Choosing the confidence allocation over the uniform one is not
   established.** Mean -2.7%, median -3.0%, but sd 4.7 across scenes and both
   paired tests land at p ~ 0.13-0.15. At n=8 this is indistinguishable from
   zero.

The default stays at `--refiner-conf-fade 0.45`, and the reasoning in the help
text is rewritten to match: it is on because it is never much worse (worst case
+2.9%) and is numerically ahead on mean and median, **not because it is
measurably better.** `--refiner-uniform-fade 0.45` is within noise of it.

This retires the framing of §17.66.7-§17.66.11. The -2.8% that looked like a
stable constant across two maps was a draw from a distribution with sd 4.7;
seeing it twice in a row was the coincidence Kimi warned about in round 30 when
he told me not to read the decimal agreement. **He was right for a reason I did
not appreciate at the time: the agreement was not evidence of precision, it was
a small sample of a wide distribution.**

## 17.74 Crowding-fade LOSES at matched budget: the map variable is not the mechanism

Kimi's round-32 pre-registration: *crowding-fade at acc alpha matched to
conf-fade's 9.59 should land at or below conf's 0.0732; if it beats conf, the
mechanism arc closes on the map variable and the head-side statistics become
description.*

```
o-0.6 map, all arms                lpips     psnr     injected acc_alpha
none                              0.0920   27.2051        26.65
uniform 0.45                      0.0775   27.4765        12.50
conf 0.45                         0.0732   27.6614         9.59
crowd T=4.0  (matched, 9.36)      0.0806   27.4268         9.36
crowd T=4.5              (10.36)  0.0815   27.4057        10.36
combo: crowd 4.5 + conf 0.45      0.0703   27.6976         6.12
```

**Crowding-fade at matched budget scores 0.0806 — worse than conf (0.0732) AND
worse than uniform (0.0775).** Pre-registration falsified, and in the direction
that hurts the account rather than helping it.

This is the cleanest evidence yet against §17.71.7's decomposition. If two
thirds of conf-fade's value were "reaching a lower accumulated alpha", then a
lever that targets accumulated alpha DIRECTLY, and provably reaches the same
value, should capture at least that two thirds. It captures none of it: it does
worse than the uniform fade that reaches a HIGHER acc_alpha (12.50).

So accumulated alpha is not the operative variable. Reaching a given alpha
budget is not what pays; **where the alpha is removed from is what pays**, and
crowded voxels are the wrong place to take it from. Note crowd-fade needs a
heavier mean multiplier (0.389) than conf (0.55) to reach the same budget, i.e.
it removes more opacity for the same alpha reduction -- an inefficient
allocation by construction.

The combo arm is the best cell in the table (0.0703) but is **not dose- or
budget-matched** (acc_alpha 6.12 against conf's 9.59), so it cannot be scored
against Kimi's decision rule. What it does establish is that the two levers are
not redundant: stacking them beats either alone, so confidence carries
information that voxel crowding does not. A budget-matched combo arm is the
outstanding measurement.

### 17.74.1 Running score of mechanism accounts

```
channel present/absent          falsified 17.70.2 (coupling ordering inverted)
crowding / acc-alpha budget     falsified 17.70.4, reinstated 17.71.6 by an
                                instrument fix, falsified again HERE by direct test
self-limiting allocation        falsified 17.70.6 (consequence half)
2/3 budget + 1/3 allocation     falsified HERE (the 2/3 term does not transfer
                                to a lever that targets the budget directly)
```

Four accounts, five falsifications, one of which was an instrument artifact in
both directions. **The empirical position is unchanged and is worth stating
plainly: thinning works enormously (-32%, p<1e-4, 8/8), and no mechanism yet
proposed survives contact with a direct test.** The lever ships on its measured
effect, and this file does not claim to explain it.

---

## 17.75 The causal test closes: the channel is PHOTOMETRIC, not geometric

Both arms of Kimi's round-24 design are now in, trained identically (40 epochs,
same seed, same family, same resolution) and probed identically (5 pairs of
Replica office0, clean input, the protocol behind every row of §17.64).

```
head                  scale p90    opacity median   mean    frac>0.9   frac>0.99
base                    9.778          1.0000      1.0000    100.0%     100.0%
replica (clean)        22.768          1.0000      0.9993     99.8%      99.4%
noisy   (content)      26.182          1.0000      1.0000    100.0%     100.0%
photo   (photometry)   17.367          0.9912      0.9293     83.1%      52.1%
```

**Only the photometric arm moves opacity.** Against the clean head's 99.8%:

```
content     +0.2 pp   (99.8 -> 100.0)   no recruitment at all
photometry -16.7 pp   (99.8 ->  83.1)   and frac>0.99 halves, 99.4 -> 52.1
```

That reverses the round-24 hypothesis, which predicted geometric conflict would
be the channel and photometric jitter would not be. The measured dissociation is
the other way round.

### 17.75.1 Scoring the pre-registration exactly as written

Kimi's stated rule was **"photometry arm ends with frac>0.9 > 70%; if it comes
out <50%, the exposure-hedge slice is real and my colour-sufficiency model is
wrong."** Measured 83.1%. **By his own threshold his prediction holds** — the
opacity channel is not strongly recruited, and his argument (a global x0.8-1.25
shift is exactly representable by scaling `f_dc`, so opacity never needs to be
called on) survives as the account of why it is only partial.

Both things are true and the file records both: his numeric prediction passed,
and the *direction* of the round-24 channel is falsified. The contrast between
arms does not depend on the threshold — it is 16.9 pp wide between two arms that
differ in nothing but the perturbation.

### 17.75.2 The two degradations have opposite signatures, on both axes

```
                       scale p90 vs clean head     opacity frac>0.9
content (geometric)         +15%   (22.8 -> 26.2)      unchanged
photometry (appearance)     -24%   (22.8 -> 17.4)      -16.7 pp
```

Two distinct response modes, cleanly separated:

- **Matching uncertainty is answered with scale**, and opacity is not touched.
- **Appearance disagreement is answered with opacity**, and the scale tail
  actually shrinks.

This is the first result in the arc that assigns each output channel a job, and
it was produced by a controlled pair rather than by correlating outputs — which
matters, given that every correlational account in §17.70-17.74 has been
falsified.

### 17.75.3 What still does not explain TUM

Neither arm reaches TUM's 12.5%. Photometry gets 83.1% from 99.8%, moving about
one sixth of the distance on the frac>0.9 scale and rather more on frac>0.99.
So per-view brightness jitter is *a* real driver and is nowhere near sufficient.

The remaining candidates are the ones no image-space perturbation of a rendered
dataset can produce, and they are now better motivated than when I proposed them
in round 28, because the surviving driver is photometric rather than geometric:
specular drift, white balance, rolling shutter, motion blur -- appearance
disagreements that are view- and scene-correlated rather than an i.i.d. global
gain. Replica has none of them by construction. Depth-supervision structure
(§17.66.3's `invalid`, partial -0.198) remains the other live member and is the
one that is not photometric at all.

### 17.75.4 The corrected verdict printer, first clean use

```
base           : 0.1009
head-only best : 0.0749   => BEATS base by 25.7%
```

This run has no opacity penalty, so the render loss and the training objective
coincide and 25.7% is a real quality number. Recorded as the contrast case for
§17.70.9's o-0.6 run, whose 94.1% was 92% penalty term. Comparable val figures:
photo psnr 21.00 / lpips 0.1283 against the clean head's 21.23 / 0.1084, so the
degradation costs val quality as expected and the head still beats base.

### 17.74.2 Budget-matched combo: crowding contributes nothing, and dilutes conf

The outstanding cell from §17.74, now matched. All on the o-0.6 map:

```
arm                                  lpips     injected acc_alpha
none                                0.0920        26.65
uniform 0.45                        0.0775        12.50
crowd  T=4.0        (matched)       0.0806         9.36
combo6 crowd 6.0 + conf 0.30        0.0747         9.63   <- budget-matched
combo8 crowd 8.0 + conf 0.30        0.0755        11.03
conf   0.45                         0.0732         9.59   <- best
```

Scored against Kimi's round-32 decision rule:

- **combo (0.0747) >> crowd alone (0.0806)** -> confidence carries information
  that voxel crowding does not. His stated consequence: ship the combination.
- But **combo is also WORSE than conf alone (0.0747 vs 0.0732)** at the same
  budget, which his rule did not anticipate. Adding the crowding term to
  confidence does not add anything; it dilutes.

So at a matched alpha budget the ordering is unambiguous:

```
conf alone  0.0732   <   combo  0.0747   <   uniform  0.0775   <   crowd  0.0806
```

**Crowding-fade is the worst allocation tested, including worse than spreading
the dose uniformly.** The lever Kimi proposed as the line's "final form" — the
one that targets the map property directly — is refuted, and refuted twice: it
loses alone, and it degrades conf-fade when mixed in. `--crowd-fade` stays in
the codebase as a measured negative, not as a shipping lever.

This also closes the §17.71.7 decomposition for good. There is no sense in which
"reaching a lower accumulated alpha" is two thirds of the benefit: three
different levers reach ~9.5 and score 0.0732, 0.0747 and 0.0806.

## 17.76 The photometric channel produces ALLOCATION, not just a level shift

Co-location probe on the photo head, the same instrument as §17.66:

```
photo head, Replica office0, opacity median 0.9906, frac>0.9 81.2%
                    invalid      edge   disagree      conf     depth
OPACITY   raw        0.0479   -0.0693   -0.2273    0.0985    0.2581
          |depth     0.1338   -0.0689   -0.2684    0.1270    0.0710
MAXSCALE  |depth     0.0019    0.1341    0.1277   -0.4174    0.0428
```

**`disagree` is the strongest opacity predictor at -0.268 partial**, against
`conf` at +0.127. Compare the TUM head, where the ranking is the other way round
(conf +0.213, disagree -0.096).

So training under per-view brightness jitter does not merely lower the opacity
level — it makes opacity **predicted by local photometric disagreement**, which
is the specific pattern the perturbation would predict if opacity were serving
as an appearance-disagreement hedge. Perturb photometry, and opacity reorganizes
around photometry.

That is a causal demonstration of the channel rather than a correlational one,
and it is the strongest single piece of mechanism evidence in this arc — notable
because every correlational account here (§17.70-17.74) has been falsified while
the two controlled-pair results (§17.75.2's dissociation and this) both stand.

`invalid` is +0.134 here and must be ignored: Replica's rendered depth is
complete, so the predictor has no variance on this family and the sign is noise.

---

## 17.77 The real-family deployment lands OUTSIDE the pre-registered band, and the line's scope has to be restated

Kimi's round-31 risk-reducer #2, the run he called "the only one that answers
does this line matter in a real deployment". Full online SLAM, TUM desk, TUM
head, ~2300 steps, conf-fade at the new default against conf-fade disabled.

His PRE-REGISTERED band: lpips -3% to -8%, psnr +-0.1 dB, black +<0.3pp, and
explicitly *"if it lands outside that, the whole line's scope statement needs
rewriting."*

```
arm        psnr       lpips
nofade   14.1646     0.3718
conf     14.2152     0.3696      -0.6% lpips, +0.05 dB
```

**-0.6%. Outside the band, on the low side.** psnr is inside it. So by his own
criterion the scope statement is what changes, not the lever.

### 17.77.1 The magnitude is largely Replica-specific; the sign is not

Set against the Replica online A/B on the same harness:

```
family/head              online conf-fade vs no fade
Replica office0, head          -12.6%
TUM desk, TUM head              -0.6%
```

and the offline TUM sequences now coming in tell the same story — thinning
against no thinning is -1.8% on tum room and -0.3% on tum 360, against -32% mean
across eight Replica scenes.

**This is the Replica-artifact risk Kimi flagged in round 31, and it is real for
the effect SIZE.** It is not a sign flip: thinning does not hurt on TUM, it
simply buys very little there. The headline "-32%, p<1e-4, 8/8" in §17.72.1 is a
statement about Replica and must never be quoted without the family attached.

### 17.77.2 The leading explanation, and the experiment that isolates it

The TUM head un-saturates itself (median opacity 0.60, frac>0.9 12.5%), so its
baked map has little haze left for an injection-time fade to remove. The Replica
head stays saturated at 99.8%. On that account the difference is a property of
the MAP, not of the dataset.

The confound is obvious: TUM and Replica differ in sensor, geometry, exposure,
trajectory and absolute map quality (14 dB vs 27 dB), any of which could produce
the same contrast.

**The isolating experiment is running now**: the same TUM desk sequence, same
harness, same everything, with the BASE head (100% saturated) instead of the TUM
head. PRE-REGISTERED: if saturation is the explanation, the base head gains
substantially more than the TUM head's -0.6%. If it also gains ~0, saturation is
not the explanation and the contrast belongs to the dataset.

Against that, one caution already on record: the o-0.6 head is un-saturated
(frac>0.9 = 0.0%, median 0.5614 — close to the TUM head's 0.60) and still gains
-20.4% offline. So per-Gaussian saturation cannot be the whole story, and if the
base-head arm comes back large, the account will be "saturation matters on TUM"
rather than "saturation explains the levers".

### 17.77.3 Saturation IS the explanation on TUM, and the pre-registration lands

Same sequence, same harness, same everything except the head:

```
TUM desk, online, ~2300 steps      psnr      lpips     conf-fade effect
BASE head   (frac>0.9 = 100%)
  no fade                        14.1743    0.4035
  conf-fade 0.45                 14.0596    0.3775      -6.4%   -0.11 dB
TUM head    (frac>0.9 =  12.5%)
  no fade                        14.1646    0.3718
  conf-fade 0.45                 14.2152    0.3696      -0.6%   +0.05 dB
```

**A 10x difference in the lever's value, from the head alone.** The
pre-registration in §17.77.2 is confirmed: on TUM, map saturation is what
decides whether injection-time thinning pays.

And note where -6.4% falls: **inside Kimi's pre-registered -3 to -8% band.** His
band was right; it was right for a *saturated* map, and the TUM head had already
done the job the lever exists to do.

### 17.77.4 The 2x2 the two arms accidentally completed

Reading the four cells as a factorial — head training vs injection thinning:

```
                    no fade     conf-fade
base head           0.4035       0.3775
TUM head            0.3718       0.3696

head training alone   -7.9%
injection lever alone -6.4%
both                  -8.4%
```

**The two routes do nearly the same job, and stacking them adds almost nothing**
(-8.4% against -7.9% for head training alone). Head training is slightly the
better of the two on this sequence, and the lever recovers most of its value on
an untrained head.

That is a genuinely useful engineering statement, and it is the first time the
two halves of this project have been measured against each other on one axis in
one harness. It also gives the lever a clear role: **it is what you use when you
cannot retrain the head for the deployment family.**

Caveat, stated because it limits the claim: one sequence, and the o-0.6 head
remains a counterexample to any pure saturation account — it is un-saturated
(frac>0.9 = 0.0%, median 0.5614, essentially the TUM head's 0.60) and still gains
-20.4% offline on Replica. So saturation decides the WITHIN-TUM contrast
decisively and still does not explain the cross-family one. The 2x2 is repeating
on tum room now.

### 17.77.5 TUM allocation sweep: conf and uniform are indistinguishable there too

```
sequence   none      uniform    conf     conf vs uniform   thinning vs none
room      0.4583     0.4502    0.4501       -0.02%             -1.8%
360       0.4406     0.4399    0.4419       +0.45%             -0.3%
teddy     0.4456     0.4371    0.4367       -0.09%             -1.9%
```

Three ties and one marginal loss, on a real-sensor family with a head that has
already un-saturated itself — so both the allocation question and the thinning
question come out near zero here, for the reason §17.77.3 identifies.

Combined with Replica's p=0.13, the position on the allocation choice is now
firm: **conf-fade vs uniform-fade is not measurably different on any family
tested.** The default stays where it is on the "never much worse, numerically
ahead" grounds already written into the flag's help, and nothing stronger should
be claimed for it.

---

## 17.78 The allocation question, closed: no measurable difference, on 12 maps across two families

```
TUM (4 sequences, TUM head)     room -0.02%   360 +0.45%   teddy -0.09%   plant +1.16%
                                mean +0.38%,  wins 2/4,  paired t p=0.279
Replica (8 scenes, Replica head)                          mean -2.68%,  wins 5/8,  p=0.132

COMBINED  n=12    mean -1.66%   median -0.06%   wins 7/12
                  paired t p=0.231    Wilcoxon p=0.380
                  95% CI on the mean   -3.96% .. +0.63%
```

**The confidence interval contains zero and the median is -0.06%.** After a dose
sweep, two families, twelve maps and an online A/B, `--conf-fade` and
`--uniform-fade` are not distinguishable.

Note the family split, which is the only structure in the data: conf is ahead on
Replica (-2.7%) and marginally behind on TUM (+0.4%). §17.77.3 explains why any
allocation effect must be small on TUM — the TUM head has already un-saturated
itself, so total thinning only buys -0.3% to -4.1% there, and a third-order
question about how to allocate a small dose cannot resolve above noise.

**The default is left at `--refiner-conf-fade 0.45`** — not because it is better,
but because it is the configuration both deployment A/Bs were actually run with,
and changing a default on no evidence is churn. The flag's help now states the
CI and says plainly that `--refiner-uniform-fade 0.45` is equally defensible and
simpler, and that effort belongs on *whether thinning is on*, not on which
allocation.

### 17.78.1 What this retires

§17.66.7 read a -2.8% agreement across two maps as a stable effect and built
three sections of mechanism on it. The effect was a draw from a distribution
whose 95% CI spans -4.0% to +0.6%. Every mechanism account in §17.70-17.74 was
an attempt to explain a difference that is not established to exist.

That is the single most expensive error of the session, and it is not an
instrument bug — the numbers were all correct. **It was reading a two-point
agreement as precision instead of asking for a confidence interval, and then
spending five experiments explaining it.** The instrument bugs cost hours; this
cost the whole mechanism arc.

The rule, for the file: **before explaining an effect, put an interval on it.**
A difference of a few percent measured on two maps needs n before it needs a
mechanism.

### 17.77.6 tum room replicates the 2x2 (partial), and two operational near-misses

```
tum room, online              psnr      lpips
base head, no fade          12.5912    0.4801
base head, conf-fade        12.6953    0.4470     lever alone  -6.9%
TUM head,  no fade          13.0125    0.4325     head alone   -9.9%
TUM head,  conf-fade          (running)
```

Against desk's -6.4% / -7.9%, the two effects replicate closely on a second
sequence, and the ordering (head training slightly ahead of the lever) holds.

**Near-miss 1: a flag validation caught a real misconfiguration.** The Replica
saturation run died immediately on `--refiner-gpu 1 is not a visible device`
-- the flag indexes VISIBLE devices, so with `CUDA_VISIBLE_DEVICES` pinned to a
single card the only valid value is 0. That check was added in an earlier
session precisely because a bad refiner device used to kill the refiner
subprocess while the main run completed looking normal. It did its job: the run
failed loudly instead of silently producing a map with no refinement. The three
earlier deployment scripts were checked against this and were correct
(`--refiner-gpu 0`), and their logs contain no errors and reach `snapshot v10`,
so §17.66.11's -12.6% and §17.77's numbers stand.

**Near-miss 2, which was luck rather than care.** Fixing that bug, I `sed`-ed
three scripts including one that was still running, and the replacement changed
the byte length of a line. **Bash reads scripts lazily by byte offset**, so
editing a running script can make it execute garbage from a shifted position.
This session already produced one artifact of exactly that kind -- the
`cf_tumhead.out` log that came back as NUL bytes and had to be re-run. The job
survived here only because bash had already buffered the loop.

Rule: **never edit a script that is currently executing.** Copy it, edit the
copy, and launch the copy. The cost of the copy is nothing; the cost of the
failure mode is a silently corrupted run that looks like a result.

---

## 17.79 In the ONLINE harness, head saturation separates the lever's value perfectly — and the offline harness was overstating it tenfold

The cross-family test from §17.77.2 came back, and with it the four online
deployment A/Bs now form a complete picture.

```
online A/B, conf-fade vs no fade      head frac>0.9    lpips effect
Replica office0, plain head               99.8%          -12.6%
TUM desk,        base head               100.0%           -6.4%
Replica office0, o-0.6 head                0.0%           -1.8%
TUM desk,        TUM head                 12.5%           -0.6%
```

**Perfect separation, 4/4, across both families.** Saturated heads gain -6% to
-13%; un-saturated heads gain -0.6% to -1.8%. The two clusters do not overlap
and the split is not confounded with family — each family contributes one point
to each cluster.

### 17.79.1 The offline harness overstates the lever by an order of magnitude

The o-0.6 head is the case that makes this unmissable, because the baselines
match almost exactly:

```
o-0.6 head, Replica office0        no-fade lpips    conf-fade    effect
offline, 600 steps, frozen blob        0.0920         0.0732     -20.4%
online,  ~2300 steps, growing map      0.0913         0.0897      -1.8%
```

Same head, same sequence, baselines within 0.8% of each other — and the lever's
measured benefit differs **eleven-fold**.

The explanation is the optimizer. Injection-time thinning is a *prior on the
initial opacities*; given enough steps the optimizer re-derives whatever
opacities the data supports and the prior washes out. 600 offline steps leave it
largely intact; ~2300 online steps mostly erase it. What survives online is only
the part the optimizer cannot fix on its own — which is exactly the haze a
saturated head bakes in, and exactly nothing when the head has already
un-saturated itself.

This is standing rule 5 (cross-harness absolutes do not mix) in a sharper form
that the rule as written does not cover: **it is not only absolute scores that
fail to transfer between harnesses, it is the measured EFFECT SIZE of an
intervention.** Every offline lever number in §17.66-17.74 is therefore an upper
bound on deployment value, not an estimate of it. The rule is amended.

### 17.79.2 This rehabilitates the one diagnostic I had written off

§17.64 originally said "arm it when the map's accumulated alpha is high and the
head has not un-saturated itself." I retired the arming column wholesale in
§17.67 after the tail RATIO failed on four cells, and §17.70.7 generalized that
to "no parameter statistic of the head has ever gated a lever."

That generalization was measured entirely in the offline harness, where — per
§17.79.1 — the lever's value is dominated by an effect the optimizer would have
removed anyway. **In the harness that ships, the saturation fraction gates the
lever correctly on every cell tested.** The tail ratio remains dead; the
saturation fraction does not.

Stated with the caution this deserves: n=4, perfect separation, and I have been
wrong about a gating statistic four times before — every one of those times
offline. An out-of-sample cell is running now (Replica office1, plain saturated
head), pre-registered to land in the saturated cluster at worse than -3%.

### 17.79.3 What this implies for the default, pending that cell

If the gate holds, `--refiner-conf-fade 0.45` unconditionally is the wrong
default: it buys -6% to -13% on a saturated head and ~1% on an un-saturated one,
and `frac(opacity > 0.9)` is measurable at load time from a single pair. The
right default is to arm the lever on that measurement. Not changed yet — one
out-of-sample cell first, and a gate that fires on the wrong side would be worse
than no gate.

### 17.79.4 The 2x2 replicates

```
                     desk                room
lever alone         -6.4%               -6.9%
head alone          -7.9%               -9.9%
both                -8.4%              -10.4%
```

Both sequences agree: head training and the injection lever do nearly the same
job, head training is slightly ahead, and stacking adds about half a point.
Consistent with §17.79.1's account — head training removes the haze at source,
so the lever has less left to remove.

### 17.79.5 The gate is falsified on its first out-of-sample cell

Pre-registered in §17.79.2, before the run: Replica office1 with the plain
(99.8% saturated) head should land in the saturated cluster, worse than -3%.

```
Replica office1, plain head, online     psnr      lpips
  no fade                             22.3723    0.1103
  conf-fade 0.45                      22.3235    0.1085     -1.6%
```

**-1.6%. It lands in the UN-saturated cluster.** The gate is wrong on the first
cell that could have falsified it.

Worse for the account, the comparison that isolates the variable:

```
plain Replica head (99.8% saturated), online conf-fade effect
  office0     -12.6%
  office1      -1.6%
```

**Same head, same family, same harness — an 8x difference from the SCENE
alone.** Whatever decides the lever's online value, it is not a property of the
head, because the head is identical in those two rows.

So the four-cell perfect separation in §17.79 was a coincidence of which scenes
happened to be paired with which heads, and §17.79.2's "this rehabilitates the
saturation fraction" is **withdrawn**. §17.70.7's rule stands unamended after
all, and now at five exhibits:

```
tail ratio                4 cells, no predictive information
saturation fraction       perfect on 4 cells, falsified on the 5th
opacity-conf coupling     inverted
accumulated alpha         ordered 3, failed the 4th
untreated acc_alpha       failed (Kimi, round 32)
```

I should have caught this before writing §17.79.2. The four cells varied head
AND scene together and I read the separation as being about the head; office1
was already on record in §17.72 as the scene where conf-fade behaved oddly
(the only offline loss at that point). The confound was visible in data I had
already collected.

### 17.79.6 What actually survives, and what the default now rests on

Every online cell measured, all families and heads:

```
Replica office0, plain head    -12.6%
TUM desk,        base head      -6.4%
Replica office0, o-0.6 head     -1.8%
Replica office1, plain head     -1.6%
TUM desk,        TUM head       -0.6%
```

Range -0.6% to -12.6%, **beneficial in every cell, harmful in none**, and not
predicted by any head- or map-level statistic tried. That is a weaker claim than
a gate, and it is the one the default should rest on:

> Leave conf-fade on unconditionally. It has never been harmful in the online
> harness, it is occasionally worth -12%, and nothing available at load time
> predicts which case you are in — so the expected value of gating it is
> negative once the risk of gating it wrongly is counted.

This is also the correct reading of §17.79.1: the online effect is bounded below
by zero because the optimizer removes a prior it disagrees with. A prior that
washes out when wrong and persists when right is exactly the kind of
intervention that should be left on by default.

More online cells are running to test the "never harmful" claim, since it is now
load-bearing for the default and rests on five points.

### 17.79.7 The gate fails on the un-saturated side too

Second out-of-sample cell, pre-registered in §17.79.3's companion run: the o-0.9
penalty head (frac>0.9 = 0.0%, median opacity 0.8395) on Replica office0 should
land in the un-saturated cluster, better than -3%.

```
Replica office0, o-0.9 head, online     psnr      lpips
  no fade                             26.8403    0.0889
  conf-fade 0.45                      26.8770    0.0842     -5.3%
```

**-5.3%**, between the two clusters and much nearer the saturated one. The
prediction misses on this side as well.

Both out-of-sample cells fail, in opposite directions — a saturated head
predicted large that came in small (office1, -1.6%), and an un-saturated head
predicted small that came in large (o-0.9, -5.3%). Testing both sides is what
made the falsification unambiguous rather than a near-miss; a gate confirmed
only where it predicts "large" would have looked half-alive.

Note also that the o-0.9 and o-0.6 heads sit at 0.0% saturation on the same
scene with the same harness and give -5.3% and -1.8%. Two heads, identical on
the gating statistic, differing threefold in the outcome it was supposed to
predict.

### 17.79.8 The "never harmful" record, which is what the default rests on

```
online A/B, conf-fade vs no fade                     lpips effect
Replica office0, plain head                             -12.6%
Replica room0,   plain head                              -8.9%
TUM desk,        base head                               -6.4%
Replica office0, o-0.9 head                              -5.3%
Replica office0, o-0.6 head                              -1.8%
Replica office1, plain head                              -1.6%
TUM desk,        TUM head                                -0.6%
```

**7 cells, 7 beneficial, none harmful.** Range -0.6% to -12.6%, mean about
-5.3%, spanning two families, four heads and five scenes. Nothing predicts the
magnitude; the sign is so far invariant.

That asymmetry is the whole case for the default, and it has a mechanism behind
it (§17.79.1): the lever is a prior on initial opacity, and the optimizer
removes a prior it disagrees with while keeping one it agrees with. A prior that
washes out when wrong and persists when right should be on by default, and no
gate is needed precisely because the downside is bounded near zero.

Four more cells are running to push this to eleven, since it is the one claim in
the arc that the shipped configuration depends on.

### 17.79.9 A third family produces the first harmful cell — and it is negligible

Pre-registered in §17.79.8's companion run: the sign stays negative on 7-scenes
and euroc, because a positive cell would break the default's only remaining
justification.

```
7-scenes chess, 7-scenes head, online     psnr      lpips
  no fade                               13.6824    0.4889
  conf-fade 0.45                        13.6163    0.4897     +0.16%
```

**+0.16%.** The first harmful online cell in eleven, and psnr also falls
(-0.07 dB). "Never harmful" is falsified as an absolute.

The practical conclusion strengthens rather than weakens, because the magnitude
is what matters for a default and it is negligible. Full online record:

```
Replica office3, plain head    -12.7%      TUM desk,  base head       -6.4%
Replica office0, plain head    -12.6%      Replica office0, o-0.9     -5.3%
Replica office2, plain head    -10.2%      Replica office0, o-0.6     -1.8%
Replica room0,   plain head     -8.9%      Replica office1, plain     -1.6%
Replica room1,   plain head     -7.1%      TUM desk,  TUM head        -0.6%
                                           7-scenes chess, 7sc head   +0.16%
```

Read with a noise floor -- the refinement seed sigma is real (§17.51) and cells
under about 1% should be called ties rather than results:

```
clear benefit   9 cells   -1.6% to -12.7%
tie             2 cells   TUM head -0.6%, 7-scenes chess +0.16%
clear harm      0 cells
```

**The correct claim is not "never harmful" but "downside bounded at the noise
floor".** That is what a default needs, and it is now quantified rather than
asserted: the worst outcome observed over eleven cells, three families, five
heads and seven scenes is +0.16%, against a best of -12.7%.

The mechanism from §17.79.1 predicts exactly this shape and should be stated as
the reason the asymmetry is expected rather than lucky: the lever is a prior on
initial opacity, the optimizer discards a prior it disagrees with, so the
downside is bounded by how much damage survives ~2300 steps of disagreement --
near zero -- while the upside is whatever haze the optimizer cannot remove on
its own.

Note 7-scenes chess is also the weakest map in the set (psnr 13.7), and §17.77.1
already recorded that low-quality bakes are where these levers do least. euroc
is still running.

### 17.79.10 Fourth family, and the online record closed at twelve cells

```
euroc MH_01_easy, euroc head, online     psnr      lpips
  no fade                              13.6544    0.3697
  conf-fade 0.45                       13.6319    0.3489     -5.6%
```

Complete online record, `--refiner-conf-fade 0.45` against no fade:

```
Replica office3, plain head    -12.7%      euroc MH_01, euroc head    -5.6%
Replica office0, plain head    -12.6%      Replica office0, o-0.9     -5.3%
Replica office2, plain head    -10.2%      Replica office0, o-0.6     -1.8%
Replica room0,   plain head     -8.9%      Replica office1, plain     -1.6%
Replica room1,   plain head     -7.1%      TUM desk, TUM head         -0.6%
TUM desk,        base head      -6.4%      7-scenes chess, 7sc head   +0.16%
```

**12 cells, 4 families, 6 heads, 8 scenes: 10 clear benefit, 2 ties, 0 harm.**
Range -0.6% to -12.7%, single positive cell +0.16%.

This is the result the shipped default rests on, and it is now broad enough to
state without the family caveat that §17.77.1 had to attach to the offline
numbers. The magnitude remains unpredicted by any statistic tried; the
distribution of outcomes is what justifies the default, not any ability to
anticipate a given map's number.

## 17.80 Re-measuring the offline levers online

§17.79.1's 11x discrepancy applies in principle to every lever in this file
whose record is offline. The cap is the one that matters, for a reason the
conf-fade case did not have: **the cap carries a measured COST** -- psnr -0.1 to
-0.5 dB in every offline cell (§17.68) and a black-fraction rise on euroc. A
lever that keeps its cost and loses its benefit under the harness that ships
would be actively harmful, not merely useless.

PRE-REGISTERED, before the run: the cap's online benefit falls far below its
-16.7% offline figure on Replica office0, by the same mechanism -- the optimizer
re-derives scales it disagrees with, just as it re-derives opacities. If instead
it holds near -16%, then the 11x discrepancy is specific to OPACITY priors and
does not generalize to geometry priors, which would be the more interesting
outcome and would need explaining.

Running alongside it: the head-training route through the online refiner on
7-scenes and euroc, with conf-fade off in both arms so it isolates the training
route from the injection lever. Kimi's round-33 question 2 is the motivation --
if head training survives the harness where the injection lever largely does
not, that is the strongest available argument for the training route, and it has
never been measured that way on these two families.

### 17.80.1 The cap SURVIVES the online harness, and gains psnr instead of costing it

Both halves of this reverse a documented conclusion.

```
Replica office0, plain head, online     psnr      lpips
  no cap                              26.5926    0.0995
  scale-cap 0.009778                  26.6280    0.0897     -9.8%,  +0.035 dB
```

```
the same lever and map, offline 600 steps (17.67)
  none 0.0959 -> cap 0.0799                                 -16.7%
```

**The cap keeps most of its benefit online (-9.8% against -16.7%).** That is
nothing like the opacity prior's collapse (-20.4% -> -1.8%, eleven-fold), so the
pre-registration in §17.80 resolves on its second branch:

> **The 11x harness discrepancy is specific to OPACITY priors and does not
> generalize to geometry priors.**

Which needs an account, and the honest one is that I do not have a confirmed
mechanism. The plausible reading: opacity is a per-Gaussian free parameter that
the optimizer can move directly and cheaply toward whatever the photometric loss
wants, so a prior on it is erased in a few hundred steps. The cap does not just
bias a parameter, it removes coverage that oversized Gaussians were providing,
which changes which basin the optimization is in rather than only where it
starts. That is a hypothesis, and given this session's record on hypotheses it is
recorded as one.

**The psnr sign also flips.** Every offline cap cell cost psnr (-0.11 to -0.51 dB
across four families, §17.68), which I wrote up as "the cap is not a free win
anywhere". Online on this map it GAINS psnr (+0.035 dB) while also improving
lpips. So that characterization was an artifact of the offline harness and is
withdrawn pending the confirmation run.

This matters more than the conf-fade result did, because the cap was the lever
carrying a measured cost, and §17.80's worry was that it would keep the cost and
lose the benefit. The measurement says the opposite on both counts.

A second family (euroc, offline -7.3% lpips / -0.12 dB psnr) is running to
confirm before either reversal is treated as established. One map is not a
result, and two documented conclusions are at stake.

### 17.80.2 Head training on 7-scenes chess: nothing in deployment

```
7-scenes chess, online, conf-fade OFF in both arms     psnr      lpips
  base checkpoint                                    13.5719    0.4897
  7-scenes head                                      13.6681    0.4904    +0.14%
```

psnr +0.096 dB, lpips +0.14% -- a wash, and on the lpips axis marginally the
wrong way. The 7-scenes head beat base on its val split during training, and
that advantage does not reach the deployed map on this sequence.

Recorded as a single cell with euroc still running, and it is the first
deployment measurement of the head-training route on a family other than TUM and
Replica. It is also the same sequence that produced the only positive conf-fade
cell (§17.79.9), which is consistent with §17.77.1's observation that the
weakest bakes (psnr 13.7 here) are where all of these levers do least -- but
consistency with an existing story is not evidence for it, and the honest
statement is that 7-scenes chess is where two independent interventions have now
both come out at zero.

## 17.81 Why the head persists and the injected prior does not: the optimizer un-saturates by itself

Kimi's round-33 measurement, computed on the shipped post-polish artifacts
(`scripts/diag_attractor.py`), TUM desk, both arms with conf-fade OFF:

```
post-polish map        n           opacity median  frac>0.9  maxscale med   p90
base checkpoint     2,396,900         0.2645        12.1%      2.893 mm   10.491
TUM head            1,860,034         0.1284         9.2%      2.545 mm    7.489
```

Two things fall out, and the first is the one that explains §17.79.1.

**The optimizer un-saturates on its own, drastically.** The base checkpoint
injects opacity ~1.0 for essentially every Gaussian (§17.58, frac>0.9 = 100%).
After ~2300 online steps its map sits at **median 0.2645, with 33.9% of
Gaussians below 0.1**. The polish does not merely tolerate un-saturation, it
performs it, going far further than any fade we inject.

That is the mechanism for the eleven-fold decay, and it is now measured rather
than hypothesized: **an injected opacity prior washes out because the optimizer
was going to remove that opacity anyway.** `--conf-fade 0.45` multiplies
opacities by a mean 0.55; the optimizer takes them to 0.26 unaided. The prior is
a head start on a journey the optimizer completes without it, so its value is
whatever the optimizer had not finished by the time the budget ran out -- which
is exactly why the effect shrinks as steps grow and why it varies by scene.

**Head training moves the attractor.** The two maps do not converge to the same
place: median opacity 0.2645 vs 0.1284, a factor of two, plus a 28.6% lighter
scale tail and 22% fewer Gaussians surviving the cull. So the head does not
merely start the optimizer somewhere better, it changes where the optimizer
ENDS. Kimi's phrasing is the right one for the write-up: *the optimizer does not
restore what the data never asked to restore.*

This resolves Kimi's round-33 question 2 in the direction that favours the
training route, and it is the cleanest mechanistic statement produced in this
whole arc -- notably, again, from a controlled comparison of artifacts rather
than from correlating statistics within one map.

### 17.81.1 What it implies for the two routes

```
head training      changes the converged state          persists under budget
injection prior    changes the starting state           decays with budget
scale cap          removes coverage, not just a value    persists (17.80.1)
```

The cap fits this frame and is no longer anomalous: it does not bias a parameter
the optimizer is already moving, it removes geometry the optimizer would have to
re-create, which is a different kind of intervention. That was offered as a
hypothesis in §17.80.1 and this table is consistent with it, though it is still
not a direct test of it.

The practical corollary is worth stating plainly for the write-up: **if you can
retrain the head for your deployment family, do that; the injection lever is
what you use when you cannot, and its value falls as your refinement budget
rises.**

## 17.82 Head training through the online refiner: the largest deployment effect in the project

Base checkpoint vs family head, full online SLAM, `--refiner-conf-fade 0` in
BOTH arms so this isolates the training route from the injection lever
completely.

```
family / sequence          base          head           lpips      psnr
euroc MH_01_easy       13.0482/0.4555  13.6660/0.3606   -20.8%   +0.62 dB
TUM desk (17.77.4)              /0.4035         /0.3718    -7.9%
TUM room (17.77.6)              /0.4801         /0.4325    -9.9%
7-scenes chess         13.5719/0.4897  13.6681/0.4904    +0.14%  +0.10 dB
```

**euroc at -20.8% lpips and +0.62 dB is the largest single effect measured for
any intervention in this project, in the harness that ships.** For comparison,
the best injection-lever cell over twelve online A/Bs was -12.7%, and its median
was about -5%.

Three of four families show a real deployment benefit from head-only training;
7-scenes chess is a wash on both metrics. That sequence is also where the
injection lever produced its only positive cell (§17.79.9), so it is now the
site of two independent interventions coming out at zero -- worth noting as a
pattern, though with n=1 sequence per family here it is not yet a family-level
statement.

Read together with §17.81, this is the project's central engineering claim and
it now has a mechanism behind it:

```
head training      moves the optimizer's converged state   persists under budget
                   (opacity median 0.2645 -> 0.1284)       -7.9% to -20.8%
injection prior    moves the starting state only           decays with budget
                   (optimizer reaches 0.26 unaided)        -0.6% to -12.7%
```

**If you can retrain the head for your deployment family, do that.** The
injection lever is what you use when you cannot, and its value falls as the
refinement budget rises.

Caveats kept with the claim: one sequence per family for euroc and 7-scenes, and
the two TUM cells come from the §17.77 2x2 rather than a dedicated run. eth3d is
running as the fifth family, on a modal-shape sequence (table_3) rather than an
`sfm_*` one, since those five would hit the §17.55 resolution failure with this
head.

### 17.80.3 CORRECTION: the cap's online survival was one map, and the second map contradicts it

§17.80.1 reported, on Replica office0, that the cap keeps most of its offline
benefit online and gains psnr instead of costing it, and drew from that the
conclusion that the harness discrepancy is specific to opacity priors. The
confirmation cell was already running when that was written; it disagrees.

```
                        offline    online     retained    online psnr
Replica office0         -16.7%     -9.8%        59%        +0.035 dB
euroc MH_01_easy         -7.3%     -0.5%         7%        -0.135 dB
   (conf-fade, for scale: -20.4%   -1.8%         9%)
```

**On euroc the cap retains 7% of its offline benefit -- the same order as the
opacity prior's 9% -- and it costs psnr there, as it did offline.**

So both halves of §17.80.1 are withdrawn:

- "The 11x discrepancy is specific to opacity priors" is **not supported**. The
  cap decays like an opacity prior on one family and largely survives on
  another. What varies is the map, not the parameter type.
- "The cap gains psnr online" was one map. It costs psnr on the second, matching
  every offline cell. The original §17.68 characterization -- the cap trades
  psnr for lpips -- stands after all.

I reported that reversal after a single cell while the confirmation run was in
flight. The run existed precisely because one map is not a result, and I should
have waited for it before writing the conclusion rather than writing the
conclusion and the caveat together.

The surviving statement about the cap is weaker and matches everything else in
this file: **its online value is map-dependent and unpredictable, ranging from
-9.8% to -0.5%, and it carries a psnr cost at least some of the time.** Combined
with §17.68's finding that no tail statistic predicts where it pays, the cap is
not a default and should be armed only after a per-map offline on/off arm --
which, given §17.79.1, will itself overstate the online benefit by an unknown
factor. That is a lever with a poor cost-to-confidence ratio and the write-up
should say so.

### 17.82.1 eth3d: the largest deployment effect yet

```
eth3d table_3, online, conf-fade OFF both arms     psnr      lpips
  base checkpoint                                13.2213    0.5575
  eth3d head                                     14.0378    0.3951    -29.1%  +0.82 dB
```

Head-training deployment record, five families:

```
eth3d table_3       -29.1%   +0.82 dB
euroc MH_01_easy    -20.8%   +0.62 dB
TUM room             -9.9%
TUM desk             -7.9%
7-scenes chess       +0.14%  +0.10 dB
```

Four of five families show a large deployment gain; 7-scenes is the lone
exception. The two largest are the two families whose heads were trained on
self-predicted pseudo-depth rather than sensor depth, which is the opposite of
what I would have guessed and is worth flagging for the write-up rather than
explained away here.

Second sequences are running for eth3d, euroc and 7-scenes -- the last because a
single sequence cannot separate "the 7-scenes head is weak" from "chess is a
hard sequence", and that distinction decides whether the exception is a family
result or a scene result.

---

## 17.83 STATE OF PLAY — read this instead of §17.64, which is superseded

§17.64 was written earlier the same day and predates everything below. Where the
two disagree, this section wins.

### 17.83.1 The three results that would go in a paper

```
1  HEAD-ONLY TRAINING, measured through the ONLINE refiner (17.82)
   eth3d  -29.1% lpips / +0.82 dB      TUM room  -9.9%
   euroc  -20.8% lpips / +0.62 dB      TUM desk  -7.9%
                                       7-scenes  +0.14%  (the one exception)
   Mechanism (17.81): it moves the optimizer's CONVERGED state, not its start.
   Post-polish opacity median 0.2645 (base) vs 0.1284 (head).

2  INJECTION THINNING, 12 online cells, 4 families, 6 heads, 8 scenes (17.79.10)
   10 clear benefit (-1.6% to -12.7%), 2 ties, 0 harm. Worst cell +0.16%.
   Mechanism (17.81): the optimizer un-saturates ITSELF from ~1.0 to 0.26, so
   the lever is ACCELERATION ALONG A PATH THE OPTIMIZER ALREADY WALKS. Value
   ~ (distance along that path) x (budget scarcity); decays as polish converges;
   downside bounded because a prior the optimizer overtakes cannot hurt much.

3  A CAUSAL DISSOCIATION between the head's two output channels (17.75, 17.76)
   content degradation (geometric)   -> scale tail +15%, opacity unchanged
   photometric jitter (appearance)   -> scale tail -24%, opacity -16.7 pp
   and the ALLOCATION half: on the photo head, opacity's strongest predictor is
   local photometric disagreement (-0.268 partial) where on the TUM head it is
   backbone confidence (+0.213). Perturb photometry, opacity reorganizes around
   photometry. Both come from controlled pairs, not correlations.
```

### 17.83.2 The methodology record, which Kimi argues is a co-lead not an appendix

```
five gating statistics falsified by out-of-sample data
   tail ratio (4 cells) | saturation fraction (perfect on 4, dead on 2 more)
   opacity-conf coupling (INVERTED) | accumulated alpha | untreated acc alpha
four instrument failures caught, one of which I built this session
   hp_alpha out of calibration | argsort on a ~100%-tied array
   single-sequence roundtrip probe on a heterogeneous release
   acc_alpha's eps-linearization culled by the rasterizer, biased along the
     exact axis it was built to compare
five mechanism accounts proposed and falsified, plus one falsified by a broken
   instrument and then REINSTATED by the fix
```

Portable rules earned, in the order they cost the most:

```
1  Before explaining an effect, put an INTERVAL on it. Rounds 30-32 explained a
   -2.8% "constant" that was a two-point sample of an sd-4.7 distribution whose
   95% CI is -4.0% .. +0.6%. Every number was correct. This cost the whole
   mechanism arc and no instrument was broken.
2  An instrument's validation regime must OVERLAP its usage regime. acc_alpha
   was validated on un-faded maps and used only on faded ones.
3  Cross-harness, the measured EFFECT SIZE of an intervention does not transfer,
   not just absolute scores. Offline 600-step overstated the opacity lever 11x.
4  Diagnostics nominate; per-blob measurement decides. Five statistics, zero
   successes as a-priori gates.
5  Never edit a script that is currently executing (bash reads lazily by byte
   offset; one log came back as NUL bytes).
```

### 17.83.3 Kimi's lever taxonomy, with the correction the euroc cell forces

His round-34 framing, which the data mostly supports:

```
opacity fades       budget bridges -- acceleration only, ship ungated
cap-type edits      content removals -- persist because the optimizer is
                    indifferent to what it did not need
head training       BASIN SELECTION -- the only intervention that moves the
                    endpoint. This is the sentence the training case rests on.
```

**The middle row does not survive the euroc cell**, which landed after he wrote
it: the cap retains 59% of its offline benefit on Replica office0 and **7% on
euroc** -- the same order as the opacity prior's 9%. So "cap-type edits persist"
is a one-map result. The corrected row is: *cap-type edits persist on some maps
and not others, unpredictably, and carry a psnr cost at least some of the time*
(§17.80.3).

Rows 1 and 3 stand and are the two that matter.

### 17.83.4 What ships

```
--refiner-conf-fade 0.45     DEFAULT ON, ungated. Justified by the 12-cell
                             distribution, NOT by any ability to predict which
                             map benefits. conf-vs-uniform is a coin flip
                             (n=12, median -0.06%, CI contains zero).
--refiner-uniform-fade       equally defensible, simpler, within noise
--refiner-scale-cap          NOT a default. Arm per map, knowing the offline
                             arm will overstate the online value.
--crowd-fade                 measured NEGATIVE, kept as a record (17.74.2)
head training                do it if you can retrain for the family
```

### 17.83.5 Open, in priority order

```
1  second sequences for eth3d/euroc/7-scenes head deployment (RUNNING) -- the
   headline rests on one sequence per family for three of five
2  whether 7-scenes is a weak head or chess is a hard sequence (RUNNING: fire,
   office)
3  why the two LARGEST head gains are the two pseudo-depth families (eth3d,
   euroc) -- opposite of expectation, currently unexplained
4  the per-depth-rank gradient probe, now with a TIME-RESOLVED prediction from
   Kimi: unblocking difference large at iter ~50, gone by ~1000
5  17.34 re-analysis under current defaults; head-training seed ensemble;
   Replica mesh controlled-trajectory design
```

### 17.82.2 Second sequences: the headline holds, and 7-scenes is a SCENE effect not a family effect

```
family / sequence            base            head            lpips     psnr
eth3d table_3            13.2213/0.5575  14.0378/0.3951     -29.1%   +0.82 dB
eth3d sofa_1             22.0717/0.2313  23.1519/0.1811     -21.7%   +1.08 dB
euroc MH_01_easy         13.0482/0.4555  13.6660/0.3606     -20.8%   +0.62 dB
TUM room                        /0.4801         /0.4325      -9.9%
TUM desk                        /0.4035         /0.3718      -7.9%
euroc V1_01_easy         13.2728/0.4749  13.2233/0.4458      -6.1%   -0.05 dB
7-scenes office          11.3933/0.5498  11.6121/0.5228      -4.9%   +0.22 dB
7-scenes chess           13.5719/0.4897  13.6681/0.4904      +0.14%  +0.10 dB
```

**8 cells, 5 families, 7 benefit, 1 null.** Every family has at least one clear
positive cell.

**RETRACTED, see 17.82.3.** On two cells (chess +0.14%, office -4.9%) I wrote
that the exception was a scene rather than a family. The third 7-scenes sequence
falsifies that; the family reading was right and I called it early off n=2.

Within-family spread is large: eth3d -29.1% and -21.7%; euroc -20.8% and -6.1%.
So the family-level numbers should be reported as ranges, never as point
estimates from one sequence -- which is exactly the error §17.72 caught for the
injection lever when office1 broke a two-map pattern.

Data limitation worth recording for anyone repeating this: **only chess, office
and pumpkin ship `groundtruth.txt` in this 7-scenes release.** fire, heads,
redkitchen and stairs cannot be scored by `eval_map_quality.py` at all; the fire
cell was run and had to be discarded after the fact. Check for the file before
spending two SLAM runs on a sequence.

pumpkin (the third and last evaluable 7-scenes sequence) and a third TUM
sequence are running.

### 17.82.3 The third 7-scenes cell: it IS a family effect, and my n=2 call was wrong

```
7-scenes pumpkin      base 14.6259/0.4852    head 14.7660/0.4852    0.00%  +0.14 dB
```

lpips identical to four decimals. Full 7-scenes record: **chess +0.14%, pumpkin
0.00%, office -4.9%** -- two nulls and one modest gain, against every other
family having at least one cell past -20%.

So §17.82.2's "the exception is a scene, not a family" is **withdrawn**. I wrote
it after office reversed chess, on two cells, in the same session in which I had
already recorded (§17.72, §17.78.1) that two-point patterns in this project have
repeatedly failed at n=3. The correction is not that the data changed; it is
that I made a family-level call off two sequences immediately after writing a
rule against exactly that.

Full head-training deployment table, ten cells, five families:

```
eth3d table_3     -29.1%  +0.82 dB        TUM room           -9.9%
TUM plant         -25.9%  +1.29 dB        TUM desk           -7.9%
eth3d sofa_1      -21.7%  +1.08 dB        euroc V1_01        -6.1%  -0.05 dB
euroc MH_01       -20.8%  +0.62 dB        7-scenes office    -4.9%  +0.22 dB
                                          7-scenes pumpkin    0.00% +0.14 dB
                                          7-scenes chess     +0.14% +0.10 dB
```

By family, as ranges rather than point estimates:

```
eth3d      -21.7% to -29.1%   (2 cells)   strongest, both cells large
TUM         -7.9% to -25.9%   (3 cells)   widest spread
euroc       -6.1% to -20.8%   (2 cells)
7-scenes    +0.14% to -4.9%   (3 cells)   the weak family
```

**Four of five families gain substantially; 7-scenes is genuinely weaker.** Why
is unexplained. Bake quality is not it -- 7-scenes office (psnr 11.4) is the
weakest map in the table and is the only 7-scenes cell that gains, while chess
at 13.7 gains nothing. Recorded as open.

Third sequences for eth3d and euroc and a fourth for TUM are running, since
those families rest on two or three cells with spreads of 7 to 18 points.

### 17.82.4 Third/fourth cells, and a repeat offence

```
eth3d plant_1     base 20.3776/0.4088   head 20.1810/0.3796    -7.1%   -0.20 dB
TUM teddy         base 12.8566/0.5304   head 13.1675/0.4439   -16.3%   +0.31 dB
euroc V2_01_easy  UNSCORABLE -- no groundtruth.txt, two SLAM runs wasted
```

eth3d's third cell is **-7.1%**, far below its other two (-29.1%, -21.7%), so
the family range widens to -7.1% .. -29.1%. TUM's fourth is -16.3%, inside its
-7.9% .. -25.9% band. Neither family is well summarized by a single number, and
the spread within eth3d is now as large as the spread between families.

**The repeat offence.** §17.82.2 recorded, one turn earlier, that four of seven
7-scenes sequences ship no `groundtruth.txt` and that the check costs one line
and belongs before the runs. I then queued euroc V2_01_easy without checking and
lost two more SLAM runs the same way. Only 2 of 11 euroc sequences are scorable
(MH_01_easy, V1_01_easy) -- both already used -- so **euroc is capped at two
cells with this release** and its range cannot be firmed further.

Writing the lesson down did not prevent the repeat, so it is now a script:
`scripts/check_evaluable.py`, which exits non-zero on any unscorable sequence
and is invoked as a gate before the batch launches rather than as advice.

```
euroc      2/11 scorable   MH_01_easy, V1_01_easy
7-scenes   3/7  scorable   chess, office, pumpkin
eth3d      widely present
TUM        present
Replica    written from traj.txt by the dataloader on first load
```

That is the second time this session that a rule recorded in prose failed and a
mechanical check was needed instead -- the first being the `--refiner-gpu`
validation that caught a bad device index (§17.77.6). The pattern is worth
stating: **in this project, prose rules have not survived contact with my own
next batch; the ones that hold are the ones that fail a command.**

---

---

## 17.84 交接记录：一次会话内跨压缩的进度重复，已修复；round 33(Kimi计数35)的净新增内容

**先说清楚发生了什么，因为它值得作为方法论记录。** 在这次会话的某个点上，我
基于一份陈旧的理解重新推导了 §17.75/17.76/17.78 已经确立的结论(光度学2x2的
反转结果、crowd-fade 在预算匹配下的失败)，把它们当作"新发现"追加到了文件末尾，
编号也撞车成了 §17.73/17.74/17.75。追加的内容和真实记录在数字上完全吻合(例如
两边都测出 frac>0.9 从 99.8%→83.1%，opacity-disagree partial 都是 -0.268，
crowd T=4.0 在 vox=3cm 下都精确落在 acc_alpha=9.36)——这不是巧合，是同一组
实验在会话被压缩后被我不知情地重跑了一遍。发现方式：在处理 Kimi round 33 的
回复时，它引用了一些我发出的 prompt 里根本没有的具体数字(`combo` 臂、crowd
vs conf 的精确排序)，这个不一致promot我去读 SKILL.md 的真实全文，才发现
§17.76-17.83 早已存在且更完整。**处理方式是直接删除重复内容，而不是保留两份
相互矛盾的记录**——已删除的部分见本节之前的 git 历史。

**真实的当前进度基准是 §17.83(STATE OF PLAY)+ §17.82.4**，不是我之前误以为的
§17.75。以下只记录在修复之后、基于真实状态做出的净新增工作。

### 17.84.1 Kimi round 33 回复中，独立验证之后仍然成立的部分

Kimi 的回复引用了具体数字来源不明的 "12-map, CI [-3.96,+0.63]"——但这个结论
本身在真实记录里已经用**更严格的方式**确立了(§17.78"closed: no measurable
difference, on 12 maps",§17.83.4 "median -0.06%, CI contains zero")，不需要
我再重新验证一遍。真正新增的是他这一轮给出的两个方案性判断：

**Q1(是否需要给 fade 加"光度学职责"防护)：不需要。** 论证:该顾虑(fade 可能
破坏正在做光度学调和工作的 opacity)在量级上已经被现有测量界定——TUM-head
地图上(opacity 承担光度学工作最多的部署场景，disagree partial 仅 -0.096)，
uniform 和 conf 两个 fade 打平，代价 0.08-0.09 dB。饱和头(Replica/base)上
完全没有光度学工作可破坏，而这正是杠杆收益最大的地方。不加防护；改为免费
预案:**若任何 fade 分支未来在某张图上出现损失，先对照 `disagree` 场做定位
分析，再决定要不要设计防护。**

**Q2(更强的光度学扰动 vs crowd-fade 变体，哪个优先)：更强光度学扰动优先。**
论证:crowd-fade(体素版)在预算匹配下已经**两次**被证伪(§17.74.2:单独更差、
混入 conf 还会稀释 conf)，继续做它的射线版本是在抢救一个已经告负的杠杆；
而更强的光度学扰动是"一次训练"就可能同时完成因果解释 AND 产出一个训练配方
级别的修复——如果光度学增强能自然教会透明度，就可能从根源上替代 relu 惩罚
这条路线。他给出更锐利的机制:纯全局亮度抖动(现有 photo 臂)几乎能被逐
Gaussian 的颜色残差完全兜住(一个标量)，尚且把 frac>0.9 从 99.4%→52.1%(按
frac>0.99 口径)打了对折；**空间变化型**的光度学扰动(需要的修正量取决于该点
落在目标视角画面里的位置，而不只是取决于它来自源图的哪个像素)按源像素预测
的颜色残差没法预先兜住，理应招募更多 opacity。

### 17.84.2 已实现并已在 GPU0 上启动:`replica-photo-spatial`

`splatt3r_core/data/replica/replica.py` 新增 degrade 模式
`"photometry-spatial"`：对每对视角中的一个视角施加

```
per-channel WB 抖动    每通道独立 gain ~ U(0.9, 1.1)，刻意保持温和，
                      因为这部分本身接近能被颜色通道兜住的范围
径向渐晕(vignetting)  中心随机偏移(图像 35%-65% 范围内)，强度 v~U(0.25,0.55)，
                      falloff = 1 - v*(r/r_max)^2
```

已在 `train_lora_per_scene.py` 的 `FAMILIES` 注册为 `replica-photo-spatial`
(288,512 分辨率，与其它 Replica 变体一致)，coverage cache 硬链接自
`replica_train/val_...pkl`(仅像素内容改变，几何/位姿/覆盖度不变，标准做法，
见 §17.65.1 对 `replica-noisy` 的同一处理)。`exp_head_only.py` 的
`--family` choices 里原本缺这一项，已补上(否则直接报 `invalid choice`，
第一次启动就是这样失败并重启的)。

**预注册(Kimi round 33，启动之前写下)**：40 epoch 训完后，
`scripts/diag_colocation.py` 探测该头在 Replica office0 上的
`frac(opacity>0.9)` 应落在 **30%-60%** 区间——比 plain photometry 的 83.1%
更接近 TUM 的 12.5%，但不会打平。若真落在这个区间，"结构化光度学分歧"就
基本坐实为真实数据集不饱和的主因，depth-mask/invalid 结构分摊剩余部分；若仍
停在 70-80%，说明光度学只是次要通道，mask 候选原因通过排除法接管大部分解释。

正在训练中，训完后立刻用 `diag_colocation.py` 读数、送 Kimi 独立分析。

### 17.84.3 GPU1:补充 conf-vs-uniform 样本(office3/4, room1/2)——低优先级确认性数据

由于 §17.78/§17.83.4 已经用 12 张图把这个问题判定为"统计上不可区分、经费
无关紧要"，这个补充实验**不解决任何悬而未决的问题**，只是把样本量从 12
扩到最多 16，进一步收窄一个已经足够窄的置信区间。跑起来的原因是它在我发现
进度重复*之前*就已经启动，且不影响其它工作，就让它按原计划跑完作为额外确
认数据；跑完不会改变 §17.83.4 的结论，也不会因此重新打开这个问题。跑完后
GPU1 应转去 §17.83.5 列出的真正未解决问题，优先级最高的是:为什么两个用
伪深度(pseudo-depth)监督的家族(eth3d、euroc)反而是 head-training 收益
最大的两个家族——这与直觉相反且目前无解释。

---

## 17.85 Round 34: the pseudo-depth account is mechanically dead; the coverage account is falsified by measurement; un-saturation has a clean counterexample

### 17.85.1 Kimi actually read the code, and it kills my proposed mechanism at the root

I proposed (round 34): pseudo-depth trains the head in the network's own
self-consistent frame, so there's no train/deploy scale mismatch, while real
sensor depth forces a correction toward an external metric scale the deployed
SLAM pipeline may not honour.

Kimi checked the actual training code rather than reasoning about it, and the
mechanism does not exist: **the head-only loss is purely photometric** --
`1.0*masked_MSE + 0.25*masked_LPIPS`, in pixel space only. Gaussian means are
`pts3d.detach()`, frozen backbone output, never trainable in this recipe. The
MASt3R geometric loss is disabled (`mast3r_loss_weight=None`). **`is_metric_scale`
is a no-op in the entire head-only path** -- its only consumer is that disabled
loss. Depth never supervises the head, in any frame, metric or otherwise. My
account had no pipe to act through. Fixed the misleading comment at the two
call sites (`splatt3r_core/data/{eth3d,euroc}/*.py`) so the next reader does not
make the same assumption.

Depth touches training in exactly two non-differentiable places: the loss mask
(`calculate_in_frustum_mask`: depth>0, in-frustum, cross-view depth agreement
within 0.1m) and pair selection (`compute_coverage`, same test plus a 0.5m
camera prefilter). Kimi's sharper point: for euroc/eth3d these run in **mixed
units** -- network-scale pseudo-depth against metric GT poses -- while TUM/
7-scenes run in consistent metric units throughout. If scale-frame consistency
mattered, it should hurt the pseudo-depth families, not help them. My account's
mechanism, where it exists in the code at all, predicts the wrong sign.

### 17.85.2 Free check (a): supervision coverage, measured -- and it falsifies the leading alternative

Kimi's own leading replacement candidate was supervision coverage: TUM's real
depth masks out its ~40% sensor-invalid pixels (round 29), while pseudo-depth
families only lose their `conf<1.5` zeros, so they should supervise far more of
the image. Predicted: TUM ~60%, euroc/eth3d ~85%+, Replica ~100%.

Measured directly (`calculate_loss_mask` on 40 real training batches per
family, before spending any GPU-hour on a retrain):

```
family      mean supervised fraction    median     sd
tum               60.0%                 59.2%      10.2
7-scenes          65.1%                 67.6%       8.9
euroc             39.5%                 40.9%      15.3   <- LOWEST, not highest
eth3d             61.9%                 63.3%      12.5   <- ~= TUM, not >>
replica           79.8%                 81.7%      11.2   <- high, not ~100%
```

**Falsified.** TUM's 60% is correctly predicted, but euroc has the LOWEST
coverage of any family (39.5%) while having the SECOND-LARGEST deployment gain
(-20.8%), and eth3d's 61.9% is statistically indistinguishable from TUM's 60.0%
despite a 3x larger deployment gain (-29.1% vs -9.9%). Coverage fraction orders
nothing. This is exactly the check Kimi flagged might make the retrain
unnecessary -- it did, by ruling the leading candidate out before the
GPU-hour was spent.

### 17.85.3 Free check (b): un-saturation degree, and 7-scenes is a real counterexample, not a pending one

Kimi's second candidate, which needed no new data for four families and fit
them in perfect rank order: eth3d 0.24 (-29.1%), euroc 0.38 (-20.8%), TUM 0.66
(-9.9%), Replica 0.998 (null/negative pre-penalty). He flagged 7-scenes (median
0.4537 from the round-27 table, deployment null) as the one cell that breaks
the ordering, and left it "pending bake soundness" rather than counted against
the account.

Measured directly (identical 5-pair probe protocol, `checkpoints/head_only_long
/7-scenes/head_best.pt` on `datasets/7-scenes/chess`):

```
base            opacity median 1.0000   frac>0.9 100.0%   frac<0.1  0.0%
7-scenes head   opacity median 0.4789   frac>0.9  31.4%   frac<0.1  4.4%
```

Confirms the round-27 figure (0.4537 vs 0.4789 -- different probe draws, same
regime) rather than an artifact. **7-scenes IS substantially un-saturated** --
more than TUM (0.66), comparable to euroc (0.38) -- yet its deployment gain is
null (+0.14%). This is not a cell waiting on a bake-soundness check; it is a
direct, measured counterexample to "un-saturation degree orders deployment
gain." The account explains 4 of 5 families and is wrong on the fifth in a way
that cannot be explained by insufficient data.

### 17.85.4 Where this leaves the question

Both of Kimi's own candidates are now damaged by direct measurement rather than
argument: coverage is falsified outright; un-saturation is a 4/5 fit with one
clean counterexample. What is left standing, unfalsified because untested:
headroom (eth3d/euroc bases were the worst in the matrix) and content/
distribution distance (euroc's machine-hall imagery is far from the backbone's
training distribution). Neither has been measured against the actual deployment
gains yet. Sent to Kimi for the next round rather than pursued alone, since the
"is 7-scenes' bake even sound" question and the headroom/content confounds both
need judgment calls about which controlled experiment actually discriminates
them -- the same kind of design question the last several rounds have needed
him for.

### 17.85.5 Round 35: both replacement candidates are now dead, and euroc's low coverage explains itself

Kimi owned both kills cleanly. New detail he added: euroc's 39.5% coverage
(the lowest of any family, §17.85.2) is not an isolated anomaly -- it is the
**mixed-unit mask signature itself**, the mechanism his round-34 reply
predicted would exist (network-scale pseudo-depth vs metric GT poses, tested
against a 0.1m atol). euroc's machine-hall scenes likely carry a larger network-
vs-metric scale error than eth3d's smaller rooms, so its `atol=0.1` consistency
test fails on more pixels -- 61.9% eth3d coverage vs 39.5% euroc is consistent
with that. **Recorded as a genuine recipe quirk, unresolved**: the mask's
geometric-consistency test is miscalibrated in mixed units for both pseudo-depth
families, and it excludes more than half of euroc's pixels from supervision.
Whether this is worth fixing is orthogonal to the deployment-gain question --
euroc gained the second-most despite it, so "more supervision would help" does
not follow from "the mask is measurably wrong for this family."

Three mechanism candidates for the depth-source pattern have now died on
contact with data: scale-frame (killed by the plumbing report, §17.85.1),
coverage (killed by direct measurement, §17.85.2), un-saturation (killed by
direct measurement on the fifth cell, §17.85.3).

### 17.85.6 The design that replaces family-level pattern-matching: predict a cell before running it

Kimi's reframe: headroom and content-distance were never two candidates, they
are one variable measured two ways. **Base deficit is the observable form of
distribution distance** -- a family/sequence far from the backbone's training
distribution bakes a bad base map, and a bad base map is what "headroom" means.
So the predictor to test is **per-sequence base val deficit**, not family
membership and not depth source.

Existing val-gain numbers from the base-vs-head matrix are already suggestively
ordered: euroc (+3.69 dB) ~= eth3d (+3.15 dB) >> 7-scenes (+2.15 dB) >
Replica (+1.81 dB) > TUM (+1.74 dB). The two cells that break a clean ordering
against deployment gain are exactly the two with independent explanations
already on record (7-scenes' bake-soundness question, Replica's saturation
story) -- but Kimi's point, correctly, is that **consistency on five points is
not a test.** A pattern across five families is a hypothesis generator, not a
conclusion; the conclusion lives in a cell predicted before it is run.

**The design, three steps:**

1. Compute base val deficit for every training sequence in every family --
   free, no SLAM run, the existing eval harness does this.
2. Correlate against the deployment gains already measured, at the SEQUENCE
   level (not family maxima) -- roughly 8-10 points rather than 5.
3. **Pre-register and run the separating cells**: (i) the highest-deficit
   REAL-depth sequence not yet deployed -- family/depth-source predicts a
   small gain, base-deficit predicts a large one; (ii) the lowest-deficit
   eth3d/euroc sequence -- family/depth-source predicts large, base-deficit
   predicts small. If both land on the base-deficit side, the account is
   confirmed out-of-sample -- the first mechanism in this entire inquiry to
   earn that status rather than fit-after-the-fact. If they do not, the
   correct final entry is "family-level variation of unknown origin, n too
   small" -- stated plainly rather than chased with a fifth account.

**Standing rules added, verbatim, because this is the fourth mechanism hunt in
the project to end in a falsified account:**

> A pattern across five families is a hypothesis generator, never a
> conclusion; the conclusion lives in the cell you predict before you run it.
> Report all sequences, never family maxima -- "best sequence each" is a
> selection statistic that manufactures patterns.

### 17.85.7 What the depth-source pattern is now: effect real, cause unassigned

Kimi's precision, worth keeping exact: the DEPLOYMENT GAINS on eth3d/euroc are
real measurements (-29%/-21% against a 0.17% floor, not noise). What has died
three times is the ATTRIBUTION of those gains to depth source. The correct
current entry is *"large deployment gains measured on eth3d/euroc; attribution
unresolved; base deficit is the live untested candidate"* -- not "no effect,"
and not "pseudo-depth causes it." The depth-source framing is retired from this
point forward; base val deficit is what gets tested next.

Base-deficit computation and the separating-cell selection are next, before any
further deployment A/Bs are queued for this question.

### 17.85.8 Base-deficit tested on data already in hand: also dead, no new GPU-hour needed

Kimi's step 1-2 (compute base deficit per sequence, correlate against existing
deployment gains) doesn't require new measurements -- every deployment A/B
already reports both base and head lpips. Compiled the 11 sequences with
recorded base lpips and gain%:

```
base_lpips   gain      sequence
0.5575      -29.1%     eth3d table_3
0.5498       -4.9%     7-scenes office
0.5304      -16.3%     TUM teddy
0.4897       +0.1%     7-scenes chess
0.4852        0.0%     7-scenes pumpkin
0.4801       -9.9%     TUM room
0.4749       -6.1%     euroc V1_01
0.4555      -20.8%     euroc MH_01
0.4088       -7.1%     eth3d plant_1
0.4035       -7.9%     TUM desk
0.2313      -21.7%     eth3d sofa_1

Pearson r  = 0.143
Spearman rho = 0.145
```

**Essentially no correlation.** The single most damaging point: eth3d sofa_1
has the LOWEST base deficit of any sequence (0.2313 -- the base is already
good there) and the SECOND-LARGEST gain (-21.7%) -- directly against the
headroom account. 7-scenes office has the second-worst base (0.5498) and only
a modest gain (-4.9%). Existing data refutes base-deficit before any separating
cell needs to be run.

**Fourth candidate dead by direct measurement**, joining scale-frame (plumbing),
coverage (measurement), un-saturation (measurement). No mechanism proposed for
the depth-source pattern across five rounds has survived contact with data.
Sent back to Kimi rather than pursued further alone -- per his own round-35
rule, this is exactly the point at which "family-level variation of unknown
origin, n too small" becomes the responsible entry, and whether to write that
now or spend the two GPU-hours on separating cells anyway (in case lpips is
the wrong deficit metric, e.g. psnr-based deficit or the two-view val loss
before any SLAM run) is a judgment call handed back rather than made alone.

### 17.85.9 Round 36: closed. The 7-scenes cell is not obviously broken; neither metric variant would have changed the answer

Kimi's argument for closing without further metric variants: both killer points
are metric-robust. eth3d sofa_1 has the best lpips AND would have the best psnr
(the two rarely disagree on "is this base good"); 7-scenes office inverts under
lpips, would invert under psnr, and inverts under the pre-SLAM val loss too
(base val psnr 13.71 vs TUM's 15.09, yet null deployment gain). Base-psnr and
pre-SLAM-val are re-measurements of the same near-zero correlation with
different noise -- not queued.

The one non-metric-variant action he named -- checking 7-scenes' round-29
bake-soundness flag, never resolved -- is free (log analysis, no GPU) and
worth doing before writing the final sentence. Checked:

```
RELOCALIZING / Failed to relocalize count, all four 7-scenes deployment logs: 0
```

No tracking loss in any of the four runs. And the psnr range is not anomalous
against the other real-sensor families measured in the SAME harness:

```
eth3d table_3     base psnr 13.2213
euroc MH_01       base psnr 13.0482
7-scenes chess    base psnr 13.5719
7-scenes pumpkin  base psnr 14.6259
7-scenes office   base psnr 11.3933
```

7-scenes sits inside the same 11-15 psnr band as eth3d/euroc/TUM. The round-29
suspicion of broken bakes was most likely an artifact of comparing this band to
Replica's 21-27 psnr regime (synthetic, exact geometry) rather than to the other
real-sensor deployments it actually belongs with. **The cell is not obviously
broken.** It stands as real data, and it remains the point that kills every
deficit account tested.

### 17.85.10 CLOSED: family-level variation of unknown origin, n too small

Final scorecard, four mechanism candidates for "why do eth3d/euroc gain most
from head training", each killed by a pre-registered check rather than by
argument:

```
scale-frame consistency    DEAD -- inert in the training path (plumbing, 17.85.1)
supervision coverage       DEAD -- anti-correlated in the decisive cell (17.85.2)
opacity un-saturation      DEAD -- clean counterexample, 7-scenes (17.85.3)
base deficit (headroom)    DEAD -- rho=0.145 on 11 sequences, sofa_1 kills it (17.85.8)
```

Per Kimi's framing, adopted for the write-up: this is not "we don't know" but
a **negative-results result** -- the four candidates any reader would propose
have each been pre-registered and killed, which is itself the useful content.
The reporting rule for the paper: give the full per-sequence table (all
measured gains, no family maxima), state plainly that the variation is
unattributed after four tested candidates, and invite a fifth mechanism only
if it comes with a pre-registered out-of-sample cell -- the standard this
inquiry finally held itself to on the fourth attempt.

**The depth-source framing is retired.** What remains open and worth one more
experiment, per Kimi: not another correlation, but the one CAUSAL test never
run -- see §17.85.11.

### 17.85.11 The one remaining causal test: TUM head trained on pseudo-depth

Every check so far has been correlational (across families, across sequences).
Kimi's argument for one more experiment, not a fifth correlation: **train a TUM
head on pseudo-depth instead of real sensor depth** -- same family, same
scenes, same recipe, same seed structure, ONE variable moved. This is the only
design in the whole inquiry that can answer "does depth-supervision source
matter at all" causally rather than by pattern-matching across confounded
families.

Pre-registered:

```
pseudo-TUM deployment gain ~= real-TUM's -9.9%
    -> depth source is dead, PERMANENTLY, by a controlled negative rather
       than an exhausted correlation list. Stronger than the correlational
       close in 17.85.10.
pseudo-TUM deployment gain >> real-TUM's -9.9%
    -> depth source causally matters after all, and the mechanism hunt
       reopens with a proven lever to explain rather than a correlation
       to explain away.
```

Launching now: `scripts/precompute_pseudo_depth.py` for TUM (idempotent),
then a 40-epoch head-only run identical to the existing TUM recipe except for
depth source, then a deployment A/B against the existing real-depth TUM head.

---

## 17.86 The causal test lands: depth source is dead, permanently, by a controlled negative

### 17.86.1 E1: TUM room, real-depth head vs pseudo-depth head, everything else fixed

```
                psnr        lpips
real-depth    13.0996      0.4346
pseudo-depth  13.1369      0.4397     +1.17% relative (pseudo slightly worse, negligible)
```

**Statistically indistinguishable.** The delta (1.17%) is far inside the noise
this project has already characterized (scene-level sd ~4.7%, §17.75.1-style
CIs routinely spanning several percent on n=1 comparisons). Reproducibility
check: the real-depth arm's lpips here (0.4346) matches the previously-recorded
real-depth TUM-room number (0.4325, §17.77.6/17.82) to within 0.5% relative --
same head, same behaviour, confirming this run is not an anomalous bake.

**Kimi's pre-registered null is confirmed exactly.** Same family, same scenes,
same recipe, same seed structure -- the ONE variable moved (depth source) and
deployment gain did not move with it. This is the first CAUSAL result in the
entire depth-source inquiry, after five rounds of correlational analysis and
four dead candidates (§17.85). It closes the question more strongly than the
correlational analysis could: not "no pattern found" but "the one variable
directly implicated by the family split, moved directly, produces no effect."

**Depth supervision source is retired as an explanation for the eth3d/euroc
deployment gains, permanently.** Whatever drives those gains, it is not which
kind of depth (real sensor vs self-predicted) trained the head -- consistent
with, and now doubly confirmed by, the plumbing finding that depth never
supervises the head's parameters at all in this recipe (§17.85.1).

### 17.86.2 E2: the spatially-varying photometry arm deploys, and lands exactly where predicted

Deployment A/B, Replica office0, base checkpoint vs `replica-photo-spatial`
head (per-channel WB jitter + radial vignetting on one view, §17.84.2):

```
              psnr        lpips
base        26.4177      0.1086
photo-spatial 26.6552    0.0934     -14.0% lpips   +0.24 dB
```

A real, solid deployment gain -- comparable in size to the o-0.9 opacity
penalty's deployment result (+0.20 dB / -15.2%, §17.69), but produced by
teaching the channel through realistic photometric augmentation rather than by
imposing it with a loss penalty.

**The opacity readout confirms the pre-registration.** Measured via
`diag_colocation.py`, identical protocol to every other head:

```
photo-spatial head    opacity median 0.8344   frac>0.9 34.7%
```

Kimi's round-33 pre-registration was **frac>0.9 in [30%, 60%]** -- lands right
at the bottom of the predicted range, and materially lower than plain
photometry's 83.1% (frac>0.99 comparison would be sharper still, matching the
pattern from §17.73.1). The mechanistic account -- spatially-varying photometric
disagreement, which a per-source-pixel colour residual cannot fully absorb
because the needed correction depends on where a point lands in the TARGET
frame -- predicted more opacity recruitment than global brightness alone, and
got it.

Still short of TUM's unaided 12.5%, consistent with §17.73.3's standing
conclusion: photometric disagreement is A driver, not sufficient alone.

One change worth flagging: this head's leading opacity predictor is `conf`
(+0.310 partial), NOT `disagree` (-0.071, weak) -- the OPPOSITE of the plain
photometry head, whose leading predictor was `disagree` (-0.268). So the
spatially-varying perturbation recruited MORE opacity overall (34.7% vs 83.1%
frac>0.9) while reorganizing WHAT that opacity tracks -- toward backbone
confidence rather than toward local photometric disagreement. Not yet
understood; flagged for the next round rather than explained here.

### 17.86.3 Standing state of both open lines, updated

```
depth-source pattern (eth3d/euroc gain most)   CLOSED -- unattributed, and now
                                                the one causal lever available
                                                was tested and found inert
photometric-augmentation-as-recipe-fix         POSITIVE, replicated at deployment:
                                                -14.0% lpips / +0.24 dB, opacity
                                                recruitment landed in the
                                                pre-registered range
```

### 17.86.4 Free check on candidate 1: the vignetting field does not correlate with conf. Candidate 1 refuted.

Tested without any new training: applied a FRESH, out-of-training-sample
vignetting field (identical formula to `replica.py`'s degrade, new random
centre/strength) to held-out Replica office0 pairs, ran the photo-spatial head,
and correlated the KNOWN ground-truth vignetting field directly against the
head's predicted opacity and predicted backbone confidence on the same pixel
grid:

```
n=737,280 pixels, 5 pairs
spearman(vignette_falloff, opacity) = +0.032   (~0)
spearman(vignette_falloff, conf)    = -0.037   (~0)
```

**Both negligible.** The vignetted region is NOT where backbone confidence
drops, and it is NOT where the head's predicted opacity drops either. Candidate
1 (conf is a confound of the vignetting perturbation's own spatial structure)
is refuted -- this was a free, out-of-sample generalization test (fresh random
vignette parameters, not the ones any specific training pair saw), so the null
result is not an artifact of testing on memorized noise.

This leaves candidate 2 (the recipe determines which channel the head reaches
for; spatial photometric structure gets learned as a confidence-like signal for
reasons not yet identified) as the only surviving account, and a third
possibility neither of us named: the reorganization is a training-dynamics
artifact of this specific 40-epoch run (single seed) rather than a property of
the perturbation family at all. Sent to Kimi with the refutation rather than
pursued further alone.

### 17.86.5 Round 37: Kimi's unifying account, and the mixed arm

Kimi's answer to the conf/disagree puzzle subsumes both candidates I proposed:
**opacity allocates to wherever the training-time damage field lives; what it
appears to "track" is whichever observable best marks that field.** Global
brightness jitter leaves damage distributed by the scene's own disagreement
structure -> opacity keys to `disagree`. A radial vignette concentrates damage
peripherally -> opacity keys to whatever best marks that region, and if `conf`
does, opacity tracks `conf` -- not because conf causes anything, but because it
is the best available marker of where the perturbation hurt. Under this
account the flip is the prediction, not an anomaly.

His proposed decisive check: compute per-pixel Δloss (perturbed vs clean,
frozen base model, inference only, no training) for each perturbation on a
handful of pairs, then correlate each head's own opacity against its own
perturbation's damage field. Queued for the next free GPU slot -- both GPUs are
currently committed to real training/deployment runs, and building this
correctly means going through the actual render+loss harness rather than a
pixel-difference proxy, which is worth doing right rather than quickly.

His sharper read on my candidate 1 (conf as vignette sensor): conf comes from
the FROZEN backbone (§17.85.1, means/conf are `pts3d.detach()`/frozen pointmap
output) -- head training cannot change it. So candidate 1 can only be about
whether the frozen backbone's conf ALREADY responds to vignetting, not about
the head learning to make it respond. §17.86.4's free check answered this
directly (conf does not move under a fresh, out-of-sample vignette) and by his
logic that kills candidate 1 at the root, independent of the damage-map result.

**Mixed arm launched** (GPU1): `--family replica-mixed`, new `ReplicaData`
degrade `"mixed"` applying content (noise+blur) and photometry-spatial
(WB+vignette) together per view, same rng object advanced through both blocks
so neither perturbation's own calibration changes relative to running alone.
Pre-registered: both channels engage without interference -> frac>0.9 drops to
30-50% AND scale p90 inflates >=10% on the same head. If one suppresses the
other, that is itself a finding (a shared, competed-for budget).

Per Kimi: this is the last mechanism experiment with real decision value:
"beyond this cell, diminishing returns." The office1 confirmatory run (guarding
against a repeat of §17.72's single-scene reversal) is running on GPU0.

Minor documentation bug noticed while building the mixed mode, not yet fixed:
the class docstring says photometry/photometry-spatial degrade the "TARGET
view only," but `get_view`'s rng is seeded by `(sequence, view_idx)`
independently for every call regardless of whether it is serving a context or
target role, so in practice EVERY view gets its own independent draw. This
does not invalidate the measured results (independent per-view jitter still
produces inter-view appearance disagreement, which is what the causal story
needed), but the docstring overclaims a target-only asymmetry the code does
not implement. Flagged for a future cleanup pass, not urgent.

### 17.86.6 office1 confirms: positive on both scenes, not another office1 reversal

```
Replica office1        psnr        lpips
base                 21.7742      0.1202
photo-spatial        22.2909      0.1123      -6.6% lpips   +0.52 dB
```

Positive on both metrics, same direction as office0 (-14.0% lpips/+0.24 dB),
smaller in relative lpips magnitude. **Not a repeat of §17.72's office1
reversal** (where conf-fade flipped sign entirely on this exact scene) --
here the sign holds, only the size differs, which is the expected and
unremarkable kind of within-family spread this project has learned to report
as a range rather than suppress:

```
photo-spatial deployment gain range (n=2): -6.6% to -14.0% lpips, both positive
```

Two-scene confirmation is not conclusive on its own (this project has been
burned by n=2 calls before -- §17.82.3's 7-scenes correction), but it clears
the specific risk this run was checking for. The photo-spatial deployment
result stands as replicated, not scene-specific.

### 17.86.7 E1 second-sequence confirmation: desk agrees, and the sign flips -- exactly what a true null looks like

```
TUM desk                psnr        lpips
real-depth head       14.0955      0.3755
pseudo-depth head     14.1822      0.3720     -0.93% relative (pseudo slightly BETTER)
```

Room showed pseudo slightly worse (+1.17%); desk shows pseudo slightly better
(-0.93%). **Noise straddling zero in opposite directions on the two sequences
is a stronger confirmation of the null than two same-direction near-zero deltas
would have been** -- a consistent small bias in one direction across both
sequences would have at least raised the question of a systematic (if small)
effect; sign flip with both magnitudes inside the project's established noise
floor is the signature of no real difference at all.

E1 is now confirmed on two TUM sequences, matching the same replication
discipline just applied to the photo-spatial result. Depth source stays closed.

### 17.86.8 Four-scene range for the photo-spatial deployment result

```
scene       base            photo-spatial       lpips     psnr
office0   26.4177/0.1086    26.6552/0.0934      -14.0%   +0.24 dB
office1   21.7742/0.1202    22.2909/0.1123       -6.6%   +0.52 dB
room0     24.9794/0.1198    25.5077/0.0968      -19.2%   +0.53 dB
office2   20.8521/0.1629    21.2616/0.1329      -18.4%   +0.41 dB

range: -6.6% to -19.2% lpips, mean -14.55%, all four positive on both metrics
```

Consistently positive across four different Replica scenes -- the deployment
claim for photo-spatial should be reported as this range, not office0's -14.0%
alone (Kimi's point: "office0's -14.0% should be reported as the top of a
range, not the number" -- room0 turned out larger still, so office0 was
actually mid-range, which only reinforces the point that a single scene is not
representative in either direction). office2 (4th scene, confirmatory, run
2026-08-17) lands near the top of the existing range rather than outside it --
no change to the range bounds, mean shifts from -13.3% to -14.55%.

---

## 17.87 Kimi's causal test for the trajectory story: falsified, in the wrong direction

Same baked map (TUM room, base checkpoint, `logs/frames_head`), two supervision
subsets split by LOCAL translation/rotation ratio (`diag_within_seq.py`'s exact
`local_ratio()`, 227 frames each, matched counts, clean separation: low subset
ratio range [0.002,0.007], high subset [0.012,0.036], zero overlap). Held-out
set is independent of frames-traj content by construction (confirmed by
reading `refine_local.py` before relying on it: held-out frames come from
`gt_pairs`/`kf_idx`, never from the frames-traj file), so both arms share the
identical 50-frame eval set.

```
supervision subset          iter 0 lpips   iter 600 lpips   gain
low ratio  (rotation-heavy)    0.5964         0.5385        -9.71%
high ratio (translation-heavy) 0.5964         0.5522        -7.41%
```

**Kimi's pre-registration**: high-parallax supervision should yield >= 2x the
gain of low-parallax supervision. **Measured: the opposite sign of the
prediction.** Rotation-heavy supervision produced the LARGER gain (-9.71% vs
-7.41%), not the smaller one -- not a null result, a reversal.

### 17.87.1 What this adds to the existing record

This is the second independent measurement to turn against the
supervision-parallax mediator specifically (as opposed to the cross-sequence
correlation itself, which stands unchallenged at r=-0.978). §17.38's
within-sequence test already found the mediator sign-inconsistent (4 negative,
3 positive across 7 sequences) despite having more dynamic range than the raw
ratio. This causal manipulation -- not just an observational correlation --
now shows the mediator predicting the wrong sign on the one sequence tested.

Two mediator failures, one correlational and inconsistent, one causal and
reversed, are a stronger joint statement than either alone: **supervision
parallax, however it is operationalized, does not appear to be the channel.**
The cross-sequence correlation (motion type vs scene type, r=-0.978) remains
real and unexplained; what is now further weakened is any specific causal
story for WHY it holds.

### 17.87.2 Honest caveats before this closes anything

n=1 sequence, n=1 causal manipulation. Per this project's own repeatedly
re-learned rule (§17.72, §17.82.3), a single reversal should not be promoted to
"the mediator is dead" any more than a single confirmation should have been
promoted to "the mediator is proven." The result is reported and sent to Kimi
rather than acted on alone.

One live confound specific to this design, flagged in the pre-registration and
not yet checked: the two supervision subsets may cover different scene content
(rotation-heavy frames tend to sweep more distant surfaces; translation-heavy
frames tend to densely re-observe near ones), so "low beats high" could reflect
which SURFACES got supervised rather than which MOTION TYPE did. The held-out
coverage overlap between the two subsets was pre-registered to be reported
alongside the gains and has not yet been computed.

### 17.87.3 The coverage confound, checked: the two subsets occupy the same region

```
subset   n     centroid            pos spread (mean/max)   view-dir spread (mean)
low     227   [1.21,-0.27,0.37]        1.08 / 1.52 m              45.9 deg
high    227   [1.23,-0.08,0.24]        0.98 / 1.50 m              38.9 deg

centroid distance (low vs high) = 0.233 m
```

The two subsets' camera centroids sit 0.23m apart against a ~1.5m position
spread within each -- they occupy essentially the same region of the room, not
different areas. Position spread and view-direction spread are also similar
between subsets. This weakens the gross scene-coverage confound (the two
supervision sets are not obviously looking at different parts of the room) but
does not rule out finer content differences (e.g. near-surface vs far-surface
emphasis at matched centroids), which the held-out-pixel-coverage check
Kimi's pre-registration specified would settle more precisely.

### 17.87.4 Second sequence (desk): confirms the ORIGINAL direction, not the room reversal

```
TUM desk, same design, matched 102-frame subsets
supervision           iter 0 lpips   iter 600 lpips   gain
low (rotation-heavy)     0.5071         0.4342        -14.38%
high (translation-heavy) 0.5071         0.3875        -23.59%     1.64x low's gain
```

Coverage check, same method as room: centroid distance 0.24m against ~0.7-1.3m
spread within each subset -- same region, weak confound, consistent with the
room check.

**Desk confirms Kimi's original pre-registration direction** (high beats low,
1.64x -- short of his stated >=2x bar but the right sign and a meaningful
margin). Room reversed it. Two sequences, opposite signs:

```
sequence   low gain    high gain    high/low ratio    direction
room        -9.71%      -7.41%          0.76x         REVERSED
desk       -14.38%     -23.59%          1.64x         confirmed (below 2x bar)
```

### 17.87.5 Verdict, reached without Kimi (quota exhausted this round): the causal manipulation reproduces the EXACT sign-inconsistency already on record, and that is itself the finding

This was meant to be a discriminating test -- Kimi's design, run twice. It did
discriminate, just not between "mediator real" and "mediator absent." What it
discriminated was whether the causal result would look like the CORRELATIONAL
record already in §17.38, and it does, closely:

```
§17.38 (correlational, 7 sequences)   4 negative, 3 positive -- sign-inconsistent
§17.87 (causal, 2 sequences)          1 reversed, 1 confirmed -- sign-inconsistent
```

A mediator that is causally manipulated and still flips sign between two
sequences of the SAME family, using the SAME map-building pipeline and the
SAME split methodology, is not a measurement-noise story -- it is the same
result the correlational test already gave, obtained a different way. Two
independent methods (observe vs manipulate) landing on the same
sign-inconsistent texture is more informative than either alone would be, and
it is informative in the direction of killing the mediator, not supporting it.

**Verdict, by the standard Kimi set one round earlier** ("if it's flat, you
demote the whole trajectory story to map-content association, mechanism
unknown"): a sign-inconsistent causal result meets or exceeds that bar for
demotion. Applying it without waiting for his return, since the logic follows
directly from data already in hand and does not require his judgment to
complete -- if this call turns out wrong when he is back, it costs a
correction, not a wasted experiment.

**Kimi's audit (round 40), after the fact: closure sound, wording wrong.**
"Two independent mediator tests both failed" overstates it -- desk was
directionally POSITIVE (1.64x, just below the pre-registered 2x bar), only
room reversed. The accurate phrase is **"sign-inconsistent across sequences
and methods"** -- which is the stronger argument for closing, not a weaker
one: the causal manipulation and the observational test (§17.38's 4-neg/3-pos)
independently landed on the SAME inconsistent texture, rather than one clean
failure and one ambiguous result. On whether a reversal is stronger evidence
against a mediator than a flat result would be: yes, precisely -- flat says
"no evidence the channel exists" (weak against); opposite signs across
sequences say "no single-direction channel exists" (strong against, and
positively excludes the universal-mechanism version of the claim). Closing
after two sequences, without a third, was also correct: this is a demotion,
not a promotion, and demotions need less evidence.

**One magnitude finding survives the demotion and belongs in the record
regardless**: the supervision-subset effect is LARGE on both sequences
(desk: -23.59% vs -14.38%, a 9-point spread; room: -9.71% vs -7.41%), just
sign-inconsistent in direction. Supervision composition has a large,
scene-dependent effect on refinement gain -- the sharpest version yet of this
session's recurring theme that scene effects dominate lever effects, worth
keeping even though the specific parallax-ratio mechanism is dead.

**Recorded conclusion**: the trajectory/parallax story demotes to *an
unexplained seven-sequence association (r=-0.978), with two independent tests
of its leading candidate mediator (supervision parallax) both failing to
support it* -- §17.38 correlationally (sign-inconsistent), §17.87 causally
(sign-inconsistent, on the same two sequences the mediator was tested on). No
further mechanism hunting on this line without a genuinely new candidate
mediator, which nothing in the current record suggests. The association itself
stays in the file as a real, measured, unexplained seven-point correlation --
worth reporting as exactly that, not as a finding with a story attached.

A third sequence was considered and not run: the marginal value of a third
coin flip on an already sign-inconsistent pattern is low, and this project's
own standing rule (§17.83.5, restated by Kimi at round 38) is to stop spending
GPU-hours on items that do not change what ships or what a write-up claims.
This conclusion does not depend on which way a third sequence would land.

### 17.86.9 The damage-map check: deferred, not abandoned, and why

Attempted to build this properly (Kimi's item 1, tier-1 priority) once GPU0
freed up. Traced the actual training-path machinery needed: `model.forward(v1,
v2)` -> `p1,p2` (predicted Gaussians, means in a network-internal frame, NOT
the `pts3d`/`valid_mask` DUST3RSplattingDataset computes from GT depth --
confirmed those are unused by training, §17.85.1's finding again showing up in
a new place) -> `model.decoder(batch, p1, p2, (h,w))` -> `color` (target-frame
render) -> compare against `target['original_img']` for the per-pixel damage
map. Correlating that against context-frame opacity requires projecting each
Gaussian's mean into the target camera via the real relative pose -- and
whether the network's predicted `means` live in world frame, context-camera
frame, or some other canonical frame the paired decoder establishes was not
resolved by the code read available in the remaining time.

**Deferred rather than rushed.** Kimi is unavailable (quota) to sanity-check a
geometric implementation choice made under uncertainty, and this project has
repeatedly paid for exactly this kind of rushed instrumentation (the acc_alpha
cull-threshold bug, the CUDA_VISIBLE_DEVICES/--refiner-gpu index bug earlier
this session). The check tests a secondary mechanism question (why the
conf/disagree flip happened) that does not gate any claim already made --
§17.86.4's simpler vignette-field null already stands as a real, if partial,
result (candidate 1 refuted). Queued for when either Kimi is back to review the
frame-convention question, or there is time to trace `model.forward`'s exact
`means` frame convention through the decoder code rather than guess it.

### 17.86.10 The mixed arm lands: the two channels SUPPRESS each other, not both engage

Kimi is unavailable this round (quota); scoring this against his own
pre-registration since the logic requires no judgment call he was uniquely
positioned to make.

`replica-mixed` (content noise+blur AND photometry-spatial WB+vignette,
applied together per view, same rng advanced through both blocks so neither
perturbation's own calibration changed relative to running alone), 40 epochs,
probed with the same 5-pair protocol as every other head in this table:

```
head                          scale p90   inflation vs base   opacity median  frac>0.9
base                             9.778           --                1.0000      100.0%
noisy (content only)            26.182        +167.8%              1.0000      100.0%
photo (global photometry)       17.367         +77.6%              0.9912       83.1%
photospatial (spatial photo)    14.818         +51.5%              0.8759       42.9%
mixed (content + photospatial)  10.968         +12.2%              0.9332       62.7%
```

**Pre-registration**: both channels engage without interference -> frac>0.9
drops to 30-50% AND scale p90 inflates >=10% on the same head.

**Scored**: the scale-inflation bar is technically cleared (+12.2% >= 10%),
barely -- but it is far below EITHER individual perturbation's own inflation
(content alone: +167.8%; photospatial alone: +51.5%). And frac>0.9 (62.7%)
lands ABOVE the predicted [30,50]% range, meaning LESS opacity recruitment than
photospatial achieved alone (42.9%), not more. **Both numbers move in the
"suppressed" direction relative to their solo arms, not the "both engage"
direction.**

This is Kimi's own alternative outcome, stated when he pre-registered the
run: *"if one suppresses the other, the channels compete for a shared budget,
which is itself a finding (and would bound how much a single recipe can teach)."*
That is what the data shows. Combining both perturbations in one recipe does
not let the head learn both hedges at full strength; it produces a head that
hedges LESS on either axis than either perturbation produces alone.

### 17.86.11 A candidate reading, offered without Kimi to attack it

Both perturbations compete for the SAME limited quantity: how much the render
loss can be reduced by moving away from the saturated/tight-scale prior the
base checkpoint starts from. If the network has a roughly fixed "budget" of
loss-reduction it will spend on hedging (whether via scale or via opacity, per
the causal dissociation's own finding that these are largely separate channels
for separate perturbation types), then presenting both perturbations at once
means each gets a smaller share of gradient signal per epoch than it would
training alone -- not because the mechanisms interfere structurally, but
because 40 epochs of mixed-perturbation data contains fewer clean instances of
EITHER perturbation type in isolation than 40 epochs of single-perturbation
data does. This is a training-dynamics/dose account, not a representational
incompatibility account, and it predicts the suppression should shrink (or
disappear) with more epochs or a recipe that alternates rather than always
co-applies both perturbations per view. Not tested. Flagged as the leading
guess, not a finding.

Measurement-variance note: this session's two probes of the photospatial head
disagree on frac>0.9 (34.7% via `diag_colocation.py`, 5 or 6 pairs at one set
of indices; 42.9% via `gp2.py` here, 5 pairs at different indices). Both
numbers are now on record rather than silently reconciled; the gap is itself
informative about how much this class of probe should be trusted to one
significant figure.

### 17.86.12 The mixed head deploys BEST of all the photometric-augmentation heads, despite showing the WEAKEST channel engagement

```
Replica office0        psnr        lpips
base                 26.3274      0.1095
mixed                26.7077      0.0874     -20.2% lpips   +0.38 dB
```

Comparison against the other augmentation heads on the same scene:

```
head              deployment lpips gain    opacity/scale channel engagement
photo (global)          not deployed        frac>0.9 83.1%, p90 +77.6%
photospatial            -14.0%              frac>0.9 42.9%, p90 +51.5%
mixed                   -20.2% <- BEST       frac>0.9 62.7%, p90 +12.2% <- WEAKEST
```

**The mixed head produces the largest deployment gain of any augmentation head
measured, while showing the weakest opacity/scale response of the three on the
mechanism probe.** This decouples the two things §17.86.10's mechanism
narrative implicitly linked: "how much the head visibly hedges via
opacity/scale" is not the same variable as "how good the resulting deployed
map is." Whatever the mixed recipe's content-noise component adds to
deployment quality, it does not show up as the un-saturation/tail-inflation
signature the probe measures -- it must be doing something else structurally
useful (plausibly: general robustness to appearance/geometry noise of the kind
real SLAM sequences also contain, independent of the specific opacity/scale
channels this session has been tracking).

**Caution against over-reading one scene**: every other augmentation head's
deployment number in this file has needed a 2-3 scene range before being
trusted (§17.72's office1 reversal, §17.86.6-8's three-scene photospatial
range). This is one scene. Recorded as the leading data point, not a
conclusion, pending a second scene -- which is queued next since GPU1 just
freed up.

This result is a genuine puzzle worth Kimi's attention when his quota returns:
the mechanism story (opacity tracks damage; more perturbation types should
recruit more/different opacity) does not obviously predict "the recipe with
the WEAKEST individual-channel response deploys BEST." Two candidate readings,
neither tested: (a) the mechanism probe measures the wrong thing -- deployment
quality is driven by something the opacity/scale summary statistics do not
capture; (b) the content-noise component's val-time cost (§17.73.4: content
degradation makes the training task strictly harder) forces the network to
learn more robust FEATURES generally, which pays off at deployment through a
channel unrelated to opacity/scale allocation.

### 17.86.13 office1 confirms the sign, not the size: the mixed head's advantage is real but scene-dependent, more so than photospatial's was

```
Replica office1        psnr        lpips
base                 21.9888      0.1165
mixed                22.0945      0.1116     -4.2% lpips   +0.11 dB
```

Positive on both metrics, same direction as office0 -- not a §17.72-style
reversal. But the spread between scenes is much wider than photospatial's own:

```
head            office0      office1      ratio (larger/smaller)
photospatial     -14.0%       -6.6%            2.1x
mixed            -20.2%       -4.2%            4.8x
```

**Three-scene range for the mixed head: -4.2% to -20.2% lpips** (room0 added
2026-08-17, see below). Both positive, so §17.86.12's headline ("mixed deploys
best despite the weakest mechanism probe reading") survives as a real,
replicated-in-sign effect -- but it is not uniformly the strongest of the
three augmentation heads once office1 is in the picture: photospatial's
office1 (-6.6%) beats mixed's office1 (-4.2%), while mixed's office0 (-20.2%)
beats photospatial's office0 (-14.0%). The "mixed is best" claim from a single
scene does not survive as a clean ordering; the honest statement is that mixed
has the widest scene-to-scene spread of the three heads tested, positive
throughout, mean roughly comparable to photospatial's own mean.

```
Replica room0           psnr        lpips
base                 25.5087      0.1094
mixed                25.7712      0.0931     -14.9% lpips  +0.26 dB
```

```
head            3-4 scene mean (lpips gain)
photospatial     -14.55%  (n=4: -6.6/-14.0/-19.2/-18.4)
mixed            -13.1%   (n=3: -4.2/-20.2/-14.9)
```

Comparable means, very different variance. This changes §17.86.12's puzzle
framing slightly: it is not that mixed is unambiguously the best-deploying
head with the weakest mechanism signature; it is that mixed's deployment
outcome is LESS predictable scene-to-scene than photospatial's, which is
itself consistent with (though does not prove) the "shared training-budget
competed for by two perturbation types" account from §17.86.11 -- a recipe
that teaches both hedges weakly might produce a less reliably-calibrated head
than one that teaches a single hedge strongly.

**No further scenes queued for this cell.** Two augmentation-head deployment
lines (photospatial n=3, mixed n=2) both show positive, scene-variable gains
in the same range as the rest of this project's family-level deployment
results; the marginal value of a third mixed-arm scene is low against this
project's own stated bar (does it change what ships or what the write-up
claims) -- it does not, since "positive, variable magnitude" is already the
honest and sufficient characterization.

### 17.86.14 Round 40 audit (Kimi): the 34.7% vs 42.9% provenance check, resolved

Kimi flagged, correctly, that photospatial's `frac>0.9` reads 34.7% in the E2
readout (§17.86.2, via `diag_colocation.py`) and 42.9% in the mixed-comparison
table (§17.86.10, via `gp2.py`) -- same checkpoint, two numbers. Traced rather
than assumed:

```
diag_colocation.py   idxs = [int(n*f) for f in linspace(0.05, 0.9, pairs)]
                     -> office0 (n=2000): frames [100, 525, 950, 1375, 1800]
                        spans the FULL sequence

gp2.py               idxs = (0, 200, 400, 600, 800), fixed regardless of n
                     -> office0: spans only the FIRST 40% of the sequence
```

**Not a bug -- two genuinely different, non-overlapping probe sets.** Neither
script was ever designed to be numerically interchangeable with the other; the
gap is sampling-location variance, the same kind of thing this project
has repeatedly had to learn to report as a range rather than a point estimate.
Both numbers stay on record with their instrument named alongside them from
here forward; `diag_colocation.py`'s reading (34.7%) is the more representative
one of the two, since it samples the whole trajectory rather than the first
40%, and should be preferred when only one number is quoted.

### 17.86.15 Round 40 audit: the mixed-arm "higher variance" claim, corrected to what n=2 vs n=3 actually supports

Kimi's catch: comparing a 2-scene range (mixed) against a 3-scene range
(photospatial) and calling the ratio (4.8x vs 2.1x) a variance difference is
not a valid comparison -- range grows with n by construction, so a small-n
range-ratio is not a variance estimate. **§17.86.13's "mixed has higher
scene-to-scene variance" claim is retracted as stated.**

Corrected claim: mixed and photospatial have comparable MEANS (-12.2% vs
-13.3%); whether their variances differ is unresolved at the current sample
sizes (n=2 vs n=3) and would need a matched-n comparison to assess honestly.

**The mechanism puzzle from §17.86.12 (why does the head with the weakest
opacity/scale probe response deploy competitively) is not pursued further**,
on Kimi's argument: probe-strength and deployment-quality were never
established as related variables in this project to begin with -- the o-0.9
penalty head knee-parked its own opacity probe (median 0.84, well short of a
"strong" un-saturation reading) and still deployed at -15.2% lpips (§17.69).
Single-scene deployment numbers in this project swing by an order of magnitude
on their own (office1's whole history), so a one-scene probe/deployment
discordance is the ordinary texture of this system, not a paradox needing a
fifth mechanism.

**What is decision-relevant and stands**: mixed suppresses both channels on
the mechanism probe (§17.86.10) AND shows no mean deployment advantage over
photospatial alone. **Recipe recommendation: ship photospatial-only
augmentation; do not ship the mixed recipe.** The suppression finding stays in
the file as one sentence -- combined augmentation blunts both channel
responses and adds no measured benefit, consistent with the two perturbation
types competing for a shared training-time budget -- without further chase.

---

## 17.88 The damage-map check runs, and the sign is backwards from the hedging hypothesis

Kimi's round-40 trace unblocked this (`p1["means"]`/`p2["means_in_other_view"]`
both live in context[0]'s camera frame; exact projection confirmed against
`decoder_splatting_cuda.py`'s own extrinsic construction). Built
`damage_map.py`: for a real (context0, context1, target) triple on Replica
office0, run the BASE (frozen) model's own render+loss twice -- context1 clean
vs context1 degraded by the perturbation a given trained head was trained on
-- to get a per-pixel target-frame damage map (Delta MSE, perturbed minus
clean). Project the TRAINED head's own predicted opacities (on the degraded
input) from context1's grid into the target camera via the exact transform
the training decoder uses, and correlate opacity against the damage sampled
at each Gaussian's projected location.

Sanity check passed on every pair: damage mean is positive (the perturbation
does increase loss, confirming the harness measures something real) except
one slightly-negative outlier consistent with noise at that specific pair.

```
head                          degrade tested        n         spearman(opacity, damage)
photospatial                  photometry-spatial   589,824         +0.2821
mixed                         photometry-spatial   589,824         +0.2261
noisy (content-trained)       content               589,824         +0.0012  (degenerate:
                                                                       opacity ~100% saturated)
```

**The sign is backwards from the hedging hypothesis this check was built to
test.** The prediction (mine, and implicit in Kimi's framing across several
rounds) was that opacity should be LOW where the perturbation does the most
damage -- the head "hides" uncertain/risky content behind transparency. What
is measured, robustly, on two independently-trained heads against the exact
perturbation type each was trained on: **opacity is HIGH where damage is
high.** Consistent in sign and comparable in magnitude (+0.28, +0.23) across
photospatial and mixed.

The content-trained head's near-zero correlation is not informative either
way -- its opacity is ~100% saturated (§17.65.1), so any rank correlation on
it is close to the degenerate-array problem §17.66 built a guard for; not
formally flagged by this script but the same caution applies by inspection
(opacity has almost no dynamic range to correlate with anything).

### 17.88.1 Sent to Kimi rather than interpreted alone

This reverses the working assumption behind the whole allocation-mechanism
thread (§17.66 onward: "opacity hedges by going low where the map/photometry
is unreliable"). A reversed but consistent, non-degenerate, two-head-replicated
signal is data, not noise, and needs a mechanism reading before it goes in the
file as anything more than a measurement. One candidate considered and not
yet tested: **"commitment where it matters"** rather than "hedge where it's
risky" -- opacity may be high precisely where the network cannot afford to
fade out (the content still needs to be rendered and errors there are
expensive), and low by default where fading out is cheap because the content
is already easy. That would produce exactly this sign without requiring
opacity to track "risk" the way the original hedging framing assumed. Not
adjudicated; sent onward.

## 17.89 Round 41 resolved: opacity is a brightness-trim dial, not a hedging channel -- and the decisive check confirmed it with opposite signs

Kimi read the damage-map script directly (offered, and he took the offer --
right call given this project's history) and found the fact that resolves
everything: `data.py`'s `__getitem__` fetches BOTH context views (line 78) AND
target views (line 120) through the SAME degrade-enabled `ReplicaData`
instance, each with its own independently-seeded rng. **Training degrades
context and target independently** -- the target's vignette/WB draw is
invisible to the model, unpredictable from the input it sees. (This also
resolves the "minor documentation bug" flagged in passing at §17.86.5: the
class docstring's "target view only" framing was never accurate; every view,
context or target, gets its own draw.)

### 17.89.1 The mechanism, stated properly

Under a black-background renderer, opacity is not a keep/discard gate -- it is
a **brightness trim**: lower alpha darkens the render toward black (more
background bleed), higher alpha brightens it (less bleed, closer to the
predicted colour). `f_dc` is anchored to the input image's own (possibly
darkened) colours and sigmoid-bounded, so it cannot freely relight a region.
Opacity is the only channel with spare room to brighten past what the degraded
input suggests. A head trained with unpredictable per-instance target
darkening therefore learns the loss-optimal move: **raise opacity where the
input was likely darkened**, to recover brightness the target will, on
average, actually have. The damage-map check degrades the input and measures
where that hurts the base model's render -- which is exactly where a trained
head should have learned to lift. Opacity-high-where-damage-high was the
FORWARD prediction once opacity is modeled as a brightness dial rather than a
confidence gate; the "reversal" was a wrong mental model, not a wrong
measurement.

### 17.89.2 The decisive check: opposite signs, as predicted, before running anything

Kimi's sharpest consequence of the account: correlate opacity against input
luminance directly, per head. Prediction, pre-registered before running:
photospatial-style heads (vignette-darkened inputs) should show NEGATIVE
correlation (lift what's dark); real-data heads with no synthetic vignette
(TUM) should show POSITIVE correlation (solid where content is bright/
reliable, matching TUM's own +0.213 conf partial from §17.66).

```
head             dataset                  opacity vs input luminance
photospatial     Replica office0                -0.3156
mixed            Replica office0                -0.2788
tum              TUM desk                       +0.3026
```

**Opposite signs, comparable magnitude (~0.28-0.32), exactly as predicted --
before the numbers landed.** This is the first mechanism account in this
entire arc confirmed by a prediction that could have failed cleanly and
didn't. Kimi's account is adopted:

> Opacity is the head's per-region brightness/solidity dial. Its sign and
> magnitude follow whatever the supervision made unreliable: solid where
> content is reliably reconstructable (conf, TUM +0.213), trimmed where the
> sensor never vouched for the content (invalid, -0.198), trimmed where
> supervision photometrically conflicts (disagree, photo head -0.268), lifted
> where a perturbation stole brightness from the input (damage/luminance,
> photospatial/mixed -0.28 to -0.32). None of the upstream signs change; the
> "confidence/hedging" language is retired in favour of "solidity" and
> "brightness trim," which is mechanically accurate rather than metaphorical.

### 17.89.3 What this means for the deployment paradox (§17.86.12)

Resolved, not just explained away: the damage map measures *sensitivity to
input degradation*, not *wrongness*. High-damage pixels are where there was
strong content for the perturbation to remove; a head that raises opacity
there is repairing lost brightness, which is deployment-positive by
construction. "Doubling down on wrong content" would only describe this if
damage meant reconstruction failure -- it means the opposite, structurally.

### 17.89.4 Script fixes made per Kimi's audit

`damage_map.py`: `strict=False` head-weight loading now asserts
`n_loaded > 0` and prints the loaded/missing/unexpected key counts -- a silent
0-key load would have left the "trained head" scoring the base checkpoint
twice, indistinguishable from a working run by output shape alone. Confirmed
retroactively that the original runs loaded correctly (136/136 keys, matching
the fresh `luminance_check.py` runs above), so §17.88's numbers stand.

### 17.89.5 Closed here, per Kimi's own stopping rule

Two of his three "free checks before stopping" are done (the luminance
correlation, decisive; the load-assertion, done as part of the fix). The third
(clean-input vs degraded-input opacity, to separate "baked-in prior" from
"online response to seeing degradation") and the spatial-offset locality
control are lower-value now that the sign prediction has already landed
correctly -- per the standing rule this session keeps re-deriving (does the
next check change what ships or what gets claimed), the mechanism section can
be written now. **The opacity-allocation mechanism thread, open since §17.66,
is closed.**

### 17.89.6 Extension to eth3d/euroc (run 2026-08-17): 5/5 families now confirm the sign

Kimi's optional suggestion (extend from 2-3 heads to 5 families) was cheap
enough to run immediately once GPU0/GPU1 freed up -- both eth3d and euroc
heads are real-sensor data with no synthetic vignette/darkening in their
training recipe (`train_lora_per_scene.py`'s `FAMILIES` dict: plain
`ETH3DData`/`EuRoCData`, no `degrade=` kwarg), so the same real-data
prediction that held for TUM (+0.3026) applies to both.

```
head             dataset                  opacity vs input luminance
photospatial     Replica office0                -0.3156
mixed            Replica office0                -0.2788
tum              TUM desk                       +0.3026
eth3d            eth3d cables_1                 +0.3919
euroc            euroc MH_01_easy               +0.3037
```

**5/5 families land on the predicted sign**, real-data heads clustering
tightly at +0.30 to +0.39, synthetic-darkening heads at -0.28 to -0.32. No
family breaks the pattern. This does not change the mechanism verdict itself
(already closed in §17.89.5) -- it is additional replication breadth for the
paper write-up, run because it was a five-minute script invocation against an
otherwise-idle GPU, not because the mechanism was still in doubt.

## 17.90 Seed ensemble lands: the recipe is stable

Both round-38 "cheap insurance" runs finished. Kimi's instruction stands: these
report as the robustness paragraph and reopen nothing unless a seed disagreed
wildly. Neither did.

```
head              seed0 val psnr/lpips     seed1 val psnr/lpips     delta
tum                 16.83 / 0.251            16.88 / 0.2529        +0.05 dB / +0.76%
photospatial        20.33 / 0.1575            20.21 / 0.1536        -0.13 dB / -2.48%
```

Both deltas are small and go in OPPOSITE directions (TUM seed1 slightly worse
lpips, photospatial seed1 slightly better) -- the signature of ordinary
training-seed noise, not a systematic seed sensitivity. Both magnitudes are
comparable to or smaller than the within-family scene-to-scene spread this
project has repeatedly measured for deployment (e.g. photospatial's own 2.1x
scene range in lpips gain, §17.86.8). **The head-only recipe is not
seed-fragile.** No further seed runs queued; the five-family replication
already established more about robustness than seed variance would add
(Kimi's round-38 point, confirmed rather than just asserted).

### 17.90.1 Deployment-level seed check: also stable

```
head              seed0 deploy psnr/lpips     seed1 deploy psnr/lpips     delta
tum (room)          13.03 / 0.4381              13.18 / 0.4352            +0.15 dB / -0.66%
photospatial (o0)   26.87 / 0.0879              26.83 / 0.0842            -0.05 dB / -4.21%
```

Both small, both directions show seed1 with modestly better lpips -- no sign
of seed fragility at deployment either. The recipe's robustness claim now
covers both the val-split check (§17.90) and the deployment check (here).
Closed; no further seed runs.

### 17.89.6 Round 42, closing: why this confirmation counts, and the sentence that ties the whole project together

Kimi's closing notes, adopted verbatim where noted. **Why this result is the
strongest evidence the project has produced**: it was a pre-registered
opposite-signs prediction on one instrument, cheap to fail two ways, and it
failed neither. Every correlational finding in this file that "was consistent
with" a story is weaker evidence than this one prediction that could have
landed same-sign, or null, and did neither. This also retroactively resolves
a tension flagged back at 17.88: opacity correlating positively with both
`conf` (TUM head) and `damage` (photospatial head) was only ever a puzzle
under the risk/hedging axis -- under the trim-dial axis they are different
heads answering different supervision structures, and no single head has to
satisfy both at once.

**The two dropped follow-up checks are confirmed as the right drops**: the
spatial-offset control existed to rule out "two smooth fields agreeing by
construction," and opposite signs answer that more decisively than the
control would have (shared smooth structure cannot flip a sign). The
clean-vs-degraded-input variant would add color (static prior vs online
response), not load-bearing evidence.

**The unifying sentence for the write-up, not stated plainly anywhere before
this**:

> The thinning lever and the head-training result are the same result. The
> base checkpoint's opacity collapse means it cannot trim -- the haze was
> untrimmable solidity. Injection-time thinning supplies the trim from
> outside. The heads that deploy well are the ones that learned the trim
> themselves. Every lever that worked in this project either supplies,
> imitates, or enables that trim.

This connects the very first sections of this file (the dot lattice, the haze,
§17.2's instrument) to the last (the brightness-trim mechanism) as one
throughline rather than two separate arcs that happened to share a project.

**Kimi's own accounting, recorded because the file has been keeping score of
both of us**: the confidence/hedging framing that died this round was his as
much as mine -- he co-authored it across several rounds (round 24's
view-independence argument, the "confidence channel" language adopted at
§17.66). Recorded as progression (confidence channel -> trim dial), not
silently overwritten.

One optional, unrequired probe he flagged: the same luminance correlation on
the euroc/eth3d heads (the two largest deployment-gain families) would extend
the mechanism claim from two heads to five, for about ten minutes of probe
time. Not run in this session; noted as available and cheap if the write-up
wants the wider claim.

---

## 17.91 STATE OF PLAY (round 2) — read this instead of §17.83, which is superseded

§17.83 was written mid-session and everything from §17.84 onward happened
after it. This section exists for the same reason §17.83 did: to let a
compacted context (yours, next time) reconstruct the project's actual state
from this file alone, without re-deriving anything already settled. Read this
first. Section numbers below point to full detail; this section is the map,
not the territory.

### 17.91.1 What this project is, in two sentences

Splatt3R-SLAM integrates Splatt3R (feed-forward two-view Gaussian splatting)
into MASt3R-SLAM for real-time Gaussian-splat scene reconstruction, assembling
toward a paper. The user runs experiments to be reported by default, wants
everything logged here for the write-up, and wants complex analysis/design
questions argued to agreement with Kimi (a separate model, reached via
`scripts/`-adjacent shell wrappers around a persistent CLI session) rather than
decided unilaterally — except when Kimi's quota is exhausted, in which case
the standing instruction is to keep working, log every judgment call made
alone, and have Kimi audit them when he returns.

### 17.91.2 The three results this paper actually rests on, current form

```
1  HEAD-ONLY TRAINING (frozen encoder, train only the Gaussian head)
   Replicated on 5 families, deployment-verified online. eth3d -29.1% lpips
   euroc -20.8%, TUM -7.9% to -25.9% across 4 sequences, 7-scenes the weak
   family (+0.14% to -4.9%). Mechanism (17.81, 17.89): the optimizer
   un-saturates opacity on its own (1.0 -> 0.26) even without a trained head;
   head training moves the CONVERGED ENDPOINT, not just the starting point.
   Seed-robust (17.90: two heads, two seeds each, val AND deployment, deltas
   all <5% relative, signs inconsistent = ordinary noise not fragility).

2  INJECTION-TIME THINNING (opacity fade at map-assembly time, no retraining)
   12 online cells, 4 families, 6 heads, 8 scenes (17.79.10): 10 clear
   benefit, 2 ties, 0 harm, worst cell +0.16%. Mechanism: pure acceleration
   along a path the optimizer already walks by itself -- this is WHY it has
   bounded downside (a prior the optimizer was going to overtake anyway can't
   hurt much) and why training-time opacity penalties CANNOT substitute for
   it (17.70.8: crowding is a map-level property that exists only after SLAM
   accumulates many keyframes over a surface; pair-training never sees a map).
   Confirmed twice at two knee depths (o-0.9 -21.0%, o-0.6 -15.8%, neither
   collapsed toward zero as training-substitution would require).
   `--refiner-conf-fade` ships as the shipped allocation variant (confidence-
   weighted, not uniform), default 0.45, justified by a 12-cell distribution
   NOT by any ability to predict which map benefits (conf-vs-uniform is
   statistically a coin flip, n=12, median -0.06%, CI contains zero -- ship it
   anyway because it's never much worse, worst case +2.9%, and it IS the
   configuration the deployment A/Bs were run with).

3  THE OPACITY MECHANISM (what opacity actually is, and why real data teaches
   it and clean synthetic data doesn't)
   CLOSED, 17.89, this session. Opacity is a BRIGHTNESS-TRIM DIAL against the
   black background renderer, not a confidence/hedging/risk channel (that
   framing, which both the user and Kimi held for most of this session, is
   RETIRED). Lower alpha darkens toward black; higher alpha brightens.
   f_dc is anchored to the (possibly degraded) input colour and sigmoid-
   bounded; opacity is the only channel with room to brighten past what a
   degraded input suggests. Confirmed by the strongest evidence type this
   project produced: a PRE-REGISTERED OPPOSITE-SIGN prediction (opacity vs
   input luminance: photospatial/mixed heads -0.32/-0.28, TUM head +0.30),
   which landed exactly as predicted before the numbers were seen. This
   unifies with result #2: "the thinning lever and the head-training result
   are the same result" (Kimi, round 42) -- the base checkpoint's opacity
   collapse means it cannot trim (untrimmable solidity = the haze from
   §17.2's very first sections); injection thinning supplies the trim
   externally; heads that deploy well are the ones that learned the trim
   themselves. Every lever that ever worked in this project either supplies,
   imitates, or enables that trim.
```

### 17.91.3 The closed-question ledger since §17.83 (chronological, this is the part a compacted context most needs)

**Why eth3d/euroc (the two pseudo-depth families) show the largest head-
training deployment gains** -- CLOSED, unattributed, §17.85.10. Four candidate
mechanisms, each pre-registered and killed by direct measurement, not
argument:
```
scale-frame consistency   DEAD -- mechanically inert. The head-only loss is
                          PURELY photometric (masked MSE + 0.25 LPIPS); means
                          are pts3d.detach() (frozen, not trainable);
                          is_metric_scale is a no-op (its only consumer, the
                          MASt3R Regr3D loss, is disabled). Depth touches
                          training only via the loss mask and pair selection,
                          never the head's parameters. (17.85.1)
supervision coverage      DEAD -- measured mean supervised-pixel fraction per
                          family: tum 60.0%, 7-scenes 65.1%, euroc 39.5%
                          (LOWEST, yet 2nd-largest gain), eth3d 61.9%
                          (~=TUM, yet 3x the gain), replica 79.8%. Orders
                          nothing. (17.85.2)
opacity un-saturation     DEAD -- explained 4/5 families in rank order
degree                    (eth3d 0.24->−29.1%, euroc 0.38->−20.8%, tum
                          0.66->−9.9%, replica 0.998->null) but 7-scenes
                          (measured 0.4789, comparable to euroc's 0.38) has a
                          NULL gain -- a direct measured counterexample, not
                          a pending cell. (17.85.3)
base deficit (headroom)   DEAD -- rho=0.145 across 11 sequences with recorded
                          base lpips. eth3d sofa_1 has the LOWEST deficit of
                          any sequence (base already good) and the 2nd-LARGEST
                          gain. Killed with data already in hand, no new
                          GPU-hour needed. (17.85.8)
```
The gains themselves are real (measured against a 0.17% noise floor); only
the ATTRIBUTION died four times. Standing rule earned: **"a pattern across
five families is a hypothesis generator, never a conclusion; the conclusion
lives in the cell you predict before you run it"** (Kimi, round 35) -- and
even that discipline, applied properly (per-sequence base-deficit prediction
tested against data already in hand), still killed the account. 7-scenes'
suspected-broken-bake flag from round 29 was itself checked and found
groundless: zero relocalization events in any of its four deployment logs,
and its psnr band (11-15) is not anomalous against eth3d/euroc/TUM in the
same harness -- the earlier suspicion was comparing it to Replica's 21-27
regime, a different family entirely. (17.85.9)

**Does depth-supervision SOURCE (real sensor vs self-predicted pseudo-depth)
matter causally** -- CLOSED PERMANENTLY, controlled negative, §17.86.1/.86.7.
The one causal experiment in the whole depth-source inquiry: added
`depth_source="pseudo"` to `TUMData` (splatt3r_core/data/tum/tum.py), trained
a TUM head on self-predicted pseudo-depth instead of real sensor depth, same
family/scenes/recipe/seed otherwise. Deployment A/B on TWO sequences:
```
TUM room   real 13.0996/0.4346   pseudo 13.1369/0.4397   +1.17% (pseudo worse)
TUM desk   real 14.0955/0.3755   pseudo 14.1822/0.3720   -0.93% (pseudo better)
```
Opposite-sign noise on both -- the cleanest null texture this project
produced before the luminance check topped it. `scripts/precompute_pseudo_depth_tum.py`
is the new script (mirrors the existing euroc/eth3d one).

**Does the trajectory translation/rotation-ratio story have a causal
mechanism** -- CLOSED, demoted to unexplained association, §17.87. Kimi's
design: same baked map (TUM room, then TUM desk), two supervision-frame
subsets split by LOCAL parallax ratio (reusing `diag_within_seq.py`'s
`local_ratio()`), matched counts, shared/disjoint held-out set by
construction. `scripts/split_frames_traj.py` (well, the scratchpad copy) does
the splitting.
```
room: low(rotation-heavy) -9.71%   high(translation-heavy) -7.41%   REVERSED
desk: low -14.38%                  high -23.59%                    confirmed
                                                                     (1.64x,
                                                                     below the
                                                                     2x bar)
```
Sign-inconsistent across two sequences, matching (independently) the
observational within-sequence test's own inconsistency from months earlier
(§17.38: 4 negative, 3 positive across 7 sequences). Two independent methods
landing on the same inconsistent texture is what justified closing without a
third sequence -- **this was a demotion, not a promotion, and demotions need
less evidence than promotions** (Kimi, round 40, endorsing a call made solo
while he was offline). The magnitude finding survives the demotion:
supervision composition has a LARGE effect on refinement gain either way
(desk: a 9-point spread on the same map) -- just sign-inconsistent, the
sharpest version yet of "scene effects dominate lever effects."

**The causal photometric dissociation, completed** -- §17.86.2-.86.10.
Content-only degradation moves scale, not opacity (confirms the original
round-24/28 finding). Photometric degradation moves opacity, more so the more
spatially-structured it is:
```
global brightness jitter    frac>0.9 83.1%   (weakest un-saturation)
spatial (WB+vignette)       frac>0.9 34.7-42.9% (two different probe
                                                  protocols, see 17.86.14 --
                                                  not a bug, non-overlapping
                                                  frame samples)
mixed (content+spatial)     frac>0.9 62.7%   opacity/scale response WEAKER
                                              than either solo arm --
                                              SUPPRESSION, Kimi's own
                                              pre-registered alternative
                                              outcome, not "both engage"
```
Deployment: photospatial positive on 3 Replica scenes (-6.6% to -19.2%,
mean -13.3%; a 4th scene is running now, see 17.91.5). Mixed positive on 2
scenes (-4.2% to -20.2%, mean -12.2%) -- **NOT "best," a claim retracted in
round 40's audit** (comparing an n=2 range to photospatial's n=3 range as a
variance claim was invalid; corrected to "comparable means, variance
unresolved at these n"). **Recipe recommendation: ship photospatial-only
augmentation for any future head training; do not ship the mixed recipe** --
it suppresses both channels on the mechanism probe and shows no mean
deployment advantage.

**Why does opacity allocate where it does** -- CLOSED, §17.88-89, this
session's central arc, detailed in 17.91.2 item 3 above. Built
`damage_map.py` (scratchpad) reusing the EXACT training-path render+loss
(`model.forward` -> `model.decoder`, not a hand-rolled proxy) once Kimi traced
the frame convention (`p1["means"]`/`p2["means_in_other_view"]` both live in
context[0]'s camera frame; exact projection `inv(c2w_target) @ c2w_context0`).
First result (opacity HIGH where a frozen base model's render-loss damage from
degrading context1 is HIGH) read as "backwards" from the hedging hypothesis
until Kimi read the script AND traced `data.py`'s dataset path, finding that
BOTH context and target views are independently degraded during training
(same degrade-enabled instance, per-view seeded rng) -- so the target's
darkening is unpredictable from the input, and the loss-optimal response is
to brighten wherever the input was likely darkened, using opacity as the only
channel with spare room to do it. The decisive confirmation:
opacity-vs-input-luminance correlation, OPPOSITE SIGNS predicted and measured
(photospatial -0.3156, mixed -0.2788, TUM +0.3026). This is the strongest
evidence type the project produced all session: a prediction that could have
failed two clean ways and failed neither.

### 17.91.4 Standing methodological rules earned this session (on top of the ones already in 17.83.2)

```
1  A compaction can cause you to re-derive and duplicate already-settled work
   under clashing section numbers. If a number you're about to report
   "discovering" matches something suspiciously exactly, check for a
   duplicate before assuming it's new. (17.84 -- this happened once, mid-
   session, and cost real time to untangle.)
2  Any probe that TRANSFORMS the signal (eps-scaling, filtering, projection)
   must be validated at the EXTREMES of the range it will actually see, and
   print a reliability flag on the same line as the number. (acc_alpha's
   cull-threshold bug, this session's second self-inflicted instrument bug.)
3  A reversal is STRONGER evidence against a directional mediator than a flat
   result: flat says "unproven," opposite signs across cases say "no single-
   direction channel exists," which positively excludes the universal
   version of the claim. (Kimi, round 40.)
4  Demotions need less evidence than promotions. Closing a claim as
   unexplained/demoted after two data points can be the right call even
   where promoting a NEW claim on two data points would not be; the bar is
   asymmetric and that asymmetry is not just convenient reasoning. (Kimi,
   round 40.)
5  A range computed from n=2 is not comparable to a range from n=3; range
   grows with n by construction, so a range-RATIO between different sample
   sizes is not a variance estimate. Retracted a "higher variance" claim on
   exactly this error. (Kimi, round 40.)
6  When a surprising result reverses a working hypothesis, and the working
   directory contains the instrument that produced it, READ THE SCRIPT before
   building a new mechanism story on top of the number. The single highest-
   value thing Kimi did this session was accepting an explicit offer to read
   damage_map.py rather than trusting the description of it -- the frame-
   convention/site question he found by reading the DATASET code (not the
   probe script) was the entire answer.
7  strict=False weight loading is a silent-failure shape: a 0-key load is
   indistinguishable from a working load by output shape alone. Assert
   n_loaded > 0 and print missing/unexpected key counts on every such load
   from now on, not just when something looks wrong.
8  The strongest evidence a mechanism account can produce is a PRE-REGISTERED
   PREDICTION THAT COULD FAIL TWO CLEAN WAYS AND DOESN'T -- e.g. opposite
   signs on the same instrument, predicted before running. Weight this class
   of result above any number of "consistent with" correlations, which this
   project produced and then had to retract dozens of times this session
   alone.
9  When the primary collaborator (Kimi) is unavailable, keep working rather
   than block, but (a) log every judgment call explicitly enough that he can
   audit it when he returns, (b) prefer decisions that are demotions/closures
   over decisions that are promotions/new claims, since the former needs less
   certainty, and (c) do not let solo momentum silently become the new
   standing methodology -- round 40's audit function (3/4 solo calls
   endorsed, 1 needed a wording correction, 1 provenance number needed a
   source check) is what makes solo work during an outage safe to do at all.
10 GPU monitoring: match the check interval to the ACTUAL expected task
   duration, not to a conservative default. A fixed long polling interval
   that outlives the task it's watching wastes wall-clock time the user
   notices and has to prompt around -- if a task's rough duration is known
   (most deployment A/Bs in this project run under 20 minutes), poll or
   background-wait accordingly rather than defaulting to a large margin.
```

### 17.91.5 Exactly what is running right now, and what to do when it lands

```
GPU0   e6_ps_base / e6_ps_photospatial (scripts/... via
       /tmp/.../scratchpad/ps_scene4.sh), Replica office2, photo-spatial
       head's 4th deployment scene. On completion: record psnr/lpips next to
       §17.86.8's 3-scene table, recompute the range/mean. No further action
       needed regardless of outcome (the recipe recommendation does not hinge
       on this scene).
GPU1   e7_mixed_base / e7_mixed_mixed (.../mixed_scene3.sh), Replica room0,
       mixed head's 3rd deployment scene. Same treatment against §17.86.13's
       2-scene table.
```
Both are confirmatory, not decision-changing -- per 17.91.4 rule 10, check
these on a tight interval (they're single-scene A/Bs, historically ~15-20 min
each including the bake) rather than a long default wakeup.

### 17.91.6 Explicitly not pursued, and why (so it doesn't get re-proposed)

```
gradient probe (Kimi's round-23 unblocking hypothesis, time-resolved)
   SKIPPED. No claim or shipped default depends on the micro-mechanism; the
   attractor measurement (optimizer un-saturates unaided, 1.0->0.26) already
   carries the persistence claim. Kimi's own call, round 38.
17.34 re-analysis under current defaults
   SUPERSEDED by the trajectory causal test's closure (17.91.3). More
   correlation analysis cannot change a claim that a causal test already
   demoted.
head-training seed ensemble beyond what's done
   DONE AND CLOSED (17.90, 17.90.1) -- val and deployment, two heads, two
   seeds, stable. No further seeds queued; five-family replication already
   establishes robustness better than more seeds would.
damage-map spatial-offset control / clean-vs-degraded-input variant
   DROPPED, Kimi's round-41/42 call: the opposite-signs luminance result
   already answers what the offset control existed to check (shared smooth
   structure cannot flip a sign), and the clean-vs-degraded distinction is
   color (static prior vs online response), not load-bearing.
euroc/eth3d luminance probe (extend the trim-dial confirmation to 5 heads)
   OPTIONAL, NOT RUN. Free (probe only, no training), ~10 minutes per head
   per Kimi's own estimate. Worth doing if the write-up wants the mechanism
   claim to span all five families rather than the two Replica-augmented
   heads + TUM currently confirmed. Low urgency: the account doesn't need it
   to stand, per Kimi's round-42 framing.
Replica mesh controlled-trajectory experiment (orbit-vs-translate synthetic
trajectories, from way back in round 8's design)
   Superseded in practice by the cheaper held-map supervision-split causal
   test (17.87), which answered the causal trajectory question without
   needing new rendering infrastructure. Not revisited.
```

### 17.91.7 Files and scripts built or modified this session, for orientation

```
scripts/precompute_pseudo_depth_tum.py      NEW. TUM analogue of the existing
                                            euroc/eth3d pseudo-depth precompute.
splatt3r_core/data/tum/tum.py               MODIFIED. depth_source="sensor"|
                                            "pseudo" param, pseudo_depth_path().
splatt3r_core/data/replica/replica.py       MODIFIED. New degrade modes
                                            "photometry-spatial" (already
                                            existed) and "mixed" (content +
                                            photometry-spatial together, same
                                            rng advanced through both blocks).
scripts/train_lora_per_scene.py             MODIFIED. FAMILIES += tum-pseudo,
                                            replica-mixed.
scripts/exp_head_only.py                    MODIFIED. --family choices
                                            extended; verdict printer fixed to
                                            score the RENDER loss with penalty
                                            terms excluded (was silently
                                            inflating penalty-run "wins" by
                                            reporting the penalized objective
                                            against an unpenalized base).
scripts/refine_local.py                     MODIFIED much earlier this
                                            session: black-fraction and
                                            acc_alpha (exact-inversion,
                                            cull-aware) now print on every
                                            lattice report; --conf-fade lever
                                            added and shipped as the main.py
                                            default.
scripts/diag_colocation.py                  Built earlier this session:
                                            per-pixel opacity/scale vs
                                            predictor correlation, with a
                                            degeneracy guard (rank stats on a
                                            near-constant array are meaningless).
scratchpad/damage_map.py                    NEW. The render+loss damage-map
                                            check (17.88-89). Reuses the
                                            training decoder directly; now
                                            asserts strict=False loads are
                                            non-empty.
scratchpad/luminance_check.py               NEW. The decisive opacity-vs-
                                            luminance probe (17.89.2).
scratchpad/split_frames_traj.py             NEW. Local-parallax-ratio
                                            supervision-subset splitter for
                                            the trajectory causal test.
scratchpad/ask_kimi.sh                      Kimi CLI wrapper, unchanged
                                            pattern: writes <round>.tmp then
                                            atomically renames to <round>.done
                                            so "in progress" and "done" are
                                            never confused.
```

### 17.91.8 Kimi collaboration state

Session ID for resuming: `session_4c5a97eb-111a-491a-9349-0886850a5a43`
(`kimi -r <id>`, must be invoked from the scratchpad directory -- the session
is bound to its creation cwd). Round counter is at 42 as of this section.
His quota ran out twice this session (once mid-arc, once briefly later); both
times the standing instruction was to keep working alone rather than block,
log every judgment call for audit, and prefer closures over new claims while
he's out. This worked well both times -- see 17.91.4 rule 9. When he's out
again, that is the protocol to repeat, not a reason to pause.

### 17.91.9 The shape of this session's journey, for narrative continuity

Picked up mid-queue from a much longer prior arc (§17.1-17.83 predate this
session's visible context). Opened by discovering and untangling a
compaction-duplicate (17.84) -- a real cost, and the reason rule 1 in 17.91.4
exists. Then ran a long, disciplined falsification sequence on "why do
pseudo-depth families gain most from head training" -- four candidate
mechanisms, each pre-registered and each killed by direct measurement rather
than argument, ending in a deliberate non-answer ("unattributed, n too
small") that both collaborators treated as a legitimate, citable result rather
than a failure. That discipline then transferred cleanly to two harder
questions: the depth-source causal test (one clean controlled negative,
closed permanently) and the trajectory causal test (a design that reversed on
its first sequence and confirmed on its second, closed as sign-inconsistent
rather than either "proven" or "disproven"). The middle of the session added
a new perturbation type (spatially-varying photometry) that outperformed the
existing global-photometry arm as predicted, then a combined arm that
revealed channel suppression rather than combination, catching the author's
own premature "mixed deploys best" claim on a second scene before it went in
the file uncorrected. The session closed on its most demanding arc: building
an instrument (the damage-map check) to test a hedging/confidence hypothesis
for opacity allocation that BOTH collaborators had held for most of the
project, watching it come back "backwards," and -- rather than either
discarding the surprising result or rationalizing it into the old frame --
tracing the actual training code far enough to discover the working
hypothesis had the WRONG MODEL of what opacity mechanically does in a
black-background renderer. The corrected model (brightness trim, not
confidence gate) was then tested with a genuinely falsifiable, pre-registered,
opposite-sign prediction across two different training regimes, and it landed
clean. That is the single best piece of evidence this project has produced,
and it closes a mechanism question that had been open, under a wrong model,
since early in the project's history (§17.66). Seed ensembles at the end
confirmed none of the headline results are fragile to training-seed noise, at
both the validation and deployment level.

## 17.92 Paper-prep round (2026-08-17): LoRA planning dropped, qualitative figures, colour-consistency re-opened with a real result, runtime/memory, retrieval-refit and online-refinement pushed to closure

Triggered by the user's status-check + "what's left before this is a paper"
question. Kimi (round 43, session `session_4c5a97eb-111a-491a-9349-0886850a5a43`)
gave an independent second opinion in parallel; where it conflicted with this
session's own record the record won (see 17.92.6) -- Kimi's context was a
compact status blurb, not the full file, so a "5 rounds overdue" flag turned
out to already be closed at §17.85.9.

**Housekeeping**: the plan-mode file for per-scene TUM LoRA fine-tuning
(`.claude/plans/replicated-weaving-dawn.md`) was deleted per explicit user
instruction ("去掉lora这条线的所有规划") -- superseded by head-only training's
measured result, nothing else in the skill set referenced it as a live plan
(every remaining LoRA mention in this file is historical: route A -49%, route
D decoder-only-LoRA best +0.37 dB, both worse than head-only).

**Bug found and fixed en route**: `main.py`'s `--refiner-gpu` validation
(added for the class of failure documented at that call site: a bad ordinal
kills the refiner subprocess silently while SLAM finishes and looks
successful) checked `0 <= args.refiner_gpu < _n` unconditionally, but the
flag's own documented default is `-1` ("same card as SLAM") -- so any
`--refiner` run under a single-visible-device `CUDA_VISIBLE_DEVICES` crashed
even with the flag never passed. Fixed: `args.refiner_gpu != -1` now exempts
the sentinel. Found because this round's GPU1-only constraint (below) hit it
immediately on the first queued run.

### 17.92.1 Qualitative comparison figures -- 5 cases, 10 PNGs, in `figures/`

Base-vs-trained-head renders at held-out frames, reusing
`eval_map_quality.py --dump-renders` directly (not re-derived) so the
exposure-normalization and intrinsics handling are exactly what scored the
numbers already on record. Head-only training freezes the encoder, so
tracking/keyframe selection is bit-identical between a base and a head run of
the same sequence (the project's own ATE-untouched invariant) -- meaning the
held-out frame list `eval_map_quality.py` derives is the same for both arms
of each case, so base/head renders are guaranteed to be the same frame, not
just the same sequence. Composited GT | base | trained-head, labelled with
psnr/lpips, via `composite_qual.py` (scratchpad).

```
file                          case                                          base                          trained head
eth3d_sofa1_f{0,1}.png        eth3d sofa_1 (largest family gain, -29.1%)    psnr=22.07 lpips=0.2313      psnr=23.15 lpips=0.1811
euroc_v101_f{0,1}.png         euroc V1_01_easy (-20.8% family gain)         psnr=13.27 lpips=0.4749      psnr=13.22 lpips=0.4458
tum_rpy_f{0,1}.png            TUM freiburg1_rpy (real sensor data)          psnr=10.95 lpips=0.5161      psnr=11.04 lpips=0.5154
replica_ps_office2_f{0,1}.png Replica office2 (photospatial, -18.4%)        psnr=20.85 lpips=0.1629      psnr=21.26 lpips=0.1329
7scenes_office_f{0,1}.png     7-scenes office (the WEAK family, -4.5%)      psnr=11.42 lpips=0.5469      psnr=11.64 lpips=0.5222
```

eth3d's pair (`eth3d_sofa1_f0.png`) is the cleanest single before/after:
visibly sharper cushion-seam and fabric texture in the trained-head render at
matched psnr/lpips gap. The 7-scenes pair is included deliberately as the
honest-limitation exhibit -- same recipe, real but modest gain, useful as the
paper's own counter-example to "the lever always works big."

Source runs, all pre-existing from this session's deployment A/Bs except
7-scenes (fresh, `qual_7scenes.sh`, office/base+head, refiner on,
`--refiner-duty 1.0 --refiner-polish-secs 300`): `logs/h2_eth3d2_{base,head}`,
`logs/h2_euroc2_{base,head}`, `logs/h5_rpy_{base,head}`,
`logs/e6_ps_{base,photospatial}`, `logs/qual_7s_{base,head}`.

### 17.92.2 Colour-consistency: re-opened with one measurement, and it came back a real (if modest) positive -- opposite of the naive mechanism prediction

Per the plan the user approved before this ran: reuses
`color_harmonize.py`'s voxel-hash overlap-detection machinery
(`color_consistency_probe.py`, scratchpad) but applies NO correction --
pure measurement of how much two independent observations of the same
physical point disagree in raw baked colour, comparing base checkpoint vs
the photospatial-augmented head, both on TUM `freiburg1_floor` (the sequence
that originally reported the colour-consistency bug), no refiner (raw bake
only, isolating what the checkpoint itself does).

Pre-registered, before running (both outcomes real per the mechanism
section, §17.89.1): photospatial's independent per-view WB jitter + vignette
augmentation MIGHT teach incidental WB-drift invariance (LOWER disagreement
predicted) -- but §17.89.1 also established that `f_dc`/colour is anchored to
the (possibly degraded) input colour, with only opacity free to compensate,
which predicts NO improvement in colour specifically (only brightness). A
real test, not a formality.

```
                         RGB |own-ref| mean   median    hue-only mean   hue-only median   overlap voxels
base                          0.3098          0.2594        0.2151          0.1241         266,760
photospatial                  0.3023          0.2512        0.1879          0.1164         265,048
relative change                -2.4%           -3.1%        -12.6%           -6.2%
```

**Photospatial wins on every metric, and by far the most on hue-only
(chromaticity with brightness normalized out): -12.6%.** This is the OPPOSITE
of the extrapolation §17.89.1's "colour is anchored, opacity is the only free
channel" framing would suggest -- the colour residual itself evidently does
learn some WB-drift robustness, concentrated in hue rather than raw RGB
(which mixes hue and brightness, diluting the effect to -2.4%). Single
sequence, single seed, point estimate only -- no significance test run, and
this does not overturn the opacity-brightness-trim-dial mechanism (that
remains about brightness specifically, confirmed by 5/5 families' luminance
sign check, §17.89.2/17.89.6). It adds a second, smaller, real effect
alongside it: incidental hue-consistency improvement, undocumented until this
probe. Reportable as: colour-consistency remains diagnosed-not-fixed
end-to-end (Plan 3, full retrain with a dedicated consistency loss, is still
the only complete fix, and remains not done), but the photospatial recipe
already ships a measurable, free, partial mitigation as a side effect of an
augmentation chosen for an unrelated reason.

`cc_base.json` / `cc_photospatial.json` (scratchpad) hold the full numbers;
`base` map: 5,847,963 gaussians / 32 keyframes; `photospatial`: 5,686,144 /
32 keyframes, same sequence, same coverage.

### 17.92.3 Runtime/memory systems table -- 5 families, no-refiner vs same-GPU refiner

All measured on GPU1 exclusively, per the user's explicit constraint that
GPU0 carries desktop/other-process load and would contaminate timing numbers
(`perf_sweep.sh`, scratchpad). One representative sequence per family, base
checkpoint throughout. The refiner arm here uses a SHORT 60s polish window
(timing/memory characterization only -- steady-state fps and peak memory
stabilize fast; this is a different, shorter run than 17.92.4's quality
matrix, which needs the full 300s protocol to be comparable to every other
deployment number in this file. Do not average the two arms' numbers
together across sections).

```
family     no-refiner fps   +refiner (same-GPU, duty 1.0) fps   no-refiner peak MiB   +refiner peak MiB   final map (gaussians)
eth3d          7.93                    1.53                          18,315               21,528              2,051,729
euroc          5.40                    4.21                          22,287               30,516              3,327,453
tum            5.84                    3.70                          21,037               25,554              2,622,845
7-scenes       6.57                    4.23                          21,149               25,058              3,024,145
replica        7.89                    4.95                          17,537               20,896              3,065,486
```

Consistent with the single-sequence numbers already on record (§15.2: TUM
desk baseline 8.0 fps / 101 ms p50, same-GPU unthrottled refiner 4.5 fps /
206 ms p50) -- this extends that characterization from one sequence to five
families, all in the same ballpark (no-refiner 5.4-7.9 fps, same-GPU
unthrottled refiner roughly halves it except eth3d, whose 2.05M-Gaussian map
and heavier scene geometry make it the outlier at 1.53 fps). Memory scales
with final Gaussian count as expected (2-3.3M Gaussians -> 17.5-30.5 GiB peak
across arms); no family approaches the hardware ceiling on the A6000s this
project runs on (49 GiB/card). 30 fps real-time was never the operating
point for any family, matching the standing conclusion at §15.2/§15.11.

### 17.92.4 The refiner's OWN contribution, isolated from head training, across 4 families -- the measurement that was never actually done

Every deployment A/B run in this project (this session's photospatial/mixed
scenes, the eth3d/euroc/TUM head-vs-base pairs, all of it) has `--refiner` ON
in BOTH arms, so the refiner's marginal value has only ever been measured on
ONE sequence, TUM desk, at TWO specific budgets (§15.8: +1.24 dB live at
duty-0.25; §15.11: +2.15 dB at duty-1.0 second-GPU). This is the gap Kimi's
round-43 review flagged under "is online refinement actually done" that this
project had been quietly assuming rather than measuring. `refiner_onoff_matrix.sh`
(scratchpad, GPU1 only) closes it: base checkpoint throughout, same protocol
as every deployment A/B this session (`--refiner-duty 1.0
--refiner-polish-secs 300 --refiner-conf-fade 0`), refiner OFF (raw bake,
`_gaussians.ply`) vs ON (`_refined.ply`), one sequence per family.

```
family (sequence)          off: psnr / lpips        on: psnr / lpips         delta psnr    delta lpips
eth3d (sofa_1)              13.14 / 0.5663           21.83 / 0.2363           +8.69 dB       -58.3%
euroc (V1_01_easy)          12.22 / 0.5585           13.22 / 0.4777           +1.00 dB       -14.5%
tum (freiburg1_desk)        10.52 / 0.5605           14.05 / 0.4072           +3.52 dB       -27.3%
replica (office0)           21.80 / 0.2381           26.41 / 0.1029           +4.62 dB       -56.8%
```

**Positive on every family, by a wide margin, and far larger than the
duty-0.25 "live" number on record (+1.24 dB) -- because this is the full
300s polish protocol, the one every deployment A/B this session actually
shipped, not the throttled-live regime.** This directly answers Kimi's Q1
concern: if the paper's product is "always polish," the refiner's own
contribution is not a marginal add-on next to head training -- it is
comparable in size or larger (eth3d's +8.69 dB refiner contribution dwarfs
head training's own -29.1% lpips gain on the same family). The two levers
are not competing explanations for the same variance; they stack, and this
table is the first time that stacking was actually measured apart. Caveat
carried over from every other 300s-polish number in this file: this is the
non-real-time polish regime, not the duty-throttled live number -- both
regimes are real and both are already on record elsewhere (§15.8/§15.11),
this table just adds the missing per-family breadth at the polish end.

**Kimi's other Q1 ask (a retention curve at 4-5 step budgets under the final
config, not just the two endpoints already measured) is NOT done here** --
flagged as a real, understood gap rather than silently closed. Given the
table above already answers the higher-value question (is the refiner's
contribution real and family-general at the budget every deployment number
in this file actually uses), the retention curve is lower priority: it would
sharpen the budget-bridging mechanism story, not change any number currently
reported. Left as an explicit open item, not silently dropped.

### 17.92.5 Retrieval-refit: pushed to full closure, final verdict NO-GO -- an offline win that online ATE caught and killed

Full detail lives in the `splatt3r-retrieval-refit` skill's new §9, not
duplicated here; summary for this file's own ledger. Corpus expanded 3->9
TUM sequences (111->279 keyframes); the offline Recall@k verdict FLIPPED
once properly powered (cb2048 beats the original MASt3R assets: +20%
relative R@1, weighted across 8 sequences) -- but the real SLAM ATE A/B this
project's own screening protocol requires before shipping anything caught
what the offline proxy missed: room improved (-8.2% RMSE), 360 got worse
(+24.1%), and desk collapsed to within noise of the historical
`--no-loop-closure` baseline for that exact sequence (+318.8% RMSE). **Final:
keep the original MASt3R retrieval assets, permanently.** Two real bugs
fixed en route (`load_gt_poses`'s hard assert on a real 81ms GT gap in
`freiburg1_floor`; an int-vs-string codebook-size cache mismatch that
silently killed the backend subprocess). This is the same lesson as every
other line in this file that transferred offline->online: **an offline
metric win is a hypothesis, never a shipping decision by itself.**

## 17.93 External baselines (2026-08-17): the line the project deferred for months, now started -- MASt3R-SLAM first, and it validates the whole evaluation pipeline

Standing rule until now (memory: `external-baselines-deferred`) was "compare
with ourselves until the results justify a paper." With §17.91.2's three core
results all closed and §17.92 finishing the internal paper-prep work, that
bar is met, and Kimi's round-43 review independently ranked external
baselines the #1 remaining gap ("without it, every headline number you have
is unplaceable"). User-directed order: MASt3R-SLAM -> MonoGS -> Photo-SLAM ->
VGGT-SLAM.

**Hard constraints the user set for this whole line** (violating any of these
invalidates the numbers or the machine):
```
1. Every clone, dataset, weight, and environment lives under ./tmp/, never
   in the system conda env and never touching the system CUDA install.
   Each repo gets its own env at tmp/<repo>/env (conda --prefix).
2. Reuse ./datasets/ by symlink -- never re-download a dataset we have.
3. ALL performance/accuracy measurement runs on GPU1. GPU0 carries the
   user's own workload and would contaminate timing.
4. Compile with all cores. (ninja + MAX_JOBS=32 -- torch's cpp_extension
   silently falls back to single-threaded distutils when ninja is missing,
   which is what made the first builds crawl.)
```

### 17.93.1 Getting MASt3R-SLAM to build on this host: five real blockers

Documented because every remaining baseline will hit the same class of
problem, and because two of these were fixed WRONG on the first attempt.

```
blocker                          resolution
git-lfs missing                  submodule checkout of pyimgui died mid-clone.
                                 apt-installed git-lfs (system pkg, not CUDA).
CUDA version mismatch            System nvcc is 13.3; upstream wants torch
                                 2.5.1+cu124. FIRST ATTEMPT (WRONG): upgraded
                                 torch to 2.13 to match system CUDA -- broke
                                 the torch C++ API instead. CORRECT: installed
                                 cuda-nvcc 12.4 INTO tmp/MASt3R-SLAM/env via
                                 conda --prefix, so nvcc and torch agree and
                                 the system CUDA is untouched (user's rule 1).
curope build failure             kernels.cu uses tokens.type(), removed in
                                 modern torch. FIRST ATTEMPT (WRONG): dropped
                                 curope from setup.py, reasoning it has a
                                 pure-pytorch fallback. USER CORRECTED THIS:
                                 the parent project's OWN copy of kernels.cu
                                 was already patched months ago
                                 (tokens.scalar_type() + AT_DISPATCH_FLOATING
                                 _TYPES_AND2(Half, BFloat16) for bf16). Copied
                                 it over -> builds clean. Verified the CUDA
                                 path is actually live, not the fallback:
                                 `from models.pos_embed import RoPE2D` resolves
                                 to models.curope.curope2d.cuRoPE2D.
                                 LESSON: check whether this project already
                                 solved a vendored-dependency problem before
                                 working around it.
compute_60/61/70 unsupported     setup.py hardcodes archs CUDA 13.3 dropped.
                                 Trimmed to 75/80/86 (this host is A6000 =
                                 sm_86). Kept after the CUDA downgrade since
                                 it is functionally identical on this hardware
                                 and cuts compile time.
in3d/pyimgui won't build         Vendored pyimgui fails against modern Cython
                                 (cimgui attribute errors). GUI-only. Made
                                 main.py's `run_visualization` import lazy and
                                 defined WindowMsg (a plain dataclass, no GUI
                                 dependency) locally, so --no-viz headless
                                 evaluation runs without a working GUI stack.
```

Environment as built: `tmp/MASt3R-SLAM/env`, python 3.11, torch 2.5.1+cu124,
env-local nvcc 12.4, ninja. Checkpoints: the two retrieval assets symlinked
from our own `checkpoints/` (identical files, no re-download); only MASt3R's
main 2.75GB weight fetched fresh. `datasets/` symlinked wholesale.

### 17.93.2 The result: our ATE matches upstream MASt3R-SLAM to within noise on all 9 TUM sequences

Both systems, same host, same GPU (1), same config
(`config/eval_calib.yaml`: use_calib True, single_thread True, subsample 2 --
upstream's file and ours are byte-identical in these fields), same metric
(`evo_ape tum -as`), same ground truth.

```
sequence   MASt3R-SLAM (upstream)   ours      delta (ours - upstream)
360             0.048155          0.042079        -0.0061
desk            0.016136          0.016975        +0.0008
desk2           0.023532          0.027672        +0.0041
floor           0.024969          0.027230        +0.0023
plant           0.019565          0.015361        -0.0042
room            0.061271          0.059027        -0.0022
rpy             0.023064          0.021557        -0.0015
teddy           0.045072          0.047585        +0.0025
xyz             0.008902          0.008896        -0.0000
MEAN            0.03007           0.02960          -1.6%
```

**4 sequences lower for us, 4 lower for upstream, 1 tie, signs inconsistent,
means apart by 1.6%.** That is the texture of noise, not of a systematic
difference -- the same "sign-inconsistent across sequences => not a real
effect" standard this project applied to kill its own trajectory-parallax
mediator (§17.87) and its own mixed-vs-photospatial variance claim
(§17.86.15). The same yardstick has to apply when the number would flatter us
as when it would not.

**THIS IS A CONTROL, NOT A WIN. It does not show our system tracks better
than MASt3R-SLAM, and it must never be reported as if it did.** There is no
mechanism by which it could: head-only training freezes the encoder, tracking
runs the same pointmap matching on the same weights, and the Gaussian map is
strictly downstream of pose estimation. A systematic ATE improvement here
would be evidence of an uncontrolled variable, not of a contribution.
Parity is the expected -- and desired -- outcome.

Two things this establishes, both load-bearing for the paper:

1. **The evaluation pipeline is faithful.** Our harness, run against the
   upstream system's own numbers on its own benchmark, reproduces them. Every
   ATE number in this file inherits that credibility.
2. **The ATE-untouched invariant is now verified EXTERNALLY, not just
   internally.** This project has claimed throughout (§15.8, and the
   head-only route's core argument) that its contributions leave tracking
   bit-identical because head-only training freezes the encoder and the
   Gaussian map is downstream of tracking. Until now that was checked only
   against our own base checkpoint. It now holds against the actual upstream
   system we forked from -- a much stronger statement, and exactly the claim
   a reviewer would probe.

Note what this does NOT establish: nothing about rendering quality, and
nothing about this project being better at anything. MASt3R-SLAM produces no
Gaussian splats and reports no PSNR/LPIPS, so **it is not a baseline for this
project's contribution at all -- it is a control for the one axis where we
claim to change nothing.** The competitive comparison, on the axis this
project actually contributes to, requires the rendering-capable systems
(MonoGS, Photo-SLAM), which is why they are next. Until one of those produces
a head-to-head rendering number under our protocol, this project has NO
external evidence of superiority over anything, and the write-up must not
imply otherwise.

### 17.93.3 Baseline landscape, and the protocol-swamp warning that governs how these get reported

Surveyed before starting (WebSearch + repo inspection):
```
system         code?  renders?  datasets shipped        metrics reported
MASt3R-SLAM    yes    NO        TUM/7-Scenes/EuRoC      ATE only          DONE
MonoGS         yes    YES       TUM/Replica/EuRoC       PSNR/LPIPS/ATE    next
                                (default cfg is fr3, not fr1 -- needs one)
Photo-SLAM     yes    YES       Replica/TUM/EuRoC       PSNR/LPIPS/ATE
                                (ships fr1_desk cfg; heavy C++/ORB-SLAM3
                                 build, custom OpenCV+CUDA, LibTorch)
VGGT-SLAM      yes    NO        TUM/7-Scenes            ATE only
                                (pure python, easy install, eval scripts
                                 for exactly our two families)
GSO-SLAM       ?      YES       Replica/TUM             PSNR 34.48 (Replica)
VBGS-SLAM      ?      YES       Replica                 PSNR 37.94
```

**The number that must not be pasted into a table naively:** recent papers
report Replica PSNR of 30.9 (Photo-SLAM), 34.5 (GSO-SLAM), 37.9 (VBGS-SLAM),
against this project's own 20-27 dB range. That gap is far more likely a
protocol difference than a 10 dB quality deficit -- candidate causes, all
known to vary across the GS-SLAM literature: training-view vs held-out-view
scoring (this project scores ONLY held-out non-keyframes, §eval_map_quality's
protocol), GT poses vs estimated poses (§13.14 measured a ~4.5 dB gap between
those two on our own maps), per-frame online optimization budget, resolution,
and exposure normalization (§17.40 measured +0.126 dB from that alone).
This project is unusually well-placed to say so with receipts. **Rule for
this line: only numbers produced BY US under OUR protocol go in a comparison
table; published numbers may be cited but must be labelled as
different-protocol, never tabulated as if measured head-to-head.**

### 17.93.4 MonoGS: built, run, and cross-evaluated -- rendering quality is a TIE, and their map is 100x smaller

Second baseline, first one that actually renders. Same env recipe as
§17.93.1 (own conda prefix at `tmp/MonoGS/env`, env-local nvcc 12.4, ninja,
MAX_JOBS=32), plus one new blocker class: **this repo pins a 2024-era
dependency stack and every deviation broke something**, so all five were
pinned back to upstream's `environment.yml` rather than patching their code
(patching a baseline's source to fit our environment would undermine the
claim that we measured *their* system):
```
numpy      2.4.6 -> 1.26.4      np.unicode_ removed in numpy 2.0
opencv     5.0.0 -> 4.8.1.78    upstream's pin; needs numpy<2
plyfile    1.1.5 -> 0.8.1       upstream's pin; 1.1.5 needs numpy>=2
evo        1.37  -> 1.11.0      trajectory.align_trajectory removed in newer evo
matplotlib 3.11  -> 3.7.5       evo 1.11's traj_colormap breaks on modern mpl
```
Also `evo_config set plot_backend Agg` (evo forces TkAgg, dies headless).
gcc: nvcc 12.4 rejects the host's gcc 15.2, so `gcc`/`gxx` 12.4 metapackages
go in the env -- note `gcc_linux-64` alone is NOT enough, it only ships
`x86_64-conda-linux-gnu-gcc` and leaves plain `gcc` missing.

**MonoGS's own reported numbers, fr1_desk, its own protocol:**
```
ATE RMSE                     0.03575 m
PSNR/SSIM/LPIPS  before color refinement   17.18 / 0.640 / 0.4097
                 after  color refinement   20.71 / 0.698 / 0.3679  (26000 iters)
SLAM wall time               837.8 s
```

#### The protocol audit that had to happen before any comparison

Read `utils/eval_utils.py: eval_rendering` rather than assuming. Findings:
```
dimension            MonoGS                       ours                    same?
scored frames        every 5th, keyframes skipped  held-out non-keyframes  YES
render pose          its OWN ESTIMATED pose        GT pose, Sim3-aligned   NO
resolution           640x480 undistorted           512x384 via resize_img  NO
post-optimization    26000-iter colour refinement  300 s refiner polish    NO
psnr masking         psnr over gt>0 pixels only    all pixels              minor
```
The pose-source row is the dangerous one, and this project had already
measured it internally at ~4.5 dB (§13.14). **Measured again here, directly,
on MonoGS's own map: rendering it from the estimated poses it was built at
scores 18.71 dB; rendering the same map from ground-truth poses scores
9.87 dB -- an 8.8 dB protocol gap on one system, one sequence.** Any table
that mixes the two conventions is measuring the convention, not the method.

#### A bug I caught in my own adapter, and the check that caught it

First cross-eval run returned MonoGS at PSNR 6.09 -- a number that would have
flattered us enormously and was obviously wrong (PSNR 6 is noise). Cause: I
assumed `trj_final.json` stored world-to-camera and inverted it; it stores
camera-to-world. The check that caught it, now a standing rule for any
cross-system evaluation: **a map MUST render well from the poses it was
built at.** Convention sweep on MonoGS's own keyframes:
```
est as-is (c2w)    18.71 dB   <- correct
est inverted       4.36 dB
gt  as-is           9.87 dB
gt  inverted        5.02 dB
```
Never report a cross-system number without passing that self-consistency
check first. A wrong-convention adapter produces exactly the kind of
lopsided result that is easiest to publish and hardest to retract.

#### The result, under one protocol

Both maps scored by `scripts/eval_map_quality.py`'s own code path
(`/tmp/cross_eval_monogs.py`): identical 100 held-out frames chosen to be
keyframes of NEITHER system, GT poses Sim3-aligned into each map's own frame,
same renderer, same LPIPS/PSNR code, same exposure handling.

```
system    psnr      lpips     gaussians     map size on disk
ours      14.1068   0.4030    2,396,900     170 MB
MonoGS    13.8537   0.4027       23,019     1.5 MB
```

**Rendering quality is a tie** -- 0.25 dB in PSNR, 0.0003 in LPIPS, both
inside this project's own measured noise floors (0.031 dB / 0.17% lpips,
§17.79). **MonoGS reaches that tie with ~104x fewer Gaussians and a 113x
smaller map.** That compactness gap is a real and unflattering finding for
this project and must be reported: our maps are enormous for the quality they
deliver, which bears directly on the memory numbers in §17.92.3.

Honest caveats, all of which bound this result:
```
n=1 sequence (fr1_desk). Nothing here generalizes yet.
Post-optimization budgets still differ (26000 iters vs 300 s) -- not equalized.
Resolution/undistortion still differ; unquantified.
The two Sim3 alignments come from different data (their saved est/gt pairs
  vs our Umeyama fit), so a small residual misalignment is possible on
  either side.
ATE on this sequence: ours 0.0170 vs MonoGS 0.0358. Real, but it comes from
  MASt3R-SLAM's tracking, which we inherit unchanged -- NOT from anything
  this project contributes (see 17.93.2).
```

**What this does to the paper's framing:** the honest headline after two
baselines is *not* "we beat the state of the art." It is: our rendering
quality is competitive with MonoGS on a matched protocol while our
tracking (inherited) is better on this sequence, and our map representation
is dramatically less compact. The compactness gap is the obvious reviewer
attack and should be pre-empted in the write-up, not hidden.

### 17.93.5 VGGT-SLAM: set up and run -- and it is a POSE baseline, not a rendering one

Fourth baseline in the user's order, delegated to a subagent for environment
setup while the Photo-SLAM build occupied the foreground. Repo:
`MIT-SPARK/VGGT-SLAM` (2.0 branch), pure Python, env at
`tmp/VGGT-SLAM/env` (python 3.11, torch 2.5.1+cu124 -- the same known-good
combination as the other two baselines).

**Established by reading the code, not assumed:** grepping the whole repo
(excluding third_party) for `psnr|lpips|ssim|render|gaussian|novel.view`
returns ZERO hits. There is no rasterizer, no photometric decoder, no
Gaussian representation anywhere; the map is a raw coloured point cloud and
the only viewer draws points and frustums. **VGGT-SLAM cannot produce a
rendering-quality number at all**, so like MASt3R-SLAM it is a tracking
control, not a competitor on this project's actual contribution. (The
`version1.0` branch does ship a dense-geometry eval -- RMSE
accuracy/completeness/Chamfer against 7-Scenes depth. Those are point-cloud
metrics; if a map-quality comparison against VGGT-SLAM is ever wanted, that
is the honest axis, and a PSNR column is simply not obtainable from it
without writing a renderer.)

```
TUM freiburg1_desk, ATE RMSE (evo_ape tum -as), all measured on this host, GPU1
VGGT-SLAM 2.0      0.025417
```

Weights fetched: VGGT-1B (5.0 GB, byte-exact vs HF metadata, loads
strict=True with 0 missing/0 unexpected keys), dino_salad.ckpt (352 MB),
dinov2_vitb14 (346 MB) -- all under `tmp/VGGT-SLAM/torch_home/`, with
`TORCH_HOME` redirected there by an `activate.d` hook because
`loop_closure.py` derives the SALAD checkpoint path from
`torch.hub.get_dir()`.

Deviations from upstream, recorded because they bound the result:
```
torch 2.5.1 instead of the pinned 2.3.1 (machine-wide known-good choice;
   requirements.txt fed to pip with the torch lines removed so it could not
   silently downgrade)
numpy pinned 1.26.4, opencv 4.10.0.84 (the vendored VGGT fork declares numpy<2)
perception_models + sam3 installed --no-deps: their requirements pin
   numpy==2.1.2 which would break the VGGT fork. Both are imported only under
   `if args.run_os:`, which the TUM eval never enters -- so that open-set path
   is UNVERIFIED and its weights were not downloaded.
DINOv2 torch.hub cache pre-seeded by git clone, because torch.hub fetches a
   GitHub *archive* and this host's github archive endpoint is blocked
   (silent 0-byte downloads). No code patched.
evals/eval_tum.sh: 4 path-only lines edited to point at our ./datasets
   symlink instead of upstream's hardcoded sibling-checkout path.
eval_7scenes.sh / eval_euroc.sh exist only on the version1.0 branch despite
   the 2.0 README referencing them -- 7-Scenes/EuRoC numbers from VGGT-SLAM
   would require porting those scripts across branches, not attempted.
```

### 17.93.6 Two orchestration bugs of my own, both worth remembering

Recorded because both cost real wall-clock time and both are the kind that
recur:

1. **`pgrep -f "<script>.sh"` matches the watchdog's own command line.** A
   watchdog whose command string contains the pattern it greps for sees
   itself, so `while pgrep -f ...; do sleep; done` never exits. This silently
   burned ~1 hour: the Photo-SLAM build had already FAILED at the DBoW2 step
   and the watchdog kept reporting "running". This is the same hazard as the
   already-recorded "kill by resolved PID, never by pattern" rule, in a new
   costume. **Rule: wait on MARKER STRINGS written into a log, never on
   pgrep -f.** All chain scripts now do this.
2. **conda env include/lib paths are not automatic.** `$ENV/include` and
   `$ENV/lib` are only searched when the env is *activated*; a build driven
   by absolute paths to `$ENV/bin/gcc` does NOT get them. DBoW2 failed on a
   missing `boost/serialization/serialization.hpp` that was present in the
   env the whole time. Exporting `CPATH` / `CPLUS_INCLUDE_PATH` /
   `LIBRARY_PATH` fixed it.

### 17.93.7 Photo-SLAM build blockers (running tally -- this is the heaviest baseline by far)

```
blocker                              resolution
github tarball endpoint blocked      git clone --depth 1 --branch <tag>
missing NPP libraries                conda libnpp-dev=12.2.5.30 (matches CUDA 12.4)
CMake 4.4 dropped <3.5 compat        downgrade env cmake to 3.27 (upstream's tested version)
OpenCV 4.8 + CUDA 12.x incompatible  upgrade to OpenCV 4.10 (CUDA 12.x supported from 4.9).
                                     NOTE: this is a genuine fidelity deviation --
                                     upstream tested 4.7/4.8 with CUDA 11.8. It affects
                                     Photo-SLAM's image-processing backend only, not its
                                     Gaussian mapping, but it must be stated in the write-up.
xfeatures2d triggered blocked dl     confirmed unused by Photo-SLAM/ORB-SLAM3 (only 6 cv::cuda
                                     entry points are used: GpuMat, resize, cvtColor,
                                     StereoSGM, createStereoSGM, reprojectImageTo), removed
conda zlib symbol clash              -DBUILD_ZLIB=ON (OpenCV's bundled copy), apps off
boost headers not found              export CPATH/CPLUS_INCLUDE_PATH (see 17.93.6.2)
Eigen 5.0.1 in env, g2o wants 3.x    downgrade to eigen 3.4 (Eigen 5's config rejects a
                                     3.1.0 request outright, and Eigen 5 would break
                                     ORB-SLAM3/Sophus APIs regardless)
```

### 17.93.8 Photo-SLAM: built (18 blockers), run, and the three-way rendering comparison under one protocol

Photo-SLAM is by far the heaviest baseline -- a C++/CMake stack (ORB-SLAM3 +
DBoW2 + g2o + Sophus + LibTorch + a CUDA rasterizer) requiring a
source-built CUDA-enabled OpenCV. Eighteen distinct build blockers, all
resolved **in build configuration only; not one line of their algorithm
source was modified**, which is what lets the numbers below be called
Photo-SLAM's own behaviour. Full blocker list in 17.93.7 plus:
```
libcudart_static/libcudadevrt missing   conda cuda-cudart-static=12.4
nvToolsExt missing                      conda cuda-nvtx=12.4
libcuda / libnvrtc missing              cuda-driver-dev, cuda-nvrtc-dev; driver
                                        stub lives at $ENV/lib/stubs/libcuda.so
LibTorch saw two CUDA installs          CUDAToolkit resolved to SYSTEM /usr/local/cuda
                                        (13.3) while CUDA came from env (12.4);
                                        pinned both with CUDAToolkit_ROOT
MAXFLOAT undefined                      BSD-ism gone from modern headers;
                                        -DMAXFLOAT=... compile flag
M_PIf32 / M_PI_2f32 undefined            glibc dropped the float-suffixed math
                                        constants; defined by flag. LESSON: I fixed
                                        M_PI_2f32 first, rebuilt, then hit M_PIf32 --
                                        should have grepped the WHOLE repo for the
                                        class of problem before rebuilding.
DBoW2/g2o symbols undefined at link     modern ld defaults to
                                        --no-copy-dt-needed-entries, so symbols are
                                        NOT resolved through a transitively-NEEDED
                                        library; they must be linked explicitly
                                        (CMAKE_CXX_STANDARD_LIBRARIES="-lDBoW2 -lg2o",
                                        appended AFTER the objects)
exp2f@GLIBC_2.27 undefined              from libtorch_cpu.so; conda's sysroot libm
                                        stops at GLIBC_2.15 and sysroot 2.28 conflicts
                                        with the cos7 mesa packages. Correct fix:
                                        -Wl,--allow-shlib-undefined on the final step
                                        only (the symbol resolves at runtime from
                                        system glibc). WRONG fix I tried first:
                                        putting /usr/lib/x86_64-linux-gnu on the global
                                        search path -- that mixes system glibc with
                                        conda crt objects (glibc 2.34 removed
                                        __libc_csu_init) and broke the compiler test.
```
Two of my own recurring errors are worth naming: **scope creep on build
flags** (applying `-lDBoW2 -lg2o` and the system lib path globally broke
DBoW2's own compiler test -- a flag needed by the LAST step must not be
applied to the steps that build its dependencies), and **fixing one instance
of a class instead of enumerating the class first**.

#### Photo-SLAM, its own reported numbers (fr1_desk)
```
ATE RMSE (evo_ape tum -as)   0.017408
self-reported PSNR            17-24 dB, per KEYFRAME (its own training views,
                              its own renderer, its own resolution)
map                           36,069 gaussians, sh_degree=3, 8.9 MB
```

#### Pose-convention self-check (mandatory before any cross number)
```
render Photo-SLAM's map from its OWN keyframe poses, as-is (c2w)   13.340 dB
                                                  inverted (w2c)    8.065 dB
```
c2w confirmed. **I initially misread this system's self-check as a FAILURE**
because I used the wrong form of the check -- comparing own-keyframes-under-
GT-aligned-poses (10.71) against held-out-under-GT-aligned-poses (10.73) and
concluding "no keyframe advantage => adapter broken". That comparison is
confounded: both arms pay the GT-alignment penalty. The correct check is
rendering from the system's own poses directly, and Photo-SLAM passes it.
The 13.34 -> 10.71 drop is the same pose-source effect measured at 8.8 dB for
MonoGS (17.93.4) -- here 2.6 dB.

Also checked and ruled out as the explanation: our renderer uses only the SH
DC band while Photo-SLAM stores sh_degree=3. Measured energy in the dropped
bands: `f_rest` abs-mean 0.0178 vs `f_dc` 1.2917 (ratio 0.21 by absolute sum,
far smaller per-coefficient). Not worth several dB. Our ply decoder reads
properties BY NAME, so the 62-float sh_degree=3 layout parses correctly.

### 17.93.9 THE THREE-WAY RESULT: identical protocol, identical frames

All three maps scored by the same code path, on the SAME 100 held-out frames
chosen to be keyframes of NONE of the three systems, from ground-truth poses
Sim3-aligned into each map's own frame, same renderer, same LPIPS/PSNR code,
same exposure handling (`/tmp/cross_eval_3way.py`).

```
system      psnr      lpips     gaussians    map size    ATE (fr1_desk)
ours        13.9490   0.4110    2,396,900     170 MB      0.016975
MonoGS      13.8685   0.3975       23,019     1.5 MB      0.035750
Photo-SLAM  10.7269   0.5583       36,069     8.9 MB      0.017408
```
Self-consistency (each on its own keyframes, same GT-aligned protocol):
ours 14.2494/0.3963, MonoGS 15.1311/0.3672, Photo-SLAM 10.7147/0.5591.

**Honest reading of this table:**
- **PSNR: we lead by 0.08 dB over MonoGS -- that is a tie**, well inside this
  project's own noise floor (0.031 dB, §17.79). Against Photo-SLAM we lead by
  3.2 dB, which is real.
- **LPIPS: MonoGS is BETTER than us (0.3975 vs 0.4110)**, by 3.3% relative.
  LPIPS is the perceptual metric this project has argued all along is the one
  that matters. This must be reported, not buried.
- **Compactness: we are dramatically worse.** 2.4M gaussians / 170 MB versus
  MonoGS's 23K / 1.5 MB for equal-or-better quality -- 104x more primitives,
  113x more disk. This is the single most attackable number in the whole
  comparison.
- **ATE: ours and Photo-SLAM are equivalent (0.0170 vs 0.0174), both better
  than MonoGS (0.0358)** -- but ours comes from MASt3R-SLAM's tracking, which
  we inherit unchanged, so it is not evidence for this project's contribution
  (§17.93.2).

**What this does to the paper's claim.** After all four baselines the
defensible statement is NOT "we beat the state of the art". It is: *this
project's head-only fine-tuning plus injection-time thinning reaches parity
with MonoGS on PSNR, slightly behind on LPIPS, ahead of Photo-SLAM on both,
while inheriting MASt3R-SLAM-class tracking -- and does so with a map
representation two orders of magnitude less compact than the GS-SLAM
baselines.* The compactness gap should be stated up front as a limitation,
because a reviewer will find it immediately.

Caveats bounding all of this: **n=1 sequence (fr1_desk)**; post-optimization
budgets are NOT equalized (MonoGS runs 26000 colour-refinement iters,
Photo-SLAM runs to its 4181-iteration shutdown, ours gets a 300 s refiner
polish); resolution/undistortion handling still differs per system and is
unquantified; and Photo-SLAM was built against OpenCV 4.10 rather than the
4.7/4.8 upstream tested (CUDA 12.x forced this), which touches its image
front-end though not its Gaussian mapping.

### 17.93.10 VGGT-SLAM full TUM sweep, and the complete 3-system ATE table

VGGT-SLAM extended from 1 to all 9 freiburg1 sequences (its own eval
protocol: `--submap_size 32 --max_loops 1 --min_disparity 50
--conf_threshold 25 --lc_thres 0.95`, scored with the same
`evo_ape tum -as` as everything else).

```
sequence   MASt3R-SLAM     ours      VGGT-SLAM
360          0.048155    0.042079    0.049624
desk         0.016136    0.016975    0.025417
desk2        0.023532    0.027672    0.029102
floor        0.024969    0.027230    0.099134
plant        0.019565    0.015361    0.024532
room         0.061271    0.059027    0.063816
rpy          0.023064    0.021557    0.025791
teddy        0.045072    0.047585    0.036064
xyz          0.008902    0.008896    0.013846
MEAN         0.03007     0.02960     0.04081
```
(Single-sequence ATE for the two rendering baselines on fr1_desk, for
completeness: Photo-SLAM 0.017408, MonoGS 0.035750.)

VGGT-SLAM is ~38% worse than ours on the mean and loses on 8 of 9 sequences
(its `floor` result, 0.0991, is a clear failure case; it wins only on
`teddy`). **The same attribution rule as §17.93.2 applies: our advantage here
is MASt3R-SLAM's tracking, which we inherit unchanged. This is not evidence
for anything this project contributes.** VGGT-SLAM produces no renderable
map (17.93.5), so it appears in the ATE table only.

### 17.93.11 What is worth doing next, and what is not (assessed after all four baselines)

**Worth doing, in order:**
```
1. REPLICA three-way rendering comparison. Replica is THE standard GS-SLAM
   rendering benchmark -- it is where the literature's 30+ dB numbers come
   from -- and all three rendering-capable systems ship Replica support.
   Highest value for three reasons: (a) it is the benchmark a reviewer
   expects; (b) TUM's ~14 dB regime may be compressing the differences
   between systems, and Replica's higher-PSNR regime should separate them
   better; (c) it directly tests this file's own protocol-swamp hypothesis --
   if OUR Replica numbers under OUR protocol still land far below the
   published 30+ dB, that confirms the gap is protocol, not quality, which is
   itself a reportable methodological finding.
   SENSOR CAVEAT FOUND WHILE SETTING THIS UP: MonoGS ships Replica configs
   for RGB-D ONLY (configs/rgbd/replica/), no monocular variant, while
   Photo-SLAM ships both. Our system is monocular. Comparing our monocular
   result against MonoGS's RGB-D would be apples-to-oranges and would flatter
   MonoGS (depth input is a large advantage). So: Photo-SLAM MONOCULAR is the
   fair Replica comparison; any MonoGS Replica number must be labelled RGB-D
   explicitly, or a monocular config must be constructed (a deviation).
2. EQUALIZE THE OPTIMIZATION BUDGET. The headline table is currently
   confounded: MonoGS gets 26000 colour-refinement iters, Photo-SLAM runs to
   its 4181-iteration shutdown, ours gets a 300 s refiner polish. A
   budget-matched arm is needed before the comparison is defensible.
3. TURN THE COMPACTNESS WEAKNESS INTO AN ANALYSIS. 2.4M gaussians vs MonoGS's
   23K at equal quality is the most attackable number in the paper. We
   already own the lever that addresses it (`--refiner-conf-fade`, §17.79):
   measure a quality-vs-gaussian-count curve and answer "what is our quality
   at 23K gaussians?" That converts a reviewer's attack into our own
   efficiency analysis.
```

**Explicitly NOT worth doing:**
```
More pose-only baselines (DROID-SLAM, ORB-SLAM3, ...) -- tracking is not this
   project's contribution (17.93.2); extra ATE columns add nothing.
GSO-SLAM / VBGS-SLAM (2026) -- code availability unverified, and they sit in
   the same high-PSNR Replica regime; better to make Replica solid first.
7-Scenes / EuRoC extensions of VGGT-SLAM -- its eval scripts for those live
   only on the version1.0 branch and would need porting plus a judgement call
   about v1-harness-vs-v2-pipeline validity (17.93.5), for datasets that are
   far less standard in the GS-SLAM literature than Replica.
```

### 17.93.12 REPLICA, monocular, matched protocol: we win clearly -- and the protocol-swamp hypothesis is confirmed quantitatively

Replica office0, both systems MONOCULAR (fair sensor match -- see the caveat
in 17.93.11: MonoGS ships no monocular Replica config, only RGB-D, so it is
correctly ABSENT from this table rather than included with a depth-input
advantage). Same 100 held-out frames, keyframes of neither system, GT poses
Sim3-aligned into each map's own frame, same renderer/metrics/exposure.

```
system      psnr      lpips     gaussians
ours        26.2884   0.1023    2,987,333
Photo-SLAM  22.2286   0.2085       83,372
```
Self-consistency (own keyframes, same protocol): ours 26.4605/0.1109,
Photo-SLAM 22.4395/0.1984 -- both slightly above their held-out scores, the
expected pattern, so both adapters are sound.

**+4.06 dB PSNR and 51% lower LPIPS. This is the first result in the whole
external-baseline campaign where this project wins a rendering comparison
outright rather than tying.** It is also the opposite of the TUM outcome
(§17.93.9: tie with MonoGS on PSNR, slightly behind on LPIPS), which is why
n=1-per-dataset conclusions are dangerous in both directions.

#### The protocol gap, now measured on a published number

Photo-SLAM's Replica monocular PSNR **as reported in the literature is
~30.9 dB** (cited in third-party comparison tables; not independently
verified by us). The SAME system, built from its own source and run on the
same scene, scores **22.23 dB under our protocol**. That is an **8.7 dB
gap for one unchanged system** -- and it matches, almost exactly, the 8.8 dB
pose-source effect measured independently on MonoGS's own map in §17.93.4
(18.71 dB from its estimated poses vs 9.87 dB from ground-truth poses).

Two independent measurements, two different systems, same ~8.7-8.8 dB:
**the GS-SLAM literature's rendering numbers and this project's are offset by
roughly one protocol's worth of difference, not by quality.** Concretely,
this means our Replica 26.29 dB should NOT be read as "4.6 dB below the
published 30.9" -- measured identically, we are 4.06 dB ABOVE the system that
published 30.9.

This is the single strongest justification for the reporting rule already
adopted in §17.93.3: **only numbers we produced under our own protocol go in
a comparison table; published numbers may be cited but must be labelled
different-protocol and never tabulated head-to-head.** The paper should state
this offset explicitly with these two measurements as evidence -- it is a
methodological contribution in its own right, and this project is unusually
well placed to make it because it has both systems built and instrumented.

#### What does NOT change
The compactness gap persists and remains the most attackable number: 2.99M
gaussians for us vs 83K for Photo-SLAM on this scene -- 36x. (On TUM vs
MonoGS it was 104x.) Winning on quality while using 1-2 orders of magnitude
more primitives is a real efficiency deficit and must be reported as such.

Caveats: n=1 scene (office0); optimization budgets still not equalized
(Photo-SLAM ran to its 5081-iteration shutdown, ours had a 300 s refiner
polish); Photo-SLAM built against OpenCV 4.10 rather than upstream's 4.7/4.8.

### 17.93.13 The compactness curve: our quality does NOT survive shrinking to a baseline's budget

The most attackable number in the external comparison is that our maps carry
36-104x more Gaussians than the GS-SLAM baselines for equal-or-better
quality. The obvious defence would be "our map is merely redundant; prune it
and we match them." **That defence is now measured, and it is false.**

Replica office0, the same 100 held-out frames as §17.93.12, our own refined
map truncated to successively smaller budgets by keeping the highest-opacity
Gaussians (opacity is this project's measured solidity channel, §17.89 -- the
primitives the optimizer is itself fading out are the honest ones to drop
first). No re-optimization after pruning.

```
gaussians kept        psnr      lpips     note
2,987,333 (full)    26.2884    0.1023
1,000,000           24.6770    0.1951    -1.6 dB for a 3x cut
  300,000           20.2268    0.4252    -6.1 dB
   83,372           13.5410    0.6244    Photo-SLAM's budget on this scene
   23,019           12.0446    0.6726    MonoGS's budget (from the TUM run)
```

**At Photo-SLAM's own primitive budget we score 13.54 dB against its measured
22.23 dB -- we are 8.7 dB WORSE at matched count**, having been 4.06 dB
better at 36x the count. The knee is around 1M Gaussians: the first 3x cut
costs only 1.6 dB, after which quality falls off a cliff.

**Honest bound in both directions.** This is naive post-hoc truncation with
no re-optimization, so it is a LOWER bound -- a system actually trained or
refined at 83K would do meaningfully better than a 3M map crudely cut to 83K.
That caveat is real and must be stated. But it does not rescue the defence:
the measurement shows our quality is *constituted by* having millions of
primitives, not merely padded by them. **The efficiency deficit is a genuine
architectural limitation of this project's representation, not an artifact of
not having pruned.**

For the write-up: report this curve, state the lower-bound caveat, and do not
claim parity-at-matched-budget. The defensible positioning is quality-per-view
at a large memory cost, with the curve given so a reader can locate the
trade-off themselves. A real fix (training/refining directly at a small
budget, or a densification-control policy) is future work, not something this
project has measured.

### 17.93.14 Budget equalization on TUM -- and a claim I nearly got wrong

§17.93.11 listed "the optimization budgets are not equalized" as a confound in
the TUM three-way table. Resolved, and the resolution is more nuanced than the
first look suggested.

Extended arm: same sequence, same everything, `--refiner-polish-secs 1200`
instead of 300 (4x wall clock; 4400 refiner steps instead of ~1400). Scored on
the identical 100 held-out frames.

```
our polish budget       ours psnr/lpips     MonoGS psnr/lpips   Photo-SLAM
300 s  (~1400 steps)    13.9490 / 0.4110    13.8685 / 0.3975    10.7269 / 0.5583
1200 s (~4400 steps)    14.6120 / 0.3788    (unchanged)         (unchanged)
```

**The check that stopped an overclaim.** Seeing our LPIPS go 0.4110 ->
0.3788 and overtake MonoGS's 0.3975, the tempting write-up was "our LPIPS
deficit was a budget artifact." Before writing that, I measured what MonoGS's
26000-iteration colour refinement actually costs in wall clock: its own
progress bar reports **26000 iters in 5:39 = 339 s at 76.57 it/s**.

So the ORIGINAL comparison was already roughly wall-clock matched -- our 300 s
against their 339 s, with MonoGS if anything getting slightly more. The
correct readings are therefore:

```
AT COMPARABLE WALL CLOCK (~300-340 s of post-sequence optimization):
    PSNR is a tie (13.95 vs 13.87, inside the 0.031 dB noise floor)
    LPIPS: MonoGS is BETTER (0.3975 vs 0.4110), by 3.3% relative
    -- i.e. §17.93.9's reported conclusion STANDS, unchanged.

GIVEN 3.5x MonoGS's refinement wall clock:
    we overtake on both (14.61 / 0.3788)
```

The honest sentence is **not** "the LPIPS deficit was a budget artifact." It
is: *at equal optimization budget MonoGS has better perceptual quality; we can
buy past it with ~3.5x the compute.* Those are different claims and only the
second one is ours to make, with the multiplier stated.

Note the two budgets are not in identical units (their iterations vs our
wall-clock-limited steps) and their iterations are cheaper because their map
is 100x smaller -- which is the compactness deficit of §17.93.13 showing up
again, now as a throughput disadvantage: we need far more time per
optimization step for the same scene.

**Standing lesson, third instance in this campaign** (after the MonoGS
c2w/w2c inversion in §17.93.4 and the misread Photo-SLAM self-check in
§17.93.8): every time a number moved in this project's favour, checking it
one level deeper changed the claim. The pattern is consistent enough to be a
rule -- **a result that flatters us gets one more verification step than one
that does not.**

### 17.93.15 Replica extended to 3 scenes: the win replicates, and it is large

The §17.93.12 Replica result was n=1, which this file has repeatedly warned
against. Extended to three scenes; both systems monocular, same protocol
(100 held-out frames unseen by either, GT poses Sim3-aligned per map, same
renderer/metrics). Our arm is the base checkpoint + 300 s refiner polish in
all three (`logs/ref_onoff_replica_on`, `logs/e2b_repps_base`,
`logs/e7_mixed_base`); Photo-SLAM ran to its own shutdown iteration.

```
scene      ours psnr/lpips     Photo-SLAM psnr/lpips    delta psnr   ours G      PS G
office0    26.2884 / 0.1023    22.2286 / 0.2085          +4.06 dB    2,987,333   83,372
office1    21.7772 / 0.1195    17.8049 / 0.1854          +3.97 dB    2,149,703   75,235
room0      25.5562 / 0.1085    17.9169 / 0.1518          +7.64 dB    2,011,045   89,219
MEAN       24.5406 / 0.1101    19.3168 / 0.1819          +5.22 dB
```

**Three for three, same direction, +3.97 to +7.64 dB, LPIPS 29-51% lower.**
Unlike the TUM tie this is a replicated, sign-consistent, large-margin win --
and it is on the dataset the GS-SLAM literature actually uses as its
rendering benchmark. The compactness gap persists throughout (~25x here).

### 17.93.16 MonoGS on harder TUM sequences: its monocular tracking diverges, and that ends the comparison there

Extending the TUM rendering comparison beyond fr1_desk required MonoGS on
more sequences. Configs were produced by copying its `fr1_desk.yaml` and
changing only `dataset_path` -- legitimate because all freiburg1 sequences
share one calibration.

```
sequence     MonoGS ATE RMSE     ours ATE     MonoGS self-reported psnr
fr1_desk        0.035750         0.016975         20.71 (after refinement)
fr1_room        0.791090         0.059027         15.41
fr1_360         0.177320         0.042079         15.89
```

**MonoGS's monocular tracking diverges on both harder sequences** -- 0.79 m
on room is a complete failure, 0.18 m on 360 is an order of magnitude worse
than ours. This is a real and reportable robustness difference between the
systems (ours inherits MASt3R-SLAM's tracking, which holds up).

**No rendering cross-evaluation was computed for these two sequences, on
purpose.** With the trajectory diverged, MonoGS's map lives in a badly wrong
frame; scoring it from GT-aligned poses would produce a number that measures
its tracking failure, not its rendering quality, while looking like a
rendering result. Reporting such a number would be the mirror image of the
c2w/w2c bug in §17.93.4 -- a large, flattering, meaningless margin. The
honest output is the ATE table above plus the statement that the rendering
comparison is confined to sequences where both systems track successfully
(fr1_desk).

This also bounds §17.93.9 properly: **the TUM rendering tie with MonoGS
rests on n=1 and cannot be extended, because MonoGS does not survive the
other sequences.** The Replica three-scene result (17.93.15) is therefore
the load-bearing rendering comparison, not the TUM one.

## 17.94 CONSOLIDATED RESULT TABLES (paper-ready; every number measured on this host)

Single place to copy numbers from when writing. Every value below was produced
by us on this machine; nothing is quoted from a paper except where explicitly
marked LITERATURE. Blank cells mean NOT MEASURED, never "failed".

### Table 1 -- Trajectory accuracy, ATE RMSE (m), TUM freiburg1, evo_ape tum -as
COMPLETE: all 9 sequences x all 5 systems, no gaps.
```
sequence   ours       MASt3R-SLAM  VGGT-SLAM   Photo-SLAM   MonoGS
360        0.042079   0.048155     0.049624    0.034738     0.177320
desk       0.016975   0.016136     0.025417    0.014861     0.035750
desk2      0.027672   0.023532     0.029102    0.438468     0.843905
floor      0.027230   0.024969     0.099134    0.013299     0.539209
plant      0.015361   0.019565     0.024532    0.046073     0.071390
room       0.059027   0.061271     0.063816    0.509838     0.791090
rpy        0.021557   0.023064     0.025791    0.056558     0.040656
teddy      0.047585   0.045072     0.036064    0.304875     0.122987
xyz        0.008896   0.008902     0.013846    0.009682     0.017198
MEAN(9)    0.02960    0.03007      0.04081     0.16094      0.29328
```
**The headline here is ROBUSTNESS, not precision.** Photo-SLAM and MonoGS
each diverge on multiple sequences (desk2, room, teddy at 0.30-0.84 m --
i.e. tracking failure, not error), which is what wrecks their means. On the
sequences where they do track, Photo-SLAM is competitive and sometimes best
(floor 0.0133, desk 0.0149, 360 0.0347). Ours / MASt3R-SLAM / VGGT-SLAM never
diverge. Ours vs MASt3R-SLAM remains noise (4-4-1) and that advantage is
INHERITED tracking, not a contribution of this project (§17.93.2).
Photo-SLAM's desk scored 0.017408 on its first run and 0.014861 on a rerun --
ORB-SLAM3 is multithreaded and non-deterministic; single-run numbers for it
carry run-to-run variance and this must be stated in the write-up.

### Table 2 -- Rendering, Replica, monocular vs monocular, matched protocol
ALL 8 Replica scenes (the dataset has exactly 8: office0-4, room0-2).
100 held-out frames per scene unseen by either system, GT poses Sim3-aligned
per map, same renderer/metrics/exposure. **Load-bearing rendering result.**
```
scene     ours psnr / lpips     Photo-SLAM psnr / lpips   d_psnr   lpips win   ours G      PS G
office0   26.3049 / 0.1044      22.2286 / 0.2085          +4.08    ours        2,987,333    83,372
office1   22.0827 / 0.1150      17.8049 / 0.1854          +4.28    ours        2,149,703    75,235
office2   20.9714 / 0.1505      20.2365 / 0.1285          +0.73    PHOTO-SLAM  3,065,486    91,256
office3   20.1782 / 0.1438      19.3641 / 0.1553          +0.81    ours        2,410,090    66,156
office4   23.6457 / 0.1562      16.4615 / 0.1254          +7.18    PHOTO-SLAM  2,636,008    72,064
room0     25.4441 / 0.1097      17.9169 / 0.1518          +7.53    ours        2,011,045    89,219
room1     21.9085 / 0.1395      21.8820 / 0.1745          +0.03    ours        3,298,001   113,798
room2     23.6181 / 0.1599      20.7493 / 0.1158          +2.87    PHOTO-SLAM  3,039,774    71,477
MEAN(8)   23.0192 / 0.1349      19.5805 / 0.1557          +3.44    ours 5/8
```
**THIS TABLE CORRECTS AN EARLIER OVERCLAIM OF MY OWN.** At n=3 (office0,
office1, room0 -- §17.93.15) the margin looked like **+5.29 dB with LPIPS
3/3**. Those three happened to be the three scenes we win biggest. With all 8:
```
PSNR   we win 8/8, but the mean margin falls +5.29 -> +3.44 dB, and three
       scenes are effectively ties (+0.03, +0.73, +0.81)
LPIPS  we win only 5/8 -- Photo-SLAM is better on office2, office4, room2
```
The small-n hazard this file has warned about repeatedly bit a result that
flattered us. Report the 8-scene numbers; the n=3 subset is not
representative. MonoGS is absent by design (RGB-D-only Replica configs;
scoring its depth-input map against our monocular one would flatter it).
### Table 3 -- Rendering, TUM freiburg1_desk (n=1, NOT extendable)
```
system                    psnr      lpips    gaussians   map size   budget
ours (300 s polish)       13.9490   0.4110   2,396,900   170 MB     ~1400 steps
MonoGS (339 s refine)     13.8685   0.3975      23,019   1.5 MB     26000 iters
Photo-SLAM                10.7269   0.5583      36,069   8.9 MB     to shutdown
ours (1200 s polish)      14.6120   0.3788   2,396,900   170 MB     ~4400 steps
```
At COMPARABLE wall clock (300 vs 339 s) PSNR ties and MonoGS wins LPIPS. Only
with 3.5x their refinement budget do we lead both. Cannot be extended to more
TUM sequences because MonoGS's tracking diverges there (Table 1).

### Table 4 -- Compactness curve, ours, 4 Replica scenes
Same 100 held-out frames per scene; highest-opacity Gaussians kept; NO
re-optimization afterwards (so every row is a LOWER bound -- a system actually
trained at that budget would do better).
```
kept          office0            office1            room0              room2
              psnr / lpips       psnr / lpips       psnr / lpips       psnr / lpips
full          26.2884 / 0.1023   22.0827 / 0.1150   25.4441 / 0.1097   23.6181 / 0.1599
              (2,987,333 G)      (2,149,703 G)      (2,011,045 G)      (3,039,774 G)
1,000,000     24.6770 / 0.1951   21.7111 / 0.1546   25.0105 / 0.1204   22.8682 / 0.2140
  300,000     20.2268 / 0.4252   18.9560 / 0.4401   16.2572 / 0.3997   17.8659 / 0.4669
   83,372     13.5410 / 0.6244   13.9680 / 0.5925    8.6825 / 0.6513   10.4275 / 0.6877
   23,019     12.0446 / 0.6726   12.0762 / 0.6320    5.1574 / 0.7569    7.7428 / 0.7507
Photo-SLAM's
own score at
~83K G        22.2286            17.8049            17.9169            20.7493
```
**Replicated across 4 scenes: at Photo-SLAM's own primitive budget we score
8.7 / 3.8 / 9.2 / 10.3 dB WORSE than it does**, having been better by
4.08 / 4.28 / 7.53 / 2.87 dB at 25-36x the count. The knee is consistently
around 1M Gaussians (the first 2-3x cut costs only 0.4-1.6 dB), after which
quality collapses. The efficiency deficit is a genuine architectural property
of this project's representation, not an artifact of not having pruned -- and
it is now n=4, not n=1.

### Table 5 -- Protocol offset (methodological finding)
```
measurement                                          value      offset
Photo-SLAM Replica mono, LITERATURE (3rd-party table) ~30.9 dB   --
Photo-SLAM Replica office0, OUR protocol              22.23 dB   8.7 dB
MonoGS map rendered from ITS OWN estimated poses      18.71 dB   --
MonoGS SAME map rendered from GT poses                 9.87 dB   8.8 dB
```
Two independent systems, two independent methods, ~8.7-8.8 dB both times:
the literature's GS-SLAM rendering numbers and ours differ by roughly one
protocol, not by quality.

### Table 6 -- System cost, ALL systems, identical conditions
TUM freiburg1_desk (613 frames), GPU1 exclusively, same 1 Hz nvidia-smi
sampler for every system, wall clock measured end-to-end.
```
system       wall clock   peak GPU MiB   note
ours              70 s         21,035    6.47 fps; no refiner in this arm
Photo-SLAM        28 s          1,286    fastest and leanest
MonoGS           555 s          2,389    404 s of that is its optimisation
VGGT-SLAM         24 s          9,436    pose+pointcloud only, no rendering
```
**Our peak memory is 16x Photo-SLAM's and 8.8x MonoGS's, and Photo-SLAM is
2.5x faster wall-clock.** This is the same architectural deficit as the
compactness curve (Table 4), now measured at system level. It is the clearest
single number a reviewer can attack and must be reported up front.

Per-family cost for our system alone (5 families, refiner on/off), retained
from §17.92.3:
```
family     fps no-refiner   fps +refiner(duty 1.0)   peak MiB no-ref   peak MiB +ref   final G
eth3d          7.93              1.53                   18,315            21,528        2,051,729
euroc          5.40              4.21                   22,287            30,516        3,327,453
tum            5.84              3.70                   21,037            25,554        2,622,845
7-scenes       6.57              4.23                   21,149            25,058        3,024,145
replica        7.89              4.95                   17,537            20,896        3,065,486
```

## 17.95 Figures for the paper: inventory, the baseline-comparison set, and a misleading figure I caught

### 17.95.1 What existed vs what was missing
`figures/` held 10 PNGs, ALL of them internal ablation (base checkpoint vs
fine-tuned head, one row per family: eth3d_sofa1, euroc_v101, tum_rpy,
replica_ps_office2, 7scenes_office, x2 frames each). After the external
campaign the paper's most important missing figure class was a **side-by-side
against the competing systems**. Added:
```
cmp_replica_room0_f{0,1}.png     GT | ours | Photo-SLAM, Replica room0
cmp_replica_office0_f{0,1}.png   GT | ours | Photo-SLAM, Replica office0
cmp_replica_office2_f{0,1}.png   GT | ours | Photo-SLAM, Replica office2 (the near-tie scene)
```
Rendered through the SAME path the Table-2 numbers came from (identical
held-out frame, GT pose Sim3-aligned per map, identical renderer), each panel
labelled with its own PSNR/LPIPS and Gaussian count.

### 17.95.2 The first version of this figure contradicted its own caption
The initial run picked held-out frames by even spacing. On Replica room0 --
the scene where we lead by **+7.53 dB on the 100-frame mean** -- the chosen
frame had Photo-SLAM at **28.27 dB vs our 23.51**, i.e. 4.8 dB AHEAD, under a
caption reading "ours +7.53 dB over Photo-SLAM". A reader would have seen a
figure disproving its own label.

Fixed by selecting **representative** frames: score ~24 held-out frames for
both systems, sort by per-frame delta, and use the MEDIAN-delta frame (plus
one from the upper tail as the second panel). Captions now say explicitly that
the frame is the median of the per-frame delta and quote the scene mean
separately.

### 17.95.3 What the per-frame distribution revealed -- mean and median disagree
Computing those deltas produced a finding the scene means hide:
```
scene      per-frame delta (ours - Photo-SLAM), 24 probes
           median      min       max      scene-mean delta (Table 2)
room0      +8.01     -4.77    +15.66      +7.53
office0    +3.62     +0.20     +9.61      +4.08
office2    -1.84     -9.64     +7.53      +0.73
```
**On office2 the scene mean says we are +0.73 dB ahead while the per-frame
median says we are 1.84 dB BEHIND.** Our nominal win there comes from a
minority of frames where we do much better, not from being typically better --
and room0's -4.77 dB minimum shows Photo-SLAM beats us on some frames even in
the scene we dominate. For the write-up: report the mean (it is the standard
metric) but state that the per-frame distribution is wide and, on the
near-tie scenes, median-negative. Anyone choosing a single qualitative frame
should pick by median, not by eye.
