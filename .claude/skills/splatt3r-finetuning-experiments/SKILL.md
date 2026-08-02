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
incompleteness, not optimization failure.

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

- Generalization beyond TUM desk/room for the online numbers (the offline
  family-level head gains are separately measured, §8.1; causal/seam work
  is two sequences, one seed each).
- Real-time 30 fps: the deployable point is ~7.6 fps at 1.86M Gaussians on
  an A6000 pair.
- Viz display of the refined map: publication verified, consumer
  implemented but never drawn (headless box).
- Long-sequence behaviour of the dedup lifecycle (desk fires it once, at
  the tail, with no recovery budget).
