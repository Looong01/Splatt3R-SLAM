---
name: splatt3r-retrieval-refit
description: De-MASt3R-ification of the loop-closure retrieval subsystem — refitting the PCA whitening and rebuilding the ASMK codebook on dumped Splatt3R encoder features, per dataset family, with threshold recalibration and an accept/reject decision framework. Read before touching splatt3r_slam/retrieval_database.py, splatt3r_slam/retrieval_dump.py, main.py's --no-loop-closure / --dump-retrieval-features / --retriever-path flags, scripts/eval_retrieval_ab.sh, anything under checkpoints/retrieval/, or the vendored mast3r/retrieval/ + thirdparty/asmk/ fitting code. CLOSED (§9, 2026-08-17): offline Recall@k favored a refit codebook once the corpus was properly powered, but real SLAM ATE A/B caught a catastrophic regression on one of three sequences that the offline proxy missed — final verdict NO-GO, keep the original MASt3R assets permanently, stages 2-3 moot.
metadata:
  type: reference
---

# Splatt3R-SLAM: retrieval subsystem de-MASt3R-ification (whitening refit + codebook rebuild)

## 1. Background and verified facts

The loop-closure retrieval subsystem is the **only** part of the SLAM
pipeline still consuming MASt3R assets. Everything else (encoder, decoder,
Gaussian heads) is Splatt3R; this subsystem loads two files:

- `load_retriever()` (`splatt3r_slam/splatt3r_utils.py:92-112`) loads
  `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth`
  (default when `retriever_path=None`) and passes the **Splatt3R encoder**
  as backbone (`backbone=splatt3r_model.encoder`).
- The ASMK codebook path is **derived from the .pth filename**
  (`splatt3r_core/src/mast3r_src/mast3r/retrieval/processor.py:87-90`):
  split basename on `_`, drop the last token (`trainingfree`), append
  `_codebook.pkl` →
  `checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl`.
  Any replacement .pth must keep a matching sibling `_codebook.pkl` name.

### What is actually inside the .pth (dumped and verified 2026-07-27)

Top-level keys `['model', 'args']`. `model` contains exactly two tensors:

- `prewhiten.m` — (1, 1024), float64 (PCA mean)
- `prewhiten.p` — (1024, 1024), float64 (PCA projection)

`args` (verified via `torch.load`):

- `hdims=''` → empty projector list → `projector = nn.Identity()`
  (`retrieval/model.py:148-149`), so feature dim stays `backbone_dim=1024`
- `postwhiten=None` → `postwhiten = nn.Identity()` (`model.py:126`)
- `prewhiten=0`, `featweights='l2norm'` → attention is just the per-token
  L2 norm (`model.py:132-134`)
- `nfeat=300.0` (top-300 tokens kept by `how_select_local`, `model.py:88-104`)
- `nclusters='64k'` (asmk parses this to 65536,
  `thirdparty/asmk/asmk/codebook.py:15-17`)
- `whiten_nimages=30000`, `codebook_nimages=30000`
- `freeze_backbone=1`, `imsize=512`, `residual=0`

**Conclusion: the entire retrieval head is a single PCA whitening. There are
no trained layers.** Refitting it on new features is a minutes-scale
computation (one covariance eigensolve + one k-means), not a training run.

### The feature stream is already Splatt3R-native

`RetrievalDatabase` (`splatt3r_slam/retrieval_database.py:9-41`) subclasses
the MASt3R `Retriever` but skips image loading entirely: `prep_features()`
consumes `frame.feat` directly — the raw Splatt3R encoder feature
(`encoder._encode_image()`, same tensor the SLAM frontend uses), and applies
`prewhiten → projector(Identity) → attention(L2 norm) → postwhiten(Identity)
→ how_select_local(top-300)`. So the retrieval pipeline is
`[Splatt3R encoder features] → [MASt3R-fitted PCA whitening] → [MASt3R-fitted
64k codebook]`. Only the two bracketed MASt3R-fitted artifacts are foreign.

### Why a refit can only help in two distinct ways

The base Splatt3R checkpoint **freezes its encoder**
(`splatt3r_core/main.py:76-78` — `self.encoder.requires_grad_(False)`, only
the two `gaussian_dpt.dpt` heads are trainable). The encoder is structurally
the MASt3R ViT-L encoder, so base-checkpoint `frame.feat` should be
numerically close to what the MASt3R whitening was fit on. Two benefit
channels, keep them separate:

1. **Domain specialization** — refitting on in-domain features (the actual
   SLAM keyframe distribution of one dataset family) instead of the
   original generic 30k-image corpus. Available even for the base model.
2. **Drift compensation** — only exists once a LoRA adapter actually
   changes encoder weights (see `splatt3r-lora-finetuning` skill). Then the
   old whitening is measurably wrong for the new feature distribution.

A **zero-drift check** (stage 1) quantifies channel 2: refit whitening on
base features and numerically compare m/p against the existing tensors.

### Retrieval serves two features — the ablation flag only cuts one

- **Loop-closure candidates**: `run_backend()` (`main.py:145-154`) calls
  `retrieval_database.update(...)` on every keyframe and feeds returned
  candidates into the factor graph as loop edges.
- **Relocalization**: `relocalization()` (`main.py:59-102`,
  Mode.RELOC path) queries the same database with
  `add_after_query=False`, then re-adds on success.

`--no-loop-closure` only discards the candidate list at `main.py:152-153`
(`config.get("no_loop_closure", False)` — old yamls without the key fall
back safely). `update()` is still always called (comment at
`main.py:140-144`): the keyframe must still be *inserted* into the database
(`add_after_query=True`) because relocalization depends on it. So arm (b)
of the A/B experiment measures loop-closure contribution only; reloc is
live in both arms.

## 2. Goal and decision criteria

**Goal: best per-dataset accuracy, not novelty of assets.** Refitted assets
are a means, not a deliverable.

**Acceptance criteria** (all under `single_thread: True` deterministic
protocol, `config/eval_calib.yaml`):

- Loop-closure-sensitive sequences show a **significant** ATE improvement
  with the new assets, and
- remaining sequences are **not worse** (within noise).

**Keeping the existing MASt3R assets is a legitimate outcome.** If the
refit doesn't beat them, revert and document.

### Three-arm decision table

| Observation | Meaning | Action |
|---|---|---|
| (a) ≈ (b) | Retrieval contributes nothing on this family | Don't invest further in this family's assets |
| (a) ≫ (b) and new assets ≥ (a) | Retrieval matters; refit at least matches | Replace assets (per family) |
| (a) ≫ (b) but new assets < (a) | Retrieval matters; MASt3R assets still best | Keep MASt3R assets, document |

((a) = loop closure ON with existing assets, (b) = `--no-loop-closure`.)

## 3. Stage 0 — IMPLEMENTED: feature dump + A/B baseline

### `--no-loop-closure` (main.py:267-273)

`store_true`; injected into the global config before the backend process
spawns (`main.py:289`, `config["no_loop_closure"] = args.no_loop_closure`)
so `run_backend` sees it via `set_global_config(cfg)`.

### `--dump-retrieval-features DIR` (main.py:274-284)

Dumps each keyframe's `frame.feat` (the raw Splatt3R encoder feature fed to
the retrieval head) to `DIR/<seq_name>/feat_<kf_idx:06d>.npy` plus a
`metadata.jsonl` (one JSON record per keyframe: `kf_idx`, `frame_id`,
`timestamp`, `img_shape`, `feat_shape`). Implemented in
`splatt3r_slam/retrieval_dump.py` (`RetrievalFeatureDumper`):

- Mounted at the **two** `keyframes.append()` sites in `main.py`:
  INIT mode (`main.py:460-461`) and TRACKING new-keyframe
  (`main.py:532-533`).
- `frame.feat` is **provably non-None at both mount points** (docstring in
  `retrieval_dump.py:1-20`): INIT frames get it from
  `splatt3r_inference_mono()` (`splatt3r_utils.py:591-594`), TRACKING
  keyframes from `tracker.track() → splatt3r_match_asymmetric() →
  splatt3r_asymmetric_inference()` (`splatt3r_utils.py:667-669`), both
  before `keyframes.append(frame)` — and `SharedKeyframes.__setitem__`
  (`frame.py:358`) would raise otherwise. The `None` check in `dump()` is
  a defensive guard only.
- Stored array drops the leading batch dim: `_encode_image` returns
  (1, N, 1024), saved as (N, 1024); true shape recorded in metadata.
- Existing non-empty dump dir → warn + overwrite (matches the project's
  rerun semantics for stale trajectory files).

### `scripts/eval_retrieval_ab.sh [a|b|all]`

- Arm **a**: loop closure ON + dump to `logs/retrieval_features/<seq>/`
  (this simultaneously builds the fitting corpus for stage 1).
- Arm **b**: `--no-loop-closure` (reloc kept).
- Sequences (TUM): `freiburg1_room` and `freiburg1_360` (loop-closure
  sensitive), `freiburg1_desk` (control).
- Config `config/eval_calib.yaml` (`single_thread: True`, `use_calib:
  True`, `subsample: 2`) — matches `scripts/eval_tum.sh` protocol.
- Trajectories land in `logs/retrieval_ab_<arm>/<seq>.txt`; summary table
  via `evo_ape tum <gt> <est> -as` (same invocation as `eval_tum.sh`).

### Sequence-selection principle for other families

Pick sequences with **long trajectories and trajectory shapes that revisit**
previously-mapped areas (loop-shaped or back-and-forth paths) as the
sensitive set, plus one short/open trajectory as control. Without revisits,
loop closure is structurally irrelevant and the A/B comparison is noise.

## 4. Stage 1 — NOT implemented: per-family asset fitting (design)

Corpus: all `feat_*.npy` from arm-(a) dumps of every sequence in one
dataset family (expect ~30–60k keyframes × 300 tokens... in practice
subsample to ~300–600k descriptors per family).

### Whitening refit

- Use `pcawhitenlearn_shrinkage(X, s=1.0)`
  (`splatt3r_core/src/mast3r_src/mast3r/retrieval/model.py:21-38`) directly
  on the dumped patch features stacked as (N_desc, 1024). It returns
  `(m, P.T)` — `m` is (1, 1024), the returned matrix is already transposed
  to the (1024, 1024) layout `Whitener.p` expects.
- Note `retrieval/model.py` also has `RetrievalModel.reinitialize_whitening()`
  (`:165+`), but it drives an image dataset through the backbone — bypass
  it; we already have features, call the shrinkage function directly.

### Codebook rebuild

- Use the vendored asmk: `thirdparty/asmk/asmk/codebook.py` (`Codebook.train`
  → `index_factory.cluster`, faiss k-means; see usage pattern in
  `thirdparty/asmk/examples/demo_how.py:41-59`:
  `ASMKMethod.initialize_untrained(params)` →
  `asmk.train_codebook(vecs, cache_path=...)`). Keep the exact
  `asmk_params` dict from `processor.py:91-97` (binary kernel, build
  `multiple_assignment: 1`, query `multiple_assignment: 5`,
  `similarity: {threshold 0.0, alpha 3.0}`, `use_idf: False`) so query-time
  behavior is unchanged — only `train_codebook.codebook.size` varies.
- **Size must drop from 64k: sweep 8k / 16k / 32k.** Rationale: a
  per-family corpus of ~300–600k descriptors gives <10 samples per cluster
  at 64k — centroids are noise. The original 64k was fit on 30k *images*
  (millions of descriptors) from a broad corpus; a narrow per-family corpus
  supports far fewer clusters. asmk accepts `'8k'`-style strings
  (`codebook.py:15-17`).

### Save format (must match what `Retriever.__init__` loads)

```
whitening .pth: {'model': {'prewhiten.m': (1,1024) float64,
                           'prewhiten.p': (1024,1024) float64},
                 'args': <original args namespace, unchanged>}
```

Keep `args` identical (including `nclusters='64k'` is fine — but then the
codebook filename/size must still line up; simpler: set
`args.nclusters` to the swept value, since `processor.py:97` feeds it into
`asmk_params` and `Codebook.__init__` asserts centroid count == size).
Codebook: `<same_dir>/<derived_name>_codebook.pkl` per the
`processor.py:87-90` naming rule.

### Zero-drift empirical check (do this first, it's cheap)

Refit whitening on **base-checkpoint** features, then compare m/p
numerically against the existing tensors (relative Frobenius error, and
cosine of principal eigenvectors). Small error ⇒ encoder features match
MASt3R's distribution ⇒ any base-model gain is domain specialization only.
Large error ⇒ something already drifted; investigate before trusting either
asset set.

### Screening protocol

1. **Offline Recall@k first**: use ground-truth poses (available in the
   dump metadata timestamps + dataset GT) to define true neighbors by pose
   distance + view-angle thresholds; score retrieval with old vs new
   whitening × codebook size. Cheap, deterministic, no SLAM runs.
2. Only candidates that clearly win offline proceed to **full SLAM ATE**
   runs (stage-0 protocol, arm (a) with swapped assets).

Keep the shared (original MASt3R) assets untouched as the control group in
every comparison.

## 5. Stage 2 — NOT implemented: threshold recalibration + asset layout

- `config/base.yaml:55-61`: `retrieval.min_thresh: 5e-3` (and `k: 3`) were
  tuned for the MASt3R feature/score distribution. **Any new feature space
  (new whitening and/or codebook) shifts ASMK score distributions — the
  thresholds must be re-swept per family**, or the system-level result
  degrades even if Recall@k improved. Sweep on the sensitive sequences,
  verify no regression on controls.
- Asset organization (one directory per family, old assets left in place
  for rollback):

```
checkpoints/retrieval/<family>/
    whitening.pth      # refit prewhiten.m/p + original args
    codebook.pkl       # rebuilt, size from the 8k/16k/32k sweep
    thresholds.yaml    # recalibrated retrieval.k / min_thresh (+ reloc.*)
    meta.json          # corpus stats, codebook size, zero-drift numbers,
                       # Recall@k + ATE vs baseline, date
```

- Wire-up: `load_retriever(model, retriever_path=...)` already accepts a
  path; expose it as a CLI arg/config key rather than hardcoding. Keep the
  default pointing at the original MASt3R assets so absence of the new
  directory tree reproduces current behavior exactly.

## 6. Stage 3 — BLOCKED: refit on LoRA-adapted features

Blocked on the per-family LoRA adapters (see `splatt3r-lora-finetuning`
skill; TUM-family training running in background as of 2026-07-27). Once an
adapter exists:

1. Run stage-0 arm (a) with `--lora <adapter>` to dump features through the
   adapted encoder (LoRA is hot-swapped onto `model.encoder`,
   `splatt3r_utils.py:80-89`, and the retriever uses that same encoder —
   nothing extra to wire).
2. Repeat stage 1 + 2 on those features. This is where **both** benefit
   channels (drift compensation + domain specialization) apply, so the
   expected gain is largest here — but the same accept/reject criteria and
   the "keep MASt3R assets if not beaten" rule still hold.

## 7. Risk register

- **Domain-specialization gain may be ≈ 0.** The zero-drift check will
  likely show base features match MASt3R's distribution closely; the
  original whitening was fit on 30k images of broad data, which may
  already cover these families' feature statistics. Mitigation: that's a
  legitimate negative result — decision table, row 3.
- **Codebook sparsity.** Per-family corpus ≪ original corpus; too many
  clusters = noisy centroids, too few = coarse quantization. Hence the
  mandatory 8k/16k/32k sweep instead of assuming 64k.
- **Threshold miscalibration degrades the system even with better
  retrieval.** ASMK scores are distribution-dependent; skipping stage 2
  can turn a Recall@k win into an ATE loss (bad loop edges are worse than
  none — they corrupt the factor graph).
- **asmk pipeline fiddliness.** The vendored asmk + faiss GPU k-means path
  (`thirdparty/asmk/`, note the `FaissGpuL2Index` CPU fallback shim in
  `processor.py:16-30`) is old and brittle; fit codebook through the same
  `ASMKMethod` entry points the runtime uses (`initialize_untrained` →
  `train_codebook(vecs, cache_path=...)`), not a parallel implementation,
  so the saved pickle is guaranteed load-compatible.
- **Measurement noise.** All accept/reject comparisons must use
  `single_thread: True` (deterministic ordering) with the same config
  (`config/eval_calib.yaml`) and the same `evo_ape tum -as` invocation as
  stage 0; multithreaded runs make A/B differences uninterpretable.

## 8. Current status (2026-07-27)

**Implemented (stage 0):**

- `main.py:267-284` — `--no-loop-closure`, `--dump-retrieval-features`
  flags; config injection at `main.py:289`; dump mounts at
  `main.py:460-461` (INIT) and `main.py:532-533` (TRACKING keyframe).
- `splatt3r_slam/retrieval_dump.py` — `RetrievalFeatureDumper`.
- `scripts/eval_retrieval_ab.sh` — arms a/b/all + evo_ape summary.

**Baseline results (2026-07-27, TUM, eval_calib/single_thread, evo_ape -as):**

| sequence | a (LC on) | b (LC off) | ATE increase |
|---|---|---|---|
| fr1_room | 0.0590 | 0.0828 | +40% |
| fr1_360 | 0.0421 | 0.0770 | +83% |
| fr1_desk ("no-loop" control!) | 0.0170 | 0.0711 | +319% |

Verdict: (a) >> (b) on ALL sequences, including the supposedly
loop-free control — retrieval edges are not just dramatic loop
closures but also short/medium-baseline revisit constraints; without
them the pose graph degenerates to a chain (each kf constrained only
to idx-1) and drift accumulates fast. The retrieval subsystem is
highly valuable everywhere, and `--no-loop-closure` is an ablation
tool only, never an operating mode. NOTE (correction): this baseline
only proves "retrieval vs no retrieval" — it does NOT by itself
justify refitting the assets; that decision needs (a) vs (c), which
the Recall experiment below answers (NO-GO on base features).
Dumped corpus: 51/46/14 keyframes (room/360/desk), 334MB, under
`logs/retrieval_features/` — NOTE: 14 kfs for desk is far too few for
codebook fitting; collect more sequences per family in stage 1.
Also fixed during the runs: `eval_retrieval_ab.sh` rmse extraction
(evo indents the `rmse` line, `^rmse` never matched), and a main-loop
backend-liveness watchdog in `main.py` (backend death used to hang the
main process forever, leaving GPU-memory-holding orphans).

**Recall@k offline verdict (2026-07-27, `scripts/eval_retrieval_recall.py`,
`logs/retrieval_recall/results.json`):** fitted new whitening on all 85,248
dumped base features + new faiss-CPU codebooks (2048/8192) and compared
against the MASt3R assets, plus a codebook-free global-spoc ablation that
isolates whitening quality. Result: **refit whitening on base features is
NOT better (global-spoc R@1 slightly worse on all 3 sequences), and the
codebook comparison is inconclusive (sample-starved: 33k descriptors vs
>=319k recommended for 8192 clusters). Stage 1 on base features is a NO-GO
-- keep the MASt3R assets.** This confirms the encoder-frozen argument:
the existing whitening was fit on 30k images of the SAME feature
distribution; 111 keyframes of domain data cannot beat it. The only
remaining motivation for refitting is LoRA drift (stage 3, blocked on
adapters). Side observation: codebook-free global-spoc brute force beat
every ASMK config on fr1_360 (R@1 0.63 vs 0.50), so the ASMK quantization
stage itself may be the retrieval bottleneck -- worth revisiting if
retrieval quality ever becomes the priority.

**Superseded by §9 below (2026-08-17): corpus expanded, offline verdict
flipped, then online SLAM ATE caught what offline missed. Final: NO-GO,
keep MASt3R assets, permanently — see §9 for why this closes stage 3 too
(the LoRA route stage 3 was blocked on no longer exists; see §9.4).**

## 9. Closed (2026-08-17): corpus expanded, offline flipped, online caught a regression offline missed -- final verdict NO-GO

Triggered by a user request to push this line to full closure. The
33k-descriptor corpus above was always flagged as sample-starved for the
codebook comparison specifically (whitening-only was conclusive already);
this section removes that caveat with real data, then adds the online check
the original screening protocol (§4) always called for but this line never
actually ran.

### 9.1 Corpus expansion: 3 -> 9 TUM sequences, 33k -> 214k/84k descriptors

`--dump-retrieval-features` re-run on all 6 remaining freiburg1 sequences
(`desk2`, `floor`, `plant`, `rpy`, `teddy`, `xyz`) alongside the original 3
(`room`, `360`, `desk`). 279 keyframes total (up from 111): 214,272 raw
per-keyframe descriptors (768 tokens/kf, used for whitening) / ~84k top-300
attention-selected descriptors (used for codebook training) -- still short
of the >=319k "recommended" full-power threshold for 8192 clusters, but a
2.5x improvement, and `eval_retrieval_recall.py` (`SEQUENCES` extended,
`NEW_CB_SIZES` extended to `[2048, 8192, 16384, 32768]`) now flags the
under-powered sizes explicitly at cluster time (asmk's own "please provide
at least N training points" warning fired exactly as predicted for
16384/32768 -- those two sizes are directional only, not trusted).

Real bug hit and fixed en route: `load_gt_poses`'s hard `assert` on
GT-timestamp association tolerance (20 ms) crashed the whole 9-sequence run
on one `freiburg1_floor` frame with an 81 ms real gap in that sequence's GT
log. Changed to exclude the offending keyframe (with a printed warning)
rather than abort the entire evaluation over one sequence's data gap.

### 9.2 Offline Recall@k, re-run on the expanded corpus: verdict FLIPS from the 33k-descriptor test

Weighted by valid-query count per sequence (`teddy`'s n_valid=2 is near-
degenerate and excluded from the weighted read, though its raw numbers are
in `logs/retrieval_recall/results.json`):

```
config                        weighted R@1   weighted R@5
a_old_white_old_cb64k (MASt3R)   0.314          0.762
c_new_white_new_cb2048           0.376          0.795
c_new_white_new_cb8192           0.338          0.786
```

**cb2048 beats the original MASt3R assets by +20% relative R@1, +4%
relative R@5, weighted across 8 sequences (teddy excluded).** This reverses
the original 33k-descriptor verdict ("refit whitening on base features is
NOT better") specifically because that test was underpowered for the
codebook comparison, exactly as flagged at the time. Per this skill's own
screening protocol (§4): an offline win proceeds to full SLAM ATE before
shipping -- never ships on Recall@k alone.

### 9.3 Real SLAM ATE A/B: the online check catches a regression the offline proxy missed

Assets saved in deployable form (`checkpoints/retrieval/tum/tum_refit_{trainingfree.pth,codebook.pkl}`,
`scratchpad/save_retrieval_assets.py`) and wired through a new `--retriever-path`
CLI flag (`main.py`, `load_retriever(model, retriever_path=cfg.get("retriever_path"))`
in `run_backend`, propagated through the global config dict the same way
`no_loop_closure` already was). Real bug hit and fixed here too: the codebook
cache was trained with `size=2048` (a plain int, `eval_retrieval_recall.py`'s
`NEW_CB_SIZES`) but the saved deploy `args.nclusters` was written as the
string `'2k'` -- numerically identical after `Codebook`'s own parsing, but
`asmk_method.train_codebook`'s cache-validity check does exact dict equality
on the PRE-parse value (`cdb.params == step_params['codebook']`), so `'2k'
!= 2048` failed the check and silently killed the backend subprocess (visible
only as `main.py`: "backend process died unexpectedly" at the driving-script
level; the real `AssertionError` was buried in the per-sequence log). Fixed
by keeping `args.nclusters` as the same int type the codebook was actually
trained with. Confirmed with a 25s smoke run before committing to the full
A/B.

```
sequence   old ATE RMSE   new (cb2048) ATE RMSE   change
room        0.0590          0.0542                 -8.2%  (better)
360         0.0421          0.0522                +24.1%  (worse)
desk        0.0170          0.0711                +318.8% (catastrophic)
```

**desk's new-assets RMSE (0.0711) matches almost exactly the historical
`--no-loop-closure` baseline for the same sequence (0.0711, §8's stage-0
table) -- the refit assets are producing loop-closure edges on desk that are
functionally as useless as having no loop closure at all**, despite desk
contributing keyframes to the SAME fitting corpus as every other sequence.
One win (room), one loss (360), one collapse (desk): not a close call, and
not explained by corpus coverage (desk was IN the fitting corpus, at 14
keyframes -- the smallest single contribution, but that's a fitting-time
property, irrelevant to a shared cross-sequence asset's later use at
inference time on desk specifically).

### 9.4 Final verdict: NO-GO, permanently -- keep the MASt3R retrieval assets

Per this skill's own decision table (§2): "Keeping the existing MASt3R
assets is a legitimate outcome. If the refit doesn't beat them, revert and
document." One catastrophic regression out of three sequences fails that
bar even though the offline proxy and one of three online sequences favored
the refit. This is the same lesson this project's OTHER line
(splatt3r-finetuning-experiments) has hit repeatedly and independently:
**an offline metric win is a hypothesis, not a result; only the online
SLAM-level check is load-bearing.** The refit assets are not deleted
(`checkpoints/retrieval/tum/`) in case a future, larger corpus or a
different codebook-training protocol is worth revisiting, but they are not
wired in as a default and `--retriever-path` defaults to `None` (original
MASt3R assets) exactly as before this work started.

**This closes stage 1 with a real, online-confirmed answer (not a sample-
starved inconclusive one), which makes stage 2 (threshold recalibration)
moot -- there is no shipped asset change to recalibrate thresholds around.
Stage 3 (LoRA-feature refit) is doubly moot: its blocking premise (a LoRA
adapter that changes encoder weights) no longer exists in this project at
all -- encoder-LoRA was measured catastrophically negative and dropped from
all planning this session (splatt3r-lora-finetuning skill), and head-only
training (the surviving, shipped fine-tuning route) never touches the
encoder, so there is no future scenario in this project where retrieval
features drift from what the original MASt3R assets were fit on.**
`--retriever-path` and the asset-saving script remain as reusable
infrastructure (e.g. for a future, much larger real-world corpus), but the
retrieval-refit line itself is closed, not paused.
