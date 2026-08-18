"""EXPERIMENT: does upstream's own recipe -- frozen encoder, train ONLY the
Gaussian head -- beat the base checkpoint, where encoder-LoRA measurably did not?

Why this exists
---------------
The encoder-LoRA route was measured against base under an identical protocol
(150 held-out tum val samples, same resolution/stride/mask settings) and lost
on every metric:

    BASE(no LoRA)  val/loss=0.1976  mse=0.1123  psnr=9.50  lpips=0.3414
    LoRA(best)     val/loss=0.2951  mse=0.1814  psnr=7.41  lpips=0.4545   (-49%)

Rendered images confirm the failure mode is geometric, not perceptual: the
LoRA render is washed out by a haze of oversized, semi-transparent Gaussians
(measured separately: p99 predicted scale 85x the base model's, 0.35% pinned
at the clamp ceiling, screen-coverage proxy 1821x).

Leading hypothesis for the root cause: we unfroze what upstream deliberately
froze. `splatt3r_core/main.py`'s own MAST3RGaussians.__init__ does
`self.encoder.requires_grad_(False)` and re-enables only
`downstream_head{1,2}.gaussian_dpt.dpt`, and its forward() puts `torch.no_grad()`
around _encode_symmetrized/_decoder but NOT around _downstream_head. Meanwhile
157,870,984 params of matching/pointmap heads stay frozen on top of that same
encoder -- so perturbing the encoder feeds out-of-distribution features to a
large frozen consumer that never gets to adapt, and the Gaussian head is left
chasing a moving target.

This script therefore trains the ONE configuration upstream proved works, and
scores it against base with the same protocol used for the LoRA verdict. It is
deliberately a standalone experiment, not a change to the production training
path: if the hypothesis holds, integration comes after.

Deliberately held constant vs. the LoRA runs (one variable at a time):
  - same data pipeline, coverage cache, val split, resolution
  - NO scale penalty (that was a patch for the explosion this should prevent)
  - NO normalize_exposure yet (a real train/inference mismatch worth fixing,
    but adding it here would confound the encoder-freezing result)

Usage:
    cd /home/share-v5/Codes/Splatt3R-SLAM
    python3 scripts/exp_head_only.py            # train + score
    python3 scripts/exp_head_only.py --eval-only  # score an existing head ckpt
"""
import argparse
import math
import os
import pickle
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))
os.chdir(CORE)

import mast3r.utils.path_to_dust3r  # noqa
import torch

import utils.loss_mask as loss_mask
from data.data import DUST3RSplattingDataset
from data.tum.tum import TUMData
from main import MAST3RGaussians
from utils import geometry

# --- experiment configuration -------------------------------------------
RES = (512, 384)          # tum's deployment resolution (see train_lora_per_scene FAMILIES)
VAL_FRACTION = 0.15
STRIDE = 1                # full density; the encoder is under no_grad now, so
                          # the activation pressure that forced stride=2 is gone
BATCH = 2
LR = 1e-5                 # upstream configs/main.yaml's opt.lr
GRAD_CLIP = 0.5           # upstream's gradient_clip_val
EPOCHS = 6
SAMPLES_PER_SEQ = 200     # DUST3RSplattingDataset num_epochs_per_epoch; smaller
                          # than training's 1000 so an epoch is minutes, not hours
VAL_SAMPLES = 150         # matches the base-vs-LoRA verdict protocol exactly
DEV = "cuda:0"

FAMILY = "tum"
WARMUP_STEPS = 0          # only needed once BATCH is raised, see --batch

BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "head_only", "tum")
COV_TRAIN = os.path.join(REPO_ROOT, "checkpoints", "lora_coverage_cache",
                         "tum_train_valfrac0.15_pos0.5.pkl")
COV_VAL = os.path.join(REPO_ROOT, "checkpoints", "lora_coverage_cache",
                       "tum_val_valfrac0.15_pos0.5.pkl")


def configure_family(family, batch, lr):
    """Point the module-level config at `family` and scale LR for `batch`.

    LR scaling uses the SQUARE-ROOT rule (lr * sqrt(batch/2)), not the linear
    rule: linear scaling was derived for SGD with momentum, while for
    Adam-family optimizers the update is already normalized by the gradient's
    second moment, so linear scaling overshoots. It would also be actively
    dangerous here -- the encoder-LoRA runs collapsed at lr=1e-4, and linear
    scaling from batch 2 to 24 lands exactly there (1.2e-4).

    Fewer, larger steps is a real change, not just a speed knob: at equal
    samples seen, batch 24 takes 12x fewer optimizer steps than batch 2. That
    is the standard large-batch generalization tradeoff, and it is why warmup
    is switched on with the batch increase rather than left at 0.
    """
    global FAMILY, RES, COV_TRAIN, COV_VAL, BATCH, LR, WARMUP_STEPS
    import train_lora_per_scene as T

    FAMILY = family
    RES = T.FAMILIES[family][3]
    tag = f"valfrac{T.VAL_FRACTION}_pos{T.COVERAGE_POS_THRESHOLD}"
    COV_TRAIN = os.path.join(T.COVERAGE_CACHE_ROOT, f"{family}_train_{tag}.pkl")
    COV_VAL = os.path.join(T.COVERAGE_CACHE_ROOT, f"{family}_val_{tag}.pkl")

    BATCH = batch
    LR = lr if lr is not None else LR * math.sqrt(batch / 2)
    WARMUP_STEPS = 0 if batch <= 2 else 100


class MAST3RGaussiansHeadOnly(MAST3RGaussians):
    """Upstream's forward(), plus the numerical guard the rasterizer needs.

    The guard (nan_to_num + clamp on scales, quaternion normalization) is a
    no-op in the healthy regime -- the base model's largest predicted scale is
    0.0231, three orders of magnitude below the 2.0 ceiling -- so it does not
    alter training, it only prevents a degenerate prediction from corrupting
    memory inside the CUDA rasterizer. Notably absent: the scale PENALTY from
    the LoRA path, which does shape the loss and was only ever needed because
    encoder training made scales explode.
    """

    def forward(self, view1, view2):
        import einops
        from utils import sh_utils

        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = \
                self.encoder._encode_symmetrized(view1, view2)
            dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)

        pred1 = self.encoder._downstream_head(1, [t.float() for t in dec1], shape1)
        pred2 = self.encoder._downstream_head(2, [t.float() for t in dec2], shape2)

        for pred in (pred1, pred2):
            pred["scales"] = torch.nan_to_num(
                pred["scales"], nan=1e-6, posinf=2.0, neginf=1e-6
            ).clamp(min=1e-6, max=2.0)
            rot = torch.nan_to_num(pred["rotations"], nan=0.0, posinf=0.0, neginf=0.0)
            pred["rotations"] = rot / rot.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        pred1["covariances"] = geometry.build_covariance(pred1["scales"], pred1["rotations"])
        pred2["covariances"] = geometry.build_covariance(pred2["scales"], pred2["rotations"])

        new_sh1 = torch.zeros_like(pred1["sh"])
        new_sh2 = torch.zeros_like(pred2["sh"])
        new_sh1[..., 0] = sh_utils.RGB2SH(einops.rearrange(view1["original_img"], "b c h w -> b h w c"))
        new_sh2[..., 0] = sh_utils.RGB2SH(einops.rearrange(view2["original_img"], "b c h w -> b h w c"))
        pred1["sh"] = pred1["sh"] + new_sh1
        pred2["sh"] = pred2["sh"] + new_sh2

        pred2["pts3d_in_other_view"] = pred2.pop("pts3d")
        pred2["means_in_other_view"] = pred2.pop("means")
        return pred1, pred2


def head_params(model):
    """The only trainable tensors: both Gaussian DPT heads."""
    return [p for n, p in model.encoder.named_parameters()
            if "gaussian_dpt" in n and p.requires_grad]


def head_state_dict(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.encoder.state_dict().items() if "gaussian_dpt" in n}


def to_dev(view):
    return {k: (v.to(DEV) if torch.is_tensor(v) else v) for k, v in view.items()}


def batch_to_dev(batch):
    return {"context": [to_dev(v) for v in batch["context"]],
            "target": [to_dev(v) for v in batch["target"]]}


OPACITY_PENALTY = 0.0
OPACITY_TARGET = 0.9


def step_loss(model, batch):
    b = batch_to_dev(batch)
    v1, v2 = b["context"]
    _, _, h, w = v1["img"].shape
    p1, p2 = model.forward(v1, v2)
    color, _ = model.decoder(b, p1, p2, (h, w))
    m = loss_mask.calculate_loss_mask(b)
    out = model.calculate_loss(b, v1, v2, p1, p2, color, m,
                               apply_mask=True, average_over_mask=True,
                               calculate_ssim=False)
    if OPACITY_PENALTY > 0:
        # ANTI-SATURATION. The base checkpoint predicts opacity 1.0 for
        # essentially every Gaussian (17.58) and the Replica head keeps it
        # there, while the TUM head learns to un-saturate -- and only the
        # un-saturated head improves a baked map. This penalises only the
        # saturated tail, so a healthy distribution is untouched: it asks the
        # head to stop using full opacity everywhere, not to prefer any
        # particular value.
        pen = 0.0
        for p in (p1, p2):
            o = p["opacities"] if "opacities" in p else p.get("opacity")
            if o is not None:
                pen = pen + torch.relu(o - OPACITY_TARGET).mean()
        # The penalty goes into the TRAINING objective only. Scoring the
        # verdict on the penalized loss compares the base checkpoint -- which
        # predicts opacity 1.0 almost everywhere and therefore eats the maximum
        # possible penalty -- against a head that was trained to avoid it, and
        # reports the gap as a quality win. That is how the o-0.6 run came to
        # print "BEATS base by 94.1%" while its val psnr was actually 0.13 dB
        # WORSE than the plain head's (17.69.1-17.69.2). Keep the render-loss
        # component separate so the verdict can be scored on it.
        out = (out[0] + OPACITY_PENALTY * pen, ) + tuple(out[1:]) + (out[0],)
    else:
        out = tuple(out) + (out[0],)
    return out


EVAL_SEED = 1234


@torch.no_grad()
def evaluate(model, loader, tag):
    """Re-seed before every evaluation.

    DUST3RSplattingDataset.__getitem__ re-samples its context/target view
    triplet on EVERY call, via unseeded random.randint/choice/sample
    (data/data.py:132,143,170). A fixed index list is therefore
    deterministic in index only, NOT in content -- every evaluation scored
    a different draw of samples. Measured cost: three evaluations of the
    byte-identical base checkpoint scored psnr 9.7268 / 9.8648 / 10.3221
    (lpips 0.2801 / 0.2856 / 0.2793), a ~0.6 dB band from sampling alone.
    Large effects survive that -- route A's -2.09 dB, and route B's gain,
    which re-measured on a fixed draw at +1.00 dB vs the +1.11 dB seen
    across draws -- but per-epoch deltas of ~0.1 dB do not, and no
    cross-run comparison is valid without this. Seeding here makes every
    evaluate() call score the exact same 150 triplets.
    """
    import random as _random

    # random.seed() alone only covers num_workers=0. With workers, sampling
    # happens in the child processes, which torch seeds as base_seed +
    # worker_id, where base_seed is drawn from torch's global RNG when the
    # iterator is created -- so torch.manual_seed here is what actually
    # pins the worker draws.
    #
    # Both are GLOBAL, and evaluate() runs at the end of every training
    # epoch, so seeding without restoring would reset the training RNG to
    # the same state each time and make every epoch train on an identical
    # draw. Snapshot and restore around the evaluation.
    _py_state = _random.getstate()
    _torch_state = torch.get_rng_state()
    _random.seed(EVAL_SEED)
    torch.manual_seed(EVAL_SEED)
    try:
        return _evaluate_seeded(model, loader, tag)
    finally:
        _random.setstate(_py_state)
        torch.set_rng_state(_torch_state)


@torch.no_grad()
def _evaluate_seeded(model, loader, tag):
    model.eval()
    tl = tm = tp = tr = 0.0
    n = 0
    for batch in loader:
        loss, mse, lp, render_only = step_loss(model, batch)[:4]
        tl += loss.item(); tm += mse.item(); tp += lp.item()
        tr += render_only.item(); n += 1
    mse_avg = tm / n
    # calculate_loss's mse sums squared error over all 3 colour channels but
    # divides by mask.sum(), which counts PIXELS -- so its "mse" is 3x the
    # per-channel MSE, and a PSNR taken straight from it is 10*log10(3) =
    # 4.77 dB too low. Verified directly (scripts/diag_psnr.py): base scores
    # 10.3221 as-reported vs 15.0933 per-channel. The loss itself is left
    # alone -- the factor is constant, so it only acts as a 3x weighting of
    # mse against lpips, and changing it mid-project would silently change
    # the training objective. Only the reported PSNR is corrected here.
    # Every relative result in this project is unaffected: base and all
    # trained checkpoints went through the identical path.
    psnr = -10 * math.log10(mse_avg / 3.0)
    print(f"  {tag:>16} | val/loss={tl/n:.4f}  mse={mse_avg:.4f}  "
          f"psnr={psnr:.4f}  lpips={tp/n:.4f}  (n={n})", flush=True)
    # Second return value is the RENDER loss with any penalty term excluded --
    # the only one comparable across runs with different penalty settings.
    return tl / n, tr / n


def build_loaders():
    """Data for FAMILY, whose per-family resolution and coverage cache are the
    same ones train_lora_per_scene.py and precompute_coverage.py use, so a run
    here is directly comparable to the production training path."""
    import train_lora_per_scene as T

    data_cls, root, extra, _res = T.FAMILIES[FAMILY]
    train_data = data_cls(root, "train", val_fraction=VAL_FRACTION, **extra)
    val_data = data_cls(root, "val", val_fraction=VAL_FRACTION, **extra)
    with open(COV_TRAIN, "rb") as f:
        cov_tr = pickle.load(f)
    with open(COV_VAL, "rb") as f:
        cov_va = pickle.load(f)

    tr_ds = DUST3RSplattingDataset(train_data, cov_tr, resolution=RES,
                                   num_epochs_per_epoch=SAMPLES_PER_SEQ)
    va_ds = DUST3RSplattingDataset(val_data, cov_va, resolution=RES,
                                   num_epochs_per_epoch=100)
    # Same deterministic val subset the base-vs-LoRA verdict used, so the
    # numbers here are directly comparable to it.
    va_sub = torch.utils.data.Subset(
        va_ds, list(range(0, len(va_ds), max(1, len(va_ds) // VAL_SAMPLES))))

    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=8)
    va = torch.utils.data.DataLoader(va_sub, batch_size=1, shuffle=False, num_workers=4)
    return tr, va


def main():
    global OUT_DIR, EPOCHS

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS,
                    help="the 6-epoch default was chosen to make a run take "
                         "minutes; neither route had converged by then")
    ap.add_argument("--out", default=OUT_DIR,
                    help="checkpoint dir; override so a long run does not "
                         "overwrite the 6-epoch result it is compared against")
    ap.add_argument("--family", default=FAMILY,
                    choices=("tum", "tum-pseudo", "7-scenes", "euroc", "eth3d", "replica",
                             "replica-noisy", "replica-photo", "replica-photo-spatial",
                             "replica-mixed"))
    ap.add_argument("--batch", type=int, default=BATCH,
                    help="batch>2 buys at most 12%% throughput (measured, see "
                         "exp_batch_scan.py) and changes the optimization, so "
                         "it needs the LR scaling and warmup that come with it")
    ap.add_argument("--lr", type=float, default=None,
                    help="explicit LR; default is sqrt-scaled from batch")
    ap.add_argument("--opacity-penalty", type=float, default=0.0, metavar="W",
                    help="weight on relu(opacity-0.9), i.e. a penalty on the "
                         "saturated tail only. Kimi round 24: if this yields an "
                         "unsaturated head that keeps the val gain and flips "
                         "the baked sign, it is simultaneously the evidence "
                         "for the saturation mechanism and the fix for it.")
    ap.add_argument("--opacity-target", type=float, default=0.9, metavar="T",
                    help="the knee of the penalty: relu(opacity - T). T=0.9 "
                         "only clips the saturated tail and left the median at "
                         "0.84, which is why external thinning still bought "
                         "21%% on that head (17.60) -- the >0.9 fraction said "
                         "'un-saturated' while the accumulated alpha did not. "
                         "T=0.6 targets the level the TUM head reached on its "
                         "own (median 0.66).")
    ap.add_argument("--seed", type=int, default=None,
                    help="training seed (data order + any stochastic op). "
                         "Every head-training result so far is single-seed, "
                         "while the refinement experiments measured a seed "
                         "sigma of 0.66 dB under the same protocol -- so the "
                         "smaller diagonal claims (VGG-LPIPS puts three "
                         "families at -1.7%% to -4.9%%) currently have no "
                         "noise bound at all. evaluate() snapshots and "
                         "restores RNG state, so this does not disturb the "
                         "fixed evaluation draw.")
    args = ap.parse_args()

    if args.seed is not None:
        import random as _r
        _r.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    global OPACITY_PENALTY, OPACITY_TARGET
    OPACITY_PENALTY = args.opacity_penalty
    OPACITY_TARGET = args.opacity_target
    configure_family(args.family, args.batch, args.lr)
    # This module does os.chdir(CORE) at import, long before argparse runs, so
    # a relative --out would silently resolve under splatt3r_core/ instead of
    # the repo root. Anchor it explicitly.
    OUT_DIR = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    EPOCHS = args.epochs
    os.makedirs(OUT_DIR, exist_ok=True)
    tr, va = build_loaders()
    print(f"train samples/epoch={len(tr.dataset)}  val samples={len(va.dataset)}  "
          f"res={RES}  stride={STRIDE}  batch={BATCH}  lr={LR:.3g}  "
          f"warmup={WARMUP_STEPS}  family={FAMILY}  seed={args.seed}\n", flush=True)

    model = MAST3RGaussiansHeadOnly.load_from_checkpoint(BASE_CKPT, map_location=DEV).to(DEV)
    model.decoder.spatial_stride = STRIDE

    n_tr = sum(p.numel() for p in head_params(model))
    n_all = sum(p.numel() for p in model.encoder.parameters())
    print(f"trainable (Gaussian head only): {n_tr:,} / {n_all:,} "
          f"({100*n_tr/n_all:.2f}%)\n", flush=True)

    print("=== baseline: untouched base checkpoint ===", flush=True)
    base_loss, base_render = evaluate(model, va, "BASE")
    if args.eval_only:
        return

    opt = torch.optim.AdamW(head_params(model), lr=LR, weight_decay=0.05)
    best = base_loss
    best_render = base_render
    print(f"\n=== training (target to beat: {base_loss:.4f}) ===", flush=True)

    gstep = 0
    n_skipped = 0
    max_gnorm = 0.0
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(tr):
            if gstep < WARMUP_STEPS:
                # Linear warmup. At batch 2 this is disabled (WARMUP_STEPS=0);
                # it exists because a sqrt-scaled LR applied from step 0 to a
                # head whose optimizer state is empty is the classic way to
                # wreck a pretrained initialization in the first few updates.
                for g in opt.param_groups:
                    g["lr"] = LR * (gstep + 1) / WARMUP_STEPS
            elif gstep == WARMUP_STEPS and WARMUP_STEPS:
                for g in opt.param_groups:
                    g["lr"] = LR
            gstep += 1

            loss = step_loss(model, batch)[0]

            # Skip the update on a non-finite loss instead of letting it reach
            # the weights. One poisoned step is unrecoverable: NaN parameters
            # produce NaN gradients forever, so training silently continues to
            # completion while learning nothing. The ETH3D run did exactly
            # that -- fine through epoch 27, NaN within the first 50 steps of
            # epoch 28, then twelve dead epochs that still printed as progress.
            # A single bad batch should cost one step, not the rest of the run.
            if not torch.isfinite(loss):
                n_skipped += 1
                print(f"  ep{ep} step{i} SKIPPED: non-finite loss "
                      f"({loss.item()}); {n_skipped} skipped so far", flush=True)
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(head_params(model), GRAD_CLIP)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                print(f"  ep{ep} step{i} SKIPPED: non-finite grad norm; "
                      f"{n_skipped} skipped so far", flush=True)
                opt.zero_grad(set_to_none=True)
                continue

            max_gnorm = max(max_gnorm, float(gnorm))
            opt.step()
            if i % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  ep{ep} step{i}/{len(tr)} loss={loss.item():.4f} "
                      f"gnorm={float(gnorm):.3f} max={max_gnorm:.3f} "
                      f"peak_mem={mem:.1f}GiB", flush=True)
        vl, vr = evaluate(model, va, f"epoch {ep}")
        if n_skipped:
            print(f"  ep{ep} skipped {n_skipped} non-finite steps in total",
                  flush=True)
        print(f"  ep{ep} done in {time.time()-t0:.0f}s "
              f"({'BETTER' if vl < base_loss else 'worse'} than base)", flush=True)
        if vl < best:
            best = vl
            best_render = vr
            torch.save(head_state_dict(model), os.path.join(OUT_DIR, "head_best.pt"))
            print(f"  -> new best {vl:.4f}, saved", flush=True)

    print(f"\n=== VERDICT ===")
    print(f"  skipped steps  : {n_skipped}  (max grad norm seen {max_gnorm:.3f})")
    # Scored on the RENDER loss with penalty terms excluded (17.69.2). The
    # penalized loss is what training optimizes, but comparing it across runs
    # with different penalty settings is meaningless: the base checkpoint is
    # maximally saturated and so eats the largest possible penalty, which shows
    # up as a spurious quality win for any penalty run.
    print(f"  base   (render): {base_render:.4f}")
    print(f"  head-only best : {best_render:.4f}   (train objective {best:.4f})")
    if best_render < base_render:
        print(f"  => head-only BEATS base by "
              f"{(base_render-best_render)/base_render*100:.1f}%")
    else:
        print(f"  => head-only does NOT beat base "
              f"({(best_render-base_render)/base_render*100:.1f}% worse)")


if __name__ == "__main__":
    main()
