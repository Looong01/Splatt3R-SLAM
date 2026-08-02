"""EXPERIMENT: does raising the LPIPS loss weight buy perceptual quality that
the upstream 1.0/0.25 mse:lpips split leaves on the table?

Follow-up to scripts/exp_head_only.py, which established that upstream's recipe
(frozen encoder, train only the Gaussian head) beats base by 16.5%. Three
independent observations from that run motivate this one:

  1. psnr improved monotonically (9.73 -> 10.84, +1.11 dB) while lpips stopped
     improving after epoch 1 (0.2590 -> 0.2593 over the remaining 4 epochs).
  2. Rendered images: the head-only model removes base's blotchy splash
     artifacts, but its wood-grain texture is visibly smoother/softer.
  3. The stated motivation for fine-tuning at all is the blur complaint
     (see the splatt3r-gaussian-map skill) -- and MSE is precisely the term
     that rewards blur, since averaging minimizes squared error under
     uncertainty. Optimizing an MSE-dominated objective to fix blur works
     against itself.

Note the nominal 4:1 weighting is less lopsided in practice: at epoch 3 the
actual contributions were 1.0*0.0885 = 0.0885 (mse, 57%) vs 0.25*0.2633 =
0.0658 (lpips, 43%). LPIPS_WEIGHT=1.0 below flips that to roughly 1:3.

DECISION CRITERION, fixed in advance (val/loss is NOT usable here -- changing
the weights changes what it means, so it cannot be compared against the
head-only run's 0.1473 or base's 0.1765; this is the same class of mistake as
comparing val/loss across different decoder strides):

    ACCEPT if lpips improves meaningfully over the head-only run's 0.2593
    AND psnr stays at or above base's 9.7268.

Everything else is held identical to exp_head_only.py.

--- original header follows ---

EXPERIMENT: does upstream's own recipe -- frozen encoder, train ONLY the
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
LPIPS_WEIGHT = 1.0        # THE variable under test (upstream: 0.25)
EPOCHS = 6
SAMPLES_PER_SEQ = 200     # DUST3RSplattingDataset num_epochs_per_epoch; smaller
                          # than training's 1000 so an epoch is minutes, not hours
VAL_SAMPLES = 150         # matches the base-vs-LoRA verdict protocol exactly
DEV = "cuda:0"

BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "head_only_lpips", "tum")
COV_TRAIN = os.path.join(REPO_ROOT, "checkpoints", "lora_coverage_cache",
                         "tum_train_valfrac0.15_pos0.5.pkl")
COV_VAL = os.path.join(REPO_ROOT, "checkpoints", "lora_coverage_cache",
                       "tum_val_valfrac0.15_pos0.5.pkl")


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


def step_loss(model, batch):
    b = batch_to_dev(batch)
    v1, v2 = b["context"]
    _, _, h, w = v1["img"].shape
    p1, p2 = model.forward(v1, v2)
    color, _ = model.decoder(b, p1, p2, (h, w))
    m = loss_mask.calculate_loss_mask(b)
    return model.calculate_loss(b, v1, v2, p1, p2, color, m,
                                apply_mask=True, average_over_mask=True,
                                calculate_ssim=False)


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
    tl = tm = tp = 0.0
    n = 0
    for batch in loader:
        loss, mse, lp = step_loss(model, batch)
        tl += loss.item(); tm += mse.item(); tp += lp.item(); n += 1
    mse_avg = tm / n
    print(f"  {tag:>16} | val/loss={tl/n:.4f}  mse={mse_avg:.4f}  "
          f"psnr={-10*math.log10(mse_avg):.4f}  lpips={tp/n:.4f}  (n={n})", flush=True)
    return tl / n


def build_loaders():
    train_data = TUMData(os.path.join(REPO_ROOT, "datasets", "tum"), "train",
                         val_fraction=VAL_FRACTION)
    val_data = TUMData(os.path.join(REPO_ROOT, "datasets", "tum"), "val",
                       val_fraction=VAL_FRACTION)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tr, va = build_loaders()
    print(f"train samples/epoch={len(tr.dataset)}  val samples={len(va.dataset)}  "
          f"res={RES}  stride={STRIDE}  batch={BATCH}  lr={LR}\n", flush=True)

    model = MAST3RGaussiansHeadOnly.load_from_checkpoint(BASE_CKPT, map_location=DEV).to(DEV)
    model.decoder.spatial_stride = STRIDE
    # calculate_loss() reads these off self.config at call time, so overriding
    # here changes both the training signal and what val/loss reports.
    print(f"loss weights: mse={model.config.loss.mse_loss_weight} "
          f"lpips={model.config.loss.lpips_loss_weight} -> {LPIPS_WEIGHT}", flush=True)
    model.config.loss.lpips_loss_weight = LPIPS_WEIGHT

    n_tr = sum(p.numel() for p in head_params(model))
    n_all = sum(p.numel() for p in model.encoder.parameters())
    print(f"trainable (Gaussian head only): {n_tr:,} / {n_all:,} "
          f"({100*n_tr/n_all:.2f}%)\n", flush=True)

    print("=== baseline: untouched base checkpoint ===", flush=True)
    base_loss = evaluate(model, va, "BASE")
    if args.eval_only:
        return

    opt = torch.optim.AdamW(head_params(model), lr=LR, weight_decay=0.05)
    best = base_loss
    print(f"\n=== training (target to beat: {base_loss:.4f}) ===", flush=True)

    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(tr):
            loss, _, _ = step_loss(model, batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head_params(model), GRAD_CLIP)
            opt.step()
            if i % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  ep{ep} step{i}/{len(tr)} loss={loss.item():.4f} "
                      f"peak_mem={mem:.1f}GiB", flush=True)
        vl = evaluate(model, va, f"epoch {ep}")
        print(f"  ep{ep} done in {time.time()-t0:.0f}s "
              f"({'BETTER' if vl < base_loss else 'worse'} than base)", flush=True)
        if vl < best:
            best = vl
            torch.save(head_state_dict(model), os.path.join(OUT_DIR, "head_best.pt"))
            print(f"  -> new best {vl:.4f}, saved", flush=True)

    print(f"\n=== VERDICT ===")
    print(f"  base           : {base_loss:.4f}")
    print(f"  head-only best : {best:.4f}")
    if best < base_loss:
        print(f"  => head-only BEATS base by {(base_loss-best)/base_loss*100:.1f}%")
    else:
        print(f"  => head-only does NOT beat base "
              f"({(best-base_loss)/base_loss*100:.1f}% worse)")


if __name__ == "__main__":
    main()
