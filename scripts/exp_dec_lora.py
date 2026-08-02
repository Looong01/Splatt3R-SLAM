"""Route D — DECODER-only LoRA ablation, isolating 3.3's mechanism.

Route A (encoder LoRA) measured -49% vs base: unfreezing the encoder fed
out-of-distribution features to ~158M frozen matching/pointmap-head params
and drove a Gaussian scale explosion (skill 3.1-3.3). Route B (frozen
encoder, Gaussian head only) measured +1.00 dB. The open question 3.3 left:
is DECODER adaptation also toxic, or was the encoder specifically the
problem? The decoder sits downstream of _encode_image, so LoRA there leaves
retrieval features and matching inputs BIT-IDENTICAL to base by
construction -- which is also why, whichever way this goes, the
retrieval-asset refit (splatt3r-retrieval-refit) stays closed.

Arm D = route B's exact protocol (frozen encoder, trainable Gaussian head,
same data, same seeded eval draw) plus LoRA r=8/alpha=16 on the decoder
blocks ONLY:

    encoder (_encode_symmetrized)  : frozen, under no_grad  -> identical
    decoder (dec_blocks, dec_blocks2): LoRA adapters, trained
    Gaussian DPT heads             : fully trained, as route B

Read the outcome against route B's numbers (same protocol): base psnr
10.3221 / lpips 0.2793, route B 6-epoch 11.3217 / 0.2620 (+1.00 dB). D >> B
means decoder capacity was the limiter all along; D ~= B means the head
already saturates what the frozen features support; D collapses means
adaptation anywhere upstream of the head is destabilizing and route B is
the end of the line. The p99-scale canary per eval watches for 3.1's
explosion signature.

Usage:
    python3 scripts/exp_dec_lora.py                 # train + score (TUM, 6 ep)
    python3 scripts/exp_dec_lora.py --eval-only     # score an existing run
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
from main import MAST3RGaussians
from utils import geometry

# --- experiment configuration (mirrors exp_head_only.py route B) ----------
RES = (512, 384)
VAL_FRACTION = 0.15
STRIDE = 1
BATCH = 2
LR = 1e-5
GRAD_CLIP = 0.5
EPOCHS = 6
SAMPLES_PER_SEQ = 200
VAL_SAMPLES = 150
FAMILY = "tum"
BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
OUT_DIR = os.path.join(REPO_ROOT, "checkpoints", "exp_dec_lora")
DEV = "cuda"
LORA_R, LORA_ALPHA = 8, 16
DEC_TARGET = r"dec_blocks2?\.\d+\.(attn|cross_attn|mlp)\.(qkv|proj|projq|projk|projv|fc1|fc2)$"
GAUSSIAN_HEAD_MODULES = ["downstream_head1.gaussian_dpt.dpt",
                         "downstream_head2.gaussian_dpt.dpt"]


class MAST3RGaussiansDecLoRA(MAST3RGaussians):
    """Route B's forward, but only _encode_symmetrized is under no_grad.

    Encoder output is bit-identical to the frozen base (retrieval/matching
    features untouched by construction); gradients flow through the decoder
    so the LoRA adapters train; the Gaussian heads train exactly as route B.
    """

    def forward(self, view1, view2):
        import einops
        from utils import sh_utils

        base = (self.encoder.get_base_model()
                if hasattr(self.encoder, "get_base_model") else self.encoder)
        with torch.no_grad():
            (shape1, shape2), (feat1, feat2), (pos1, pos2) = \
                base._encode_symmetrized(view1, view2)
        dec1, dec2 = base._decoder(feat1, pos1, feat2, pos2)

        pred1 = base._downstream_head(1, [t.float() for t in dec1], shape1)
        pred2 = base._downstream_head(2, [t.float() for t in dec2], shape2)

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


def attach_dec_lora(model, r=LORA_R, alpha=LORA_ALPHA):
    """peft LoRA scoped to the decoder blocks by full-name regex. The
    Gaussian heads ride along as modules_to_save (the gotcha documented in
    splatt3r_core/lora.py: without it, peft save/resume silently drops all
    head training)."""
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        target_modules=DEC_TARGET,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.0,
        modules_to_save=GAUSSIAN_HEAD_MODULES,
    )
    model.encoder = get_peft_model(model.encoder, lora_cfg)
    return model


def trainable_params(model):
    """LoRA adapters + Gaussian heads (encoder-scoped; the LPIPS criterion's
    linear layers also carry requires_grad but are never optimized, an
    upstream quirk)."""
    return [p for n, p in model.encoder.named_parameters() if p.requires_grad]


def dec_lora_state_dict(model):
    return {n: p.detach().cpu().clone()
            for n, p in model.encoder.state_dict().items()
            if ("lora_" in n) or ("gaussian_dpt" in n)}


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
    """Seeded fixed-draw evaluation, byte-identical protocol to
    exp_head_only.py (see its comment: an unseeded draw costs ~0.6 dB)."""
    import random as _random

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
    scale_p99 = 0.0
    for batch in loader:
        loss, mse, lp = step_loss(model, batch)
        tl += loss.item(); tm += mse.item(); tp += lp.item(); n += 1
        # 3.1 canary: predicted Gaussian scales, 85x explosion in route A.
        v1, v2 = batch_to_dev(batch)["context"]
        p1, p2 = model.forward(v1, v2)
        s = torch.cat([p1["scales"].flatten(), p2["scales"].flatten()])
        scale_p99 = max(scale_p99, float(torch.quantile(s, 0.99)))
    mse_avg = tm / n
    psnr = -10 * math.log10(mse_avg / 3.0)  # see exp_head_only.py on the /3
    print(f"  {tag:>16} | val/loss={tl/n:.4f}  mse={mse_avg:.4f}  "
          f"psnr={psnr:.4f}  lpips={tp/n:.4f}  scale_p99={scale_p99:.4f}  (n={n})",
          flush=True)
    return tl / n


def build_loaders():
    import train_lora_per_scene as T

    data_cls, root, extra, _res = T.FAMILIES[FAMILY]
    train_data = data_cls(root, "train", val_fraction=VAL_FRACTION, **extra)
    val_data = data_cls(root, "val", val_fraction=VAL_FRACTION, **extra)
    tag = f"valfrac{T.VAL_FRACTION}_pos{T.COVERAGE_POS_THRESHOLD}"
    with open(os.path.join(T.COVERAGE_CACHE_ROOT, f"{FAMILY}_train_{tag}.pkl"), "rb") as f:
        cov_tr = pickle.load(f)
    with open(os.path.join(T.COVERAGE_CACHE_ROOT, f"{FAMILY}_val_{tag}.pkl"), "rb") as f:
        cov_va = pickle.load(f)

    tr_ds = DUST3RSplattingDataset(train_data, cov_tr, resolution=RES,
                                   num_epochs_per_epoch=SAMPLES_PER_SEQ)
    va_ds = DUST3RSplattingDataset(val_data, cov_va, resolution=RES,
                                   num_epochs_per_epoch=100)
    va_sub = torch.utils.data.Subset(
        va_ds, list(range(0, len(va_ds), max(1, len(va_ds) // VAL_SAMPLES))))

    tr = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=8)
    va = torch.utils.data.DataLoader(va_sub, batch_size=1, shuffle=False, num_workers=4)
    return tr, va


def main():
    global OUT_DIR, EPOCHS

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    if args.seed is not None:
        import random as _r
        _r.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    OUT_DIR = args.out if os.path.isabs(args.out) else os.path.join(REPO_ROOT, args.out)
    EPOCHS = args.epochs
    os.makedirs(OUT_DIR, exist_ok=True)
    tr, va = build_loaders()
    print(f"route D (decoder-LoRA r={LORA_R} a={LORA_ALPHA})  "
          f"train samples/epoch={len(tr.dataset)}  val samples={len(va.dataset)}  "
          f"res={RES}  batch={BATCH}  lr={LR:.3g}  family={FAMILY}  seed={args.seed}\n",
          flush=True)

    model = MAST3RGaussiansDecLoRA.load_from_checkpoint(BASE_CKPT, map_location=DEV).to(DEV)
    model.decoder.spatial_stride = STRIDE

    print("=== baseline: untouched base checkpoint ===", flush=True)
    base_loss = evaluate(model, va, "BASE")
    if args.eval_only:
        return

    model = attach_dec_lora(model)
    model.train()
    n_tr = sum(p.numel() for p in trainable_params(model))
    print(f"trainable (decoder-LoRA + head): {n_tr:,}\n", flush=True)

    opt = torch.optim.AdamW(trainable_params(model), lr=LR, weight_decay=0.05)
    best = base_loss
    print(f"=== training (target to beat: {base_loss:.4f}) ===", flush=True)

    n_skipped = 0
    max_gnorm = 0.0
    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        for i, batch in enumerate(tr):
            loss, _, _ = step_loss(model, batch)
            if not torch.isfinite(loss):
                n_skipped += 1
                print(f"  ep{ep} step{i} SKIPPED: non-finite loss", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(trainable_params(model), GRAD_CLIP)
            if not torch.isfinite(gnorm):
                n_skipped += 1
                print(f"  ep{ep} step{i} SKIPPED: non-finite grad norm", flush=True)
                opt.zero_grad(set_to_none=True)
                continue
            max_gnorm = max(max_gnorm, float(gnorm))
            opt.step()
            if i % 50 == 0:
                mem = torch.cuda.max_memory_allocated() / 2**30
                print(f"  ep{ep} step{i}/{len(tr)} loss={loss.item():.4f} "
                      f"gnorm={float(gnorm):.3f} max={max_gnorm:.3f} "
                      f"peak_mem={mem:.1f}GiB", flush=True)
        vl = evaluate(model, va, f"epoch {ep}")
        print(f"  ep{ep} done in {time.time()-t0:.0f}s "
              f"({'BETTER' if vl < base_loss else 'worse'} than base)", flush=True)
        if vl < best:
            best = vl
            torch.save(dec_lora_state_dict(model), os.path.join(OUT_DIR, "dec_lora_best.pt"))
            print(f"  -> new best {vl:.4f}, saved", flush=True)

    print(f"\n=== VERDICT (route D) ===")
    print(f"  skipped steps : {n_skipped}  (max grad norm {max_gnorm:.3f})")
    print(f"  base          : {base_loss:.4f}")
    print(f"  dec-LoRA best : {best:.4f}")
    print(f"  reference     : route B same protocol = +1.00 dB psnr over base")


if __name__ == "__main__":
    main()
