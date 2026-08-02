"""Diagnose why the reported PSNR is ~12 dB when 3DGS papers report 25-35 dB.

Two candidate causes, both measured here rather than argued:

1. **Channel normalization.** `MAST3RGaussians.calculate_loss` computes

       mse = (rgb_l2_loss * mask[:, :, None, ...]).sum() / mask.sum()

   where `rgb_l2_loss` is (b, v, c, h, w) and `mask` is (b, v, h, w). The
   numerator sums over all 3 colour channels; the denominator counts pixels
   only. The result is therefore 3x the per-channel MSE, which understates
   PSNR by 10*log10(3) = 4.77 dB. As a *loss* this is harmless (a constant
   factor, equivalent to weighting mse 3x against lpips); as a *reported
   metric* it is simply wrong.

2. **Masked-out area.** The target view is only partly covered by the two
   context views -- the renders show large black regions where nothing was
   reconstructed. If those pixels reach the metric they would dominate it.
   `calculate_loss_mask` is supposed to exclude them; this script reports the
   actual covered fraction so the assumption is checked rather than trusted.

Neither affects any *relative* result in this project: base and every trained
checkpoint were scored through the identical code path, so all reported deltas
stand. Only the absolute PSNR scale is at issue.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/diag_psnr.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import exp_head_only as E  # noqa: E402
import utils.loss_mask as loss_mask  # noqa: E402

LONG_CKPT = os.path.join(E.REPO_ROOT, "checkpoints", "head_only_long", "tum",
                         "head_best.pt")


@torch.no_grad()
def measure(model, loader, tag):
    model.eval()
    n = 0
    as_reported = 0.0     # what evaluate() prints: sum over channels / pixels
    per_channel = 0.0     # sum over channels / (pixels * channels)
    covered = 0.0         # fraction of target pixels the mask keeps

    for b in loader:
        b = E.batch_to_dev(b)
        v1, v2 = b["context"]
        _, _, h, w = v1["img"].shape
        p1, p2 = model.forward(v1, v2)
        color, _ = model.decoder(b, p1, p2, (h, w))
        mask = loss_mask.calculate_loss_mask(b)

        target = torch.stack([t["original_img"] for t in b["target"]], dim=1)
        sq = (color - target) ** 2                      # (b, v, c, h, w)
        m = mask[:, :, None, ...]                       # (b, v, 1, h, w)
        num = (sq * m).sum()
        pixels = mask.sum().clamp(min=1)

        as_reported += (num / pixels).item()
        per_channel += (num / (pixels * sq.shape[2])).item()
        covered += mask.float().mean().item()
        n += 1

    ar, pc, cov = as_reported / n, per_channel / n, covered / n
    print(f"  {tag:>8} | as-reported mse={ar:.4f} psnr={-10*math.log10(ar):7.4f}   "
          f"per-channel mse={pc:.4f} psnr={-10*math.log10(pc):7.4f}   "
          f"mask covers {cov*100:5.1f}% of target pixels", flush=True)


def main():
    _, va = E.build_loaders()
    model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
        E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
    model.decoder.spatial_stride = E.STRIDE

    import random
    random.seed(E.EVAL_SEED)
    torch.manual_seed(E.EVAL_SEED)
    measure(model, va, "base")

    if os.path.exists(LONG_CKPT):
        sd = torch.load(LONG_CKPT, map_location=E.DEV)
        model.encoder.load_state_dict(sd, strict=False)
        random.seed(E.EVAL_SEED)
        torch.manual_seed(E.EVAL_SEED)
        measure(model, va, "long40")


if __name__ == "__main__":
    main()
