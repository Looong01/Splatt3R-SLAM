"""Render GT / base / head-only(lpips 0.25) / head-only(lpips 1.0) on the SAME
seeded validation samples.

The measurement protocol's standing rule is that no fine-tuning verdict is
believed on scalars alone -- metrics did not reveal that the encoder-LoRA
failure was geometric rather than perceptual, but a single rendered triplet
did, immediately. This script is the visual half of the route B vs. route C
comparison, whose scalars are close enough that images decide it:

    base       mse 0.0929  psnr 10.3221  lpips 0.2793
    route B    mse 0.0738  psnr 11.3217  lpips 0.2620
    route C    mse 0.0763  psnr 11.1722  lpips 0.2565

Route C trades 0.15 dB of psnr for 0.0055 of lpips. Whether that is worth
taking is a question about texture vs. smoothing that the numbers cannot
answer -- LPIPS is a proxy for the blur complaint that motivated this work, not
the complaint itself.

All four columns come from the identical sample, drawn under EVAL_SEED, so
differences are attributable to the weights alone (see the seeding note in
exp_head_only.evaluate).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/exp_render_compare.py [--n 6]
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import exp_head_only as E  # noqa: E402  (does sys.path/chdir setup on import)

CKPT_B = os.path.join(E.REPO_ROOT, "checkpoints", "head_only", "tum", "head_best.pt")
CKPT_C = os.path.join(E.REPO_ROOT, "checkpoints", "head_only_lpips", "tum", "head_best.pt")
CKPT_LONG = os.path.join(E.REPO_ROOT, "checkpoints", "head_only_long", "tum", "head_best.pt")
OUT = os.path.join(E.REPO_ROOT, "logs", "render_compare")


def to_png(t):
    """(3,H,W) float in [0,1] -> HWC uint8."""
    a = t.detach().float().clamp(0, 1).cpu().numpy()
    return (np.transpose(a, (1, 2, 0)) * 255).round().astype(np.uint8)


@torch.no_grad()
def render(model, batch):
    b = E.batch_to_dev(batch)
    v1, v2 = b["context"]
    _, _, h, w = v1["img"].shape
    p1, p2 = model.forward(v1, v2)
    color, _ = model.decoder(b, p1, p2, (h, w))
    return color


def load_head(model, path):
    sd = torch.load(path, map_location=E.DEV)
    missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all("gaussian_dpt" not in k for k in missing), "a head key failed to load"
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6, help="how many samples to render")
    ap.add_argument("--family", default="tum",
                    choices=("tum", "7-scenes", "euroc", "eth3d", "replica"))
    ap.add_argument("--head", default=None,
                    help="a single head checkpoint to compare against base; "
                         "overrides the tum-specific routeB/routeC/long40 set")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    # The module's constants are tum's. Any other family needs its resolution
    # and coverage caches rewired before build_loaders() is called, exactly as
    # exp_head_only.main does.
    if args.family != "tum":
        E.configure_family(args.family, E.BATCH, None)
    out_dir = args.out or (OUT if args.family == "tum" else OUT + "_" + args.family)
    # anchor to the repo root: this module chdir()s into splatt3r_core at
    # import, so a relative --out lands in splatt3r_core/logs/ instead
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(
        E.REPO_ROOT, out_dir)
    globals()["OUT"] = out_dir
    OUT_ = out_dir
    os.makedirs(OUT_, exist_ok=True)

    _, va = E.build_loaders()

    # Same draw the metrics were computed on.
    random.seed(E.EVAL_SEED)
    torch.manual_seed(E.EVAL_SEED)
    batches = []
    for i, batch in enumerate(va):
        if i >= args.n:
            break
        batches.append(batch)
    print(f"collected {len(batches)} samples under seed {E.EVAL_SEED}", flush=True)

    model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
        E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
    model.decoder.spatial_stride = E.STRIDE
    model.eval()

    # Ground truth, written once.
    for i, batch in enumerate(batches):
        gt = batch["target"][0]["original_img"][0]
        _imwrite(os.path.join(OUT_, f"s{i}_gt.png"), to_png(gt))

    if args.head:
        # absolute: this module chdir()s into splatt3r_core at import, so a
        # relative --head resolves there. Third time today (17.45's .npz,
        # 17.49's --referee-scales).
        h = args.head if os.path.isabs(args.head) else os.path.join(
            E.REPO_ROOT, args.head)
        variants = [("base", None), ("head", h)]
    else:
        variants = [("base", None), ("routeB", CKPT_B), ("routeC", CKPT_C)]
    if not args.head and os.path.exists(CKPT_LONG):
        # The 40-epoch run: +1.77 dB / -10.2% lpips vs base, against the
        # 6-epoch route B's +1.00 dB / -6.2%. This is the column that matters.
        variants.append(("long40", CKPT_LONG))
    for tag, ckpt in variants:
        if ckpt is not None:
            load_head(model, ckpt)
        model.eval()
        for i, batch in enumerate(batches):
            color = render(model, batch)
            _imwrite(os.path.join(OUT_, f"s{i}_{tag}.png"), to_png(color[0, 0]))
        print(f"wrote {len(batches)} renders for {tag}", flush=True)

    print(f"\noutput: {OUT_}", flush=True)


def _imwrite(path, arr):
    import cv2
    cv2.imwrite(path, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
