"""Cross-family evaluation: does head fine-tuning generalize, or memorize its domain?

The head-only runs each trained on one dataset family and were scored on
held-out frames of THAT family. `data/common.py: split_train_val` takes the
last 15% of frames *within each sequence*, so validation frames come from
sequences the model trained on -- adjacent-frame leakage is avoided (the split
is contiguous and unshuffled, deliberately) but the scenes themselves are seen.
Every headline number in this project is therefore an IN-DOMAIN adaptation
measurement, not evidence of generalization.

This script separates the two, at zero training cost, by scoring every trained
head on every family's held-out set:

    rows    = checkpoint (base, tum-head, 7-scenes-head, euroc-head)
    columns = family whose val split is being scored

The diagonal reproduces the published per-family numbers. The off-diagonal
answers the question a reviewer asks first: on a family it never trained on,
does a fine-tuned head still beat base, or does it fall below?

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/eval_cross_family.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import exp_head_only as E  # noqa: E402


def use_vgg_lpips(model):
    """Swap the loss's LPIPS trunk to a real VGG.

    `splatt3r_core/main.py` writes `lpips.LPIPS('vgg', spatial=True)`, but the
    package signature is `LPIPS(pretrained=True, net='alex', ...)` -- 'vgg'
    binds to `pretrained` and the trunk stays AlexNet. Every LPIPS number in
    this project is therefore AlexNet-LPIPS, confirmed by `trunk [alex]` in the
    training logs.

    That matters most for the cross-family matrix, where the "fine-tuning is
    actively harmful across domains" reading rests almost entirely on the lpips
    column. One cell is diagnostic: tum-head on EuRoC moves psnr by -0.15 while
    lpips rises 48.5% -- flat pixel error with an exploding perceptual metric is
    the signature of a metric reacting to low-level statistics, not of semantic
    degradation. Re-scoring with the intended VGG trunk decides whether that
    wording survives.
    """
    import lpips as lpips_lib

    model.lpips_criterion = lpips_lib.LPIPS(net="vgg", spatial=True).to(E.DEV)
    return model

FAMILIES = ("tum", "7-scenes", "euroc", "eth3d")
HEADS = {
    "base": None,
    "tum-head": "checkpoints/head_only_long/tum/head_best.pt",
    "7-scenes-head": "checkpoints/head_only_long/7-scenes/head_best.pt",
    "euroc-head": "checkpoints/head_only_long/euroc/head_best.pt",
    "eth3d-head": "checkpoints/head_only_long/eth3d/head_best.pt",
}


def load_head(model, rel):
    sd = torch.load(os.path.join(E.REPO_ROOT, rel), map_location=E.DEV)
    missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
    assert not unexpected and not [k for k in missing if "gaussian_dpt" in k]


def main():
    results = {}
    for fam in FAMILIES:
        E.configure_family(fam, 2, 1e-5)
        _, va = E.build_loaders()
        model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
            E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
        model.decoder.spatial_stride = E.STRIDE
        if os.environ.get("USE_VGG_LPIPS"):
            use_vgg_lpips(model)
        base_sd = {k: v.clone() for k, v in model.encoder.state_dict().items()
                   if "gaussian_dpt" in k}

        for name, rel in HEADS.items():
            if rel is None:
                model.encoder.load_state_dict(base_sd, strict=False)
            else:
                path = os.path.join(E.REPO_ROOT, rel)
                if not os.path.exists(path):
                    print(f"  [skip] {name}: {path} missing", flush=True)
                    continue
                load_head(model, rel)
            print(f"[{fam}] {name}", flush=True)
            E.evaluate(model, va, name)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
