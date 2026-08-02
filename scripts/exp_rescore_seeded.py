"""Re-score the head-only run's saved checkpoint against base on the SAME
fixed sample draw.

Why this exists
---------------
Every earlier number in this experiment series was measured on a *different*
draw of validation samples. DUST3RSplattingDataset.__getitem__ re-samples its
context/target triplet on every call (unseeded random.randint/choice/sample,
data/data.py:132,143,170), so pinning the index list pinned only the indices,
not their contents. Three separate evaluations of the *identical* base
checkpoint scored:

    psnr 9.7268  / lpips 0.2801     (head-only run, unseeded draw)
    psnr 9.8648  / lpips 0.2856     (lpips run, unseeded draw)
    psnr 10.3221 / lpips 0.2793     (lpips run, seeded draw)

-- a ~0.6 dB spread from sampling alone. The head-only run's headline result
(+1.11 dB over base) was one draw against another draw, so it is not
trustworthy at that magnitude. This script settles it: base and the trained
head are scored on the byte-identical 150 triplets, so any difference is
attributable to the weights and nothing else.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/exp_rescore_seeded.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import exp_head_only as E  # noqa: E402  (does sys.path/chdir setup on import)

HEAD_CKPT = os.path.join(E.REPO_ROOT, "checkpoints", "head_only", "tum", "head_best.pt")


def main():
    _, va = E.build_loaders()
    print(f"val samples={len(va.dataset)}  res={E.RES}  stride={E.STRIDE}  "
          f"seed={E.EVAL_SEED}\n", flush=True)

    model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
        E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
    model.decoder.spatial_stride = E.STRIDE

    print("=== base checkpoint, seeded draw ===", flush=True)
    E.evaluate(model, va, "BASE")

    sd = torch.load(HEAD_CKPT, map_location=E.DEV)
    missing, unexpected = model.encoder.load_state_dict(sd, strict=False)
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
    assert all("gaussian_dpt" not in k for k in missing), "a head key failed to load"
    print(f"\nloaded {len(sd)} head tensors from {HEAD_CKPT}", flush=True)

    print("=== head-only trained checkpoint, SAME seeded draw ===", flush=True)
    E.evaluate(model, va, "HEAD")


if __name__ == "__main__":
    main()
