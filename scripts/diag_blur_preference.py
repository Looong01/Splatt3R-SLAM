"""Does the training loss actually prefer a blurrier prediction?

The question
------------
`splatt3r_core/main.py` trains on `1.0*mse + 0.25*lpips`, where the lpips term
is AlexNet-LPIPS (the trunk argument is bound to the wrong parameter, see the
splatt3r-finetuning-experiments skill). LPIPS is known to be sensitive to
sub-pixel misalignment (ST-LPIPS, Ghildyal & Liu ECCV 2022; E-LPIPS, Kettunen
et al. NeurIPS 2019), and our predictions are renderings whose geometry carries
residual registration error against the target view. The hypothesis under test
is whether that combination makes the loss *actively reward blur*, as opposed
to merely failing to reward sharpness.

The distinction matters and is testable with one inequality, no training:

    LPIPS(blur(pred), target) < LPIPS(pred, target)   -> actively rewards blur
    LPIPS(blur(pred), target) > LPIPS(pred, target)   -> the term still pushes
                                                         toward sharpness, it
                                                         is just weakened

Counter-arguments this has to survive (from review): the super-resolution
literature finds that *adding* a perceptual term to L2 reduces blur (Johnson
2016, SRGAN, EnhanceNet), and MSE alone already produces blur by converging to
the conditional mean under irreducible uncertainty -- which two-view NVS has in
abundance (occlusions, unseen content). So a positive result here does not make
LPIPS "the cause" of blur; it would only show the term is pulling the wrong way
on this data.

What is measured, per sample, on the seeded validation draw:

  - mse and lpips of the prediction as-is
  - the same after Gaussian-blurring the PREDICTION (sigma 1 and 2 px)
  - the same after translating the TARGET by 1-3 px, which separates an
    intrinsic blur preference from a misalignment-induced one
  - all of it under both AlexNet and VGG trunks, since AlexNet's stride-4 conv1
    and aggressive pooling alias more than VGG's early layers and should
    therefore show steeper shift sensitivity

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/diag_blur_preference.py [--n 60]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

import exp_head_only as E  # noqa: E402

BLUR_SIGMAS = (1.0, 2.0)
# Symmetric control (review point): is AlexNet-LPIPS's 3x blur sensitivity
# ANTI-BLUR, or merely PRO-high-frequency? If injecting noise raises alex less
# than vgg -- or not at all -- then what it rewards is high-frequency *energy*
# rather than *correct* high frequency, and as a loss it would encourage
# texture noise. Blur and noise are the two opposite deviations from a correct
# image, so the pair separates the two readings.
NOISE_SIGMAS = (0.02, 0.05)
SHIFTS_PX = (0, 1, 2, 3)


def gaussian_blur(x, sigma):
    if sigma <= 0:
        return x
    k = int(2 * round(3 * sigma) + 1)
    g = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    c = x.shape[1]
    x = F.conv2d(x, g.view(1, 1, 1, k).expand(c, 1, 1, k), padding=(0, k // 2), groups=c)
    return F.conv2d(x, g.view(1, 1, k, 1).expand(c, 1, k, 1), padding=(k // 2, 0), groups=c)


def shift(x, px):
    """Translate diagonally by px, replicating the border.

    A real registration error is a small rigid/projective offset; a diagonal
    integer shift is the cheapest stand-in that moves both axes at once.
    """
    if px == 0:
        return x
    return F.pad(x, (px, 0, px, 0), mode="replicate")[..., :-px or None, :-px or None]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--head", default="checkpoints/head_only_long/tum/head_best.pt")
    args = ap.parse_args()

    import lpips as lpips_lib
    import random

    _, va = E.build_loaders()
    model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
        E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
    model.decoder.spatial_stride = E.STRIDE
    hp = os.path.join(E.REPO_ROOT, args.head)
    if os.path.exists(hp):
        model.encoder.load_state_dict(torch.load(hp, map_location=E.DEV), strict=False)
        print(f"loaded head: {args.head}", flush=True)
    model.eval()

    trunks = {n: lpips_lib.LPIPS(net=n).to(E.DEV) for n in ("alex", "vgg")}

    random.seed(E.EVAL_SEED)
    torch.manual_seed(E.EVAL_SEED)

    # acc[(shift, sigma)] = [mse_sum, alex_sum, vgg_sum]
    # key: (shift, kind, amount); kind is "blur" or "noise"
    perturb = [("blur", 0.0)] + [("blur", b) for b in BLUR_SIGMAS] \
              + [("noise", z) for z in NOISE_SIGMAS]
    acc = {(s, k, a): [0.0, 0.0, 0.0] for s in SHIFTS_PX for k, a in perturb}
    n = 0
    for i, batch in enumerate(va):
        if i >= args.n:
            break
        b = E.batch_to_dev(batch)
        v1, v2 = b["context"]
        _, _, h, w = v1["img"].shape
        p1, p2 = model.forward(v1, v2)
        color, _ = model.decoder(b, p1, p2, (h, w))
        pred = color[0, 0][None].clamp(0, 1)
        tgt = b["target"][0]["original_img"][0][None].clamp(0, 1)

        for s in SHIFTS_PX:
            t = shift(tgt, s)
            for kind, amt in perturb:
                if kind == "blur":
                    p = gaussian_blur(pred, amt)
                else:
                    g = torch.Generator(device=pred.device).manual_seed(E.EVAL_SEED + i)
                    p = pred + amt * torch.randn(pred.shape, generator=g,
                                                 device=pred.device)
                # MSE is measured on the UNCLAMPED perturbation. Clamping to
                # [0,1] first makes iid noise appear to *reduce* MSE, which is
                # impossible for zero-mean noise (E[MSE] must rise by sigma^2):
                # for pixels near the bounds -- and TUM has a lot of near-black
                # ones -- the half of the noise pushed out of range is clipped
                # back toward the boundary, i.e. back toward the target, while
                # the error-reducing half survives. An earlier version clamped
                # here and produced "noise at sigma=0.02 improves MSE by 0.70%",
                # read as evidence the prediction was over-smooth. It was an
                # artifact: measured 0.70% vs the iid prediction of +0.67%, and
                # at sigma=0.05 measured +0.34% vs predicted +4.17% -- a gap
                # widening with sigma, the signature of truncation.
                acc[(s, kind, amt)][0] += torch.mean((p - t) ** 2).item()
                # LPIPS needs [0,1] input, so it necessarily sees the clamped
                # version; that only shrinks the perturbation, making the
                # perceptual penalties reported here conservative.
                p = p.clamp(0, 1)
                for j, name in enumerate(("alex", "vgg")):
                    acc[(s, kind, amt)][1 + j] += trunks[name](p, t, normalize=True).mean().item()
        n += 1

    print(f"\nn={n}   prediction blurred with sigma; target shifted by px")
    print(f"{'shift':>6} {'perturb':>10} {'mse':>9} {'alex':>9} {'vgg':>9}   "
          f"{'Δmse%':>8} {'Δalex%':>8} {'Δvgg%':>8}  (Δ vs sigma=0 at same shift)")
    for s in SHIFTS_PX:
        base = [v / n for v in acc[(s, "blur", 0.0)]]
        for kind, amt in perturb:
            v = [x / n for x in acc[(s, kind, amt)]]
            d = [(v[k] - base[k]) / base[k] * 100 for k in range(3)]
            print(f"{s:>6} {kind[:4]:>5}{amt:>5.2f} {v[0]:>9.4f} {v[1]:>9.4f} {v[2]:>9.4f}   "
                  f"{d[0]:>+8.2f} {d[1]:>+8.2f} {d[2]:>+8.2f}")

    print("\nA negative Δ means blurring the prediction IMPROVED that term.")
    print("Shift sensitivity (sigma=0, Δ vs shift=0):")
    z = [v / n for v in acc[(0, "blur", 0.0)]]
    for s in SHIFTS_PX[1:]:
        v = [x / n for x in acc[(s, "blur", 0.0)]]
        print(f"  shift={s}px  mse {(v[0]-z[0])/z[0]*100:+.1f}%   "
              f"alex {(v[1]-z[1])/z[1]*100:+.1f}%   vgg {(v[2]-z[2])/z[2]*100:+.1f}%")


if __name__ == "__main__":
    main()
