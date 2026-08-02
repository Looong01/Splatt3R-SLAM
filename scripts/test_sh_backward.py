"""Close the `sh[14]` bug class: check EVERY spherical-harmonic coefficient.

The sh[14] defect was of the kind "a hand port of the forward polynomial missed
one term". Verifying the six terms that *were* changed is the wrong response to
that; the right one is to check all 16 coefficients at every degree, which is
what this does. (Suggested by review, and cheap enough that there is no excuse
not to make it a permanent gate.)

Two independent chains have to be right, and they fail differently:

  dL_dsh[k]      the gradient w.r.t. each coefficient. Tested by plain central
                 differences on the coefficient itself -- and here a plain FD
                 IS trustworthy, unlike anywhere else in this rasterizer,
                 because an SH coefficient changes only colour. Geometry, radii
                 and tile assignment are untouched, so the staircase that makes
                 FD useless for camera/mean gradients (see
                 test_camera_gradient.py) simply is not present.

  dRGBd{x,y,z}   the derivative w.r.t. the view direction, which feeds
                 dL_dmeans and dL_dcampos. Tested through dL_dcampos with the
                 translation-cancellation estimator, since campos enters ONLY
                 through the direction -- so a defect anywhere in that chain
                 shows up here. This is the test that originally caught
                 sh[14].

Together they cover both ways a mis-ported term can hide. Run at degrees 1, 2
and 3: a term is only exercised at its own degree and above.

Usage:  python3 scripts/test_sh_backward.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "splatt3r_core"))

import torch

import test_camera_gradient as T


def build(n, degree, device, seed=5):
    g = torch.Generator().manual_seed(seed)
    means = torch.cat([torch.rand((n, 2), generator=g) * 2.0 - 1.0,
                       torch.rand((n, 1), generator=g) * 1.0 + 3.0], 1).to(device)
    scales = (torch.rand((n, 3), generator=g) * 0.10 + 0.20).to(device)
    covs = torch.diag_embed(scales ** 2).to(device)
    n_sh = (degree + 1) ** 2
    sh = (torch.rand((n, 3, n_sh), generator=g) * 0.5).to(device)
    opa = (torch.rand((n,), generator=g) * 0.2 + 0.10).to(device)
    wt = torch.randn((3, T.H, T.W), generator=torch.Generator().manual_seed(7)).to(device)
    c2w = torch.eye(4, device=device)
    c2w[2, 3] = -1.5
    view = torch.inverse(c2w).T.contiguous()
    proj = torch.tensor([[2., 0, 0, 0], [0, 2., 0, 0],
                         [0, 0, 1.0001, 1.], [0, 0, -0.010001, 0]], device=device)
    return means, covs, sh, opa, wt, view, (view @ proj).contiguous(), c2w[:3, 3].contiguous()


def run_degree(degree, n, device):
    means, covs, sh, opa, wt, view, projm, campos = build(n, degree, device)
    n_sh = (degree + 1) ** 2

    def loss(m, s, c):
        return (T.raster_direct(m, covs, s, opa, view, projm, c, degree, device) * wt).sum()

    s = sh.clone().requires_grad_(True)
    c = campos.clone().requires_grad_(True)
    loss(means, s, c).backward()
    g_sh, g_c = s.grad.clone(), c.grad.clone()

    print(f"\n=== sh_degree={degree}  ({n_sh} coefficients, N={n}) ===")
    worst_k, worst = -1, 0.0
    h = 2e-3
    for k in range(n_sh):
        # Perturb coefficient k of every Gaussian, all 3 channels, together:
        # the directional derivative is then sum_i sum_ch dL/dsh[i,ch,k].
        d = torch.zeros_like(sh)
        d[:, :, k] = h
        with torch.no_grad():
            fd = ((loss(means, sh + d, campos) - loss(means, sh - d, campos)) / (2 * h)).item()
        an = g_sh[:, :, k].sum().item()
        rel = abs(fd - an) / max(abs(fd), abs(an), 1e-6)
        flag = "   <-- MISMATCH" if rel > 2e-2 else ""
        print(f"  sh[{k:2d}]  analytic={an:+12.4f}  fd={fd:+12.4f}  rel={rel:.2e}{flag}")
        if rel > worst:
            worst, worst_k = rel, k

    # Direction chain, via campos (see module docstring).
    imp = []
    for i in range(3):
        e = torch.zeros(3, device=device)
        e[i] = h
        with torch.no_grad():
            fm = ((loss(means + e, sh, campos) - loss(means - e, sh, campos)) / (2 * h)).item()
            fb = ((loss(means + e, sh, campos + e)
                   - loss(means - e, sh, campos - e)) / (2 * h)).item()
        imp.append(fb - fm)
    t = torch.tensor(imp, device=device)
    scale = max(t.abs().max().item(), g_c.abs().max().item(), 1e-6)
    dir_rel = ((t - g_c).abs().max() / scale).item()
    print(f"  direction chain (via dL_dcampos): analytic="
          f"{[f'{v:+.4f}' for v in g_c.tolist()]}  implied={[f'{v:+.4f}' for v in imp]}"
          f"  rel={dir_rel:.2e}")
    print(f"  worst coefficient: sh[{worst_k}] at {worst:.2e}")
    return worst, dir_rel


def main():
    device = "cuda"
    os.chdir(os.path.join(REPO_ROOT, "splatt3r_core"))
    T.H, T.W = 256, 256
    ok = True
    for degree in (1, 2, 3):
        w, d = run_degree(degree, 150, device)
        if w > 2e-2:
            print(f"  FAIL: a coefficient gradient is wrong at degree {degree}")
            ok = False
        if d > 3e-2:
            print(f"  FAIL: the view-direction chain is wrong at degree {degree}")
            ok = False
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
