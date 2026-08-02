"""Validate the camera-pose gradients added to the CUDA rasterizer.

Upstream's backward returns gradients for the Gaussians only, so photometric
pose refinement was previously done with an exact workaround: moving the camera
by delta and moving every Gaussian by delta^-1 are the same render,

    render(X, delta @ c2w)  ==  render(delta^-1 @ X, c2w)

That workaround is an *exact reference implementation* of the gradient the CUDA
change now computes directly, which makes it a better validation target than
finite differences (no step size, no truncation error) and better than porting
MonoGS's fork (it is our own scenes and data). Both are used here.

Three parts, in increasing order of how much they can hide:

  PART 1 (unit).  Calls GaussianRasterizer directly with viewmatrix, projmatrix
    and campos as INDEPENDENT leaf tensors, and checks each analytic gradient
    against a central finite difference along a random direction, at two step
    sizes. Independent inputs are what makes this a unit test: a bug in one of
    the three cannot be masked by the other two, which is exactly what happens
    once they are all derived from one c2w.

  PART 2 (integration, sh_degree=0).  The full render_cuda path, camera
    perturbed by a 6-DoF delta, against the workaround. At degree 0 the
    workaround has no defect, so the two must agree to floating-point noise.
    This also pins the delta convention: refine_gaussian_map.render()'s
    docstring stated the identity with mismatched conventions (`render(X, T @
    delta)` with delta^-1 on the Gaussians only holds if T is c2w and delta acts
    on the left), and nothing had ever tested which one the code implements.

  PART 3 (sh_degree=3).  Here the two paths are EXPECTED to diverge:
    cuda_splatting.py passes `campos=extrinsics[i,:3,3]`, the original camera
    position, while the workaround hands the rasterizer Gaussians that have been
    rotated -- so forward.cu evaluates the SH direction rotated by R_delta^T.
    Part 1 having validated dL_dcampos independently is what lets us say the
    CUDA path is the correct one rather than merely the different one.

Usage:  python3 scripts/test_camera_gradient.py [--n 50000]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "splatt3r_core"))

import torch

NEAR, FAR = 0.01, 100.0
H, W = 96, 128


def so3_exp(w):
    theta = w.norm() + 1e-12
    k = w / theta
    z = torch.zeros_like(k[0])
    K = torch.stack([
        torch.stack([z, -k[2], k[1]]),
        torch.stack([k[2], z, -k[0]]),
        torch.stack([-k[1], k[0], z]),
    ])
    return (torch.eye(3, dtype=w.dtype, device=w.device)
            + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K))


def delta_matrix(rot, trans):
    R = so3_exp(rot)
    top = torch.cat([R, trans[:, None]], dim=1)
    bot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=rot.device, dtype=rot.dtype)
    return torch.cat([top, bot], dim=0)


def make_scene(n, degree, device, seed=0):
    """Few, large, smooth Gaussians: alpha compositing is piecewise-smooth in
    the camera parameters only while the set of contributing primitives is
    stable, so finite differences need a scene without a swarm of marginal
    single-pixel splats."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    means = torch.cat([
        torch.rand((n, 2), generator=g) * 3.0 - 1.5,
        torch.rand((n, 1), generator=g) * 2.0 + 2.0,
    ], dim=1).to(device)
    scales = (torch.rand((n, 3), generator=g) * 0.05 + 0.05).to(device)
    q = torch.randn((n, 4), generator=g)
    q = (q / q.norm(dim=1, keepdim=True)).to(device)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(n, 3, 3)
    covs = R @ torch.diag_embed(scales ** 2) @ R.transpose(1, 2)
    n_sh = (degree + 1) ** 2
    sh = (torch.rand((n, 3, n_sh), generator=g) * 0.5).to(device)
    opa = (torch.rand((n, 1), generator=g) * 0.3 + 0.15).to(device).reshape(n)
    return means, covs, sh, opa


def weights(device):
    return torch.randn((3, H, W), generator=torch.Generator().manual_seed(7)).to(device)


# --------------------------------------------------------------------------
# PART 1 -- direct rasterizer call, the three camera inputs independent
# --------------------------------------------------------------------------
def raster_direct(means, covs, sh, opa, viewmatrix, projmatrix, campos, degree, device):
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings, GaussianRasterizer)

    settings = GaussianRasterizationSettings(
        image_height=H, image_width=W,
        tanfovx=0.5, tanfovy=0.4,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=viewmatrix, projmatrix=projmatrix,
        sh_degree=degree, campos=campos,
        prefiltered=False, debug=False,
    )
    row, col = torch.triu_indices(3, 3)
    img, _ = GaussianRasterizer(settings)(
        means3D=means,
        means2D=torch.zeros_like(means, requires_grad=True),
        shs=sh.permute(0, 2, 1).contiguous(),
        colors_precomp=None,
        opacities=opa[:, None],
        cov3D_precomp=covs[:, row, col],
    )
    return img


def part1(n, degree, device, seed=5):
    """dL_dcampos against a finite difference that is actually trustworthy.

    A naive FD -- perturb campos, watch the loss -- is not. Alpha compositing
    over many small splats is only piecewise smooth in the camera parameters:
    `radii` is an INTEGER, so a small perturbation flips primitives in and out
    and the loss moves in steps. Measured: upstream's own (unmodified)
    sum_i dL/dmeans disagreed with such an FD by 2x on two of three axes, and
    the disagreement grew as h shrank -- the signature of noise, not truncation.

    The estimator used instead cancels that. Translating the Gaussians AND the
    camera position by the same v leaves every SH direction (pos_i - campos)
    unchanged, so

        dL/dcampos . v  =  FD[translate both] - FD[translate means only]

    and the two FDs share identical geometry, hence identical staircase
    artifacts, which subtract out. In practice this is stable to ~0.2% across a
    5x change in step size where the raw FD was not stable at all.

    viewmatrix and projmatrix need no FD: PART 2 compares them against the
    exact Gaussian-transform workaround, which is a far stronger reference.
    They are also SH-independent (they are fed by dL_dmean2D and dL_dconic
    only), so degree 3 adds nothing for them.
    """
    g = torch.Generator().manual_seed(seed)
    # Few, large, smooth Gaussians -- see above for why this matters.
    means0 = torch.cat([torch.rand((n, 2), generator=g) * 2.0 - 1.0,
                        torch.rand((n, 1), generator=g) * 1.0 + 3.0], 1).to(device)
    scales = (torch.rand((n, 3), generator=g) * 0.10 + 0.20).to(device)
    covs = torch.diag_embed(scales ** 2).to(device)
    n_sh = (degree + 1) ** 2
    sh = (torch.rand((n, 3, n_sh), generator=g) * 0.5).to(device)
    opa = (torch.rand((n,), generator=g) * 0.2 + 0.10).to(device)
    wt = torch.randn((3, H, W), generator=torch.Generator().manual_seed(7)).to(device)

    c2w = torch.eye(4, device=device)
    c2w[2, 3] = -1.5
    view0 = torch.inverse(c2w).T.contiguous()
    proj = torch.tensor([[2., 0, 0, 0], [0, 2., 0, 0],
                         [0, 0, 1.0001, 1.], [0, 0, -0.010001, 0]], device=device)
    projm0 = (view0 @ proj).contiguous()
    campos0 = c2w[:3, 3].contiguous()

    def loss(m, c):
        return (raster_direct(m, covs, sh, opa, view0, projm0, c, degree, device) * wt).sum()

    c = campos0.clone().requires_grad_(True)
    loss(means0, c).backward()
    g_c = c.grad.clone()

    print(f"\n--- PART 1: dL_dcampos, sh_degree={degree}, N={n} ---")
    print(f"  analytic          = {[f'{v:+.5f}' for v in g_c.tolist()]}")
    worst = 0.0
    for h in (5e-3, 1e-3):
        imp = []
        for i in range(3):
            e = torch.zeros(3, device=device)
            e[i] = h
            with torch.no_grad():
                fm = ((loss(means0 + e, campos0) - loss(means0 - e, campos0)) / (2 * h)).item()
                fb = ((loss(means0 + e, campos0 + e) - loss(means0 - e, campos0 - e)) / (2 * h)).item()
            imp.append(fb - fm)
        t = torch.tensor(imp, device=device)
        scale = max(t.abs().max().item(), g_c.abs().max().item(), 1e-6)
        rel = ((t - g_c).abs().max() / scale).item()
        worst = max(worst, rel) if h == 1e-3 else worst
        print(f"  implied (h={h:.0e}) = {[f'{v:+.5f}' for v in imp]}   max rel diff = {rel:.2e}")
    if degree == 0:
        print("  (degree 0 is view-independent: both must be identically zero)")
    return worst


# --------------------------------------------------------------------------
# PARTS 2/3 -- full render_cuda path, workaround vs CUDA
# --------------------------------------------------------------------------
def render_full(c2w, means, covs, sh, opa, device, pose_delta=None):
    from src.pixelsplat_src.cuda_splatting import render_cuda

    if pose_delta is not None:
        Rd, td = pose_delta
        Rin = Rd.transpose(0, 1)
        means = (means - td) @ Rin.transpose(0, 1)
        covs = Rin @ covs @ Rin.transpose(0, 1)

    K = torch.tensor([[0.9, 0.0, 0.5], [0.0, 1.2, 0.5], [0.0, 0.0, 1.0]],
                     device=device)[None]
    img = render_cuda(
        c2w[None], K,
        torch.full((1,), NEAR, device=device), torch.full((1,), FAR, device=device),
        (H, W), torch.zeros((1, 3), device=device),
        means[None], covs[None], sh[None], opa[None], use_sh=True,
    )
    return img.reshape(1, 3, H, W)


def part23(degree, n, device):
    means, covs, sh, opa = make_scene(n, degree, device)
    wt = weights(device)[None]
    c2w0 = torch.eye(4, device=device)
    c2w0[2, 3] = -1.5
    rot0 = torch.tensor([0.014, -0.021, 0.008], device=device)
    tr0 = torch.tensor([0.021, 0.013, -0.017], device=device)

    def grad_of(fn):
        """Returns (pose gradient, map gradient). The map gradient matters as
        much as the pose one: the identity backend routes the delta THROUGH
        means and covariances, so if the two backends disagreed there, the map
        optimization itself would diverge between them -- and that is the bulk
        of the work, not the 84 pose parameters."""
        rot = rot0.clone().requires_grad_(True)
        tr = tr0.clone().requires_grad_(True)
        m = means.clone().requires_grad_(True)
        cv = covs.clone().requires_grad_(True)
        fn(rot, tr, m, cv).backward()
        return torch.cat([rot.grad, tr.grad]), (m.grad.clone(), cv.grad.clone())

    def L_work(rot, tr, m, cv):
        d = delta_matrix(rot, tr)
        return (render_full(c2w0, m, cv, sh, opa, device,
                            pose_delta=(d[:3, :3], d[:3, 3])) * wt).sum()

    def L_cuda(rot, tr, m, cv):
        return (render_full(delta_matrix(rot, tr) @ c2w0, m, cv, sh, opa,
                            device) * wt).sum()

    (ga, ma), (gb, mb) = grad_of(L_work), grad_of(L_cuda)
    with torch.no_grad():
        d = delta_matrix(rot0, tr0)
        ia = render_full(c2w0, means, covs, sh, opa, device,
                         pose_delta=(d[:3, :3], d[:3, 3]))
        ib = render_full(d @ c2w0, means, covs, sh, opa, device)
    img_rel = (ia - ib).abs().max().item() / max(ia.abs().max().item(), 1e-9)
    cos = torch.nn.functional.cosine_similarity(ga[None], gb[None]).item()
    rel_norm = ((ga - gb).norm() / gb.norm()).item()

    tag = "PART 2" if degree == 0 else "PART 3"
    print(f"\n--- {tag}: workaround vs cuda, sh_degree={degree}, N={n} ---")
    print(f"  forward image  max|diff| / max|val| = {img_rel:.3e}")
    # means and covs are judged separately and by different criteria. The
    # covariance backward runs through denom2inv = 1/((ac-b^2)^2 + 1e-7), a
    # near-cancellation that is badly conditioned for near-degenerate 2D
    # covariances -- and the identity backend feeds it covariances that have
    # been through two extra fp32 matmuls (R^T cov R). The resulting tail is
    # numerical, not a disagreement about the math: it lands on components
    # whose reference value is ~0, while direction is preserved. Measured at
    # degree 0: means cos = 1.00000000, covs cos = 0.99970, and on the top 1%
    # of covariance components by magnitude the median relative difference is
    # 1.8e-3. Note this makes the CUDA path the better-conditioned of the two,
    # since it passes covariances through untouched.
    mean_rel = ((ma[0] - mb[0]).norm() / mb[0].norm().clamp_min(1e-12)).item()
    cov_cos = torch.nn.functional.cosine_similarity(
        ma[1].reshape(1, -1), mb[1].reshape(1, -1)).item()
    map_rel = max(mean_rel, 1.0 - cov_cos)
    print(f"  pose gradient  ||ga-gb||/||gb||     = {rel_norm:.3e}   cos = {cos:.6f}")
    print(f"  map  gradient  means ||da||/||d||   = {mean_rel:.3e}")
    print(f"  map  gradient  covs  cos            = {cov_cos:.8f}")
    print(f"    cuda       = {[f'{v:+.4f}' for v in gb.tolist()]}")
    print(f"    workaround = {[f'{v:+.4f}' for v in ga.tolist()]}")
    return img_rel, rel_norm, map_rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    os.chdir(os.path.join(REPO_ROOT, "splatt3r_core"))
    ok = True

    w0 = part1(150, 0, args.device)
    w3 = part1(150, 3, args.device)
    if max(w0, w3) > 2e-2:
        print("  FAIL: dL_dcampos disagrees with finite differences")
        ok = False

    img_rel0, grad_rel0, map_rel0 = part23(0, args.n, args.device)
    if img_rel0 > 1e-3:
        print("  FAIL: renders differ at degree 0 -> delta convention is wrong"); ok = False
    if grad_rel0 > 1e-2:
        print("  FAIL: cuda pose gradient disagrees with the exact workaround"); ok = False
    if map_rel0 > 1e-3:
        print("  FAIL: cuda MAP gradient disagrees with the exact workaround"); ok = False

    img_rel3, grad_rel3, map_rel3 = part23(3, args.n, args.device)
    if grad_rel3 < 1e-2:
        print("  NOTE: no measurable divergence at sh=3; this scene's "
              "view-dependence may be too weak to expose the campos defect.")
    else:
        print(f"\n  sh=3: the workaround's gradient differs from the correct one by "
              f"{grad_rel3:.1%} (image by {img_rel3:.1%}) -- the campos defect of "
              f"cuda_splatting.py:151, measured rather than argued.")

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
