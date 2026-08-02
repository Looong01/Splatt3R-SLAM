"""Per-scene 3DGS optimization of a persisted Gaussian map, with a control arm.

Why this exists
---------------
The fine-tuned head produces maps that score ~11-12 dB when re-rendered from
held-out poses (§9.2 of the splatt3r-finetuning-experiments skill). Published
3DGS numbers are 25-35 dB, but those come from *per-scene optimization*:
hundreds of views fitted to one scene for minutes. Splatt3R is feed-forward
from two images, so the two are not comparable tasks. Per-scene optimization on
top of the predicted map is the one route that closes that gap.

It also changes what is being claimed. "Feed-forward prediction, +1.78 dB over
the released checkpoint" becomes "Splatt3R as an initializer for 3DGS", which
is only a result if the initialization actually matters. Hence the control arm:

    --init map     the SLAM map (Splatt3R's prediction) as initialization
    --init random  the same NUMBER of Gaussians, randomly placed in the map's
                   bounding box, random colours, isotropic scales set from the
                   mean nearest-neighbour spacing

Same optimizer, same iteration count, same held-out frames. If both arms land
in the same place, the fine-tuning contributed nothing to the final quality and
the dB gain belongs to the optimizer -- that is a real, publishable negative
result and it must not be discovered after the fact.

What this does NOT do (deliberately, for now)
---------------------------------------------
No densification/pruning (INRIA's adaptive density control). Adding it in the
first version would confound the question: a big gain could then be attributed
to densification rather than to initialization. Pure parameter optimization
first; densification is a separate, later variable.

Parameterization follows INRIA: the free variables are the PRE-activation
values (log scales, logit opacity, raw quaternion, SH DC), so scales stay
positive and opacity stays in (0,1) with no constraints. Per-parameter learning
rates are taken from the reference implementation, with the positional LR
scaled by scene extent as it is there.

Usage:
    python3 scripts/refine_gaussian_map.py \
        --ply logs/head_ate_head/rgbd_dataset_freiburg1_desk_gaussians.ply \
        --traj logs/head_ate_head/rgbd_dataset_freiburg1_desk.txt \
        --dataset datasets/tum/rgbd_dataset_freiburg1_desk \
        --init map --iters 3000
"""
import argparse
import math
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))

import numpy as np
import torch
import torch.nn.functional as F

from eval_map_quality import NEAR, FAR, associate, load_tum_traj, umeyama_sim3
from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply

C0 = 0.28209479177387814

# INRIA 3DGS reference learning rates.
LR_MEANS = 1.6e-4      # scaled by scene extent below
LR_F_DC = 2.5e-3
LR_F_REST = 2.5e-3 / 20.0   # INRIA trains the higher SH bands 20x slower
SH_UPGRADE_INTERVAL = 1000  # raise the active degree every N iterations
LR_OPACITY = 5e-2
LR_SCALE = 5e-3
LR_ROT = 1e-3
DSSIM_WEIGHT = 0.2     # loss = (1-w)*L1 + w*(1-SSIM)


LR_POSE_ROT = 1e-3
LR_POSE_TRANS = 1e-3


def so3_exp(w):
    """Axis-angle (N,3) -> rotation matrices (N,3,3), Rodrigues."""
    theta = w.norm(dim=1, keepdim=True).clamp_min(1e-12)
    k = w / theta
    K = torch.zeros((w.shape[0], 3, 3), device=w.device, dtype=w.dtype)
    K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2];  K[:, 1, 2] = -k[:, 0]
    K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    th = theta[:, :, None]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    return I + th.sin() * K + (1 - th.cos()) * (K @ K)


class PoseDeltas(torch.nn.Module):
    """A learnable 6-DoF correction per supervision view.

    The deployable protocol supervises at SLAM-estimated poses and scores
    13.998 dB; the identical keyframes at ground-truth poses score 15.810.
    That 1.81 dB is pose error, and ground truth is not available at run time.
    Photometric pose refinement -- treating the supervision poses as variables
    initialized at the estimate -- is the standard substitute (MonoGS and
    related online systems do exactly this). Whether it actually recovers that
    1.81 dB here is the one untested assumption in the real-time plan, so this
    exists to measure it offline before any of it is built into the SLAM loop.
    """

    def __init__(self, c2w_init):
        super().__init__()
        self.register_buffer("R0", c2w_init[:, :3, :3].clone())
        self.register_buffer("t0", c2w_init[:, :3, 3].clone())
        n = c2w_init.shape[0]
        self.rot = torch.nn.Parameter(torch.zeros((n, 3), device=c2w_init.device))
        self.trans = torch.nn.Parameter(torch.zeros((n, 3), device=c2w_init.device))

    def delta(self, i):
        """(R, t) of the camera correction, to be applied inversely to the map."""
        R = so3_exp(self.rot[i:i + 1])[0]
        return R, self.trans[i]

    def c2w(self, i):
        R = so3_exp(self.rot[i:i + 1]) @ self.R0[i:i + 1]
        t = self.t0[i:i + 1] + self.trans[i:i + 1]
        T = torch.eye(4, device=R.device, dtype=R.dtype)[None].repeat(1, 1, 1)
        T = torch.cat([torch.cat([R, t[..., None]], dim=2),
                       T[:, 3:, :]], dim=1)
        return T[0]

    def drift(self):
        """Mean translation correction magnitude, for reporting."""
        return self.trans.norm(dim=1).mean().item()


def gaussian_window(size=11, sigma=1.5, device="cuda"):
    g = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(g ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].expand(3, 1, size, size).contiguous()


def ssim(a, b, win):
    pad = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=pad, groups=3)
    mu_b = F.conv2d(b, win, padding=pad, groups=3)
    mu_a2, mu_b2, mu_ab = mu_a ** 2, mu_b ** 2, mu_a * mu_b
    sa = F.conv2d(a * a, win, padding=pad, groups=3) - mu_a2
    sb = F.conv2d(b * b, win, padding=pad, groups=3) - mu_b2
    sab = F.conv2d(a * b, win, padding=pad, groups=3) - mu_ab
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_ab + c1) * (2 * sab + c2)) /
            ((mu_a2 + mu_b2 + c1) * (sa + sb + c2))).mean()


class GaussianModel(torch.nn.Module):
    """Free variables in pre-activation space, exactly as INRIA parameterizes."""

    def __init__(self, means, log_scales, quat_wxyz, logit_opacity, f_dc,
                 sh_degree=0):
        super().__init__()
        self.means = torch.nn.Parameter(means)
        self.log_scales = torch.nn.Parameter(log_scales)
        self.quat = torch.nn.Parameter(quat_wxyz)
        self.logit_opacity = torch.nn.Parameter(logit_opacity)
        self.f_dc = torch.nn.Parameter(f_dc)
        # Higher spherical-harmonic bands. Without them every Gaussian carries
        # ONE colour regardless of viewing direction, so specularities and
        # view-dependent shading cannot be represented at all -- and "the same
        # surface looks different on a revisit" is one of the two problems that
        # started this project (see the splatt3r-color-consistency skill).
        # Degree 3 = 16 coefficients per channel, of which f_dc is the first.
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        n_rest = (sh_degree + 1) ** 2 - 1
        self.f_rest = torch.nn.Parameter(
            torch.zeros((means.shape[0], 3, n_rest), device=means.device)
            if n_rest else torch.zeros((means.shape[0], 3, 0), device=means.device))

    def sh_coeffs(self):
        """(N,3,K) with K = (active_degree+1)^2, DC first."""
        k = (self.active_sh_degree + 1) ** 2 - 1
        if k == 0 or self.f_rest.shape[-1] == 0:
            return self.f_dc[..., None]
        return torch.cat([self.f_dc[..., None], self.f_rest[..., :k]], dim=-1)

    def maybe_upgrade_sh(self, it):
        """INRIA raises the active degree progressively; optimizing every band
        from step 0 is unstable because the high-frequency directions get
        gradient before the DC term has settled."""
        if (self.max_sh_degree and self.active_sh_degree < self.max_sh_degree
                and it > 0 and it % SH_UPGRADE_INTERVAL == 0):
            self.active_sh_degree += 1
            return True
        return False

    def covariances(self):
        from utils.geometry import build_covariance

        q = self.quat / self.quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        # build_covariance takes (x,y,z,w) -- utils/geometry.quaternion_to_matrix
        # unbinds `i, j, k, r` with the comment "Order changed to match scipy
        # format!" -- while the 3DGS .ply format and gaussian_ply_codec both use
        # (w,x,y,z). Feeding the stored order straight in rotates every
        # component by one position, which scrambles each Gaussian's
        # orientation: median 71% relative error against the covariance the
        # decoder computes with scipy.
        #
        # On the INITIAL map this costs almost nothing (12.3719 vs 12.4416
        # scored correctly) because the Gaussians are only mildly anisotropic
        # (median condition number 4.5) and 1.86M of them overlap heavily -- so
        # every refinement result measured before this fix remains valid; the
        # optimizer simply worked in an oddly parameterized but self-consistent
        # frame. It is fatal on SAVE, though: after optimization the
        # orientations carry real information, and the two misreadings do NOT
        # cancel (the codec writes w,x,y,z; this read x,y,z,w), which is the
        # 1.32 dB that a saved map lost on reload.
        q = torch.cat([q[..., 1:4], q[..., 0:1]], dim=-1)
        return build_covariance(self.log_scales.exp(), q)

    def opacity(self):
        return torch.sigmoid(self.logit_opacity)

    def param_groups(self, extent):
        return [
            # INRIA scales the positional LR by scene extent; a map spanning
            # metres and one spanning centimetres otherwise need different LRs.
            {"params": [self.means], "lr": LR_MEANS * extent, "name": "means"},
            {"params": [self.f_dc], "lr": LR_F_DC, "name": "f_dc"},
            {"params": [self.f_rest], "lr": LR_F_REST, "name": "f_rest"},
            {"params": [self.logit_opacity], "lr": LR_OPACITY, "name": "opacity"},
            {"params": [self.log_scales], "lr": LR_SCALE, "name": "scale"},
            {"params": [self.quat], "lr": LR_ROT, "name": "rotation"},
        ]


# --- INRIA adaptive density control ------------------------------------
# Defaults from the reference implementation. Both arms of the factorial use
# the identical schedule, so densification cannot favour one initialization.
DENSIFY_FROM = 500
DENSIFY_UNTIL = 15000
DENSIFY_INTERVAL = 100
DENSIFY_GRAD_THRESHOLD = 2e-4
PERCENT_DENSE = 0.01        # clone below this fraction of scene extent, else split
SPLIT_N = 2
SPLIT_SCALE_DIVISOR = 1.6
PRUNE_MIN_OPACITY = 0.005
OPACITY_RESET_INTERVAL = 3000
# Hard cap, sized from the LARGEST map in use, not the smallest. desk starts at
# 1.86M, but room (7.35M) and 360 (7.27M) both exceed a 6M cap outright, which
# silently forced clone_mask/split_mask empty and turned "densification on" into
# prune-only -- the logs read `clone=0 split=0 pruned=1004` for every round, and
# the room/360 densify cells of the first multi-scene sweep had to be discarded.
MAX_GAUSSIANS = 12_000_000


def _replace_params(model, opt, new_tensors, keep=None, n_new=0):
    """Swap every parameter for a resized version, carrying Adam state along.

    Adam keeps per-element `exp_avg` / `exp_avg_sq`. Assigning a new Parameter
    leaves state of the wrong shape behind, so the state has to be rebuilt with
    **the same index/concat operation that produced the parameters**: surviving
    Gaussians keep their momentum, newly created ones start at zero. That is
    INRIA's optimizer surgery.

    Getting this wrong is not a subtle penalty. An earlier version zeroed
    `exp_avg`/`exp_avg_sq` for *every* Gaussian on *every* densification round
    — i.e. wiped Adam's momentum every 100 iterations — while the no-densify
    control kept its momentum throughout. Every densify cell then read ~5 dB
    below its control at both view counts, with the same rise-to-1500,
    fall-by-3000 shape, which looked like a finding about densification and was
    an artifact of the optimizer being repeatedly reset.

    keep:  bool mask over the concatenated (old ++ new) rows, or None when the
           parameter count is unchanged (e.g. the opacity reset).
    n_new: how many rows were appended after the original block.
    """
    # param_groups() labels the groups for readability ("opacity", "scale",
    # "rotation"); the module attributes they map to are the pre-activation
    # names. Keep the translation in one place.
    attr_of = {"means": "means", "f_dc": "f_dc", "f_rest": "f_rest",
               "opacity": "logit_opacity", "scale": "log_scales",
               "rotation": "quat"}
    for group in opt.param_groups:
        if group["name"] not in attr_of:
            continue  # pose deltas: fixed count, untouched by densification
        name = attr_of[group["name"]]
        old = group["params"][0]
        new = new_tensors[name]
        state = opt.state.pop(old, None)
        p = torch.nn.Parameter(new.contiguous().requires_grad_(True))
        if state is not None:
            if keep is None:
                pass  # same shape; carry momentum through untouched
            else:
                for k in ("exp_avg", "exp_avg_sq"):
                    old_s = state[k]
                    pad = torch.zeros((n_new, *old_s.shape[1:]),
                                      dtype=old_s.dtype, device=old_s.device)
                    state[k] = torch.cat([old_s, pad], dim=0)[keep].contiguous()
            opt.state[p] = state
        group["params"][0] = p
        setattr(model, name, p)
    assert set(attr_of.values()) == set(new_tensors), new_tensors.keys()


def _gather(model, idx):
    return {
        "means": model.means[idx], "f_dc": model.f_dc[idx],
        "f_rest": model.f_rest[idx],
        "logit_opacity": model.logit_opacity[idx],
        "log_scales": model.log_scales[idx], "quat": model.quat[idx],
    }


def densify_and_prune(model, opt, stats, extent, noop=False, mode="full"):
    """One clone/split/prune round. Returns (n_clone, n_split, n_pruned)."""
    grads = stats["grad_accum"] / stats["denom"].clamp_min(1)
    grads = torch.nan_to_num(grads, nan=0.0)
    big = model.log_scales.exp().max(dim=1).values > PERCENT_DENSE * extent
    selected = grads >= DENSIFY_GRAD_THRESHOLD
    if noop or mode == "prune-only":
        selected = torch.zeros_like(selected)

    clone_mask = selected & ~big
    split_mask = selected & big
    room = MAX_GAUSSIANS - model.means.shape[0]
    if room <= 0:
        print(f"    [warn] at the {MAX_GAUSSIANS:,} cap "
              f"({model.means.shape[0]:,} gaussians): clone/split disabled, "
              f"this is prune-only and NOT a test of densification", flush=True)
        clone_mask[:] = False
        split_mask[:] = False

    parts = [{k: v for k, v in _gather(model, slice(None)).items()}]

    if clone_mask.any():
        parts.append(_gather(model, clone_mask))

    if split_mask.any():
        idx = split_mask.nonzero(as_tuple=True)[0]
        scales = model.log_scales[idx].exp()
        for _ in range(SPLIT_N):
            # Sample the offspring's centre from the parent's own covariance,
            # in the parent's local frame -- the split children should cover
            # the volume the parent was covering, not sit on top of it.
            noise = torch.randn_like(scales) * scales
            child = _gather(model, idx)
            child["means"] = child["means"] + noise
            child["log_scales"] = child["log_scales"] - math.log(SPLIT_SCALE_DIVISOR)
            parts.append(child)

    cat = {k: torch.cat([p[k] for p in parts], dim=0) for k in parts[0]}
    n_before = model.means.shape[0]

    # The split parents are replaced by their children.
    keep = torch.ones(cat["means"].shape[0], dtype=torch.bool, device=cat["means"].device)
    keep[:n_before][split_mask] = False
    # Prune transparent Gaussians (INRIA's opacity criterion).
    if not noop and mode != "clone-only":
        keep &= torch.sigmoid(cat["logit_opacity"]) >= PRUNE_MIN_OPACITY

    n_new = cat["means"].shape[0] - n_before
    cat = {k: v[keep] for k, v in cat.items()}
    _replace_params(model, opt, cat, keep=keep, n_new=n_new)
    return (int(clone_mask.sum()), int(split_mask.sum()),
            int((~keep).sum()), model.means.shape[0])


def reset_opacity(model, opt):
    """INRIA periodically caps opacity to force re-justification.

    Without this, floaters that happen to sit in front of the camera keep their
    high opacity forever and never get pruned.
    """
    capped = torch.minimum(model.logit_opacity,
                           torch.full_like(model.logit_opacity, math.log(0.01 / 0.99)))
    new = _gather(model, slice(None))
    new["logit_opacity"] = capped
    _replace_params(model, opt, new)


def uniform_subsample(seq, n):
    """n items spread over the WHOLE of seq, endpoints included.

    Replaces `seq[::max(1, len(seq)//n)][:n]`, which looks uniform and is not.
    That idiom picks a stride by integer division and then truncates, so once n
    exceeds len(seq)/2 the stride collapses to 1 and the result is a contiguous
    PREFIX. Measured on the 284-frame desk candidate pool:

        n=25 -> 93% of the trajectory      n=100 -> 70%
        n=50 -> 87%                        n=150 -> 53%   <- worst
                                           n=250 -> 88%

    This silently confounded the view-count axis with trajectory coverage and
    produced a non-monotone dip at exactly n=150 in BOTH pose modes, which is
    what exposed it -- a real effect has no reason to be worst at one interior
    view count and recover afterwards.
    """
    if n >= len(seq):
        return list(seq)
    idx = np.linspace(0, len(seq) - 1, n).round().astype(int)
    # round() can collide on adjacent indices; dict.fromkeys keeps order.
    return [seq[i] for i in dict.fromkeys(idx.tolist())]


def render(model, c2w, K, hw, device, extras=False, pose_delta=None,
           pose_backend="identity"):
    """`pose_delta` = (R (3,3), t (3,)) -- the CAMERA correction, either way.

    Two backends compute the same gradient:

      identity  applies delta^-1 to the Gaussians and leaves the camera alone,
                which works on an unmodified rasterizer. O(N) transform per
                rendered view per step.
      cuda      composes delta onto the camera and lets the rasterizer's own
                dL/d(viewmatrix), dL/d(projmatrix) and dL/d(campos) carry the
                gradient (see thirdparty/diff-gaussian-rasterization-modified).
                O(1) scene-side, and correct for sh_degree >= 1, which the
                identity is not.

    Validated equal to 7.5e-6 relative at sh_degree=0 by
    scripts/test_camera_gradient.py.

    The compiled rasterizer's backward returns gradients for means3D, means2D,
    sh, colours, opacities, scales and rotations -- and for nothing else. There
    is no dL/dviewmatrix, so making the supervision poses `nn.Parameter`s and
    rendering with them produces exactly zero pose gradient (measured: mean
    translation correction 0.0000 after 1000 iterations).

    But moving the camera by delta and moving every Gaussian by delta^-1 are
    the same rendering, exactly:

        render(X, T @ delta)  ==  render(delta^-1 @ X, T)

    and the second form differentiates, because it only touches means and
    covariances. So pose refinement needs no CUDA change at all -- just a rigid
    transform of the map per supervision view, which is one matmul over 1.86M
    primitives.
    """
    from src.pixelsplat_src.cuda_splatting import render_cuda

    h, w = hw
    extrinsics = torch.as_tensor(c2w, dtype=torch.float32, device=device)[None]
    intrinsics = torch.as_tensor(K, dtype=torch.float32, device=device)[None].clone()
    intrinsics[:, 0, :] /= w
    intrinsics[:, 1, :] /= h

    means, covs = model.means, model.covariances()
    if pose_delta is not None and pose_backend == "cuda":
        # delta composed onto the camera: c2w' = delta @ c2w. Same convention as
        # the branch below, so the two are directly comparable.
        Rd, td = pose_delta
        bot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=Rd.dtype)
        dmat = torch.cat([torch.cat([Rd, td[:, None]], dim=1), bot], dim=0)
        extrinsics = (dmat @ extrinsics[0])[None]
    elif pose_delta is not None:
        # Inverse of the camera delta, applied to the map.
        Rd, td = pose_delta
        Rin = Rd.transpose(0, 1)
        means = (means - td) @ Rin.transpose(0, 1)
        covs = Rin @ covs @ Rin.transpose(0, 1)

    out = render_cuda(
        extrinsics,
        intrinsics,
        torch.full((1,), NEAR, device=device),
        torch.full((1,), FAR, device=device),
        (h, w),
        torch.zeros((1, 3), device=device),
        means[None],
        covs[None],
        model.sh_coeffs()[None],
        model.opacity()[None],
        use_sh=True,
        return_extras=extras,
    )
    if extras:
        img, radii, mean_grads = out
        return img.reshape(1, 3, h, w), radii[0], mean_grads[0]
    return out.reshape(1, 3, h, w)


def build_random_init(g, device, n_points=None):
    """Control arm: random geometry, no information from Splatt3R.

    `n_points=None` matches the map's Gaussian COUNT. That looked like the fair
    comparison and is not: filling a room-sized bounding box with 7.35M
    Gaussians at opacity 0.1 saturates alpha within the first few of the
    hundreds crossed by any ray, so only a thin front shell ever receives
    gradient. Measured (logs/probe_clone.log): after 900 iterations at LR 5e-2
    the opacity was still 0.1002 and the scale still 0.03852, with p99 equal to
    the mean -- the field never moved off its initialization. Two independent
    runs then ended 0.001 dB apart despite a 0.66 dB seed sigma, because the
    outcome was set by the initialization and not by the optimization.

    Standard 3DGS initializes ~100k points -- two orders of magnitude fewer --
    precisely so the field is not saturated, and *grows* it with densification.
    Matching the count instead of the regime produced an unoptimizable fog and
    invalidated every map-vs-random comparison built on it.

    Pass `n_points` (with --densify) for the real recipe.
    """
    n = n_points or g["n"]
    lo = g["means"].min(0).values
    hi = g["means"].max(0).values
    means = torch.rand((n, 3), device=device) * (hi - lo) + lo
    vol = float((hi - lo).prod().clamp_min(1e-9))
    spacing = (vol / max(n, 1)) ** (1.0 / 3.0)
    log_scales = torch.full((n, 3), math.log(max(spacing, 1e-6)), device=device)
    quat = torch.zeros((n, 4), device=device)
    quat[:, 0] = 1.0
    logit_opacity = torch.full((n,), math.log(0.1 / 0.9), device=device)
    f_dc = (torch.rand((n, 3), device=device) - 0.5) / C0
    return means, log_scales, quat, logit_opacity, f_dc


@torch.no_grad()
def score(model, frames, K, device, lp):
    model.eval()
    tm = tl = 0.0
    for c2w, target in frames:
        pred = render(model, c2w, K, target.shape[-2:], device).clamp(0, 1)
        tm += torch.mean((pred - target) ** 2).item()
        tl += lp(pred, target, normalize=True).mean().item()
    model.train()
    n = len(frames)
    mse = tm / n
    return mse, -10 * math.log10(max(mse, 1e-12)), tl / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True)
    ap.add_argument("--traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--init", choices=("map", "random"), default="map")
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--train-source", choices=("keyframes", "all"), default="keyframes",
                    help="'keyframes' reproduces the first experiment (~14 views on "
                         "desk); 'all' adds non-keyframes under ORACLE ground-truth "
                         "poses -- see the note in main()")
    ap.add_argument("--tag", default="", help="label for the log line")
    ap.add_argument("--densify", action="store_true",
                    help="enable INRIA adaptive density control. The 2x2 "
                         "factorial {map,random} x {off,on} needs all four "
                         "cells: random-init without densification is a known "
                         "handicap, so the off-only comparison overstates how "
                         "much the initialization contributes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--optimize-poses", action="store_true",
                    help="treat the supervision poses as variables, initialized "
                         "at the SLAM estimate. Measured at ~17 iterations/sec "
                         "on desk (1.86M gaussians) against a ~61 s sequence, an "
                         "online background optimizer can afford roughly 300-1500 "
                         "iterations -- so this is only interesting inside that "
                         "budget, not at the 9000-30000 that reach 19.9 dB "
                         "offline.")
    ap.add_argument("--alternate", type=int, default=0,
                    help="alternate between map-only and pose-only phases every "
                         "N iterations. 0 = joint (the old behaviour, which the "
                         "injection test shows suppresses pose recovery to "
                         "8-15% against 41-58% pose-only).")
    ap.add_argument("--freeze-map", action="store_true",
                    help="optimize the poses only, holding every Gaussian "
                         "parameter fixed. Separates 'the poses cannot be "
                         "recovered' from 'the map absorbed the error first'.")
    ap.add_argument("--perturb-poses", type=float, default=0.0,
                    help="inject a known rigid perturbation (translation sigma "
                         "in metres, rotation scaled to match the real "
                         "error ratio) into the supervision poses before "
                         "optimizing them, and report how much of it is "
                         "recovered. Run on a map optimized at ground-truth "
                         "poses, this separates 'the method is intrinsically "
                         "weak' from 'the SLAM map is co-adapted to its own "
                         "pose errors'.")
    ap.add_argument("--save-poses", default=None,
                    help="npz with the initial and optimized c2w of every "
                         "supervision view, for scoring the correction against "
                         "ground truth offline.")
    ap.add_argument("--pose-lr", type=float, default=None,
                    help="learning rate for the 6-DoF pose corrections. The "
                         "default (1e-3) is 2.07x the map's positional LR "
                         "(1.6e-4 * extent = 4.84e-4 on desk), where the "
                         "standard advice for joint pose/map optimization is "
                         "1-2 ORDERS smaller. Suspected cause of the monotone "
                         "lpips degradation with view count in the "
                         "pose-optimized arm (0.426 at 14 views -> 0.527 at "
                         "250) while psnr stays flat: weakly-constrained pose "
                         "parameters absorbing photometric noise.")
    ap.add_argument("--pose-backend", choices=("identity", "cuda"), default="identity",
                    help="how --optimize-poses gets its gradient. 'identity' "
                         "transforms the Gaussians by delta^-1 (works on a "
                         "stock rasterizer, O(N) per view per step); 'cuda' "
                         "composes delta onto the camera and uses the "
                         "rasterizer's camera gradients (O(1) scene-side, and "
                         "the only correct one at sh_degree >= 1).")
    ap.add_argument("--frames-traj", default=None,
                    help="<seq>_frames.txt from main.py: estimated poses for "
                         "every tracked frame, not just keyframes. With "
                         "--deployable this replaces the ~14-keyframe "
                         "supervision set with --n-train views drawn from it, "
                         "still using no ground truth. Held-out frames are "
                         "excluded first.")
    ap.add_argument("--kf-gt-poses", action="store_true",
                    help="with --deployable: supervise on the SAME keyframes "
                         "but at GROUND-TRUTH poses. Isolates pose quality from "
                         "view count -- the two things that separate the "
                         "deployable 13.998 from the oracle 19.08, and which "
                         "were initially (wrongly) attributed entirely to view "
                         "count. Not deployable; a decomposition probe.")
    ap.add_argument("--deployable", action="store_true",
                    help="supervise ONLY with what a shipped system has: the "
                         "keyframes the SLAM run actually selected, at the "
                         "poses it actually estimated. The default protocol "
                         "uses ground-truth poses and non-keyframes, which is "
                         "an oracle upper bound -- useful for locating the "
                         "ceiling, not for claiming a deliverable. The gap "
                         "between the two is the part of the reported gain that "
                         "cannot be shipped.")
    ap.add_argument("--sh-degree", type=int, default=0,
                    help="spherical-harmonic degree for the refined map. 0 (the "
                         "default so far) gives every Gaussian a single "
                         "view-independent colour; 3 is the 3DGS standard and "
                         "is what lets a surface change appearance with viewing "
                         "angle.")
    ap.add_argument("--save-ply", default=None,
                    help="write the refined map as a 3DGS .ply. Nothing has "
                         "ever persisted an optimized map, so the 19 dB result "
                         "has never been looked at as an image and its GEOMETRY "
                         "has never been measured -- and per-scene refinement "
                         "optimizes a photometric loss only, so it is free to "
                         "trade geometry for appearance (floaters that render "
                         "correctly from the training views but sit in the "
                         "wrong place). base->finetuned was checked for exactly "
                         "this (depth L1 -8..27%%, coverage -1pp); "
                         "12.37->19.11 was not.")
    ap.add_argument("--n-random", type=int, default=None,
                    help="Gaussian count for --init random. Default matches "
                         "the map's count, which is DEGENERATE at these sizes "
                         "(see build_random_init). Use ~100000 with --densify "
                         "for the standard 3DGS regime.")
    ap.add_argument("--densify-mode", default="full",
                    choices=("full", "clone-only", "prune-only"),
                    help="isolate which structural operation carries the "
                         "effect. On room/n=150/random, full densification "
                         "cost 4.9 dB while touching 0.4% of the Gaussians, "
                         "and the A/A control cleared the code path -- so the "
                         "damage is in clone/split or in prune, and guessing "
                         "which has already gone wrong three times here")
    ap.add_argument("--densify-noop", action="store_true",
                    help="A/A control: run the ENTIRE densify code path "
                         "(extras rendering, gradient accumulation, "
                         "_replace_params every interval) but with clone, "
                         "split and prune all disabled, so the Gaussian set is "
                         "unchanged. Any difference from --densify off is "
                         "attributable to the path, not to densification. "
                         "Added because room/n=150/random lost 4.57 dB while "
                         "its Gaussian count moved by 0.4% -- a degradation "
                         "that structural change cannot explain")
    ap.add_argument("--n-held", type=int, default=50)
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import lpips as lpips_lib
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img

    load_config(args.config)
    dev = args.device

    dataset = load_dataset(args.dataset)
    ds_ts = np.array([float(t) for t in dataset.timestamps])
    est_ts, est_T = load_tum_traj(args.traj)
    gt_ts, gt_T = load_tum_traj(os.path.join(args.dataset, "groundtruth.txt"))

    pairs = associate(est_ts, gt_ts)
    s, R, t = umeyama_sim3(
        np.array([est_T[i, :3, 3] for i, _ in pairs]),
        np.array([gt_T[j, :3, 3] for _, j in pairs]),
    )
    Rt = R.T

    def to_map(c2w_gt):
        m = np.eye(4)
        m[:3, :3] = Rt @ c2w_gt[:3, :3]
        m[:3, 3] = Rt @ (c2w_gt[:3, 3] - t) / s
        return m

    # Held-out frames are carved out FIRST and never touched again -- not by
    # optimization and not by the Sim3 fit -- so growing the training set
    # cannot eat into them.
    #
    # Training views are then drawn from everything else. With --train-source
    # keyframes that is the 14-ish frames the SLAM run selected, which is what
    # the first experiment used and what made it plateau by iteration 1500 on
    # a training loss still falling 4x. With `all`, non-keyframes join too.
    #
    # ORACLE PROTOCOL: non-keyframe supervision uses GROUND-TRUTH poses, since
    # save_traj only persists keyframes. The map was built online from
    # *estimated* poses, so this mixes initialization quality with pose
    # accuracy and is an upper bound, not a deployable pipeline. It is run
    # this way deliberately -- the question here is how far a given
    # initialization can be refined, not what a shipped system would achieve --
    # but it must be labelled as an oracle wherever it is reported.
    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    non_kf = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    # NOT switched to uniform_subsample: this defines the held-out test set that
    # every previously recorded number was scored against (12.372 init, 13.998
    # baseline, 15.810 GT ceiling, ...). Changing it would silently invalidate
    # every cross-run comparison in the skill. At n_held=50 out of 599 the
    # stride is 11 and the span is 90%, so the bias is small; it is frozen for
    # comparability, not because it is ideal.
    held_c = non_kf[:: max(1, len(non_kf) // args.n_held)][: args.n_held]
    held_set = {i for i, _ in held_c}

    pool = [(i, j) for i, j in gt_pairs
            if i not in held_set and (args.train_source == "all" or i in kf_idx)]
    frm_ts = frm_T = None
    if args.deployable and args.frames_traj:
        # DEPLOYABLE + unlocked views: supervision from every tracked frame, at
        # the pose SLAM itself estimated for it (main.py's <seq>_frames.txt,
        # written by evaluate.FramePoseLog). No ground truth anywhere on this
        # path -- unlike --train-source all, which pairs non-keyframes with
        # GROUND-TRUTH poses and is therefore an oracle.
        #
        # These poses live in the map's own frame already (they are composed
        # from the same keyframe poses the map was baked from), so they get no
        # Sim3 mapping, exactly like the --deployable keyframe path.
        frm_ts, frm_T = load_tum_traj(args.frames_traj)
        fr_pairs = associate(frm_ts, ds_ts)
        # Held-out frames are non-keyframes, and so is most of this file --
        # without this filter the training set would swallow the test set and
        # every number below would be meaningless.
        cand = [(fi, di) for fi, di in fr_pairs if di not in held_set]
        sel = uniform_subsample(cand, args.n_train)
        train_c = [("FRAME", fi) for fi, _di in sel]
        n_kf_sel = sum(1 for _fi, di in sel if di in kf_idx)
        print(f"frames-traj: {len(frm_ts)} poses, {len(fr_pairs)} associated, "
              f"{len(cand)} after held-out filter, {len(sel)} sampled "
              f"({n_kf_sel} of them keyframes)", flush=True)
    elif args.deployable and args.kf_gt_poses:
        # Same keyframes, ground-truth poses: pair each keyframe timestamp with
        # its ground-truth pose, mapped into the map frame like any other GT
        # pose. Difference from --deployable alone is the pose source only.
        kf_gt = associate(est_ts, gt_ts)
        train_c = [("KFGT", j) for _i, j in kf_gt]
    elif args.deployable:
        # est_T is already in the map's own frame -- the map was baked from
        # these very poses -- so no Sim3 mapping is applied to them, unlike the
        # ground-truth path below.
        train_c = [("EST", i) for i in range(len(est_ts))]
    else:
        train_c = uniform_subsample(pool, args.n_train)
    print(f"train frames={len(train_c)} (source={args.train_source}, "
          f"pool={len(pool)})  held-out={len(held_c)}", flush=True)

    K = dataset.camera_intrinsics.K_frame

    def load_frames(cands):
        out = []
        for di, gj in cands:
            if di == "KFGT":
                k = associate(np.array([gt_ts[gj]]), ds_ts)
                if not k:
                    continue
                img = resize_img(dataset.get_image(k[0][1]), dataset.img_size)["img"]
                target = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
                out.append((to_map(gt_T[gj]), target))
                continue
            if di == "FRAME":
                # gj indexes the frames trajectory; its pose is already in the
                # map frame, so it is used raw (no to_map()).
                k = associate(np.array([frm_ts[gj]]), ds_ts)
                if not k:
                    continue
                img = resize_img(dataset.get_image(k[0][1]), dataset.img_size)["img"]
                target = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
                out.append((frm_T[gj], target))
                continue
            if di == "EST":
                # Keyframe supervision at the estimated pose. Find the dataset
                # frame whose timestamp matches this keyframe.
                k = associate(np.array([est_ts[gj]]), ds_ts)
                if not k:
                    continue
                img = resize_img(dataset.get_image(k[0][1]), dataset.img_size)["img"]
                target = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
                out.append((est_T[gj], target))
                continue
            img = resize_img(dataset.get_image(di), dataset.img_size)["img"]
            # resize_img returns ImgNorm'd [-1,1]; the rasterizer emits [0,1].
            target = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
            out.append((to_map(gt_T[gj]), target))
        return out

    train_frames = load_frames(train_c)
    held_frames = load_frames(held_c)

    g = decode_gaussians_from_ply(args.ply, device=dev)
    print(f"map: {g['n']:,} gaussians  init={args.init}", flush=True)

    if args.init == "map":
        init = (g["means"], g["log_scales"], g["quat_wxyz"], g["logit_opacity"], g["f_dc"])
    else:
        init = build_random_init(g, dev, args.n_random)

    model = GaussianModel(*[x.clone().float() for x in init],
                          sh_degree=args.sh_degree).to(dev)
    extent = float((g["means"].max(0).values - g["means"].min(0).values).norm() / 2)
    # --freeze-map: the map has ~1.86M x 14 free parameters against the poses'
    # 50 x 6. When both move, the map can absorb a pose perturbation far faster
    # than the poses can undo it -- which is the leading explanation for the
    # controlled injection test recovering nothing even though the photometric
    # loss is steeply sensitive to displacement (scripts/diag_pose_observability
    # measures +19% mse at 5 mm, +141% at 2 cm). Freezing the map isolates the
    # pose problem from that race.
    groups = [] if args.freeze_map else model.param_groups(extent)
    opt = torch.optim.Adam(groups if groups else [
        {"params": [model.means], "lr": 0.0, "name": "means"}], eps=1e-15)
    poses = None
    true_c2w = None
    if args.optimize_poses:
        init_c2w = torch.stack([torch.as_tensor(c, dtype=torch.float32, device=dev)
                                for c, _t in train_frames])
        if args.perturb_poses > 0:
            # CONTROLLED RECOVERY TEST. Everything measured so far conflates two
            # things: whether photometric pose refinement CAN recover a known
            # pose error, and whether it can do so from an initialization where
            # the map was built with those very errors baked in (the SLAM map is
            # co-adapted to its own estimated poses -- the starting point sits in
            # a basin that is self-consistent with them).
            #
            # Running this on a map optimized at GROUND-TRUTH poses, then
            # injecting a known perturbation of the same magnitude as the real
            # estimation error, separates them: high recovery means the method
            # works and the real-world 5.2% is a co-adaptation/basin problem;
            # low recovery means photometric pose observability is intrinsically
            # poor on this scene and no amount of tuning will open the 4.5 dB
            # gate -- which is the case for moving the work into the SLAM backend.
            true_c2w = init_c2w.clone()
            g = torch.Generator(device="cpu").manual_seed(args.seed + 1000)
            n = init_c2w.shape[0]
            dt = torch.randn((n, 3), generator=g) * args.perturb_poses
            # Rotation perturbed proportionally: the real estimate's rotation
            # error is ~1.7 deg against ~0.024 m of translation error.
            dr = torch.randn((n, 3), generator=g) * (args.perturb_poses * 1.24)
            R = so3_exp(dr.to(dev))
            init_c2w = init_c2w.clone()
            init_c2w[:, :3, :3] = R @ init_c2w[:, :3, :3]
            init_c2w[:, :3, 3] = init_c2w[:, :3, 3] + dt.to(dev)
            # The perturbation has to reach the RENDERER, not just PoseDeltas'
            # R0/t0 buffers -- delta(i) is built from rot/trans alone and never
            # touches them, so an earlier version of this left the injected
            # error in an unused buffer while the training loop kept rendering
            # from the original poses. The optimizer then started at the optimum
            # with no gradient, wandered by ~0.01 m, and the "recovery" metric
            # scored that wandering against a perturbation that was never
            # applied -- reporting zero recovery from an experiment that did
            # nothing at all.
            train_frames = [(init_c2w[i].detach().cpu().numpy(), train_frames[i][1])
                            for i in range(len(train_frames))]
            err0 = (init_c2w[:, :3, 3] - true_c2w[:, :3, 3]).norm(dim=1)
            print(f"  INJECTED pose perturbation: mean |dt| = {err0.mean():.4f} m "
                  f"over {n} views (sigma={args.perturb_poses})", flush=True)
        poses = PoseDeltas(init_c2w).to(dev)
        lr_pose = args.pose_lr if args.pose_lr is not None else None
        opt.add_param_group({"params": [poses.rot],
                             "lr": lr_pose if lr_pose is not None else LR_POSE_ROT,
                             "name": "pose_rot"})
        opt.add_param_group({"params": [poses.trans],
                             "lr": lr_pose if lr_pose is not None else LR_POSE_TRANS,
                             "name": "pose_trans"})
        print(f"  pose LR = {lr_pose if lr_pose is not None else LR_POSE_ROT:.1e} "
              f"(map positional LR = {LR_MEANS * extent:.2e})", flush=True)
        print(f"  optimizing {init_c2w.shape[0]} supervision poses", flush=True)

    win = gaussian_window(device=dev)
    lp = lpips_lib.LPIPS(net="alex").to(dev)

    mse, psnr, lpv = score(model, held_frames, K, dev, lp)
    def save_refined(path):
        from splatt3r_slam.gaussian_ply_codec import encode_gaussians_for_ply
        from plyfile import PlyData, PlyElement

        with torch.no_grad():
            cov = model.covariances().double().cpu()
            row, col = torch.triu_indices(3, 3)
            attrs = encode_gaussians_for_ply(
                model.means.detach().cpu(), cov[:, row, col],
                # NO clamp. encode_gaussians_for_ply inverts SH2RGB, so this
                # round-trips f_dc exactly -- but only if it is not clipped on
                # the way out. f_dc is a free pre-activation parameter and the
                # optimizer routinely drives it outside the [0,1] RGB box:
                # measured 2.9% of channels pinned at a bound, costing 1.32 dB
                # (a map scoring 19.0375 reloaded at 17.7225). The clamp came
                # from the SLAM-export path, where colours are genuinely
                # in-range; for a refined map it is simply wrong.
                (model.f_dc.detach().cpu() * C0 + 0.5),
                model.opacity().detach().cpu())
        names = (["x", "y", "z", "nx", "ny", "nz"]
                 + [f"f_dc_{i}" for i in range(3)] + ["opacity"]
                 + [f"scale_{i}" for i in range(3)] + [f"rot_{i}" for i in range(4)])
        el = np.empty(attrs.shape[0], dtype=[(n_, "f4") for n_ in names])
        for i, nm in enumerate(names):
            el[nm] = attrs[:, i]
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        PlyData([PlyElement.describe(el, "vertex")]).write(path)
        print(f"  saved refined map: {path} ({attrs.shape[0]:,} gaussians)", flush=True)

    tag = f"[{args.tag}] " if args.tag else ""
    print(f"\n  {tag}iter     0 | held-out psnr={psnr:7.4f}  lpips={lpv:.4f}  (init)", flush=True)

    densify_on = args.densify or args.densify_noop
    rng = np.random.default_rng(args.seed)
    n_g = model.means.shape[0]
    stats = {"grad_accum": torch.zeros(n_g, device=dev),
             "denom": torch.zeros(n_g, device=dev)}

    # --alternate: the controlled injection test showed that a FREE map absorbs
    # an injected pose error before the poses can undo it -- recovery 8-15%
    # jointly against 41-58% with the map frozen. Every deployable number in
    # this project was measured jointly, so its pose optimization was running in
    # the suppressed regime. Alternating gives the poses phases in which they
    # are the only thing that can explain the residual.
    map_names = {"means", "f_dc", "f_rest", "opacity", "scale", "rotation"}
    base_lr = {g["name"]: g["lr"] for g in opt.param_groups}

    def set_phase(pose_phase):
        for gp in opt.param_groups:
            is_pose = gp["name"].startswith("pose")
            gp["lr"] = base_lr[gp["name"]] if (is_pose == pose_phase) else 0.0

    for it in range(1, args.iters + 1):
        if args.alternate > 0 and poses is not None:
            set_phase(((it - 1) // args.alternate) % 2 == 1)
        fi = int(rng.integers(len(train_frames)))
        c2w, target = train_frames[fi]
        pdelta = poses.delta(fi) if poses is not None else None
        pbk = args.pose_backend
        if densify_on:
            pred, radii, mean_grads = render(
                model, c2w, K, target.shape[-2:], dev, extras=True,
                pose_delta=pdelta, pose_backend=pbk)
        else:
            pred = render(model, c2w, K, target.shape[-2:], dev,
                          pose_delta=pdelta, pose_backend=pbk)
            radii = mean_grads = None
        l1 = (pred - target).abs().mean()
        loss = (1 - DSSIM_WEIGHT) * l1 + DSSIM_WEIGHT * (1 - ssim(pred.clamp(0, 1), target, win))

        opt.zero_grad(set_to_none=True)
        loss.backward()

        if densify_on:
            # Accumulate the screen-space position gradient for VISIBLE
            # Gaussians only (radii > 0). Averaging over all Gaussians instead
            # would dilute the signal by however many happen to be off-screen
            # in this view, which varies wildly frame to frame.
            with torch.no_grad():
                vis = radii > 0
                g = mean_grads.grad
                if g is not None:
                    stats["grad_accum"][vis] += g[vis, :2].norm(dim=-1)
                    stats["denom"][vis] += 1

        opt.step()

        if densify_on and DENSIFY_FROM <= it <= DENSIFY_UNTIL and it % DENSIFY_INTERVAL == 0:
            nc, ns, npr, ntot = densify_and_prune(model, opt, stats, extent,
                                                  noop=args.densify_noop,
                                                  mode=args.densify_mode)
            stats = {"grad_accum": torch.zeros(ntot, device=dev),
                     "denom": torch.zeros(ntot, device=dev)}
            with torch.no_grad():
                # Distribution probes. Two independent clone-only runs ended at
                # 8.6284 and 8.6294 dB -- a 0.001 dB match when the arm's seed
                # sigma is 0.66 dB, which is the signature of collapse onto a
                # single degenerate state rather than a quality tradeoff. Train
                # loss also rose (0.29 vs 0.125 for prune-only), so the model
                # fits its own training views worse. These probes are here to
                # show WHICH quantity runs away.
                def q99(x):
                    # torch.quantile caps out around 16M elements and raises
                    # "input tensor is too large"; log_scales.exp() on a 7.35M
                    # Gaussian map is 22M. Subsample instead -- a 1M-element
                    # sample pins the 99th percentile far tighter than the
                    # precision this probe needs.
                    f = x.reshape(-1)
                    if f.numel() > 1_000_000:
                        f = f[torch.randint(f.numel(), (1_000_000,), device=f.device)]
                    return f.quantile(0.99)

                op = model.opacity()
                sc = model.log_scales.exp()
                print(f"  {tag}iter {it:5} | densify clone={nc} split={ns} "
                      f"pruned={npr} -> {ntot:,} gaussians | "
                      f"opacity mean={op.mean():.4f} p99={q99(op):.4f} "
                      f"frac>0.9={(op > 0.9).float().mean()*100:.2f}% | "
                      f"scale mean={sc.mean():.5f} p99={q99(sc):.5f}",
                      flush=True)

        if model.maybe_upgrade_sh(it):
            print(f"  {tag}iter {it:5} | SH degree -> {model.active_sh_degree}",
                  flush=True)

        if it % args.eval_every == 0 or it == args.iters:
            mse, psnr, lpv = score(model, held_frames, K, dev, lp)
            mem = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"  {tag}iter {it:5} | held-out psnr={psnr:7.4f}  lpips={lpv:.4f}  "
                  f"train_loss={loss.item():.4f}  peak={mem:.1f}GiB", flush=True)

        # The opacity reset runs AFTER the evaluation, and never on the final
        # iteration. It caps every Gaussian's opacity at 0.01 to force floaters
        # to re-justify themselves over the iterations that follow; scoring a
        # map in that state measures a deliberately crippled model, not a
        # result. An earlier version reset before evaluating, and since the
        # reset interval (3000) equalled the run length (3000), every densify
        # cell reported its final number from a near-transparent map -- 7.18 dB
        # at iteration 3000 where iteration 1500 had scored 14.42.
        if (densify_on and not args.densify_noop and it < args.iters
                and it % OPACITY_RESET_INTERVAL == 0):
            reset_opacity(model, opt)
            print(f"  {tag}iter {it:5} | opacity reset", flush=True)


    if poses is not None:
        print(f"  mean pose translation correction: {poses.drift():.4f}", flush=True)

    if poses is not None and true_c2w is not None:
        with torch.no_grad():
            # The optimized camera is delta @ c2w_init, so its centre is
            # R_delta @ t_init + t_delta (the same convention pinned by
            # scripts/test_camera_gradient.py PART 2).
            opt_t = torch.stack([
                poses.delta(i)[0] @ init_c2w[i, :3, 3] + poses.delta(i)[1]
                for i in range(len(train_frames))])
            e_before = (init_c2w[:, :3, 3] - true_c2w[:, :3, 3]).norm(dim=1)
            e_after = (opt_t - true_c2w[:, :3, 3]).norm(dim=1)
        rec = 1.0 - e_after.mean().item() / e_before.mean().item()
        print(f"  RECOVERY: injected {e_before.mean():.4f} m -> residual "
              f"{e_after.mean():.4f} m  = {100*rec:.1f}% recovered "
              f"({100*(e_after < e_before).float().mean():.0f}% of views improved)",
              flush=True)

    if poses is not None and args.save_poses:
        # Persist initial and optimized camera poses so the correction can be
        # scored against ground truth offline (scripts/diag_pose_correction.py).
        # `mean drift` alone cannot distinguish a pose refiner from a
        # photometric sponge: both move the cameras, only one moves them
        # towards the truth.
        with torch.no_grad():
            init = torch.stack([torch.as_tensor(c, dtype=torch.float32)
                                for c, _ in train_frames])
            opt_c2w = []
            for i in range(len(train_frames)):
                Rd, td = poses.delta(i)
                bot = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=Rd.device)
                dmat = torch.cat([torch.cat([Rd, td[:, None]], dim=1), bot], dim=0)
                opt_c2w.append((dmat.cpu() @ init[i]))
            opt_c2w = torch.stack(opt_c2w)
        os.makedirs(os.path.dirname(args.save_poses) or ".", exist_ok=True)
        np.savez(args.save_poses,
                 init=init.numpy(), opt=opt_c2w.numpy(),
                 kind=np.array([k for k, _ in train_c], dtype=object),
                 ref=np.array([v for _, v in train_c]),
                 tag=args.tag)
        print(f"  saved {len(opt_c2w)} pose pairs -> {args.save_poses}", flush=True)

    if args.save_ply:
        save_refined(args.save_ply)


if __name__ == "__main__":
    main()
