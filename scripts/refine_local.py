"""Refinement on a TRAJECTORY-ANCHORED map — stages 1-3 of the online plan.

This is `refine_gaussian_map.py`'s job done on the parameterization the online
system needs. The difference is not cosmetic: the offline script optimizes a
world-space map baked from the SLAM run's estimated poses, and that bake is what
creates co-adaptation. Measured, same perturbation and protocol, differing only
in which poses the map was built from (skill 13.12b):

    map built at ground-truth poses   +51.8% of an injected error recovered
    map built at SLAM poses           -22.2%   -- pushed FURTHER from truth

Here the Gaussians stay in their owning keyframe's camera frame and are composed
through that keyframe's *current* pose on every render, so a pose correction
re-deforms the map instead of invalidating what was learned.

Stages, each checkable on its own:

  --stage 1   world-space control. Flattens the keyframe Gaussians once, exactly
              like the offline script, and optimizes that. Exists only to prove
              the plumbing here reproduces the known offline number before the
              parameterization changes underneath it.
  --stage 2   local parameterization, poses fixed. MUST match stage 1 within
              noise; if it does not, the composition is wrong and
              scripts/test_camera_gradient.py is the reference.
  --stage 3   local parameterization plus a synthetic pose correction injected
              mid-run, to verify the map follows a loop closure rigidly. This
              is the property no offline result has ever had. Three modes:
              'global' (self-consistent control), 'perkf' (iid per-keyframe,
              supervision frozen -- the unfaithful version), 'block' (one Sim3
              to the second half of the trajectory, supervision views carried
              by their anchor keyframes -- the faithful seam test, item b',
              with a three-way held-out split: low-only / high-only / seam).

Usage:
    python3 scripts/refine_local.py --kfgauss logs/frames_head/<seq>_kfgauss.pt \\
        --traj logs/frames_head/<seq>.txt --frames-traj logs/frames_head/<seq>_frames.txt \\
        --dataset datasets/tum/<seq> --stage 2 --n-train 50 --iters 3000
"""
import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))

import numpy as np
import torch
import torch.nn.functional as F

from eval_map_quality import NEAR, FAR, associate, load_tum_traj, umeyama_sim3

DSSIM_WEIGHT = 0.2


def render(means_w, cov_w, rgb, opacity, c2w, K, hw, device):
    from src.pixelsplat_src.cuda_splatting import render_cuda
    h, w = hw
    ext = torch.as_tensor(c2w, dtype=torch.float32, device=device)[None]
    intr = torch.as_tensor(K, dtype=torch.float32, device=device)[None].clone()
    intr[:, 0, :] /= w
    intr[:, 1, :] /= h
    row, col = torch.triu_indices(3, 3)
    img = render_cuda(
        ext, intr,
        torch.full((1,), NEAR, device=device), torch.full((1,), FAR, device=device),
        (h, w), torch.zeros((1, 3), device=device),
        means_w[None], cov_w[None],
        ((rgb - 0.5) / 0.28209479177387814)[:, :, None][None],
        opacity.reshape(-1)[None],
        use_sh=True)
    return img.reshape(1, 3, h, w)


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kfgauss", required=True, help="<seq>_kfgauss.pt from main.py")
    ap.add_argument("--traj", required=True)
    ap.add_argument("--frames-traj", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--stage", type=int, default=2, choices=(1, 2, 3))
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--n-held", type=int, default=50)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--perturb-kf-at", type=int, default=1500,
                    help="stage 3: iteration at which to inject a synthetic "
                         "keyframe-pose correction")
    ap.add_argument("--perturb-kf-sigma", type=float, default=0.05)
    ap.add_argument("--perturb-kf-rot", type=float, default=0.087,
                    help="stage 3: rotation sigma in radians (~5 deg)")
    ap.add_argument("--perturb-kf-scale", type=float, default=0.05,
                    help="stage 3: Sim3 scale sigma. A loop closure is a "
                         "similarity, and the scale path (cov -> s^2) is "
                         "otherwise never exercised.")
    ap.add_argument("--n-perturb", type=int, default=10,
                    help="stage 3: how many successive corrections to apply. "
                         "One passing does not mean ten do -- incremental "
                         "write-back bugs accumulate per event.")
    ap.add_argument("--reanchor", choices=("none", "gt"), default="none",
                    help="replace every keyframe's estimated anchor pose with "
                         "its ground-truth pose before optimizing. Measures how "
                         "much of the pose gap is placement (recoverable by "
                         "re-anchoring) versus baked into the per-keyframe "
                         "prediction itself (not recoverable).")
    ap.add_argument("--perturb-mode", choices=("global", "perkf", "block"), default="global",
                    help="stage 3: 'global' applies one Sim3 to every keyframe "
                         "AND the cameras -- self-consistent, so it cannot tear "
                         "a seam. 'perkf' gives each keyframe its own "
                         "correction, which is what a pose-graph solve produces "
                         "and what can actually split a shared surface. 'block' "
                         "applies one Sim3 to the SECOND HALF of the trajectory "
                         "(a loop-closure-like block correction) AND moves each "
                         "supervision view with its anchor keyframe -- the "
                         "faithful version of the seam test (item b').")
    ap.add_argument("--freeze-supervision", action="store_true",
                    help="stage 3 block mode only: do NOT move supervision "
                         "views with their anchors. The unfaithful control arm "
                         "-- the real system never does this (FramePoseLog "
                         "recomputes anchored poses), so any extra damage here "
                         "measures the value of the faithful treatment.")
    ap.add_argument("--dedup-voxel", type=float, default=0.0,
                    help="de-clustering ablation (gates item c): before "
                         "optimization, quantize world-space means into voxels "
                         "of this edge length; where a voxel holds Gaussians "
                         "from >=2 keyframes, keep only the EARLIEST "
                         "keyframe's and delete the rest. Tests the "
                         "two-shells explanation of the overlap-quality "
                         "anomaly (skill 13.14).")
    ap.add_argument("--min-confidence", type=float, default=1.5,
                    help="injection-density knob. The map is one Gaussian per "
                         "pixel per keyframe -- 46 x 512x384 = 9.0M raw on 360, "
                         "7.27M after the default filtering, against 1-2M for a "
                         "comparable room in standard 3DGS. Raising this keeps "
                         "fewer, more confident Gaussians. Per-step cost is "
                         "linear in the count, so the honest comparison between "
                         "densities is at matched WALL CLOCK, not matched "
                         "iterations: a sparser map is worse per step and gets "
                         "more steps for the same seconds.")
    ap.add_argument("--aa-sigma", type=float, default=0.0, metavar="TAU",
                    help="3D smoothing filter at injection: sigma = TAU * the "
                         "local lattice pitch (distance to the nearest 4-neighbour "
                         "on the source pixel grid). Closes the inter-Gaussian "
                         "gaps that read as a dot lattice / moire from any view "
                         "other than the source. 0.5 puts the midpoint between "
                         "two neighbours at 1 sigma. 0 = off.")
    ap.add_argument("--lr-means", type=float, default=1.6e-4)
    ap.add_argument("--lr-f-dc", type=float, default=2.5e-3)
    ap.add_argument("--lr-opacity", type=float, default=5e-2)
    ap.add_argument("--lr-scale", type=float, default=5e-3)
    ap.add_argument("--lr-rot", type=float, default=1e-3)
    ap.add_argument("--kf-pose-lr", type=float, default=0.0, metavar="LR",
                    help="free the keyframe TRANSLATIONS. 17.48 showed a "
                         "per-cluster scale correction applied alone makes the "
                         "render worse even when it makes the geometry 6.4x "
                         "more accurate, because SLAM fitted each pose to that "
                         "keyframe's own biased pointmap and the error is "
                         "common-mode with the pose. Solving the two together "
                         "is the only form in which the correction can pay.")
    ap.add_argument("--referee-scales", default=None, metavar="NPY",
                    help="per-keyframe corrections from diag_referee.py, used "
                         "as a PRIOR on kf_log_depth rather than as an edit")
    ap.add_argument("--referee-weight", type=float, default=0.0, metavar="W",
                    help="weight of the referee prior")
    ap.add_argument("--kf-depth-lr", type=float, default=0.0,
                    help="learnable per-keyframe log depth scale: slides each "
                         "cluster along its own view rays. Targets the seams, "
                         "which 17.4 re-diagnosed as a misplaced cluster "
                         "floating as a veil, not a colour step. One scalar per "
                         "keyframe. 0 = off.")
    ap.add_argument("--ss-loss", type=int, default=1, metavar="N",
                    help="render the supervision view at Nx and average-pool "
                         "back to native before the loss. The rasterizer point-"
                         "samples at pixel centres, so at native rate the loss "
                         "is blind to everything between the sample points -- "
                         "including the gaps between Gaussians, which is why "
                         "the optimizer is free to open them. Pooling an Nx "
                         "render is area integration by Monte Carlo: the same "
                         "effect as a 2D Mip filter, with no CUDA change. Costs "
                         "N^2 rasterizer pixels, and the rasterizer is ~10%% of "
                         "a culled step.")
    ap.add_argument("--freeze-means-after", type=int, default=0, metavar="N",
                    help="run with the positional rate live for N steps, then "
                         "set it to zero. The hypothesis (Kimi, round 4) is "
                         "that the positions' one remaining job once the band "
                         "limit is a constraint is fixing real geometry error, "
                         "which needs DIRECTED motion and is only available "
                         "early while the gradient carries more signal than "
                         "noise; after that the same rate is pure jitter. "
                         "0 = off.")
    ap.add_argument("--freeze-opacity-after", type=int, default=0, metavar="N",
                    help="same, for opacity. Identifies the slow second carrier "
                         "that still lifts hp_alpha from 0.0016 to 0.0037-0.0073 "
                         "once the means are frozen.")
    ap.add_argument("--consistency", type=float, default=0.0, metavar="W",
                    help="weight on the cross-cluster consistency loss: two "
                         "keyframes' Gaussians in one voxel are penalized for "
                         "disagreeing about position and colour. The only "
                         "repair route for the seams that needs no external "
                         "truth (17.29), and the only signal that can tell "
                         "--kf-depth-lr which way to slide a cluster, which "
                         "photometry provably cannot (17.16, 17.17).")
    ap.add_argument("--consistency-voxel", type=float, default=0.02,
                    help="voxel size for the consistency term, metres. The "
                         "scale that matters is the DISAGREEMENT scale, not the "
                         "lattice pitch: a veil sits 4-10 cm in front of the "
                         "true surface (17.16's +-2-5%% at 2 m), so a voxel "
                         "sized off the 2 mm pitch puts the two layers in "
                         "different voxels and never compares them. Default was "
                         "0.02 for exactly that wrong reason and measured "
                         "-1..-3%% on the warp deficit; see the sweep in 17.30.")
    ap.add_argument("--consistency-rgb", type=float, default=1.0,
                    help="relative weight of the colour term inside the "
                         "consistency loss; 0 makes it purely geometric")
    ap.add_argument("--pitch-lr", action="store_true",
                    help="scale the positional learning rate by each Gaussian's "
                         "own lattice pitch instead of by scene extent, via a "
                         "dimensionless residual parameterization. One step of "
                         "--lr-means then displaces a Gaussian by that fraction "
                         "of its own spacing. Adam is scale-invariant in the "
                         "gradient, so this cannot be done by reweighting "
                         "gradients. Requires --aa-sigma/--aa-hard-floor.")
    ap.add_argument("--dither", type=float, default=0.0, metavar="K",
                    help="jitter each Gaussian by K*pitch IN THE PLANE "
                         "perpendicular to its own view ray. Kimi round 27's "
                         "third arm against the transparency lattice: the "
                         "artifact is a coherent modulation of the weight "
                         "field, so breaking the centres' lattice coherence "
                         "should scatter it into incoherent grain. It is "
                         "explicitly NOT judged by hp_alpha (which measures "
                         "energy, not coherence) -- it passes or fails on "
                         "stress-view images.")
    ap.add_argument("--scale-cap", type=float, default=0.0, metavar="M",
                    help="clamp every Gaussian scale to at most M metres. The "
                         "other half of Kimi's round-24 decomposition: the "
                         "Replica head's baked regression comes with a scale "
                         "tail (p90 +133%%, max axis +100%%) as well as a "
                         "saturated opacity, and only a 2x2 separates which "
                         "one costs the dB. Cap at the BASE head's p90.")
    ap.add_argument("--uniform-fade", type=float, default=0.0, metavar="D",
                    help="THE CONTROL FOR --streak-opacity (Kimi, round 21). "
                         "The elongation selector fires on 64-98%% of all "
                         "Gaussians (17.53), so the criterion may be doing no "
                         "work at all: multiply EVERY opacity by (1-D) instead, "
                         "with D matched to the selector's mean deficit. If "
                         "uniform ~= selected, the lever is a global thinning "
                         "prior and the elongation framing is wrong.")
    ap.add_argument("--conf-fade", type=float, default=0.0, metavar="D",
                    help="17.66.4's confidence-weighted thinning, the arm for "
                         "--uniform-fade. Same mean thinning, but SPENT WHERE "
                         "IT IS EARNED: opacity *= 1 - D*2*(1-conf_norm), with "
                         "conf_norm the per-keyframe rank of the backbone's own "
                         "confidence in [0,1]. Motivated by 17.66: on the TUM "
                         "head, opacity's strongest surviving predictor is that "
                         "confidence (+0.213 partial on depth), so a trained "
                         "head already encodes this and an untrained one (base, "
                         "Replica) does not. The factor 2 makes the MEAN "
                         "multiplier (1-D), matching --uniform-fade D, so the "
                         "two arms differ in allocation and not in dose -- "
                         "without that they would not be comparable at all.")
    ap.add_argument("--crowd-fade", type=float, default=0.0, metavar="T",
                    help="crowding-weighted thinning: target T units of "
                         "accumulated alpha per occupied voxel, "
                         "opacity *= clamp(T/sum(opacity in voxel), 0.1, 1). "
                         "Kimi's round-32 lever, and the sharpest consequence "
                         "of the account in 17.71.7 -- if two thirds of "
                         "conf-fade's value is reaching a lower alpha budget, "
                         "then confidence is a PROXY and this targets the map "
                         "property directly. Absolute units, no rank "
                         "normalization: crowding is geometric and comparable "
                         "across keyframes, unlike the confidence head. "
                         "Combinable with --conf-fade to test whether "
                         "confidence carries anything beyond geometry.")
    ap.add_argument("--crowd-voxel", type=float, default=0.0, metavar="M",
                    help="voxel edge in metres for --crowd-fade. 0 (default) "
                         "picks 4x the median Gaussian extent, so a voxel holds "
                         "the Gaussians that genuinely overlap.")
    ap.add_argument("--streak-opacity", type=float, default=0.0, metavar="K",
                    help="hide ray-elongated Gaussians instead of shrinking "
                         "them: opacity *= min(1, K*pitch/max_scale). Clamping "
                         "the long axis was harmful at every setting (17.16) "
                         "because it shrinks the footprint; this leaves the "
                         "footprint alone and lets the surface behind show "
                         "through. K ~ 1.5 is the suggested start. Watch the "
                         "black fraction: a rise >1%% means there was nothing "
                         "behind after all.")
    ap.add_argument("--max-anisotropy", type=float, default=0.0,
                    help="clamp each Gaussian's long axis to at most this many "
                         "times its short axis at injection. Targets the "
                         "trailing streaks at depth edges: uncertainty along "
                         "the ray makes the head emit a needle pointing at the "
                         "camera, which max_scale=0.5 does not catch because "
                         "its absolute size is fine. Clamped, never deleted -- "
                         "deleting leaves holes at silhouettes.")
    ap.add_argument("--aa-hard-floor", action="store_true",
                    help="hold the --aa-sigma band limit as a per-Gaussian "
                         "CONSTRAINT (added in quadrature at every forward) "
                         "instead of folding it into the initial scales. The "
                         "difference only shows up once the optimizer runs: "
                         "folded in, it is an initial condition Adam may walk "
                         "off, since (scale down, opacity up) is nearly "
                         "loss-preserving under point-sampled rasterization.")
    ap.add_argument("--aa-compensate-opacity", action="store_true",
                    help="Mip-Splatting's energy correction alongside --aa-sigma. "
                         "Off by default: the perforation IS an alpha deficit, "
                         "and this gives back exactly what closes it.")
    ap.add_argument("--exposure", action="store_true",
                    help="per-supervision-frame affine colour correction "
                         "(per-channel gain+bias, 6 params/frame) applied to the "
                         "RENDER before the loss. TUM freiburg1 runs the camera's "
                         "auto-exposure, so one surface is baked at different "
                         "brightnesses by different keyframes and the seam is "
                         "visible; without this the L1 term can only average the "
                         "conflict, which blurs instead of reconciling.")
    ap.add_argument("--exposure-lr", type=float, default=1e-3)
    ap.add_argument("--exposure-recenter", action="store_true",
                    help="fold the mean exposure back into f_dc at the end. "
                         "MEASURED HARMFUL (17.8): costs a further 0.106 dB raw "
                         "psnr and moves fit psnr by 0.005, because the mean of "
                         "the per-frame gains is an artifact of the training set "
                         "rather than a global exposure the map should adopt. "
                         "Kept as the control that establishes this.")
    ap.add_argument("--cull", type=int, default=0, metavar="TILES",
                    help="lever 1 (skill 16.8): submit only the Gaussians whose "
                         "block AABB intersects the view frustum. TILES^2 blocks "
                         "per keyframe; 4 measured 2.1x with zero false "
                         "negatives. 0 = off (submit the whole map).")
    ap.add_argument("--cull-exact", action="store_true",
                    help="per-Gaussian frustum test with each Gaussian's own "
                         "3-sigma footprint, instead of --cull's block AABBs. "
                         "Keeps 4.7-11.8%% of the map against the block test's "
                         "28-50%%, matches `radii > 0` to 0.1 pp, and its own "
                         "cost is 10.7 ms against 137 ms. Supersedes --cull.")
    ap.add_argument("--cull-margin", type=float, default=1.3,
                    help="frustum widening for --cull; loose on purpose, a false "
                         "positive costs throughput and a false negative loses "
                         "gradient")
    ap.add_argument("--mask-cache", type=int, default=0, metavar="REFRESH",
                    help="lever 4 (skill 16.11): per-view cache of the measured "
                         "gradient support, re-observed every REFRESH visits of "
                         "that view. Occlusion-aware, unlike --cull, and unlike "
                         "--cull it is an approximation. 0 = off. Requires --cull "
                         "(the observe step falls back to the culled set).")
    ap.add_argument("--mask-no-union", action="store_true",
                    help="on refresh, replace the cached support instead of "
                         "OR-ing it with the previous observation")
    ap.add_argument("--lattice-report", action="store_true",
                    help="report self-consistency vs a 2x render and the alpha "
                         "lattice amplitude at start and end -- the only "
                         "measurement that can see whether the optimizer "
                         "removed the band limit")
    ap.add_argument("--seam-report", action="store_true",
                    help="split held-out views by multi-keyframe overlap and "
                         "score them separately. Per-keyframe rigid "
                         "re-anchoring can tear seams where two clusters cover "
                         "one surface, and a global score cannot see it.")
    ap.add_argument("--config", default="config/eval_calib.yaml")
    ap.add_argument("--save-ply", default=None, metavar="PATH",
                    help="write the refined map as a 3DGS .ply so the "
                         "fly-through metric (scripts/diag_flythrough.py) can "
                         "score it -- the photometric numbers this script "
                         "prints are provably blind to the seams (17.17)")
    ap.add_argument("--save-renders", default=None, metavar="DIR",
                    help="after the final iteration, render every held-out "
                         "view at NATIVE resolution and write pred + GT PNGs "
                         "to DIR (publication-grade figures; the GUI capture "
                         "path renders above native resolution and is not "
                         "suitable for that).")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.chdir(REPO_ROOT)
    import lpips as lpips_lib
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.splatt3r_utils import resize_img
    from splatt3r_slam.refiner import LocalGaussianMap, sim3_to_mat, gaussians_from_keyframe
    from refine_gaussian_map import uniform_subsample

    load_config(args.config)
    os.chdir(CORE)
    dev = args.device
    torch.manual_seed(args.seed)

    ds = load_dataset(os.path.join(REPO_ROOT, args.dataset))
    ds_ts = np.array([float(t) for t in ds.timestamps])
    est_ts, est_T = load_tum_traj(os.path.join(REPO_ROOT, args.traj))
    frm_ts, frm_T = load_tum_traj(os.path.join(REPO_ROOT, args.frames_traj))
    gt_ts, gt_T = load_tum_traj(os.path.join(REPO_ROOT, args.dataset, "groundtruth.txt"))

    pairs = associate(est_ts, gt_ts)
    s_, R_, t_ = umeyama_sim3(np.array([est_T[i, :3, 3] for i, _ in pairs]),
                              np.array([gt_T[j, :3, 3] for _, j in pairs]))
    Rt = R_.T

    def to_map(c2w_gt):
        m = np.eye(4)
        m[:3, :3] = Rt @ c2w_gt[:3, :3]
        m[:3, 3] = Rt @ (c2w_gt[:3, 3] - t_) / s_
        return m

    # Held-out set built exactly as refine_gaussian_map.py builds it, so the
    # numbers here are directly comparable to every offline result.
    kf_idx = {j for _, j in associate(est_ts, ds_ts)}
    gt_pairs = associate(ds_ts, gt_ts)
    non_kf = [(i, j) for i, j in gt_pairs if i not in kf_idx]
    held_c = non_kf[:: max(1, len(non_kf) // args.n_held)][: args.n_held]
    held_set = {i for i, _ in held_c}

    # Supervision: estimated poses for tracked frames, no ground truth (the
    # deployable protocol). Held-out frames removed first.
    fr_pairs = associate(frm_ts, ds_ts)
    cand = [(fi, di) for fi, di in fr_pairs if di not in held_set]
    sel = uniform_subsample(cand, args.n_train)

    def load_views(items, mapped):
        """items are (pose_idx, dataset_idx) for the estimated path and
        (dataset_idx, gt_idx) for the held-out path -- the two orderings come
        from associate() being called with its arguments the other way round,
        so they are unpacked separately rather than with a shared alias."""
        out = []
        for a, b in items:
            di = a if mapped else b
            img = resize_img(ds.get_image(di), ds.img_size)["img"]
            tgt = torch.as_tensor(img, dtype=torch.float32, device=dev) * 0.5 + 0.5
            out.append((to_map(gt_T[b]) if mapped else frm_T[a], tgt))
        return out

    # Supervision poses are the SLAM estimates (no ground truth); held-out
    # views are scored at mapped ground-truth poses, exactly as every offline
    # number in the skill was.
    train_frames = load_views(sel, mapped=False)
    held_frames = load_views(held_c, mapped=True)
    # FramePoseLog anchor rule (splatt3r_slam/evaluate.py:86): each tracked
    # frame is anchored to the LATEST keyframe at tracking time, and when the
    # backend corrects that keyframe the frame's world pose is recomputed
    # through the correction. --perturb-mode block replays exactly that.
    anchors = np.clip(
        np.searchsorted(est_ts, [ds_ts[di] for _, di in sel], side="right") - 1,
        0, None)
    train_frames = [(c2w, tgt, int(a)) for (c2w, tgt), a in zip(train_frames, anchors)]
    print(f"train={len(train_frames)}  held-out={len(held_frames)}", flush=True)

    # --- build the map ---
    blob = torch.load(os.path.join(REPO_ROOT, args.kfgauss), map_location="cpu")
    parts = []
    kf_pose_data = []
    for k, kf in enumerate(blob["keyframes"]):
        h, w = int(kf["img_shape"][0]), int(kf["img_shape"][1])
        local = {key: kf[key].to(dev) for key in
                 ("means", "scales", "rotations", "sh", "opacities", "conf")}
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev,
                                      min_confidence=args.min_confidence,
                                      aa_sigma_scale=args.aa_sigma,
                                      aa_compensate_opacity=args.aa_compensate_opacity,
                                      max_anisotropy=args.max_anisotropy,
                                      streak_opacity=args.streak_opacity,
                                      hard_floor=args.aa_hard_floor,
                                      want_conf=args.conf_fade > 0)
        if got is not None and args.dither > 0 and len(got) > 6:
            got = list(got)
            m = got[0]
            pitch = got[6] / max(args.aa_sigma, 1e-9)
            ray = m / m.norm(dim=1, keepdim=True).clamp_min(1e-9)
            r = torch.randn_like(m)
            r = r - (r * ray).sum(1, keepdim=True) * ray      # project out the ray
            r = r / r.norm(dim=1, keepdim=True).clamp_min(1e-9)
            got[0] = m + r * (args.dither * pitch)[:, None]
            got = tuple(got)
        if got is not None and args.scale_cap > 0:
            got = list(got)
            got[1] = got[1].clamp(max=args.scale_cap)
            got = tuple(got)
        if got is not None and args.uniform_fade > 0:
            got = list(got)
            got[4] = got[4] * (1.0 - args.uniform_fade)
            got = tuple(got)
        if got is not None and args.conf_fade > 0:
            got = list(got)
            conf = got[-1]
            # Rank-normalize WITHIN the keyframe. The raw confidence head is
            # not calibrated across frames -- its scale drifts with texture and
            # exposure -- so a global threshold would thin whole keyframes
            # rather than the uncertain parts of each, which is a different
            # (and unintended) intervention. Ranking makes conf_norm uniform on
            # [0,1] per keyframe by construction, which is also what makes the
            # mean multiplier exactly (1-D) and the dose comparable to
            # --uniform-fade.
            r = torch.argsort(torch.argsort(conf)).float()
            conf_norm = r / max(len(r) - 1, 1)
            # Floor at 0.1, Kimi's round-29 guard. The multiplier at
            # conf_norm=0 is 1-2D, which reaches ZERO at D=0.5 -- and a
            # multiplier of zero is deletion, not thinning, which is the round-5
            # failure re-entering silently through the extreme tail. The floor
            # keeps this a thinning prior at every D. At the D=0.45 used for the
            # 17.66.4 arms the untruncated minimum is exactly 0.1, so the floor
            # is a no-op there and does not disturb those measurements.
            mult = (1.0 - args.conf_fade * 2.0 * (1.0 - conf_norm)).clamp(0.1, 1.0)
            got[4] = got[4] * mult
            got = tuple(got[:-1])
        if got is None:
            continue
        parts.append(got)
        kf_pose_data.append(kf["T_WC"].to(dev))
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in kf_pose_data]))

    if args.crowd_fade > 0:
        # CROWDING-WEIGHTED FADE (Kimi round 32). If 2/3 of conf-fade's value is
        # "reach a lower accumulated alpha for the same dose" (17.71.7), then
        # confidence is only a PROXY for crowding and the direct variable should
        # do better. This targets the map property itself.
        #
        # Applied HERE, after assembly, and not per-keyframe like the other two
        # fades -- and that placement is the whole point. Within one keyframe
        # there is exactly one Gaussian per pixel, so there is no stack to
        # measure; crowding only exists once several keyframes' Gaussians land
        # on the same surface. It is a map-level property and the head never
        # sees a map (17.70.8), which is also why no training-time penalty
        # could ever have priced it.
        #
        # NO rank normalization, unlike --conf-fade. Rank-per-keyframe forces
        # every keyframe to give up the same fraction regardless of how crowded
        # it actually is -- a defensible compromise for an uncalibrated
        # confidence head, but a bug for crowding, which is a geometric quantity
        # in absolute units and comparable across keyframes by construction.
        #
        # The multiplier targets a fixed alpha budget per occupied voxel:
        #   mult = clamp(T / sum(opacity in voxel), 0.1, 1)
        # so a voxel already at or below the target is untouched and one at 4x
        # the target is thinned 4x. Floored at 0.1 for the same reason as
        # --conf-fade: a zero multiplier is deletion, not thinning.
        means_w = torch.cat([
            (kf_mats[i, :3, :3] @ p[0].T).T + kf_mats[i, :3, 3]
            for i, p in enumerate(parts)])
        opas_all = torch.cat([p[4].reshape(-1) for p in parts])
        vox = args.crowd_voxel
        if vox <= 0:
            # default: 4x the median Gaussian extent, so a voxel holds the
            # Gaussians that genuinely overlap rather than an arbitrary volume
            vox = 4.0 * float(torch.cat([p[1] for p in parts]).median())
        keys = torch.floor(means_w / vox).to(torch.int64)
        keys = keys - keys.min(0).values
        span = keys.max(0).values + 1
        flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
        uniq, inv = torch.unique(flat, return_inverse=True)
        vox_alpha = torch.zeros(len(uniq), device=dev, dtype=opas_all.dtype)
        vox_alpha.index_add_(0, inv, opas_all)
        mult_all = (args.crowd_fade / vox_alpha.clamp_min(1e-6))[inv].clamp(0.1, 1.0)
        print(f"  crowd-fade: voxel={vox*100:.2f} cm  occupied={len(uniq):,}  "
              f"alpha/voxel median={float(vox_alpha.median()):.2f}  "
              f"mult mean={float(mult_all.mean()):.3f} "
              f"min={float(mult_all.min()):.3f}", flush=True)
        off = 0
        for i, p in enumerate(parts):
            n = p[4].shape[0]
            p = list(p)
            p[4] = p[4] * mult_all[off:off + n]
            parts[i] = tuple(p)
            off += n

    if args.reanchor == "gt":
        # THE value proposition of a trajectory-anchored map, in one number.
        # Each keyframe's Gaussians were predicted in its own camera frame and
        # are merely PLACED by its pose. If the placement is what is wrong --
        # rather than the prediction -- then swapping the estimated anchors for
        # ground-truth ones should recover a large part of the 4.5 dB pose gap
        # with no optimization at all. A baked world-space map cannot do this;
        # the pose error is already fused into its geometry.
        #
        # Recovers toward ~19 -> re-anchoring is the mechanism, seams are
        # negligible, P1 wins outright. Recovers little -> either seams tear, or
        # the head has already absorbed the pose error INSIDE each cluster, and
        # no amount of re-anchoring helps.
        # Associate through the TRAJECTORY FILE's timestamps, not through
        # kf["frame_id"]. main.py:320 calls dataset.subsample(), so inside the
        # SLAM run frame_id indexes the SUBSAMPLED stream (307 entries on desk)
        # while every analysis script here loads the dataset unsubsampled (613).
        # Indexing ds_ts with a raw frame_id therefore lands on the wrong frame,
        # and increasingly so along the sequence: an earlier version of this did
        # exactly that and produced keyframe/ground-truth mismatches of up to
        # 1.8 m and 126 deg, reported as a 4.7 dB "re-anchoring loss".
        # est_ts[i] is keyframe i's own timestamp, verified to match the dumped
        # T_WC line for line.
        kf_gt = []
        n_miss = 0
        for i_kf in range(len(blob["keyframes"])):
            g = associate(np.array([est_ts[i_kf]]), gt_ts)
            if not g:
                n_miss += 1
                kf_gt.append(None)
                continue
            kf_gt.append(to_map(gt_T[g[0][1]]))
        keep = [i for i, m in enumerate(kf_gt) if m is not None]
        new_mats = kf_mats.clone()
        for i in keep:
            g_i = torch.as_tensor(kf_gt[i], dtype=torch.float32, device=dev)
            # Keep this keyframe's OWN Sim3 scale. T_WC is a Sim3 and its scale
            # is folded into the rotation block by sim3_to_mat, so it also sets
            # the physical size of that keyframe's Gaussian cluster. Measured
            # here: the per-keyframe scales run 0.685-1.039, nowhere near 1.
            # Overwriting the whole 3x3 with the (scale-free) mapped ground-truth
            # rotation silently resizes every cluster by up to 46% -- an earlier
            # version of this did exactly that and reported a 4.4 dB "loss" that
            # was entirely the resize.
            s_kf = torch.linalg.det(kf_mats[i, :3, :3]).abs() ** (1.0 / 3.0)
            new_mats[i, :3, :3] = s_kf * g_i[:3, :3]
            new_mats[i, :3, 3] = g_i[:3, 3]
        kf_mats = new_mats
        print(f"  RE-ANCHORED {len(keep)}/{len(kf_gt)} keyframes to ground-truth "
              f"poses ({n_miss} unmatched)", flush=True)
    means, scales, rots, rgb, opas, kf_id = [
        torch.cat([p[i] for p in parts]) for i in range(6)]
    floor = (torch.cat([p[6] for p in parts]) if args.aa_hard_floor else None)
    print(f"map: {means.shape[0]:,} gaussians over {len(parts)} keyframes", flush=True)

    if args.dedup_voxel > 0 and args.stage != 1:
        # De-clustering ablation (gates item c). Where several keyframes'
        # clusters share one voxel, keep the EARLIEST owner's Gaussians and
        # delete the rest -- if the overlap-quality anomaly is two shells
        # interfering, the overlap class should improve disproportionately.
        with torch.no_grad():
            A = kf_mats[kf_id, :3, :3]
            b = kf_mats[kf_id, :3, 3]
            mw = torch.einsum("mij,mj->mi", A, means) + b
            vox = torch.floor(mw / args.dedup_voxel).long()
            vox = vox - vox.min(0).values
            B = int(vox.max().item()) + 2
            vox_key = (vox[:, 0] * B + vox[:, 1]) * B + vox[:, 2]
            n_kf_ = int(kf_id.max().item()) + 1
            order = torch.argsort(vox_key * n_kf_ + kf_id)  # by voxel, then kf
            vk_s, kf_s = vox_key[order], kf_id[order]
            bounds = torch.ones_like(vk_s, dtype=torch.bool)
            bounds[1:] = vk_s[1:] != vk_s[:-1]
            group_id = torch.cumsum(bounds, 0) - 1
            min_kf = kf_s[bounds][group_id]  # each group's earliest keyframe
            keep_sorted = kf_s == min_kf
            keep = torch.empty_like(keep_sorted)
            keep[order] = keep_sorted
            n_del = int((~keep).sum())
            print(f"dedup: voxel {args.dedup_voxel} m -> delete {n_del:,}/"
                  f"{means.shape[0]:,} ({100.0 * n_del / means.shape[0]:.1f}%) "
                  f"from shared voxels", flush=True)
            means, scales, rots, rgb, opas, kf_id = [
                t[keep] for t in (means, scales, rots, rgb, opas, kf_id)]
            if floor is not None:
                floor = floor[keep]

    if args.stage == 1:
        # World-space control: bake once, forget the keyframe attribution.
        A = kf_mats[kf_id, :3, :3]
        b = kf_mats[kf_id, :3, 3]
        means = torch.einsum("mij,mj->mi", A, means) + b
        from utils.geometry import build_covariance
        q = rots / rots.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        cov = A @ build_covariance(scales, q) @ A.transpose(1, 2)
        # Re-express as scale/rotation so the same parameterization is optimized.
        U, S, _ = torch.linalg.svd(cov.double())
        det = torch.linalg.det(U)
        U[det < 0, :, 0] *= -1
        scales = S.clamp_min(1e-16).sqrt().float()
        from scipy.spatial.transform import Rotation
        rots = torch.as_tensor(
            Rotation.from_matrix(U.cpu().numpy()).as_quat(), dtype=torch.float32,
            device=dev)
        kf_id = torch.zeros_like(kf_id)
        kf_mats = torch.eye(4, device=dev)[None]

    if args.stage == 1 and floor is not None:
        # Stage 1 re-diagonalizes the world-space covariance, so a scale floor
        # expressed against the LOCAL axes no longer refers to anything. Rather
        # than silently apply it to the wrong quantity, fold it in before the
        # SVD -- stage 1 is a plumbing control and does not need the constraint
        # property, only the same geometry.
        raise SystemExit("--aa-hard-floor is meaningless under --stage 1; "
                         "use --aa-sigma (folded into the scales) there")
    model = LocalGaussianMap(
        means, scales, rots, rgb, opas, kf_id, scale_floor=floor,
        # pitch = floor / aa_sigma recovers the raw lattice spacing, so the
        # positional rate can be expressed per-Gaussian without carrying a
        # second copy of it.
        pitch=(floor / args.aa_sigma if args.pitch_lr and floor is not None
               else None)).to(dev)
    if args.pitch_lr and floor is None:
        raise SystemExit("--pitch-lr needs --aa-sigma > 0 --aa-hard-floor "
                         "(the pitch is only computed on that path)")
    extent = float((means.max(0).values - means.min(0).values).norm() / 2)
    opt = torch.optim.Adam(
        model.param_groups(extent, lr_means=args.lr_means, lr_f_dc=args.lr_f_dc,
                           lr_opacity=args.lr_opacity, lr_scale=args.lr_scale,
                           lr_rot=args.lr_rot,
                           lr_kf_depth=args.kf_depth_lr), eps=1e-15)
    model._depth_free = args.kf_depth_lr > 0

    # Free keyframe translations (3 DoF each). Rotation is left fixed: the
    # quantity that trades off against a per-cluster depth scale is where the
    # cluster sits along the view ray, not how it is oriented.
    kf_dt = None
    if args.kf_pose_lr > 0:
        kf_dt = torch.nn.Parameter(torch.zeros(kf_mats.shape[0], 3, device=dev))
        opt.add_param_group({"params": [kf_dt], "lr": args.kf_pose_lr,
                             "name": "kf_dt"})

    referee_log = None
    if args.referee_scales and args.referee_weight > 0:
        # absolute: this script chdir()s into splatt3r_core, and a relative
        # path resolves there instead -- the same trap that silently ate
        # every control-arm .npz in 17.45
        rs = np.load(os.path.join(REPO_ROOT, args.referee_scales)
                     if not os.path.isabs(args.referee_scales)
                     else args.referee_scales)
        assert len(rs) == kf_mats.shape[0], (
            f"{len(rs)} referee scales for {kf_mats.shape[0]} keyframes")
        referee_log = torch.log(torch.as_tensor(rs, dtype=torch.float32,
                                                device=dev))
        referee_log = referee_log - referee_log.median()
        print(f"  referee prior: {len(rs)} keyframes, weight "
              f"{args.referee_weight}, median |s-1| = "
              f"{float((referee_log.exp() - 1).abs().median()) * 100:.2f}%",
              flush=True)

    def live_mats():
        """kf_mats with the free translations applied, differentiably."""
        if kf_dt is None:
            return kf_mats
        m = kf_mats.clone()
        m[:, :3, 3] = m[:, :3, 3] + kf_dt
        return m

    win = gaussian_window(device=dev)
    lp = lpips_lib.LPIPS(net="alex").to(dev)
    K = ds.camera_intrinsics.K_frame
    rng = np.random.default_rng(args.seed)
    tag = f"[{args.tag}] " if args.tag else ""

    @torch.no_grad()
    def classify_seams():
        """Split the held-out views by whether they see MULTI-KEYFRAME overlap.

        Per-keyframe rigid re-anchoring is this representation's structural
        weakness: where two keyframes' Gaussian clusters cover the same surface,
        a pose update pulls them in different directions and the seam can tear
        into a double surface. A global score cannot see that -- the torn region
        is a small fraction of the image -- so the held-out set is split and
        reported separately.

        Coverage per keyframe is obtained by rendering that keyframe's cluster
        alone with white colour on a black background, which approximates its
        accumulated alpha. Done once, at startup.
        """
        ones = torch.ones_like(model.f_dc)
        mw, cw = model.world(kf_mats)
        op = model.opacity()
        n_kf = int(kf_id.max().item()) + 1
        frac = []
        for c2w, tgt in held_frames:
            # Distance prefilter: a keyframe cluster anchored metres away
            # cannot cover this view. Classification-only heuristic; on desk
            # (all keyframes within ~2 m) it is a no-op, on room-style
            # trajectories it is what makes per-keyframe coverage affordable.
            vp = torch.as_tensor(c2w, device=dev)[:3, 3]
            near_k = (kf_mats[:, :3, 3] - vp).norm(dim=1) < 5.0
            cov_count = torch.zeros(tgt.shape[-2:], device=dev)
            for k in range(n_kf):
                if not bool(near_k[k]):
                    continue
                m = kf_id == k
                if not bool(m.any()):
                    continue
                a = render(mw[m], cw[m], (ones[m] * 0 + 1.0), op[m], c2w, K,
                           tgt.shape[-2:], dev).clamp(0, 1).mean(1)[0]
                cov_count += (a > 0.5).float()
            frac.append(float((cov_count >= 2).float().mean()))
        frac = np.array(frac)
        thr = float(np.median(frac))
        overlap = [i for i in range(len(frac)) if frac[i] > thr]
        single = [i for i in range(len(frac)) if frac[i] <= thr]
        print(f"  seam split: overlap-fraction median {thr:.3f}  "
              f"({len(overlap)} overlap-heavy / {len(single)} single-coverage views)",
              flush=True)
        return overlap, single

    @torch.no_grad()
    def classify_block(split):
        """Three-way split of held-out views by WHICH SIDE of a block
        correction covers them: only keyframes < split (the unperturbed side),
        only keyframes >= split (the corrected side), or BOTH -- the seam
        class, the only views that can tear. Computed on CURRENT geometry, so
        call it right after a perturbation: that is the state the optimizer
        has to deal with. A keyframe counts as covering a view when its solo
        render puts alpha > 0.5 on at least 5% of pixels, so stray specks do
        not manufacture seam membership.
        """
        ones = torch.ones_like(model.f_dc)
        mw, cw = model.world(kf_mats)
        op = model.opacity()
        n_kf = int(kf_id.max().item()) + 1
        lo, hi, mix, none = [], [], [], []
        for i, (c2w, tgt) in enumerate(held_frames):
            vp = torch.as_tensor(c2w, device=dev)[:3, 3]
            near_k = (kf_mats[:, :3, 3] - vp).norm(dim=1) < 5.0
            ks = set()
            for k in range(n_kf):
                if not bool(near_k[k]):
                    continue
                m = kf_id == k
                if not bool(m.any()):
                    continue
                a = render(mw[m], cw[m], (ones[m] * 0 + 1.0), op[m], c2w, K,
                           tgt.shape[-2:], dev).clamp(0, 1).mean(1)[0]
                if float((a > 0.5).float().mean()) > 0.05:
                    ks.add(k)
            has_lo = any(k < split for k in ks)
            has_hi = any(k >= split for k in ks)
            if has_lo and has_hi:
                mix.append(i)
            elif has_hi:
                hi.append(i)
            elif has_lo:
                lo.append(i)
            else:
                none.append(i)
        print(f"  block split at kf {split}: held-out {len(lo)} low-only / "
              f"{len(hi)} high-only / {len(mix)} seam / {len(none)} uncovered",
              flush=True)
        return lo, hi, mix

    @torch.no_grad()
    def affine_fit(pred, tgt):
        """Per-channel least-squares gain+bias taking pred onto tgt.

        Not a way to make the numbers look better. TUM freiburg1 runs the
        camera's auto-exposure, so a held-out frame carries its own exposure --
        a property of that frame, not of the map, and one no single map can
        match for every frame at once. Scoring both arms after removing that
        nuisance is what isolates the map; the un-fitted number stays reported
        beside it because that is what a deployed render actually looks like.

        Fitted over rendered pixels only: unmapped background is exactly zero
        in all three channels and would otherwise drag the fit toward black.
        """
        m = (pred.sum(1, keepdim=True) > 0).float()
        n = m.sum().clamp_min(1.0)
        mx = (pred * m).sum((0, 2, 3)) / n
        my = (tgt * m).sum((0, 2, 3)) / n
        dx = (pred - mx.view(1, 3, 1, 1)) * m
        dy = (tgt - my.view(1, 3, 1, 1)) * m
        a = ((dx * dy).sum((0, 2, 3)) / (dx * dx).sum((0, 2, 3)).clamp_min(1e-8))
        a = a.clamp(0.2, 5.0)
        b = my - a * mx
        return (pred * a.view(1, 3, 1, 1) + b.view(1, 3, 1, 1)).clamp(0, 1)

    @torch.no_grad()
    def evaluate(subset=None, fitted=False):
        mses, lps = [], []
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        views = held_frames if subset is None else [held_frames[i] for i in subset]
        for c2w, tgt in views:
            pred = render(mw, cw, rgb_, op_, c2w, K, tgt.shape[-2:], dev).clamp(0, 1)
            if fitted:
                pred = affine_fit(pred, tgt)
            mses.append(torch.mean((pred - tgt) ** 2).item())
            lps.append(lp(pred * 2 - 1, tgt * 2 - 1).item())
        mse = sum(mses) / len(mses)
        return -10 * math.log10(max(mse, 1e-12)), sum(lps) / len(lps)

    @torch.no_grad()
    def lattice_report(label):
        """Self-consistency against a 2x render, and the lattice amplitude in
        the alpha channel. See scripts/diag_lattice.py for the derivation; the
        point of running it here is that it is the only number that can tell
        whether the optimizer walked the band limit back off, which the
        native-rate psnr provably cannot see (it moved 0.01 dB across the whole
        tau sweep while the alpha lattice changed 13x)."""
        if not args.lattice_report:
            return
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        ones = torch.ones_like(rgb_)
        K2 = torch.as_tensor(K, dtype=torch.float32, device=dev).clone()
        K2[:2] *= 2
        selfp, hpa, blackf, accum, culled = [], [], [], [], []
        for c2w, tgt in held_frames[::5]:
            h, w = tgt.shape[-2:]
            r1 = render(mw, cw, rgb_, op_, c2w, K, (h, w), dev).clamp(0, 1)
            rs = render(mw, cw, rgb_, op_, c2w, K2, (2 * h, 2 * w), dev).clamp(0, 1)
            selfp.append(-10 * math.log10(max(float(
                torch.mean((r1 - F.avg_pool2d(rs, 2)) ** 2)), 1e-12)))
            a1 = render(mw, cw, ones, op_, c2w, K, (h, w), dev).clamp(0, 1)
            ass = render(mw, cw, ones, op_, c2w, K2, (2 * h, 2 * w), dev).clamp(0, 1)
            cov = F.interpolate((a1.mean(1, keepdim=True) > 0.1).float(),
                                scale_factor=2) > 0.5
            blur = F.avg_pool2d(F.pad(ass, (1, 1, 1, 1), mode="replicate"),
                                3, stride=1)
            hp = (ass - blur).abs().mean(1, keepdim=True)
            m = cov.float()
            n = m.sum().clamp_min(1)
            mu = (hp * m).sum() / n
            hpa.append(float((((hp - mu) ** 2 * m).sum() / n) ** 0.5))
            # Black fraction: pixels the map fails to cover at all. This is the
            # FIRST thing to check for any opacity-reducing lever (Kimi,
            # round 29) -- it was the measured signature of the 17.5 removal
            # failure, and a thinning prior that has quietly become a deletion
            # prior shows up here before it shows up in psnr. The help text for
            # --streak-opacity has told the reader to watch it since 17.16
            # while nothing in this script computed it.
            blackf.append(float((a1.mean(1) < 0.02).float().mean()))
            # Accumulated alpha (total alpha MASS per pixel, sum_k alpha_k),
            # which 17.64 established as the operational target for the
            # thinning levers and which until now was only ever measured ad
            # hoc. It is not `a1`: composited alpha saturates at 1 and cannot
            # exceed it, while the quantity that ordered the four maps
            # (6.17/6.66/4.45/3.16) is the uncapped sum.
            #
            # Recovered by rendering at opacity eps*alpha and inverting the
            # compositing law EXACTLY rather than to first order:
            #
            #   A = 1 - prod_k (1 - eps*a_k)   =>   sum_k a_k = -ln(1-A)/eps
            #
            # (exact whenever each individual eps*a_k is small, which the eps
            # below guarantees; it does NOT require eps*sum(a) to be small,
            # which is what the first-order form needed and what made it
            # inaccurate on crowded maps).
            #
            # eps=0.05, NOT 0.01. The rasterizer culls Gaussians whose alpha
            # falls below ~1/255, so on a THINNED map (median opacity 0.25
            # after a D=0.55 fade) eps=0.01 puts every Gaussian at 0.0025 --
            # under the cutoff -- and the render comes back exactly zero. That
            # is not a small bias: it silently reported acc_alpha=0.00 for the
            # injected map while the same map measured 3.49 after optimization
            # raised the opacities back over the threshold, and it biased
            # exactly the arms this metric was built to compare, since conf-fade
            # and uniform-fade differ precisely in how many Gaussians they push
            # into the low-alpha tail.
            eps = 0.05
            aeps = render(mw, cw, ones, op_ * eps, c2w, K, (h, w), dev).clamp(0, 1)
            # native-resolution coverage; `cov` above is the 2x grid used for
            # the lattice statistic and has the wrong shape to index this
            covn = a1.mean(1) > 0.1
            if bool(covn.any()):
                A = aeps.mean(1)[covn].clamp(max=1 - 1e-6)
                accum.append(float((-torch.log1p(-A) / eps).mean()))
            else:
                accum.append(float("nan"))
            # Guard: report the share of Gaussians still under the cull
            # threshold, because when it is large the number above is an
            # underestimate and must not be compared across arms.
            culled.append(float((op_ * eps < 1.0 / 255).float().mean()))
        print(f"  lattice[{label}]: self-psnr={sum(selfp) / len(selfp):6.2f} dB  "
              f"hp_alpha={sum(hpa) / len(hpa):.5f}  "
              f"black={100 * sum(blackf) / len(blackf):.2f}%  "
              f"acc_alpha={sum(accum) / len(accum):.2f}"
              + (f"  [UNRELIABLE: {100 * sum(culled) / len(culled):.0f}% of "
                 f"gaussians under the rasterizer cull threshold]"
                 if sum(culled) / len(culled) > 0.02 else ""), flush=True)

    perturb_iters = [args.perturb_kf_at + i * max(1, args.iters // (args.n_perturb + 2))
                     for i in range(args.n_perturb)]
    perturb_iters = [i for i in perturb_iters if i < args.iters]

    seam_overlap, seam_single = (classify_seams() if args.seam_report else (None, None))
    block_classes = None  # (lo, hi, mix) held-out split, set at each block perturbation

    def report(it, extra=""):
        p, l = evaluate()
        line = f"  {tag}iter {it:5} | held-out psnr={p:7.4f}  lpips={l:.4f}"
        pf, lf = evaluate(fitted=True)
        line += f" | fit psnr={pf:7.4f}  lpips={lf:.4f}"
        if block_classes is not None:
            lo, hi, mix = block_classes
            for name, sub in (("low", lo), ("high", hi), ("seam", mix)):
                if sub:
                    ps, ls = evaluate(sub)
                    line += f" | {name} {ps:7.4f}/{ls:.4f}"
        if args.seam_report:
            po, lo_ = evaluate(seam_overlap)
            ps, ls = evaluate(seam_single)
            line += f" | overlap {po:7.4f}/{lo_:.4f}  single {ps:7.4f}/{ls:.4f}"
        print(line + extra, flush=True)
        return p, l

    # Per-supervision-frame exposure. Six free parameters per frame, applied to
    # the RENDER, never to the target: the map must still explain the target,
    # it is just allowed to do so up to that frame's exposure. Initialised to
    # the identity (log-gain 0, bias 0) so step 0 is unchanged.
    #
    # The cheat this could become, and why it does not: a per-frame affine has
    # only 6 degrees of freedom against ~590k pixels, and it is spatially
    # constant, so it cannot absorb any structured error -- no geometry, no
    # blur, no seam INSIDE a frame. What it can absorb is exactly what
    # auto-exposure produces, a global gain and offset. Held-out frames get no
    # such parameters, which is what keeps the reported number honest.
    exposure = None
    if args.exposure:
        exposure = torch.zeros((len(train_frames), 2, 3), device=dev,
                               requires_grad=True)
        opt.add_param_group({"params": [exposure], "lr": args.exposure_lr,
                             "name": "exposure"})
        print(f"  exposure: {len(train_frames)} frames x 6 params, "
              f"lr={args.exposure_lr}", flush=True)

    mask_cache = None
    if args.mask_cache:
        from splatt3r_slam.refiner import ViewMaskCache
        if not args.cull:
            print("  --mask-cache without --cull: observe steps will submit the "
                  "whole map, which is the slow path this is meant to avoid",
                  flush=True)
        mask_cache = ViewMaskCache(refresh=args.mask_cache,
                                   union=not args.mask_no_union)
    if args.cull or args.cull_exact or mask_cache is not None:
        print(f"  culling: mode={'exact' if args.cull_exact else 'block'} "
              f"tiles={args.cull} margin={args.cull_margin} "
              f"mask_cache={args.mask_cache}", flush=True)

    lattice_report("init")
    report(0, "  (init)")

    for it in range(1, args.iters + 1):
        if args.stage == 3 and it in perturb_iters:
            # Simulate a loop closure. A real correction is a Sim3 -- rotation,
            # translation AND scale -- so a translation-only test would leave
            # the whole scale path unexercised (cov picks up s^2, and screen
            # size feeds the densification thresholds). Applied as a global
            # similarity so the supervision poses can move with it and quality
            # is genuinely expected to be preserved: the point of the test is
            # that the MAP follows, not that it is robust to inconsistency.
            #
            # Repeated, not once: a single correction passing does not mean ten
            # do. Any place that writes world-space values back into local
            # parameters accumulates error per event, so the map is always
            # re-composed from the canonical local parameters and the current
            # poses, never updated incrementally.
            n_done = perturb_iters.index(it) + 1
            g = torch.Generator(device="cpu").manual_seed(args.seed + 77 + it)
            axis = torch.randn(3, generator=g); axis = axis / axis.norm()
            ang = float(torch.randn(1, generator=g)) * args.perturb_kf_rot
            Kx = torch.tensor([[0, -axis[2], axis[1]],
                               [axis[2], 0, -axis[0]],
                               [-axis[1], axis[0], 0]], dtype=torch.float64)
            Rp = (torch.eye(3, dtype=torch.float64) + math.sin(ang) * Kx
                  + (1 - math.cos(ang)) * (Kx @ Kx))
            sc = 1.0 + float(torch.randn(1, generator=g)) * args.perturb_kf_scale
            tp = torch.randn(3, generator=g).double() * args.perturb_kf_sigma
            S = torch.eye(4, dtype=torch.float64)
            S[:3, :3] = sc * Rp
            S[:3, 3] = tp
            Sd = S.to(dev).float()

            if args.perturb_mode == "global":
                # Everything moves together. This checks that the composition is
                # correct and self-consistent -- but it CANNOT tear a seam,
                # because every cluster is carried rigidly by the same
                # transform. Real loop closures are differential.
                kf_mats = Sd @ kf_mats
                train_frames = [(S.numpy() @ np.asarray(c2w, dtype=np.float64), tgt, a)
                                for c2w, tgt, a in train_frames]
                held_frames = [(S.numpy() @ np.asarray(c2w, dtype=np.float64), tgt)
                               for c2w, tgt in held_frames]
            elif args.perturb_mode == "perkf":
                # DIFFERENTIAL: each keyframe gets its own small correction, as
                # a pose-graph solve actually produces. This is the arm that can
                # tear seams -- where two keyframes' clusters cover one surface
                # and are now pulled apart, the shared surface splits. The
                # supervision cameras are NOT moved, because there is no single
                # transform that would move them consistently; the map is
                # therefore expected to LOSE quality here, and the question is
                # how much and whether it is concentrated in overlap regions.
                nk = kf_mats.shape[0]
                gd = torch.Generator(device="cpu").manual_seed(args.seed + 991 + it)
                dts = (torch.randn((nk, 3), generator=gd)
                       * args.perturb_kf_sigma).to(dev)
                kf_mats = kf_mats.clone()
                kf_mats[:, :3, 3] += dts
            elif args.perturb_mode == "block":
                # FAITHFUL differential test (item b'). A real loop closure
                # corrects a contiguous block of the trajectory roughly rigidly
                # against the older map, so ONE Sim3 goes to the second half of
                # the keyframes. Supervision views move WITH their anchor
                # keyframe -- that is what FramePoseLog guarantees online --
                # unless --freeze-supervision (the unfaithful control). Within a
                # corrected block, views and clusters move together and stay
                # exactly self-consistent; the only photometric tension left is
                # cross-block, concentrated in the seam held-out class.
                # Held-out cameras stay at ground truth.
                split = kf_mats.shape[0] // 2
                kf_mats_pre = kf_mats.clone()
                kf_mats = kf_mats.clone()
                kf_mats[split:] = Sd @ kf_mats[split:]
                if not args.freeze_supervision:
                    S_np = S.numpy()
                    train_frames = [
                        (S_np @ np.asarray(c2w, dtype=np.float64) if a >= split
                         else c2w, tgt, a)
                        for c2w, tgt, a in train_frames]
                block_classes = classify_block(split)
                # Recompute the two-way overlap split on the perturbed
                # geometry too -- on desk the three-way split can degenerate
                # (every keyframe covers most views, so everything is "seam"),
                # and the median-overlap split is then the only differentiator.
                if args.seam_report:
                    seam_overlap, seam_single = classify_seams()
                # "before" row for those same view classes: the pre-perturbation
                # state scored under the post-perturbation split.
                kf_mats_new = kf_mats
                kf_mats = kf_mats_pre
                for name, sub in zip(("low", "high", "seam"), block_classes):
                    if sub:
                        pb, lb = evaluate(sub)
                        print(f"    pre-perturb {name}: psnr={pb:7.4f}  lpips={lb:.4f}",
                              flush=True)
                kf_mats = kf_mats_new
            # Every cached support was observed against the OLD poses. A
            # correction moves clusters relative to cameras, so occlusion order
            # changes and the cached sets are no longer even approximately
            # right; re-observe from scratch. This is the refresh policy 16.11
            # said the design needs.
            if mask_cache is not None:
                mask_cache.drop()
            report(it, f"  <- LOOP-CLOSURE #{n_done} "
                       f"(rot {math.degrees(ang):+.1f} deg, scale {sc:.4f}, "
                       f"|t| {float(tp.norm()):.3f} m)")

        for nm, when in (("means", args.freeze_means_after),
                         ("opacity", args.freeze_opacity_after)):
            if when and it == when + 1:
                for g in opt.param_groups:
                    if g["name"] == nm:
                        g["lr"] = 0.0
                print(f"  froze {nm} at iter {it - 1}", flush=True)

        fi = int(rng.integers(len(train_frames)))
        c2w, tgt, _ = train_frames[fi]
        # Submit as few Gaussians as is sound. Composing all of them when one
        # view can only see ~13% is where the step actually goes (16.7): the
        # composition's backward is ~54% of a step against 10.4% for the
        # rasterizer's forward, and the rasterizer's own frustum cull cannot
        # help with work we did before calling it.
        idx, observe = (mask_cache.take(fi) if mask_cache is not None
                        else (None, False))
        if idx is None and args.cull_exact:
            idx = model.visible_exact(kf_mats, c2w, K, tgt.shape[-2:])
        elif idx is None and args.cull:
            idx = model.visible_subset(kf_mats, c2w, K, tgt.shape[-2:],
                                       tiles=args.cull, margin=args.cull_margin)
        mw, cw = model.world(live_mats(), idx)
        if args.ss_loss > 1:
            n_ss = args.ss_loss
            Kss = torch.as_tensor(K, dtype=torch.float32, device=dev).clone()
            Kss[:2] *= n_ss
            h_, w_ = tgt.shape[-2:]
            pred = F.avg_pool2d(
                render(mw, cw, model.rgb(idx), model.opacity(idx), c2w, Kss,
                       (h_ * n_ss, w_ * n_ss), dev), n_ss)
        else:
            pred = render(mw, cw, model.rgb(idx), model.opacity(idx), c2w, K,
                          tgt.shape[-2:], dev)
        if exposure is not None:
            g, b = exposure[fi, 0], exposure[fi, 1]
            pred = pred * g.exp().view(1, 3, 1, 1) + b.view(1, 3, 1, 1)
        loss = ((1 - DSSIM_WEIGHT) * (pred - tgt).abs().mean()
                + DSSIM_WEIGHT * (1 - ssim(pred.clamp(0, 1), tgt, win)))
        if args.consistency > 0:
            from splatt3r_slam.refiner import cross_cluster_loss
            # Computed on the CULLED subset, not the whole map: the term is a
            # per-voxel statistic, so a random view-sized sample of voxels is an
            # unbiased estimate of it and costs nothing extra.
            loss = loss + args.consistency * cross_cluster_loss(
                model, kf_mats, args.consistency_voxel, idx,
                w_rgb=args.consistency_rgb)
        if referee_log is not None:
            # A PRIOR, not an edit: the photometric term is free to disagree,
            # and what is being tested is whether the two together find a
            # better optimum than either alone.
            d = model.kf_log_depth.flatten() - referee_log
            loss = loss + args.referee_weight * (d ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if observe:
            mask_cache.observe(fi, model, idx)
        opt.step()

        if it % args.eval_every == 0:
            report(it, f"  train_loss={loss.item():.4f}")

    if args.exposure and args.exposure_recenter:
        # Fold the MEAN exposure back into the map. This was built to recover
        # the 0.159 dB of raw psnr that --exposure costs, on the theory that the
        # per-frame parameters were holding a global brightness the deployed map
        # then failed to carry. IT DOES NOT WORK: measured -0.106 dB more raw
        # psnr and -0.005 fit psnr (17.8). The mean of the per-frame gains is an
        # artifact of the training set, not a level the map should adopt, and
        # the transform is only equivalent where accumulated alpha is 1 anyway.
        with torch.no_grad():
            g_mu = exposure[:, 0].exp().mean(0)
            b_mu = exposure[:, 1].mean(0)
            rgb_new = (model.rgb() * g_mu + b_mu).clamp(0, 1)
            model.f_dc.copy_((rgb_new - 0.5) / 0.28209479177387814)
            exposure[:, 0] -= g_mu.log()
            exposure[:, 1] -= b_mu
        print(f"  exposure recentred into the map: gain {g_mu.tolist()} "
              f"bias {b_mu.tolist()}", flush=True)
        report(args.iters, "  (after recentring)")

    if args.save_ply:
        import pathlib as _pl
        from splatt3r_slam.refiner import save_refined_map
        outp = _pl.Path(args.save_ply)
        if not outp.is_absolute():
            outp = _pl.Path(REPO_ROOT) / outp
        outp.parent.mkdir(parents=True, exist_ok=True)
        save_refined_map(str(outp), model, kf_mats)

    lattice_report("final")
    if args.kf_depth_lr > 0:
        with torch.no_grad():
            d = model.kf_log_depth.exp()
        print(f"  kf depth scales: mean {float(d.mean()):.5f} "
              f"std {float(d.std()):.5f} "
              f"range [{float(d.min()):.5f}, {float(d.max()):.5f}]", flush=True)
        # the whole vector, so it can be correlated against an external
        # reference's per-cluster corrections (route R): if the photometric
        # optimum and the referee disagree, the map+trajectory is at a joint
        # optimum that absolute depth correctness does not describe
        print("  kf depth vector: "
              + " ".join(f"{float(v):.4f}" for v in d.flatten()), flush=True)
    if args.exposure:
        with torch.no_grad():
            g = exposure[:, 0].exp()
            b = exposure[:, 1]
        print(f"  exposure gains  {g.mean(0).tolist()} +- "
              f"{g.std(0).tolist()}\n  exposure biases {b.mean(0).tolist()} +- "
              f"{b.std(0).tolist()}", flush=True)
    if args.save_renders:
        # Publication figures: native-resolution held-out renders (pred + GT).
        # GUI captures render above the map's native resolution and show
        # sampling artifacts that are not in the map itself.
        import pathlib
        from PIL import Image as PILImage
        # This script os.chdir()s into splatt3r_core at startup; anchor a
        # relative output dir to the repo root or figures land in
        # splatt3r_core/logs/ (the same trap exp_head_only.py's --out had).
        out_dir = pathlib.Path(args.save_renders)
        if not out_dir.is_absolute():
            out_dir = pathlib.Path(REPO_ROOT) / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        with torch.no_grad():
            for i, (c2w, tgt) in enumerate(held_frames):
                pred = render(mw, cw, rgb_, op_, c2w, K, tgt.shape[-2:], dev)
                for name, img in (("pred", pred), ("gt", tgt)):
                    arr = (img.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
                           * 255).astype(np.uint8)
                    PILImage.fromarray(arr).save(
                        out_dir / f"view{i:03d}_{name}.png")
        print(f"  saved {len(held_frames)} pred/gt render pairs -> {out_dir}",
              flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
