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
    ap.add_argument("--seam-report", action="store_true",
                    help="split held-out views by multi-keyframe overlap and "
                         "score them separately. Per-keyframe rigid "
                         "re-anchoring can tear seams where two clusters cover "
                         "one surface, and a global score cannot see it.")
    ap.add_argument("--config", default="config/eval_calib.yaml")
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
        got = gaussians_from_keyframe(local, kf["img"].to(dev), h, w, k, dev)
        if got is None:
            continue
        parts.append(got)
        kf_pose_data.append(kf["T_WC"].to(dev))
    kf_mats = sim3_to_mat(torch.stack([p.reshape(-1) for p in kf_pose_data]))

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

    model = LocalGaussianMap(means, scales, rots, rgb, opas, kf_id).to(dev)
    extent = float((means.max(0).values - means.min(0).values).norm() / 2)
    opt = torch.optim.Adam(model.param_groups(extent), eps=1e-15)
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
    def evaluate(subset=None):
        mses, lps = [], []
        mw, cw = model.world(kf_mats)
        rgb_, op_ = model.rgb(), model.opacity()
        views = held_frames if subset is None else [held_frames[i] for i in subset]
        for c2w, tgt in views:
            pred = render(mw, cw, rgb_, op_, c2w, K, tgt.shape[-2:], dev).clamp(0, 1)
            mses.append(torch.mean((pred - tgt) ** 2).item())
            lps.append(lp(pred * 2 - 1, tgt * 2 - 1).item())
        mse = sum(mses) / len(mses)
        return -10 * math.log10(max(mse, 1e-12)), sum(lps) / len(lps)

    perturb_iters = [args.perturb_kf_at + i * max(1, args.iters // (args.n_perturb + 2))
                     for i in range(args.n_perturb)]
    perturb_iters = [i for i in perturb_iters if i < args.iters]

    seam_overlap, seam_single = (classify_seams() if args.seam_report else (None, None))
    block_classes = None  # (lo, hi, mix) held-out split, set at each block perturbation

    def report(it, extra=""):
        p, l = evaluate()
        line = f"  {tag}iter {it:5} | held-out psnr={p:7.4f}  lpips={l:.4f}"
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
            report(it, f"  <- LOOP-CLOSURE #{n_done} "
                       f"(rot {math.degrees(ang):+.1f} deg, scale {sc:.4f}, "
                       f"|t| {float(tp.norm()):.3f} m)")

        fi = int(rng.integers(len(train_frames)))
        c2w, tgt, _ = train_frames[fi]
        mw, cw = model.world(kf_mats)
        pred = render(mw, cw, model.rgb(), model.opacity(), c2w, K,
                      tgt.shape[-2:], dev)
        loss = ((1 - DSSIM_WEIGHT) * (pred - tgt).abs().mean()
                + DSSIM_WEIGHT * (1 - ssim(pred.clamp(0, 1), tgt, win)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it % args.eval_every == 0:
            report(it, f"  train_loss={loss.item():.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
