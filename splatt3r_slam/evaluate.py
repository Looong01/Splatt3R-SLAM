import pathlib
from typing import Optional
import cv2
import lietorch
import numpy as np
import torch
from splatt3r_slam.dataloader import Intrinsics
from splatt3r_slam.frame import SharedKeyframes
from splatt3r_slam.lietorch_utils import as_SE3
from splatt3r_slam.config import config
from splatt3r_slam.geometry import constrain_points_to_ray
from plyfile import PlyData, PlyElement


def prepare_savedir(args, dataset):
    save_dir = pathlib.Path("logs")
    if args.save_as != "default":
        save_dir = save_dir / args.save_as
    save_dir.mkdir(exist_ok=True, parents=True)
    seq_name = dataset.dataset_path.stem
    return save_dir, seq_name


def save_traj(
    logdir,
    logfile,
    timestamps,
    frames: SharedKeyframes,
    intrinsics: Optional[Intrinsics] = None,
):
    # log
    logdir = pathlib.Path(logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / logfile
    with open(logfile, "w") as f:
        # for keyframe_id in frames.keyframe_ids:
        for i in range(len(frames)):
            keyframe = frames[i]
            t = timestamps[keyframe.frame_id]
            if intrinsics is None:
                T_WC = as_SE3(keyframe.T_WC)
            else:
                T_WC = intrinsics.refine_pose_with_calibration(keyframe)
            x, y, z, qx, qy, qz, qw = T_WC.data.numpy().reshape(-1)
            f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")


class FramePoseLog:
    """Estimated poses for EVERY tracked frame, stored relative to an anchor keyframe.

    save_traj() persists keyframes only -- ~14 of them on TUM desk. That is the
    binding constraint on per-scene refinement: with ground-truth poses, going
    from 14 supervision views to 150 is worth 2.43 dB, the largest single lever
    measured (splatt3r-finetuning-experiments skill, 13.2), and no deployable
    protocol can use it because the poses of the other frames are thrown away.
    They do exist at runtime -- tracker.track() computes frame.T_WC for every
    frame -- they were simply never written down.

    Why anchor-relative rather than world-frame. A non-keyframe's pose is never
    revisited by the pose-graph backend: it is estimated once, against the
    keyframe current at that moment, and then forgotten. A world-frame copy
    taken at tracking time therefore goes stale as soon as loop closure or
    local BA moves the surrounding keyframes, while the Gaussian map does NOT
    go stale -- it is re-baked from each keyframe's *current* T_WC every time
    it is drawn or exported (see SharedKeyframes.gs_means). Mixing the two
    would supervise a corrected map with uncorrected poses, which is precisely
    the error mode the refinement is supposed to remove.

    Storing T_anchor_frame instead keeps them consistent by construction: the
    anchor's correction carries its frames with it, and the world pose is
    recomputed at export time from whatever the anchor's pose has become.

    Recording costs one Sim3 compose per frame and no host sync -- the relative
    pose is kept on the GPU and only moved to the CPU when saving.
    """

    def __init__(self):
        # frame_id -> (anchor_keyframe_idx, T_anchor_frame data (1,8) or None)
        # None marks a frame that became a keyframe itself, whose pose should
        # be read straight from the keyframe buffer (it gets corrected there).
        self._rec = {}

    def record(self, frame_id, keyframes, T_WC_frame):
        """Log a tracked non-keyframe against the keyframe it was tracked from.

        Returns (anchor_idx, rel Sim3 data) so the caller can hand the same
        pair to the refiner's supervision store without recomputing the
        compose; None when there is no anchor yet.
        """
        with keyframes.lock:
            anchor = len(keyframes) - 1
            if anchor < 0:
                return None
            T_WC_anchor = lietorch.Sim3(keyframes.T_WC[anchor])
        rel = (T_WC_anchor.inv() * T_WC_frame).data.detach().clone()
        self._rec[int(frame_id)] = (anchor, rel)
        return anchor, rel

    def record_keyframe(self, frame_id, kf_idx):
        """Log a frame that became keyframe `kf_idx`; its pose is authoritative."""
        self._rec[int(frame_id)] = (int(kf_idx), None)

    def __len__(self):
        return len(self._rec)

    def save(self, logdir, logfile, timestamps, keyframes: SharedKeyframes):
        """Write every logged frame in TUM format, poses resolved against current anchors."""
        logdir = pathlib.Path(logdir)
        logdir.mkdir(exist_ok=True, parents=True)
        n_written = 0
        with keyframes.lock:
            n_kf = len(keyframes)
            with open(logdir / logfile, "w") as f:
                for frame_id in sorted(self._rec):
                    anchor, rel = self._rec[frame_id]
                    if anchor >= n_kf:
                        # The anchor keyframe was rolled back (pop_last) after
                        # this frame was tracked; its pose no longer exists.
                        continue
                    T_WC_anchor = lietorch.Sim3(keyframes.T_WC[anchor])
                    T_WC = T_WC_anchor if rel is None else T_WC_anchor * lietorch.Sim3(rel)
                    x, y, z, qx, qy, qz, qw = as_SE3(T_WC).data.numpy().reshape(-1)
                    f.write(
                        f"{timestamps[frame_id]} {x} {y} {z} {qx} {qy} {qz} {qw}\n"
                    )
                    n_written += 1
        print(
            f"[frame-traj] wrote {n_written}/{len(self._rec)} frame poses "
            f"({n_kf} keyframes) -> {logdir / logfile}"
        )
        return n_written


def save_keyframe_gaussians(path, keyframes: SharedKeyframes):
    """Dump every keyframe's CAMERA-SPACE Gaussians plus its current pose.

    The exported `<seq>_gaussians.ply` is baked into world space and therefore
    loses which keyframe each Gaussian belongs to -- which is precisely the
    information the online refiner needs, because it keeps the map a function
    of the trajectory rather than a fixed artifact (see splatt3r_slam/refiner.py
    and the co-adaptation measurement in the splatt3r-finetuning-experiments
    skill, 13.12b).

    Online the refiner reads these straight out of shared memory; this exists so
    the same code can be validated offline against a finished run.
    """
    import numpy as np

    out = {"n": 0, "keyframes": []}
    with keyframes.lock:
        n = len(keyframes)
        for idx in range(n):
            local = keyframes.get_gaussians_local(idx)
            if local is None:
                continue
            kf = keyframes[idx]
            out["keyframes"].append({
                "kf_idx": idx,
                "frame_id": int(kf.frame_id),
                "T_WC": kf.T_WC.data.detach().cpu().clone(),
                "img": kf.img.detach().cpu().clone(),
                "img_shape": kf.img.shape[-2:],
                **{k: v.detach().cpu().clone() for k, v in local.items()},
            })
        out["n"] = len(out["keyframes"])
    pathlib.Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(out, path)
    print(f"[kf-gaussians] wrote {out['n']} keyframes -> {path}")
    return out["n"]


def save_reconstruction(savedir, filename, keyframes, c_conf_threshold):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    pointclouds = []
    colors = []
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        if config["use_calib"]:
            X_canon = constrain_points_to_ray(
                keyframe.img_shape.flatten()[:2], keyframe.X_canon[None], keyframe.K
            )
            keyframe.X_canon = X_canon.squeeze(0)
        pW = keyframe.T_WC.act(keyframe.X_canon).cpu().numpy().reshape(-1, 3)
        color = (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8).reshape(-1, 3)
        valid = (
            keyframe.get_average_conf().cpu().numpy().astype(np.float32).reshape(-1)
            > c_conf_threshold
        )
        pointclouds.append(pW[valid])
        colors.append(color[valid])
    pointclouds = np.concatenate(pointclouds, axis=0)
    colors = np.concatenate(colors, axis=0)

    save_ply(savedir / filename, pointclouds, colors)


def save_keyframes(savedir, timestamps, keyframes: SharedKeyframes):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(exist_ok=True, parents=True)
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        t = timestamps[keyframe.frame_id]
        filename = savedir / f"{t}.png"
        cv2.imwrite(
            str(filename),
            cv2.cvtColor(
                (keyframe.uimg.cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR
            ),
        )


def save_ply(filename, points, colors):
    colors = colors.astype(np.uint8)
    # Combine XYZ and RGB into a structured array
    pcd = np.empty(
        len(points),
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    pcd["x"], pcd["y"], pcd["z"] = points.T
    pcd["red"], pcd["green"], pcd["blue"] = colors.T
    vertex_element = PlyElement.describe(pcd, "vertex")
    ply_data = PlyData([vertex_element], text=False)
    ply_data.write(filename)


def save_gaussian_map(
    filename,
    keyframes: SharedKeyframes,
    spatial_stride=1,
    depth_max_percentile=0.98,
    max_scale=0.5,
    min_confidence=1.5,
    min_opacity=0.3,
    keyframe_stride=1,
):
    """Persist the accumulated Gaussian map as a standard 3DGS .ply.

    This is the project's headline artifact (its increment over upstream
    MASt3R-SLAM) and previously had no persistence at all: the Gaussians
    lived only in the viz process's GPU memory and vanished with it. The
    `<seq>.ply` written by save_reconstruction() is a *point cloud* --
    positions and colours only -- so it cannot be used to re-render the
    scene from a new viewpoint or opened in a Gaussian-splatting viewer.

    Every keyframe's camera-space Gaussians are baked into world space
    using that keyframe's CURRENT pose (bake_gaussians_world, the same
    function the live viewer uses), so pose-graph corrections up to the
    moment of the call are reflected -- there is no stale copy.

    Output follows the INRIA 3DGS .ply convention, which stores
    PRE-activation values, while this model's head emits POST-activation
    ones (catmlp_dpt_head.py: reg_dense_scales does .exp(),
    reg_dense_opacities does .sigmoid()). Writing them unconverted would
    make any standard viewer apply the activation a second time. So this
    inverts each one on the way out:
      opacity : sigmoid(x) -> logit
      scale   : exp(x)     -> log
      colour  : SH->RGB    -> SH DC coefficient
    Rotation/scale are recovered from the world-space covariance by SVD,
    and the quaternion is reordered scipy's (x,y,z,w) -> 3DGS's (w,x,y,z).
    """
    from splatt3r_slam.splatt3r_utils import bake_gaussians_world
    from splatt3r_slam.gaussian_ply_codec import gaussians_to_ply_element

    all_means, all_cov, all_rgb, all_opa = [], [], [], []
    with keyframes.lock:
        n = len(keyframes)
        # keyframe_stride>1 bakes only every Nth keyframe. Its purpose is the
        # missing cell of a 2x2: the amortized pipeline is only ever measured
        # with ALL keyframes fused (~50 on desk), while per-scene optimization
        # has been measured at 14 and 50 supervision views. Without an
        # {amortized x ~14 keyframes} number there is no way to tell whether
        # the amortized side is information-limited or saturated, and the whole
        # "the bottleneck is amortized inference" reading rests on that
        # distinction. Tracking is unaffected -- this only changes which
        # keyframes are baked into the exported map.
        for kf_idx in range(0, n, keyframe_stride):
            local = keyframes.get_gaussians_local(kf_idx)
            if local is None:
                continue
            keyframe = keyframes[kf_idx]
            _, h, w = keyframe.img.shape[-3:]
            baked = bake_gaussians_world(
                local,
                keyframe.img,
                h,
                w,
                keyframe.T_WC,
                spatial_stride=spatial_stride,
                depth_max_percentile=depth_max_percentile,
                max_scale=max_scale,
                min_confidence=min_confidence,
                min_opacity=min_opacity,
                # This is a persisted artifact, not a viewport draw: the
                # stride-proportional scale inflation inside
                # bake_gaussians_world exists purely so subsampled splats
                # keep visually touching, and would write splats up to
                # `spatial_stride` times too large into the .ply.
                inflate_scales_for_stride=False,
            )
            if baked is None:
                continue
            means, cov_tri, rgb, opa = baked
            all_means.append(means)
            all_cov.append(cov_tri)
            all_rgb.append(rgb)
            all_opa.append(opa)

    if not all_means:
        print(f"[gaussian-map] nothing to export (no keyframes with Gaussians) -- skipping {filename}")
        return 0

    means = torch.cat(all_means).float()
    cov_tri = torch.cat(all_cov).float()
    rgb = torch.cat(all_rgb).float()
    opa = torch.cat(all_opa).float().reshape(-1)

    # The encode path (cov -> SVD U -> sqrt(S) scales, det<0 fix, scipy
    # quaternion, log/logit/f_dc inverse transforms) lives in
    # gaussian_ply_codec so scripts/test_gaussian_ply_roundtrip.py can
    # exercise the exact same code. It also recovers rotation/scale from the
    # world-space covariance by SVD and handles det=-1 reflections.
    # gaussians_to_ply_element additionally appends uchar red/green/blue so
    # generic viewers (MeshLab) show colour instead of a default swatch.
    PlyData([gaussians_to_ply_element(means, cov_tri, rgb, opa)]).write(str(filename))
    n_gauss = means.shape[0]
    print(f"[gaussian-map] wrote {n_gauss} Gaussians -> {filename}")
    return n_gauss
