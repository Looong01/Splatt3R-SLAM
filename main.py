"""
Main script for Splatt3R-SLAM
This version uses Splatt3R (with Gaussian Splatting) instead of MASt3R.
"""

import argparse
import datetime
import pathlib
import time
import warnings
import cv2
import lietorch
import torch
import tqdm
import yaml
import sys
from typing import Optional

# Suppress known safe warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*weights_only.*", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=".*The parameter 'pretrained' is deprecated.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore", message=".*Arguments other than a weight enum.*", category=UserWarning
)
from splatt3r_slam.global_opt import FactorGraph

from splatt3r_slam.config import load_config, config, set_global_config
from splatt3r_slam.dataloader import Intrinsics, load_dataset
import splatt3r_slam.evaluate as eval
from splatt3r_slam.frame import (
    Mode,
    SharedKeyframes,
    SharedStates,
    SharedGaussians,
    create_frame,
)
from splatt3r_slam.splatt3r_utils import (
    load_splatt3r,
    load_retriever,
    splatt3r_inference_mono,
    splatt3r_render,
    gaussians_to_world,
)
from splatt3r_slam.multiprocess_utils import new_queue, try_get_msg
from splatt3r_slam.tracker import FrameTracker
from splatt3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp


def should_append_gaussians(
    add_new_kf: bool,
    frame_idx: int,
    current_T_WC: lietorch.Sim3,
    last_append_T_WC: Optional[lietorch.Sim3],
    last_append_frame_idx: int,
    min_translation: float,
    min_frame_gap: int,
) -> bool:
    if add_new_kf:
        return True
    if last_append_T_WC is None:
        return True
    if (frame_idx - last_append_frame_idx) < min_frame_gap:
        return False

    t_cur = current_T_WC.matrix()[0, :3, 3]
    t_last = last_append_T_WC.matrix()[0, :3, 3]
    translation = torch.linalg.norm(t_cur - t_last).item()
    return translation >= min_translation


def relocalization(frame, keyframes, factor_graph, retrieval_database):
    # we are adding and then removing from the keyframe, so we need to be careful.
    # The lock slows viz down but safer this way...
    with keyframes.lock:
        kf_idx = []
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=False,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds
        successful_loop_closure = False
        if kf_idx:
            keyframes.append(frame)
            n_kf = len(keyframes)
            kf_idx = list(kf_idx)  # convert to list
            frame_idx = [n_kf - 1] * len(kf_idx)
            print("RELOCALIZING against kf ", n_kf - 1, " and ", kf_idx)
            if factor_graph.add_factors(
                frame_idx,
                kf_idx,
                config["reloc"]["min_match_frac"],
                is_reloc=config["reloc"]["strict"],
            ):
                retrieval_database.update(
                    frame,
                    add_after_query=True,
                    k=config["retrieval"]["k"],
                    min_thresh=config["retrieval"]["min_thresh"],
                )
                print("Success! Relocalized")
                successful_loop_closure = True
                keyframes.T_WC[n_kf - 1] = keyframes.T_WC[kf_idx[0]].clone()
            else:
                keyframes.pop_last()
                print("Failed to relocalize")

        if successful_loop_closure:
            if config["use_calib"]:
                factor_graph.solve_GN_calib()
            else:
                factor_graph.solve_GN_rays()
        return successful_loop_closure


def run_backend(cfg, model, states, keyframes, K):
    set_global_config(cfg)

    device = keyframes.device
    factor_graph = FactorGraph(model, keyframes, K, device)
    retrieval_database = load_retriever(model)

    mode = states.get_mode()
    while mode is not Mode.TERMINATED:
        mode = states.get_mode()
        if mode == Mode.INIT or states.is_paused():
            time.sleep(0.01)
            continue
        if mode == Mode.RELOC:
            frame = states.get_frame()
            success = relocalization(frame, keyframes, factor_graph, retrieval_database)
            if success:
                states.set_mode(Mode.TRACKING)
            states.dequeue_reloc()
            continue
        idx = -1
        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks[0]
        if idx == -1:
            time.sleep(0.01)
            continue

        # Graph Construction
        kf_idx = []
        # k to previous consecutive keyframes
        n_consec = 1
        for j in range(min(n_consec, idx)):
            kf_idx.append(idx - 1 - j)
        frame = keyframes[idx]
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        kf_idx += retrieval_inds

        lc_inds = set(retrieval_inds)
        lc_inds.discard(idx - 1)
        if len(lc_inds) > 0:
            print("Database retrieval", idx, ": ", lc_inds)

        kf_idx = set(kf_idx)  # Remove duplicates by using set
        kf_idx.discard(idx)  # Remove current kf idx if included
        kf_idx = list(kf_idx)  # convert to list
        frame_idx = [idx] * len(kf_idx)
        if kf_idx:
            factor_graph.add_factors(
                kf_idx, frame_idx, config["local_opt"]["min_match_frac"]
            )

        with states.lock:
            states.edges_ii[:] = factor_graph.ii.cpu().tolist()
            states.edges_jj[:] = factor_graph.jj.cpu().tolist()

        if config["use_calib"]:
            factor_graph.solve_GN_calib()
        else:
            factor_graph.solve_GN_rays()

        with states.lock:
            if len(states.global_optimizer_tasks) > 0:
                idx = states.global_optimizer_tasks.pop(0)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/tum/rgbd_dataset_freiburg1_desk")
    parser.add_argument("--config", default="config/base.yaml")
    parser.add_argument("--save-as", default="default")
    parser.add_argument("--no-viz", action="store_true")
    parser.add_argument("--calib", default="")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to Splatt3R checkpoint (downloads if not provided)",
    )
    parser.add_argument(
        "--render-gaussians",
        action="store_true",
        default=True,
        help="Enable Gaussian Splatting rendering and save per-frame PNGs (default: True)",
    )
    parser.add_argument(
        "--no-render-gaussians",
        action="store_true",
        help="Disable Gaussian Splatting rendering and per-frame PNG saving",
    )
    parser.add_argument(
        "--render-dir",
        default="logs/gaussian_renders",
        help="Directory to save Gaussian-rendered images (default: logs/gaussian_renders)",
    )
    parser.add_argument(
        "--max-gaussians",
        type=int,
        default=4 * 1024 * 1024,
        help="Max number of Gaussians in shared buffer (default: 4194304)",
    )
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=4,
        help="Spatial stride for subsampling Gaussians per frame (default: 4, stride=1 means no subsampling)",
    )
    parser.add_argument(
        "--depth-max-percentile",
        type=float,
        default=0.98,
        help="Filter out Gaussians deeper than this depth percentile (default: 0.98). "
        "Set 1.0 to disable depth filtering.",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=1.0,
        help="Remove Gaussians whose max scale axis exceeds this value (default: 1.0). "
        "Large scales indicate hallucinated splash artifacts.",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=1.5,
        help="Remove Gaussians at pixels with pointmap confidence below this (default: 1.5). "
        "Set 0 to disable confidence filtering.",
    )
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.6,
        help="Keep top N%% of Gaussians per frame ranked by quality score (default: 0.6). "
        "1.0 disables the quality-percentile filter.",
    )

    args = parser.parse_args()

    load_config(args.config)
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    main2viz = new_queue(manager, args.no_viz)
    viz2main = new_queue(manager, args.no_viz)

    dataset = load_dataset(args.dataset)
    dataset.subsample(config["dataset"]["subsample"])
    h, w = dataset.get_img_shape()[0]

    if args.calib:
        with open(args.calib, "r") as f:
            intrinsics = yaml.load(f, Loader=yaml.SafeLoader)
        config["use_calib"] = True
        dataset.use_calibration = True
        dataset.camera_intrinsics = Intrinsics.from_calib(
            dataset.img_size,
            intrinsics["width"],
            intrinsics["height"],
            intrinsics["calibration"],
        )

    keyframes = SharedKeyframes(manager, h, w)
    states = SharedStates(manager, h, w)
    shared_gaussians = SharedGaussians(manager, max_gaussians=args.max_gaussians)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, shared_gaussians, main2viz, viz2main),
            kwargs=dict(
                spatial_stride=args.spatial_stride,
                max_gaussians=args.max_gaussians,
                depth_max_percentile=args.depth_max_percentile,
                max_scale=args.max_scale,
                min_confidence=args.min_confidence,
                keep_ratio=args.keep_ratio,
            ),
        )
        viz.start()

    # Load Splatt3R model instead of MASt3R
    print("Loading Splatt3R model...")
    model = load_splatt3r(path=args.checkpoint, device=device)
    model.share_memory()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        print("[Warning] No calibration provided for this dataset!")
        sys.exit(0)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        traj_file = save_dir / f"{seq_name}.txt"
        recon_file = save_dir / f"{seq_name}.ply"
        if traj_file.exists():
            traj_file.unlink()
        if recon_file.exists():
            recon_file.unlink()

    tracker = FrameTracker(model, keyframes, device)
    last_msg = WindowMsg(
        spatial_stride=args.spatial_stride,
        max_gaussians=args.max_gaussians,
        depth_max_percentile=args.depth_max_percentile,
        max_scale=args.max_scale,
        min_confidence=args.min_confidence,
        keep_ratio=args.keep_ratio,
    )

    # Gaussian rendering setup
    render_gaussians = args.render_gaussians and not args.no_render_gaussians
    spatial_stride = args.spatial_stride
    # Avoid repeatedly appending near-identical Gaussians from almost the same view.
    gs_append_min_translation = 0.12  # meters
    gs_append_min_frame_gap = 3  # frames
    last_gs_append_T_WC = None
    last_gs_append_frame_idx = -(10**9)
    render_dir = None
    if render_gaussians:
        render_dir = pathlib.Path(args.render_dir)
        render_dir.mkdir(exist_ok=True, parents=True)
        print(f"[Gaussian Rendering] Enabled. Saving to {render_dir}")
    print(
        f"[Gaussians] max_gaussians={args.max_gaussians}, spatial_stride={spatial_stride}"
    )
    print(
        f"[Gaussians] splash filter: depth_max_pct={args.depth_max_percentile}, "
        f"max_scale={args.max_scale}, min_conf={args.min_confidence}, "
        f"keep_ratio={args.keep_ratio}"
    )

    # Enable gaussian splatting visualization whenever the viz window is active
    enable_gs_viz = not args.no_viz

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    i = 0
    fps_timer = time.time()

    frames = []

    while True:
        mode = states.get_mode()
        msg = try_get_msg(viz2main)
        last_msg = msg if msg is not None else last_msg

        # Update runtime params from GUI sliders
        spatial_stride = last_msg.spatial_stride
        shared_gaussians.max_gaussians = last_msg.max_gaussians
        # Splash-filter params (live from GUI or CLI defaults in headless mode)
        depth_max_pct = last_msg.depth_max_percentile
        gs_max_scale = last_msg.max_scale
        gs_min_conf = last_msg.min_confidence
        gs_keep_ratio = last_msg.keep_ratio

        if last_msg.is_terminated:
            states.set_mode(Mode.TERMINATED)
            break

        if last_msg.is_paused and not last_msg.next:
            states.pause()
            time.sleep(0.01)
            continue

        if not last_msg.is_paused:
            states.unpause()

        if i == len(dataset):
            states.set_mode(Mode.TERMINATED)
            break

        timestamp, img = dataset[i]
        if save_frames:
            frames.append(img)

        # get frames last camera pose
        T_WC = (
            lietorch.Sim3.Identity(1, device=device)
            if i == 0
            else states.get_frame().T_WC
        )
        frame = create_frame(i, img, T_WC, img_size=dataset.img_size, device=device)

        if mode == Mode.INIT:
            # Initialize via mono inference with Splatt3R
            X_init, C_init = splatt3r_inference_mono(model, frame)
            frame.update_pointmap(X_init, C_init)
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            states.set_mode(Mode.TRACKING)
            states.set_frame(frame)

            # --- Gaussian Splatting: accumulate world-space Gaussians ---
            if enable_gs_viz or render_gaussians:
                gs_world = gaussians_to_world(
                    frame,
                    include_cross=False,
                    spatial_stride=spatial_stride,
                    depth_max_percentile=depth_max_pct,
                    max_scale=gs_max_scale,
                    min_confidence=gs_min_conf,
                    keep_ratio=gs_keep_ratio,
                )
                if gs_world is not None:
                    means_w, cov_w, colors_w, opas_w, quality_w = gs_world
                    if enable_gs_viz:
                        shared_gaussians.replace_in_voxels(
                            means_w, quality_w, voxel_size=0.05
                        )
                        shared_gaussians.append(
                            means_w,
                            cov_w,
                            colors_w,
                            opas_w,
                            kf_idx=len(keyframes) - 1,
                            opacity_threshold=0.3,
                            quality=quality_w,
                        )
                        last_gs_append_T_WC = frame.T_WC
                        last_gs_append_frame_idx = i
                    if render_gaussians:
                        rendered = splatt3r_render(model, frame, frame, K=K)
                        if rendered is not None:
                            rendered_img = (
                                rendered[0, 0].cpu().clamp(0, 1).permute(1, 2, 0)
                            )
                            rendered_np = (rendered_img.numpy() * 255).astype("uint8")
                            rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(
                                str(render_dir / f"gs_init_{i:06d}.png"), rendered_bgr
                            )

            i += 1
            continue

        if mode == Mode.TRACKING:
            add_new_kf, match_info, try_reloc = tracker.track(frame)
            if try_reloc:
                states.set_mode(Mode.RELOC)
            states.set_frame(frame)

            should_append = should_append_gaussians(
                add_new_kf=add_new_kf,
                frame_idx=i,
                current_T_WC=frame.T_WC,
                last_append_T_WC=last_gs_append_T_WC,
                last_append_frame_idx=last_gs_append_frame_idx,
                min_translation=gs_append_min_translation,
                min_frame_gap=gs_append_min_frame_gap,
            )

            # --- Gaussian Splatting: accumulate world-space Gaussians every tracked frame ---
            if (enable_gs_viz or render_gaussians) and not try_reloc and should_append:
                gs_world = gaussians_to_world(
                    frame,
                    include_cross=False,
                    spatial_stride=spatial_stride,
                    depth_max_percentile=depth_max_pct,
                    max_scale=gs_max_scale,
                    min_confidence=gs_min_conf,
                    keep_ratio=gs_keep_ratio,
                )
                if gs_world is not None:
                    means_w, cov_w, colors_w, opas_w, quality_w = gs_world
                    if enable_gs_viz:
                        shared_gaussians.replace_in_voxels(
                            means_w, quality_w, voxel_size=0.05
                        )
                        shared_gaussians.append(
                            means_w,
                            cov_w,
                            colors_w,
                            opas_w,
                            kf_idx=len(keyframes),
                            opacity_threshold=0.3,
                            quality=quality_w,
                        )
                        last_gs_append_T_WC = frame.T_WC
                        last_gs_append_frame_idx = i
            if render_gaussians and not try_reloc:
                keyframe = keyframes.last_keyframe()
                if keyframe is not None:
                    rendered = splatt3r_render(
                        model,
                        frame,
                        keyframe,
                        K=K,
                        target_T_WC=frame.T_WC,
                    )
                    if rendered is not None:
                        rendered_img = rendered[0, 0].cpu().clamp(0, 1).permute(1, 2, 0)
                        rendered_np = (rendered_img.numpy() * 255).astype("uint8")
                        rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(
                            str(render_dir / f"gs_track_{i:06d}.png"), rendered_bgr
                        )

        elif mode == Mode.RELOC:
            X, C = splatt3r_inference_mono(model, frame)
            frame.update_pointmap(X, C)
            states.set_frame(frame)
            states.queue_reloc()
            # In single threaded mode, make sure relocalization happen for every frame
            while config["single_thread"]:
                with states.lock:
                    if states.reloc_sem.value == 0:
                        break
                time.sleep(0.01)

        else:
            raise Exception("Invalid mode")

        if add_new_kf:
            keyframes.append(frame)
            states.queue_global_optimization(len(keyframes) - 1)
            # In single threaded mode, wait for the backend to finish
            while config["single_thread"]:
                with states.lock:
                    if len(states.global_optimizer_tasks) == 0:
                        break
                time.sleep(0.01)
        # log time
        if i % 30 == 0:
            FPS = i / (time.time() - fps_timer)
            print(f"FPS: {FPS}")
        i += 1

    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
        eval.save_reconstruction(
            save_dir,
            f"{seq_name}.ply",
            keyframes,
            last_msg.C_conf_threshold,
        )
        eval.save_keyframes(
            save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
        )
    if save_frames:
        savedir = pathlib.Path(f"logs/frames/{datetime_now}")
        savedir.mkdir(exist_ok=True, parents=True)
        for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
            frame = (frame * 255).clip(0, 255)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{savedir}/{i}.png", frame)

    print("done")
    backend.join()
    if not args.no_viz:
        viz.join()
