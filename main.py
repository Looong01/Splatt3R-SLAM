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
import numpy as np
import torch
import tqdm
import yaml
import sys
import os

# torch >= 2.6 defaults torch.load to weights_only=True, which rejects the
# pickled objects (omegaconf DictConfig, argparse Namespace, faiss indices)
# inside the MASt3R/Splatt3R checkpoints. These checkpoints come from trusted
# sources, so restore the old default globally.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

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
    create_frame,
)
from splatt3r_slam.splatt3r_utils import (
    load_splatt3r,
    load_retriever,
    splatt3r_inference_mono,
    splatt3r_render,
)
from splatt3r_slam.multiprocess_utils import new_queue, try_get_msg
from splatt3r_slam.refiner import RefinedMapSnapshot, SupervisionFrames, run_refiner
from splatt3r_slam.retrieval_dump import RetrievalFeatureDumper
from splatt3r_slam.tracker import FrameTracker
from splatt3r_slam.visualization import WindowMsg, run_visualization
import torch.multiprocessing as mp


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
    retrieval_database = load_retriever(model, retriever_path=cfg.get("retriever_path"))

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
        # NOTE: update() must always be called, even with no_loop_closure on:
        # besides querying candidates it inserts this keyframe into the
        # retrieval database (add_after_query=True), which relocalization
        # (Mode.RELOC) depends on. We only discard the returned loop
        # candidates so no loop-closure edges enter the factor graph.
        retrieval_inds = retrieval_database.update(
            frame,
            add_after_query=True,
            k=config["retrieval"]["k"],
            min_thresh=config["retrieval"]["min_thresh"],
        )
        # config.get() fallback: old config yamls have no such key.
        if config.get("no_loop_closure", False):
            retrieval_inds = []
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


# The one thinning dose that ships on. 0.45 is the top of the measured range
# (skill 17.66.10), NOT a located optimum -- conf-fade's advantage over uniform
# was still growing at the last dose tested, and doses above 0.5 both go
# untested and stop being dose-matched, because the 0.1 floor begins to clip.
CONF_FADE_DEFAULT = 0.45


if __name__ == "__main__":
    mp.set_start_method("spawn")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    device = "cuda:0"
    save_frames = False
    datetime_now = str(datetime.datetime.now()).replace(" ", "_")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-keyframe-gaussians", action="store_true",
                        help="also write <seq>_kfgauss.pt: each keyframe's "
                             "camera-space Gaussians and current pose, which "
                             "the baked .ply cannot express")
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
        "--map-keyframe-stride",
        type=int,
        default=1,
        help="bake only every Nth keyframe into the exported Gaussian map. "
             "Used to measure the amortized pipeline at a reduced view count; "
             "does not affect tracking or the trajectory.",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="Path to a fine-tuned Gaussian-head state_dict (e.g. "
        "checkpoints/head_only/tum/head_best.pt) to load on top of "
        "--checkpoint. This is the fine-tuning route measured to beat the "
        "released weights; the encoder is left untouched. See the "
        "splatt3r-finetuning-experiments skill.",
    )
    parser.add_argument(
        "--lora",
        default=None,
        help="Path to a trained LoRA adapter directory (e.g. "
        "checkpoints/lora/<scene>/) to hot-load on top of --checkpoint. "
        "NEGATIVE RESULT: this route measured 49%% worse than the released "
        "checkpoint; kept only to reproduce that. Prefer --head. "
        "Requires `pip install peft`.",
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
        default=16 * 1024 * 1024,
        help="Target Gaussian render budget across all keyframes (default: 16777216). "
        "The live per-keyframe baker coarsens its stride to stay under this. "
        "Raise further for a denser map at the cost of more VRAM/render time.",
    )
    parser.add_argument(
        "--spatial-stride",
        type=int,
        default=1,
        help="Spatial stride for subsampling Gaussians per frame (default: 1, full "
        "per-pixel density). WARNING: this setting -- and stride=2 too, just less "
        "reliably -- has been observed to crash the vendored CUDA gaussian "
        "rasterizer with an illegal memory access once enough Gaussians accumulate. "
        "The 'spatial stride' GUI slider adjusts this live if you hit instability; "
        "see the splatt3r-gaussian-map skill for what's been tried.",
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
        "--no-loop-closure",
        action="store_true",
        help="Disable loop-closure edges produced by retrieval-database candidate "
        "queries in the backend pose graph. Relocalization (reloc) still queries "
        "the database and is unaffected. Intended for ablation experiments.",
    )
    parser.add_argument(
        "--retriever-path",
        default=None,
        help="Path to a retrieval whitening .pth (with a matching "
        "<name>_codebook.pkl sibling, see splatt3r-retrieval-refit skill "
        "section 5). Default None reproduces current behaviour exactly: "
        "the original MASt3R retrieval assets.",
    )
    parser.add_argument(
        "--dump-retrieval-features",
        default=None,
        metavar="DIR",
        help="Dump each keyframe's retrieval feature (frame.feat, the raw "
        "Splatt3R encoder feature fed to the retrieval head) to DIR/<seq_name>/ "
        "as feat_<kf_idx>.npy + metadata.jsonl, for refitting the retrieval "
        "whitening/codebook.",
    )
    parser.add_argument(
        "--frame-timing",
        default=None,
        metavar="CSV",
        help="Write per-frame wall-clock timing (mode, tracking ms, backend-wait "
             "ms, total iteration ms) to CSV, for GPU-contention measurement "
             "(experiment (g) in the splatt3r-finetuning-experiments skill). "
             "The INIT frame is not logged.",
    )
    parser.add_argument(
        "--refiner",
        action="store_true",
        help="Run the online map-refinement process (P1 stage 4): optimizes a "
             "keyframe-local Gaussian map in the background, supervised by "
             "anchor-carried tracked frames, and writes <seq>_refined.ply at "
             "the end. Requires use_calib (a fixed K to render through).",
    )
    parser.add_argument(
        "--refiner-streak-opacity",
        type=float,
        default=0.5,
        help="hide ray-elongated Gaussians at injection: opacity *= "
             "min(1, K*pitch/max_scale). ON by default at K=0.5: -10.4%% lpips "
             "ONLINE (17.42), -5.0%% offline (17.32). "
             "(skill 17.32) and NOT yet tested online, which after 17.21 is the "
             "only thing that could license it as a default. Requires "
             "--refiner-aa-sigma > 0: the criterion is defined against the "
             "lattice pitch and there is no sound fallback without it.",
    )
    parser.add_argument(
        "--refiner-polish-tol",
        type=float,
        default=0.0,
        help="stop the polish phase once the training loss improves by less "
             "than this fraction over one logging window, instead of burning "
             "the full --refiner-polish-secs. Sequences need 2.4x different "
             "step counts (desk 1581, room 1856, 360 3821; skill 17.25), so a "
             "fixed duration either wastes time or starves the hard ones. "
             "0 = off, which is what ships until this is measured.",
    )
    parser.add_argument(
        "--refiner-uniform-fade",
        type=float,
        default=0.0,
        help="multiply every injected opacity by (1-D). The measured form of "
             "the thinning prior (skill 17.56): the elongation selector it "
             "replaces was shown to contribute nothing, and on a deep fade to "
             "be actively worse. Arm it when the map's accumulated alpha is "
             "high and the head has not un-saturated itself. See "
             "--refiner-conf-fade for a strictly better allocation of the same "
             "dose. (An earlier version of this help claimed a measured lattice "
             "cost from 17.59; that was RETRACTED in 17.63 -- the images show "
             "thinning REDUCES the visible lattice.)",
    )
    parser.add_argument(
        "--refiner-conf-fade",
        type=float,
        default=None,               # resolved to CONF_FADE_DEFAULT below
        help=f"thin by the same mean dose as --refiner-uniform-fade but allocate "
             f"it by the backbone's own confidence: opacity *= "
             f"1-D*2*(1-rank(conf)), floored at 0.1. DEFAULT "
             f"{CONF_FADE_DEFAULT}; pass 0 to disable. WHAT IS ACTUALLY "
             f"ESTABLISHED: turning thinning ON is decisive -- across 8 Replica "
             f"scenes, faded beats unfaded by -32%% to -34%% lpips, paired "
             f"p<1e-4. Choosing THIS allocation over the uniform one is NOT: "
             f"over 12 maps on two families conf wins 7, mean -1.7%%, MEDIAN "
             f"-0.06%%, 95%% CI -4.0%% to +0.6%% -- the interval contains zero "
             f"and both paired tests sit at p>0.2 (17.78). It is the default "
             f"because it is the configuration the deployment A/Bs were run "
             f"with and it is never much worse (worst case +2.9%%), NOT because "
             f"it is measurably better. --refiner-uniform-fade "
             f"{CONF_FADE_DEFAULT} is an equally defensible choice and is "
             f"simpler. Do not spend effort tuning between them; spend it on "
             f"whether thinning is on at all, which is where the effect is. The dose {CONF_FADE_DEFAULT} is the top of the measured "
             f"range and not a located optimum; doses above 0.5 are untested "
             f"AND stop being dose-matched to uniform, "
             f"because the 0.1 floor starts clipping. Supersedes "
             f"--refiner-uniform-fade; setting that one explicitly turns this "
             f"off.",
    )
    parser.add_argument(
        "--refiner-scale-cap",
        type=float,
        default=0.0,
        help="clamp injected Gaussian scales to at most this many metres, "
             "anchored to the BASE checkpoint's p90 on the deployment family. "
             "Measured -3.5%% to -16.7%% lpips across four families, but it COSTS "
             "psnr in every one (-0.1 to -0.5 dB) and raises black fraction, so "
             "it is not a free win. Do NOT try to predict whether it pays from "
             "the head's p90 ratio: that diagnostic was tested on four cells and "
             "carries no predictive information (17.67, 17.68). Measure it per "
             "map with an offline on/off arm.",
    )
    parser.add_argument(
        "--refiner-polish-patience",
        type=int,
        default=3,
        help="how many consecutive windows must sit below --refiner-polish-tol "
             "before the polish stops. 1 was measured too trigger-happy: it "
             "fired on a window whose mean loss had risen while the map was "
             "still improving.",
    )
    parser.add_argument(
        "--refiner-unfreeze-in-polish",
        action="store_true",
        help="restore the positional learning rate when the post-sequence "
             "polish phase starts. Only meaningful with --refiner-freeze-means. "
             "The decisive test for skill 17.22's conflict-averaging "
             "explanation of the online lpips deficit.",
    )
    parser.add_argument(
        "--refiner-freeze-means",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="hold the Gaussian centres fixed. ON by default, and measured "
             "optimal at every budget tested: at the ~300 in-sequence steps a "
             "large map gets, letting positions move buys 0.35 dB psnr and "
             "costs 9.7%% lpips (skill 17.12/17.13/17.14). Pass "
             "--no-refiner-freeze-means to let them move.",
    )
    parser.add_argument(
        "--refiner-min-confidence",
        type=float,
        default=1.5,
        help="confidence threshold at injection for the REFINER's map. 0 by "
             "default, which is the opposite of the 4.0 that skill 16.9 "
             "recommended: that result was measured at matched wall clock while "
             "throughput was binding, and exact culling removed that constraint. "
             "At the ~300 in-sequence steps a large map gets, 0.0 beats 1.5 by "
             "+0.43 dB with 43%% less unmapped area, and 4.0 loses 1.94 dB "
             "(skill 17.16). Offline runs with thousands of steps should raise "
             "it -- 4.0 takes psnr there, at 4%% of lpips.",
    )
    parser.add_argument(
        "--refiner-aa-sigma",
        type=float,
        default=0.5,
        help="band limit at injection, as a multiple of each Gaussian's own "
             "lattice pitch, held as a constraint. 0 disables. Without it the "
             "map is perforated between its own sample points and every render "
             "above the source sampling rate shows a halftone lattice (17.2).",
    )
    parser.add_argument(
        "--refiner-duty",
        type=float,
        default=0.25,
        help="Refiner GPU duty cycle (default 0.25). An unthrottled refiner "
             "doubles tracker latency on one card (skill 15.2); the process "
             "sleeps between iterations to hold this share of its unthrottled "
             "rate. Steps are dropped, never frames.",
    )
    parser.add_argument(
        "--refiner-gpu",
        type=int,
        default=-1,
        help="GPU index the refiner COMPUTES on (default -1 = same card as "
             "SLAM). (g) measured cross-GPU contention as noise, so on a "
             "two-card box '--refiner-gpu 1' (launched with both cards "
             "visible) moves the render loop off the tracking card; only "
             "small per-keyframe copies still touch it.",
    )
    parser.add_argument(
        "--refiner-dedup-voxel",
        type=float,
        default=0.0,
        help="Voxel edge (m) for the de-clustering lifecycle (skill 15.7): "
             "when the refined map exceeds --refiner-max-gaussians after a "
             "keyframe injection, shared voxels keep only the earliest "
             "keyframe's Gaussians. 0 disables.",
    )
    parser.add_argument(
        "--refiner-max-gaussians",
        type=int,
        default=4_000_000,
        help="Map size that triggers a dedup pass (default 4M).",
    )
    parser.add_argument(
        "--save-gs-view",
        default=None,
        metavar="DIR",
        help="Save the interactive 3DGS map render (what the GUI window "
             "shows) to DIR as PNGs, for offline analysis. Writes every "
             "--gs-view-stride-th rendered frame as gs_map_%%08d.png.",
    )
    parser.add_argument(
        "--gs-view-stride",
        type=int,
        default=1,
        help="Save every Nth GUI-rendered frame (default 1 = every frame).",
    )
    parser.add_argument(
        "--gs-scale-inflate",
        type=float,
        default=1.0,
        help="Display-only uniform splat-size multiplier (default 1.0). "
             "~1.15-1.3 closes sub-pixel gaps (black speckle) when the map "
             "is rendered above its capture resolution and low-passes moiré. "
             "Never written into the map or exports; the GUI 'GS scale "
             "inflate' slider adjusts it live.",
    )
    parser.add_argument(
        "--refiner-polish-secs",
        type=float,
        default=0.0,
        help="After the dataset ends, keep the refiner optimizing with the "
             "final poses for this many seconds before termination (default "
             "0 = stop immediately). The measured win of online refinement "
             "concentrates in the first few hundred steps (skill 15.1), and "
             "big maps are supervision-starved at sequence end -- this is "
             "the cheap way to buy them back.",
    )

    args = parser.parse_args()

    # Validate flag combinations here, not in the refiner subprocess. The
    # streak lever needs the lattice pitch, which is only computed when the
    # band limit is on; getting that wrong used to raise inside `run_refiner`,
    # which killed the refiner and let the rest of the run finish looking
    # normal -- a full sequence with no refined map and no obvious reason why.
    if args.refiner_streak_opacity > 0 and args.refiner_aa_sigma <= 0:
        parser.error(
            "--refiner-streak-opacity needs --refiner-aa-sigma > 0: the "
            "elongation criterion is defined against the measured lattice "
            "pitch, which is only computed when the band limit is on. "
            "--refiner-aa-sigma 0.5 is the measured setting.")
    # --refiner-gpu indexes VISIBLE devices, so under CUDA_VISIBLE_DEVICES=1 the
    # only valid value is 0. Getting this wrong raises "invalid device ordinal"
    # INSIDE the refiner subprocess, which kills the refiner while SLAM runs to
    # completion and writes every artifact except the refined map -- a full
    # sequence that looks successful and silently measures the unrefined
    # baseline. Same failure shape as the --refiner-streak-opacity case below,
    # and the third time this class has cost a run.
    if args.refiner:
        import torch as _t
        _n = _t.cuda.device_count()
        if args.refiner_gpu is not None and args.refiner_gpu != -1 and not (0 <= args.refiner_gpu < _n):
            parser.error(
                f"--refiner-gpu {args.refiner_gpu} is not a visible device: "
                f"torch sees {_n} ({', '.join(f'cuda:{i}' for i in range(_n))}). "
                f"Note this indexes VISIBLE devices -- with CUDA_VISIBLE_DEVICES "
                f"set to a single card, the only valid value is 0.")
    # conf-fade is on by default, so a bare `--refiner-uniform-fade 0.45` must
    # not trip the mutual-exclusion check -- the user asked for one lever, not
    # two. Explicitly choosing the uniform allocation turns the default off;
    # only setting BOTH explicitly is the error, because the two multiply and
    # the combined dose lands well past where thinning turns harmful (17.56).
    if args.refiner_uniform_fade > 0:
        if args.refiner_conf_fade is not None and args.refiner_conf_fade > 0:
            parser.error(
                "--refiner-conf-fade and --refiner-uniform-fade are two "
                "allocations of the SAME thinning dose, not two levers to "
                "stack. Together they multiply, so the effective dose is far "
                "past what either was measured at and past the point where "
                "thinning turns harmful (17.56). Pick one; --refiner-conf-fade "
                "was measured better or equal on every map tested "
                "(17.66.7-17.66.11).")
        args.refiner_conf_fade = 0.0
    elif args.refiner_conf_fade is None:
        args.refiner_conf_fade = CONF_FADE_DEFAULT
    if args.refiner_polish_tol > 0 and args.refiner_polish_secs <= 0:
        parser.error("--refiner-polish-tol only does something during the "
                     "polish phase; set --refiner-polish-secs as well.")
    if args.refiner_unfreeze_in_polish and not args.refiner_freeze_means:
        parser.error("--refiner-unfreeze-in-polish requires "
                     "--refiner-freeze-means; there is nothing to unfreeze.")

    load_config(args.config)
    # Injected before the backend process is spawned below, so run_backend
    # picks it up via set_global_config(cfg).
    config["no_loop_closure"] = args.no_loop_closure
    config["retriever_path"] = args.retriever_path
    print(args.dataset)
    print(config)

    manager = mp.Manager()
    # Set when the post-sequence polish phase begins. The refiner is a separate
    # process and cannot otherwise tell -- main just sleeps while it runs on.
    polish_flag = manager.Value("i", 0)
    polish_done = manager.Value("i", 0)
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

    # Load Splatt3R model instead of MASt3R. Loaded before SharedKeyframes so
    # we know the model's SH degree and can size the per-keyframe local
    # Gaussian storage buffer correctly (see gs_sh_dim below).
    print("Loading Splatt3R model...")
    model = load_splatt3r(path=args.checkpoint, device=device,
                          lora_path=args.lora, head_path=args.head)
    model.share_memory()
    gs_sh_dim = getattr(model.encoder, "sh_degree", 1)

    keyframes = SharedKeyframes(manager, h, w, gs_sh_dim=gs_sh_dim)
    states = SharedStates(manager, h, w)

    # Refiner-side shared structures are created before the viz process so
    # the viewer can subscribe to the refined-map snapshot channel from the
    # start (the refiner process itself spawns after the backend).
    sup_frames = None
    snapshot = None
    if args.refiner:
        sup_frames = SupervisionFrames(manager, h, w)
        snapshot = RefinedMapSnapshot(
            manager, args.refiner_max_gaussians + 1_000_000)

    if not args.no_viz:
        viz = mp.Process(
            target=run_visualization,
            args=(config, states, keyframes, main2viz, viz2main),
            kwargs=dict(
                spatial_stride=args.spatial_stride,
                max_gaussians=args.max_gaussians,
                depth_max_percentile=args.depth_max_percentile,
                max_scale=args.max_scale,
                min_confidence=args.min_confidence,
                keyframe_stride=args.map_keyframe_stride,
                refined_snapshot=snapshot,
                save_gs_view=args.save_gs_view,
                gs_view_stride=args.gs_view_stride,
                gs_scale_inflate=args.gs_scale_inflate,
            ),
        )
        viz.start()

    has_calib = dataset.has_calib()
    use_calib = config["use_calib"]

    if use_calib and not has_calib:
        # This check runs AFTER viz.start() above, so a bare sys.exit()
        # here would leave the viz process running forever (it does not
        # act on Mode.TERMINATED on its own) -- a batch/eval script would
        # hang. Tear it down explicitly, and exit nonzero: this is a
        # misconfiguration, and exiting 0 makes shell `&&` chains and CI
        # treat a failed run as successful.
        print("[Error] No calibration provided for this dataset!", file=sys.stderr)
        if not args.no_viz:
            viz.terminate()
            viz.join()
        sys.exit(1)
    K = None
    if use_calib:
        K = torch.from_numpy(dataset.camera_intrinsics.K_frame).to(
            device, dtype=torch.float32
        )
        keyframes.set_intrinsics(K)

    # remove the trajectory from the previous run
    if dataset.save_results:
        save_dir, seq_name = eval.prepare_savedir(args, dataset)
        # Include the Gaussian map: if this run ends up not writing one
        # (crash, or nothing survived filtering), a stale file from a
        # previous run would otherwise sit next to freshly-written
        # trajectory/reconstruction files and look current.
        for stale in (
            save_dir / f"{seq_name}.txt",
            save_dir / f"{seq_name}.ply",
            save_dir / f"{seq_name}_gaussians.ply",
            save_dir / f"{seq_name}_refined.ply",
        ):
            if stale.exists():
                stale.unlink()

    tracker = FrameTracker(model, keyframes, device)
    frame_pose_log = eval.FramePoseLog()
    last_msg = WindowMsg(
        spatial_stride=args.spatial_stride, max_gaussians=args.max_gaussians
    )

    # Optional per-keyframe retrieval-feature dump (main process, at the two
    # keyframes.append() sites below). seq_name matches prepare_savedir().
    retrieval_dumper = None
    if args.dump_retrieval_features:
        retrieval_dumper = RetrievalFeatureDumper(
            args.dump_retrieval_features, dataset.dataset_path.stem
        )

    # Per-frame Gaussian-rendered PNG preview (novel-view render from the
    # frame's own just-computed local Gaussians -- independent of the
    # accumulated map, which the viz process now owns entirely).
    render_gaussians = args.render_gaussians and not args.no_render_gaussians
    render_dir = None
    if render_gaussians:
        render_dir = pathlib.Path(args.render_dir)
        render_dir.mkdir(exist_ok=True, parents=True)
        print(f"[Gaussian Rendering] Enabled. Saving to {render_dir}")
    print(
        f"[Gaussians] map baking now lives in the viz process "
        f"(max_gaussians={args.max_gaussians}, spatial_stride={args.spatial_stride}, "
        f"depth_max_pct={args.depth_max_percentile}, max_scale={args.max_scale}, "
        f"min_conf={args.min_confidence})"
    )

    backend = mp.Process(target=run_backend, args=(config, model, states, keyframes, K))
    backend.start()

    # Online map refinement (P1 stage 4): a third worker process optimizing a
    # keyframe-local map against anchor-carried supervision. Needs a fixed K
    # to render through, so it is calib-only.
    refiner = None
    sup_rng = np.random.default_rng(1)
    if args.refiner:
        if K is None:
            print("[refiner] requires use_calib (fixed intrinsics); disabled",
                  file=sys.stderr)
        else:
            refined_path = None
            if dataset.save_results:
                _sd, _sn = eval.prepare_savedir(args, dataset)
                refined_path = str(_sd / f"{_sn}_refined.ply")
            refiner_device = (
                f"cuda:{args.refiner_gpu}" if args.refiner_gpu >= 0 else None)
            refiner = mp.Process(
                target=run_refiner,
                args=(config, states, keyframes, sup_frames, K.cpu().numpy()),
                kwargs=dict(save_path=refined_path,
                            duty_cycle=args.refiner_duty,
                            device=refiner_device,
                            dedup_voxel=args.refiner_dedup_voxel,
                            max_gaussians=args.refiner_max_gaussians,
                            aa_sigma=args.refiner_aa_sigma,
                            min_confidence=args.refiner_min_confidence,
                            streak_opacity=args.refiner_streak_opacity,
                            freeze_means=args.refiner_freeze_means,
                            polish_flag=polish_flag,
                            unfreeze_in_polish=args.refiner_unfreeze_in_polish,
                            polish_done=polish_done,
                            polish_tol=args.refiner_polish_tol,
                            polish_patience=args.refiner_polish_patience,
                            uniform_fade=args.refiner_uniform_fade,
                            conf_fade=args.refiner_conf_fade,
                            scale_cap=args.refiner_scale_cap,
                            snapshot=snapshot),
            )
            refiner.start()
            print(f"[refiner] started (duty_cycle={args.refiner_duty}, "
                  f"device={refiner_device or 'same'}, save={refined_path})")

    i = 0
    fps_timer = time.time()

    frames = []

    # Set to True only once the normal teardown below (drain -> TERMINATED
    # -> backend.join -> save -> viz.join) has completed. The finally
    # block uses it to tell a clean exit apart from an exception unwinding
    # the main loop (KeyboardInterrupt, or e.g. an IndexError from the
    # frame pipeline): only the latter needs best-effort process cleanup,
    # otherwise the backend/viz processes would be left running as
    # orphans.
    clean_exit = False
    timing_f = open(args.frame_timing, "w") if args.frame_timing else None
    if timing_f is not None:
        timing_f.write("frame,timestamp,mode,track_ms,backend_wait_ms,iter_ms\n")
    try:
        while True:
            # Watchdog: if the backend process dies (e.g. OOM in the retriever),
            # fail fast instead of blocking forever on queues it will never
            # drain -- the finally block below then reaps any remaining
            # children instead of leaving GPU-memory-holding orphans.
            if not backend.is_alive():
                raise RuntimeError("backend process died unexpectedly")
            mode = states.get_mode()
            msg = try_get_msg(viz2main)
            last_msg = msg if msg is not None else last_msg

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
                if not (refiner is not None and args.refiner_polish_secs > 0):
                    states.set_mode(Mode.TERMINATED)
                break

            timestamp, img = dataset[i]
            t_iter0 = time.perf_counter()
            t_track = 0.0
            t_wait = 0.0
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
                frame_pose_log.record_keyframe(i, len(keyframes) - 1)
                if sup_frames is not None:
                    sup_frames.offer(
                        (frame.uimg * 255).round().to(torch.uint8),
                        len(keyframes) - 1,
                        torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.]),
                        sup_rng)
                states.queue_global_optimization(len(keyframes) - 1)
                if retrieval_dumper is not None:
                    retrieval_dumper.dump(len(keyframes) - 1, frame, timestamp)
                states.set_mode(Mode.TRACKING)
                states.set_frame(frame)

                # Local (camera-space) Gaussians were just written into
                # frame.gaussian_pred by splatt3r_inference_mono() above, and
                # keyframes.append() (via SharedKeyframes.__setitem__) already
                # persisted them into the keyframe's shared-memory slot. The
                # viz process bakes them to world space itself, live, using
                # this keyframe's current T_WC -- nothing more to do here.
                if render_gaussians:
                    rendered = splatt3r_render(model, frame, frame, K=K)
                    if rendered is not None:
                        rendered_img = rendered[0, 0].cpu().clamp(0, 1).permute(1, 2, 0)
                        rendered_np = (rendered_img.numpy() * 255).astype("uint8")
                        rendered_bgr = cv2.cvtColor(rendered_np, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(
                            str(render_dir / f"gs_init_{i:06d}.png"), rendered_bgr
                        )

                i += 1
                continue

            if mode == Mode.TRACKING:
                _t = time.perf_counter()
                add_new_kf, match_info, try_reloc = tracker.track(frame)
                t_track += time.perf_counter() - _t
                if try_reloc:
                    states.set_mode(Mode.RELOC)
                states.set_frame(frame)

                # Every tracked frame's pose, kept relative to the keyframe it
                # was tracked against (see FramePoseLog). Frames that go on to
                # become keyframes are re-tagged below, since their own pose
                # then gets corrected by the backend and is the better source.
                if not try_reloc:
                    anchor_rel = frame_pose_log.record(i, keyframes, frame.T_WC)
                    # The same anchor-relative pair feeds the refiner's
                    # supervision store: its poses re-compose through the
                    # anchor's CURRENT pose at sample time, so supervision
                    # follows loop closures (skill 15.5).
                    if sup_frames is not None and anchor_rel is not None:
                        sup_frames.offer(
                            (frame.uimg * 255).round().to(torch.uint8),
                            anchor_rel[0], anchor_rel[1], sup_rng)

                # If this frame becomes a keyframe, tracker.track() already
                # attached its local-space gaussian_pred; keyframes.append()
                # below persists it. Non-keyframe tracked frames never enter
                # the map (only real keyframe poses are ever corrected by the
                # pose-graph backend, so only keyframe-tagged Gaussians can be
                # safely re-baked -- see the splatt3r-gaussian-map skill).
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
                _t = time.perf_counter()
                while config["single_thread"]:
                    if not backend.is_alive():
                        raise RuntimeError("backend process died unexpectedly")
                    with states.lock:
                        if states.reloc_sem.value == 0:
                            break
                    time.sleep(0.01)
                t_wait += time.perf_counter() - _t

            else:
                raise Exception("Invalid mode")

            if add_new_kf:
                keyframes.append(frame)
                frame_pose_log.record_keyframe(i, len(keyframes) - 1)
                states.queue_global_optimization(len(keyframes) - 1)
                if sup_frames is not None:
                    # A new keyframe supervises with its own pose: anchor =
                    # itself, rel = identity Sim3 (txyz, quat xyzw, scale).
                    sup_frames.offer(
                        (frame.uimg * 255).round().to(torch.uint8),
                        len(keyframes) - 1,
                        torch.tensor([0., 0., 0., 0., 0., 0., 1., 1.]),
                        sup_rng)
                if retrieval_dumper is not None:
                    retrieval_dumper.dump(len(keyframes) - 1, frame, timestamp)
                # In single threaded mode, wait for the backend to finish
                _t = time.perf_counter()
                while config["single_thread"]:
                    if not backend.is_alive():
                        raise RuntimeError("backend process died unexpectedly")
                    with states.lock:
                        if len(states.global_optimizer_tasks) == 0:
                            break
                    time.sleep(0.01)
                t_wait += time.perf_counter() - _t
            # log time
            if i % 30 == 0:
                FPS = i / (time.time() - fps_timer)
                print(f"FPS: {FPS}")
            if timing_f is not None:
                timing_f.write(
                    f"{i},{timestamp},{mode.name},{t_track * 1e3:.3f},"
                    f"{t_wait * 1e3:.3f},{(time.perf_counter() - t_iter0) * 1e3:.3f}\n"
                )
                if i % 50 == 0:
                    timing_f.flush()
            i += 1

        # Shut the backend down and join it BEFORE reading keyframes out for
        # saving. The backend runs global optimization asynchronously and
        # writes its corrections back into the shared `keyframes`; saving
        # first (as this used to) serializes a pose graph missing whatever
        # optimization was still outstanding when the dataset ran out.
        #
        # Joining alone is NOT enough: run_backend loops on
        # `while mode is not Mode.TERMINATED`, checked at the top of each
        # iteration, so setting TERMINATED makes it abandon everything still
        # queued in states.global_optimizer_tasks -- join() would then only
        # cover the single task already in flight. Drain the queue first
        # (same wait the single_thread path above uses), then terminate.
        #
        # Bounded, because an already-dead backend would never drain it and
        # this must not hang a finished run forever.
        #
        # If the viz GUI is paused, the backend idles in its pause branch
        # and never consumes global_optimizer_tasks -- the drain below
        # would then sit out the full 120s timeout doing nothing. Unpause
        # first so the backend can finish whatever is still queued.
        states.unpause()
        drain_deadline = time.time() + 120.0
        while time.time() < drain_deadline:
            with states.lock:
                if len(states.global_optimizer_tasks) == 0:
                    break
            if not backend.is_alive():
                print("[warn] backend exited with optimization tasks still queued", file=sys.stderr)
                break
            time.sleep(0.01)
        else:
            with states.lock:
                n_left = len(states.global_optimizer_tasks)
            print(
                f"[warn] timed out draining backend queue ({n_left} task(s) left); "
                f"saved results may omit the last pose corrections",
                file=sys.stderr,
            )

        # Post-sequence polish: with the backend drained (poses final) but the
        # refiner still alive, keep optimizing for --refiner-polish-secs. This
        # is where big, supervision-starved maps buy their quality back.
        if refiner is not None and args.refiner_polish_secs > 0:
            print(f"[refiner] polish phase: {args.refiner_polish_secs:.0f}s "
                  f"with final poses", flush=True)
            polish_flag.value = 1
            polish_end = time.time() + args.refiner_polish_secs
            while (time.time() < polish_end and refiner.is_alive()
                   and not polish_done.value):
                time.sleep(0.5)

        states.set_mode(Mode.TERMINATED)
        backend.join()
        if refiner is not None:
            # Join AFTER the backend: the refiner's final map save composes
            # through the keyframes' poses, and those are only final once the
            # drained backend has written its last corrections.
            refiner.join(timeout=120)
            if refiner.is_alive():
                print("[warn] refiner did not exit in 120s; terminating",
                      file=sys.stderr)
                refiner.terminate()
                refiner.join(timeout=10)
            # Publication-side check of the viewer snapshot channel.
            print(f"[refiner] snapshot v{snapshot.version.value}, "
                  f"{snapshot.count.value:,} gaussians published")

        if dataset.save_results:
            save_dir, seq_name = eval.prepare_savedir(args, dataset)
            eval.save_traj(save_dir, f"{seq_name}.txt", dataset.timestamps, keyframes)
            # Same trajectory, but every tracked frame rather than the ~14
            # keyframes -- the supervision-view bottleneck for per-scene
            # refinement. Poses are resolved against their anchor keyframe's
            # CURRENT (post-loop-closure) pose at this point, so this file and
            # the Gaussian map written below share one frame of reference.
            frame_pose_log.save(
                save_dir, f"{seq_name}_frames.txt", dataset.timestamps, keyframes
            )
            eval.save_reconstruction(
                save_dir,
                f"{seq_name}.ply",
                keyframes,
                last_msg.C_conf_threshold,
            )
            eval.save_keyframes(
                save_dir / "keyframes" / seq_name, dataset.timestamps, keyframes
            )
            # The Gaussian map itself -- this project's actual output, and
            # until now the only one that was never persisted (the .ply above
            # is a plain point cloud: positions/colours, no covariance,
            # opacity or SH, so it cannot be re-rendered from a new view).
            # Written in standard 3DGS .ply form so external viewers can open
            # it directly. Uses the same CLI knobs as the live renderer so the
            # export matches what was on screen.
            if args.dump_keyframe_gaussians:
                # Camera-space Gaussians per keyframe, for the online refiner's
                # offline validation path (see splatt3r_slam/refiner.py).
                eval.save_keyframe_gaussians(
                    save_dir / f"{seq_name}_kfgauss.pt", keyframes
                )
            eval.save_gaussian_map(
                save_dir / f"{seq_name}_gaussians.ply",
                keyframes,
                spatial_stride=args.spatial_stride,
                depth_max_percentile=args.depth_max_percentile,
                max_scale=args.max_scale,
                min_confidence=args.min_confidence,
            )
        if save_frames:
            savedir = pathlib.Path(f"logs/frames/{datetime_now}")
            savedir.mkdir(exist_ok=True, parents=True)
            for i, frame in tqdm.tqdm(enumerate(frames), total=len(frames)):
                frame = (frame * 255).clip(0, 255)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{savedir}/{i}.png", frame)

        print("done")
        if not args.no_viz:
            # Normal path: block until the user closes the viz window.
            viz.join()
        clean_exit = True
    finally:
        if timing_f is not None:
            timing_f.close()
        if not clean_exit:
            # Exception path only. Best-effort, idempotent teardown: each
            # step is individually guarded so cleanup can never mask the
            # original exception, and each is a no-op once the process it
            # targets is already dead. (On the normal path this whole
            # block is skipped via clean_exit; even if it ran, the backend
            # would already be joined and the viz process already reaped
            # after the user closed the window.)
            try:
                states.set_mode(Mode.TERMINATED)
            except Exception:
                pass
            try:
                if backend.is_alive():
                    backend.join(timeout=30)
                    if backend.is_alive():
                        backend.terminate()
                        backend.join(timeout=10)
            except Exception:
                pass
            if refiner is not None:
                try:
                    if refiner.is_alive():
                        refiner.terminate()
                        refiner.join(timeout=10)
                except Exception:
                    pass
            if not args.no_viz:
                try:
                    if viz.is_alive():
                        viz.terminate()
                        viz.join(timeout=10)
                except Exception:
                    pass
