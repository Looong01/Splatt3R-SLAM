"""Save actual Splatt3R intermediate predictions and recorded TUM trajectories."""
import sys
import json
import hashlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/"scripts")]
import numpy as np
import torch
import lietorch
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = ROOT/"docs/Thesis/ral/fig/overview_assets"

def main():
    from splatt3r_slam.config import load_config
    from splatt3r_slam.dataloader import load_dataset
    from splatt3r_slam.frame import create_frame
    from splatt3r_slam.splatt3r_utils import load_splatt3r, splatt3r_inference_mono, splatt3r_render
    from paper.render_refinement_pair import main as render_pair
    OUT.mkdir(parents=True, exist_ok=True)
    load_config(str(ROOT/"config/eval_calib.yaml"))
    ds = load_dataset(str(ROOT/"datasets/tum/rgbd_dataset_freiburg1_desk"))
    model = load_splatt3r(device="cuda:0")
    frame = create_frame(452, ds.get_image(452), lietorch.Sim3.Identity(1, device="cuda:0"),
                         img_size=ds.img_size, device="cuda:0")
    with torch.no_grad():
        points, confidence = splatt3r_inference_mono(model, frame)
        local = frame.gaussian_pred["means"][0].cpu().numpy()
        rgb = frame.uimg.cpu().numpy()
        Image.fromarray((rgb.clip(0,1)*255).astype("uint8")).save(OUT/"stage_rgb.png")
        z = local[...,2]
        lo,hi = np.quantile(z[np.isfinite(z)&(z>0)], [.02,.98])
        color = plt.get_cmap("viridis")(np.clip((z-lo)/(hi-lo),0,1))[...,:3]
        Image.fromarray((color*255).astype("uint8")).save(OUT/"stage_pointmap.png")
        intrinsics = torch.as_tensor(ds.camera_intrinsics.K_frame, device="cuda:0")
        rendered = splatt3r_render(model, frame, frame, K=intrinsics)
        a = rendered.reshape(-1,3,*rendered.shape[-2:])[0].permute(1,2,0).cpu().numpy()
        Image.fromarray((a.clip(0,1)*255).astype("uint8")).save(OUT/"stage_local_render.png")
        np.savez_compressed(OUT/"stage_prediction.npz", means=local,
                            pointmap=points.cpu().numpy(), confidence=confidence.cpu().numpy())
    del model,frame,rendered
    torch.cuda.empty_cache()
    xyz,colors = local.reshape(-1,3)[::12],rgb.reshape(-1,3)[::12]
    mask = np.isfinite(xyz).all(1)&(xyz[:,2]>lo)&(xyz[:,2]<hi)
    fig = plt.figure(figsize=(4,3),dpi=160)
    ax = fig.add_axes([0,0,1,1],projection="3d")
    ax.scatter(*xyz[mask].T,c=colors[mask],s=.7,linewidths=0)
    ax.view_init(elev=-65,azim=-90)
    ax.set_axis_off()
    fig.savefig(OUT/"stage_local_gaussians.png")
    plt.close(fig)
    run = ROOT/"logs/ref_onoff_tum_on"
    for kind,suffix in [("trajectory","_frames.txt"),("keyframes",".txt")]:
        rows = np.loadtxt(run/("rgbd_dataset_freiburg1_desk"+suffix))
        fig,ax = plt.subplots(figsize=(4,2.6),dpi=160)
        ax.plot(rows[:,1],rows[:,3],color="#376687",lw=1)
        if kind=="keyframes":
            ax.scatter(rows[:,1],rows[:,3],c=np.arange(len(rows)),cmap="viridis",s=6)
        ax.set_aspect("equal",adjustable="datalim")
        ax.axis("off")
        fig.tight_layout(pad=.15)
        fig.savefig(OUT/f"stage_{kind}.png")
        plt.close(fig)
    # Fill the large overview panels with visible data rather than white margins.
    im = Image.open(OUT/"stage_local_gaussians.png").convert("RGB")
    pixels = np.asarray(im)
    ys, xs = np.where(pixels.min(2) < 240)
    crop = (max(0, int(xs.min())-15), max(0, int(ys.min())-15),
            min(im.width, int(xs.max())+16), min(im.height, int(ys.max())+16))
    im.crop(crop).save(OUT/"stage_local_gaussians_tight.png")
    rows = np.loadtxt(run/"rgbd_dataset_freiburg1_desk.txt")
    fig, ax = plt.subplots(figsize=(5,1), dpi=220)
    ax.plot(rows[:,1], rows[:,3], color="#376687", lw=2)
    ax.scatter(rows[:,1], rows[:,3], c=np.arange(len(rows)), cmap="viridis", s=12)
    ax.axis("off")
    fig.subplots_adjust(left=.03, right=.97, bottom=.08, top=.92)
    fig.savefig(OUT/"stage_keyframes_panel.png")
    plt.close(fig)
    render_pair()
    images = sorted(OUT.glob("stage_*.png"))+[OUT/"before.png",OUT/"after.png"]
    (OUT/"system_stages_manifest.json").write_text(json.dumps({
        "scene":"TUM freiburg1_desk","prediction_frame":452,"head":"released",
        "prediction_path":"splatt3r_inference_mono + splatt3r_render",
        "local_gaussian_visual":"predicted Gaussian centres coloured by source RGB",
        "pointmap_visual":"predicted z, 2%-98% colour range",
        "keyframe_visual":"recorded keyframe centres, not inferred graph edges",
        "recorded_run":str(run.relative_to(ROOT)),
        "refinement_protocol":json.loads((OUT/"metadata.json").read_text()),
        "sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in images}
    },indent=2)+"\n")

if __name__=="__main__":
    main()
