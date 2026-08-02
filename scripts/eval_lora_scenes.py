"""Base-vs-LoRA novel-view rendering comparison: one representative scene
from EACH of the four trained families (tum, 7-scenes, euroc, eth3d).

Loads the base checkpoint ONCE. A pristine deepcopy of the base encoder
is kept aside and used for every "base" render; each family's trained
LoRA adapter is loaded with peft.PeftModel.from_pretrained onto a fresh
one-shot throwaway deepcopy of that pristine encoder -- never onto the
shared base, because from_pretrained injects the adapter in-place into
the very object it is given and would otherwise contaminate every
subsequent base render. Adapter selection per family: the final adapter
at checkpoints/lora/<family>/ (written when training completes), else
best/ (lowest val/loss, tracked specifically for eval), else the
highest-numbered epoch_N/ (last resort -- may be post-collapse weights).
For each family's representative scene, a held-out target frame
(from that scene's val split, never trained on) is rendered through the
actual CUDA rasterizer (model.decoder, the same path training/validation
uses) from two other context frames, and compared pixel-for-pixel against
the real ground-truth frame at that pose via MSE/PSNR/LPIPS -- both with
and without that family's LoRA active. Samples whose target view has an
all-zero valid_mask are skipped from the metric averages (they would
report mse/lpips=0, i.e. psnr=inf) and counted separately in the summary.
See the splatt3r-lora-finetuning skill for why this is the right
comparison method (and why an earlier self-view SH-dump comparison
wasn't).

PREREQUISITE: run scripts/train_lora_per_scene.py first -- this script
errors clearly (FileNotFoundError) if a family's adapter directory
doesn't exist yet.

Usage:
    cd /home/share-v5/Codes/Splatt3R-SLAM
    python3 scripts/eval_lora_scenes.py
"""
import copy
import os
import re
import sys

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Derived from this file's own location (scripts/ -> repo root), not
# hardcoded -- the previous absolute path silently broke the moment the
# repo was cloned or moved anywhere else.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE = os.path.join(REPO_ROOT, "splatt3r_core")
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(CORE, "src", "pixelsplat_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src"))
sys.path.insert(0, os.path.join(CORE, "src", "mast3r_src", "dust3r"))
os.chdir(CORE)

import mast3r.utils.path_to_dust3r  # noqa
import numpy as np
import torch
from peft import PeftModel
from PIL import Image

from data.data import DUST3RSplattingTestDataset
from data.eth3d.eth3d import ETH3DData
from data.euroc.euroc import EuRoCData
from data.sevenscenes.sevenscenes import SevenScenesData
from data.tum.tum import TUMData
from lora import MAST3RGaussiansLoRA
from utils import loss_mask as loss_mask_module

DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets")
BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
LORA_ROOT = os.path.join(REPO_ROOT, "checkpoints", "lora")
OUT_DIR = os.path.join(REPO_ROOT, "logs", "lora_eval")
VAL_FRACTION = 0.15

# Must match scripts/train_lora_per_scene.py's FAMILIES, including the
# per-family resolution -- an adapter trained at one resolution has to be
# evaluated at that same resolution for the numbers to mean anything. See
# that file for how each value was derived (it is the resolution SLAM
# inference actually feeds the model for that dataset). Order is (W, H).
FAMILIES = {
    "tum": (TUMData, os.path.join(DATASETS_ROOT, "tum"), {}, (512, 384)),
    # NOTE: temporarily reduced to tum only -- the other three families have
    # no trained adapter yet, and this script raises on a missing one.
}

N_SAMPLES_PER_SCENE = 3
CONTEXT_GAP = 15  # frame-index gap between the two context views


def pick_samples(n_val, n_samples=N_SAMPLES_PER_SCENE, gap=CONTEXT_GAP):
    gap = min(gap, max(2, n_val - 1))
    starts = np.linspace(0, max(0, n_val - gap - 1), n_samples).astype(int)
    samples = []
    for s in starts:
        c1 = int(s)
        c2 = min(c1 + gap, n_val - 1)
        t = (c1 + c2) // 2
        if t != c1 and t != c2:
            samples.append((c1, c2, t))
    return samples


def to_device(view, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in view.items()}


@torch.no_grad()
def render_and_score(model, batch, device):
    view1 = to_device(batch["context"][0], device)
    view2 = to_device(batch["context"][1], device)
    batch_dev = {
        "context": [view1, view2],
        "target": [to_device(tv, device) for tv in batch["target"]],
    }
    _, _, h, w = batch_dev["context"][0]["img"].shape

    pred1, pred2 = model.forward(view1, view2)
    color, _ = model.decoder(batch_dev, pred1, pred2, (h, w))
    mask = loss_mask_module.calculate_loss_mask(batch_dev)
    _, mse, lpips_val = model.calculate_loss(
        batch_dev, view1, view2, pred1, pred2, color, mask,
        apply_mask=True, average_over_mask=True, calculate_ssim=False,
    )
    return color, mse.item(), lpips_val.item()


def save_render(color, path):
    img = color[0, 0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def save_gt(view_target, path):
    img = view_target["original_img"][0].clamp(0, 1).permute(1, 2, 0).float().cpu().numpy()
    Image.fromarray((img * 255).astype(np.uint8)).save(path)


def main():
    device = "cuda"
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading base checkpoint (once, reused across all families)...")
    model = MAST3RGaussiansLoRA.load_from_checkpoint(BASE_CKPT, map_location=device).to(device)
    model.eval()
    # `base_encoder = model.encoder` alone is NOT a pristine reference:
    # PeftModel.from_pretrained(base_encoder, ...) injects the adapter into
    # THAT VERY OBJECT (peft_model.base_model.model is base_encoder), so
    # every subsequent "base" render would silently run through LoRA, and
    # each family would additionally inherit the previous family's adapter.
    # Verified directly rather than assumed -- note a type-name check is
    # useless here, since peft's lora layer class is also called `Linear`;
    # the decisive test is that the same base module's forward() output
    # changes after from_pretrained(). Keeping a deepcopy that is never
    # handed to peft is what actually guarantees an uncontaminated base.
    pristine_encoder = copy.deepcopy(model.encoder)

    results = []

    for family_name, (data_cls, family_root, extra_kwargs, resolution) in FAMILIES.items():
        print(f"\n{'=' * 70}\nEvaluating family: {family_name}\n{'=' * 70}")
        # The final adapter is written to <family>/ only when a family
        # trains to completion. A run that was interrupted, or that
        # exhausted its retries, leaves the directory present but without
        # an adapter_config.json -- an isdir() check alone would pass and
        # peft would then fail with a much less obvious error. Fallback
        # order: best/ BEFORE the newest epoch_N/ -- best/ tracks the
        # lowest val/loss specifically for eval purposes (see
        # train_lora_per_scene.py's find_resume_checkpoint comment), while
        # the newest epoch_N/ may be post-collapse weights: training has
        # previously collapsed and then NOT recovered for a long stretch,
        # so "most recent" is not "best". epoch_N/ is the last resort, for
        # runs that never completed a validation epoch.
        family_root_dir = os.path.join(LORA_ROOT, family_name)

        def _is_adapter(d):
            return os.path.isfile(os.path.join(d, "adapter_config.json"))

        lora_dir = None
        if _is_adapter(family_root_dir):
            lora_dir = family_root_dir
        elif _is_adapter(os.path.join(family_root_dir, "best")):
            lora_dir = os.path.join(family_root_dir, "best")
        elif os.path.isdir(family_root_dir):
            epochs = []
            for name in os.listdir(family_root_dir):
                m = re.match(r"epoch_(\d+)$", name)
                if m and _is_adapter(os.path.join(family_root_dir, name)):
                    epochs.append((int(m.group(1)), os.path.join(family_root_dir, name)))
            if epochs:
                lora_dir = max(epochs)[1]

        if lora_dir is None:
            raise FileNotFoundError(
                f"No usable LoRA adapter under {family_root_dir} (looked for "
                f"adapter_config.json there, in best/, and in epoch_N/) -- "
                f"run scripts/train_lora_per_scene.py first."
            )
        if lora_dir == family_root_dir:
            print(f"  adapter: final adapter at {lora_dir}")
        else:
            print(f"  note: no final adapter; evaluating {os.path.basename(lora_dir)}/ instead ({lora_dir})")
        # best/ carries a val_loss.txt sidecar written by the training
        # script's best-tracking callback -- print it when present so a
        # human can judge how good the selected weights actually are.
        val_loss_sidecar = os.path.join(lora_dir, "val_loss.txt")
        if os.path.isfile(val_loss_sidecar):
            with open(val_loss_sidecar) as f:
                print(f"  val_loss.txt: {f.read().strip()}")

        val_data = data_cls(family_root, "val", val_fraction=VAL_FRACTION, **extra_kwargs)
        if len(val_data.sequences) == 0:
            raise RuntimeError(f"No usable val sequences for family '{family_name}' under {family_root}.")
        # Representative scene: just the first one found. Edit this line
        # to pick a specific sequence by name instead, e.g.:
        #   scene = "rgbd_dataset_freiburg1_desk" if family_name == "tum" else val_data.sequences[0]
        scene = val_data.sequences[0]
        n_val = len(val_data.color_paths[scene])
        print(f"  representative scene: {scene} ({n_val} val frames)")

        scene_out_dir = os.path.join(OUT_DIR, family_name, scene.replace("/", "_"))
        os.makedirs(scene_out_dir, exist_ok=True)

        samples = pick_samples(n_val)
        print(f"  evaluating {len(samples)} held-out samples: {samples}")
        test_samples = [(scene, c1, c2, t) for c1, c2, t in samples]
        test_ds = DUST3RSplattingTestDataset(val_data, test_samples, resolution)
        loader = torch.utils.data.DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)

        mse_base_list, lpips_base_list = [], []
        mse_lora_list, lpips_lora_list = [], []
        n_skipped_empty = 0

        # One throwaway copy per family, so this family's adapter can never
        # leak into the pristine base or into the next family's evaluation.
        lora_encoder = PeftModel.from_pretrained(
            copy.deepcopy(pristine_encoder), lora_dir
        ).to(device)
        lora_encoder.eval()

        for i, batch in enumerate(loader):
            # data.py keeps samples whose views have an all-zero valid_mask
            # (assert -> warn; e.g. an all-zero pseudo-depth frame from
            # euroc/eth3d). For the TARGET view that means an all-zero
            # loss mask, and calculate_loss's clamp(min=1000) denominator
            # then reports mse=lpips=0 -- a "perfect" sample that would
            # poison the averages (psnr=inf in the extreme). Skip those
            # samples from the metric accumulation entirely and count them
            # separately.
            if not bool(batch["target"][0]["valid_mask"].any()):
                n_skipped_empty += 1
                print(f"  sample {i}: skipped (empty target valid_mask)")
                continue

            model.encoder = pristine_encoder
            color_base, mse_base, lpips_base = render_and_score(model, batch, device)
            mse_base_list.append(mse_base)
            lpips_base_list.append(lpips_base)

            model.encoder = lora_encoder
            color_lora, mse_lora, lpips_lora = render_and_score(model, batch, device)
            mse_lora_list.append(mse_lora)
            lpips_lora_list.append(lpips_lora)

            save_render(color_base, os.path.join(scene_out_dir, f"sample{i}_base.png"))
            save_render(color_lora, os.path.join(scene_out_dir, f"sample{i}_lora.png"))
            save_gt(to_device(batch["target"][0], device), os.path.join(scene_out_dir, f"sample{i}_gt.png"))

            print(
                f"  sample {i}: base mse={mse_base:.4f} lpips={lpips_base:.4f}  |  "
                f"lora mse={mse_lora:.4f} lpips={lpips_lora:.4f}"
            )

        if n_skipped_empty:
            print(f"  skipped {n_skipped_empty} empty-mask sample(s)")

        if not mse_base_list:
            # Every sample had an empty mask -- report no metrics rather
            # than a nan/inf row that looks like data.
            print(f"  WARNING: all {len(samples)} samples had an empty target valid_mask; no metrics for this scene")
            results.append((family_name, scene, None, None, None, None, None, None, n_skipped_empty))
        else:
            mse_base_avg = float(np.mean(mse_base_list))
            lpips_base_avg = float(np.mean(lpips_base_list))
            mse_lora_avg = float(np.mean(mse_lora_list))
            lpips_lora_avg = float(np.mean(lpips_lora_list))
            psnr_base_avg = -10.0 * np.log10(mse_base_avg)
            psnr_lora_avg = -10.0 * np.log10(mse_lora_avg)

            results.append((family_name, scene, mse_base_avg, psnr_base_avg, lpips_base_avg, mse_lora_avg, psnr_lora_avg, lpips_lora_avg, n_skipped_empty))
            print(f"  scene avg: base  mse={mse_base_avg:.4f} psnr={psnr_base_avg:.2f} lpips={lpips_base_avg:.4f}")
            print(f"             lora  mse={mse_lora_avg:.4f} psnr={psnr_lora_avg:.2f} lpips={lpips_lora_avg:.4f}")
            print(f"  renders saved to {scene_out_dir}/")

        # Release this family's throwaway encoder copy before building the
        # next one -- otherwise all four accumulate on the GPU.
        model.encoder = pristine_encoder
        del lora_encoder
        torch.cuda.empty_cache()

    print(f"\n{'=' * 100}\nSUMMARY (lower mse/lpips = better, higher psnr = better)\n{'=' * 100}")
    header = f"{'family':<12}{'scene':<35}{'mse(base)':>10}{'mse(lora)':>10}{'psnr(base)':>12}{'psnr(lora)':>12}{'lpips(base)':>12}{'lpips(lora)':>12}"
    print(header)

    def _fmt(v, width, prec):
        return f"{'n/a':>{width}}" if v is None else f"{v:>{width}.{prec}f}"

    total_skipped = 0
    for family_name, scene, mb, pb, lb, ml, pl, ll, n_skipped in results:
        total_skipped += n_skipped
        print(
            f"{family_name:<12}{scene:<35}"
            f"{_fmt(mb, 10, 4)}{_fmt(ml, 10, 4)}"
            f"{_fmt(pb, 12, 2)}{_fmt(pl, 12, 2)}"
            f"{_fmt(lb, 12, 4)}{_fmt(ll, 12, 4)}"
        )
    if total_skipped:
        print(f"skipped {total_skipped} empty-mask samples (all-zero target valid_mask; excluded from the averages above)")


if __name__ == "__main__":
    main()
