"""Train one adapter/head per DATASET FAMILY (tum, 7-scenes, euroc, eth3d),
pooling ALL locally-downloaded sequences/scenes within each family into a
single training set. See the splatt3r-lora-finetuning skill for the full
design rationale, the earlier single-scene proof-of-concept this replaces,
and important caveats (especially for euroc/eth3d's self-predicted
pseudo-depth).

TWO MODES, selected by HEAD_ONLY below:
  - HEAD_ONLY=True (current): control experiment after the LoRA
    fine-tuning was judged a failure. Frozen encoder, only the Gaussian
    dpt heads train (upstream's own recipe, no peft), plain state_dict
    checkpoints under checkpoints/headonly/<family>/.
  - HEAD_ONLY=False: the historical LoRA mode (peft adapters on the
    encoder + trainable heads), checkpoints under checkpoints/lora/.

PREREQUISITE for euroc and eth3d: run scripts/precompute_pseudo_depth.py
first. tum and 7-scenes need no preprocessing (real sensor depth).

Usage:
    cd /home/share-v5/Codes/Splatt3R-SLAM
    python3 scripts/train_lora_per_scene.py                  # all 4 families, one process
    python3 scripts/train_lora_per_scene.py tum 7-scenes      # just these

RECOMMENDED instead, for the real dual-GPU run: scripts/train_lora_all_families.sh,
which launches a fresh `python3` process per family rather than looping
internally -- see that script's header for why (repeated DDP process-group
init/teardown across sequential trainer.fit() calls in one process was
never actually run end-to-end in this repo).

Trains sequentially, one family at a time, each on BOTH GPUs via DDP
(devices=2 below -- edit if your machine doesn't have 2 GPUs). Each
family gets its own independently-initialized LoRA (not continued/shared
across families). Safe to Ctrl-C between families -- a family's adapter
is written after its training finishes AND periodically during training
(see CHECKPOINT_EVERY_N_EPOCHS below -- important for a run this long:
a family interrupted mid-training can resume-worthy state from its last
periodic checkpoint under checkpoints/lora/<family>/epoch_<N>/, though
this script itself doesn't auto-resume from one -- that's a manual
`ckpt_path=` addition to trainer.fit() if/when needed).

BATCH_SIZE below is a reasoned estimate for dual RTX A6000 (48GB each,
96GB total), NOT empirically profiled at this resolution/model -- watch
`nvidia-smi` during the first few minutes of a real run and adjust up
(more headroom) or down (OOM) before trusting it for an unattended
multi-hour/day run.
"""
import os
import pickle
import re
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
# The last OOM (BATCH_SIZE=1, bf16-mixed) reported 2.40GiB "reserved by
# PyTorch but unallocated" alongside the failed 4.85GiB ask with only
# 3.89GiB free -- real fragmentation, not just "not enough memory total"
# (3.89 + 2.40 > 4.85). expandable_segments lets the allocator grow/reuse
# existing reserved segments instead of needing one fresh contiguous
# block, which is exactly what the torch.OutOfMemoryError message itself
# suggests for this pattern. Must be set before the first CUDA context
# init, hence before `import torch` below. Zero behavior change if there's
# no fragmentation to reclaim -- safe to leave on regardless of whether it
# fully fixes this specific OOM.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Lightning's DDP subprocess launcher re-execs this script for every extra
# GPU rank via `[sys.executable, os.path.abspath(sys.argv[0])] + sys.argv[1:]`
# (lightning/fabric/strategies/launchers/subprocess_script.py:_basic_subprocess_cmd),
# and that os.path.abspath() call happens lazily, inside trainer.fit() --
# i.e. AFTER the os.chdir(CORE) below has already changed the cwd. Since we're
# always invoked as a cwd-relative path ("scripts/train_lora_per_scene.py"),
# resolving it against the wrong (post-chdir) cwd silently points at a
# nonexistent file (`splatt3r_core/scripts/...`), so rank 1+ fails to even
# start and Lightning kills rank 0 along with it -- this is why a DEVICES=2
# run was observed stuck on a single GPU with low VRAM for its whole
# pre-training setup phase and then crashing the instant it reached
# trainer.fit(). Resolving to an absolute path now, before the chdir, makes
# the later abspath() call in Lightning a no-op on an already-correct path.
sys.argv[0] = os.path.abspath(sys.argv[0])

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
import lightning as L
import torch

from data.common import compute_coverage
from data.data import DUST3RSplattingDataset
from data.eth3d.eth3d import ETH3DData
from data.euroc.euroc import EuRoCData
from data.sevenscenes.sevenscenes import SevenScenesData
from data.replica.replica import ReplicaData
from data.tum.tum import TUMData
from lora import MAST3RGaussiansLoRA, attach_lora
from main import MAST3RGaussians

DATASETS_ROOT = os.path.join(REPO_ROOT, "datasets")
BASE_CKPT = os.path.join(REPO_ROOT, "checkpoints", "epoch=19-step=1200.ckpt")
LORA_OUT_ROOT = os.path.join(REPO_ROOT, "checkpoints", "lora")

# --- Head-only control experiment ----------------------------------------
# HEAD_ONLY=True: control experiment after the LoRA fine-tuning experiment
# was judged a failure (adapter ~49% worse than base, a geometric-level
# collapse). Working hypothesis: LoRA unfroze the encoder that upstream
# Splatt3R deliberately freezes (main.py: MAST3RGaussians.__init__
# requires_grad_(False) on the encoder, True only on the Gaussian dpt
# heads), while the frozen ~158M-param matching/pointmap head depends on
# the ORIGINAL encoder feature distribution. This mode therefore trains
# the upstream-validated recipe instead: everything frozen except the two
# Gaussian dpt heads, no peft at all, scale penalty off (lora.py
# SCALE_PENALTY_ENABLED=False -- it was a patch for a LoRA-specific
# pathology, i.e. a confound to remove). Checkpoints are plain state_dicts
# of the trainable head params under checkpoints/headonly/<family>/ (kept
# separate from checkpoints/lora/ so the two checkpoint formats never
# mix). Set False to reproduce the historical LoRA runs.
HEAD_ONLY = True
HEADONLY_OUT_ROOT = os.path.join(REPO_ROOT, "checkpoints", "headonly")
# Under DDP, rank 1+ re-execs this whole script from scratch (see the
# sys.argv[0] fix above) and would otherwise redo the expensive coverage
# computation independently and in parallel with rank 0 -- wasted GPU time
# that doubles the pre-training setup latency for every family. Rank 0
# always finishes computing (and caches) coverage before it reaches
# trainer.fit(), i.e. before rank 1 is even spawned, so rank 1 always finds
# a warm cache here instead. Deterministic given the same downloaded data
# (see data/common.py: split_train_val, compute_coverage -- no randomness),
# so sharing this way across ranks is safe. NOTE: the cache is keyed on
# VAL_FRACTION/COVERAGE_POS_THRESHOLD but NOT on the downloaded data itself
# -- delete checkpoints/lora_coverage_cache/ if you add/remove sequences
# under datasets/<family>/.
COVERAGE_CACHE_ROOT = os.path.join(REPO_ROOT, "checkpoints", "lora_coverage_cache")

# name -> (Data class, family root dir, extra kwargs, resolution)
#
# Resolution is PER-FAMILY, and each value is exactly what SLAM inference
# will feed the model for that dataset -- not a free choice. Derived by
# replaying splatt3r_slam/splatt3r_utils.py:712 resize_img(size=512)
# (resize long edge to 512, then crop both dims down to a multiple of 16)
# against each dataset's real on-disk image size:
#
#   family     native      aspect   -> inference res   tokens
#   tum        640x480     1.333       512x384          768
#   7-scenes   640x480     1.333       512x384          768
#   euroc      752x480     1.567       512x320          640
#   eth3d      739x458     1.614       512x304          608
#   (previously a single global 512x512 for all             1024)
#
# Sizes verified against the actual files, not assumed: all 15 ETH3D
# scenes, all 7 sampled 7-Scenes sequences and all 11 EuRoC sequences
# report one consistent size each.
#
# Why per-family rather than one global value: the four families have
# three different aspect ratios, so no single resolution can match all of
# them -- and matching matters here specifically because each family gets
# its own LoRA that is hot-swapped in at inference time for that dataset.
# Training a family's adapter at a different resolution than the one it
# will be deployed at puts it on a different input distribution than it
# was tuned for. Training one family per process (see
# train_lora_all_families.sh) makes per-family resolution free to do.
# Bonus: every one of these is cheaper than the old 512x512 -- attention
# memory scales with token_count^2, so these run at ~56%/56%/39%/35% of
# the old attention footprint.
#
# ORDER IS (W, H), not (H, W) -- confirmed by reading the consumer:
# dust3r/datasets/utils/cropping.py:61 does
# `input_resolution = np.array(image.size)  # (W,H)` and compares the
# requested resolution against it directly.
FAMILIES = {
    "tum": (TUMData, os.path.join(DATASETS_ROOT, "tum"), {}, (512, 384)),
    "7-scenes": (SevenScenesData, os.path.join(DATASETS_ROOT, "7-scenes"), {}, (512, 384)),
    "euroc": (EuRoCData, os.path.join(DATASETS_ROOT, "euroc"), {}, (512, 320)),
    "eth3d": (ETH3DData, os.path.join(DATASETS_ROOT, "eth3d"), {"max_scenes": 15}, (512, 304)),  # max_scenes matches precompute_pseudo_depth.py's MAX_ETH3D_SCENES
    # Replica (NICE-SLAM rendered release): exact poses, exact complete depth,
    # constant exposure. The only family here without sensor noise in the
    # supervision, and the benchmark the recent Gaussian-SLAM literature
    # reports on. 1200x680 source, so the same 4:3-ish 512x384 as tum.
    # 512x288, NOT 512x384. Replica is 1200x680 (16:9), so the SLAM side's
    # resize_img (long side 512, crop to a multiple of 16) feeds the network
    # 512x288. Training at 512x384 centre-crops to 4:3 instead, i.e. a
    # different crop of a different aspect -- measured cost of getting this
    # wrong: +1.84 dB on the val split, -4.0 dB on the baked map in an actual
    # SLAM run (17.55). euroc's (512,320) and eth3d's (512,304) are each their
    # family's deployment shape for exactly this reason.
    "replica": (ReplicaData, os.path.join(DATASETS_ROOT, "Replica"), {}, (512, 288)),
    "replica-noisy": (ReplicaData, os.path.join(DATASETS_ROOT, "Replica"),
                      {"degrade": "content"}, (512, 288)),
    "replica-photo": (ReplicaData, os.path.join(DATASETS_ROOT, "Replica"),
                      {"degrade": "photometry"}, (512, 288)),
}

# --- Training hyperparameters -------------------------------------------
# Tuned for dual RTX A6000 (2x48GB=96GB VRAM, 32 CPU cores) -- see the
# splatt3r-lora-finetuning skill for the upstream configs/main.yaml
# cross-check most of these were originally based on, and for why
# BATCH_SIZE/NUM_WORKERS specifically are reasoned estimates, not
# profiled numbers.
# Resolution is no longer a single global constant -- it is per-family,
# in the FAMILIES dict above. History, for context on the numbers there:
# a global 512x512 (copied from upstream configs/main.yaml, which trained
# on square ScanNet++ crops) was briefly dropped to 384x384 during the OOM
# phase, then restored to 512x512 once scale_invariant=False fixed the
# underlying rasterizer blowup that had actually been eating the headroom.
# Measured at 512x512: ~32GiB of 49GiB per GPU at BATCH_SIZE=2, closely
# matching a from-architecture estimate of ~34GiB -- which also settled an
# open question, since the estimate would have been ~45GiB had autocast
# kept softmax in fp32. It does not; the attention matrices are bf16.
# All four per-family resolutions are strictly cheaper than that 512x512
# measurement, so headroom is not a concern at these settings.
LORA_R = 8
LORA_ALPHA = 16
# ("qkv", "proj") alone was always meant as a starting point (see the
# splatt3r-lora-finetuning skill's original design doc: "target_modules
# as a starting point, fc1/fc2 left alone, widen if not enough") -- widened
# now that val/psnr has been plateauing around ~9 despite the
# modules_to_save fix (Gaussian head weights now actually persisting
# across resumes -- see the skill). Attention-only LoRA may simply not
# give the backbone enough adaptation capacity for a task this different
# from MASt3R's original pointmap-regression pretraining. fc1/fc2 match
# 100 additional Linear layers (mlp.fc1/fc2 in every enc_blocks/dec_blocks
# transformer block, confirmed by listing named_modules() directly, not
# assumed). NOTE: resuming from a checkpoint saved under the old
# ("qkv","proj") config would silently keep using ITS OWN saved
# adapter_config.json (PeftModel.from_pretrained ignores this constant
# entirely in that path) -- any such old checkpoint needs to be moved
# aside first, or this change has no effect.
LORA_TARGET_MODULES = ("qkv", "proj", "fc1", "fc2")
# Both lowered after a real run's metrics.csv showed a training collapse:
# loss stable ~0.1-0.2 for all of epoch 0, then spiked 0.59->1.28 within a
# single 10-step logging interval partway through epoch 1 and never
# recovered for the remaining 86 epochs it ran (flat ~1.2, val/loss
# matching) -- see the splatt3r-lora-finetuning skill for the full
# analysis. LR=1e-4 with clip=0.5 wasn't enough to bound the damage from
# whatever produced that one outlier gradient (suspected: a batch with a
# near-empty loss mask -- calculate_loss()'s division only guards the
# exact mask.sum()==0 case via .clamp(min=1), not a small-but-nonzero
# mask, which can still blow up the loss/gradient magnitude). LR=2e-5
# matches the upstream configs/main.yaml value's order of magnitude
# (1e-5) rather than the 10x-higher guess this started at; clip=0.1 is a
# much tighter hard bound on the per-step gradient norm regardless of
# what produces an outlier gradient. Switching Adam->AdamW (lora.py) is a
# separate, complementary change -- do not rely on it alone to prevent a
# repeat of this collapse, it doesn't cap single-step gradient magnitude.
LR = 2e-5
GRADIENT_CLIP_VAL = 0.1
# See find_resume_checkpoint() usage below: resumed runs use LR *
# RESUME_LR_FACTOR instead of plain LR, to avoid AdamW's momentum/variance
# reset causing an oversized effective first step on an already-decent
# starting point.
# 0.5 experiment REVERTED same day (2026-07-27): raising resume LR to 1e-5
# made the run OOM inside epoch 0 (run15 at 2e-6 survived ~5 epochs per
# cycle) -- consistent with the rasterizer num_rendered explosion being
# driven by how fast predicted scales grow (epochs-to-OOM scaled ~1/LR).
# Back to 2e-6 until the num_rendered growth itself is addressed; see
# [mem] probe output in the run logs for the mechanism discrimination.
RESUME_LR_FACTOR = 0.1

# History of the LoRA-mode value (=2): 1 -> 2 (2026-07-27). The scale
# penalty (lora.py SCALE_PENALTY_THRESHOLD) successfully held scales in
# the healthy range (p99 ~0.2-0.4, down from 6-8) and OOM was gone -- yet
# run20 attempt 1 STILL died at epoch-0 end with the classic illegal
# memory access (rasterizer_impl.cu:330, forward identifyTileRanges,
# snapshot_fw.dump captured 2026-07-27 20:34). That falsified "giant
# Gaussians cause the rasterizer crashes": that crash family had a
# scale-INDEPENDENT driver, most plausibly the per-view Gaussian COUNT
# (196k at stride=1) overflowing some internal indexing, as suspected in
# the decoder's own comment. So stride went back to 2 for LoRA mode.
#
# HEAD_ONLY mode uses stride=1 (full-resolution Gaussian prediction), the
# value upstream actually trained with: the scale-invariant=False fix and
# the crash history above are all from LoRA-pathology runs, and with the
# encoder frozen the head's predictions start sane (base checkpoint), so
# the mitigations that motivated stride=2 are not expected to be needed.
# If the rasterizer crash DOES come back in head-only mode, stride=2 is
# the first knob to re-enable.
if HEAD_ONLY:
    GAUSSIAN_SPATIAL_STRIDE = 1
else:
    GAUSSIAN_SPATIAL_STRIDE = 2

DEVICES = 2  # both A6000s via DDP; set to 1 (and DDP_STRATEGY=None) for single-GPU
DDP_STRATEGY = "ddp_find_unused_parameters_true"  # required: LoRA+frozen-backbone leaves most params unused per step, same choice upstream's own multi-GPU config makes

if HEAD_ONLY:
    # PER GPU -- effective global batch = BATCH_SIZE * DEVICES. batch=2 is
    # upstream's value and is expected to be stable here: the earlier
    # batch=2 OOMs were LoRA-pathology-driven (MAST3RGaussiansLoRA.forward
    # removes the no_grad wrap, so activations through the ENTIRE 731M
    # backbone were retained for backward, and runaway predicted scales
    # inflated the rasterizer's held-for-backward buffers). Head-only uses
    # MAST3RGaussians.forward, which keeps the encoder/decoder under
    # torch.no_grad() -- only the Gaussian-head forward builds a gradient
    # graph -- and the frozen backbone can't drive scale growth. Fallback
    # plan if OOM still occurs: BATCH_SIZE=1 first, then
    # GAUSSIAN_SPATIAL_STRIDE=2 above.
    BATCH_SIZE = 2
else:
    BATCH_SIZE = 1  # PER GPU -- effective global batch = BATCH_SIZE * DEVICES = 2. 2 -> 1 (2026-07-27): the training OOM is a single-step rasterizer-buffer explosion whose held-for-backward buffers scale with (b * num_target_views) renders per step; once the resumed-from best's predicted scales are large enough, batch=2 (6 held renders) OOMs inside epoch 0 on EVERY attempt (run16/run17), so training was fully blocked. Batch=1 halves the held renders (~41GiB peak -> ~22GiB), unblocking resume from best=0.3021. If scales keep growing and this too hits the ceiling, the next knobs are GAUSSIAN_SPATIAL_STRIDE=2 (above) and tightening lora.py's scales clamp (2.0).
NUM_EPOCHS_PER_EPOCH = 1000  # samples drawn per Lightning "epoch" PER SEQUENCE (DUST3RSplattingDataset semantics), across ALL of a family's sequences
MAX_EPOCHS = 100  # -> for e.g. tum (9 sequences): (9*1000/2)*100 = 450,000 global steps at BATCH_SIZE=1. "As long as possible" per the user's request -- long but finite; see CHECKPOINT_EVERY_N_EPOCHS for a way to stop early with a usable result.
NUM_WORKERS = 16  # PER GPU process (DDP spawns one DataLoader per rank) -- 2 ranks * 16 = 32 worker processes, saturating the 32 cores alongside the 2 main training processes
COVERAGE_POS_THRESHOLD = 0.5  # metres; coarse candidate-pair filter, see data/common.py: compute_coverage
VAL_FRACTION = 0.15
CHECKPOINT_EVERY_N_EPOCHS = 10  # periodic LoRA adapter save during a long run -- see SaveLoRAAdapterCallback
LOG_EVERY_N_STEPS = 10  # how often Lightning logs to CSVLogger AND (see PrintMetricsCallback) prints a plain flushed line


def find_resume_checkpoint(out_dir):
    """Find the most-recently-saved checkpoint for a family, if any, so a
    crashed run can warm-start instead of losing all progress. Added
    after the rasterizer's recurring illegal-memory-access crash (see the
    splatt3r-lora-finetuning skill) kept eventually killing long runs even
    after several rounds of fixes pushed it further out each time --
    training now survives multiple full epochs before it can recur, so
    resuming instead of restarting from scratch each time is worth it.

    NOT a true Lightning resume: only the model WEIGHTS carry over
    (enable_checkpointing=False -- we never save full trainer state), so
    the optimizer's momentum, the LR scheduler's position, and the epoch
    counter all restart at 0 on every relaunch. Good enough for "don't
    lose the trained weights themselves," not for "continue the exact
    training trajectory."

    Prefers the highest-numbered <out_dir>/epoch_N/ over best/ -- best/
    tracks lowest val/loss specifically for eval purposes and can lag well
    behind the most recent training progress. The marker file identifying
    a valid checkpoint dir is mode-dependent: adapter_config.json (peft,
    LoRA mode) or head.ckpt (plain state_dict, HEAD_ONLY mode) -- see
    save_training_checkpoint().

    Returns (resume_path_or_None, highest_epoch_number_on_disk). The
    second value feeds SaveLoRAAdapterCallback's epoch_offset so a resumed
    run's periodic saves keep counting up instead of overwriting the
    pre-crash epoch_N directories.
    """
    marker = "head.ckpt" if HEAD_ONLY else "adapter_config.json"
    if not os.path.isdir(out_dir):
        return None, 0
    epoch_dirs = []
    for name in os.listdir(out_dir):
        m = re.match(r"epoch_(\d+)$", name)
        if m and os.path.isfile(os.path.join(out_dir, name, marker)):
            epoch_dirs.append((int(m.group(1)), os.path.join(out_dir, name)))
    if epoch_dirs:
        epoch_dirs.sort()
        return epoch_dirs[-1][1], epoch_dirs[-1][0]
    best_dir = os.path.join(out_dir, "best")
    if os.path.isfile(os.path.join(best_dir, marker)):
        return best_dir, 0
    return None, 0


class MAST3RGaussiansHeadOnly(MAST3RGaussians):
    """MAST3RGaussians with a requires_grad-filtered optimizer, for the
    HEAD_ONLY control experiment.

    The base class (not MAST3RGaussiansLoRA-without-peft) is the right
    starting point here: base forward() keeps the encoder/decoder under
    torch.no_grad() -- upstream's validated recipe, only the Gaussian-head
    forward builds a gradient graph -- whereas the LoRA subclass's
    forward() retains activations through the whole 731M backbone whether
    or not peft is attached (that memory profile was part of the LoRA
    pathology). The only thing the base class gets wrong for this use is
    configure_optimizers() (main.py), which passes ALL encoder parameters
    including the ~690M frozen ones to Adam; overridden here to filter on
    requires_grad, same as lora.py's version. AdamW + weight_decay=0.05 +
    MultiStepLR kept identical to lora.py's configure_optimizers (matches
    upstream configs/main.yaml), so the LoRA and head-only experiments
    differ only in WHAT is trainable, not in the optimizer setup.

    Trainable params are exactly what MAST3RGaussians.__init__ leaves
    requires_grad=True: downstream_head{1,2}.gaussian_dpt.dpt (~40.4M).
    """

    def configure_optimizers(self):
        params = [p for p in self.encoder.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=self.config.opt.lr, weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [max(1, self.config.opt.epochs // 2)], gamma=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }


def trainable_state_dict(model):
    """State dict of just the trainable Gaussian-head parameters in
    HEAD_ONLY mode. Scoped to model.encoder (with the "encoder." prefix
    re-added) -- NOT model-wide requires_grad: the LPIPS criterion's own
    linear layers also carry requires_grad=True (upstream's own quirk;
    they're used in the loss but were never in any optimizer), and saving
    them would just add noise. The result is exactly the params
    MAST3RGaussiansHeadOnly.configure_optimizers() optimizes --
    downstream_head{1,2}.gaussian_dpt.dpt, ~40.4M, ~160MB per checkpoint
    instead of duplicating the ~3GB frozen backbone, which BASE_CKPT
    already provides on every resume."""
    trainable = {"encoder." + n for n, p in model.encoder.named_parameters() if p.requires_grad}
    return {k: v for k, v in model.state_dict().items() if k in trainable}


def save_training_checkpoint(model, out_dir):
    """Mode-aware periodic/best save: peft adapter dir (LoRA mode) or a
    plain trainable-params state_dict at <out_dir>/head.ckpt (HEAD_ONLY
    mode). find_resume_checkpoint() keys off the same two filenames."""
    os.makedirs(out_dir, exist_ok=True)
    if HEAD_ONLY:
        torch.save(trainable_state_dict(model), os.path.join(out_dir, "head.ckpt"))
    else:
        model.encoder.save_pretrained(out_dir)


def load_training_checkpoint(model, ckpt_dir, device):
    """HEAD_ONLY-mode inverse of save_training_checkpoint(): load the
    trainable-params state_dict over a freshly BASE_CKPT-loaded model.
    (LoRA-mode resume is handled inside attach_lora(resume_from=...)
    instead.) Returns nothing; asserts the checkpoint contained no
    unexpected keys."""
    sd = torch.load(os.path.join(ckpt_dir, "head.ckpt"), map_location=device)
    _missing, unexpected = model.load_state_dict(sd, strict=False)
    # missing = frozen backbone + buffers, already correct from BASE_CKPT.
    assert not unexpected, f"unexpected keys in {ckpt_dir}/head.ckpt: {unexpected}"


def build_pooled_coverage(data, device, cache_path=None):
    """compute_coverage() is per-sequence; call it once per sequence in
    this (possibly multi-sequence) family Data object and combine into
    the {sequence: {i: {j: frac}}} shape DUST3RSplattingDataset expects.
    Cached to disk (see COVERAGE_CACHE_ROOT above) so a DDP rank 1+ re-exec
    of this script reuses rank 0's result instead of recomputing.
    """
    if cache_path is not None and os.path.exists(cache_path):
        print(f"    loading cached coverage from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    coverage = {}
    for sequence in data.sequences:
        n = len(data.color_paths[sequence])
        print(f"    computing coverage for {sequence} ({n} frames)...")
        t0 = time.time()
        coverage[sequence] = compute_coverage(data, sequence, device=device, pos_threshold=COVERAGE_POS_THRESHOLD)
        print(f"      done in {time.time() - t0:.1f}s")

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(coverage, f)
        print(f"    cached coverage to {cache_path}")
    return coverage


class SaveLoRAAdapterCallback(L.Callback):
    """Periodically save the training checkpoint mid-training (not
    Lightning's own full-model checkpointing, which we don't want --
    enable_checkpointing stays False). Important for a run this long:
    gives a usable artifact if training is stopped early or crashes, not
    just an all-or-nothing save at the very end. Rank-0 only, to avoid
    every DDP process racing to write the same files. What exactly gets
    written (peft adapter vs head.ckpt state_dict) dispatches on HEAD_ONLY
    inside save_training_checkpoint().
    """

    def __init__(self, out_root, every_n_epochs, epoch_offset=0):
        self.out_root = out_root
        self.every_n_epochs = every_n_epochs
        # trainer.current_epoch restarts at 0 on every resumed process (we
        # never save full Lightning state -- see find_resume_checkpoint),
        # so without an offset a resumed run's "epoch_10" would overwrite
        # the pre-crash epoch_10, silently destroying the older (and
        # possibly better) checkpoint. epoch_offset carries the highest
        # epoch number already on disk so numbering keeps increasing
        # across restarts.
        self.epoch_offset = epoch_offset

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.every_n_epochs != 0:
            return
        out_dir = os.path.join(self.out_root, f"epoch_{epoch + self.epoch_offset}")
        save_training_checkpoint(pl_module, out_dir)
        print(f"  [checkpoint] saved training checkpoint at epoch {epoch + self.epoch_offset} -> {out_dir}")


class SaveBestAdapterCallback(L.Callback):
    """Tracks val/loss across epochs and overwrites a single "best" LoRA
    adapter snapshot whenever a new low is reached -- separate from
    SaveLoRAAdapterCallback's periodic every-N-epochs schedule, which
    saves on a fixed cadence regardless of whether that particular epoch
    was actually good.

    Uses VAL loss, not train loss: train/loss is a single noisy per-step
    number (batch_size=2, see the "is step 980 the best?" discussion in
    the splatt3r-lora-finetuning skill -- a single low step doesn't mean
    the model was actually better there). val/loss is averaged over the
    whole held-out val set once per epoch, so it's the number that
    actually reflects real progress, and it's exactly what psnr is
    derived from (`-10*mse.log10()`) so there's no separate "highest
    psnr" criterion to reconcile -- lowest val/loss and highest val/psnr
    always agree.

    Rank-0 only, to avoid every DDP process racing to write the same
    files. Skips Lightning's sanity-check validation pass (one no-training
    forward pass before epoch 0 starts) so that doesn't get recorded as
    "epoch 0's" result.
    """

    def __init__(self, out_dir):
        self.out_dir = out_dir
        # Read back a val_loss.txt sidecar from a previous run's best/, if
        # present, instead of always starting from +inf. Without this, a
        # resumed run (see find_resume_checkpoint()) would happily
        # overwrite an already-good best/ checkpoint with a worse one just
        # because this fresh process's own best_val_loss started at +inf
        # and had no memory of what "best" actually meant before the
        # restart -- caught in practice: a resumed run's first epoch
        # produced val/loss=0.2366, worse than the 0.2151 checkpoint it
        # had just resumed from, and without this fix would have
        # overwritten it.
        sidecar = os.path.join(out_dir, "val_loss.txt")
        if os.path.isfile(sidecar):
            with open(sidecar) as f:
                self.best_val_loss = float(f.read().strip())
            print(f"  [best] resuming best-tracking from {sidecar}: {self.best_val_loss:.4f}")
        else:
            self.best_val_loss = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking:
            return
        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is None:
            return
        val_loss = val_loss.item()
        if val_loss >= self.best_val_loss:
            return
        self.best_val_loss = val_loss
        save_training_checkpoint(pl_module, self.out_dir)
        with open(os.path.join(self.out_dir, "val_loss.txt"), "w") as f:
            f.write(str(val_loss))
        val_psnr = trainer.callback_metrics.get("val/psnr")
        psnr_str = f", val/psnr={val_psnr.item():.4f}" if val_psnr is not None else ""
        print(
            f"  [best] new best val/loss={val_loss:.4f}{psnr_str} "
            f"at epoch {trainer.current_epoch} -> saved to {self.out_dir}",
            flush=True,
        )


class PrintMetricsCallback(L.Callback):
    """Plain, flushed print()s of the loss/metrics Lightning already logs
    to CSVLogger -- for readability in a log file, since the actual
    numbers (see the splatt3r-lora-finetuning skill) never showed up in
    full_run.log at all: piping stdout through `tee` makes it a non-TTY,
    which (a) makes Python fully block-buffer stdout instead of line-
    buffering, so nothing appears until a flush/exit, and (b) makes
    Lightning's default Rich progress bar not redraw properly (it's
    designed for an interactive terminal's cursor control, not a pipe).
    `enable_progress_bar=False` on the Trainer turns that bar off entirely
    -- this callback is what replaces it with something a log file can
    actually show. Rank-0 only, to avoid every DDP process printing the
    same line; metrics come from `trainer.callback_metrics`, the same
    dict the (now-disabled) progress bar would have read from.
    """

    def __init__(self, every_n_steps):
        self.every_n_steps = every_n_steps

    def _print(self, tag, trainer, keys):
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        parts = [f"epoch={trainer.current_epoch}", f"step={trainer.global_step}"]
        for key in keys:
            value = metrics.get(key)
            if value is not None:
                parts.append(f"{key}={value.item():.4f}")
        print(f"  [{tag}] " + " ".join(parts), flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step % self.every_n_steps != 0:
            return
        self._print("train", trainer, ("train/loss", "train/mse", "train/psnr", "train/lpips", "train/scale_pen", "train/scale_p99"))

    def on_validation_epoch_end(self, trainer, pl_module):
        self._print("val", trainer, ("val/loss", "val/mse", "val/psnr", "val/lpips"))
        # Memory probe to distinguish a true leak (allocated grows
        # monotonically per epoch) from allocator caching of data-dependent
        # rasterizer buffer peaks (only reserved grows). Rank-0 only.
        if trainer.is_global_zero:
            print(
                f"  [mem] epoch={trainer.current_epoch} "
                f"allocated={torch.cuda.memory_allocated() / 2**30:.2f}GiB "
                f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}GiB",
                flush=True,
            )


def train_one_family(family_name: str):
    # Under DDP, Lightning launches one subprocess per GPU and sets
    # LOCAL_RANK in its environment -- each rank must build its model on
    # ITS OWN device, not a hardcoded "cuda" (which would put every rank
    # on GPU 0). Do this before trainer.fit(), since model construction
    # happens in plain user code here, not inside a Lightning-managed hook.
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = f"cuda:{local_rank}"

    data_cls, family_root, extra_kwargs, resolution = FAMILIES[family_name]
    out_root = HEADONLY_OUT_ROOT if HEAD_ONLY else LORA_OUT_ROOT
    out_dir = os.path.join(out_root, family_name)
    log_dir = os.path.join(REPO_ROOT, "logs", "headonly_training" if HEAD_ONLY else "lora_training", family_name)
    os.makedirs(log_dir, exist_ok=True)

    mode = "HEAD-ONLY (frozen encoder, Gaussian dpt heads only)" if HEAD_ONLY else "LoRA"
    print(f"\n{'=' * 70}\nTraining {mode} for family: {family_name}\n{'=' * 70}")
    t0 = time.time()

    resume_from, epoch_offset = find_resume_checkpoint(out_dir)
    if HEAD_ONLY:
        model = MAST3RGaussiansHeadOnly.load_from_checkpoint(BASE_CKPT, map_location=device)
        if resume_from is not None:
            print(f"Loading base model + resuming Gaussian head from {resume_from} ...")
            load_training_checkpoint(model, resume_from, device)
        else:
            print("Loading base model (encoder frozen, Gaussian dpt heads trainable)...")
    else:
        if resume_from is not None:
            print(f"Loading base model + resuming LoRA from {resume_from} ...")
        else:
            print("Loading base model + attaching a fresh LoRA...")
        model = MAST3RGaussiansLoRA.load_from_checkpoint(BASE_CKPT, map_location=device)
        model = attach_lora(
            model, r=LORA_R, alpha=LORA_ALPHA, target_modules=LORA_TARGET_MODULES, resume_from=resume_from
        )
    # RESUME_LR_FACTOR, not plain LR, when resuming: caught in practice
    # (see the splatt3r-lora-finetuning skill) -- a resumed run's val/loss
    # got steadily WORSE for 4 straight epochs (0.2151 -> 0.2366 -> 0.2623
    # -> 0.2824), not noise, a real trend. Resuming only carries over the
    # LoRA weights (enable_checkpointing=False -- no optimizer state is
    # saved), so AdamW's momentum/variance estimates restart at zero. Adam's
    # bias correction inflates the effective step size while those
    # estimates are still small, so the first several steps after a
    # restart take unusually large steps -- fine when starting from a bad
    # random init, actively harmful when starting from an already-decent
    # point, since it can shove the weights straight out of the good
    # region they were already in. A reduced LR for resumed runs gives
    # AdamW a gentler restart instead of full-strength steps applied to an
    # already-converged-ish point.
    model.config.opt.lr = LR * RESUME_LR_FACTOR if resume_from is not None else LR
    model.config.opt.epochs = MAX_EPOCHS
    model.config.data.batch_size = BATCH_SIZE
    model.decoder.spatial_stride = GAUSSIAN_SPATIAL_STRIDE
    model = model.to(device)

    n_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.encoder.parameters())
    print(f"Trainable encoder params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")

    print(f"Building {family_name} train/val data (pooling all sequences under {family_root})...")
    train_data = data_cls(family_root, "train", val_fraction=VAL_FRACTION, **extra_kwargs)
    val_data = data_cls(family_root, "val", val_fraction=VAL_FRACTION, **extra_kwargs)
    if len(train_data.sequences) == 0:
        raise RuntimeError(
            f"No usable sequences found for family '{family_name}' under {family_root}. "
            f"For euroc/eth3d, did you run scripts/precompute_pseudo_depth.py first?"
        )
    n_train_total = sum(len(train_data.color_paths[s]) for s in train_data.sequences)
    print(f"  {len(train_data.sequences)} sequences, {n_train_total} total train frames: {train_data.sequences}")

    cache_tag = f"valfrac{VAL_FRACTION}_pos{COVERAGE_POS_THRESHOLD}"
    print("  computing coverage matrices (train)...")
    train_coverage = build_pooled_coverage(
        train_data, device,
        cache_path=os.path.join(COVERAGE_CACHE_ROOT, f"{family_name}_train_{cache_tag}.pkl"),
    )
    print("  computing coverage matrices (val)...")
    val_coverage = build_pooled_coverage(
        val_data, device,
        cache_path=os.path.join(COVERAGE_CACHE_ROOT, f"{family_name}_val_{cache_tag}.pkl"),
    )

    train_ds = DUST3RSplattingDataset(
        train_data, train_coverage, resolution=resolution, num_epochs_per_epoch=NUM_EPOCHS_PER_EPOCH
    )
    val_ds = DUST3RSplattingDataset(
        val_data, val_coverage, resolution=resolution, num_epochs_per_epoch=max(20, NUM_EPOCHS_PER_EPOCH // 10)
    )
    print(f"  train_ds len={len(train_ds)}, val_ds len={len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    trainer = L.Trainer(
        accelerator="gpu",
        devices=DEVICES,
        strategy=DDP_STRATEGY if DEVICES > 1 else "auto",
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=GRADIENT_CLIP_VAL,
        # Without this, Trainer defaults to full fp32. That matters a lot
        # more than BATCH_SIZE here: MAST3RGaussiansLoRA.forward() removed
        # the base model's torch.no_grad() wrap so LoRA can get gradients,
        # which means activations through the ENTIRE 731M-param backbone
        # (not just the 42.8M trainable LoRA params) are retained for
        # backward -- and DUST3RSplattingDataset renders 3 target views per
        # sample (data/data.py: num_target_views=3, fixed, independent of
        # BATCH_SIZE), each a separate CUDA Gaussian-rasterizer call with
        # its own large geom/binning/img buffers. That's why dropping
        # BATCH_SIZE 12->3->2 barely moved the OOM: the floor was already
        # close to 47GB regardless of batch size.
        #
        # "bf16-mixed", not "16-mixed": first attempt at bf16-mixed crashed
        # with `NotImplementedError: "rope_2d_cuda" not implemented for
        # 'BFloat16'` -- the vendored custom CUDA RoPE kernel every
        # attention block calls (croco/models/curope/kernels.cu:101) was
        # compiled with AT_DISPATCH_FLOATING_TYPES_AND_HALF, which covers
        # float32/float64/float16 but not bfloat16. Fell back to
        # "16-mixed" (fp16) at the time, but then patched and rebuilt the
        # kernel instead (AT_DISPATCH_FLOATING_TYPES_AND2 with Half AND
        # BFloat16 -- verified with a standalone CUDA smoke test: finite,
        # non-identity output for float32/float16/bfloat16 all three), so
        # bf16 is back and preferred -- unlike fp16, its exponent range
        # matches fp32's, so no GradScaler/loss-scaling risk of NaN/inf.
        # Also patched src/pixelsplat_src/cuda_splatting.py:render_cuda()
        # to force its inputs back to float32 right before the rasterizer
        # call: that CUDA extension (thirdparty/diff-gaussian-rasterization-
        # modified) is hardcoded to torch::kFloat32 with no dtype dispatch
        # at all (unlike the RoPE kernel), so it would otherwise hit a
        # dtype-mismatch error on the bf16 tensors autocast produces
        # upstream in the Gaussian head.
        precision="bf16-mixed",
        logger=L.pytorch.loggers.CSVLogger(save_dir=log_dir),
        enable_checkpointing=False,
        # Rich progress bar off, PrintMetricsCallback on -- see that
        # class's docstring: the progress bar doesn't render right (and
        # loss was never visible at all) once stdout is piped through
        # `tee` to a log file, which is how this has actually been run.
        enable_progress_bar=False,
        callbacks=[
            SaveLoRAAdapterCallback(out_dir, CHECKPOINT_EVERY_N_EPOCHS, epoch_offset=epoch_offset),
            SaveBestAdapterCallback(os.path.join(out_dir, "best")),
            PrintMetricsCallback(LOG_EVERY_N_STEPS),
        ],
        log_every_n_steps=LOG_EVERY_N_STEPS,
        default_root_dir=log_dir,
    )
    global_batch = BATCH_SIZE * DEVICES
    print(f"Starting training ({MAX_EPOCHS} epochs x {len(train_ds) // global_batch} steps/epoch, "
          f"global batch size {global_batch} across {DEVICES} GPU(s))...")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if trainer.is_global_zero:
        save_training_checkpoint(model, out_dir)
        print(f"Saved final training checkpoint to {out_dir}")
        print(f"Family {family_name} done in {time.time() - t0:.1f}s total.")

    del model, trainer, train_loader, val_loader
    torch.cuda.empty_cache()


if __name__ == "__main__":
    requested = sys.argv[1:] if len(sys.argv) > 1 else list(FAMILIES.keys())
    unknown = [f for f in requested if f not in FAMILIES]
    if unknown:
        raise SystemExit(f"Unknown famil{'y' if len(unknown)==1 else 'ies'}: {unknown}. Choices: {list(FAMILIES.keys())}")

    print(f"Training {'head-only (HEAD_ONLY=True)' if HEAD_ONLY else 'LoRA adapters'} for {len(requested)} families: {requested}")
    overall_t0 = time.time()
    for family_name in requested:
        train_one_family(family_name)
    print(f"\nAll {len(requested)} families done in {time.time() - overall_t0:.1f}s total.")
    print(f"Checkpoints saved under: {(HEADONLY_OUT_ROOT if HEAD_ONLY else LORA_OUT_ROOT)}/<family>/")
