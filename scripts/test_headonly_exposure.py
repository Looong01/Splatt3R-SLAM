"""Self-test for the HEAD_ONLY control-experiment implementation -- NO
training is run. Verifies, on CPU:

  (a) head-only mode's trainable parameters are exactly the Gaussian dpt
      heads (~40.4M), and MAST3RGaussiansHeadOnly.configure_optimizers()
      passes exactly those params (not the ~690M frozen backbone) to the
      optimizer;
  (b) SequenceExposureLock (data/common.py) applies ONE locked gain per
      sequence -- the same gain to two different frames of the same
      sequence -- with the gain clamped to [0.4, 2.5];
  (c) the plain-state_dict checkpoint path (trainable_state_dict /
      save_training_checkpoint / load_training_checkpoint /
      find_resume_checkpoint) round-trips losslessly.

Usage:
    /home/share-v5/miniconda3/envs/splatt3r-slam/bin/python scripts/test_headonly_exposure.py
"""
import os
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

# Importing the training module runs its sys.path/chdir setup and gives us
# HEAD_ONLY, MAST3RGaussiansHeadOnly, the checkpoint helpers and BASE_CKPT.
import train_lora_per_scene as t  # noqa: E402

import torch  # noqa: E402

from data.common import NORMALIZE_EXPOSURE, SequenceExposureLock  # noqa: E402

failures = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        failures.append(name)


print("=" * 70)
print("(a) head-only trainable parameters")
print("=" * 70)
assert t.HEAD_ONLY, "HEAD_ONLY is False -- this test is for the head-only mode"
model = t.MAST3RGaussiansHeadOnly.load_from_checkpoint(t.BASE_CKPT, map_location="cpu")
# Scope to model.encoder: the LPIPS criterion's linear layers also carry
# requires_grad=True (upstream's own quirk -- used in the loss but never
# in any optimizer, in upstream's recipe or ours), so a model-wide
# requires_grad count would include 1,152 params that are never trained.
trainable = [(n, p.numel()) for n, p in model.encoder.named_parameters() if p.requires_grad]
n_trainable = sum(c for _, c in trainable)
n_total = sum(p.numel() for p in model.parameters())
print(f"  trainable (encoder-scoped): {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.3f}%)")
check("trainable params only in gaussian dpt heads",
      all("gaussian_dpt.dpt" in n for n, _ in trainable),
      f"{len(trainable)} tensors, e.g. {trainable[0][0]}")
check("trainable count ~40.4M (expected 40,405,916)",
      n_trainable == 40_405_916, f"got {n_trainable:,}")

model.config.opt.lr = t.LR
model.config.opt.epochs = t.MAX_EPOCHS
opt = model.configure_optimizers()["optimizer"]
n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
check("configure_optimizers covers exactly the trainable params",
      n_opt == n_trainable, f"optimizer params {n_opt:,} vs trainable {n_trainable:,}")

print("=" * 70)
print("(b) exposure normalization: one locked gain per sequence")
print("=" * 70)
check("NORMALIZE_EXPOSURE switch is on", NORMALIZE_EXPOSURE)
rng = np.random.default_rng(0)
# frame0: dark-ish image, mean ~0.30; frame1: brighter, mean ~0.45 --
# different content, same sequence.
frame0 = (rng.random((32, 48, 3)) * 0.2 + 0.2 * 255 / 255 * 255).astype(np.uint8)
frame0 = (rng.random((32, 48, 3)) * 51 + 51).astype(np.uint8)   # mean ~0.30
frame1 = (rng.random((32, 48, 3)) * 51 + 89).astype(np.uint8)   # mean ~0.45
lock = SequenceExposureLock()
gain = lock.lock("seqA", frame0)
out0 = lock.apply(frame0, "seqA")
out1 = lock.apply(frame1, "seqA")
expected1 = (np.clip(frame1.astype(np.float32) / 255.0 * gain, 0.0, 1.0) * 255.0).round().astype(np.uint8)
check("same locked gain applied to two frames of one sequence",
      np.array_equal(out1, expected1),
      f"gain={np.round(gain, 4).tolist()}")
check("gain within [0.4, 2.5]",
      bool(np.all(gain >= 0.4) and np.all(gain <= 2.5)),
      f"gain={np.round(gain, 4).tolist()}")
mean0_after = out0.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0
check("first frame canonicalized towards target_mean=0.5",
      bool(np.all(np.abs(mean0_after - 0.5) < 0.05)),
      f"mean after lock-frame apply={np.round(mean0_after, 4).tolist()}")
# A second sequence gets its OWN independent gain (and its own lock frame).
frameB = (rng.random((32, 48, 3)) * 51 + 150).astype(np.uint8)  # brighter seq
gainB = lock.lock("seqB", frameB)
check("different sequence -> different locked gain",
      not np.allclose(gain, gainB),
      f"gainA={np.round(gain, 3).tolist()} gainB={np.round(gainB, 3).tolist()}")
# Clamp behaviour: an extremely dark first frame must clamp to 2.5.
dark = np.full((32, 48, 3), 5, dtype=np.uint8)
gainC = lock.lock("seqC", dark)
check("extreme dark first frame clamps gain to 2.5",
      bool(np.allclose(gainC, 2.5)), f"gain={np.round(gainC, 4).tolist()}")

print("=" * 70)
print("(c) checkpoint state_dict round-trip")
print("=" * 70)
with tempfile.TemporaryDirectory() as tmp:
    epoch_dir = os.path.join(tmp, "epoch_3")
    # Save via the same helper the training callbacks use.
    t.save_training_checkpoint(model, epoch_dir)
    ckpt_path = os.path.join(epoch_dir, "head.ckpt")
    check("head.ckpt written", os.path.isfile(ckpt_path), ckpt_path)
    saved = torch.load(ckpt_path, map_location="cpu")
    check("checkpoint contains only gaussian head keys",
          all(k.startswith("encoder.") and "gaussian_dpt.dpt" in k for k in saved)
          and len(saved) == len(trainable),
          f"{len(saved)} tensors")

    original = {n: p.detach().clone() for n, p in model.encoder.named_parameters() if p.requires_grad}
    # Perturb the head, then restore from the checkpoint.
    with torch.no_grad():
        for n, p in model.encoder.named_parameters():
            if p.requires_grad:
                p.add_(1.0)
    t.load_training_checkpoint(model, epoch_dir, "cpu")
    restored_ok = all(
        torch.equal(p, original[n]) for n, p in model.encoder.named_parameters() if p.requires_grad
    )
    check("save -> perturb -> load restores exact head weights", restored_ok)

    # find_resume_checkpoint prefers the highest epoch_N/ with head.ckpt.
    os.makedirs(os.path.join(tmp, "epoch_7"))
    torch.save(saved, os.path.join(tmp, "epoch_7", "head.ckpt"))
    os.makedirs(os.path.join(tmp, "best"))
    torch.save(saved, os.path.join(tmp, "best", "head.ckpt"))
    resume, epoch_offset = t.find_resume_checkpoint(tmp)
    check("find_resume_checkpoint prefers highest epoch_N",
          resume.endswith("epoch_7") and epoch_offset == 7,
          f"resume={resume}, offset={epoch_offset}")
    # A peft adapter dir (adapter_config.json, no head.ckpt) must NOT be
    # picked up in HEAD_ONLY mode -- the two formats never mix.
    with tempfile.TemporaryDirectory() as tmp2:
        peft_dir = os.path.join(tmp2, "epoch_5")
        os.makedirs(peft_dir)
        open(os.path.join(peft_dir, "adapter_config.json"), "w").write("{}")
        resume2, _ = t.find_resume_checkpoint(tmp2)
        check("peft-format checkpoint ignored in HEAD_ONLY mode", resume2 is None)

print("=" * 70)
if failures:
    print(f"SELF-TEST FAILED: {failures}")
    sys.exit(1)
print("SELF-TEST OK: all checks passed.")
