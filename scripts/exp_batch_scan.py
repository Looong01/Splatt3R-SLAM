"""Measure training throughput and peak memory vs. batch size, per family.

Motivation: the head-only runs use batch=2 and peak at 6.3 GiB on a 49 GiB
A6000, which looks like a large waste. It partly is -- but GPU *utilization*
during the TUM run measured 89%, so the headroom is not straightforwardly
convertible into speed, and the honest answer needs numbers rather than a
glance at nvidia-smi.

What this measures, per family and batch size:
  - s/step and samples/s (median over timed steps, after warmup)
  - peak allocated memory

What it deliberately does not do: change the training recipe. Batch size is
not a free knob here -- at a fixed LR, doubling the batch halves the number of
optimizer steps per epoch, which is a different optimization problem, so any
batch change has to be paid for with an LR retune and a fresh base measurement
before its results can be compared to the existing runs. This script only
establishes what the throughput would buy, so that decision can be made on
evidence.

Peak memory is also data-dependent in this model: the 3DGS rasterizer's
buffers scale with `num_rendered`, which depends on the predicted Gaussian
scales. The reported peak is therefore a sample, not a bound -- treat it as
indicative and keep headroom.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 scripts/exp_batch_scan.py 7-scenes
    CUDA_VISIBLE_DEVICES=1 python3 scripts/exp_batch_scan.py --all --batches 2,4,8,16
"""
import argparse
import os
import pickle
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import exp_head_only as E  # noqa: E402  (sys.path/chdir setup)
import train_lora_per_scene as T  # noqa: E402

WARMUP_STEPS = 3
TIMED_STEPS = 12


def build_loader(family, batch):
    data_cls, root, extra, res = T.FAMILIES[family]
    tag = f"valfrac{T.VAL_FRACTION}_pos{T.COVERAGE_POS_THRESHOLD}"
    cov_path = os.path.join(T.COVERAGE_CACHE_ROOT, f"{family}_train_{tag}.pkl")
    if not os.path.exists(cov_path):
        raise FileNotFoundError(
            f"no coverage cache for '{family}': {cov_path}\n"
            f"run: python3 scripts/precompute_coverage.py {family}")

    data = data_cls(root, "train", val_fraction=T.VAL_FRACTION, **extra)
    with open(cov_path, "rb") as f:
        cov = pickle.load(f)
    ds = E.DUST3RSplattingDataset(data, cov, resolution=res,
                                  num_epochs_per_epoch=E.SAMPLES_PER_SEQ)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                         num_workers=8, drop_last=True)
    return loader, res


def scan_family(family, batches):
    print(f"\n{'=' * 72}\n{family}\n{'=' * 72}", flush=True)

    model = E.MAST3RGaussiansHeadOnly.load_from_checkpoint(
        E.BASE_CKPT, map_location=E.DEV).to(E.DEV)
    model.decoder.spatial_stride = E.STRIDE
    opt = torch.optim.AdamW(E.head_params(model), lr=E.LR, weight_decay=0.05)

    rows = []
    for batch in batches:
        try:
            loader, res = build_loader(family, batch)
        except FileNotFoundError as e:
            print(e, flush=True)
            return

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        times = []
        try:
            for i, b in enumerate(loader):
                if i >= WARMUP_STEPS + TIMED_STEPS:
                    break
                torch.cuda.synchronize()
                t0 = time.time()
                loss, _, _ = E.step_loss(model, b)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(E.head_params(model), E.GRAD_CLIP)
                opt.step()
                torch.cuda.synchronize()
                if i >= WARMUP_STEPS:
                    times.append(time.time() - t0)
        except torch.cuda.OutOfMemoryError:
            print(f"  batch={batch:<3} OOM", flush=True)
            torch.cuda.empty_cache()
            continue

        s = statistics.median(times)
        peak = torch.cuda.max_memory_allocated() / 2**30
        rows.append((batch, s, batch / s, peak))
        print(f"  batch={batch:<3} {s:.3f} s/step  {batch / s:5.2f} samples/s  "
              f"peak={peak:5.2f} GiB  res={res}", flush=True)

    if rows:
        base_rate = rows[0][2]
        print(f"\n  relative throughput (batch={rows[0][0]} = 1.00x):", flush=True)
        for batch, _s, rate, _p in rows:
            print(f"    batch={batch:<3} {rate / base_rate:.2f}x", flush=True)

    del model, opt
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family", nargs="?", choices=sorted(T.FAMILIES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--batches", default="2,4,8,16")
    args = ap.parse_args()
    if not args.all and args.family is None:
        ap.error("give a family name or --all")

    batches = [int(x) for x in args.batches.split(",")]
    families = sorted(T.FAMILIES) if args.all else [args.family]
    for f in families:
        scan_family(f, batches)


if __name__ == "__main__":
    main()
