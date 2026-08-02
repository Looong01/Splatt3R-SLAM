"""Regenerate the live-progress block in the finetuning-experiments skill.

Every number in that skill is supposed to be traceable to a log, and the
hand-transcribed progress updates were not: on two occasions an epoch's metrics
were reported with values that appear nowhere in either log (epoch 22 and
epoch 27 of the TUM run), and had to be retracted. Parsing the logs removes
that failure mode entirely.

Rewrites only the region between the PROGRESS markers in SKILL.md, so the
prose around it is never touched.

Usage:
    python3 scripts/update_skill_progress.py
"""
import math
import os
import re
from datetime import datetime

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(REPO_ROOT, ".claude", "skills",
                     "splatt3r-finetuning-experiments", "SKILL.md")
BEGIN = "<!-- PROGRESS:BEGIN -->"
END = "<!-- PROGRESS:END -->"

RUNS = [
    ("TUM", "batch 2, lr 1e-5", 40, "logs/exp_head_only_long.log"),
    ("7-Scenes", "batch 8, lr 2e-5, warmup 100", 40, "logs/exp_head_only_7scenes.log"),
    ("EuRoC", "batch 4, lr 1.41e-5, warmup 100 (batch 8 OOMed, see §7.2)", 40, "logs/exp_head_only_euroc_b4.log"),
    ("ETH3D", "batch 4, lr 1.41e-5, warmup 100", 40, "logs/exp_head_only_eth3d.log"),
]

# `nan` is matched deliberately, not skipped. An earlier version accepted only
# [\d.]+ and silently dropped the 12 NaN epochs of the ETH3D run, so the
# progress table read "28/40 epochs" and looked like a run still in flight when
# training had in fact diverged at epoch 28 and never recovered. A failure that
# formats as ordinary progress is worse than no report at all.
ROW = re.compile(r"(BASE|epoch (\d+)) \| val/loss=(nan|[\d.]+)\s+mse=(nan|[\d.]+)\s+"
                 r"psnr=(nan|[\d.]+)\s+lpips=(nan|[\d.]+)")


def parse(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = ROW.search(line)
            if m:
                mse = float(m.group(4))
                if math.isnan(mse):
                    rows.append({"tag": m.group(1), "nan": True})
                    continue
                rows.append({
                    "tag": m.group(1), "nan": False,
                    "loss": float(m.group(3)), "mse": mse,
                    # Recomputed, not taken from the log: calculate_loss's
                    # "mse" sums over 3 colour channels but divides by pixel
                    # count, so a PSNR taken from it directly is 4.77 dB low.
                    # Logs written before that was found print the low value;
                    # deriving PSNR here makes old and new logs consistent.
                    # See scripts/diag_psnr.py.
                    "psnr": -10 * math.log10(mse / 3.0),
                    "lpips": float(m.group(6)),
                })
    return rows


def block():
    out = [BEGIN, "",
           f"*Regenerated {datetime.now():%Y-%m-%d %H:%M} by "
           f"`scripts/update_skill_progress.py` — parsed from the logs, not "
           f"transcribed.*", ""]
    for name, cfg, total, log in RUNS:
        path = os.path.join(REPO_ROOT, log)
        if not os.path.exists(path):
            out += [f"**{name}** ({cfg}) — no log yet.", ""]
            continue
        rows = parse(path)
        if len(rows) < 2:
            out += [f"**{name}** ({cfg}) — started, no epoch results yet.", ""]
            continue
        base, eps = rows[0], rows[1:]
        nan_rows = [r for r in eps if r["nan"]]
        eps = [r for r in eps if not r["nan"]]
        if not eps:
            out += [f"**{name}** ({cfg}) — **DIVERGED, no finite epochs**, `{log}`", ""]
            continue
        bp = max(eps, key=lambda r: r["psnr"])
        bl = min(eps, key=lambda r: r["lpips"])
        bv = min(eps, key=lambda r: r["loss"])
        bm = min(eps, key=lambda r: r["mse"])
        done = f"{len(eps) + len(nan_rows)}/{total}"
        # A diverged run must announce itself. Silently dropping NaN epochs
        # once made a dead ETH3D run read as "28/40, still going".
        warn = (f" — **DIVERGED to NaN at {nan_rows[0]['tag']}; "
                f"{len(nan_rows)} dead epochs follow. Best below is from "
                f"before divergence and the saved checkpoint is valid.**"
                if nan_rows else "")
        out += [
            f"**{name}** ({cfg}) — {done} epochs, `{log}`{warn}", "",
            "| metric | base | best | Δ | at |",
            "|---|---|---|---|---|",
            f"| psnr | {base['psnr']:.4f} | **{bp['psnr']:.4f}** | "
            f"**{bp['psnr'] - base['psnr']:+.2f} dB** | {bp['tag']} |",
            f"| lpips | {base['lpips']:.4f} | **{bl['lpips']:.4f}** | "
            f"**{(bl['lpips'] - base['lpips']) / base['lpips'] * 100:+.1f}%** | {bl['tag']} |",
            f"| mse | {base['mse']:.4f} | **{bm['mse']:.4f}** | "
            f"**{(bm['mse'] - base['mse']) / base['mse'] * 100:+.1f}%** | {bm['tag']} |",
            f"| val/loss | {base['loss']:.4f} | **{bv['loss']:.4f}** | "
            f"**{(bv['loss'] - base['loss']) / base['loss'] * 100:+.1f}%** | {bv['tag']} |",
            "",
        ]
    out += [END]
    return "\n".join(out)


def main():
    with open(SKILL) as f:
        s = f.read()
    if BEGIN not in s or END not in s:
        raise SystemExit(f"markers {BEGIN}/{END} not found in {SKILL}")
    head, rest = s.split(BEGIN, 1)
    _stale, tail = rest.split(END, 1)
    with open(SKILL, "w") as f:
        f.write(head + block() + tail)
    print(block())


if __name__ == "__main__":
    main()
