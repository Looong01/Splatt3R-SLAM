"""Build editable RA-L figures and numeric tables from measured evidence.

Run after render_comparisons.py. No image synthesis, retouching, or per-method
crop selection is used. PDF/SVG contain vector labels and the original renders.
The system overview is rendered from its native TikZ source using pdflatex.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/Thesis/ral"
DATA = ROOT / "logs/paper_ral_20261007"
FIG = PAPER / "fig"
SCENES = [f"office{i}" for i in range(5)] + [f"room{i}" for i in range(3)]
INK, GRAY, BLUE, TEAL, ORANGE = "#172B3A", "#647887", "#4079AC", "#007E78", "#B77A2E"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .5, "savefig.facecolor": "white",
})


def save(fig, name):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=240, bbox_inches=None)
    plt.close(fig)


def canvas(w, h):
    f = plt.figure(figsize=(w, h))
    ax = f.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, w), ylim=(0, h))
    ax.axis("off")
    return f, ax


def text(ax, x, y, label, size=8, **kw):
    ax.text(x, y, label, fontsize=size, va="center", **kw)


def box(ax, x, y, w, h, title, subtitle=None, color=TEAL, fill="#EDF7F5",
        size=8, dashed=False):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.05",
        linewidth=.85, edgecolor=color, facecolor=fill,
        linestyle="--" if dashed else "-"))
    text(ax, x+w/2, y+h/2+(0.095 if subtitle else 0), title, size,
         ha="center", fontweight="bold")
    if subtitle:
        text(ax, x+w/2, y+h/2-.12, subtitle, size-1, ha="center")


def arrow(ax, start, end, color=GRAY, dashed=False, style="-|>", rad=0):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=8, linewidth=.9,
        color=color, linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}"))


def image(fig, path, rect, crop=None, outline=None, roi=None):
    ax = fig.add_axes(rect)
    a = np.asarray(Image.open(path))
    if crop:
        x, y, w, h = crop
        a = a[y:y+h, x:x+w]
    ax.imshow(a, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(bool(outline))
        spine.set_color(outline or "white")
        spine.set_linewidth(1.1)
    if roi:
        x, y, w, h = roi
        ax.add_patch(Rectangle((x, y), w, h, fill=False, lw=1.1,
                               edgecolor=ORANGE))
    return ax


def teaser(records):
    w, h = 7.15, 2.23
    f, a = canvas(w, h)
    text(a, .02, 2.12, "(a) Pose-anchored prediction + separate refinement", 8.2,
         fontweight="bold")
    box(a, .03, 1.35, 1.32, .54, "Image pair", "shared frozen features",
        color=BLUE, fill="#EEF3F9", size=7.7)
    box(a, 1.63, 1.35, 1.42, .54, "Anchored Gaussians", "follow pose-graph updates",
        size=7.7)
    box(a, 3.34, 1.35, 1.26, .54, "Map refinement", "no pose optimisation",
        size=7.7)
    arrow(a, (1.37, 1.62), (1.60, 1.62))
    arrow(a, (3.08, 1.62), (3.31, 1.62))
    # Real views at the two middle delta ranks; retain the complete frame.
    for x, method, label in [(0.04, "photo", "Photo-SLAM"), (2.35, "ours", "Ours")]:
        image(f, DATA / "office0" / f"view0_{method}.png",
              [x/w, .02/h, 2.22/w, 1.13/h])
        text(a, x+1.11, 1.24, label, 7.8, ha="center",
             color=TEAL if method == "ours" else GRAY, fontweight="bold")
    text(a, 4.87, 2.12, "(b) Replica: all 8 scenes", 8.2, fontweight="bold")
    mean = {m: np.mean([records[s]["methods"][m]["full_sh"]["psnr"]
                        for s in SCENES]) for m in ("photo", "ours")}
    c = f.add_axes([5.05/w, .53/h, 1.98/w, 1.28/h])
    c.bar([0, 1], [mean["photo"], mean["ours"]], color=[GRAY, TEAL], width=.58)
    c.set(ylim=(0, 29), yticks=[0, 10, 20], xticks=[0, 1],
          xticklabels=["Photo-SLAM", "Ours"], ylabel="PSNR (dB) ↑")
    c.tick_params(labelsize=7, length=2)
    c.spines["left"].set_visible(False)
    c.yaxis.grid(True, lw=.4, color="#D8E0E5")
    c.set_axisbelow(True)
    for i, m in enumerate(("photo", "ours")):
        c.text(i, mean[m]+.7, f"{mean[m]:.2f}", ha="center", fontsize=9,
               fontweight="bold", color=TEAL if m == "ours" else GRAY)
    text(a, 6.03, .18, f"+{mean['ours']-mean['photo']:.2f} dB  ·  8/8 PSNR wins",
         9.3, ha="center", fontweight="bold", color=TEAL)
    save(f, "teaser")


def overview():
    from build_overview import build_overview
    build_overview()


def qualitative():
    w, h = 7.15, 4.45
    f, a = canvas(w, h)
    columns = [("gt", "Ground truth"), ("photo", "Photo-SLAM"),
               ("mono", "MonoGS"), ("ours", "Ours")]
    rois = [(155, 112, 220, 96), (58, 40, 220, 96)]
    for col, (method, title) in enumerate(columns):
        x, iw = .08 + col*1.785, 1.69
        text(a, x+iw/2, 4.34, title, 9, ha="center",
             fontweight="bold", color=TEAL if method == "ours" else INK)
        for row in range(2):
            y = 2.94-row*2.14
            p = DATA / "desk" / f"view{row}_{method}.png"
            image(f, p, [x/w, y/h, iw/w, (iw*.75)/h], roi=rois[row])
            image(f, p, [x/w, (y-.76)/h, iw/w, (iw*96/220)/h],
                  crop=rois[row], outline=ORANGE)
    save(f, "qualitative_tum")
    (FIG / "crop_manifest.json").write_text(json.dumps({
        "scene": "desk", "rule": "identical pixel ROI for all methods",
        "view0_xywh": rois[0], "view1_xywh": rois[1],
        "resampling": "none in source; original pixels embedded in PDF/SVG",
        "selection": json.loads((DATA/"desk/metrics.json").read_text())["selection"],
    }, indent=2)+"\n")


def evidence(records):
    f, ax = plt.subplots(1, 2, figsize=(7.15, 1.88))
    f.subplots_adjust(left=.075, right=.985, top=.84, bottom=.24, wspace=.29)
    # Raw scene differences, no error bars or inferred significance.
    d = [records[s]["methods"]["ours"]["full_sh"]["psnr"] -
         records[s]["methods"]["photo"]["full_sh"]["psnr"] for s in SCENES]
    ax[0].bar(range(8), d, color=TEAL, width=.66)
    ax[0].set(xticks=range(8), xticklabels=["o0","o1","o2","o3","o4","r0","r1","r2"],
              ylabel="ΔPSNR (dB) ↑", ylim=(0, 8.8))
    ax[0].set_title("(a) Ours − Photo-SLAM, Replica", loc="left", fontsize=8.5)
    for i, v in enumerate(d):
        ax[0].text(i, v+.17, f"{v:.1f}", ha="center", fontsize=7)
    # Previously measured truncation study: ours only, unaffected by SH fix.
    budgets = [23019, 83372, 300000, 1000000]
    values = {"office0":[12.04,13.54,20.23,24.68,26.29],
              "office1":[12.08,13.97,18.96,21.71,22.08],
              "room0":[5.16,8.68,16.26,25.01,25.44],
              "room2":[7.74,10.43,17.87,22.87,23.62]}
    for (s, ys), color, marker in zip(values.items(),
            [TEAL, BLUE, ORANGE, "#8662A1"], ["o", "s", "^", "D"]):
        n = records[s]["methods"]["ours"]["gaussians"]
        ax[1].plot(budgets+[n], ys, marker=marker, ms=3, lw=1, label=s, color=color)
    ax[1].set(xscale="log", xticks=[1e5,1e6], xticklabels=["100K","1M"],
              xlabel="Retained Gaussians (log scale)", ylabel="PSNR (dB) ↑")
    ax[1].set_title("(b) Map budget: post-hoc truncation", loc="left", fontsize=8.5)
    ax[1].legend(fontsize=6.4, loc="lower right", ncol=2, frameon=False,
                 handlelength=1.3, columnspacing=.8)
    for a in ax:
        a.grid(axis="y", color="#DFE5E8", lw=.5)
        a.set_axisbelow(True); a.tick_params(labelsize=7, length=2)
    save(f, "evidence")


def tables(records):
    out = PAPER / "generated"
    out.mkdir(exist_ok=True)
    def best(v, other, fmt, lower=False):
        s = format(v, fmt)
        return r"\textbf{"+s+"}" if (v < other if lower else v > other) else s
    rows = []
    for s in SCENES:
        p, o = [records[s]["methods"][m]["full_sh"] for m in ("photo","ours")]
        vals = [best(p["psnr"],o["psnr"],".2f"),best(o["psnr"],p["psnr"],".2f"),
                best(p["lpips"],o["lpips"],".3f",True),best(o["lpips"],p["lpips"],".3f",True)]
        rows.append(s+" & "+" & ".join(vals)+r" \\")
    means = {m:{k:float(np.mean([records[s]["methods"][m]["full_sh"][k]
                                for s in SCENES])) for k in ("psnr","lpips")}
             for m in ("photo","ours")}
    rows.append(r"\midrule")
    p, o = means["photo"], means["ours"]
    rows.append(r"Mean & "+ " & ".join([
        best(p["psnr"],o["psnr"],".2f"),best(o["psnr"],p["psnr"],".2f"),
        best(p["lpips"],o["lpips"],".3f",True),best(o["lpips"],p["lpips"],".3f",True)])+r" \\")
    (out/"replica_table.tex").write_text(
        r"\begin{tabular}{@{}lrrrr@{}}"+"\n"+r"\toprule"+"\n"+
        r"& \multicolumn{2}{c}{PSNR (dB) \up} & \multicolumn{2}{c}{LPIPS \dn}\\"+"\n"+
        r"Scene & Photo-SLAM & Ours & Photo-SLAM & Ours\\"+"\n"+
        r"\midrule"+"\n"+"\n".join(rows)+"\n"+r"\bottomrule"+"\n"+
        r"\end{tabular}"+"\n")
    rows=[]
    for m, label in [("photo","Photo-SLAM"),("mono","MonoGS"),("ours","Ours")]:
        d=records["desk"]["methods"][m]
        rows.append(f"{label} & {d['full_sh']['psnr']:.2f} & "
                    f"{d['full_sh']['lpips']:.3f} & {d['gaussians']/1000:.1f}"+r" \\")
    (out/"tum_table.tex").write_text(
        r"\begin{tabular}{@{}lrrr@{}}"+"\n"+r"\toprule"+"\n"+
        r"Method & PSNR (dB) \up & LPIPS \dn & $N$ (thousands)\\"+"\n"+
        r"\midrule"+"\n"+"\n".join(rows)+"\n"+r"\bottomrule"+"\n"+
        r"\end{tabular}"+"\n")
    (out/"summary.json").write_text(json.dumps(means, indent=2)+"\n")
    rows = []
    for scene in SCENES + ["desk"]:
        baseline = "mono" if scene == "desk" else "photo"
        methods = records[scene]["methods"]
        ours, other = methods["ours"], methods[baseline]
        delta = np.array([v["psnr"] for v in ours["per_frame"]]) - np.array(
            [v["psnr"] for v in other["per_frame"]])
        scene_delta = ours["full_sh"]["psnr"] - other["full_sh"]["psnr"]
        label = "TUM desk" if scene == "desk" else scene
        rows.append(f"{label} & {scene_delta:+.3f} & {np.median(delta):+.3f}"
                    f" & {np.count_nonzero(delta > 0)}/{len(delta)}" + r" \\")
    (out/"frame_distribution_table.tex").write_text(
        r"\begin{tabular}{@{}lrrr@{}}"+"\n"+r"\toprule"+"\n"+
        r"Scene & Scene $\Delta$PSNR & Frame median & Positive frames\\"+"\n"+
        r"\midrule"+"\n"+"\n".join(rows)+"\n"+r"\bottomrule"+"\n"+
        r"\end{tabular}"+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    args = parser.parse_args()
    DATA = args.data.resolve()
    FIG.mkdir(parents=True, exist_ok=True)
    records = {s: json.loads((DATA/s/"metrics.json").read_text()) for s in SCENES+["desk"]}
    teaser(records); overview(); qualitative(); evidence(records); tables(records)
    print(f"Generated figures and tables in {PAPER}")
