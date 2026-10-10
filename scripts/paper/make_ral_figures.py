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
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/Thesis/ral"
DATA = ROOT / "logs/paper_ral_20261007"
FIG = PAPER / "fig"
ASSET = FIG / "overview_assets"
SCENES = [f"office{i}" for i in range(5)] + [f"room{i}" for i in range(3)]
INK, GRAY, BLUE, TEAL, ORANGE, MAGENTA = (
    "#172B3A", "#647887", "#4079AC", "#007E78", "#B77A2E", "#A14D78")
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
    import draw_paper_diagrams as diagrams
    diagrams.DATA = DATA
    diagrams.teaser(records).render("teaser")
    diagrams.write_manifest(records)


def method_overview(variant):
    import draw_paper_diagrams as diagrams
    diagrams.DATA = DATA
    diagrams.method(variant).render(f"method_overview_{variant}")


def qualitative_replica():
    w, h = 7.15, 4.55
    f, a = canvas(w, h)
    methods = [("gt", "Ground truth"), ("photo", "Photo-SLAM"),
               ("ours", "Splatt3R-SLAM")]
    scenes = [("office0", "office0"), ("office2", "office2"),
              ("room0", "room0"), ("room2", "room2")]
    x0, iw, gap = .68, 2.05, .18
    for col, (_, label) in enumerate(methods):
        x = x0 + col * (iw + gap)
        text(a, x + iw / 2, 4.42, label, 8.5, ha="center",
             color=TEAL if col == 2 else INK, fontweight="bold")
    for row, (scene, label) in enumerate(scenes):
        y = 3.30 - row * 1.04
        text(a, .58, y + .43, label, 7, ha="right", color=BLUE,
             fontweight="bold")
        for col, (method, _) in enumerate(methods):
            x = x0 + col * (iw + gap)
            image(f, DATA / scene / f"view0_{method}.png",
                  [x/w, y/h, iw/w, .94/h],
                  outline=TEAL if method == "ours" else "#D8E0E5")
    save(f, "qualitative_replica")


def overview():
    from draw_paper_diagrams import system, write_system_tikz
    from build_overview import build_overview
    write_system_tikz(system())
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
    teaser(records)
    method_overview("ral")
    method_overview("master")
    overview()
    qualitative()
    qualitative_replica()
    evidence(records)
    tables(records)
    print(f"Generated figures and tables in {PAPER}")
