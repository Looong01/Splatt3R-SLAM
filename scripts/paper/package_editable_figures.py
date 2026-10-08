"""Package editable PowerPoint figures separately from manuscript sources."""
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "docs/EditableFigures"
PAPER_ASSET = ROOT / "docs/Thesis/ral/fig/overview_assets"
OUTPUT = ROOT / "docs/Splatt3R-SLAM-editable-figures.zip"

PPT_OUTPUTS = {
    "Splatt3R-SLAM-scientific-figures.pptx",
    "Splatt3R-SLAM-scientific-figures.pdf",
    "overview-ral.pdf", "overview-ral.svg",
    "overview-master.pdf", "overview-master.svg",
    "anchor-refinement.pdf", "anchor-refinement.svg",
    "qualitative-full.pdf", "qualitative-full.svg",
    "qualitative-replica.pdf", "qualitative-replica.svg",
    "ablations-full.pdf", "ablations-full.svg",
    "teaser-editable.pdf", "teaser-editable.svg",
    "pptx_manifest.json",
}


def package():
    files = [FIG / name for name in PPT_OUTPUTS if (FIG / name).is_file()]
    files.extend(p for p in PAPER_ASSET.rglob("*") if p.is_file())
    files.extend(ROOT / "scripts/paper" / name for name in (
        "make_overview_pptx.py", "render_refinement_pair.py"))
    files.extend(p for p in (ROOT / "logs/paper_ral_20261007").rglob("*")
                 if p.suffix in {".json", ".png"})

    prefix = "Splatt3R-SLAM-editable-figures/"
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, prefix + path.relative_to(ROOT).as_posix())
        archive.writestr(prefix + "README.md", """# Editable scientific figures

The PowerPoint deck is independent of the RA-L and MSc manuscript builds.
All labels, boxes, connectors, camera symbols, Gaussian symbols, and bars are
native editable objects. Raster elements are measured input or rendered images.

To regenerate the deck, install python-pptx and run:

    CUDA_VISIBLE_DEVICES=0 python scripts/paper/render_refinement_pair.py
    python scripts/paper/make_overview_pptx.py

The map-rendering step requires the full repository, datasets, and saved maps.
The included PPTX already embeds every raster image used by the slides.
""")
    print(f"{OUTPUT}: {OUTPUT.stat().st_size:,} bytes, {len(set(files)) + 1} files")


if __name__ == "__main__":
    package()
