"""Package the independently formatted MSc thesis and shared scientific assets."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
THESIS = ROOT / "docs/Thesis"
OUTPUT = ROOT / "docs/Thesis.zip"
PPT_ONLY = {
    "Splatt3R-SLAM-scientific-figures.pdf",
    "Splatt3R-SLAM-scientific-figures.pptx",
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
    files = [THESIS / name for name in (
        "master.tex", "master.pdf", "main.bib", "build.sh",
        "MASTER_SYNC_NOTES.md", "NARRATIVE_AUDIT.md",
        "MATHEMATICAL_CONSISTENCY_AUDIT.md",
        "SOTA_EVIDENCE_PLAN.md",
        "ral/overview-standalone.tex")]
    for folder, suffixes in (
        ("chap", {".tex"}), ("tab", {".tex"}),
        ("ral/fig", {".tikz", ".pdf", ".svg", ".png", ".json"}),
        ("ral/generated", {".tex", ".json"}),
    ):
        files.extend(p for p in (THESIS / folder).rglob("*")
                     if p.is_file() and p.suffix in suffixes
                     and p.name not in PPT_ONLY)
    files.extend(ROOT / "scripts/paper" / name for name in (
        "make_ral_figures.py", "draw_paper_diagrams.py", "build_overview.py",
        "render_refinement_pair.py"))
    files.extend(p for p in (ROOT / "logs/paper_ral_20261007").rglob("*")
                 if p.suffix in {".json", ".png"})
    prefix = "Splatt3R-SLAM-MSc/"
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, prefix + path.relative_to(ROOT).as_posix())
        archive.writestr(prefix + "README.md", """# Splatt3R-SLAM MSc thesis

Build: `cd docs/Thesis && ./build.sh master`.
Requires pdfLaTeX, BibTeX and a TeX Live/MiKTeX installation with TikZ,
report, times, natbib, unsrtnat, microtype and the other standard packages.
The PDF uses A4, 12-point text and 1.5 line spacing.

The `ral/fig` and `ral/generated` directories contain shared scientific assets,
not RA-L prose or formatting. No IEEE class is required to build the thesis.
The shared build script also recognises paper targets, whose drivers are
intentionally absent from this MSc-only archive.

To regenerate figures and data tables from the included evidence, run
`python scripts/paper/make_ral_figures.py` at this archive's root.
This requires numpy, matplotlib, Pillow, TeX with standalone, and Poppler.
The paper figures and build do not depend on PowerPoint.

Evidence includes full-SH metrics, frame indices, input hashes and rendered
images. New GPU rendering requires the full repository, datasets and maps,
which are not included. Second-reader and personalised acknowledgements
remain author placeholders. See docs/Thesis/MASTER_SYNC_NOTES.md.
""")
    print(f"{OUTPUT}: {OUTPUT.stat().st_size:,} bytes, {len(set(files)) + 1} files")


if __name__ == "__main__":
    package()
