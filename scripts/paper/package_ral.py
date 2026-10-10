"""Package the RA-L manuscript and measured render evidence without PPT files."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/Thesis"
OUTPUT = ROOT / "docs/Splatt3R-SLAM-RA-L-source.zip"
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
    files = [PAPER / name for name in (
        "ral.tex", "ral-review.tex", "ral.pdf", "ral-review.pdf",
        "main.bib", "ieeeconf.cls", "build.sh",
        "NARRATIVE_AUDIT.md", "MATHEMATICAL_CONSISTENCY_AUDIT.md",
        "SOTA_EVIDENCE_PLAN.md")]
    files.extend(p for p in (PAPER/"ral").rglob("*")
                 if p.is_file() and p.name not in PPT_ONLY)
    files.extend(ROOT / "scripts/paper" / name for name in (
        "make_ral_figures.py", "draw_paper_diagrams.py",
        "build_overview.py", "render_comparisons.py",
        "render_refinement_pair.py", "render_system_stages.py", "draw_revised_figures.py",
        "package_ral.py"))
    files.extend(p for p in (ROOT/"logs/paper_ral_20261007").rglob("*")
                 if p.suffix in (".json", ".png"))
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, "Splatt3R-SLAM-paper/" + p.relative_to(ROOT).as_posix())
        z.writestr("Splatt3R-SLAM-paper/README.md", """# Splatt3R-SLAM RA-L paper

Build from `docs/Thesis`:

    ./build.sh ral
    ./build.sh review

The two manuscript variants share `ral/document.tex`. Paper figures are
generated independently by `scripts/paper/make_ral_figures.py` and do not
depend on presentation software.
""")
    print(f"{OUTPUT}: {OUTPUT.stat().st_size:,} bytes, {len(set(files)) + 1} files")


if __name__ == "__main__":
    package()
