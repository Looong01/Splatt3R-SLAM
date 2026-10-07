"""Package the RA-L manuscript, editable figures and measured render evidence."""
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/Thesis"
OUTPUT = ROOT / "docs/Splatt3R-SLAM-RA-L-source.zip"


def package():
    files = [PAPER / name for name in (
        "ral.tex", "ral-review.tex", "ral.pdf", "ral-review.pdf",
        "main.bib", "ieeeconf.cls", "build.sh", "README.md",
        "EDITORIAL_REVISION_REPORT.md")]
    files.extend(p for p in (PAPER/"ral").rglob("*") if p.is_file())
    files.extend(ROOT / "scripts/paper" / name for name in (
        "make_ral_figures.py", "build_overview.py", "render_comparisons.py",
        "render_refinement_pair.py", "make_overview_pptx.py",
        "package_ral.py"))
    files.extend(p for p in (ROOT/"logs/paper_ral_20261007").rglob("*")
                 if p.suffix in (".json", ".png"))
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(files)):
            z.write(p, "Splatt3R-SLAM-paper/" + p.relative_to(ROOT).as_posix())
    print(f"{OUTPUT}: {OUTPUT.stat().st_size:,} bytes, {len(set(files))} files")


if __name__ == "__main__":
    package()
