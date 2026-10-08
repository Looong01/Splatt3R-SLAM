"""Create a flat, minimal Overleaf upload for the anonymous RA-L manuscript."""
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/Thesis"
OUTPUT = ROOT / "docs/Splatt3R-SLAM-RA-L-Overleaf.zip"


def package():
    files = [PAPER / name for name in (
        "ral.tex", "ral-review.tex", "main.bib", "ieeeconf.cls")]
    files.extend(
        path for path in (PAPER / "ral").rglob("*")
        if path.is_file() and path.suffix in {".tex", ".tikz", ".pdf"}
    )

    with zipfile.ZipFile(
            OUTPUT, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(files)):
            archive.write(path, path.relative_to(PAPER).as_posix())

        review_source = (PAPER / "ral-review.tex").read_text(encoding="utf-8")
        archive.writestr("main.tex", "\\pdfminorversion=5\n" + review_source)
        archive.writestr("README.md", """# Splatt3R-SLAM RA-L Overleaf package

Upload this ZIP as a new Overleaf project. The root `main.tex` builds the
anonymous RA-L review manuscript. If Overleaf does not select the settings
automatically, choose:

- Main document: `main.tex`
- Compiler: `pdfLaTeX`

`ral.tex` is the authored IEEEtran reading version. All included figure PDFs
are compatible with pdfTeX PDF 1.5.
""")

    print(f"{OUTPUT}: {OUTPUT.stat().st_size:,} bytes, {len(set(files)) + 2} files")


if __name__ == "__main__":
    package()
