"""Render the paper's native TikZ system overview with pdflatex.

The manuscript inputs the same .tikz source directly. This helper creates
standalone PDF and, when Poppler is installed, SVG/PNG copies for inspection.
"""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "docs/Thesis"


def build_overview():
    env = os.environ.copy()
    texbin = Path("/usr/local/texlive/2026/bin/x86_64-linux")
    if texbin.is_dir():
        env["PATH"] = str(texbin) + os.pathsep + env.get("PATH", "")
    compiler = shutil.which("pdflatex", path=env["PATH"])
    if not compiler:
        raise RuntimeError("pdflatex is required to render the TikZ overview")
    out = PAPER / "ral/fig"
    with tempfile.TemporaryDirectory(prefix="ral-overview-") as tmp:
        proc = subprocess.run(
            [compiler, "-interaction=nonstopmode", "-halt-on-error",
             f"-output-directory={tmp}", "-jobname=overview",
             "ral/overview-standalone.tex"],
            cwd=PAPER, env=env, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(proc.stdout)
        shutil.copyfile(Path(tmp)/"overview.pdf", out/"overview.pdf")
    if shutil.which("pdftocairo"):
        subprocess.run(["pdftocairo", "-svg", str(out/"overview.pdf"),
                        str(out/"overview.svg")], check=True)
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-singlefile", "-r", "180", "-png",
                        str(out/"overview.pdf"), str(out/"overview")], check=True)
    print(f"Rendered TikZ overview: {out/'overview.pdf'}")


if __name__ == "__main__":
    build_overview()
