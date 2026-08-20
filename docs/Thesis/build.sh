#!/bin/bash
# Build the manuscript. Works identically on Linux and Windows (MiKTeX/TeX Live)
# because the document uses only distribution-bundled Type 1 fonts via `times`.
#
#   ./build.sh          -> main.pdf
#   ./build.sh clean    -> remove build artefacts
set -e
cd "$(dirname "$0")"
[ -d /usr/local/texlive/2026/bin/x86_64-linux ] && \
  export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH

if [ "$1" = clean ]; then
  rm -f *.aux *.bbl *.blg *.brf *.log *.out *.fls *.fdb_latexmk *.synctex.gz sec/*.aux
  echo "cleaned"; exit 0
fi

command -v pdflatex >/dev/null || { echo "pdflatex not on PATH (TeX Live still installing?)"; exit 1; }

if command -v latexmk >/dev/null; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
fi
echo "--- built: $(ls -la main.pdf | awk '{print $5" bytes"}') ---"
grep -c "Warning" main.log || true
