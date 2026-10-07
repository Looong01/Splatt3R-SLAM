#!/bin/bash
# Build the manuscript. Works identically on Linux and Windows (MiKTeX/TeX Live)
# because the document uses only distribution-bundled Type 1 fonts via `times`.
#
#   ./build.sh          -> ral.pdf    (IEEE RA-L journal-style reading draft)
#   ./build.sh review   -> ral-review.pdf (anonymous ieeeconf review version)
#   ./build.sh master   -> master.pdf (MSc thesis monograph)
#   ./build.sh clean    -> remove build artefacts
set -euo pipefail
cd "$(dirname "$0")"
[ -d /usr/local/texlive/2026/bin/x86_64-linux ] && \
  export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH

if [ "${1:-}" = clean ]; then
  rm -f *.aux *.bbl *.toc *.lof *.lot *.blg *.brf *.log *.out *.fls *.fdb_latexmk *.synctex.gz chap/*.aux tab/*.aux ral/*.aux
  echo "cleaned"; exit 0
fi

command -v pdflatex >/dev/null || { echo "pdflatex not on PATH (TeX Live still installing?)"; exit 1; }

DOC=ral
case "${1:-}" in
  ral|paper|"") DOC=ral ;;
  review|ral-review) DOC=ral-review ;;
  master|thesis) DOC=master ;;
  *) echo "usage: $0 [ral|review|master|clean]"; exit 2 ;;
esac

if command -v latexmk >/dev/null; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error $DOC.tex
else
  pdflatex -interaction=nonstopmode -halt-on-error $DOC.tex
  bibtex $DOC
  pdflatex -interaction=nonstopmode -halt-on-error $DOC.tex
  pdflatex -interaction=nonstopmode -halt-on-error $DOC.tex
fi
echo "--- built: $DOC.pdf $(ls -la $DOC.pdf | awk '{print $5" bytes"}') ---"
grep -c "Warning" $DOC.log || true
