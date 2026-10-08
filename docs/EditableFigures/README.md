# Editable scientific figures

This directory is independent of the RA-L and MSc manuscript builds.
`Splatt3R-SLAM-scientific-figures.pptx` contains seven editable slides.
All text, boxes, connectors, camera symbols, Gaussian symbols, and chart bars
are native objects. Embedded raster elements are measured input or rendered
images.

Regeneration:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/paper/render_refinement_pair.py
PYTHONPATH=/path/to/python-pptx python scripts/paper/make_overview_pptx.py
```

The first command requires the full datasets and saved maps. The manuscript
figures are generated separately by `scripts/paper/make_ral_figures.py`; no
paper build reads this directory.
