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

The current manuscript has one combined method/system overview. Its native
editable source is `scripts/paper/draw_contribution_overview.py`.
`docs/Thesis/Splatt3R-SLAM-figures-1-3.pptx` now contains:

1. TUM desk teaser with Photo-SLAM comparisons and a pulled-back 3D map.
2. Combined overview centred on coupled map–camera anchors.
3. Selected real-data qualitative views with shared detail crops.

Rebuild with `python scripts/paper/draw_paper_diagrams.py --pptx`.
Text, arrows, camera symbols, Gaussians and crop frames are native editable
objects; experimental renderings remain embedded images.
Selection and metric provenance: `docs/Thesis/ral/fig/publication_assets/selection.json`.
The old GPU/CPU implementation diagram remains in `ral/fig/overview.tikz` as
archived editable source; it is not a second overview in the manuscript.
