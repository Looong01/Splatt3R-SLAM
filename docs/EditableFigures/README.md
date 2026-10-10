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

The method-navigation slides and teaser now share the white-background
Figure 1--2 geometry in `scripts/paper/draw_paper_diagrams.py`.
The dedicated four-slide deck is `docs/Thesis/Splatt3R-SLAM-figures-1-3.pptx`
(RA-L Figures 1--2, the restored Fig. 3 preview, then the MSc method variant).
Rebuild it with `python scripts/paper/draw_paper_diagrams.py --pptx`.
Fig. 1--2 labels, arrows, diagrams and crop frames are native editable
objects. Fig. 3 remains editable in `docs/Thesis/ral/fig/overview.tikz`;
its PowerPoint page is a preview. The paper does not depend on this export.
