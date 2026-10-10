# RA-L revision evidence — 2026-10-07

The requested revision replaces the default conference manuscript with a
concise IEEE RA-L paper, adds a system overview, redesigns the teaser to show
method and measured performance, and uses method-per-column qualitative
comparisons (Ground truth / Photo-SLAM / MonoGS / Ours on TUM). The MSc
monograph has now been synchronised to this scientific content while
retaining independent chapter-based formatting. The superseded conference
draft and its dedicated template have been removed at the user's request.

## Source audit

All seven `.claude/skills/*/SKILL.md` files were read, including the full
14,334-line experiment ledger. The published Splatt3R and MASt3R-SLAM PDFs
under `docs/third_party` and the existing paper sections were consulted.
Reference for presentation: Unblur-SLAM, arXiv:2603.26810v1, especially its
overview separating inherited and proposed modules.

Important corrections to carry into the revised paper:

- `main.py` offers tracked non-keyframes to `SupervisionFrames`; excluding
  keyframes in evaluation does **not** guarantee unseen optimisation views.
  Call this **non-keyframe reconstruction evaluation**, not a strict
  train/test novel-view benchmark.
- `scripts/eval_map_quality.py` discards higher SH bands. The new paper
  renderer retains all exported bands, and also records DC-only scores to
  reproduce the historical table.
- Existing all-eight-scene external-comparison maps use the **released
  Splatt3R head + refinement**, not family-adapted heads. Do not attribute
  their inter-method margin to head training. Adaptation has its own paired
  ablations in the experiment ledger.
- The evaluated configuration fixes Gaussian centres. Colour is input RGB
  plus SH residual, clamped at conversion; it is not a sigmoid colour head.
  Alpha changes do not universally brighten/darken a multicolour stack.
  Avoid presenting the old “brightness dial” interpretation as a theorem.
- The old 0.66-pixel depth-observability argument was corrected by ledger
  §17.50 (contribution-weighted baseline ~0.78 m); omit that stale claim.
- Two colour-augmented Replica heads plus three real-data heads are **five
  heads**, not five independent dataset families.
- Do not call 0.08 dB smaller than a 0.031 dB noise floor. Report the small
  absolute margin and avoid unsupported statistical-equivalence language.

## Fresh rendered evidence

Command (existing `splatt3r-slam` conda environment):

```bash
CUDA_VISIBLE_DEVICES=0 /home/share-v5/miniconda3/envs/splatt3r-slam/bin/python \
  scripts/paper/render_comparisons.py --scenes office0 office1 office2 office3
CUDA_VISIBLE_DEVICES=1 /home/share-v5/miniconda3/envs/splatt3r-slam/bin/python \
  scripts/paper/render_comparisons.py --scenes desk office4 room0 room1 room2
```

Outputs: `logs/paper_ral_20261007/<scene>/metrics.json` and two representative
views per scene, each with GT and every participating method. Each JSON holds
SHA256 and size of input PLY/trajectory, Sim3 alignment, all 100 frame indices,
per-frame metrics, full-SH and DC-only aggregates, and selection provenance.
Images use the two middle ranks of per-frame PSNR difference (Ours−MonoGS for
TUM, Ours−Photo-SLAM for Replica). No favourable-tail selection.

Replica all-eight means, full exported SH:

| Method | PSNR (dB) | LPIPS |
|---|---:|---:|
| Ours, released head + 300 s polish | 23.019214 | 0.134862 |
| Photo-SLAM | 19.482997 | 0.162673 |

Difference: +3.536218 dB; PSNR 8/8 and LPIPS 5/8 numerical wins. DC-only
Photo-SLAM values reproduce the historical table (mean 19.5805 dB), and ours
reproduces it too. PSNR within a scene is `-10 log10(mean frame MSE)`; means
above are unweighted means of scene scores.

TUM desk full-SH fresh evaluation:

| Method | PSNR (dB) | LPIPS |
|---|---:|---:|
| Ours, 300 s | 13.949010 | 0.411030 |
| MonoGS, 339 s colour refinement | 13.913588 | 0.395311 |
| Photo-SLAM | 10.665049 | 0.562209 |

TUM MonoGS keyframes are recovered from its saved GT poses and checked against
GT orientation, avoiding confusion between its 32 Hz RGB-D associated indices
and raw RGB indices. Use these newly measured values consistently; do not mix
with the historical MonoGS 13.8685/0.3975 row.

## RA-L template source

Official author instructions:
https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l

The fetched current page states 6 pages plus at most 2 extra, **double
anonymous** first submission, `ieeeconf` for first/revised submissions,
`IEEEtran` journal for final style. A search-index cache still says single
blind; prefer the freshly fetched page. Prepare an authored journal-style
reading draft and an anonymous `ieeeconf` review version. Do not add invented
acceptance dates, editor names, DOI or coauthors.

`ieeeconf.cls` downloaded unmodified from
https://ras.papercept.net/conferences/support/files/ieeeconf.zip
(ZIP: 102,812 bytes, also contains root.tex/root.pdf).
Installed journal class:
`/usr/local/texlive/2026/texmf-dist/tex/latex/ieeetran/IEEEtran.cls`.

Known author from current manuscript: Zelong Li,
`loong.li2@student.uva.nl`. Drop phone number; do not invent coauthors.

## Figure and text scope

Teaser: pose-anchored feed-forward Gaussians + separate refinement, two real
renderings, and an all-eight-scene performance plot. State protocol and
300 s polish in caption. No per-image “base PSNR=10” overlays.

System overview: input frame + current keyframe → frozen shared backbone;
pointmap/matching branch → inherited MASt3R-SLAM tracker/pose graph; Gaussian
head branch → local anchored maps → opacity thinning → refiner → renderer.
Pose graph updates both map anchors and stored anchor-relative supervision.
Distinguish offline head adaptation from online map optimisation visually.

Main qualitative figure: four columns GT / Photo-SLAM / MonoGS / Ours,
two TUM views with common crops. Replica comparisons may use three columns
where only two renderable monocular methods have valid artifacts; do not fill
missing methods with point-cloud screenshots or RGB-D results.

Keep ablations in compact tables or charts instead of the old base/ours
montages. Retain material cost and scope limits in concise academic prose.
The all-eight-scene result supports the direct claim that Splatt3R-SLAM
achieves state-of-the-art performance. Do not mix paper-reported scores into
the measured table. No submission/publication is authorized or needed.


## Delivered manuscript and verification

- `ral.tex`: authored IEEEtran journal-style draft, 7 pages including references.
- `ral-review.tex`: anonymous ieeeconf version, 7 pages; empty PDF author metadata.
- Shared content lives in `ral/`; default `build.sh` now builds RA-L.
- Four newly drawn figures: teaser, system overview, four-column TUM comparison
  with identical crops, and scene-quality/map-budget evidence. PDF, SVG, PNG and
  generation scripts are retained. No old base/head montages enter the new PDF.
- External comparisons additionally have **thinning disabled** (`conf-fade 0`),
  as documented in the original experiment configuration. Optional adaptation
  and thinning are evaluated separately, not credited for the external margin.
- Actual common-render dimensions are **512 x 288 on Replica**, **512 x 384 on
  TUM**. Dimensions are recorded in each new metrics JSON. This corrects the
  legacy blanket 512 x 384 description.
- New Replica/TUM tables are generated from metrics JSON. ATE values come from
  `tab/ate.tex`, with corrected bolding over all five methods. Paired refinement
  deltas come from ledger section 17.92.4, head adaptation from section 17.82,
  insertion experiments from 17.79, causal replay/dual-GPU measurements from
  the online-refinement record. Cost and truncation values come from `tab/cost.tex`
  and `tab/curve.tex`; only our degree-zero maps enter the truncation chart, so
  the external Photo-SLAM SH correction does not change those points.
- No new model training or SLAM reconstruction was needed: the new experiment
  re-renders saved maps with corrected full-SH evaluation and frame association.
- Both PDFs compile with no overfull boxes or undefined references/citations.
  Some underfull box notices remain from IEEE column/float composition; rendered
  pages were inspected for overlap, clipping and figure/caption readability.
  All fonts are embedded; no bitmap Type 3 text fonts are used.
- The source ZIP includes measured JSON/PNG evidence and can regenerate the
  figures on CPU and rebuild both PDFs. GPU rendering still requires the full
  project environment, datasets and saved maps referenced in the JSON files.


## User-directed system overview replacement

The user selected `docs/架构与数据流.md#L83-109` as the overview source and
requested LaTeX rendering. `ral/fig/overview.tikz` now draws the four-process
architecture: main/tracker, backend and viz on GPU 0; refiner on GPU 1;
SupervisionFrames and RefinedMapSnapshot in CPU shared memory. The figure
preserves the source's numbered flows and uses English labels for the paper.
The CPU snapshot is grouped by its actual storage location, and viz reads
SharedKeyframes as specified by the source document's Mermaid version.
The 10 mm deduplication option is explicitly optional (default voxel size is 0).

The TikZ source is included in both manuscripts as the process/device
implementation figure.
`ral/overview-standalone.tex` and `scripts/paper/build_overview.py` export
PDF/SVG/PNG copies from that source. The main method-navigation figure was
subsequently replaced as described below.

## Narrative and method-figure revision

A later editorial pass changed the title to `Splatt3R-SLAM: From Image Pairs
to Persistent Maps` and organised the paper as predict, anchor, and refine.
The process-oriented TikZ figure is included as the system implementation
overview. Separate RA-L and MSc method-navigation figures are generated by
`scripts/paper/make_ral_figures.py`; they use real TUM input and map renders,
including an unrefined/refined pair rendered at the same camera.

The RA-L abstract is approximately 155 words. Fine-grained tables now include
all 12 recorded opacity-attenuation cells and the 120/500/1000/3000 iteration
control. The paper remains seven pages in both IEEEtran and anonymous
ieeeconf formats.

## Figure 1--2 redraw, 2026-10-10

The teaser now separates the shared-anchor schematic, common-camera image
comparison with identical crops, and the eight-scene Replica mean. Grey marks
the old schematic pose; blue marks camera geometry and green marks the map.
The image and chart panels have separate gutters.

The method figure now has explicit geometry and Gaussian branches, a shared
map/camera state, and one connected Render -> Loss -> Adam -> map loop. It
retains actual before/after images and separate RA-L/MSc equation references.
The earlier experiment that also redrew Fig. 3 was reverted at the user's
request. Fig. 3 remains the original TikZ process diagram with GPU 0, GPU 1,
CPU shared-memory regions, rounded process nodes, and numbered flows.

`scripts/paper/draw_paper_diagrams.py` defines the Fig. 1--2 geometry, exports
Matplotlib PDF/SVG/PNG figures, and optionally exports native editable
PowerPoint objects. `make_ral_figures.py` uses it for Figures 1--2 and only
builds the existing Fig. 3 TikZ source. Paper builds remain independent of
PowerPoint. Exact image
hashes, the shared teaser crop, nominal sizes and minimum label sizes are
recorded in `ral/fig/figures_1_3_manifest.json`. Experimental values and
representative-view selection are unchanged.
