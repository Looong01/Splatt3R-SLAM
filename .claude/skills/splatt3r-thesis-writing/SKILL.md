---
name: splatt3r-thesis-writing
description: Writing and verification guidance for the independently formatted RA-L paper and MSc monograph in docs/Thesis. Read before changing manuscript content, figures, citations or reported results.
metadata:
  type: reference
---

# Splatt3R-SLAM manuscripts

Updated 2026-10-07. The user requested removal of the superseded conference
draft and synchronisation of the MSc scientific content with the RA-L revision.
Do not restore the removed draft, template, old montages or build target.

## Document boundaries

- `ral.tex` builds the authored IEEEtran journal-style reading draft.
- `ral-review.tex` builds the anonymous ieeeconf review draft.
- Both RA-L drivers share `ral/` prose and IEEE bibliography style.
- `master.tex` builds the MSc monograph: report class, A4, 12pt, 2.5 cm
  margins, 1.5 spacing, 12 chapters and four appendices, with `unsrtnat`.
- MSc prose is in `chap/`; its float wrappers are in `tab/`.
- Shared scientific assets are `ral/fig/`, `ral/generated/` and `main.bib`.
  Do not import the RA-L preamble, prose, captions or numbering into MSc.
- Preserve long-form explanations and negative experiments in the monograph.
  Synchronisation means consistent scientific content, not identical text.

The RA-L PDFs were verified at seven pages including references. The official
initial/revision template is ieeeconf; IEEEtran is the final journal style.
The authored file is a reading draft, not an accepted camera-ready paper.
RA-L allows six pages plus at most two extra; seven pages entails an extra-page
charge. Do not invent acceptance metadata, coauthors, DOI or editor names.
Official guidance: https://www.ieee-ras.org/publications/ra-l/information-for-authors-ra-l

The MSc formatting follows the existing project convention. No authoritative
programme-specific template or page-length requirement has been verified.
Author-supplied programme, student number and supervisors live in the title
page and must be preserved. Second reader and personalised acknowledgements
remain placeholders.

## Scientific evidence and corrections

Read `docs/Thesis/ral/REVISION_NOTES.md` and
`docs/Thesis/MASTER_SYNC_NOTES.md` for current evidence. Historical experiments
are in `splatt3r-finetuning-experiments` sections 17.50, 17.79, 17.82,
17.92--17.95, `docs/external-baselines.md`, and the online campaign records.
Historical prose is not authoritative when contradicted by current evidence.

- Fresh external rendering evidence:
  `logs/paper_ral_20261007/<scene>/metrics.json` and PNG renders.
- Retain all exported SH bands. The old `scripts/eval_map_quality.py`
  discards higher bands and is not the current external evaluation harness.
- Replica means: Ours 23.019214 dB / 0.134862 LPIPS, Photo-SLAM
  19.482997 / 0.162673. PSNR wins 8/8, LPIPS wins 5/8.
- State the result directly as state-of-the-art performance. The evidence is
  the all-eight-scene local comparison above; no new Splat-SLAM, SEGS-SLAM,
  GSO-SLAM or MyGO-Splat runs are planned for this revision. Keep those
  systems in related-work positioning and do not mix their paper-reported
  scores into the measured table.
- TUM desk: Ours 13.949010 / 0.411030, MonoGS 13.913588 / 0.395311,
  Photo-SLAM 10.665049 / 0.562209. Do not mix the old MonoGS row into this table.
- External comparison uses released head, no thinning, fixed centres and
  300 s polish. Head adaptation and thinning are separate paired experiments.
- Cameras are 100 common non-keyframes per scene, GT Sim3-aligned; resolution
  512x288 on Replica and 512x384 on TUM. MonoGS keyframe association uses
  saved GT poses, not raw indices confused with its associated frame stream.
- Non-keyframes may supervise refinement. Call this reconstruction
  evaluation, not a strict unseen-view test.
- Scene PSNR uses log of mean frame MSE; dataset mean weights scenes equally.
  Median per-frame differences and win counts are different statistics.
- Photo-SLAM's Replica budget differs. MonoGS's 339 s colour phase is close
  to our 300 s polish, but total compute is not matched.
- Opacity changes visibility of foreground and background. Black-background
  compositing is not generally monotone in luminance for multicolour stacks.
  Colour is input-RGB SH plus residual and clamping, not a sigmoid head.
  Five tested correlation heads are not five independent dataset families.
- Keep the true thinning equation:
  alpha' = alpha * clip(1 - 2 lambda (1 - rank(conf)), 0.1, 1).
  Confidence allocation is not clearly superior to uniform attenuation.
- Do not revive the universal 8.7 dB protocol-offset claim or the retracted
  0.66-pixel depth-observability explanation. Retain measured diagnostics
  with their original configurations and limited scope.
- Negative geometry-correction and LoRA experiments reject tested routes,
  not all possible geometric correction or adaptation procedures.
- Fixed-centre offline results are not a universal causal-delivery guarantee.
  State actual experiment settings separately from CLI defaults.
- Post-hoc truncation is not an optimum small-map bound and does not prove
  all memory is essential. Costs measured with refinement off exclude polish.
- SharedKeyframes uses CUDA IPC; supervision pools and snapshots are CPU
  shared memory. The GUI-enabled system has four processes.

## Figures

The main method overview is generated as native editable PowerPoint shapes by
`scripts/paper/make_overview_pptx.py`. Separate RA-L and MSc PDF/SVG exports
carry their own section and equation numbers. Real before/after map views come
from `scripts/paper/render_refinement_pair.py`. The older
`ral/fig/overview.tikz` follows `docs/架构与数据流.md` lines 83--109 and
remains as the process/device implementation figure.

`make_ral_figures.py` creates the method/performance teaser, four-column
GT / Photo-SLAM / MonoGS / Ours comparison with common crops, scene-quality
and truncation chart, and numeric tables including MSc per-frame analysis.
Inputs are measured renders, not synthetic images.
Representative frames are the two middle ranks of per-frame PSNR difference,
against MonoGS on TUM and Photo-SLAM on Replica. Preserve that selection rule
and identical crop coordinates. No per-image score overlays or old base/head
montages belong in the current manuscripts.

## Build and verify

Use pdfLaTeX and distribution fonts; do not switch to OS-dependent fontspec.
The MSc microtype, emergency stretch and compact chapter-head settings prevent
overflow at 12pt/1.5 spacing. Keep short optional captions for lists of figures
and tables. Never pad scientific prose merely to fill a short page.

From the repository root:

```bash
python scripts/paper/make_ral_figures.py
cd docs/Thesis
./build.sh master
./build.sh ral
./build.sh review
```

Check logs for undefined references/citations, overfull boxes and errors.
Inspect rendered title/front-matter and figure/table pages, verify A4 vs Letter
and embedded fonts. Check current values against metrics JSON, including
bolding across all five ATE methods. Avoid assertions of significance from
single-run numerical margins.

`scripts/paper/package_ral.py` creates `docs/Splatt3R-SLAM-RA-L-source.zip`.
`scripts/paper/package_master.py` creates the MSc-only `docs/Thesis.zip`.
Each includes scientific assets and measured evidence and must rebuild after
extraction. MSc does not need IEEE classes. RA-L does not need MSc chapters.
Neither source package contains the datasets, checkpoints or large saved maps
needed to rerun GPU reconstruction/rendering.
