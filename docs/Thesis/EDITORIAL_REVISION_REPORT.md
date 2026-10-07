# Narrative revision report

## Editorial objective

Both manuscripts now follow one claim:

> Pairwise Gaussian predictions become a persistent map when each local map
> and its supervision cameras share a keyframe anchor.

The exposition follows `problem -> idea -> mechanism -> controlled evidence ->
external result -> remaining cost`. Technical caveats remain where they affect
interpretation, but repeated defensive statements and research-process
commentary were removed.

Final title: **Splatt3R-SLAM: From Image Pairs to Persistent Maps**.

## Main changes

- RA-L abstract reduced from 215 to about 155 words.
- MSc abstract reduced from 391 to about 180 words.
- MSc introduction reduced by about 40% and organised around the shared-anchor
  idea, four research questions, and claim-linked contributions.
- Method rewritten as three verbs: **Predict, Anchor, Refine**.
- Added the scoped identity
  `T_f^{-1} T_k = T_kf^{-1}` for a map and camera sharing anchor `k`.
- Added the 6% blockwise Sim(3) anchor-following control.
- Added all 12 recorded opacity-attenuation cells and the 120/500/1000/3000
  iteration-budget control.
- Negative experiments rewritten as `question -> intervention -> end-to-end
  result -> design consequence`.
- Discussion now separates anchor scope, map size, online budget, evaluation
  scope, and statistical coverage.
- Removed anthropomorphic and self-referential phrases, including the
  "photometric term winning the argument" and "results that flatter the
  authors" passages.

The RA-L experiments section is slightly longer because it now contains the
fine-grained ablation evidence requested by the author. The abstract, method,
discussion, MSc introduction, negative-results chapter, and conclusion are
substantially shorter.

## Figure revision

`ral/fig/Splatt3R-SLAM-scientific-figures.pptx` contains six editable slides:

1. RA-L method navigation with section and equation references.
2. MSc method navigation with chapter and equation references.
3. Shared-anchor identity, controlled pose update, and refinement loop.
4. TUM GT / Photo-SLAM / MonoGS / Ours comparison with identical crops.
5. Replica multi-scene GT / Photo-SLAM / Ours comparison.
6. All 12 opacity-attenuation cells and four refiner on/off controls.

Text, boxes, connectors, camera symbols, Gaussian ellipses, and bars are native
PowerPoint objects. Raster elements are measured input or rendered images.
The overview before/after pair is rendered from the saved unrefined and
refined TUM maps at the same recorded camera. Metadata is stored in
`ral/fig/overview_assets/metadata.json`.

The RA-L and MSc overview PDFs are exported separately because their section
and equation numbers differ.

## Evidence and scope

- Replica: 23.019214 dB / 0.134862 LPIPS for Ours, versus
  19.482997 / 0.162673 for Photo-SLAM; PSNR improves on 8/8 scenes and LPIPS
  on 5/8.
- TUM desk: 13.949010 / 0.411030 for Ours, 13.913588 / 0.395311 for MonoGS,
  and 10.665049 / 0.562209 for Photo-SLAM.
- External maps use the released head, no opacity attenuation, fixed centres,
  and 300 s post-sequence refinement.
- Non-keyframes may supervise refinement. The paper therefore calls this
  reconstruction evaluation, not strict unseen-view evaluation.
- MonoGS is omitted from Replica qualitative panels because no comparable
  monocular artifact was obtained.

## Writing references

The revision follows the exposition pattern of recent Nature and Science
Robotics papers: concrete problem first, one central mechanism, claim-led
result sections, and technical detail after orientation.

- Nature 641, 1180-1187 (2025), DOI 10.1038/s41586-025-09005-y.
- Science Robotics 10(109) (2025), DOI 10.1126/scirobotics.adv1818.
- Science Robotics 11(114) (2026), DOI 10.1126/scirobotics.aej0223.

These papers are writing references only. They are not technical baselines and
were not added to `main.bib`.
