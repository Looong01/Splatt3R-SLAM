# Scientific narrative and editable figure revision

## User request and editorial direction

Rewrite both RA-L and MSc manuscripts, not just abstracts. Remove repeated
defensive statements, anthropomorphism and commentary about the writing or
research process. Explain the problem before terminology, use short familiar
English, and connect every contribution to evidence. Keep independent IEEE
and MSc formats. Deliver LaTeX, editable PPTX scientific figures, BibTeX and
compiled PDFs. The user's blur/RL example is an illustration of exposition,
not authorization to invent those modules for this method.

The screenshot specifically objects to phrases such as "the photometric term
is still winning the argument" and "a result which flatters the authors earns
one more verification step". Replace these with direct technical statements
or remove them when they add no scientific content.

Final title: **Splatt3R-SLAM: From Image Pairs to Persistent Maps**.
Central idea: local Gaussian maps and the images used to refine them share
keyframe anchors. Pose updates move both consistently; appearance refinement
can then be separated from geometric tracking.

Narrative: useful appearance maps -> fast pairwise prediction -> the
sequence-level difficulty (pose updates and overlapping predictions) ->
shared anchors -> separate refinement -> optional better initialisation ->
experiments test those claims -> measured remaining problems.

Abstract targets: RA-L about 150 words, MSc about 180 words. Keep one main
result and a short scope statement, not an inventory of every experiment.
Main text should state each limitation once in the right location. Retain
non-keyframe-supervision disclosure, separate experimental regimes, budgets,
map sizes and actual adverse results as factual experimental information.
Do not replace excessive defence with exaggerated or fabricated claims.

## Claim-to-evidence structure

1. Consistent anchor coordinates: map and supervision equations; explain
   cancellation T_f^-1 T_k = T_kf^-1 for an observation and map sharing k.
   A small coordinate-consistency identity is supportable, not a grand new
   physical theory. Scope it to that anchor; cross-anchor seams remain.
2. Refinement without a tracking update: paired on/off studies and ATE
   control; causal iteration budget and one-/two-GPU latency comparisons.
3. Initialisation matters: head-only versus released head with refinement
   controlled; no thinning / uniform / confidence-ranked attenuation,
   including all twelve recorded thinning cells and weak cases.
4. External outcome: fresh full-SH Replica and TUM tables; common protocol,
   multi-method qualitative images, map-budget and resource measurements.

The MSc should retain background, method explanation, original experiments
and appendices, rewritten in a neutral research style. Delete memoir-like
research chronology; put diagnostic evidence in organised experiments.
No arbitrary page-count padding. RA-L still targets seven pages if readable.

## Figure deliverables

Create native-shape/text/image PPTX with an overview for RA-L and a separate
MSc-numbered overview, plus detailed anchor/refinement panels and qualitative
comparison slides. Use real input frames, actual map views and actual
before/after renders. Connectors, module labels, camera glyphs, local axes,
Gaussian ellipsoids and section/equation references must be editable.
Do not flatten the overview into one screenshot inside a PPT.

Main overview follows prediction -> anchor/map assembly -> rendering/refinement,
with a geometric tracking lane and a visible supervision feedback loop.
Explain technical machinery through small concrete visuals, not a crowded
CPU/GPU memory chart. The existing TikZ process diagram can remain as the
MSc implementation figure, secondary to the method-navigation figure.

Export PPTX to PDF with LibreOffice; embed vector PDF in LaTeX. Resolve
equation/section numbers from compiled .aux files and rebuild figures until
references stabilise. Include exported PDF/SVG and original raster assets.
The generic blur and reinforcement-learning modules are absent in this system;
show the actual reservoir selection and Adam refinement loop instead.

Preserve four-column TUM (GT / Photo-SLAM / MonoGS / Ours) with common crops.
Add Replica multi-scene panels with available GT / Photo-SLAM / Ours data,
without inventing a missing monocular MonoGS reconstruction. For paper-space
constraints use a combined overview of results or MSc/supplement panels.

## Available evidence

Fresh full-SH external data are unchanged:
`logs/paper_ral_20261007/<scene>/metrics.json` and view0/view1 PNGs.
Replica 23.019214 / .134862 vs Photo 19.482997 / .162673; PSNR 8/8,
LPIPS 5/8. TUM desk Ours 13.949010 / .411030, MonoGS 13.913588 /
.395311, Photo 10.665049 / .562209. Released head, no thinning, fixed
centres, 300 s polish. Full details in `ral/REVISION_NOTES.md`.

New detailed ablation tables can use the existing ledger, not new invented data:

- `.claude/skills/splatt3r-finetuning-experiments/SKILL.md` 17.79.10
  (lines 10882+): twelve thinning LPIPS changes, confidence fade 0.45 vs off:
  Replica plain head office3 -12.7%, office0 -12.6%, office2 -10.2%,
  room0 -8.9%, room1 -7.1%, office1 -1.6%; TUM released desk -6.4%;
  EuRoC MH_01 head -5.6%; Replica office0 o-0.9 -5.3%, o-0.6 -1.8%;
  TUM adapted desk -.6%; 7-Scenes chess adapted +.16%.
- Ledger 17.92.4 (13243+): refiner off/on, base head/no thinning/300 s:
  ETH3D sofa_1 13.14/.5663 -> 21.83/.2363, delta +8.69/-58.3%;
  EuRoC V1_01 12.22/.5585 -> 13.22/.4777, +1.00/-14.5%;
  TUM desk 10.52/.5605 -> 14.05/.4072, +3.52/-27.3%;
  Replica office0 21.80/.2381 -> 26.41/.1029, +4.62/-56.8%.
  Deltas are from unrounded values, so rounded subtraction may differ .01.
- `docs/online-refinement-campaign.md` §1:
  causal/posthoc PSNR at 120/500/1000/3000 iterations:
  13.6511/13.8002, 14.3942/14.3223, 14.4747/14.3516, 14.4123/14.3747.
  These are separate controlled causal replays, not current full-SH table.
  Sampling at 500 steps, whole/early/late:
  uniform 14.40/14.09/14.72; mixed70/30 14.21/13.95/14.49;
  recent-only 14.06/13.79/14.35.
  GPU control p50: 101/103/206 ms; mean iteration125/133/223 ms;
  FPS8.0/7.5/4.5. ATE .017158 ±1e-6,306 frames/arm.
  Anchor-following block Sim3 test: 6% scale block correction, desk and room;
  fixed-camera refinement undid it around500steps, shared-anchor retained it.
  Read primary ledger if adding precise residual/overlap values to paper.
- Existing head adaptation in ledger17.82 (11057+).

Raw map paths for new genuine visuals:
`logs/ref_onoff_tum_on/rgbd_dataset_freiburg1_desk_{gaussians,refined}.ply`,
the accompanying trajectory and frames.txt; `logs/cmp_replica_office0/`.
`scripts/paper/render_comparisons.py` supplies dataset loading, full-SH
rendering, camera alignment and exact target-frame selection. Both A6000
GPUs were idle at initial check. Prefer a few renders of existing maps.

## Writing references consulted

- Nature 641,1180–1187(2025), "A foundation model for the Earth system",
  DOI10.1038/s41586-025-09005-y. Read full text on nature.com: concrete
  motivation, scoped gap, three-part method overview, claim-led result sections,
  detailed technical settings later. Use the exposition pattern, not its prose.
  https://www.nature.com/articles/s41586-025-09005-y
- Science Robotics10(109),2025, "Resilient odometry via hierarchical adaptation",
  DOI10.1126/scirobotics.adv1818. Abstract identifies environmental problem,
  hierarchical response, validation scale; technical detail follows.
  https://www.science.org/doi/10.1126/scirobotics.adv1818
- Science Robotics11(114),2026, "Fusing LiDAR and vision to generate
  high-quality reconstructions", DOI10.1126/scirobotics.aej0223, is an
  Editors' Choice summary, not a research baseline. Useful contrast of
  capabilities and motivating complementary mechanisms.

Do not add these unrelated writing examples to the SLAM bibliography merely
to cite prestigious journals. Keep them in the editorial report.

## Tools / work state

Revision completed on 2026-10-07. The RA-L abstracts and full shared prose
were rewritten around the predict--anchor--refine narrative. The MSc abstract,
introduction, technical transitions, negative-results chapter, discussion, and
conclusion were rewritten, while equations and evidence were retained.
Editable PowerPoint figures, separate RA-L/MSc overview exports, real
before/after map renders, multi-method qualitative panels, and fine-grained
ablation views were added. Verified state: MSc 65 pages, RA-L 7+7 pages.
Use `/home/share-v5/miniconda3/envs/splatt3r-slam/bin/python` for numpy,
matplotlib,Pillow,torch,lxml. python-pptx and xlsxwriter are not installed;
install in an isolated target directory and document figure dependencies.
LibreOffice, Poppler and TeX Live2026 are available. No specialized marketplace
skill was found. Main agent read thesis-writing skill fully this turn.
The whole previous manuscript was read in the preceding task; re-open each
chapter before rewriting now. Existing workspace was already dirty; preserve
unrelated changes, do not commit or contact anyone. Full rewrite is authorised.
