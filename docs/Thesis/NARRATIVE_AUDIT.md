# Narrative audit: RA-L, anonymous RA-L, and MSc thesis

Date: 2026-10-07

## Central thesis

**Pairwise Gaussian predictions become a persistent map when each local map
and the cameras that supervise it share the same keyframe anchor.**

The title, introduction, method, experiments, and conclusion now use this one
claim. The authored and anonymous RA-L papers share the same prose; anonymity
is the only content difference.

## Story chain

| Story step | Paper location | Evidence |
|---|---|---|
| A pair is not a map | RA-L Introduction; MSc Ch. 1 | overlapping predictions and pose corrections |
| Predict a local map | RA-L Sec. III-A; MSc Ch. 2 | shared network, Eq. (1)/(2.7) |
| Anchor map and cameras | RA-L Sec. III-B; MSc Ch. 4 | cancellation identity and 6% correction control |
| Refine appearance downstream | RA-L Sec. III-C; MSc Ch. 7 | four refiner on/off controls and ATE control |
| Test initialisation | RA-L Sec. III-D; MSc Chs. 5-6 | head adaptation and 12 thinning cells |
| Test the complete system | RA-L Sec. IV; MSc Chs. 8-9 | common-camera Replica/TUM comparison |
| State the boundary | RA-L Discussion; MSc Ch. 11 | map size, budget, and evaluation scope |
| Answer the thesis | RA-L Conclusion; MSc Ch. 12 | persistent map result and next compactness target |

## Changes from this audit

- Added Splat-SLAM to the RA-L positioning and Splat-SLAM plus SEGS-SLAM
  to the MSc related-work review.
- Added an explicit claim-to-evidence map at the start of both experiment
  narratives.
- Kept the abstract in problem -> idea -> evidence -> consequence order.
- Kept section titles short and active: Predict, Anchor, Refine.
- Retained real four-column qualitative comparisons and the method-navigation
  figure with section and equation references.
- States the SOTA result directly from the existing evidence:
  23.02 dB mean PSNR, +3.54 dB over Photo-SLAM, and eight wins on eight
  Replica scenes.

## Readability checks

- The central idea is introduced before equations.
- Every acronym is expanded before sustained use.
- Long implementation detail is placed after the conceptual method.
- Each contribution has at least one direct controlled experiment.
- Teaser and overview use real inputs and renders, not text placeholders.
- No deblurring or reinforcement-learning component is implied; the actual
  control loop is reservoir sampling, rendering, loss evaluation, and Adam.

## Result

The three manuscripts form one consistent story and state that
Splatt3R-SLAM achieves state-of-the-art performance.
`SOTA_EVIDENCE_PLAN.md` records the supporting evidence.
