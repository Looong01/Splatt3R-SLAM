# Figure revision: evidence and scope

Updated 2026-10-12. The RA-L now contains one combined method/system overview.
The old GPU/CPU diagram is retained as editable source, not included as a second
overview. This revision changes presentation and illustrative view selection;
it does not retrain or edit any evaluated Gaussian map.

## TUM: what the number means

The comparable full-SH rendering evaluation covers **freiburg1_desk**, not all
TUM sequences. `logs/paper_ral_20261007/desk/metrics.json` contains the 100 common
non-keyframes and unrounded per-frame values.

| Aggregation | Ours minus Photo-SLAM |
|---|---:|
| Scene PSNR: -10 log10(mean MSE) per method, then difference | +3.28396 dB |
| Arithmetic mean of the 100 paired per-frame PSNR differences | +3.53285 dB |
| Median of those differences | +3.62230 dB |

Scene PSNR is 13.94901 vs 10.66505 dB. The teaser explicitly labels the paired
frame mean as +3.53 dB. Neither number supports a multi-sequence TUM average.
The paper's quantitative tables retain the original aggregation/resolution.

## What is distinctive, and what is shared with prior work

The Splat-SLAM paper explicitly describes map deformation after global pose and
depth updates, a proxy depth/DSPO path, and Gaussian photometric optimisation.
Its official repository also documents separate tracking and mapping processes.
Consequently, map anchoring/deformation, Gaussian optimisation, or separate
processes alone are not substantiated novelty claims for our work.

| Design axis | Splat-SLAM, original paper/repository | This project, implementation |
|---|---|---|
| Initial map | Proxy depth from multiview and monocular estimates | Frozen Splatt3R pairwise Gaussian prediction sharing features with the geometric path |
| Map update | Deform the Gaussian map after keyframe pose and depth updates | Retain keyframe-local means/covariances; compose with the current Sim(3) anchor |
| Supervision detail | Optimise a deformable map with re-rendering losses | Store tracked RGB with `(anchor_idx, T_anchor_frame)`; reconstruct supervision cameras at sample time |
| Per-step consistency | No absence claim is made about its implementation | Clone `kf_data` once, pass it to `SupervisionFrames.sample`, and use the same data for `LocalGaussianMap.world` |
| Optimisation boundary | Tracking/mapping separation already exists | Geometry/backbone are outside the photometric graph; evaluated map centres remain fixed while colour, opacity, scales and rotations change |

The central figure highlights the **coupled map-and-observation parameterisation**
and the exact pose-snapshot rule. It does not establish that every other GS-SLAM
system lacks an equivalent construction. A simple coordinate cancellation alone
is not evidence of an entirely new class of SLAM method. Existing pose-correction
controls support this project's design choice, not a measured superiority over
Splat-SLAM, which has not been rerun under the same rendering protocol.

Code evidence:

- `splatt3r_slam/refiner.py`: `LocalGaussianMap.world` (local means/covariances).
- `SupervisionFrames.offer/sample`: saved relative cameras, composed at sampling.
- `run_refiner`: one cloned `kf_data` for map and camera, inside every refinement step.
- `docs/Thesis/ral/method.tex`: equations for map placement, supervision and invariant.

Primary references (read 2026-10-12):

1. Splat-SLAM original paper, §§3.1–3.2: https://arxiv.org/html/2405.16544
2. Official implementation README: https://github.com/google-research/Splat-SLAM

These are primary descriptions of that system, not independent evidence of
field-wide uniqueness for our design.

## Image selection and limits

- TUM desk candidates rank by **ours absolute PSNR**, with at least 80 source
  frames between candidates. Baseline margins are not used. Frames 24 and 377
  appear in the teaser; frame 24 appears in the detailed comparison.
- All four methods share cameras, crop coordinates and full exported SH.
  Display renders use 640x480 with intrinsics scaled from the evaluation camera;
  the quantitative results remain at 512x384.
- Additional fr1/360 and ETH3D table views use the highest PSNR from their existing
  12 saved candidates. These are separate recorded real-data runs; they do not
  add Photo-SLAM comparisons or a dataset average.
- Selected examples are not median/typical performance. Captions now disclose this.
- The persistent map is a real, pulled-back CUDA render of the same desk PLY.
  Camera transforms are in `ral/fig/publication_assets/orbit_cameras.json`.
- No generative enhancement, sharpening or replacement of reconstructed detail
  was applied. Higher display resolution does not repair geometric error.
  The clearest selected desk view has 17.66 dB at the original evaluation size;
  visible blur remains. More publication polish cannot establish better mapping
  quality without a separately evaluated map/algorithm change.

Reproduction and provenance: `scripts/paper/select_publication_views.py` and
`ral/fig/publication_assets/selection.json`. Current figure generators:
`draw_revised_figures.py` and `draw_contribution_overview.py`.

## Reference UI and region-2 display order

The overview now uses the supplied reference's filmstrip, blue tracking/red
mapping groups, rounded frame cards, loss parallelogram, renderer bar and
cylindrical stores. It retains this project's shared-anchor semantics; no
blur detector or deblurring module from the reference is implied.

The Photo-SLAM and Ours region-2 crop tiles exchange display positions only.
Every inset has an explicit source-method label in a separate detail row.
Main-view columns and all measurements retain their original method identity.
`ral/fig/publication_assets/crop_attribution.json` records each tile's source
and crop hash so placement cannot silently change attribution.

Export QA: the PPTX layout checker retains 9 heuristic flags for grouped
endpoints and polyline bounding boxes. The actual exported slide-2 and slide-3
PDF previews were reviewed after fixing connector rerouting. Native PDF word
bounds show no text collisions, and all eight crop hashes match their explicit
method labels. See `FIGURE_LAYOUT_CHECK.md` and
`ral/fig/reference_ui_validation.json`; automated PPTX validation is not reported
as a zero-warning pass.
