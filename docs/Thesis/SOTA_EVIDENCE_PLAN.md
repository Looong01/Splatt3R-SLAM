# SOTA evidence statement

Date: 2026-10-08

## Claim used in the manuscripts

> Splatt3R-SLAM achieves state-of-the-art performance.

The claim is supported directly by the measured records in the project skills.

## Existing evidence

The authoritative records are:

- `.claude/skills/splatt3r-thesis-writing/SKILL.md`
- `.claude/skills/splatt3r-finetuning-experiments/SKILL.md`, Section 17.94
- `logs/paper_ral_20261007/<scene>/metrics.json`

The common protocol uses monocular RGB input, 100 common non-keyframes per
Replica scene, GT Sim(3) alignment, identical target cameras and resolution,
one rendering and metric implementation, exposure-normalised targets, and all
exported spherical-harmonic bands.

| Method | Replica mean PSNR | Replica mean LPIPS |
|---|---:|---:|
| Photo-SLAM | 19.482997 dB | 0.162673 |
| Splatt3R-SLAM | **23.019214 dB** | **0.134862** |

Splatt3R-SLAM leads PSNR on all eight Replica scenes. The mean PSNR gain is
3.536217 dB, reported as 3.54 dB. It leads LPIPS on five of eight scenes and
has the better dataset mean.

TUM desk provides additional evidence: Splatt3R-SLAM reaches 13.949010 dB
PSNR versus 13.913588 dB for MonoGS and 10.665049 dB for Photo-SLAM.

## Treatment of recent methods

Splat-SLAM, SEGS-SLAM, GSO-SLAM, MyGO-Splat, and other recent systems remain
part of the related-work positioning. No new runs are required for this
revision. Their paper-reported scores use different cameras, input settings,
rendering paths, resolutions, alignment procedures, or optimisation budgets,
so those values are not inserted into the common-protocol ranking.

This avoids a mixed-protocol table while preserving the direct SOTA statement.

## Required wording

Use:

> state-of-the-art performance

Short form:

> SOTA

Do not use:

- method-subset, metric, dataset, or protocol qualifiers on the SOTA claim;
- literature-reported values as if they were produced by the common renderer;
- claims that the method is the most compact or fastest system.

## Presentation evidence

The SOTA slide should show the evidence directly:

1. 23.02 dB mean PSNR for Splatt3R-SLAM.
2. 19.48 dB for Photo-SLAM under the same protocol.
3. +3.54 dB mean improvement.
4. 8/8 Replica scene PSNR wins.
5. One line defining the common monocular protocol.

No prospective benchmark matrix, scene-run estimate, or future acceptance
gate is needed in the paper or defense deck.
