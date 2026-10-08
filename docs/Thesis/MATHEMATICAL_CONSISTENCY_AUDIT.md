# Mathematical consistency audit

Date: 2026-10-07

## Scope and convention

The authored and anonymous RA-L PDFs share one mathematical source. Both were
checked with the MSc thesis against the tracker, backend, refiner, rendering
code, training loss, upstream papers, and recorded metrics.

The revised notation uses `T_AB` for a transform from frame `B` to frame `A`.
`C_k` is keyframe `k`'s camera frame and `W` is the world frame. Thus
`T_WCk` is camera-to-world and
`T_CkCf = T_WCk^{-1} T_WCf`. `X_Ca^(b)` is image `b`'s pointmap expressed
in frame `C_a`. Sim(3) scale is `a_k`, Gaussian axis scale is `s_n`,
pointmap confidence is `gamma_n`, and rendered colour is `c_n(u)`.

## Corrections

1. Corrected the pointmap-fusion indices. The keyframe observation predicted
   in current-frame coordinates is `X_Cf^(k)`, matching `tracker.py:97-100`
   and MASt3R-SLAM Eq. (8).
2. Replaced mixed pose forms with the single `T_AB` convention.
3. Verified `T_WCf^{-1} T_WCk = T_CkCf^{-1}` for shared anchors.
4. Corrected 2D covariance projection to use the camera rotation and a
   2-by-3 projection Jacobian.
5. Distinguished opacity logits from activated opacity and Gaussian scale
   from Sim(3) scale.
6. Separated colour, pointmap-confidence, and match-confidence symbols.
7. Defined rank normalisation as `r_n / max(N-1,1)`, including `N=1`.
8. Distinguished VGG-LPIPS used for head training from AlexNet-LPIPS used
   for reconstruction evaluation.
9. Defined the map-to-GT similarity before writing its inverse camera mapping.
10. Corrected ATE bolding using unrounded values.
11. Restricted the unchanged-ATE statement to its TUM desk control and
    identified the 33-window sample used by the Wilcoxon test.
12. Removed an FPS value from a wall-clock row because the timing boundaries
    differ.

## Numerical checks

```text
anchor_identity_max_abs       7.772e-16
covariance_symmetry_max_abs   1.110e-16
covariance_min_eigenvalue     8.919531e-03
alignment_rotation_max_abs    1.110e-16
alignment_translation_max_abs 2.220e-16
rank normalization N=1,2,100  PASS
```

All stored scene aggregates were recomputed from their per-frame records with
zero discrepancy. Replica remains 23.019214 dB / 0.134862 LPIPS for Ours and
19.482997 dB / 0.162673 for Photo-SLAM, with 8/8 PSNR and 5/8 LPIPS wins.

No unresolved algebraic contradiction remains. Differences between the RA-L
and MSc manuscripts are expository depth, not mathematical semantics.
