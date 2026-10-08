# Splatt3R-SLAM thesis defense speaker script

Target length: 20-25 minutes, plus appendix for questions.

1. **Title.** This thesis asks how a fast image-pair reconstruction can become a map that lasts for a whole sequence.
2. **A pair is not a map.** A pair gives a useful local result. Repeating it creates many local maps, not one consistent scene.
3. **Persistent map requirements.** The map must support revisiting, pose correction, and rendering from a new camera.
4. **Why accumulation fails.** Pairwise models do not see coordinate changes, overlap, or later loop closures.
5. **The thesis.** Store the local map and the cameras that supervise it under the same keyframe anchor.
6. **Contribution chain.** Prediction supplies the prior, anchoring preserves coordinates, and refinement uses sequence evidence.
7. **Positioning.** Existing systems are mainly optimization-first or pairwise prediction. This work connects prediction to a full pose graph.
8. **Method navigation.** Use this slide as the map for the next four method slides.
9. **Predict.** One frozen shared network produces geometry for tracking and Gaussian attributes for mapping.
10. **Anchor.** The algebra says the shared anchor cancels from the relative transform. The controlled pose test checks this behavior.
11. **Refine.** The sampler chooses real tracked images. Rendering and Adam update appearance, not poses.
12. **Initialization.** Head adaptation changes prediction. Opacity attenuation changes insertion. They are evaluated separately.
13. **Evidence map.** Every contribution has a direct control and a measured observation.
14. **Protocol.** Common cameras and one renderer prevent each method from choosing an easier evaluation path.
15. **SOTA result.** Splatt3R-SLAM achieves state-of-the-art performance: 23.02 dB on Replica, 3.54 dB above Photo-SLAM, with wins on all eight scenes.
16. **Replica qualitative.** The advantage appears across different rooms, not only one selected image.
17. **TUM qualitative.** MonoGS and our method are close. The crops show where both still blur or distort detail.
18. **Tracking.** Tracking remains MASt3R-SLAM. This is a control, not a claimed tracking contribution.
19. **Refinement.** The on/off test gives the largest controlled gain on every tested family.
20. **Component evidence.** Opacity reduction helps in ten of twelve cells. Confidence allocation is not the main effect.
21. **Online budget.** Most desk improvement appears by about 500 updates. A second GPU protects tracking latency.
22. **Cost.** Dense prediction buys quality with millions of primitives and high memory. Compactness is the next systems problem.
23. **Negative results.** Several plausible fixes failed. Shared anchoring and downstream refinement remained supported.
24. **SOTA claim.** Splatt3R-SLAM achieves state-of-the-art performance. The evidence is 8/8 Replica PSNR wins and a 3.54 dB mean gain under one evaluator.
25. **Reproduction.** Every table row comes from saved maps, trajectories, common rendering, per-frame metrics, and hashes.
26. **Conclusion.** Shared anchors are the single idea that turns pairwise prediction into a persistent map.
27-29. **Appendix.** Use the implementation, SOTA evidence ledger, and setting table during questions.
