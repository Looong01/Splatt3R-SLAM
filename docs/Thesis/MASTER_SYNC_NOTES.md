# MSc content synchronisation and removal of the CVPR draft

User request: remove CVPR-specific material; bring the MSc thesis up to date
with the RA-L scientific content while keeping the two document formats
independent. This work is authorised, including deletion of the old draft.

## Document boundary

- RA-L keeps `ral.tex`, `ral-review.tex`, IEEE classes and `ral/` prose.
- MSc keeps `master.tex`, `report` A4/12pt/1.5 spacing, `chap/`, thesis title
  page, chapter numbering, front matter and appendices.
- Reuse scientific figure sources and generated tabular data, with separate
  captions/float environments. Do not input RA-L preamble or document into MSc.
- Switch MSc bibliography to distribution-provided `unsrtnat`, removing its
  dependency on the CVPR author-kit `ieeenat_fullname.bst`.
- Preserve the supplied MSc programme, student number and supervisors;
  second-reader and personalised acknowledgements remain author placeholders.

## Content synchronised

- New method/performance teaser in chapter 1; method navigation and
  process-level System Overview in chapter 4; four-column TUM
  comparison, multi-scene Replica comparison, and scene/map-budget chart in
  results.
- Fresh all-exported-SH results: Replica 23.019214 vs 19.482997 dB,
  LPIPS 0.134862 vs 0.162673, numerical wins 8/8 and 5/8. TUM desk:
  Ours 13.949010/0.411030; MonoGS 13.913588/0.395311;
  Photo-SLAM 10.665049/0.562209.
- External comparison = released head, thinning off, fixed centres, 300 s
  polish. Head adaptation/thinning and causal delivery are separate ablations.
- Non-keyframes may supervise refinement: no strict unseen-view claim.
  Replica resolution 512x288, TUM 512x384. Correct MonoGS frame association.
- Retain the measured pose-convention/pose-source diagnostics as historical
  diagnostics; remove the claim of a universal 8.7 dB protocol offset.
- Correct opacity: input-RGB SH residual + clamping, not sigmoid colour.
  Alpha changes affect foreground/background visibility and need not be
  monotone in luminance. Five tested heads are not five dataset families.
- Remove the retracted 0.66-pixel/0.057-m blindness explanation and universal
  impossibility claims about geometric corrections. Retain measured failures.
- Correct chapter 4 storage description: SharedKeyframes is CUDA IPC; pools
  and snapshots are CPU shared memory. GUI-on measurements do exist.
- Correct blanket frozen-centre, real-time, parameter-count and compactness
  claims. Naive post-hoc truncation is not an optimum small-map bound.
  CLI defaults currently enable centre freezing and aa-sigma 0.5; this is
  recorded separately from the causal study that did not support treating
  those choices as a universal fix. No runtime code was changed.
- Add Unblur-SLAM related work and current reproducibility commands.

## Cleanup completed

Removed the old conference driver/PDF/style, bibliography style, `sec/`,
unused `fig/` montages and build artefacts. The staged `main.tex` to
`cvpr.tex` rename now records removal of the old driver, so a later commit
cannot restore it. Unrelated `.vscode` changes were preserved.

Updated build.sh, both READMEs, the ignore-file comment and thesis-writing
guidance. Legitimate publication venue names remain in bibliography entries.
`package_master.py` replaces `docs/Thesis.zip` with an MSc-only package;
`package_ral.py` builds the separate RA-L package. Shared scientific assets
retain their `ral/fig` and `ral/generated` paths without importing IEEE
classes or RA-L prose into the MSc archive.

## Verification

All three PDFs compile without missing references/citations or overfull
boxes. MSc is 65 A4 pages with no LaTeX warnings or underfull boxes.
Both RA-L PDFs remain seven Letter pages; minor underfull column notices
remain. All fonts are embedded without Type 3.
The MSc retains 12 chapters and four appendices. Its figures include the
teaser, method-navigation overview, scene/budget evidence, Replica
multi-scene panel, and four-column TUM comparison. The extended per-frame
table is generated directly from the same fresh metrics JSON as the RA-L
tables.

Numeric checks recomputed PSNR from mean frame MSE and LPIPS from per-frame
values for all nine sequences, checked 100 unique frame indices per scene,
recomputed the Replica summary and 8/8 PSNR, 5/8 LPIPS wins, and compared
ATE values and bolding between the two documents. Build recorder files
confirm MSc imports neither IEEE classes nor RA-L prose, and RA-L imports
no MSc chapters. Whole-document contact sheets and figure pages were
inspected; short chapter-tail pages were removed by tightening repeated prose.

Both source archives passed integrity and content-boundary checks. Extracted
copies rebuilt all three PDFs with exactly matching extracted text and page
counts, confirming that neither depends on the other document's prose or
formatting. The ZIPs store the executable bit of build.sh; when extraction
software drops file permissions, `bash build.sh master` (or `ral` / `review`)
is equivalent. Personalised acknowledgements and second reader remain the
only visible author placeholders.
