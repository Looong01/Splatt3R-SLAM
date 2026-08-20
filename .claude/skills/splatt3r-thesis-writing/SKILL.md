---
name: splatt3r-thesis-writing
description: The CVPR-format manuscript for Splatt3R-SLAM living in docs/Thesis — its current state, exact build/verify commands, where every number and equation in it comes from, the editorial decisions behind its structure and its deliberately unflattering passages, and the open items. Read this BEFORE editing anything under docs/Thesis, before adding a figure or citation, or before changing any reported number; and read splatt3r-finetuning-experiments §17.94 (the six consolidated result tables) plus docs/external-baselines.md alongside it, since this skill deliberately does not duplicate the raw data.
metadata:
  type: reference
---

# Splatt3R-SLAM manuscript (docs/Thesis) — state, provenance, and reasoning

Written 2026-08-20. This file exists so a compacted context can resume paper
work without re-deriving anything. It covers **the writing**, not the
research; every number's raw provenance is in
`.claude/skills/splatt3r-finetuning-experiments/SKILL.md` §17.93–17.95 and in
`docs/external-baselines.md`.

---

## 1. Current state — VERIFIED BUILDING

```
docs/Thesis/
├── main.tex              13 pp; author = Zelong Li, loong.li2@student.uva.nl, +31 6 3023 2998
├── main.bib              24 entries, ALL 24 cited
├── cvpr.sty              CVPR 2026 official author-kit, UNMODIFIED
├── ieeenat_fullname.bst  official, UNMODIFIED
├── build.sh              one-shot build; `./build.sh clean` to reset
├── README.md             build notes, figure inventory, equation provenance
├── fig/
│   ├── teaser.tex        2-column strip figure (cuted `strip` env)
│   └── *.png             7 figures
└── sec/                  8 section files, ~1140 lines total with main.tex
```

**Last verified build:** 13 pages, **0 Overfull, 0 Underfull, 0 undefined
references/citations**, 10 numbered equations, 9 tables, 6 `\includegraphics`
(7 image files; `cmp_replica_room0_f0.png` is the teaser).

### Build and verify (copy-paste)
```bash
cd /home/share-v5/Codes/Splatt3R-SLAM/docs/Thesis
export PATH=/usr/local/texlive/2026/bin/x86_64-linux:$PATH
./build.sh                       # or ./build.sh clean first for a from-scratch check
grep -oE "Output written on main.pdf \([0-9]+ pages" main.log | tail -1
echo "Overfull: $(grep -cE 'Overfull \\hbox' main.log)"
echo "undefined: $(grep -ciE 'undefined (reference|citation)' main.log)"
tr '\n' ' ' < main.bbl | grep -oE '\\bibitem\[[^]]*\]\{[^}]+\}' | wc -l   # want 24
```
`build.sh` prepends the TeX Live 2026 path itself, so it also works without
the manual `export`. **Always re-run the four checks after any edit** — the
zero-warning state has been lost and recovered twice.

### Visual check
```bash
pdftoppm -png -r 55 -f 1 -l 1 main.pdf /tmp/pg   # then Read /tmp/pg-01.png
```

---

## 2. Fonts — the one thing not to "improve"

`cvpr.sty` line 30 does `\RequirePackage{times}` → URW Nimbus Roman, a Type 1
font **bundled with both TeX Live and MiKTeX**. Confirmed in `main.log`:
`utmr8a.pfb`, `utmb8a.pfb`, `ucrr8a.pfb`, plus Computer Modern math. No system
font is referenced anywhere.

**Therefore pdflatex + `times` IS the cross-platform requirement, already
satisfied.** Do NOT "upgrade" to XeLaTeX/LuaLaTeX + fontspec: that resolves
fonts from the OS (Arial, SimSun, …), which breaks the Linux/Windows parity
the user asked for. A warning to this effect is in `main.tex`'s header comment
and in `docs/Thesis/README.md`; keep both.

---

## 3. Source material and how it was gathered

| Input | Path | Used for |
|---|---|---|
| Splatt3R paper | `docs/third_party/2408.13912.pdf` | Eqs. 1–4, head architecture, loss masking |
| MASt3R-SLAM paper | `docs/third_party/2412.12392.pdf` | Eqs. 5–6, ray error, pose graph |
| Refiner implementation | `splatt3r_slam/refiner.py` (1248 lines) | Eqs. 8–9, LRs, sampling, duty cycle |
| Consolidated tables | finetuning skill §17.94 | every table in the paper |
| Baseline campaign | `docs/external-baselines.md` | protocol, offset, all comparisons |

PDF extraction used `pdftotext -layout` into `/tmp/*.txt`, then `grep -n` on
section headers — far cheaper than reading PDF pages, and it preserves the
equation context well enough to transcribe from.

CVPR template obtained by `git clone --depth 1 https://github.com/cvpr-org/author-kit`
into `/tmp/author-kit`; `cvpr.sty` and `ieeenat_fullname.bst` copied verbatim.
The kit's `cvpr.sty` already loads `cleveref` (line 496), so `\cref`/`\Cref`
work without adding the package.

---

## 4. Equation provenance (keep this table in sync with the paper)

| Eq. | Content | Source | Note |
|---|---|---|---|
| 1 | $\mu = x + \Delta$ | Splatt3R §3.3 | |
| 2 | $\Sigma = R(q)\mathrm{diag}(s)^2R(q)^\top$ | 3DGS | |
| 3 | $\alpha$-compositing | 3DGS | **load-bearing for the whole mechanism argument** |
| 4 | masked MSE + LPIPS loss | Splatt3R Eq. 3 | $\lambda_{LP}=0.25$ |
| 5 | $\mathrm{Sim}(3)$ pose + left-plus | MASt3R-SLAM Eq. 1 | |
| 6 | confidence-weighted pointmap fusion | MASt3R-SLAM Eq. 8 | |
| 7 | anchor→world composition | **ours** | why loop closures are free |
| 8 | **injection-time thinning** | **ours** | `refiner.py:1091-1094` |
| 9 | refiner objective, L1+DSSIM | **ours** | `refiner.py:1181`, `DSSIM_WEIGHT=0.2` |
| 10 | Umeyama alignment, inverted | protocol | `scripts/eval_map_quality.py` |

**Eq. 3 is the pivot of the paper's central argument.** Against a black
background the residual transmittance $\prod_l(1-\alpha_l)$ contributes zero
radiance, so lowering $\alpha$ darkens a region and raising it brightens it.
That is the mathematical basis for "opacity is a brightness-trim dial, not a
confidence gate". Before this section existed the claim was prose-only; do not
weaken it back.

### A correction that must not be undone
The first draft wrote the thinning lever as
$\alpha' = \alpha(1-\lambda(1-\hat c))$. **This did not match the code.**
`refiner.py:1091-1094` rank-normalises confidence *within each keyframe*
(the confidence head is not calibrated across frames) and applies
$$\alpha' = \alpha \cdot \mathrm{clip}\bigl(1 - 2\lambda(1-\hat c),\,0.1,\,1\bigr).$$
The factor $2$ and the floor at $0.1$ are both real and both load-bearing —
the floor keeps it a *thinning* prior rather than a *deletion* prior. The
paper now states the true form. **Rule: an equation in the paper must be
checkable against the implementation; if they disagree, the paper is wrong.**

---

## 5. Structure, and why it is this way

```
0_abstract      three results + the two comparison findings, incl. the bad news
1_intro         same arc, with the four "paragraph" headings
2_related       feed-forward recon / GS-SLAM / priors-in-SLAM / PEFT / eval practice
3_method        §Preliminaries (Eqs. 1–6) THEN §Method (Eqs. 7–9)
4_protocol      protocol, self-consistency rule, THE OFFSET
5_experiments   head-only, thinning, mechanism, Replica, TUM, ATE, refiner, cost
6_limitations   compactness curve, scope, unresolved threads
7_conclusion    results + the two transferable methodological points
```

**Deliberate choices:**

- **Preliminaries and Method are one file** (`sec/3_method.tex`) with two
  `\section`s. Splitting them was considered and rejected: the method's
  argument is a direct continuation of Eq. 3, and separating them across files
  made the equation cross-references harder to keep straight.
- **The protocol section precedes the experiments.** The ~8.7 dB offset is
  arguably the most transferable contribution; a reader must have it before
  seeing any number, otherwise they will mentally compare our 23 dB to a
  published 30.9 dB.
- **Limitations is a numbered section, not a paragraph.** The compactness
  deficit is the single most attackable result and is presented with its own
  table (`tab:curve`), not as a caveat sentence.

---

## 6. Passages that are deliberately unflattering — do not "polish" these

Every one of these was written after checking a favourable-looking result one
level deeper and finding it weaker. They are load-bearing for the paper's
credibility.

1. **Abstract** states the three-scene subset would have overstated the Replica
   margin by 1.85 dB, and that we win LPIPS on only 5/8.
2. **§5.4** reports the per-frame distribution: office2's scene mean says
   $+0.73$ dB while the per-frame **median is $-1.84$ dB**. Our nominal win
   there comes from a minority of frames.
3. **§5.5 (TUM)** states that at matched wall clock (our 300 s vs MonoGS's
   339 s) PSNR ties and **MonoGS wins LPIPS**; only at 3.5× budget do we lead
   both. The honest sentence — "we can buy past it with more compute" — is in
   the text; the earlier draft's "the deficit was a budget artefact" was
   **wrong** and was removed after measuring that MonoGS's 26 000 iterations
   take only 339 s.
4. **§5.6 (ATE)** states explicitly that parity with MASt3R-SLAM is *expected
   and not a contribution*, because tracking is inherited unchanged.
5. **§5.1** reports four falsified attributions as a negative result
   (`tab:falsified`) rather than adopting the most plausible survivor.
6. **§6.1** shows pruning our map to a baseline's budget costs 8.7–10.3 dB,
   killing the "it's just redundancy" defence, and states the lower-bound
   caveat in both directions without using it as an excuse.
7. **§7** records that three results which initially favoured us changed when
   checked deeper, and proposes the rule that a flattering result earns one
   extra verification step.

---

## 7. Figures

| File | Role | Placement |
|---|---|---|
| `cmp_replica_room0_f0.png` | teaser, 3-way one-protocol | p.1 strip |
| `cmp_replica_office0_f1.png` | 2nd baseline comparison | `figure*`, §5.4 |
| `eth3d_sofa1_f1.png` | head-only ablation, ETH3D | `figure*` row 1, §5.1 |
| `euroc_v101_f0.png` | head-only ablation, EuRoC | row 2 |
| `tum_rpy_f0.png` | head-only ablation, TUM | row 3 |
| `7scenes_office_f0.png` | head-only ablation, **weak family** | row 4 |
| `replica_ps_office2_f1.png` | photospatial head | single-col, §Method |

Sources live in
`.claude/skills/splatt3r-finetuning-experiments/figures/` (16 PNGs); these 7
were selected by the user. The 7-Scenes row is included **on purpose** so the
failure mode is visible rather than only tabulated — the caption says so.

**Frame-selection rule (do not break it):** comparison frames are chosen at
the **median of the per-frame $\Delta$PSNR**, not by eye. The first version of
the teaser used evenly-spaced frames and landed on a frame where Photo-SLAM
was 4.8 dB *ahead*, under a caption claiming we led by 7.53 dB — a figure
disproving its own label. Generator: `/tmp/make_baseline_figs.py` (scratch;
re-create from the skill if lost).

---

## 8. Open items

1. **Institution line is empty.** The email implies University of Amsterdam
   but affiliation is the author's to confirm; insert one line under the name
   in `main.tex` if wanted.
2. **Phone number is non-standard for CVPR.** Added because the user asked.
   Suggest removing before an actual CVPR submission; harmless for a preprint,
   and invisible under blind review anyway.
3. **Submission mode** is `\usepackage[final]{cvpr}` (camera-ready/preprint)
   per the user's instruction. Blind review = plain `\usepackage{cvpr}`, which
   adds line numbers and anonymises. Both alternatives are commented in
   `main.tex`.
4. **9 unused figures** remain in the skill's `figures/` directory.
5. **No supplementary material** file yet; the author kit ships a
   `sec/X_suppl.tex` pattern if one is wanted.

---

## 9. Editing recipes

```bash
# add a citation: append to main.bib, cite in sec/*.tex, rebuild, then verify
tr '\n' ' ' < main.bbl | grep -oE '\\bibitem\[[^]]*\]\{[^}]+\}' | sed 's/.*{//;s/}//' | sort > /tmp/cited
grep -oE "^@[a-z]+\{[^,]+," main.bib | sed 's/^@[a-z]*{//;s/,$//' | sort > /tmp/inbib
comm -13 /tmp/cited /tmp/inbib      # entries present but never cited
```
An earlier pass found the five dataset papers (TUM, Replica, EuRoC, ETH3D,
7-Scenes) sitting uncited while the text used all five datasets — a genuine
omission, now fixed by a `\paragraph{Datasets.}` in §4. Run this check after
every citation edit.

**Overfull boxes** have come from wide tables twice. Both fixes were the same:
shorten the column contents and move citations out of table cells into the
caption (`tab:systems`), or shorten row labels (`tab:refiner`). Locate with:
```bash
grep -A3 "Overfull \\\\hbox" main.log | head
```
then map the reported line numbers to the section file that follows in the log.


## 11. 2026-08-20 addition: §5 "Front-end geometry is not the lever"

Added in answer to the user's question "does VGGT-SLAM only do the front end,
and can/should we combine our improvements with its front-end gain?". The
answer was already fully measured in `splatt3r-finetuning-experiments`
§17.45/17.47/17.48/17.49 — and **none of it was in the manuscript**. That is
the transferable lesson: this project's negative results are its cheapest
paper material, and the failure mode is not losing them but forgetting to move
them from the experiment log onto the page.

How the gap was found, and the check to repeat before calling the paper done:
```bash
cd docs/Thesis && grep -rn -i "vggt|referee|jitter" sec/ main.bib
```
Before declaring the manuscript complete, run that grep for every major
negative-result thread in the experiment skill and confirm each is either on
the page or deliberately excluded.

**New content** (`sec/5_experiments.tex`, inserted before
`\subsection{Refinement ablations}`, plus a closing paragraph in
`sec/6_limitations.tex` "Unresolved threads"):

- `tab:scale` — the 2×2 that carries the argument: Splatt3R pairwise 5.35%,
  VGGT pairwise 10.82%, VGGT joint-16 1.68%, plus the disjoint-window
  replication (4.92% → 1.53%, 7/7, p=8.6e-8). **Keep both VGGT rows.** The
  pairwise row is what makes the claim "context, not capacity" — with only the
  joint row the table reads as "VGGT is better", which is false and is exactly
  the misreading a backbone-swap reviewer would make.
- Four paragraphs, in the order the experiments were actually run: coupling
  works → averaging cannot substitute → the referee is right and the picture
  still gets worse → the same conflict reappears inside the refiner.
- The mechanism paragraph is the load-bearing one: the scale error is
  common-mode with the pose, photometry is blind to common-mode depth error at
  these baselines (4.5% = 0.66 px at 0.057 m) but not to two clusters
  disagreeing about a surface, so correcting the map alone converts an absorbed
  error into a differential one. Photometric fit asks 0.8% where the sensor
  says 6.5%, Spearman +0.06.

**Do not soften this subsection.** It reports our own idea failing four times
and it is what lets §5 claim the front end is inherited *by decision* rather
than by omission. It joins the seven passages in §5 of this file that exist
because they are unflattering.
