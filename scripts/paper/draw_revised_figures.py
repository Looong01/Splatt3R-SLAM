"""Revised figures (2026-10): real-data teaser | multi-dataset qualitative |
'Beyond Photo-SLAM' bars kept beside the original post-hoc truncation line
chart. Reuses the editable ``Drawing`` framework so panels can export as
editable PPT shapes. The system overview stays an independent TikZ figure
(see docs/Thesis/ral/fig/overview.tikz); it is intentionally not built here.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/paper"))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from draw_paper_diagrams import (Drawing, camera, gaussians, INK, MUTED, RULE,
                                 TEAL, BLUE)

FIG = ROOT / 'docs/Thesis/ral/fig'
OLDDATA = ROOT / 'logs/paper_ral_20261007'
REAL = ROOT / 'logs/paper_real_20261010'
AMBER, ORANGE, PANEL_DARK = '#C2821E', '#AA772C', '#0E1215'
REP = [f'office{i}' for i in range(5)] + [f'room{i}' for i in range(3)]


def cover(d, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    target, src = w / h, iw / ih
    if src > target:
        cw = int(round(ih * target)); ch = ih; cx = (iw - cw)//2; cy = 0
    else:
        ch = int(round(iw/target)); cw = iw; cx = 0; cy = (ih-ch)//2
    d.image(path, x, y, w, h, crop=(cx, cy, cw, ch))


def tag(d, x, y, text, color=INK, size=7.3, align='left'):
    d.text(x, y, text, size=size, color=color, align=align)


# ---------------------------------------------------------------------------
def teaser_real():
    d = Drawing(7.15, 2.60)
    # Left: idea
    d.text(1.20, .16, 'Image pairs to a persistent map', size=8.5,
           bold=True, align='center')
    # pair
    cover(d, REAL/'desk/cand0_gt.png', .07, .52, .52, .38)
    cover(d, REAL/'desk/cand1_gt.png', .07, .94, .52, .38)
    d.line([(.61,.91),(.82,.91)], INK, 1.0, True)
    # shared anchor
    cx, cy = 1.14, .91
    d.rect(cx-.05,cy-.05,.10,.10,INK,'white')
    camera(d,cx+.04,cy-.34,.62,BLUE)
    gaussians(d,cx+.10,cy+.30,.62,TEAL)
    d.line([(cx,cy),(cx+.17,cy-.22)],BLUE,.8)
    d.line([(cx,cy),(cx+.22,cy+.22)],TEAL,.8)
    tag(d,cx,1.44,'shared anchor',TEAL,7.1,'center')
    d.line([(cx+.31,.91),(1.50,.91)],INK,1.0,True)
    # persistent map
    cover(d,REAL/'fr1_360/cand9_ours.png',1.52,.55,.62,.62)
    tag(d,1.83,.47,'persistent map',TEAL,7.1,'center')
    # claim + body
    d.text(1.20,1.72,'Map + cameras move together.',size=8.0,bold=True,
           align='center')
    d.text(1.20,2.02,'One network predicts geometry and\nappearance; pose corrections move\nevery linked Gaussian.',
           size=7.5,color=MUTED,align='center')
    d.rect(.06,2.49,.13,.10,BLUE,BLUE); tag(d,.23,2.54,'cameras',MUTED,7.2)
    d.rect(.86,2.49,.13,.10,TEAL,TEAL); tag(d,1.03,2.54,'Gaussians',MUTED,7.2)

    # ---- Right: two light-background renders, then dark 3D-primitive panel ----
    d.text(4.79,.16,'One Gaussian map, many rendered views',size=8.6,
           bold=True,align='center')
    # Two held-out renders on a light background (no dark panel).
    xr=2.52
    cover(d,REAL/'table_3/cand2_ours.png',xr,.52,1.74,.80)
    tag(d,xr+.87,.44,'ETH3D render',MUTED,7.1,'center')
    cover(d,REAL/'fr1_360/cand9_ours.png',xr,1.52,1.74,.80)
    tag(d,xr+.87,1.44,'TUM render',MUTED,7.1,'center')
    d.text(xr+.87,2.46,'TUM desk: +3.28 dB\nvs. Photo-SLAM',size=7.0,
           color=TEAL,bold=True,align='center')
    # Dark panel ONLY behind the 3D primitives, far right.
    px=4.46
    d.rect(px,.30,2.63,2.26,PANEL_DARK,PANEL_DARK,zorder=0)
    cover(d,FIG/'gauss_cloud.png',px+.08,.60,2.47,1.70)
    tag(d,px+1.31,.47,'3D Gaussian primitives','#C9D2D6',7.3,'center')
    tag(d,px+1.31,2.42,'one explicit map behind every view','#9FB0B6',7.0,'center')
    return d


# ---------------------------------------------------------------------------
def qualitative_multi():
    d = Drawing(7.15, 4.03)
    d.text(3.575,.13,'Real-world reconstructions across datasets',
           size=8.8,bold=True,align='center')
    tag(d,3.575,.35,'TUM desk: identical views and numbered detail crops',
        BLUE,7.3,'center')
    cols=[('GT',OLDDATA/'desk/view0_gt.png'),
          ('Photo-SLAM',OLDDATA/'desk/view0_photo.png'),
          ('MonoGS',OLDDATA/'desk/view0_mono.png'),
          ('Ours',OLDDATA/'desk/view0_ours.png')]
    x0,gap=.06,.07
    cw=(7.15-2*x0-3*gap)/4
    ih=cw*.75
    sy0, sh = 0, 384
    rois=[(150,205,120,90),(358,148,120,90)]   # controllers | keyboard keys
    iw2=(cw-.05)/2; ih2=iw2*90/120              # two 4:3 insets per column
    main_y=.64
    iy=main_y+ih+.07
    for i,(lab,p) in enumerate(cols):
        x=x0+i*(cw+gap)
        cover(d,p,x,main_y,cw,ih)
        tag(d,x+cw/2,.52,lab,TEAL if lab=='Ours' else INK,7.1,'center')
        for j,(px,py,pw,ph) in enumerate(rois):
            bx=x+px/512*cw; by=main_y+(py-sy0)/sh*ih
            bw=pw/512*cw; bh=ph/sh*ih
            d.rect(bx,by,bw,bh,ORANGE,'none',.9,zorder=3)
            d.text(bx+.015,by+.085,str(j+1),size=6.6,color=ORANGE,bold=True)
            ix=x+j*(iw2+.05)
            d.image(p,ix,iy,iw2,ih2,crop=(px,py,pw,ph))
            d.rect(ix,iy,iw2,ih2,ORANGE,'none',.8,zorder=3)
            d.text(ix+.015,iy+.08,str(j+1),size=6.4,color=ORANGE,bold=True)
    yr=iy+ih2+.12
    d.line([(.06,yr),(7.09,yr)],RULE,.5)
    tag(d,3.575,yr+.13,'More real scenes: ground truth (top), ours (bottom)',
        MUTED,7.4,'center')
    scenes=[('TUM fr1/360',REAL/'fr1_360/cand9'),
            ('ETH3D table',REAL/'table_3/cand2'),
            ('TUM controller',REAL/'fr1_360/cand0')]
    cw2=(7.15-2*x0-2*gap)/3
    ys=yr+.32
    for i,(lab,base) in enumerate(scenes):
        x=x0+i*(cw2+gap)
        tag(d,x+cw2/2,ys,lab,BLUE,7.1,'center')
        cover(d,str(base)+'_gt.png',x,ys+.14,cw2,.39)
        cover(d,str(base)+'_ours.png',x,ys+.14+.39+.05,cw2,.39)
    return d


# ---------------------------------------------------------------------------
def evidence():
    """Two panels: (a) Beyond Photo-SLAM per-scene gain bars (teal=Replica,
    blue=real desk), (b) the original post-hoc truncation line chart kept
    verbatim from the measured study."""
    records = {s: json.loads((OLDDATA/s/'metrics.json').read_text())
               for s in REP+['desk']}
    f, ax = plt.subplots(1, 2, figsize=(7.15, 1.98))
    f.subplots_adjust(left=.072, right=.986, top=.83, bottom=.26, wspace=.26)

    # (a) Beyond Photo-SLAM: per-scene PSNR gain, Replica + real desk.
    labels = ['o0','o1','o2','o3','o4','r0','r1','r2','desk*']
    deltas = [records[s]['methods']['ours']['full_sh']['psnr'] -
              records[s]['methods']['photo']['full_sh']['psnr']
              for s in REP+['desk']]
    colors = [TEAL]*8 + [BLUE]
    ax[0].bar(range(9), deltas, color=colors, width=.68)
    ax[0].set(xticks=range(9), xticklabels=labels, ylabel='ΔPSNR (dB) ↑',
              ylim=(0, 8.8))
    ax[0].set_title('(a) Beyond Photo-SLAM: per-scene gain', loc='left',
                    fontsize=8.5)
    for i, v in enumerate(deltas):
        ax[0].text(i, v+.17, f'{v:.1f}', ha='center', fontsize=6.6)

    # (b) Original post-hoc truncation study (ours only; unchanged numbers).
    budgets = [23019, 83372, 300000, 1000000]
    values = {'office0':[12.04,13.54,20.23,24.68,26.29],
              'office1':[12.08,13.97,18.96,21.71,22.08],
              'room0':[5.16,8.68,16.26,25.01,25.44],
              'room2':[7.74,10.43,17.87,22.87,23.62]}
    for (s, ys), color, marker in zip(values.items(),
            [TEAL, BLUE, ORANGE, '#8662A1'], ['o','s','^','D']):
        n = records[s]['methods']['ours']['gaussians']
        ax[1].plot(budgets+[n], ys, marker=marker, ms=3, lw=1, label=s,
                   color=color)
    ax[1].set(xscale='log', xticks=[1e5,1e6], xticklabels=['100K','1M'],
              xlabel='Retained Gaussians (log scale)', ylabel='PSNR (dB) ↑')
    ax[1].set_title('(b) Map budget: post-hoc truncation', loc='left',
                    fontsize=8.5)
    ax[1].legend(fontsize=6.4, loc='lower right', ncol=2, frameon=False,
                 handlelength=1.3, columnspacing=.8)
    for a in ax:
        a.grid(axis='y', color='#DFE5E8', lw=.5)
        a.set_axisbelow(True); a.tick_params(labelsize=7, length=2)
    with plt.rc_context({'pdf.fonttype': 42, 'svg.fonttype': 'none'}):
        for ext in ('pdf', 'svg', 'png'):
            f.savefig(FIG / f'evidence.{ext}', dpi=240)
    plt.close(f)


def render_all():
    teaser_real().render('teaser')
    qualitative_multi().render('qualitative_multi')
    evidence()
    print('revised figures rendered (teaser, qualitative_multi, evidence)')


if __name__ == '__main__':
    render_all()
