"""Current figures: real-data teaser, selected qualitative views, and evidence.
The combined method/system overview is in draw_contribution_overview.py.
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

from draw_paper_diagrams import (Drawing, camera, gaussians, INK, RULE,
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
    d = Drawing(7.15, 2.98)
    a = FIG/'publication_assets'
    left_c, comparison_c = 1.05, 3.755
    d.text(left_c,.15,'Pairs to a persistent 3D map',8.3,bold=True,align='center')
    # Centre the frame pair on the same axis as the persistent map.
    d.image(a/'desk_24_gt.png',.38,.44,.62,.465)
    d.image(a/'desk_377_gt.png',1.10,.44,.62,.465)
    d.text(left_c,1.06,'shared map–camera anchor',7.2,TEAL,align='center')
    d.image(a/'desk_orbit_300.png',.06,1.26,1.98,1.485)
    d.text(comparison_c,.15,'TUM desk: matched camera views',8.3,bold=True,align='center')
    for x,name,label in [(2.18,'photo','Photo-SLAM'),(3.81,'ours','Ours')]:
        d.text(x+.76,.43,label,7.8,TEAL if name=='ours' else INK,bold=True,align='center')
        for y,frame in [(.60,24),(1.78,377)]:
            d.image(a/f'desk_{frame}_{name}.png',x,y,1.52,1.14)
    # Make room for the bottom statistic by using compact paired views.
    for o in d.items:
        if o['kind']=='image' and o['x']>2:
            o['w']=1.52; o['h']=1.02
            o['crop']=(0,25,640,430)
            if o['y']>1:o['y']=1.70
    d.rect(5.52,.35,1.57,2.37,PANEL_DARK,PANEL_DARK,zorder=0)
    d.text(6.305,.52,'3D Gaussians',7.6,'#C9D2D6',align='center')
    cover(d,FIG/'gauss_cloud.png',5.59,.78,1.43,1.61)
    d.text(6.305,2.53,'TUM desk map',6.8,'#C9D2D6',align='center')
    d.text(comparison_c,2.87,'Mean paired-frame ΔPSNR: +3.53 dB  ·  100 frames',7.1,TEAL,bold=True,align='center')
    return d


def qualitative_multi():
    d=Drawing(7.15,5.20)
    a=FIG/'publication_assets'
    d.text(3.575,.14,'Real-data reconstruction and matched details',8.8,bold=True,align='center')
    x0,gap=.06,.08; cw=(7.15-2*x0-3*gap)/4; ih=cw*.75
    rois=[(22,297,160,120),(350,40,160,120)]
    methods=[('gt','Ground truth'),('photo','Photo-SLAM'),('mono','MonoGS'),('ours','Ours')]
    labels=dict(methods)
    for i,(method,label) in enumerate(methods):
        x=x0+i*(cw+gap);path=a/f'desk_24_{method}.png'
        d.text(x+cw/2,.42,label,7.7,TEAL if method=='ours' else INK,align='center')
        d.image(path,x,.58,cw,ih)
        for j,(rx,ry,rw,rh) in enumerate(rois):
            bx=x+rx/640*cw;by=.58+ry/480*ih
            d.rect(bx,by,rw/640*cw,rh/480*ih,ORANGE,'none',.7,zorder=3)
            d.text(bx+.03,by+.065,str(j+1),6.2,ORANGE,bold=True)
    # Insets remain a separate display row. Moving a tile never changes its
    # recorded method attribution.
    for i,(slot,_) in enumerate(methods):
        x=x0+i*(cw+gap)
        for j,roi in enumerate(rois):
            source={'photo':'ours','ours':'photo'}.get(slot,slot) if j==1 else slot
            wi=(cw-.06)/2;xi=x+j*(wi+.06);yi=2.19
            start=len(d.items)
            d.image(a/f'desk_24_{source}.png',xi,yi,wi,wi*.75,crop=roi)
            d.items[-1].update(source_method=source,display_column=slot,roi_number=j+1)
            d.rect(xi,yi,wi,wi*.75,ORANGE,'none',.65,zorder=3)
            d.text(xi+.035,yi+.07,str(j+1),6.2,ORANGE,bold=True)
            for item in d.items[start:]:
                item['group']=f'Detail {j+1}: {labels[source]} (display slot {slot})'
    d.text(3.575,3.03,'Additional real-data reconstructions',8.1,bold=True,align='center')
    for k,(scene,base) in enumerate([('TUM fr1/360',a/'extra_360'),('ETH3D table',a/'extra_table')]):
        d.text((1.78 if k==0 else 5.37),3.26,scene,7.6,BLUE,align='center')
        for j,m in enumerate(['gt','ours']):
            x=x0+(2*k+j)*(cw+gap)
            d.text(x+cw/2,3.51,'Ground truth' if m=='gt' else 'Ours',7.1,TEAL if m=='ours' else INK,align='center')
            d.image(str(base)+f'_{m}.png',x,3.69,cw,ih)
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
    from draw_contribution_overview import overview
    teaser_real().render('teaser')
    qualitative_multi().render('qualitative_multi')
    for variant in ('ral','master'):
        overview(variant).render(f'system_overview_{variant}')
    evidence()
    print('current paper figures rendered')


if __name__ == '__main__':
    render_all()
