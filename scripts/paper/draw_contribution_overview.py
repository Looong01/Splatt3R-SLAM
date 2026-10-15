"""Reference-style system overview with native editable UI geometry.
Filmstrip, module tabs, rounded image cards, loss block, renderer and stores
follow the supplied visual reference; all methods/data depict this project.
"""
from pathlib import Path
from draw_paper_diagrams import Drawing, camera, INK
ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT/'docs/Thesis/ral/fig/publication_assets'
STAGE = ROOT/'docs/Thesis/ral/fig/overview_assets'
NAVY, BLUE, RED = '#263B72', '#245597', '#B53D38'
PALEBLUE, PALERED, PEACH = '#EDF2FB', '#FFF0ED', '#FAE5D5'


def group_since(d, start, name):
    for o in d.items[start:]:
        o['group'] = name


def image_card(d,x,y,w,h,title,path,fill=PALEBLUE):
    start = len(d.items)
    d.ui_shape('roundrect',x,y,w,h,NAVY,fill,.85)
    d.text(x+w/2,y+.16,title,8.0,NAVY,align='center')
    iw = min(w-.20,(h-.37)*4/3)
    d.image(path,x+(w-iw)/2,y+.30,iw,iw*.75)
    group_since(d,start,title+' image card')


def overview(variant='ral'):
    d = Drawing(9.2,3.72)
    # Reference-style module boundaries and capsule titles.
    start = len(d.items)
    d.ui_shape('roundrect',3.05,.25,6.09,3.15,RED,'none',2.0)
    d.ui_shape('pill',4.68,.05,2.12,.40,RED,PALERED,2.0)
    d.text(5.74,.25,'Mapping module',10.2,INK,bold=True,align='center')
    group_since(d,start,'Mapping module boundary and tab')
    start = len(d.items)
    d.ui_shape('roundrect',.06,2.81,2.80,.58,BLUE,'none',1.8)
    d.ui_shape('pill',.48,2.61,1.96,.35,BLUE,PALEBLUE,1.8)
    d.text(1.46,2.79,'Tracking module',9.1,INK,bold=True,align='center')
    d.ui_shape('roundrect',.23,3.01,2.44,.26,NAVY,'#DCE7F7',.7)
    d.text(1.45,3.14,'MASt3R-SLAM  /  pose graph',7.4,NAVY,align='center')
    group_since(d,start,'Tracking module')
    # Real input filmstrip; the frame labels are illustrative, not timestamps.
    start = len(d.items)
    d.text(.47,.27,'Input RGB',8.7,NAVY,align='center')
    d.rect(.10,.48,.74,1.90,NAVY,'#E6EBF5',.5,zorder=0)
    for i,f in enumerate([24,377,478]):
        d.image(ASSET/f'desk_{f}_gt.png',.18,.60+i*.57,.58,.435)
    for yy in [.56+i*.14 for i in range(13)]:
        for xx in [.115,.785]:
            d.rect(xx,yy,.034,.074,'#FFFFFF','#FFFFFF',0)
    group_since(d,start,'Input RGB filmstrip')
    # Frozen shared predictor, with its real geometry output.
    start = len(d.items)
    d.ui_shape('roundrect',1.15,.76,1.45,.61,NAVY,PALEBLUE,.9)
    d.text(1.875,.96,'Frozen Splatt3R',8.7,NAVY,bold=True,align='center')
    d.text(1.875,1.20,'shared features',7.6,NAVY,align='center')
    group_since(d,start,'Frozen shared predictor')
    d.line([(.84,1.04),(1.15,1.04)],NAVY,.85,True)
    image_card(d,1.275,1.54,1.20,1.01,'Pointmap + matches',STAGE/'stage_pointmap.png')
    d.line([(1.875,1.37),(1.875,1.54)],NAVY,.85,True)
    d.line([(1.875,2.55),(1.875,2.61)],NAVY,.85,True)
    # Learned local Gaussian initialisation enters the map at the top.
    d.line([(2.60,1.06),(2.85,1.06),(2.85,.63),(8.34,.63),(8.34,.88)],NAVY,.85,True)
    d.text(3.88,.50,'predicted local Gaussians',7.4,NAVY,align='center')
    # Image cards, renderer bar and loss parallelogram from the reference UI.
    image_card(d,5.59,1.08,1.39,1.15,'Rendered RGB',ASSET/'desk_24_ours.png',PALERED)
    image_card(d,3.43,2.01,1.40,1.13,'Sampled RGB',ASSET/'desk_24_gt.png',PALERED)
    start = len(d.items)
    d.ui_shape('loss',3.60,1.17,1.46,.58,NAVY,'#F5C9C5',.9)
    d.text(4.33,1.36,'Rendering loss',8.3,NAVY,align='center')
    d.text(4.33,1.60,'L1 + D-SSIM',7.9,NAVY,align='center')
    group_since(d,start,'Photometric loss block')
    start = len(d.items)
    d.ui_shape('roundrect',7.25,.97,.25,2.26,NAVY,'#D9E3F6',.8)
    d.text(7.375,2.10,'Differentiable renderer',7.8,NAVY,align='center',rotation=90)
    group_since(d,start,'Differentiable renderer bar')
    # Cylindrical stores carry local Gaussian parameters and relative RGB cameras.
    start = len(d.items)
    d.ui_shape('cylinder',7.79,.89,1.10,1.42,NAVY,PEACH,.95)
    d.text(8.34,1.19,'Local map',8.4,NAVY,align='center')
    d.image(STAGE/'stage_local_gaussians_tight.png',8.01,1.31,.66)
    d.text(8.34,2.17,r'$\mathcal{G}_k,\,\theta_k$',8.3,RED,align='center',ppt='Gₖ, θₖ')
    group_since(d,start,'Local Gaussian map store')
    start = len(d.items)
    d.ui_shape('cylinder',7.79,2.59,1.10,.66,NAVY,PEACH,.95)
    camera(d,7.97,2.85,.58,BLUE)
    camera(d,8.45,2.85,.58,BLUE)
    d.text(8.34,3.10,'RGB cameras',7.7,NAVY,align='center')
    group_since(d,start,'Anchor-relative camera store')
    # The contribution is an explicit common anchor for map and observation poses.
    start = len(d.items)
    d.ui_shape('roundrect',5.69,2.64,1.38,.46,RED,PALERED,1.35)
    d.text(6.38,2.80,'Shared anchor k',8.0,RED,bold=True,align='center')
    d.text(6.38,2.98,'one pose snapshot',7.2,RED,align='center')
    group_since(d,start,'Shared anchor')
    d.text(8.34,2.46,'same correction',7.4,RED,align='center')
    # Data routes: true method outputs, not the blur/deblur logic of the reference.
    d.line([(7.79,1.65),(7.50,1.65)],NAVY,.85,True)
    d.line([(7.79,2.94),(7.50,2.94)],NAVY,.85,True)
    d.line([(7.25,1.65),(6.98,1.65)],NAVY,.85,True)
    d.line([(5.59,1.49),(4.983,1.49)],NAVY,.85,True)
    d.text(5.33,1.32,r'$\hat I$',8.3,NAVY,align='center',ppt='Î')
    d.line([(4.13,2.01),(4.13,1.75)],NAVY,.85,True)
    d.text(4.35,1.88,r'$I_{obs}$',8.3,NAVY,align='center',ppt='I_obs')
    d.line([(2.67,3.10),(3.19,3.10),(3.19,2.73),(3.43,2.73)],NAVY,.85,True)
    d.line([(2.67,3.25),(5.35,3.25),(5.35,2.88),(5.69,2.88)],BLUE,.85,True)
    d.text(4.38,3.19,'current keyframe poses',7.1,BLUE,align='center')
    # The same anchor update reaches both stores; parameter optimisation is separate.
    d.line([(6.38,3.10),(6.38,3.30),(9.01,3.30),(9.01,1.60),(8.89,1.60)],RED,.9,True)
    d.line([(9.01,2.92),(8.89,2.92)],RED,.9,True)
    d.line([(4.33,1.17),(4.33,.88),(7.62,.88),(7.62,1.10),(7.79,1.10)],RED,.85,True)
    d.text(6.08,.78,'optimise Gaussian parameters',7.2,RED,align='center')
    d.text(1.48,3.60,'Blue: geometry / fixed network',7.6,BLUE,align='center')
    d.text(6.00,3.60,'Red: map refinement  ·  map + cameras share the same anchor',7.6,RED,align='center')
    for item in d.items:
        if item['kind']=='line' and not item.get('group'):
            item['connector'] = True
    return d


if __name__ == '__main__':
    for v in ['ral','master']:
        overview(v).render('system_overview_'+v)
