"""Select high-fidelity illustrations without changing maps or reported metrics.
Rank by ours absolute PSNR, never by a baseline margin; preserve common cameras.
"""
import json, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'scripts/paper'),str(ROOT/'splatt3r_core')]
import numpy as np
import torch
from PIL import Image,ImageDraw
from render_comparisons import scene_paths,exported_sh,render_full,u8,fingerprint
from eval_map_quality import load_tum_traj
from splatt3r_slam.gaussian_ply_codec import decode_gaussians_from_ply
OUT=ROOT/'docs/Thesis/ral/fig/publication_assets'

@torch.no_grad()
def main():
 from splatt3r_slam.config import load_config
 from splatt3r_slam.dataloader import load_dataset
 from splatt3r_slam.image import normalize_exposure,reset_exposure_reference
 load_config(str(ROOT/'config/eval_calib.yaml'));OUT.mkdir(parents=True,exist_ok=True)
 record=json.loads((ROOT/'logs/paper_ral_20261007/desk/metrics.json').read_text())
 ds_path,methods=scene_paths('desk');ds=load_dataset(str(ds_path));ts,poses=load_tum_traj(ds_path/'groundtruth.txt')
 rows=record['methods']['ours']['per_frame'];chosen=[]
 for i in sorted(range(len(rows)),key=lambda i:-rows[i]['psnr']):
  fi=record['frames'][i]['index']
  if all(abs(fi-record['frames'][j]['index'])>=80 for j in chosen):chosen.append(i)
  if len(chosen)==5:break
 reset_exposure_reference();normalize_exposure(ds.get_image(0))
 K=np.array(ds.camera_intrinsics.K_frame).copy();K[:2]*=1.25
 for row in chosen:
  idx=record['frames'][row]['index'];rgb=normalize_exposure(ds.get_image(idx))
  im=Image.fromarray((np.clip(rgb,0,1)*255).round().astype('uint8'))
  if im.size!=(640,480):im=im.resize((640,480),Image.Resampling.LANCZOS)
  im.save(OUT/f'desk_{idx}_gt.png')
 for name,spec in methods.items():
  g=decode_gaussians_from_ply(spec['ply'],device='cuda:0');sh=exported_sh(spec['ply'],g,'cuda:0')
  align=record['methods'][name]['alignment'];s,R,t=align['s'],np.array(align['R']),np.array(align['t'])
  for row in chosen:
   idx=record['frames'][row]['index'];j=abs(ts-record['frames'][row]['timestamp']).argmin();cam=np.eye(4)
   cam[:3,:3]=R.T@poses[j,:3,:3];cam[:3,3]=R.T@(poses[j,:3,3]-t)/s
   pred=render_full(g,sh,cam,K,(480,640),'cuda:0');Image.fromarray(u8(pred)).save(OUT/f'desk_{idx}_{name}.png')
  if name=='ours':orbit_views(g,sh,s,R,t)
  del g,sh;torch.cuda.empty_cache()
 a,b=record['methods']['ours'],record['methods']['photo'];delta=[x['psnr']-y['psnr'] for x,y in zip(a['per_frame'],b['per_frame'])]
 manifest={'selection_rule':'descending ours absolute PSNR; >=80 frame separation; no baseline-margin ranking',
 'evaluation_source':'logs/paper_ral_20261007/desk/metrics.json','display_resolution':[640,480],
 'scoring_resolution':record['resolution_hw'],'reported_metrics_unchanged':True,
 'scope':'TUM freiburg1_desk only, 100 common non-keyframes',
 'scene_delta_psnr':a['full_sh']['psnr']-b['full_sh']['psnr'],
 'mean_paired_frame_delta_psnr':float(np.mean(delta)),'median_paired_frame_delta_psnr':float(np.median(delta)),
 'selected':[{'row':i,'frame':record['frames'][i]['index'],'ours_psnr':rows[i]['psnr']} for i in chosen],
 'artifacts':{n:{k:fingerprint(p) for k,p in sp.items()} for n,sp in methods.items()}}
 extras=[]
 for scene,cand,tag in [('fr1_360',11,'360'),('table_3',6,'table')]:
  for method in ['gt','ours']:
   src=ROOT/'logs/paper_real_20261010'/scene/f'cand{cand}_{method}.png'
   dst=OUT/f'extra_{tag}_{method}.png'
   dst.write_bytes(src.read_bytes())
   extras.append({'source':str(src.relative_to(ROOT)),'asset':dst.name})
 manifest['additional_scenes']={'rule':'highest PSNR among 12 saved candidates; separate recorded real-data runs','files':extras}
 (OUT/'selection.json').write_text(json.dumps(manifest,indent=2)+'\n')
 sheet=Image.new('RGB',(1280,len(chosen)*260),'white');draw=ImageDraw.Draw(sheet)
 for r,row in enumerate(chosen):
  idx=record['frames'][row]['index']
  for c,n in enumerate(['gt','photo','mono','ours']):
   sheet.paste(Image.open(OUT/f'desk_{idx}_{n}.png').resize((320,240)),(c*320,r*260+20));draw.text((c*320+4,r*260+3),f'{idx} {n}',fill='black')
 sheet.save('/tmp/desk-publication-candidates.png');print(manifest['selected'],flush=True)

def orbit_views(g,sh,s,R,t):
 from src.pixelsplat_src.cuda_splatting import render_cuda
 pts=g['means'][::80].cpu().numpy();pts=s*(pts@R.T)+t;center=np.median(pts,axis=0)
 extent=np.quantile(np.linalg.norm(pts-center,axis=1),.8);cams=[]
 for azi in [0,60,120,180,240,300]:
  az,el=np.radians([azi,48]);eye=center+extent*3.6*np.array([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),np.sin(el)])
  forward=center-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,0,1]);right/=np.linalg.norm(right);down=np.cross(forward,right)
  cam=np.eye(4);cam[:3,:3]=R.T@np.stack([right,down,forward],axis=1);cam[:3,3]=R.T@(eye-t)/s
  W,H=1000,750;k=torch.tensor([[[.95,0,.5],[0,1.267,.5],[0,0,1]]],device='cuda:0')
  image=render_cuda(torch.tensor(cam,device='cuda:0',dtype=torch.float32)[None],k,torch.tensor([.05],device='cuda:0'),torch.tensor([1000.],device='cuda:0'),(H,W),torch.ones(1,3,device='cuda:0'),g['means'][None],g['covariances'][None],sh[None],g['opacity'][None]).reshape(1,3,H,W).clamp(0,1)
  Image.fromarray(u8(image)).save(OUT/f'desk_orbit_{azi}.png');cams.append({'azimuth':azi,'camera_to_world_map':cam.tolist()})
 (OUT/'orbit_cameras.json').write_text(json.dumps(cams,indent=2)+'\n')
 sheet=Image.new('RGB',(1200,640),'white');draw=ImageDraw.Draw(sheet)
 for i,a in enumerate([0,60,120,180,240,300]):
  sheet.paste(Image.open(OUT/f'desk_orbit_{a}.png').resize((400,300)),((i%3)*400,(i//3)*320+20));draw.text(((i%3)*400+5,(i//3)*320+3),str(a),fill='black')
 sheet.save('/tmp/desk-orbit-candidates.png')

if __name__=='__main__':main()
