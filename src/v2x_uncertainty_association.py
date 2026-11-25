#!/usr/bin/env python3
"""Train-only association-only comparison: greedy, Hungarian, uncertainty Hungarian."""
from __future__ import annotations
import argparse,csv,json,hashlib,copy,math
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
import cv2,numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO
import v2x_groundplane_fusion as g
import v2x_vehicle_size_prior_baseline as sp

ROOT=Path(__file__).resolve().parents[1];ZIP=ROOT/'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip';CALIB=ROOT/'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json';WEIGHTS=ROOT/'models/yolo11s.pt';BASE=ROOT/'outputs/v2x_vehicle_size_prior_baseline/summary.json';OUT=ROOT/'outputs/v2x_uncertainty_association'
# Fixed engineering assumptions; deliberately not fitted/tuned on GT.
SIGMA_PIXEL=3.0; SIGMA_BBOX_HEIGHT_MIN=3.0; HEIGHT_REL_SIGMA={'person':.20,'car':.20,'bus':.25,'truck':.25}; MAHAL_D2_GATE=9.21; MAX_SIGMA_M=12.0; LOGDET_WEIGHT=.25
def ns(x):return g.time_ns(Path(x).name)
def med(x):return round(float(np.median(x)),4) if x else None
def det(model,img):
 r=model(img,verbose=False)[0];out=[]
 for b in r.boxes:
  n=model.names[int(b.cls[0])].lower()
  if n in g.COLORS:out.append({'label':n,'confidence':float(b.conf[0]),'bbox_xyxy':b.xyxy[0].cpu().numpy().tolist()})
 return {'detections':out}
def vehicle_xy(u,v,h,H,K,R,t,up,T,B):
 if h<=2:return None
 Z=K[1,1]*H/h;Y=(v-K[1,2])*Z/K[1,1];X=(u-K[0,2]-K[0,1]*Y)/K[0,0]*Z;centre=R.T@(np.array([X,Y,Z])-t);contact=centre-.5*H*up;q=B@T@np.r_[contact,1.]
 return q[:2]/q[3] if abs(q[3])>1e-8 and np.all(np.isfinite(q)) else None
def cov_num(f,x,sig):
 base=f(*x)
 if base is None:return None
 J=[]
 for i,s in enumerate(sig):
  xp=list(x);xm=list(x);xp[i]+=s;xm[i]-=s;yp,ym=f(*xp),f(*xm)
  if yp is None or ym is None:return None
  J.append((np.asarray(yp)-np.asarray(ym))/(2*s))
 C=np.stack(J,axis=1)@np.diag(np.asarray(sig)**2)@np.stack(J,axis=1).T
 w,V=np.linalg.eigh(C);w=np.clip(w,.05**2,MAX_SIGMA_M**2);return V@np.diag(w)@V.T
def cov_vehicle(o,K,R,t,up,T,B):
 x1,y1,x2,y2=map(float,o['bbox_xyxy']);h=y2-y1;H=sp.HEIGHT_PRIOR_M[o['label']];u=(x1+x2)/2;v=(y1+y2)/2
 return cov_num(lambda U,V,Hpx,Hp:vehicle_xy(U,V,Hpx,Hp,K,R,t,up,T,B),[u,v,h,H],[SIGMA_PIXEL,SIGMA_PIXEL,max(SIGMA_BBOX_HEIGHT_MIN,.08*h),HEIGHT_REL_SIGMA[o['label']]*H])
def cov_rsu(o,Ps,plane,B):
 x1,y1,x2,y2=map(float,o['bbox_xyxy']);u=(x1+x2)/2;v=y2
 return cov_num(lambda U,V:g.foot_to_world(Ps,[U,V],plane,B,180.),[u,v],[SIGMA_PIXEL,SIGMA_PIXEL])
def hungarian(local,remote,mode,cvl=None,cvr=None):
 n,m=len(local),len(remote);cost=np.full((n,m),1e9);meta={}
 for i,a in enumerate(local):
  for j,b in enumerate(remote):
   if g.group(a['label'])!=g.group(b['label']) or a['base_xy'] is None or b['base_xy'] is None:continue
   d=np.asarray(a['base_xy'])-np.asarray(b['base_xy']);eu=float(np.linalg.norm(d))
   if mode=='euclidean':
    lim=4.+(1.5 if a['label'] in ('bus','truck') or b['label'] in ('bus','truck') else 0.)
    if eu<=lim:cost[i,j]=eu;meta[i,j]=(eu,None)
   else:
    C=cvl[i]+cvr[j] if cvl[i] is not None and cvr[j] is not None else None
    if C is None:continue
    try:inv=np.linalg.inv(C);d2=float(d@inv@d);ld=float(np.log(max(np.linalg.det(C),1e-12)))
    except np.linalg.LinAlgError:continue
    if d2<=MAHAL_D2_GATE:cost[i,j]=d2+LOGDET_WEIGHT*ld;meta[i,j]=(eu,d2)
 rr,cc=linear_sum_assignment(cost);out=[]
 for i,j in zip(rr,cc):
  if cost[i,j]<1e8:out.append({'local_index':int(i),'remote_index':int(j),'distance_m':meta[i,j][0],'d2':meta[i,j][1]})
 return out
def score(matches,la,rb,opps):
 c=w=u=0;pairs={};gaps=[]
 for x in matches:
  a,b=la.get(x['local_index']),rb.get(x['remote_index']);pairs[x['local_index']]=x['remote_index']
  if a is None or b is None:u+=1
  elif a==b:c+=1;gaps.append(x['distance_m'])
  else:w+=1
 li={v:k for k,v in la.items()};ri={v:k for k,v in rb.items()};s=sum(pairs.get(li[o])==ri[o] for o in opps)
 return c,w,u,s,gaps
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--conf',type=float,default=.25);a=ap.parse_args()
 for p in (ZIP,CALIB,WEIGHTS,BASE):
  if not p.is_file():raise FileNotFoundError(p)
 a.out.mkdir(parents=True,exist_ok=True);rc=json.loads(CALIB.read_text());Ps=np.asarray(rc['projection_from_s110_lidar_ouster_south'],float);B=np.asarray(rc['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float);plane=g.make_ground_plane(B);K,R,t,up=sp.decompose(g.P_VEHICLE_LIDAR);model=YOLO(str(WEIGHTS));tot={k:Counter() for k in ('A_greedy','B_hungarian','C_uncertainty_hungarian')};scenes=defaultdict(lambda:{k:Counter() for k in tot});rows=[];covstat=Counter()
 with ZipFile(ZIP) as z:
  labels=sorted((ns(x),x) for x in z.namelist() if x.startswith('train/labels_point_clouds/') and x.endswith('.json'));starts=[labels[0][0]]+[labels[i][0] for i in range(1,len(labels)) if (labels[i][0]-labels[i-1][0])/1e9>.5]
  for f,(ts,lp) in enumerate(labels,1):
   doc=json.loads(z.read(lp));fr=next(iter(doc['openlabel']['frames'].values()));fp=fr['frame_properties'];ims=fp['image_file_names'];vn=next(x for x in ims if x.endswith('vehicle_camera_basler_16mm.jpg'));sn=next(x for x in ims if x.endswith('s110_camera_basler_south2_8mm.jpg'));vi=cv2.imdecode(np.frombuffer(z.read('train/images/vehicle_camera_basler_16mm/'+vn),np.uint8),cv2.IMREAD_COLOR);si=cv2.imdecode(np.frombuffer(z.read('train/images/s110_camera_basler_south2_8mm/'+sn),np.uint8),cv2.IMREAD_COLOR);T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T)
   # Exact detection/lifting inputs are frozen before any GT post-hoc lookup.
   l0=g.project_observations(det(model,vi),Pv,plane,B,a.conf,180.);local=sp.size_prior_lift(l0,K,R,t,up,T,B,vi.shape[1],vi.shape[0]);remote=g.project_observations(det(model,si),Ps,plane,B,a.conf,180.);A=g.fuse(local,remote,4.)['matched'];cvL=[cov_vehicle(x,K,R,t,up,T,B) for x in local];cvR=[cov_rsu(x,Ps,plane,B) for x in remote];covstat.update({'vehicle_cov_available':sum(x is not None for x in cvL),'rsu_cov_available':sum(x is not None for x in cvR)});Bmatch=hungarian(local,remote,'euclidean');Cmatch=hungarian(local,remote,'mahal',cvL,cvR)
   # Evaluation-only GT begins here.
   refs=g.reference_objects(doc,B,Pv,Ps,vehicle_wh=(vi.shape[1],vi.shape[0]),rsu_wh=(si.shape[1],si.shape[0]));la=g.gt_match(local,refs,'Vehicle',.3);rb=g.gt_match(remote,refs,'South2',.3);opps=set(la.values())&set(rb.values());sg=max(i+1 for i,x in enumerate(starts) if ts>=x)
   for name,match in (('A_greedy',A),('B_hungarian',Bmatch),('C_uncertainty_hungarian',Cmatch)):
    c,w,u,s,gaps=score(match,la,rb,opps);tot[name].update({'frames':1,'vehicle_detections':len(local),'rsu_detections':len(remote),'spatial_associations':len(match),'correct_association_pairs':c,'wrong_association_pairs':w,'unverifiable_association_pairs':u,'co_visible_opportunities':len(opps),'successful_opportunities':s,'unmatched_vehicle_by_association':len(local)-len({x['local_index'] for x in match}),'unmatched_rsu_by_association':len(remote)-len({x['remote_index'] for x in match})});scenes[sg][name].update({'opportunities':len(opps),'success':s})
    rows.append({'frame':f,'scene_group':sg,'condition':name,'vehicle_detections':len(local),'rsu_detections':len(remote),'co_visible_opportunities':len(opps),'spatial_associations':len(match),'correct_association_pairs':c,'wrong_association_pairs':w,'unverifiable_association_pairs':u,'successful_opportunities':s,'unmatched_vehicle_by_association':len(local)-len({x['local_index'] for x in match}),'unmatched_rsu_by_association':len(remote)-len({x['remote_index'] for x in match}),'correct_gap_median_m_conditional':med(gaps)})
   if f%25==0:print(f'completed {f}/{len(labels)} train frames',flush=True)
 with (a.out/'per_frame.csv').open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 res={}
 for k,c in tot.items():
  v=c['correct_association_pairs']+c['wrong_association_pairs'];res[k]={**dict(c),'recall_on_509_opportunities':round(c['successful_opportunities']/509,4),'verifiable_precision':round(c['correct_association_pairs']/max(1,v),4),'unmatched_vehicle_rate':round(c['unmatched_vehicle_by_association']/max(1,c['vehicle_detections']),4),'unmatched_rsu_rate':round(c['unmatched_rsu_by_association']/max(1,c['rsu_detections']),4)}
 res['scene_groups']={str(s):{k:{'opportunities':v[k]['opportunities'],'success':v[k]['success'],'recall':round(v[k]['success']/max(1,v[k]['opportunities']),4)} for k in tot} for s,v in scenes.items()}
 summary={'status':'TRAIN_ONLY_UNCERTAINTY_ASSOCIATION','frames':len(labels),'test_accessed':False,'frozen_size_prior_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),'conditions':{'A':'existing g.fuse greedy 4m Euclidean gate','B':'same 4m/class-compatible Euclidean gate + global one-to-one Hungarian','C':'class-compatible Mahalanobis d2<=9.21 + Hungarian, cost=d2+0.25*log(det(sum covariance))'},'assumed_uncertainty_parameters':{'pixel_sigma_px':SIGMA_PIXEL,'bbox_height_sigma':'max(3 px, 8% of bbox height)','class_height_relative_sigma':HEIGHT_REL_SIGMA,'covariance_eigenvalue_floor_std_m':.05,'covariance_eigenvalue_cap_std_m':MAX_SIGMA_M,'note':'Assumed engineering sensitivities, not measured/calibrated covariances and not selected from GT.'},'covariance_propagation':'Central numerical Jacobians map Vehicle bbox center/height/class-height and RSU bbox bottom-center pixels through frozen geometry into S110-base XY.','covariance_coverage':dict(covstat),'results':res,'decision_rule':'Keep C only if paired recall improves without material precision loss or uncontrolled unverifiable growth.','warnings':['GT IDs/positions/dimensions are not used before matching; they are post-hoc evaluation only.','Scene groups are train development groups, not a leakage-safe held-out split.','Log-determinant term and covariance cap prevent arbitrary uncertainty inflation from making all candidates cheap.']}
 # Fixed conclusion is deliberately based only on reported comparison, not further tuning.
 c=res['C_uncertainty_hungarian'];a0=res['A_greedy'];summary['decision']='KEEP uncertainty-aware association' if c['recall_on_509_opportunities']>a0['recall_on_509_opportunities'] and c['verifiable_precision']>=a0['verifiable_precision']-.03 and c['unverifiable_association_pairs']<=a0['unverifiable_association_pairs']*1.25 else 'REJECT uncertainty-aware association; retain simpler baseline'
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');(a.out/'README.md').write_text('# Uncertainty-aware association\n\n`python3 src/v2x_uncertainty_association.py`\n\nTrain-only; covariance assumptions are explicitly non-calibrated.\n',encoding='utf8');print(json.dumps({'results':res,'decision':summary['decision']},indent=2))
if __name__=='__main__':main()
