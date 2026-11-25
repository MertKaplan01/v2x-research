#!/usr/bin/env python3
"""Train-only deployable class-size-prior Vehicle lifting baseline.

GT is not available to size_prior_lift(), fusion, or gating.  It is read only
after both baseline and size-prior outputs are frozen for post-hoc scoring.
"""
from __future__ import annotations
import argparse,csv,json,copy
from collections import Counter,defaultdict
from pathlib import Path
from zipfile import ZipFile
import cv2,numpy as np
from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import v2x_groundplane_fusion as g

ROOT=Path(__file__).resolve().parents[1];ZIP=ROOT/'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip';CALIB=ROOT/'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json';WEIGHTS=ROOT/'models/yolo11s.pt';OUT=ROOT/'outputs/v2x_vehicle_size_prior_baseline'
# Fixed external engineering assumptions (metres), never fit from dataset GT.
HEIGHT_PRIOR_M={'person':1.70,'car':1.50,'bus':3.20,'truck':3.25}
def ns(x):return g.time_ns(Path(x).name)
def med(x):return round(float(np.median(x)),4) if x else None
def q90(x):return round(float(np.quantile(x,.9)),4) if x else None
def detector(model,img):
 r=model(img,verbose=False)[0];out=[]
 for b in r.boxes:
  lab=model.names[int(b.cls[0])].lower()
  if lab in g.COLORS:out.append({'label':lab,'confidence':float(b.conf[0]),'bbox_xyxy':b.xyxy[0].cpu().numpy().tolist()})
 return {'detections':out}
def decompose(P):
 K,R,C,*_=cv2.decomposeProjectionMatrix(P);K=K/K[2,2];C=(C[:3]/C[3]).ravel();t=-R@C
 return K,R,t,R.T@np.array([0.,-1.,0.]) # physical vertical/up direction in Vehicle LiDAR frame
def size_prior_lift(objs,K,R,t,up,T,B,width,height):
 """Use bbox height -> optical depth, then centre -> physical contact point.
 Camera coordinate convention is +Z optical depth, +Y image-down.  `up` converts
 camera-up to Vehicle-LiDAR; contact = object centre - H/2 * physical_up.
 """
 out=copy.deepcopy(objs)
 for d in out:
  x1,y1,x2,y2=map(float,d['bbox_xyxy']);h=y2-y1;H=HEIGHT_PRIOR_M[d['label']]
  flags=[]
  if h<=2: d['base_xy']=None;d['size_depth_m']=None;d['size_flags']=['too_small'];continue
  if x1<=2 or y1<=2 or x2>=width-3 or y2>=height-3:flags.append('truncated_border')
  if any(g.iou(d['bbox_xyxy'],o['bbox_xyxy'])>.10 for o in out if o is not d):flags.append('overlap_occlusion_proxy')
  u,v=(x1+x2)/2,(y1+y2)/2;Z=K[1,1]*H/h;Y=(v-K[1,2])*Z/K[1,1];X=(u-K[0,2]-K[0,1]*Y)/K[0,0]*Z
  centre=R.T@(np.array([X,Y,Z])-t);contact=centre-0.5*H*up;q=B@T@np.r_[contact,1.]
  d['base_xy']=(q[:2]/q[3]).tolist() if abs(q[3])>1e-8 and np.all(np.isfinite(q)) else None;d['size_depth_m']=float(Z);d['size_flags']=flags
 return out
def bottom(doc,oid,B):
 fr=next(iter(doc['openlabel']['frames'].values()));c=fr['objects'][oid]['object_data']['cuboid'];c=c[0] if isinstance(c,list) else c;co=g.corners_from_cuboid(c.get('val',[]))
 if co is None:return None
 q=(B@np.c_[co,np.ones(8)].T).T;ix=np.argsort(q[:,2]/q[:,3])[:4];return (q[ix,:2]/q[ix,3,None]).mean(axis=0).tolist()
def assoc(fused,la,rb):
 c=w=u=0;gaps=[];pairs={}
 for m in fused['matched']:
  a,b=la.get(m['local_index']),rb.get(m['remote_index']);pairs[m['local_index']]=m['remote_index']
  if a is None or b is None:u+=1
  elif a==b:c+=1;gaps.append(float(m['distance_m']))
  else:w+=1
 return c,w,u,gaps,pairs
def scene_group(ts,starts):return max(i+1 for i,x in enumerate(starts) if ts>=x)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);ap.add_argument('--conf',type=float,default=.25);a=ap.parse_args()
 for x in (ZIP,CALIB,WEIGHTS):
  if not x.is_file():raise FileNotFoundError(x)
 a.out.mkdir(parents=True,exist_ok=True);rc=json.loads(CALIB.read_text());Ps=np.asarray(rc['projection_from_s110_lidar_ouster_south'],float);B=np.asarray(rc['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float);plane=g.make_ground_plane(B);K,R,t,up=decompose(g.P_VEHICLE_LIDAR);model=YOLO(str(WEIGHTS));frames=[];objects=[];tot={x:Counter() for x in ('bottom_center','size_prior')};ranges={x:defaultdict(list) for x in tot};flags=defaultdict(list);scene=defaultdict(lambda:{x:Counter() for x in tot})
 with ZipFile(ZIP) as z:
  labels=sorted((ns(x),x) for x in z.namelist() if x.startswith('train/labels_point_clouds/') and x.endswith('.json'));starts=[labels[0][0]]+[labels[i][0] for i in range(1,len(labels)) if (labels[i][0]-labels[i-1][0])/1e9>.5]
  for n,(ts,lp) in enumerate(labels,1):
   doc=json.loads(z.read(lp));fr=next(iter(doc['openlabel']['frames'].values()));fp=fr['frame_properties'];ims=fp['image_file_names'];vn=next(x for x in ims if x.endswith('vehicle_camera_basler_16mm.jpg'));sn=next(x for x in ims if x.endswith('s110_camera_basler_south2_8mm.jpg'));vi=cv2.imdecode(np.frombuffer(z.read('train/images/vehicle_camera_basler_16mm/'+vn),np.uint8),cv2.IMREAD_COLOR);si=cv2.imdecode(np.frombuffer(z.read('train/images/s110_camera_basler_south2_8mm/'+sn),np.uint8),cv2.IMREAD_COLOR);T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T)
   # Freeze detections once; both lifting conditions consume the exact objects.
   local0=g.project_observations(detector(model,vi),Pv,plane,B,a.conf,180.);remote=g.project_observations(detector(model,si),Ps,plane,B,a.conf,180.);local1=size_prior_lift(local0,K,R,t,up,T,B,vi.shape[1],vi.shape[0]);refs=g.reference_objects(doc,B,Pv,Ps,vehicle_wh=(vi.shape[1],vi.shape[0]),rsu_wh=(si.shape[1],si.shape[0]))
   la=g.gt_match(local0,refs,'Vehicle',.3);rb=g.gt_match(remote,refs,'South2',.3);bottoms={r['id']:bottom(doc,r['id'],B) for r in refs};opps=set(la.values())&set(rb.values());sg=scene_group(ts,starts)
   # Inference was completed above. GT diagnostic lookup below.
   for name,local in (('bottom_center',local0),('size_prior',local1)):
    fused=g.fuse(local,remote,4.);c,w,u,gaps,pairs=assoc(fused,la,rb);li={v:k for k,v in la.items()};ri={v:k for k,v in rb.items()};success=sum(pairs.get(li[o])==ri[o] for o in opps)
    valid=sum(x['base_xy'] is not None for x in local);tot[name].update({'frames':1,'vehicle_detections':len(local),'vehicle_valid_lifts':valid,'vehicle_invalid_lifts':len(local)-valid,'spatial_associations':len(fused['matched']),'correct_association_pairs':c,'wrong_association_pairs':w,'unverifiable_association_pairs':u,'co_visible_opportunities':len(opps),'successful_opportunities':success,'unmatched_vehicle_posthoc':len(local)-len(la)});scene[sg][name].update({'opportunities':len(opps),'success':success})
    for i,oid in la.items():
     p=local[i]['base_xy'];b=bottoms.get(oid)
     if p is None or b is None:continue
     center=np.asarray(fr['objects'][oid]['object_data']['cuboid'][0]['val'][:3] if isinstance(fr['objects'][oid]['object_data']['cuboid'],list) else fr['objects'][oid]['object_data']['cuboid']['val'][:3],float);v=np.linalg.inv(T)@np.r_[center,1.];dist=float(np.linalg.norm(v[:3]/v[3]));err=float(np.linalg.norm(np.asarray(p)-np.asarray(b)));bn='0-20m' if dist<20 else ('20-40m' if dist<40 else ('40-60m' if dist<60 else '60m+'));ranges[name][bn].append(err)
     if name=='size_prior':
      for fl in local[i].get('size_flags',[]):flags[fl].append(err)
     objects.append({'frame':n,'scene_group':sg,'condition':name,'gt_id':oid,'label':local[i]['label'],'vehicle_distance_m':round(dist,4),'xy_error_to_gt_bottom_m':round(err,4),'size_depth_m':local[i].get('size_depth_m'),'size_flags':'|'.join(local[i].get('size_flags',[]))})
    frames.append({'frame':n,'scene_group':sg,'condition':name,'vehicle_detections':len(local),'vehicle_valid_lifts':valid,'vehicle_invalid_lifts':len(local)-valid,'posthoc_vehicle_2d_matches':len(la),'unmatched_vehicle_posthoc':len(local)-len(la),'co_visible_opportunities':len(opps),'spatial_associations':len(fused['matched']),'correct_association_pairs':c,'wrong_association_pairs':w,'unverifiable_association_pairs':u,'successful_opportunities':success,'correct_gap_median_m_conditional':med(gaps)})
   if n%25==0:print(f'completed {n}/{len(labels)} train frames',flush=True)
 with (a.out/'per_frame.csv').open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=frames[0].keys());w.writeheader();w.writerows(frames)
 with (a.out/'posthoc_matched_objects.csv').open('w',newline='',encoding='utf8') as h:w=csv.DictWriter(h,fieldnames=objects[0].keys());w.writeheader();w.writerows(objects)
 result={}
 for name,c in tot.items():
  allerr=[x for v in ranges[name].values() for x in v];result[name]={**dict(c),'vehicle_localization_error_to_gt_bottom_median_conditional':med(allerr),'vehicle_localization_error_to_gt_bottom_p90_conditional':q90(allerr),'association_recall_on_509_opportunities':round(c['successful_opportunities']/509,4),'verifiable_precision':round(c['correct_association_pairs']/max(1,c['correct_association_pairs']+c['wrong_association_pairs']),4),'unmatched_vehicle_rate_posthoc':round(c['unmatched_vehicle_posthoc']/max(1,c['vehicle_detections']),4),'distance_bins':{k:{'count':len(v),'median_error_m':med(v),'p90_error_m':q90(v)} for k,v in sorted(ranges[name].items())}}
 paired=defaultdict(dict);byclass=defaultdict(list)
 for x in objects:
  paired[(x['frame'],x['gt_id'])][x['condition']]=x
  if x['condition']=='size_prior':byclass[x['label']].append(float(x['xy_error_to_gt_bottom_m']))
 deltas=[float(v['bottom_center']['xy_error_to_gt_bottom_m'])-float(v['size_prior']['xy_error_to_gt_bottom_m']) for v in paired.values() if 'bottom_center' in v and 'size_prior' in v]
 result['paired_common_posthoc_vehicle_detections']={'count':len(deltas),'median_error_reduction_m_bottom_center_minus_size_prior':med(deltas),'mean_error_reduction_m_bottom_center_minus_size_prior':round(float(np.mean(deltas)),4) if deltas else None}
 result['size_prior_by_class_posthoc']={k:{'count':len(v),'median_error_to_gt_bottom_m':med(v),'p90_error_to_gt_bottom_m':q90(v)} for k,v in sorted(byclass.items())}
 result['scene_groups']={str(k):{name:{'opportunities':v[name]['opportunities'],'success':v[name]['success'],'recall':round(v[name]['success']/max(1,v[name]['opportunities']),4)} for name in tot} for k,v in scene.items()}
 fig,ax=plt.subplots(figsize=(7.5,4.4));order=['0-20m','20-40m','40-60m','60m+'];x=np.arange(4);ax.plot(x,[result['bottom_center']['distance_bins'].get(k,{}).get('median_error_m',np.nan) for k in order],'o-',label='bottom-center');ax.plot(x,[result['size_prior']['distance_bins'].get(k,{}).get('median_error_m',np.nan) for k in order],'s-',label='size-prior');ax.set(xticks=x,xticklabels=order,xlabel='GT center distance from Vehicle LiDAR (post-hoc)',ylabel='median XY error to GT bottom face (m)',title='Train-only paired Vehicle lifting comparison');ax.grid(alpha=.3);ax.legend();fig.tight_layout();fig.savefig(a.out/'error_vs_distance.png',dpi=180);plt.close(fig)
 summary={'status':'TRAIN_ONLY_DEPLOYABLE_SIZE_PRIOR_BASELINE','frames':len(labels),'test_accessed':False,'priors_m':HEIGHT_PRIOR_M,'geometry':'Z=fy*class_height/bbox_height; bbox center is back-projected through decomposed provisional P; physical contact=center-(H/2)*camera_up_in_vehicle; then Vehicle-to-S110 reference pose and S110-base transform produce base_xy.','frozen_components':'YOLO detections, provisional P, reference LiDAR pose, RSU lifting, g.fuse and 4m gate unchanged.','results':result,'size_prior_failure_flags':{k:{'matched_objects':len(v),'median_error_to_gt_bottom_m':med(v)} for k,v in flags.items()},'warnings':['Physical height priors are broad assumptions; truncation/occlusion flags are heuristic and not labels.','Errors use GT bottom-face post-hoc; no GT dimensions, positions, or IDs enter inference or association.','Train scene groups are development partitions, not leakage-safe held-out evaluation.']}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');(a.out/'README.md').write_text('# Vehicle size-prior baseline\n\n`python3 src/v2x_vehicle_size_prior_baseline.py`\n\nTrain-only; GT is post-hoc only.\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
