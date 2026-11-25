#!/usr/bin/env python3
"""One fixed-pass TUMTraf Vehicle--RSU detection-benefit evaluation.

Phase A runs the frozen operational path using images, static calibration and
the privileged per-frame reference pose only.  It writes inference records
before Phase B opens GT objects for offline evaluation.  No threshold, model,
calibration, class prior, or Hungarian rule is adjusted here.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import cv2, numpy as np
from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import v2x_groundplane_fusion as g
import v2x_vehicle_size_prior_baseline as sp
import v2x_uncertainty_association as ua

ZIP=ROOT/'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip'
CALIB=ROOT/'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json'
WEIGHTS=ROOT/'models/yolo11s.pt'
ARCHIVED=ROOT/'outputs/v2x_uncertainty_association'
OUT=ROOT/'outputs/v2x_detection_benefit'
CONF=.25; GATE=4.; IOU=.3; CONDITION='B_hungarian'

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ns(x):return g.time_ns(Path(x).name)
def det(model,img):return sp.detector(model,img)
def agg_archived():
 rows=[]
 with (ARCHIVED/'per_frame.csv').open(newline='') as h:
  for r in csv.DictReader(h):
   if r['condition']==CONDITION:rows.append(r)
 total=Counter()
 for r in rows:
  for k in ('vehicle_detections','rsu_detections','spatial_associations','correct_association_pairs','wrong_association_pairs','unverifiable_association_pairs','co_visible_opportunities','successful_opportunities','unmatched_vehicle_by_association','unmatched_rsu_by_association'):total[k]+=int(r[k])
 total['frames']=len(rows)
 frozen=json.loads((ARCHIVED/'summary.json').read_text())['results'][CONDITION]
 fields=list(total); checks={k:{'archived_csv':total[k],'frozen_summary':frozen[k],'equal':total[k]==frozen[k]} for k in fields}
 return dict(total),checks,all(v['equal'] for v in checks.values())
def scene_groups(labels):
 starts=[labels[0][0]]
 for i in range(1,len(labels)):
  if (labels[i][0]-labels[i-1][0])/1e9>.5:starts.append(labels[i][0])
 return starts
def group(ts,starts):return max(i+1 for i,x in enumerate(starts) if ts>=x)
def freeze(o):
 return {'label':o['label'],'confidence':round(float(o['confidence']),8),'bbox_xyxy':[round(float(x),5) for x in o['bbox_xyxy']],'base_xy':None if o['base_xy'] is None else [round(float(x),6) for x in o['base_xy']],'size_depth_m':None if o.get('size_depth_m') is None else round(float(o['size_depth_m']),6),'size_flags':o.get('size_flags',[])}
def status_outputs(local,remote,matches,la,rb):
 paired_l={m['local_index'] for m in matches};paired_r={m['remote_index'] for m in matches}; outputs=[]
 for m in matches:
  a,b=la.get(m['local_index']),rb.get(m['remote_index'])
  if a is not None and b is not None and a!=b:outputs.append({'kind':'incorrect_merge','ids':[],'known_ids':[a,b]})
  elif a is not None or b is not None:outputs.append({'kind':'merged_confirmed_or_partial','ids':[a if a is not None else b],'known_ids':[x for x in (a,b) if x is not None]})
  else:outputs.append({'kind':'posthoc_unmatched_merge','ids':[],'known_ids':[]})
 for i in range(len(local)):
  if i not in paired_l:outputs.append({'kind':'vehicle_unmatched','ids':[la[i]] if i in la else [],'known_ids':[la[i]] if i in la else []})
 for j in range(len(remote)):
  if j not in paired_r:outputs.append({'kind':'rsu_unmatched','ids':[rb[j]] if j in rb else [],'known_ids':[rb[j]] if j in rb else []})
 return outputs
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 for p in (ZIP,CALIB,WEIGHTS,ARCHIVED/'per_frame.csv',ARCHIVED/'summary.json'):
  if not p.is_file():raise FileNotFoundError(p)
 archived,checks,ok=agg_archived()
 if not ok:raise RuntimeError('STOP: archived Hungarian CSV disagrees with frozen summary; no inference was run.')
 rc=json.loads(CALIB.read_text());Ps=np.asarray(rc['projection_from_s110_lidar_ouster_south'],float);B=np.asarray(rc['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float);plane=g.make_ground_plane(B);K,R,t,up=sp.decompose(g.P_VEHICLE_LIDAR)
 hashes={str(p.relative_to(ROOT)):digest(p) for p in (WEIGHTS,CALIB,ROOT/'src/v2x_groundplane_fusion.py',ROOT/'src/v2x_vehicle_size_prior_baseline.py',ROOT/'src/v2x_uncertainty_association.py')}
 config={'frozen_components':'YOLO11s, provisional Vehicle projection, class-size priors, RSU ground plane, LiDAR-derived reference pose, 4 m Hungarian association','confidence':CONF,'gate_m':GATE,'iou_posthoc':IOU,'hashes':hashes,'archived_regression_checks':checks,'archived_regression_passed':ok,'test_accessed':False}
 (a.out/'frozen_run_config.json').write_text(json.dumps(config,indent=2)+'\n')
 # PHASE A: no GT object access. Frame JSON is used only for image names and the
 # per-frame reference transform consumed by the frozen operational baseline.
 records=[]
 with ZipFile(ZIP) as z:
  labels=sorted((ns(x),x) for x in z.namelist() if x.startswith('train/labels_point_clouds/') and x.endswith('.json'));starts=scene_groups(labels);model=YOLO(str(WEIGHTS))
  for number,(ts,lp) in enumerate(labels,1):
   doc=json.loads(z.read(lp));fr=next(iter(doc['openlabel']['frames'].values()));props=fr['frame_properties'];ims=props['image_file_names'];vn=next(x for x in ims if x.endswith('vehicle_camera_basler_16mm.jpg'));sn=next(x for x in ims if x.endswith('s110_camera_basler_south2_8mm.jpg'))
   vi=cv2.imdecode(np.frombuffer(z.read('train/images/vehicle_camera_basler_16mm/'+vn),np.uint8),cv2.IMREAD_COLOR);ri=cv2.imdecode(np.frombuffer(z.read('train/images/s110_camera_basler_south2_8mm/'+sn),np.uint8),cv2.IMREAD_COLOR)
   T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T)
   l0=g.project_observations(det(model,vi),Pv,plane,B,CONF,180.);local=sp.size_prior_lift(l0,K,R,t,up,T,B,vi.shape[1],vi.shape[0]);remote=g.project_observations(det(model,ri),Ps,plane,B,CONF,180.);matches=ua.hungarian(local,remote,'euclidean')
   records.append({'frame':number,'timestamp_ns':ts,'scene_group':group(ts,starts),'label_member':lp,'vehicle_image':vn,'rsu_image':sn,'vehicle': [freeze(x) for x in local],'rsu':[freeze(x) for x in remote],'matches':matches})
   if number%25==0:print(f'phase_A {number}/{len(labels)}',flush=True)
 (a.out/'frozen_inference.jsonl').write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in records))
 # PHASE B: GT is opened only after all operational records are persisted.
 total=Counter();classes=defaultdict(Counter);scenes=defaultdict(Counter);rows=[]
 with ZipFile(ZIP) as z:
  for rec in records:
   doc=json.loads(z.read(rec['label_member']));fr=next(iter(doc['openlabel']['frames'].values()));T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T)
   local=rec['vehicle'];remote=rec['rsu'];matches=rec['matches'];refs=g.reference_objects(doc,B,Pv,Ps)
   eligible={x['id']:x for x in refs if x['Vehicle'] is not None or x['South2'] is not None};la=g.gt_match(local,list(eligible.values()),'Vehicle',IOU);rb=g.gt_match(remote,list(eligible.values()),'South2',IOU)
   lv=set(la.values());rv=set(rb.values());both=lv&rv;vonly=lv-rv;ronly=rv-lv;neither=set(eligible)-lv-rv
   ronly_out=sum(eligible[x]['Vehicle'] is None for x in ronly);ronly_in=len(ronly)-ronly_out
   # Predicted association quality uses exactly the existing post-hoc rule.
   corr=wrong=unv=0;pair_map={m['local_index']:m['remote_index'] for m in matches}
   for m in matches:
    x,y=la.get(m['local_index']),rb.get(m['remote_index'])
    if x is None or y is None:unv+=1
    elif x==y:corr+=1
    else:wrong+=1
   opps=both;success=sum(pair_map.get(i)==j for oid in opps for i,j in [(next(i for i,v in la.items() if v==oid),next(j for j,v in rb.items() if v==oid))])
   outputs=status_outputs(local,remote,matches,la,rb);covered=set(x for o in outputs for x in o['ids']);known_outputs=sum(bool(o['ids']) for o in outputs);false_merges=sum(o['kind']=='incorrect_merge' for o in outputs);posthoc_unmatched=sum(not o['ids'] for o in outputs);dups=sum(max(0,n-1) for n in Counter(x for o in outputs for x in o['ids']).values())
   missed_true_merges=sum(pair_map.get(next(i for i,v in la.items() if v==oid))!=next(j for j,v in rb.items() if v==oid) for oid in both)
   total.update({'frames':1,'eligible_gt':len(eligible),'vehicle_gt_detected':len(lv),'rsu_gt_detected':len(rv),'both_detected':len(both),'vehicle_only_detected':len(vonly),'rsu_only_detected':len(ronly),'neither_detected':len(neither),'rsu_only_outside_vehicle_fov':ronly_out,'rsu_only_inside_vehicle_fov_miss':ronly_in,'oracle_union_gt_covered':len(lv|rv),'operational_fused_gt_covered':len(covered),'operational_outputs':len(outputs),'confirmed_outputs':known_outputs,'posthoc_unmatched_outputs':posthoc_unmatched,'duplicate_confirmed_gt_outputs':dups,'incorrect_merges':false_merges,'missed_true_merges':missed_true_merges,'correct_associations':corr,'incorrect_associations':wrong,'unverifiable_associations':unv,'co_visible_opportunities':len(opps),'successful_opportunities':success})
   sg=scenes[rec['scene_group']];sg.update({'eligible_gt':len(eligible),'vehicle_gt_detected':len(lv),'rsu_gt_detected':len(rv),'oracle_union_gt_covered':len(lv|rv),'operational_fused_gt_covered':len(covered),'rsu_only_detected':len(ronly),'successful_opportunities':success,'co_visible_opportunities':len(opps)})
   for oid,ref in eligible.items():
    c=classes[ref['label']];c['eligible_gt']+=1;c['vehicle_gt_detected']+=oid in lv;c['rsu_gt_detected']+=oid in rv;c['oracle_union_gt_covered']+=oid in lv|rv;c['operational_fused_gt_covered']+=oid in covered
   rows.append({'frame':rec['frame'],'timestamp_ns':rec['timestamp_ns'],'scene_group':rec['scene_group'],'eligible_gt':len(eligible),'vehicle_gt_detected':len(lv),'rsu_gt_detected':len(rv),'both_detected':len(both),'vehicle_only_detected':len(vonly),'rsu_only_detected':len(ronly),'neither_detected':len(neither),'rsu_only_outside_vehicle_fov':ronly_out,'rsu_only_inside_vehicle_fov_miss':ronly_in,'oracle_union_gt_covered':len(lv|rv),'operational_fused_gt_covered':len(covered),'operational_outputs':len(outputs),'posthoc_unmatched_outputs':posthoc_unmatched,'duplicate_confirmed_gt_outputs':dups,'incorrect_merges':false_merges,'missed_true_merges':missed_true_merges,'correct_associations':corr,'incorrect_associations':wrong,'unverifiable_associations':unv,'co_visible_opportunities':len(opps),'successful_opportunities':success})
 with (a.out/'per_frame_coverage.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 def rate(key):return total[key]/total['eligible_gt'] if total['eligible_gt'] else None
 archived_named={'frames':archived['frames'],'correct_associations':archived['correct_association_pairs'],'incorrect_associations':archived['wrong_association_pairs'],'unverifiable_associations':archived['unverifiable_association_pairs'],'co_visible_opportunities':archived['co_visible_opportunities'],'successful_opportunities':archived['successful_opportunities']}
 association_fields=tuple(archived_named)
 summary={'status':'TRAIN_ONLY_FROZEN_PIPELINE_DETECTION_BENEFIT','archived_regression_passed':ok,'rerun_association':{'matches_archived':{k:total[k]==archived_named[k] for k in association_fields},'archived':archived_named,'rerun':{k:total[k] for k in association_fields}},'eligible_population':'GT road users of supported classes whose projected cuboid box intersects at least one of the Vehicle or South2 image planes; GT is used only after frozen inference for IoU=0.3 post-hoc matching.','counts':dict(total),'recall':{'vehicle_only':rate('vehicle_gt_detected'),'rsu_only':rate('rsu_gt_detected'),'oracle_gt_deduplicated_union':rate('oracle_union_gt_covered'),'operational_fused_output':rate('operational_fused_gt_covered'),'operational_minus_vehicle':rate('operational_fused_gt_covered')-rate('vehicle_gt_detected')},'precision':{'operational_confirmed_output_fraction':total['confirmed_outputs']/total['operational_outputs'] if total['operational_outputs'] else None,'posthoc_unmatched_output_fraction':total['posthoc_unmatched_outputs']/total['operational_outputs'] if total['operational_outputs'] else None,'association_verifiable_precision':total['correct_associations']/max(1,total['correct_associations']+total['incorrect_associations']),'association_confirmed_correct_fraction_all_pairs':total['correct_associations']/max(1,total['correct_associations']+total['incorrect_associations']+total['unverifiable_associations'])},'classwise':{k:{**dict(v),**{q:v[q]/v['eligible_gt'] for q in ('vehicle_gt_detected','rsu_gt_detected','oracle_union_gt_covered','operational_fused_gt_covered')}} for k,v in sorted(classes.items())},'scene_groups':{str(k):dict(v) for k,v in scenes.items()},'warnings':['RSU-only inside Vehicle FOV denotes a post-hoc detection miss under projected GT-box FOV, not physical occlusion.','Oracle union is GT-identity-deduplicated post-hoc coverage, not deployable fusion.','Operational output coverage counts unpaired outputs and conservatively does not credit a pair that merges two distinct matched GT objects.','Correlated train frames are descriptive development evidence, not independent generalization samples.']}
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (a.out/'README.md').write_text('# Detection-benefit evaluation\n\n`python3 scripts/evaluate_detection_benefit.py` runs one fixed 400-frame inference pass after archived association regression succeeds. GT evaluation runs only after inference records are written. Train-development only.\n')
 print(json.dumps({'archived_regression_passed':ok,'rerun_matches_archived':summary['rerun_association']['matches_archived'],'recall':summary['recall']},indent=2))
if __name__=='__main__':main()
