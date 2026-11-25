#!/usr/bin/env python3
"""Bounded cache-only RSU-ground-point -> Vehicle-image feasibility gate."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys
from collections import Counter,defaultdict,deque
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import v2x_groundplane_fusion as g
CACHE=ROOT/'outputs/v2x_detection_benefit/frozen_inference.jsonl'; FUSION=ROOT/'outputs/v2x_fusion_value/summary.json'; SELECTIVE=ROOT/'outputs/v2x_selective_fusion/summary.json'
ZIP=ROOT/'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip';CALIB=ROOT/'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json';OUT=ROOT/'outputs/v2x_reprojection_feasibility';IOU=.3;EXPAND=.20
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def expanded_inside(uv,b):
 x1,y1,x2,y2=map(float,b);w=x2-x1;h=y2-y1;return x1-EXPAND*w<=uv[0]<=x2+EXPAND*w and y1-EXPAND*h<=uv[1]<=y2+EXPAND*h
def residual(uv,b):
 x1,y1,x2,y2=map(float,b);w=max(1e-8,x2-x1);h=max(1e-8,y2-y1);dx=(uv[0]-(x1+x2)/2)/w;dy=(uv[1]-(y1+y2)/2)/h;return dx,dy,float(np.hypot(dx,dy))
def projection(Pv,B,xy):
 if xy is None:return None
 q=Pv@np.linalg.inv(B)@np.array([xy[0],xy[1],0.,1.])
 if not np.all(np.isfinite(q)) or q[2]<=1e-4:return None
 uv=q[:2]/q[2]
 return uv.tolist() if np.all(np.isfinite(uv)) else None
def candidate_components(local,proj):
 adj=defaultdict(set);edges=[]
 for j,uv in enumerate(proj):
  if uv is None:continue
  for i,v in enumerate(local):
   if g.group(v['label'])!=g.group(proj_labels[j]):continue
   if expanded_inside(uv,v['bbox_xyxy']):
    u=('v',i);r=('r',j);adj[u].add(r);adj[r].add(u);edges.append((i,j))
 seen=set();comps=[]
 for root in adj:
  if root in seen:continue
  q=deque([root]);seen.add(root);nodes=[]
  while q:
   u=q.popleft();nodes.append(u)
   for v in adj[u]:
    if v not in seen:seen.add(v);q.append(v)
  comps.append((sorted(x[1] for x in nodes if x[0]=='v'),sorted(x[1] for x in nodes if x[0]=='r')))
 return comps,edges
def outputs(local,remote,accepted,la,rb):
 pl={i for i,j in accepted};pr={j for i,j in accepted};o=[];c=Counter()
 for i,j in accepted:
  a,b=la.get(i),rb.get(j)
  if a is not None and b is not None and a!=b:o.append([]);c['incorrect']+=1
  elif a is not None or b is not None:o.append([a if a is not None else b]);c['correct' if a is not None and b is not None else 'unverifiable']+=1
  else:o.append([]);c['unverifiable']+=1
 for i in range(len(local)):
  if i not in pl:o.append([la[i]] if i in la else [])
 for j in range(len(remote)):
  if j not in pr:o.append([rb[j]] if j in rb else [])
 return o,c
def evaluate_outputs(frames):
 total=Counter();rows=[]
 for r in frames:
  ids=[x for o in r['outputs'] for x in o];cover=set(ids);dup=sum(max(0,n-1) for n in Counter(ids).values());total.update({'eligible_gt':len(r['eligible']),'outputs':len(r['outputs']),'confirmed_outputs':len(ids),'posthoc_unmatched_outputs':len(r['outputs'])-len(ids),'unique_gt_covered':len(cover),'duplicates':dup,**r['score']});rows.append({'frame':r['frame'],'scene_group':r['scene_group'],'accepted_pairs':len(r['accepted']),'outputs':len(r['outputs']),'unique_gt_covered':len(cover),'duplicates':dup,'correct':r['score']['correct'],'incorrect':r['score']['incorrect'],'unverifiable':r['score']['unverifiable']})
 return total,rows
def main():
 global proj_labels
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
 for p in (CACHE,FUSION,SELECTIVE,ZIP,CALIB):
  if not p.is_file():raise FileNotFoundError(p)
 recs=[json.loads(x) for x in CACHE.read_text().splitlines() if x]
 if len(recs)!=400 or any(not r['label_member'].startswith('train/') for r in recs):raise RuntimeError('STOP: expected 400 train-only frozen cache frames')
 a.out.mkdir(parents=True,exist_ok=True);rc=json.loads(CALIB.read_text());B=np.asarray(rc['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float);Ps=np.asarray(rc['projection_from_s110_lidar_ouster_south'],float)
 # Phase A operational projection: no object/GT lookup.
 projected=[]
 with ZipFile(ZIP) as z:
  for r in recs:
   doc=json.loads(z.read(r['label_member']));T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T);uv=[projection(Pv,B,x['base_xy']) for x in r['rsu']]
   projected.append({'rec':r,'uv':uv})
 # Phase A post-hoc GT evaluation begins only now.
 stats=Counter();pairrows=[];frame_info=[];classstats=defaultdict(Counter);signed=[]
 with ZipFile(ZIP) as z:
  for item in projected:
   r=item['rec'];doc=json.loads(z.read(r['label_member']));T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T);eligible={x['id']:x for x in g.reference_objects(doc,B,Pv,Ps) if x['Vehicle'] is not None or x['South2'] is not None};la=g.gt_match(r['vehicle'],list(eligible.values()),'Vehicle',IOU);rb=g.gt_match(r['rsu'],list(eligible.values()),'South2',IOU);li={v:k for k,v in la.items()};ri={v:k for k,v in rb.items()};common=set(li)&set(ri);stats['true_co_detected_pairs']+=len(common)
   valid_this=inside_this=rank_this=0
   for oid in common:
    i,j=li[oid],ri[oid];uv=item['uv'][j]
    if uv is None:pairrows.append({'frame':r['frame'],'scene_group':r['scene_group'],'label':r['vehicle'][i]['label'],'projection_valid':False,'inside_expanded_box':False,'rank_correct_first':False});continue
    valid_this+=1;stats['valid_true_pairs']+=1;dx,dy,norm=residual(uv,r['vehicle'][i]['bbox_xyxy']);inside=expanded_inside(uv,r['vehicle'][i]['bbox_xyxy']);inside_this+=inside;stats['inside_true_box']+=inside;signed.append((dx,dy,norm));classstats[r['vehicle'][i]['label']].update({'valid':1,'inside':inside})
    cand=[(residual(uv,v['bbox_xyxy'])[2],k) for k,v in enumerate(r['vehicle']) if g.group(v['label'])==g.group(r['rsu'][j]['label'])];cand.sort();rank=(cand[0][1]==i) if cand else False;rank_this+=rank;stats['ranked_correct_first']+=rank
    comp=min((x for x,k in cand if k!=i),default=None);pairrows.append({'frame':r['frame'],'scene_group':r['scene_group'],'label':r['vehicle'][i]['label'],'projection_valid':True,'inside_expanded_box':inside,'rank_correct_first':rank,'u':round(uv[0],3),'v':round(uv[1],3),'dx_over_w':round(dx,5),'dy_over_h':round(dy,5),'norm_residual':round(norm,5),'nearest_competing_norm_residual':None if comp is None else round(comp,5),'separation_competing_minus_true':None if comp is None else round(comp-norm,5)})
   frame_info.append({'r':r,'eligible':eligible,'la':la,'rb':rb,'uv':item['uv']})
 inside_rate=stats['inside_true_box']/max(1,stats['valid_true_pairs']);rank_rate=stats['ranked_correct_first']/max(1,stats['valid_true_pairs']);med=lambda x:float(np.median(x)) if x else None
 residual_summary={'median_dx_over_width':med([x[0] for x in signed]),'median_dy_over_height':med([x[1] for x in signed]),'median_norm_residual':med([x[2] for x in signed]),'p90_norm_residual':float(np.percentile([x[2] for x in signed],90)) if signed else None,'median_true_to_competing_separation':med([x['separation_competing_minus_true'] for x in pairrows if x.get('separation_competing_minus_true') is not None])}
 gate={'inside_expanded_box_rate_at_least_60pct':inside_rate>=.60,'correct_rank_first_rate_at_least_80pct':rank_rate>=.80}; stage_b=None;outrows=[]
 # Stage B only exists after the fixed Stage A gate passes.
 if all(gate.values()):
  runtime=[];diag=Counter()
  for f in frame_info:
   r=f['r'];proj_labels=[x['label'] for x in r['rsu']];comps,edges=candidate_components(r['vehicle'],f['uv']);accepted=[]
   for ls,rs in comps:
    if len(ls)==1 and len(rs)==1:accepted.append((ls[0],rs[0]))
    else:diag['ambiguous_components']+=1
   o,sc=outputs(r['vehicle'],r['rsu'],accepted,f['la'],f['rb']);runtime.append({'frame':r['frame'],'scene_group':r['scene_group'],'eligible':f['eligible'],'outputs':o,'score':sc,'accepted':accepted});diag.update({'candidate_edges':len(edges),'components':len(comps),'accepted_pairs':len(accepted)})
  t,outrows=evaluate_outputs(runtime);naive=json.loads(FUSION.read_text())['conditions']['naive_concatenation'];sel=json.loads(SELECTIVE.read_text())['selective_fusion'];go={'coverage_at_least_1894':t['unique_gt_covered']>=1894,'incorrect_merges_at_most_15':t['incorrect']<=15,'duplicates_removed_at_least_127':naive['duplicate_confirmed_outputs']-t['duplicates']>=127,'correct_pairs_at_least_50':t['correct']>=50};stage_b={'metrics':dict(t),'diagnostics':dict(diag),'go':go,'decision':'GO' if all(go.values()) else 'STOP'}
 old=json.loads(FUSION.read_text());selective=json.loads(SELECTIVE.read_text())
 report={'status':'TRAIN_ONLY_CACHED_RSU_TO_VEHICLE_REPROJECTION_FEASIBILITY','inputs':{'frames':400,'cache_sha256':sha(CACHE),'box_expansion_each_side_fraction':EXPAND,'iou_posthoc':IOU,'inference_rerun':False,'test_accessed':False},'projection_chain':'RSU base XY at z=0 -> inverse(s110_lidar_to_s110_base) -> s110 lidar -> inverse(vehicle_lidar_to_s110_lidar) -> vehicle lidar -> provisional P_VEHICLE_LIDAR -> Vehicle pixel.','stage_a':{'true_co_detected_pairs':stats['true_co_detected_pairs'],'valid_true_pairs':stats['valid_true_pairs'],'inside_expanded_box':stats['inside_true_box'],'inside_expanded_box_rate':inside_rate,'ranked_correct_first':stats['ranked_correct_first'],'ranked_correct_first_rate':rank_rate,'residuals':residual_summary,'classwise':{k:dict(v) for k,v in classstats.items()},'gate':gate,'decision':'PASS: run fixed Stage B' if all(gate.values()) else 'STOP: Stage A geometry gate failed; Stage B not implemented'},'stage_b':stage_b,'baselines':{'naive':{k:old['conditions']['naive_concatenation'][k] for k in ('unique_gt_covered','outputs','duplicate_confirmed_outputs')},'hungarian':{k:old['conditions']['hungarian_fusion'][k] for k in ('unique_gt_covered','outputs','duplicate_confirmed_outputs','correct_associations','incorrect_associations','unverifiable_associations')},'selective':{k:selective['selective_fusion'][k] for k in ('unique_gt_covered','outputs','duplicate_confirmed_outputs','correct_associations','incorrect_associations','unverifiable_associations')}},'selective_duplicate_reconciliation':'Selective fusion accepted 53 correct pairs but removed 66 confirmed duplicate outputs because duplicate count is an output-level post-hoc one-to-one quantity. The 12 incorrect merges suppress endpoint outputs too; where an endpoint had another retained cross-view representation, suppressing it can reduce the duplicate count without constituting a correct association. Thus duplicate reduction is not equal to correct-pair count.','warnings':['GT identities and boxes are evaluated only after all RSU-point projections are frozen.','Projection uses privileged LiDAR-derived pose and provisional Vehicle calibration; strict RGB-only operation is not established.','No calibration, class sizes, boxes, model outputs, or thresholds were changed.','A failed Stage A closes this association line; it does not prove RGB association fundamentally impossible.']}
 (a.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 fields=sorted({k for x in pairrows for k in x})
 with (a.out/'true_pair_reprojection.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(pairrows)
 bad=sorted((x for x in pairrows if x.get('projection_valid')),key=lambda x:x.get('norm_residual',-1),reverse=True)[:30]
 with (a.out/'misalignment_examples.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(bad)
 if outrows:
  with (a.out/'stage_b_per_frame.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=outrows[0].keys());w.writeheader();w.writerows(outrows)
 (a.out/'README.md').write_text('# RSU-to-Vehicle reprojection feasibility\n\n`python3 scripts/evaluate_reprojection_feasibility.py` is cache-only. Stage B runs only if both fixed Stage A gates pass.\n')
 print(json.dumps({'stage_a':report['stage_a'],'stage_b':stage_b},indent=2))
if __name__=='__main__':main()
