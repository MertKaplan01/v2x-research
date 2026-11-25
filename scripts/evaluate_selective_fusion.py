#!/usr/bin/env python3
"""Cache-only isolated-component selective fusion evaluation.

Operational rule: form the frozen class-compatible Euclidean candidate graph;
merge only a connected component with one Vehicle and one RSU node.  All other
observations remain separate.  GT is read only for post-hoc evaluation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import v2x_groundplane_fusion as g

CACHE=ROOT/'outputs/v2x_detection_benefit/frozen_inference.jsonl'
FUSION=ROOT/'outputs/v2x_fusion_value/summary.json'
ZIP=ROOT/'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip'
CALIB=ROOT/'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json'
OUT=ROOT/'outputs/v2x_selective_fusion'; IOU=.3

def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def limit(a,b):
 return 4.+(1.5 if a['label'] in ('bus','truck') or b['label'] in ('bus','truck') else 0.)

def graph_components(local,remote):
 """Return edge-containing bipartite components under the existing gate."""
 adj=defaultdict(set); edges=[]
 for i,a in enumerate(local):
  if a['base_xy'] is None: continue
  for j,b in enumerate(remote):
   if b['base_xy'] is None or g.group(a['label'])!=g.group(b['label']): continue
   d=float(np.linalg.norm(np.asarray(a['base_xy'])-np.asarray(b['base_xy'])))
   if d<=limit(a,b):
    u=('v',i);v=('r',j);adj[u].add(v);adj[v].add(u);edges.append((i,j,d))
 seen=set(); comps=[]
 for root in adj:
  if root in seen:continue
  q=deque([root]);seen.add(root);nodes=[]
  while q:
   u=q.popleft();nodes.append(u)
   for v in adj[u]:
    if v not in seen:seen.add(v);q.append(v)
  ls=sorted(x[1] for x in nodes if x[0]=='v');rs=sorted(x[1] for x in nodes if x[0]=='r')
  ds=[d for i,j,d in edges if i in ls and j in rs]
  comps.append((ls,rs,ds))
 return comps,edges

def outputs_for(local,remote,accepted,la,rb):
 paired_l={i for i,j,d in accepted};paired_r={j for i,j,d in accepted};out=[];score=Counter()
 for i,j,d in accepted:
  a,b=la.get(i),rb.get(j)
  if a is not None and b is not None and a!=b:
   out.append({'kind':'incorrect_merge','ids':[]});score['incorrect_associations']+=1
  elif a is not None or b is not None:
   out.append({'kind':'merged','ids':[a if a is not None else b]})
   if a is not None and b is not None:score['correct_associations']+=1
   else:score['unverifiable_associations']+=1
  else:
   out.append({'kind':'unmatched_merge','ids':[]});score['unverifiable_associations']+=1
 for i in range(len(local)):
  if i not in paired_l: out.append({'kind':'vehicle_retained','ids':[la[i]] if i in la else []})
 for j in range(len(remote)):
  if j not in paired_r: out.append({'kind':'rsu_retained','ids':[rb[j]] if j in rb else []})
 return out,score

def enrich(dst,out,eligible,lv,score):
 ids=[x for o in out for x in o['ids']];covered=set(ids);duplicates=sum(max(0,n-1) for n in Counter(ids).values());inc=covered-lv
 dst.update({'outputs':len(out),'confirmed_outputs':len(ids),'posthoc_unmatched_outputs':len(out)-len(ids),'unique_gt_covered':len(covered),'duplicate_confirmed_outputs':duplicates,'incremental_over_vehicle':len(inc),'incremental_outside_vehicle_fov':sum(eligible[x]['Vehicle'] is None for x in inc),'incremental_inside_vehicle_fov_miss':sum(eligible[x]['Vehicle'] is not None for x in inc)})
 dst.update(score);return covered

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=OUT);a=ap.parse_args()
 for p in (CACHE,FUSION,ZIP,CALIB):
  if not p.is_file():raise FileNotFoundError(p)
 records=[json.loads(x) for x in CACHE.read_text().splitlines() if x]
 if len(records)!=400 or any(not r['label_member'].startswith('train/') for r in records):raise RuntimeError('STOP: expected exactly 400 train-only cache records')
 old=json.loads(FUSION.read_text());a.out.mkdir(parents=True,exist_ok=True)
 rc=json.loads(CALIB.read_text());B=np.asarray(rc['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float);Ps=np.asarray(rc['projection_from_s110_lidar_ouster_south'],float)
 total=Counter();classes=defaultdict(Counter);scenes=defaultdict(Counter);diag=Counter();rows=[]
 with ZipFile(ZIP) as z:
  for rec in records:
   doc=json.loads(z.read(rec['label_member']));T=g.read_transform(doc);Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T)
   eligible={x['id']:x for x in g.reference_objects(doc,B,Pv,Ps) if x['Vehicle'] is not None or x['South2'] is not None};local,remote=rec['vehicle'],rec['rsu']
   # Evaluation-only data begins after deterministic graph/acceptance creation.
   comps,edges=graph_components(local,remote);accepted=[]
   for ls,rs,ds in comps:
    if len(ls)==1 and len(rs)==1:accepted.append((ls[0],rs[0],ds[0]))
    else:diag['ambiguous_components']+=1;diag['ambiguous_vehicle_nodes']+=len(ls);diag['ambiguous_rsu_nodes']+=len(rs)
   diag.update({'frames':1,'candidate_edges':len(edges),'edge_components':len(comps),'accepted_isolated_components':len(accepted),'unmatched_vehicle_nodes':len(local)-len({i for ls,rs,ds in comps for i in ls}),'unmatched_rsu_nodes':len(remote)-len({j for ls,rs,ds in comps for j in rs})})
   la=g.gt_match(local,list(eligible.values()),'Vehicle',IOU);rb=g.gt_match(remote,list(eligible.values()),'South2',IOU);lv=set(la.values());rv=set(rb.values())
   out,score=outputs_for(local,remote,accepted,la,rb);frame=Counter({'eligible_gt':len(eligible)});covered=enrich(frame,out,eligible,lv,score)
   # Correct co-detected opportunities not selected as the corresponding pair.
   selected={i:j for i,j,d in accepted};frame['co_visible_opportunities']=len(lv&rv)
   frame['missed_true_merges']=sum(selected.get(next(i for i,v in la.items() if v==oid))!=next(j for j,v in rb.items() if v==oid) for oid in lv&rv)
   total.update(frame);scenes[rec['scene_group']].update(frame)
   for oid,ref in eligible.items():
    c=classes[ref['label']];c['eligible_gt']+=1;c['unique_gt_covered']+=oid in covered;c['incremental_outside_vehicle_fov']+=(oid in covered-lv and ref['Vehicle'] is None);c['incremental_inside_vehicle_fov_miss']+=(oid in covered-lv and ref['Vehicle'] is not None)
   rows.append({'frame':rec['frame'],'timestamp_ns':rec['timestamp_ns'],'scene_group':rec['scene_group'],'eligible_gt':len(eligible),'candidate_edges':len(edges),'edge_components':len(comps),'accepted_pairs':len(accepted),'outputs':frame['outputs'],'confirmed_outputs':frame['confirmed_outputs'],'posthoc_unmatched_outputs':frame['posthoc_unmatched_outputs'],'unique_gt_covered':frame['unique_gt_covered'],'duplicates':frame['duplicate_confirmed_outputs'],'correct_pairs':frame['correct_associations'],'incorrect_pairs':frame['incorrect_associations'],'unverifiable_pairs':frame['unverifiable_associations'],'missed_true_merges':frame['missed_true_merges']})
 def rates(d):
  q=dict(d);q['coverage_rate']=q['unique_gt_covered']/q['eligible_gt'];q['verifiable_precision']=q['correct_associations']/max(1,q['correct_associations']+q['incorrect_associations']);q['confirmed_correct_fraction_all_accepted_pairs']=q['correct_associations']/max(1,q['correct_associations']+q['incorrect_associations']+q['unverifiable_associations']);q['confirmed_output_fraction']=q['confirmed_outputs']/q['outputs'];q['posthoc_unmatched_fraction']=q['posthoc_unmatched_outputs']/q['outputs'];return q
 selective=rates(total);selective['classwise']={k:{**dict(v),'coverage_rate':v['unique_gt_covered']/v['eligible_gt']} for k,v in sorted(classes.items())};selective['scene_groups']={str(k):dict(v) for k,v in scenes.items()}
 naive=old['conditions']['naive_concatenation'];hungarian=old['conditions']['hungarian_fusion'];duplicates_removed=naive['duplicate_confirmed_outputs']-selective['duplicate_confirmed_outputs']
 go={'coverage_at_least_1894':selective['unique_gt_covered']>=1894,'verifiable_precision_at_least_90pct':selective['verifiable_precision']>=.90,'duplicates_removed_at_least_127':duplicates_removed>=127,'correct_pairs_at_least_50':selective['correct_associations']>=50}
 result={'status':'TRAIN_ONLY_CACHED_ISOLATED_COMPONENT_SELECTIVE_FUSION','rule':'Build the exact existing class-compatible Euclidean candidate graph: 4 m for person/car and the frozen 5.5 m bus/truck exception. Merge only a 1 Vehicle x 1 RSU connected component; retain every endpoint of every other component.','inputs':{'frames':400,'cache_sha256':digest(CACHE),'iou_posthoc':IOU,'inference_rerun':False,'test_accessed':False},'eligible_population':'Same 2,956 eligible GT objects used by the frozen comparison; GT did not influence graph construction or acceptance.','candidate_graph_diagnostics':dict(diag),'selective_fusion':selective,'naive_concatenation':{k:naive[k] for k in ('outputs','confirmed_outputs','posthoc_unmatched_outputs','unique_gt_covered','duplicate_confirmed_outputs','coverage_rate','confirmed_output_fraction','posthoc_unmatched_fraction','incremental_over_vehicle','incremental_outside_vehicle_fov','incremental_inside_vehicle_fov_miss','classwise','scene_groups')},'archived_hungarian':{k:hungarian[k] for k in ('outputs','confirmed_outputs','posthoc_unmatched_outputs','unique_gt_covered','duplicate_confirmed_outputs','coverage_rate','confirmed_output_fraction','posthoc_unmatched_fraction','incremental_over_vehicle','incremental_outside_vehicle_fov','incremental_inside_vehicle_fov_miss','correct_associations','incorrect_associations','unverifiable_associations','classwise','scene_groups')},'comparison':{'coverage_loss_vs_naive':naive['unique_gt_covered']-selective['unique_gt_covered'],'duplicates_removed_vs_naive':duplicates_removed,'outputs_reduced_vs_naive':naive['outputs']-selective['outputs'],'coverage_delta_vs_hungarian':selective['unique_gt_covered']-hungarian['unique_gt_covered']},'preregistered_go_criteria':go,'decision':'GO: preserve conservative reference-pose-assisted baseline' if all(go.values()) else 'STOP: criteria not all met; do not tune this rule','warnings':['Unverifiable accepted pairs are not counted correct.','Post-hoc unmatched outputs are not confirmed false positives.','Isolated graph topology does not prove a pair is physically the same object.','This retains privileged LiDAR-derived pose and provisional Vehicle calibration; train-development only, not strict RGB-only or held-out evidence.','A proposed 3D-to-2D reprojection alternative is not evaluated here; it requires separate validation of provisional camera geometry and constructed 3D boxes.']}
 if selective['unique_gt_covered']>naive['unique_gt_covered']:raise RuntimeError('accounting error: selective coverage cannot exceed naive concatenation')
 (a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
 with (a.out/'per_frame.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 (a.out/'README.md').write_text('# Isolated-Component Selective Fusion\n\nCache-only experiment: `python3 scripts/evaluate_selective_fusion.py`. It never reruns detection, lifting, or Hungarian; GT is used only post-hoc.\n')
 print(json.dumps({'selective':{k:selective[k] for k in ('outputs','unique_gt_covered','duplicate_confirmed_outputs','correct_associations','incorrect_associations','unverifiable_associations','verifiable_precision','confirmed_correct_fraction_all_accepted_pairs')},'go':go,'decision':result['decision']},indent=2))
if __name__=='__main__':main()
