#!/usr/bin/env python3
"""Compare frozen TUMTraf detections: Vehicle-only, concatenation, Hungarian.

Reads the prior frozen inference cache; does not invoke YOLO, lifting, or
association. GT is used only for this post-hoc, one-to-one IoU evaluation.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import v2x_groundplane_fusion as g

CACHE = ROOT / 'outputs/v2x_detection_benefit/frozen_inference.jsonl'
PRIOR = ROOT / 'outputs/v2x_detection_benefit/summary.json'
ZIP = ROOT / 'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip'
CALIB = ROOT / 'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json'
OUT = ROOT / 'outputs/v2x_fusion_value'
IOU = .3
CONDITIONS = ('vehicle_only', 'naive_concatenation', 'hungarian_fusion')

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def condition_outputs(name, local, remote, matches, la, rb):
    if name == 'vehicle_only':
        return [{'kind':'vehicle','ids':[la[i]] if i in la else []} for i in range(len(local))], Counter()
    if name == 'naive_concatenation':
        x=[{'kind':'vehicle','ids':[la[i]] if i in la else []} for i in range(len(local))]
        x += [{'kind':'rsu','ids':[rb[j]] if j in rb else []} for j in range(len(remote))]
        return x, Counter()
    paired_l={m['local_index'] for m in matches}; paired_r={m['remote_index'] for m in matches}
    x=[]; q=Counter()
    for m in matches:
        a,b=la.get(m['local_index']),rb.get(m['remote_index'])
        if a is not None and b is not None and a != b:
            # One operational fused output combines distinct known objects:
            # do not credit either object to coverage.
            x.append({'kind':'incorrect_merge','ids':[]}); q['incorrect_associations']+=1
        elif a is not None or b is not None:
            x.append({'kind':'merged','ids':[a if a is not None else b]})
            if a is not None and b is not None: q['correct_associations']+=1
            else: q['unverifiable_associations']+=1
        else:
            x.append({'kind':'unmatched_merge','ids':[]}); q['unverifiable_associations']+=1
    for i in range(len(local)):
        if i not in paired_l: x.append({'kind':'vehicle_unmatched','ids':[la[i]] if i in la else []})
    for j in range(len(remote)):
        if j not in paired_r: x.append({'kind':'rsu_unmatched','ids':[rb[j]] if j in rb else []})
    return x,q

def add_metrics(dst, outputs, eligible, lv, qa):
    ids=[oid for o in outputs for oid in o['ids']]
    covered=set(ids); dup=sum(max(0,n-1) for n in Counter(ids).values())
    inc=covered-lv
    dst.update({'outputs':len(outputs),'confirmed_outputs':len(ids),
                'posthoc_unmatched_outputs':len(outputs)-len(ids),
                'unique_gt_covered':len(covered),'duplicate_confirmed_outputs':dup,
                'incremental_over_vehicle':len(inc),
                'incremental_outside_vehicle_fov':sum(eligible[x]['Vehicle'] is None for x in inc),
                'incremental_inside_vehicle_fov_miss':sum(eligible[x]['Vehicle'] is not None for x in inc)})
    dst.update(qa)
    return covered

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out', type=Path, default=OUT); a=ap.parse_args()
    for p in (CACHE, PRIOR, ZIP, CALIB):
        if not p.is_file(): raise FileNotFoundError(p)
    a.out.mkdir(parents=True, exist_ok=True)
    records=[json.loads(x) for x in CACHE.read_text().splitlines() if x]
    if len(records)!=400 or any(not x['label_member'].startswith('train/') for x in records):
        raise RuntimeError('STOP: cache is not the expected 400-frame train-only record set')
    prior=json.loads(PRIOR.read_text()); rc=json.loads(CALIB.read_text())
    B=np.asarray(rc['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float)
    Ps=np.asarray(rc['projection_from_s110_lidar_ouster_south'],float)
    totals={c:Counter() for c in CONDITIONS}; classes={c:defaultdict(Counter) for c in CONDITIONS}; scenes={c:defaultdict(Counter) for c in CONDITIONS}; rows=[]
    common=Counter()
    with ZipFile(ZIP) as z:
        for rec in records:
            doc=json.loads(z.read(rec['label_member'])); T=g.read_transform(doc); Pv=g.P_VEHICLE_LIDAR@np.linalg.inv(T)
            eligible={x['id']:x for x in g.reference_objects(doc,B,Pv,Ps) if x['Vehicle'] is not None or x['South2'] is not None}
            local,remote=rec['vehicle'],rec['rsu']; la=g.gt_match(local,list(eligible.values()),'Vehicle',IOU); rb=g.gt_match(remote,list(eligible.values()),'South2',IOU)
            lv,rv=set(la.values()),set(rb.values()); common.update({'frames':1,'eligible_gt':len(eligible),'vehicle_gt_detected':len(lv),'rsu_gt_detected':len(rv),'oracle_union_gt':len(lv|rv),'both_detected':len(lv&rv),'vehicle_only_detected':len(lv-rv),'rsu_only_detected':len(rv-lv),'neither_detected':len(set(eligible)-lv-rv)})
            for name in CONDITIONS:
                outputs,qa=condition_outputs(name,local,remote,rec['matches'],la,rb)
                frame=Counter({'eligible_gt':len(eligible)}); covered=add_metrics(frame,outputs,eligible,lv,qa); totals[name].update(frame)
                sg=scenes[name][rec['scene_group']]; sg.update(frame)
                for oid,ref in eligible.items():
                    cc=classes[name][ref['label']]; cc['eligible_gt']+=1; cc['unique_gt_covered']+=oid in covered; cc['vehicle_fov_incremental_gt']+=(oid in covered-lv and ref['Vehicle'] is not None); cc['outside_fov_incremental_gt']+=(oid in covered-lv and ref['Vehicle'] is None)
                rows.append({'frame':rec['frame'],'timestamp_ns':rec['timestamp_ns'],'scene_group':rec['scene_group'],'condition':name,'eligible_gt':len(eligible),'outputs':frame['outputs'],'confirmed_outputs':frame['confirmed_outputs'],'posthoc_unmatched_outputs':frame['posthoc_unmatched_outputs'],'unique_gt_covered':frame['unique_gt_covered'],'duplicates':frame['duplicate_confirmed_outputs'],'incremental_over_vehicle':frame['incremental_over_vehicle'],'incremental_outside_vehicle_fov':frame['incremental_outside_vehicle_fov'],'incremental_inside_vehicle_fov_miss':frame['incremental_inside_vehicle_fov_miss'],'correct_associations':frame['correct_associations'],'incorrect_associations':frame['incorrect_associations'],'unverifiable_associations':frame['unverifiable_associations']})
    def pack(x):
        d=dict(x); d['coverage_rate']=d['unique_gt_covered']/common['eligible_gt']; d['confirmed_output_fraction']=d['confirmed_outputs']/d['outputs']; d['posthoc_unmatched_fraction']=d['posthoc_unmatched_outputs']/d['outputs']; return d
    result={name:pack(totals[name]) for name in CONDITIONS}
    if result['hungarian_fusion']['unique_gt_covered']!=prior['counts']['operational_fused_gt_covered']:
        raise RuntimeError('STOP: Hungarian coverage does not reconcile with frozen detection-benefit result')
    if result['naive_concatenation']['unique_gt_covered']!=common['oracle_union_gt']:
        raise RuntimeError('STOP: naive coverage does not reconcile with oracle union')
    for name in CONDITIONS:
        result[name]['classwise']={k:{**dict(v),'coverage_rate':v['unique_gt_covered']/v['eligible_gt']} for k,v in sorted(classes[name].items())}
        result[name]['scene_groups']={str(k):dict(v) for k,v in scenes[name].items()}
    trade={'outputs_reduced_vs_naive':result['naive_concatenation']['outputs']-result['hungarian_fusion']['outputs'],
           'duplicates_reduced_vs_naive':result['naive_concatenation']['duplicate_confirmed_outputs']-result['hungarian_fusion']['duplicate_confirmed_outputs'],
           'gt_coverage_lost_vs_naive':result['naive_concatenation']['unique_gt_covered']-result['hungarian_fusion']['unique_gt_covered'],
           'confirmed_output_instances_lost_vs_naive':result['naive_concatenation']['confirmed_outputs']-result['hungarian_fusion']['confirmed_outputs']}
    out={'status':'TRAIN_ONLY_CACHED_FUSION_VALUE_COMPARISON','inputs':{'frames':400,'cache_sha256':sha(CACHE),'iou_posthoc':IOU,'test_accessed':False,'inference_rerun':False},'eligible_population':'Supported-class GT objects whose cuboid projects into at least one camera; identical population for A/B/C. GT is post-hoc only.','common_detection_cases':dict(common),'conditions':result,'tradeoff_hungarian_vs_naive':trade,'reconciliation':{'hungarian_coverage':result['hungarian_fusion']['unique_gt_covered'],'prior_frozen_coverage':prior['counts']['operational_fused_gt_covered'],'naive_coverage':result['naive_concatenation']['unique_gt_covered'],'prior_oracle_union':prior['counts']['oracle_union_gt_covered']},'warnings':['Post-hoc unmatched outputs are explicitly not asserted to be confirmed false positives; annotation, class, or IoU-matching limitations remain possible.','Confirmed-output fraction is a limited post-hoc proxy, not fully verified detection precision.','Naive concatenation has no cross-view association, so it cannot have an incorrect cross-view merge; its duplicate count represents confirmed repeated object outputs across views.','Results use correlated train-development frames and privileged reference pose in the frozen lifting path; they do not establish strict camera-only or held-out performance.']}
    (a.out/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    with (a.out/'per_frame_comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (a.out/'README.md').write_text('# Frozen fusion-value comparison\n\nNo detector, lifting, or association computation is run. `python3 scripts/evaluate_fusion_value.py` evaluates the cached 400-frame train-only records against GT only after operational records were frozen.\n')
    print(json.dumps({'common':dict(common),'conditions':{k:{x:result[k][x] for x in ('outputs','unique_gt_covered','duplicate_confirmed_outputs','confirmed_output_fraction','posthoc_unmatched_outputs')} for k in CONDITIONS},'tradeoff':trade},indent=2))
if __name__=='__main__': main()
