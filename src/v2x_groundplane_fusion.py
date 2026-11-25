#!/usr/bin/env python3
"""Experimental GT-blind two-camera late fusion on TUMTraf-V2X R4 Mini.

Vehicle+South2 YOLO11s post-NMS 2D detections -> estimated object footpoints on
shared s110_base road plane -> class/group-aware matching -> RSU-only additions.
NO GT object or GT object ID is consulted in the fusion algorithm.
Frame-wise LiDAR registration matrices are read from OpenLABEL *metadata*
(simulating vehicle localization), and projected 3D GT is used ONLY AFTERWARD
in a separate diagnostic evaluator. Vehicle camera matrix is PROVISIONAL.
Not a certified calibration, real 3D detector, nor a safety mechanism.

Requires: numpy, opencv-python, matplotlib (Agg), R4 Mini zip, devkit South2 calib.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import Counter
from itertools import product
from pathlib import Path
from zipfile import ZipFile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / 'outputs/tumtraf_two_camera/paired_predictions.jsonl'
DATA = ROOT / 'data/tumtraf_r4_mini/tumtraf_v2x_cooperative_perception_dataset_mini.zip'
CALIB = ROOT / 'external/tumtraf-devkit/calib/s110_camera_basler_south2_8mm.json'
OUT = ROOT / 'outputs/v2x_groundplane_fusion'

# Provisional CoopDet3D vehicle lidar -> vehicle camera projection. VERIFY IT.
# https://github.com/tum-traffic-dataset/coopdet3d/blob/master/tools/data_converter/tumtraf_v2x_converter.py
P_VEHICLE_LIDAR = np.array([
    [1019.929965441548, -2613.286262078907, 184.6794570200418, 370.7180273597151],
    [589.8963703919744, -24.09642935106967, -2623.908527352794, -139.3143336725661],
    [0.9841844439506531, 0.1303769648075104, 0.1199281811714172, -0.1664766669273376],
], dtype=np.float64)
LABEL_MAP = {'CAR': 'car', 'BUS': 'bus', 'TRUCK': 'truck', 'PEDESTRIAN': 'person'}
COLORS = {'car': (60, 200, 240), 'bus': (90, 220, 70), 'truck': (80, 150, 245), 'person': (220, 70, 210)}


def time_ns(name: str) -> int:
    a, b = Path(name).name.split('_', 2)[:2]
    return int(a) * 10**9 + int(b.ljust(9, '0')[:9])


def read_transform(doc):
    frame = next(iter(doc['openlabel']['frames'].values()))
    val = frame['frame_properties']['transforms'][
        'vehicle_lidar_robosense_to_s110_lidar_ouster_south'
    ]['transform_src_to_dst']['matrix4x4']
    T = np.asarray(val, dtype=float).reshape(4, 4)
    if not np.all(np.isfinite(T)) or not np.allclose(T[3], [0,0,0,1], atol=1e-3):
        raise ValueError('Bad per-frame vehicle/RSU transform')
    if abs(np.linalg.det(T[:3,:3])) < 0.5:
        raise ValueError('Invalid per-frame transform rotation')
    return T


def make_ground_plane(T_lidar_base, ground_height_base=0.):
    """Plane n.X+d=0 in infrastructure LiDAR coordinates."""
    row = T_lidar_base[2].astype(float)
    n, d = row[:3], float(row[3]-ground_height_base)
    return n, d


def foot_to_world(P, uv, plane, T_lidar_base, max_range_m=180.):
    """Intersect a camera pixel ray with the assumed common ground plane.
    Returns coordinates of the intersection in s110_base, or None.
    """
    u, v = map(float, uv)
    n, d = plane
    A = np.stack([u*P[2,:3]-P[0,:3], v*P[2,:3]-P[1,:3], n])
    b = np.array([P[0,3]-u*P[2,3], P[1,3]-v*P[2,3], -d])
    if not np.all(np.isfinite(A)) or np.linalg.cond(A) > 1e8:
        return None
    xyz = np.linalg.solve(A,b)
    depth = P[2,:3] @ xyz + P[2,3]
    if not np.all(np.isfinite(xyz)) or depth <= 1e-4:
        return None
    pixel = P @ np.r_[xyz, 1.0]
    reprojected = pixel[:2] / pixel[2]
    if np.linalg.norm(reprojected-np.array([u,v])) > 1e-2:
        return None
    pbase = T_lidar_base @ np.r_[xyz,1.0]
    if abs(pbase[3]) < 1e-8 or np.linalg.norm(pbase[:2]) > max_range_m:
        return None
    return [float(pbase[0]/pbase[3]), float(pbase[1]/pbase[3])]


def base_to_camera(P, T_lidar_base, xy, ground_height_base=0.):
    xyz_base = np.array([xy[0], xy[1], ground_height_base, 1.],float)
    xyz_lidar = np.linalg.inv(T_lidar_base) @ xyz_base
    q = P @ xyz_lidar
    if not np.all(np.isfinite(q)) or q[2] <= 1e-4:
        return None
    return [float(q[0]/q[2]),float(q[1]/q[2])]


def iou(box1, box2):
    a,b,c,d = map(float,box1); e,f,g,h = map(float,box2)
    overlap = max(0.,min(c,g)-max(a,e))*max(0.,min(d,h)-max(b,f))
    union = max(0.,(c-a)*(d-b))+max(0.,(g-e)*(h-f))-overlap
    return overlap/union if union>0 else 0.


def group(label):
    return 'person' if label=='person' else 'vehicle'


def dedup(raw, conf, overlap=0.70):
    """Within-view class-agnostic NMS for conflicting bus/truck/car boxes."""
    s=sorted((dict(o) for o in raw if o['label'] in COLORS and float(o['confidence'])>=conf),
             key=lambda x:-float(x['confidence']))
    keep=[]
    for d in s:
        if any(group(d['label'])==group(k['label']) and iou(d['bbox_xyxy'],k['bbox_xyxy'])>=overlap for k in keep):
            continue
        keep.append(d)
    return keep


def project_observations(msg, P, plane, Tbase, threshold, max_range_m):
    results=[]
    for obj in dedup(msg['detections'],threshold):
        x1,y1,x2,y2 = obj['bbox_xyxy']
        uv=[0.5*(x1+x2), y2]  # bottom-center point is an approximation
        xy=foot_to_world(P,uv,plane,Tbase,max_range_m)
        results.append(dict(label=obj['label'],confidence=float(obj['confidence']),
                            bbox_xyxy=list(map(float,obj['bbox_xyxy'])),
                            base_xy=xy, bottom_pixel=uv))
    return results


def association_gate(a,b,gate_m):
    if group(a['label'])!=group(b['label']):
        return -1.
    if a['base_xy'] is None or b['base_xy'] is None:
        return -1.
    dist=float(np.linalg.norm(np.asarray(a['base_xy'])-np.asarray(b['base_xy'])))
    extra=1.5 if a['label'] in ('bus','truck') or b['label'] in ('bus','truck') else 0.
    limit=gate_m+extra
    return dist if dist<=limit else -1.


def fuse(local,remote,gate_m=4.):
    """Greedy distance association. No GT, ID, label file, image in this function."""
    # The receiver sees only two projected detection streams, not reference objects.
    used=set(); fused=[]; remote_only=[]; matched=[]; invalid=0
    for j,r in enumerate(remote):
        if r['base_xy'] is None:
            invalid+=1
            continue
        candidates=[]
        for i,a in enumerate(local):
            if i in used: continue
            dist=association_gate(a,r,gate_m)
            if dist>=0: candidates.append((dist,i))
        if candidates:
            dist,i=min(candidates)
            used.add(i)
            matched.append({'local_index':i,'remote_index':j,'distance_m':round(dist,2),
                            'vehicle_label':local[i]['label'],'rsu_label':r['label']})
        elif not any(association_gate(other,r,2.)>=0 for other in remote_only):
            obj=dict(r);obj['remote_index']=j
            remote_only.append(obj)
    fused.extend({'source':'Vehicle','label':a['label'], 'confidence':a['confidence'],
                  'base_xy':a['base_xy']} for a in local if a['base_xy'] is not None)
    fused.extend({'source':'RSU_only','label':r['label'], 'confidence':r['confidence'],
                  'base_xy':r['base_xy']} for r in remote_only)
    return dict(fused=fused,remote_only=remote_only,matched=matched,
                invalid_rsu_ray_count=invalid)


def corners_from_cuboid(values):
    a=np.asarray(values,float)
    if a.size!=10 or not np.all(np.isfinite(a)) or np.any(a[7:]<=0):
        return None
    x,y,z,w=a[3:7]; norm=np.linalg.norm([x,y,z,w])
    if norm<1e-9:return None
    x,y,z,w=np.array([x,y,z,w])/norm
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    d=a[7:]
    offsets=np.array(list(product((-1.,1.),repeat=3)))*d/2
    return offsets@R.T+a[:3]


def project_gt_box(corners,P,width,height):
    q=(P@np.c_[corners,np.ones(8)].T).T
    if np.any(q[:,2]<=1e-5):return None
    uv=q[:,:2]/q[:,2,None]
    if not np.all(np.isfinite(uv)):return None
    xy1=np.maximum(uv.min(axis=0),[0.,0.]);xy2=np.minimum(uv.max(axis=0),[width-1.,height-1.])
    if np.any(xy2<=xy1):return None
    return [*xy1,*xy2]


def reference_objects(doc,Tbase,Pv,Ps,vehicle_wh=(1920,1200),rsu_wh=(1920,1200)):
    """GT diagnostic: called ONLY after fused outputs are finalized."""
    frame=next(iter(doc['openlabel']['frames'].values()))
    refs=[]
    for oid,entry in frame.get('objects',{}).items():
        data=entry.get('object_data',{}); kind=LABEL_MAP.get(data.get('type',''))
        if kind is None:continue
        c=data.get('cuboid');c=c[0] if isinstance(c,list) and c else c
        if not isinstance(c,dict):continue
        sensor=next((a.get('val','') for a in c.get('attributes',{}).get('text',[])
                     if a.get('name')=='sensor_id'), '')
        if sensor!='s110_lidar_ouster_south':continue
        coords=c.get('val',[]);corners=corners_from_cuboid(coords)
        if corners is None:continue
        centre=(Tbase@np.r_[np.asarray(coords[:3],float),1.])[:2].tolist()
        refs.append(dict(id=oid,label=kind,base_xy=centre,
                         Vehicle=project_gt_box(corners,Pv,*vehicle_wh),
                         South2=project_gt_box(corners,Ps,*rsu_wh)))
    return refs


def gt_match(obs,refs,cam,iou_min):
    """One-to-one class+IoU GT assignment for evaluation only."""
    pairs=[]
    for i,d in enumerate(obs):
        for ref in refs:
            if d['label']!=ref['label'] or ref[cam] is None:continue
            v=iou(d['bbox_xyxy'],ref[cam])
            if v>=iou_min:pairs.append((v,i,ref['id']))
    pairs.sort(reverse=True)
    used_obs=set();used_gt=set(); matches={}
    for score,i,oid in pairs:
        if i in used_obs or oid in used_gt:continue
        used_obs.add(i);used_gt.add(oid);matches[i]=oid
    return matches


def evaluate(local,remote,combined,refs,iou_min=0.3):
    la=gt_match(local,refs,'Vehicle',iou_min)
    rb=gt_match(remote,refs,'South2',iou_min)
    local_ids=set(la.values())
    added_matches={r['remote_index']:rb[r['remote_index']]
                   for r in combined['remote_only'] if r['remote_index'] in rb}
    recovered=set(added_matches.values())-local_ids
    # Verify that recovered 2D GT matches have a remotely estimated ground
    # position close to GT centre (loose, class-dependent distance thresholds).
    lookup={o['id']:o for o in refs}
    recover_xy=[]
    for item in combined['remote_only']:
        oid=added_matches.get(item['remote_index'])
        if oid is None or oid not in recovered:continue
        gt=lookup[oid]; dist=float(np.linalg.norm(np.asarray(item['base_xy'])-np.asarray(gt['base_xy'])))
        cap=3. if gt['label']=='person' else (8. if gt['label'] in ('bus','truck') else 5.)
        if dist<=cap:recover_xy.append(oid)
    return dict(local_matched=len(local_ids),rsu_matched=len(set(rb.values())),
                added_2d_gt_proxy=len(set(added_matches.values())),
                rescued_proxy_iou=len(recovered),
                rescued_proxy_iou_and_ground=len(set(recover_xy)),
                example_rescued_ids=sorted(recovered)[:8],
                unmatched_rsu_predictions=len(combined['remote_only'])-len(added_matches))


def draw_examples(zipf,rec,local,remote,combined,Pvehicle,Tbase,outpath):
    import cv2
    panels=[]
    remote_only_ids={r['remote_index'] for r in combined['remote_only']}
    for camera,objects in [('Vehicle',local),('South2',remote)]:
        name=rec['image_paths'][camera]
        img=cv2.imdecode(np.frombuffer(zipf.read(name),dtype=np.uint8),cv2.IMREAD_COLOR)
        if img is None:raise RuntimeError('Failed image decode '+name)
        for det_i,det in enumerate(objects):
            added=(camera=='South2' and det_i in remote_only_ids)
            col=(255,0,255) if added else COLORS[det['label']]
            x1,y1,x2,y2=map(int,det['bbox_xyxy'])
            cv2.rectangle(img,(x1,y1),(x2,y2),col,4 if added else 2)
            caption=('RSU-only ' if added else '')+f"{det['label']} {det['confidence']:.2f}"
            cv2.putText(img,caption,(x1,max(26,y1-8)),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
        if camera=='Vehicle':
            for r in combined['remote_only']:
                uv=base_to_camera(Pvehicle,Tbase,r['base_xy'])
                if uv is None:continue
                u,v=uv
                if 0<=u<img.shape[1] and 0<=v<img.shape[0]:
                    cv2.drawMarker(img,(int(u),int(v)),(255,0,255),cv2.MARKER_CROSS,34,4)
                    cv2.putText(img,'RSU footpoint (approx)',(max(1,int(u)-120),max(20,int(v)-20)),
                                cv2.FONT_HERSHEY_SIMPLEX,0.52,(255,0,255),2)
        panel=cv2.resize(img,(768,480),interpolation=cv2.INTER_AREA)
        cv2.rectangle(panel,(0,0),(767,42),(20,20,20),-1)
        cv2.putText(panel,f"{camera} | {len(objects)} detections",(12,29),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,255),2)
        panels.append(panel)
    canvas=np.hstack(panels)
    footer=np.full((85,canvas.shape[1],3),27,dtype=np.uint8)
    legend=(f"Pair {rec['pair']:02d} | GT-BLIND ground-plane fusion | "
            f"matched={len(combined['matched'])}, RSU-only={len(combined['remote_only'])}, "
            f"invalid rays={combined['invalid_rsu_ray_count']}")
    cv2.putText(footer,legend,(14,30),cv2.FONT_HERSHEY_SIMPLEX,0.62,(255,255,255),2)
    cv2.putText(footer,'Pink crosses are approximate RSU ground points, NOT camera-detected 2D boxes.',
                (14,63),cv2.FONT_HERSHEY_SIMPLEX,0.58,(160,170,245),2)
    cv2.imwrite(str(outpath),np.vstack((canvas,footer)))


def plot_world(rec,local,remote,combined,outpath):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.2,6.4))
    for obs,col,label in [(local,'tab:blue','Vehicle'),(remote,'tab:orange','South2')]:
        vals=np.array([x['base_xy'] for x in obs if x['base_xy'] is not None])
        if len(vals):ax.scatter(vals[:,0],vals[:,1],c=col,s=30,alpha=.55,label=label)
    added=np.array([x['base_xy'] for x in combined['remote_only']])
    if len(added):ax.scatter(added[:,0],added[:,1],marker='x',c='magenta',s=95,label='RSU-only hypotheses')
    ax.set(xlabel='s110_base X (m)',ylabel='s110_base Y (m)',title=f"Pair {rec['pair']:02d} — assumed flat ground, provisional vehicle calib")
    ax.grid(alpha=.25);ax.axis('equal');ax.legend(loc='best')
    fig.tight_layout();fig.savefig(outpath,dpi=160);plt.close(fig)


def self_test():
    # Synthetic camera sees plane z=0 with depth constant 5m.
    P=np.array([[100.,0,0,0],[0,100.,0,0],[0,0,0,5]],float)
    T=np.eye(4);plane=(np.array([0.,0.,1.]),0.)
    # World (2,3,0) projects to pixel (40,60).
    pt=foot_to_world(P,[40.,60.],plane,T)
    assert pt is not None and np.allclose(pt,[2,3],atol=1e-6),pt
    assert np.allclose(base_to_camera(P,T,pt),[40,60],atol=1e-6)
    loc=[dict(label='bus',confidence=.9,base_xy=[2.,3.])]
    rem=[dict(label='bus',confidence=.8,base_xy=[2.4,2.9]),
         dict(label='person',confidence=.8,base_xy=[4.,1.])]
    f=fuse(loc,rem);assert len(f['matched'])==1 and len(f['remote_only'])==1, f
    assert iou([0,0,10,10],[5,5,15,15])>0.
    print('SELF-TEST OK: plane projection, inverse projection, association, unique RSU addition')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',type=Path,default=PRED)
    parser.add_argument('--zip',type=Path,default=DATA)
    parser.add_argument('--calib',type=Path,default=CALIB)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--pairs',type=int,nargs='*',help='Optional pair IDs, e.g. --pairs 1 4')
    parser.add_argument('--conf',type=float,default=.25)
    parser.add_argument('--ground-height-base-m',type=float,default=0.)
    parser.add_argument('--max-range-m',type=float,default=180.)
    parser.add_argument('--match-radius-m',type=float,default=4.)
    parser.add_argument('--evaluation-iou',type=float,default=.3)
    parser.add_argument('--no-eval',action='store_true',help='Run no object GT-based checks')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    for path in [args.predictions,args.zip,args.calib]:
        if not path.is_file():raise FileNotFoundError(path)
    args.out.mkdir(parents=True,exist_ok=True)
    calib=json.loads(args.calib.read_text())
    Ps=np.asarray(calib['projection_from_s110_lidar_ouster_south'],float)
    Tbase=np.asarray(calib['transformation_matrix_s110_lidar_ouster_south_to_s110_base'],float)
    if Ps.shape!=(3,4) or Tbase.shape!=(4,4):raise ValueError('Unexpected calibration shape')
    plane=make_ground_plane(Tbase,args.ground_height_base_m)
    rows=[]; summ=[]; preview_pairs={1,4,10,20,47,50}
    recs=[json.loads(line) for line in args.predictions.open() if line.strip()]
    if args.pairs is not None:recs=[r for r in recs if int(r['pair']) in set(args.pairs)]
    if not recs:raise RuntimeError('No matching pairs')
    with ZipFile(args.zip) as z:
        labels=sorted((time_ns(x),x) for x in z.namelist() if x.startswith('val/labels_point_clouds/') and x.endswith('.json'))
        if not labels:raise RuntimeError('No label files: per-frame registration metadata is needed')
        for rec in recs:
            idx=int(rec['pair']); ts=int(rec['streams']['South2']['capture_timestamp_ns'])
            label_ts,label_path=min(labels,key=lambda x:abs(x[0]-ts))
            doc=json.loads(z.read(label_path))
            Tvehicle=read_transform(doc)
            Pv=P_VEHICLE_LIDAR@np.linalg.inv(Tvehicle)
            local=project_observations(rec['streams']['Vehicle'],Pv,plane,Tbase,args.conf,args.max_range_m)
            remote=project_observations(rec['streams']['South2'],Ps,plane,Tbase,args.conf,args.max_range_m)
            result=fuse(local,remote,args.match_radius_m)
            result['pair']=idx;result['vehicle_ts_ns']=rec['streams']['Vehicle']['capture_timestamp_ns']
            result['south2_ts_ns']=ts;result['time_delta_ms']=rec['offset_ms']
            # Evaluation below cannot change any item in result: NO reference/GT input to fuse().
            diagnostic={}
            if not args.no_eval:
                gt=reference_objects(doc,Tbase,Pv,Ps,
                    vehicle_wh=(1920,1200),rsu_wh=(int(calib['image_width']),int(calib['image_height'])))
                diagnostic=evaluate(local,remote,result,gt,args.evaluation_iou)
            row=dict(pair=idx,offset_ms=rec['offset_ms'],label_offset_ms=round((label_ts-ts)/1e6,2),
                     local_count=len(local),remote_count=len(remote),
                     vehicle_invalid=sum(d['base_xy'] is None for d in local),
                     remote_invalid=result['invalid_rsu_ray_count'],
                     spatial_matches=len(result['matched']),remote_only=len(result['remote_only']),
                     **{k:v for k,v in diagnostic.items() if isinstance(v,(int,float))})
            rows.append(row);summ.append({'result':result,'evaluation':diagnostic})
            print(f"pair={idx:02d} | Vehicle={len(local)} RSU={len(remote)} "
                  f"matched={len(result['matched'])} RSU-only={len(result['remote_only'])} "
                  f"rescue_proxy={diagnostic.get('rescued_proxy_iou','n/a')} "
                  f"XY_checked={diagnostic.get('rescued_proxy_iou_and_ground','n/a')}")
            if idx in preview_pairs or args.pairs is not None:
                try:
                    draw_examples(z,rec,local,remote,result,Pv,Tbase,args.out/f'pair_{idx:03d}_fusion.jpg')
                    plot_world(rec,local,remote,result,args.out/f'pair_{idx:03d}_bev.png')
                except Exception as e:
                    print(f'Preview for pair {idx} skipped: {e}')
    fields=list(rows[0].keys())
    with (args.out/'fusion_per_frame.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=fields);wr.writeheader();wr.writerows(rows)
    with (args.out/'fused_objects.jsonl').open('w') as f:
        for item in summ:f.write(json.dumps(item['result'],ensure_ascii=False)+'\n')
    overview={
       'status':'GT_BLIND_EXPERIMENTAL_GROUND_PLANE_LATE_FUSION',
       'pairs':len(rows),'confidence':args.conf,'match_radius_m':args.match_radius_m,
       'ground_height_base_m_assumed':args.ground_height_base_m,
       'sum_spatial_matches':sum(r['spatial_matches'] for r in rows),
       'sum_rsu_only_hypotheses':sum(r['remote_only'] for r in rows),
       'sum_invalid_local_footpoints':sum(r['vehicle_invalid'] for r in rows),
       'sum_invalid_remote_footpoints':sum(r['remote_invalid'] for r in rows),
       'diagnostic_rescued_gt_proxy_iou':sum(r.get('rescued_proxy_iou',0) for r in rows) if not args.no_eval else None,
       'diagnostic_rescued_gt_proxy_iou_and_ground':sum(r.get('rescued_proxy_iou_and_ground',0) for r in rows) if not args.no_eval else None,
       'notes':[
          'Receiver input = YOLO11s 2D detection boxes + official RSU camera calibration + per-frame vehicle/LiDAR registration.',
          'No GT object IDs or object positions are used in inference, object matching, or creation of RSU-only hypotheses.',
          'Registration transforms come from OpenLABEL frame metadata, standing in for external localization; real V2X would require live pose estimation.',
          'The vehicle camera projection is provisional/CoopDet3D. This is NOT verified calibrated geometry.',
          'Single-view bottom-center ray intersection ASSUMES flat road at configured s110_base height; bbox feet can be wrong due to depth and occlusion.',
          'Spatial matching uses a heuristic ground-plane gate; verify duplicates/false merges visually. RSU-only is a hypothesis, NOT a verified recovered object.',
          'Diagnostic matched/reference counts use incomplete GT projections, do not measure standard precision/recall/mAP.',
          'Camera images offset in time; scene motion and registration uncertainty cause errors.',
          'NO intermediate feature fusion, ROS2 transport, or wireless measurements are performed here.',
       ]}
    (args.out/'fusion_summary.json').write_text(json.dumps(overview,indent=2))
    print('\nSaved:',args.out)
    print('REMOTE-ONLY HYPOTHESES:',overview['sum_rsu_only_hypotheses'])
    print('GT-assisted proxy rescued:',overview['diagnostic_rescued_gt_proxy_iou'],
          'XY-checked:',overview['diagnostic_rescued_gt_proxy_iou_and_ground'])

if __name__=='__main__':main()
