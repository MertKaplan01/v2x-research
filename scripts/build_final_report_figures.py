#!/usr/bin/env python3
"""Render final-report figures exclusively from saved measured summaries."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/v2x_final_research_report/figures';OUT.mkdir(parents=True,exist_ok=True)
def load(p): return json.loads((ROOT/p).read_text())
benefit=load('outputs/v2x_detection_benefit/summary.json');fusion=load('outputs/v2x_fusion_value/summary.json');select=load('outputs/v2x_selective_fusion/summary.json');reproj=load('outputs/v2x_reprojection_feasibility/summary.json');lift=load('outputs/v2x_vehicle_size_prior_baseline/summary.json');oracle=load('outputs/v2x_oracle_lifting_ablation/summary.json');pose=load('outputs/v2x_pose_dependency_audit/pose_sensitivity_summary.json')
plt.rcParams.update({'font.size':10,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})

# 1. Coverage and duplicate tradeoff
names=['Vehicle\nonly','Naive\nconcatenation','Hungarian','Selective'];cover=[902,1913,1833,1902];dups=[0,509,150,443];outs=[3120,7599,6704,7271]
fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(4)
ax[0].bar(x,cover,color=['#4C78A8','#59A14F','#E15759','#F28E2B']);ax[0].axhline(1913,color='#555',ls='--',lw=1,label='Oracle union')
ax[0].set_xticks(x,names);ax[0].set_ylabel('Unique eligible GT objects');ax[0].set_ylim(0,2100);ax[0].legend(frameon=False)
for i,v in enumerate(cover):ax[0].text(i,v+35,str(v),ha='center',fontweight='bold')
ax[1].bar(x,dups,color=['#4C78A8','#59A14F','#E15759','#F28E2B']);ax[1].set_xticks(x,names);ax[1].set_ylabel('Confirmed duplicate outputs');ax[1].set_ylim(0,580)
for i,v in enumerate(dups):ax[1].text(i,v+12,str(v),ha='center',fontweight='bold')
fig.suptitle('Coverage–deduplication trade-off (same 2,956-object train population)',fontweight='bold');fig.tight_layout();fig.savefig(OUT/'coverage_dedup_tradeoff.png',bbox_inches='tight');plt.close(fig)

# 2. Association outcomes
fig,ax=plt.subplots(figsize=(7.5,4));labels=['Hungarian','Selective'];correct=[235,53];wrong=[102,12];unv=[558,263];bottom=np.zeros(2)
for vals,color,name in [(correct,'#59A14F','Correct'),(wrong,'#E15759','Incorrect'),(unv,'#BAB0AC','Unverifiable')]:
 ax.bar(labels,vals,bottom=bottom,color=color,label=name);bottom+=vals
for i,t in enumerate(bottom):ax.text(i,t+12,str(int(t)),ha='center',fontweight='bold')
ax.set_ylabel('Accepted cross-view pairs');ax.set_ylim(0,1020);ax.legend(frameon=False,ncol=3,loc='upper center');ax.set_title('Accepted-pair outcome accounting (GT post-hoc)');fig.tight_layout();fig.savefig(OUT/'association_outcomes.png',bbox_inches='tight');plt.close(fig)

# 3. Reprojection by class
rows=[('Bus',79,67,79),('Car',272,251,252),('Truck',8,6,7),('Person',150,1,36)];x=np.arange(4);inside=[b/a*100 for _,a,b,c in rows];rank=[c/a*100 for _,a,b,c in rows]
fig,ax=plt.subplots(figsize=(8,4));w=.36;ax.bar(x-w/2,inside,w,label='Inside 20%-expanded box',color='#4C78A8');ax.bar(x+w/2,rank,w,label='Correct rank first',color='#F28E2B');ax.axhline(80,color='#555',ls='--',lw=1,label='Rank gate (80%)');ax.set_xticks(x,[a for a,*_ in rows]);ax.set_ylabel('True-pair rate (%)');ax.set_ylim(0,108);ax.legend(frameon=False,ncol=3,fontsize=8);ax.set_title('RSU-ground-point → Vehicle-image reprojection');
for i,(a,b) in enumerate(zip(inside,rank)):ax.text(i-w/2,a+2,f'{a:.1f}',ha='center',fontsize=8);ax.text(i+w/2,b+2,f'{b:.1f}',ha='center',fontsize=8)
fig.tight_layout();fig.savefig(OUT/'reprojection_class_disparity.png',bbox_inches='tight');plt.close(fig)

# 4. Localization evidence
fig,ax=plt.subplots(1,2,figsize=(10,4));
conds=['Bottom\ncenter','Size prior'];err=[25.5994,3.0533];ax[0].bar(conds,err,color=['#E15759','#59A14F']);ax[0].set_ylabel('Median Vehicle XY error (m)');ax[0].set_ylim(0,29);ax[0].set_title('Vehicle lifting: matched detections only');
for i,v in enumerate(err):ax[0].text(i,v+.7,f'{v:.2f}',ha='center',fontweight='bold')
labs=['Estimated\nboth','Vehicle\noracle','RSU\noracle','Both\noracle'];rec=[7,372,7,419];ax[1].bar(labs,rec,color=['#E15759','#4C78A8','#E15759','#59A14F']);ax[1].set_ylabel('Correct associations / 509');ax[1].set_ylim(0,460);ax[1].set_title('Privileged oracle-lifting ablation');
for i,v in enumerate(rec):ax[1].text(i,v+9,str(v),ha='center',fontweight='bold')
fig.suptitle('Measured localization bottleneck evidence',fontweight='bold');fig.tight_layout();fig.savefig(OUT/'localization_oracle_evidence.png',bbox_inches='tight');plt.close(fig)

manifest={'sources':{'detection_benefit':'outputs/v2x_detection_benefit/summary.json','fusion_value':'outputs/v2x_fusion_value/summary.json','selective':'outputs/v2x_selective_fusion/summary.json','reprojection':'outputs/v2x_reprojection_feasibility/summary.json','size_prior':'outputs/v2x_vehicle_size_prior_baseline/summary.json','oracle_lifting':'outputs/v2x_oracle_lifting_ablation/summary.json','pose_sensitivity':'outputs/v2x_pose_dependency_audit/pose_sensitivity_summary.json'},'figures':[p.name for p in sorted(OUT.glob('*.png'))]}
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
