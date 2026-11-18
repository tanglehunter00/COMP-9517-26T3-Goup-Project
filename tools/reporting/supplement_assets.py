from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUN=RUN_DIR/'controlled_diagnostics'
summary=json.loads((RUN/'summary.json').read_text(encoding='utf-8'))
assert summary['status']=='complete' and summary['seed']==42
c=summary['conditions']
sys.path.insert(0,str(ROOT))
from controlled_diagnostics import xywh_iou
from evaluate import ground_truth
gt=ground_truth(DATA_ROOT,'val')
boxes={im['id']:[] for im in gt['images']}
for a in gt['annotations']: boxes[a['image_id']].append(a['bbox'])
errors={}
for suffix in [30,50,60,70,90]:
    key=f'm4_s640_n{suffix}'
    predictions=json.loads((RUN/key/'predictions.json').read_text())
    grouped={i:[] for i in boxes}
    for p in predictions:
        if p['score']>=.25: grouped[p['image_id']].append(p)
    counts=dict(duplicate=0,localization=0,background=0)
    for image_id,rows in grouped.items():
        used=set()
        for row in sorted(rows,key=lambda p:-p['score']):
            overlaps=xywh_iou(row['bbox'],boxes[image_id])
            possible=[i for i in range(len(overlaps)) if i not in used and overlaps[i]>=.5]
            if possible: used.add(max(possible,key=lambda i:overlaps[i]))
            else:
                overlap=float(overlaps.max()) if len(overlaps) else 0
                counts['duplicate' if overlap>=.5 else 'localization' if overlap>=.1 else 'background']+=1
    assert sum(counts.values())==c[key]['FP']
    errors[str(suffix)]=counts
(RUN/'nms_error_types.json').write_text(json.dumps(errors,indent=2),encoding='utf-8')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
colors={'m2':'#2864a5','m3':'#e09d24','p2-only':'#48895c','m4':'#a44358'}
names={'m2':'YOLO11s','m3':'Faster R-CNN','p2-only':'P2-only','m4':'P2 + DySample'}
fig,ax=plt.subplots(figsize=(4.5,2.7),layout='constrained')
for m,values in summary['thresholds'].items():
    grid=values['grid']
    ax.plot([r['threshold'] for r in grid],[100*r['F1'] for r in grid],label=names[m],color=colors[m],lw=1.4)
    best=values['selected']
    ax.scatter(best['threshold'],100*best['F1'],color=colors[m],s=20)
ax.axvline(.25,color='gray',linestyle=':',lw=1)
ax.set(xlabel='Confidence threshold',ylabel='Validation F1 (%)',xlim=(.05,.95))
ax.legend(fontsize=8,ncol=2,loc='lower center')
fig.savefig(HERE/'threshold_sweep.png',dpi=240)
plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(4.5,2.7),layout='constrained')
for m in ['m2','p2-only','m4']:
    rows=[c[f'{m}_s640_n{i}'] for i in [30,50,60,70,90]]
    x=[r['nms_iou'] for r in rows]
    axes[0].plot(x,[100*r['AP50'] for r in rows],marker='o',ms=3,label=names[m],color=colors[m])
    axes[1].plot(x,[r['FP'] for r in rows],marker='o',ms=3,color=colors[m])
for ax in axes:
    ax.set_xlabel('NMS IoU')
    ax.axvline(.6,color='gray',linestyle=':',lw=1)
axes[0].set_ylabel('Validation AP50 (%)')
axes[1].set_ylabel('FP at score >= 0.25')
axes[0].legend(fontsize=7)
fig.savefig(HERE/'nms_sweep.png',dpi=240)
plt.close(fig)
print('Created validation threshold and NMS sweep figures.')
