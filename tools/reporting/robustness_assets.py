from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import json
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
RUN=RUN_DIR
sys.path.insert(0,str(ROOT))
from robustness_benchmark import corrupt
from evaluate import ground_truth
from data import image_paths,read_image
DATA=DATA_ROOT
gt=ground_truth(DATA,'val')
counts={im['id']:0 for im in gt['images']}
for a in gt['annotations']: counts[a['image_id']]+=1
chosen=next(im for im in gt['images'] if 5<=counts[im['id']]<=15)
path=next(p for p in image_paths(DATA,'val') if p.stem==chosen['stem'])
image=read_image(path)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
labels=[('clean','Clean'),('gaussian_5','Noise sigma=5'),('gaussian_15','Noise sigma=15'),('gaussian_30','Noise sigma=30'),('blur_1p5','Blur sigma=1.5'),('dark_0p5','Brightness x0.5')]
fig,axes=plt.subplots(2,3,figsize=(4.5,3.15),layout='constrained')
for ax,(condition,label) in zip(axes.flat,labels):
    ax.imshow(cv2.cvtColor(corrupt(image,path.stem,condition),cv2.COLOR_BGR2RGB))
    ax.set_title(label,fontsize=9,pad=3);ax.axis('off')
fig.savefig(HERE/'corruption_examples.jpg',dpi=250)
plt.close(fig)
(HERE/'corruption_example.json').write_text(json.dumps(dict(stem=path.stem,objects=counts[chosen['id']],selection='First validation image in sorted order with 5-15 ground-truth objects; independent of predictions.'),indent=2),encoding='utf-8')
if not (RUN/'robustness/summary.json').exists():
    print('Corruption examples created. Full metric figures await completion of all methods.')
    raise SystemExit(0)
summary=json.loads((RUN/'robustness/summary.json').read_text(encoding='utf-8'))
assert summary['status']=='complete'
methods=summary['methods']
names={'m1':'RF','m2':'YOLO11s','m3':'Faster R-CNN','p2-only':'P2-only','m4':'P2 + DySample'}
colors={'m1':'#888888','m2':'#2864a5','m3':'#e09d24','p2-only':'#48895c','m4':'#a44358'}
fig,ax=plt.subplots(figsize=(4.5,2.7),layout='constrained')
for m in ['m1','m2','m3','p2-only','m4']:
    keys=['clean','gaussian_5','gaussian_15','gaussian_30']
    ax.plot([0,5,15,30],[100*methods[m][k]['AP50:95'] for k in keys],marker='o',ms=3,lw=1.3,color=colors[m],label=names[m])
ax.set(xlabel='Gaussian noise standard deviation (0-255 scale)',ylabel='Validation AP50:95 (%)',xlim=(0,30),ylim=(0,52))
ax.legend(fontsize=8,ncol=2)
fig.savefig(HERE/'noise_robustness.png',dpi=250)
plt.close(fig)
aggregates={}
for m,conditions in methods.items():
    keys=['gaussian_5','gaussian_15','gaussian_30','blur_1p5','dark_0p5']
    aggregate={k:float(np.mean([conditions[c][k] for c in keys])) for k in ['AP50','AP50:95','AP-small','F1']}
    aggregate['AP95_drop_pp']=100*(conditions['clean']['AP50:95']-aggregate['AP50:95'])
    aggregate['AP95_retention_percent']=100*aggregate['AP50:95']/conditions['clean']['AP50:95']
    aggregates[m]=aggregate
(HERE/'robustness_aggregates.json').write_text(json.dumps(aggregates,indent=2),encoding='utf-8')
print(json.dumps(aggregates,indent=2))
