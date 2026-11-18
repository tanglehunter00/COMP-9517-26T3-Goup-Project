from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import csv, gzip, json
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

RUNS=RUN_DIR
OUT=Path(__file__).resolve().parent
DATA=DATA_ROOT
METHODS=['m1','m2','m3','p2-only','m4']
NAMES={'m1':'RF','m2':'YOLO11s','m3':'Faster R-CNN','p2-only':'P2-only','m4':'P2 + DySample'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})

def ious(a,b):
    if not len(a) or not len(b): return np.zeros((len(a),len(b)))
    a=np.asarray(a); b=np.asarray(b)
    lo=np.maximum(a[:,None,:2],b[None,:,:2])
    hi=np.minimum(a[:,None,2:],b[None,:,2:])
    inter=np.maximum(hi-lo,0).prod(-1)
    return inter/np.maximum((a[:,2:]-a[:,:2]).prod(-1)[:,None]+(b[:,2:]-b[:,:2]).prod(-1)[None,:]-inter,1e-12)

gt={}; sizes={}
for p in sorted((DATA/'images/test').glob('*.jpg')):
    w,h=Image.open(p).size; sizes[p.stem]=(w,h)
    text=(DATA/'labels/test'/f'{p.stem}.txt').read_text().strip()
    a=np.loadtxt(DATA/'labels/test'/f'{p.stem}.txt',ndmin=2) if text else np.empty((0,5))
    boxes=np.stack([(a[:,1]-a[:,3]/2)*w,(a[:,2]-a[:,4]/2)*h,(a[:,1]+a[:,3]/2)*w,(a[:,2]+a[:,4]/2)*h],axis=1)
    gt[p.stem]=boxes

summaries={}; bymethod={}; predictions={}
for method in METHODS:
    path=RUNS/method/'preds_test.csv'
    handle=path.open(newline='')
    groups=defaultdict(list)
    with handle as f:
        for r in csv.DictReader(f):
            if float(r['score'])>=.25:
                x,y,w,h=[float(r[k]) for k in ['x','y','width','height']]
                groups[r['image_id']].append([x,y,x+w,y+h,float(r['score'])])
    records={}; detail={}
    for name,boxes in gt.items():
        pred=sorted(groups[name],key=lambda x:-x[4])
        overlaps=ious([p[:4] for p in pred],boxes)
        used=set(); tp=[]; fp=[]; kinds=[]
        for i,p in enumerate(pred):
            candidates=[j for j in range(len(boxes)) if j not in used and overlaps[i,j]>=.5]
            if candidates:
                j=max(candidates,key=lambda j:overlaps[i,j]); used.add(j); tp.append((i,j))
            else:
                fp.append(i)
                best=overlaps[i].max(initial=0)
                kinds.append('duplicate' if best>=.5 else 'localization' if best>=.1 else 'background')
        fn=[j for j in range(len(boxes)) if j not in used]
        area=(boxes[:,2:]-boxes[:,:2]).prod(-1)
        small=area<1024
        records[name]=dict(gt=len(boxes),pred=len(pred),tp=len(tp),fp=len(fp),fn=len(fn),
                           small=int(small.sum()),small_tp=sum(bool(small[j]) for _,j in tp),
                           f1=2*len(tp)/max(2*len(tp)+len(fp)+len(fn),1),
                           error=abs(len(pred)-len(boxes)),signed_error=len(pred)-len(boxes))
        detail[name]=dict(pred=pred,tp=tp,fp=fp,fn=fn,kinds=kinds)
    aggregate={k:sum(r[k] for r in records.values()) for k in ['tp','fp','fn','small','small_tp']}
    aggregate.update(count_mae=float(np.mean([r['error'] for r in records.values()])),
                     count_bias=float(np.mean([r['signed_error'] for r in records.values()])),
                     exact_images=sum(r['error']==0 for r in records.values()),
                     error_types={kind:sum(d['kinds'].count(kind) for d in detail.values()) for kind in ['duplicate','localization','background']})
    ref=json.loads((RUNS/method/'test_metrics.json').read_text())
    assert [aggregate[k] for k in ['tp','fp','fn']]==[ref[k] for k in ['TP','FP','FN']],method
    summaries[method]=aggregate; bymethod[method]=records; predictions[method]=detail

sparse=[n for n,r in bymethod['m4'].items() if 3<=r['gt']<=10 and r['f1']==1]
success=sparse[len(sparse)//2]
negative=max((n for n,r in bymethod['m4'].items() if r['gt']==0),key=lambda n:bymethod['m4'][n]['fp'])
dense=max(gt,key=lambda n:len(gt[n]))
better=max((n for n,r in bymethod['m4'].items() if r['gt']>0),key=lambda n:bymethod['m4'][n]['tp']-bymethod['m2'][n]['tp'])
miss=max((n for n,r in bymethod['m4'].items() if 0<r['gt']<=20),key=lambda n:bymethod['m4'][n]['fn'])
cases=dict(success=success,negative=negative,dense=dense,better=better,miss=miss)

from matplotlib.font_manager import findfont
font=ImageFont.truetype(findfont('DejaVu Sans'),21)
for label,name in cases.items():
    panels=[]
    for method in ['m2','m4']:
        im=Image.open(DATA/'images/test'/f'{name}.jpg').convert('RGB')
        draw=ImageDraw.Draw(im)
        det=predictions[method][name]
        for j in det['fn']: draw.rectangle(tuple(gt[name][j]),outline='#00d5ff',width=3)
        for i,p in enumerate(det['pred']): draw.rectangle(tuple(p[:4]),outline='#f23b44' if i in det['fp'] else '#00e674',width=2)
        r=bymethod[method][name]
        panel=Image.new('RGB',(512,560),'white'); panel.paste(im,(0,48))
        ImageDraw.Draw(panel).text((8,10),f"{NAMES[method]}  TP {r['tp']}  FP {r['fp']}  FN {r['fn']}",font=font,fill='black')
        panels.append(panel)
    pair=Image.new('RGB',(1024,560),'white'); pair.paste(panels[0],(0,0)); pair.paste(panels[1],(512,0)); pair.save(OUT/f'case_{label}.jpg',quality=93)

logs={m:[json.loads(x) for x in (RUNS/m/'metrics.jsonl').read_text().splitlines()] for m in METHODS[1:]}
for key,file in [('AP50','curve_ap50'),('AP50:95','curve_ap')]:
    fig,ax=plt.subplots(figsize=(4.5,2.75),layout='constrained')
    for m,rows in logs.items(): ax.plot([r['epoch'] for r in rows],[r[key]*100 for r in rows],label=NAMES[m],lw=1.25)
    ax.set(xlabel='Epoch',ylabel=key+' (%)',ylim=(0,100)); ax.grid(alpha=.18); ax.legend(ncol=2,fontsize=7,loc='lower right'); fig.savefig(OUT/f'{file}.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(4.5,2.8),layout='constrained')
for m in METHODS[1:]:
    q=json.loads((RUNS/m/'preds_test.timing.json').read_text()); v=json.loads((RUNS/m/'test_metrics.json').read_text())
    ax.scatter(q['amortised_ms_per_image'],v['AP-small']*100,s=55)
    ax.annotate(NAMES[m],(q['amortised_ms_per_image'],v['AP-small']*100),xytext=(5,4 if m!='p2-only' else -12),textcoords='offset points',fontsize=7.5)
ax.set(xlabel='Amortised inference time (ms/image)',ylabel='AP-small (%)',xlim=(3.5,10.8),ylim=(30.8,32.6)); ax.grid(alpha=.18); fig.savefig(OUT/'tradeoff.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(4.5,2.75),layout='constrained')
for m,rows in logs.items():
    values=np.array([sum(r['losses'].values()) if isinstance(r['losses'],dict) else sum(r['losses']) for r in rows])
    ax.plot(np.arange(1,101),values/values[0],label=NAMES[m],lw=1.25)
ax.set(xlabel='Epoch',ylabel='Training loss / epoch-1 loss'); ax.grid(alpha=.18); ax.legend(ncol=2,fontsize=7)
fig.savefig(OUT/'loss_curves.png',dpi=220); plt.close(fig)

fig,ax=plt.subplots(figsize=(4.5,2.75),layout='constrained')
for m in METHODS[1:]:
    rows=[json.loads(x) for x in (RUNS/m/'epoch_timing.jsonl').read_text().splitlines()]
    ax.plot([r['epoch'] for r in rows],[r['epoch_wall_s'] for r in rows],label=NAMES[m],lw=1.1)
ax.set(xlabel='Epoch',ylabel='Complete epoch wall time (s)'); ax.grid(alpha=.18); ax.legend(ncol=2,fontsize=7)
fig.savefig(OUT/'epoch_times.png',dpi=220); plt.close(fig)

summary=dict(aggregates=summaries,cases={k:dict(image=v,methods={m:bymethod[m][v] for m in METHODS}) for k,v in cases.items()},
             small_objects=int(sum(((b[:,2:]-b[:,:2]).prod(-1)<1024).sum() for b in gt.values())),
             total_objects=sum(len(b) for b in gt.values()),
             selection='事后展示案例。成功例取3–10只且M4 F1=1列表中点。失败例按M4负图最大FP、最高密度、非密集最大FN选择。改进例按M4相对M2最大TP差选择。用于机制讨论，不代表总体性能。')
(OUT/'diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
