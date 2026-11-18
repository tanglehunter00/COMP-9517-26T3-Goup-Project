from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import contextlib
import io
import json
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

HERE=Path(__file__).resolve().parent
RUN=RUN_DIR
sys.path.insert(0,str(ROOT/'Yolo Improeved'))
from evaluate import ground_truth, csv_rows

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
names={'m1':'RF','m2':'YOLO11s','m3':'Faster R-CNN','p2-only':'P2-only','m4':'P2 + DySample'}
gt=ground_truth(DATA_ROOT,'test')
curves={}
for method in names:
    rows=csv_rows(RUN/method/'preds_test.csv',gt)
    with contextlib.redirect_stdout(io.StringIO()):
        truth=COCO(); truth.dataset=gt; truth.createIndex()
        detections=truth.loadRes(rows)
        evaluator=COCOeval(truth,detections,'bbox'); evaluator.params.maxDets=[1,10,300]
        evaluator.evaluate(); evaluator.accumulate()
    precision=evaluator.eval['precision'][0,:,0,0,2]
    actual=float(precision[precision>-1].mean())
    expected=json.loads((RUN/method/'test_metrics.json').read_text())['AP50']
    assert abs(actual-expected)<1e-12,(method,actual,expected)
    curves[method]=dict(recall=evaluator.params.recThrs.tolist(),precision=precision.tolist(),AP50=actual)
(HERE/'pr_curves.json').write_text(json.dumps(curves,indent=2))
fig,axis=plt.subplots(figsize=(4.5,3),layout='constrained')
for method,values in curves.items():
    axis.plot(values['recall'],values['precision'],label=names[method],lw=1.3)
axis.set(xlabel='Recall',ylabel='Interpolated precision',xlim=(0,1),ylim=(0,1.02))
axis.legend(ncol=2,fontsize=7,loc='lower left'); axis.grid(alpha=.16)
fig.savefig(HERE/'pr_curve.png',dpi=250); plt.close(fig)

m=json.loads((RUN/'m4/test_metrics.json').read_text())
fig,axis=plt.subplots(figsize=(4,3.2),layout='constrained')
matrix=np.array([[m['TP'],m['FP']],[m['FN'],np.nan]],dtype=float)
cmap=plt.get_cmap('Blues').copy(); cmap.set_bad('#eeeeee')
axis.imshow(matrix,cmap=cmap,vmin=0,vmax=m['TP'])
for i in range(2):
    for j in range(2):
        value=matrix[i,j]
        label=(f"{['TP','FP','FN'][i*2+j]}\n{int(value)}" if not np.isnan(value) else 'Not defined')
        axis.text(j,i,label,ha='center',va='center',fontsize=12,color='white' if i==j==0 else 'black')
axis.set(xticks=[0,1],yticks=[0,1],xticklabels=['Bird','Background'],yticklabels=['Bird','Background'],
         xlabel='Ground truth',ylabel='Prediction')
axis.spines[['top','right','left','bottom']].set_visible(False)
fig.savefig(HERE/'detection_confusion.png',dpi=250); plt.close(fig)

fig,ax=plt.subplots(figsize=(5,3.4),layout='constrained')
ax.set(xlim=(0,10),ylim=(0,7)); ax.axis('off')
def box(x,y,w,h,text,fill='#edf2f7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.06',edgecolor='#334155',facecolor=fill,linewidth=1))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10)
def arrow(start,end):
    ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',color='#334155',linewidth=1.1))
box(.2,5.45,2.6,1.0,'Backbone P2 [2]\n160 x 160')
box(.2,3.5,2.6,1.0,'Neck P3 [16]\n80 x 80')
box(3.5,3.5,2.6,1.0,'Upsample x2 [23]\nDySample / nearest','#fff3dc')
box(6.7,4.7,2.7,1.0,'Concat [24]\nwith backbone P2')
box(6.7,2.8,2.7,1.0,'C3k2 [25]\nP2 features')
box(6.7,.45,2.7,1.35,'Detect [26]\nP2 / P3 / P4 / P5','#dcf2e6')
box(.2,.6,3.6,1.2,'Existing P3/P4/P5 paths\n[16] / [19] / [22]')
arrow((2.8,5.95),(6.7,5.2)); arrow((2.8,4),(3.5,4)); arrow((6.1,4),(7.3,4.7))
arrow((8.05,4.7),(8.05,3.8)); arrow((8.05,2.8),(8.05,1.8)); arrow((3.8,1.2),(6.7,1.2))
ax.text(4.9,6.55,'Input 640 x 640 | layer indices in brackets',ha='center',fontsize=9)
fig.savefig(HERE/'architecture.png',dpi=250); plt.close(fig)
print('Generated verified COCO PR curves, detection accounting matrix and architecture diagram.')
