from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import csv
import hashlib
import json
import shutil
from docx import Document

HERE=Path(__file__).resolve().parent
RUN=RUN_DIR
EX=RUN/'controlled_diagnostics'
OUT=OUTPUT_DIR
summary=json.loads((EX/'summary.json').read_text(encoding='utf-8'))
plan=json.loads((EX/'plan.json').read_text(encoding='utf-8'))
assert summary['status']=='complete' and summary['split']=='val' and summary['seed']==42
assert len(summary['conditions'])==20
assert sum(len(t['grid']) for t in summary['thresholds'].values())==76
for m,digest in plan['checkpoint_sha256'].items():
    assert hashlib.sha256((RUN/m/'best.pt').read_bytes()).hexdigest()==digest
for name,metrics in summary['conditions'].items():
    metrics['checkpoint_unchanged']=True
    metrics['inference_parameters_changed']=name.endswith('zero_offset')
    (EX/name/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    assert metrics['source_split']=='val'
    assert metrics['TP']+metrics['FN']==1843
    if 'timing' in metrics:
        assert metrics['timing']['batch']==8 and len(metrics['timing']['repeat_s'])==3
        assert metrics['timing']['images']==291
        assert metrics['timing']['warmup_batches']==20
(EX/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
(EX/'metadata_correction.json').write_text(json.dumps(dict(
    note='Clarified metadata only: checkpoint_unchanged refers to files; inference_parameters_changed refers to in-memory offset intervention. Numerical predictions and metrics are unchanged.',
    plan_unchanged=True),indent=2),encoding='utf-8')
shutil.copy2(ROOT/'controlled_diagnostics.py',EX/'controlled_diagnostics.py')
shutil.copy2(HERE/'supplement_assets.py',EX/'supplement_assets.py')
doc=Document(OUT/'COMP9517_Group104_中文项目报告_5090D补充实验版.docx')
old=Document(OUT/'COMP9517_Group104_中文项目报告_5090D评审强化版.docx')
assert len(doc.tables)==15 and len(doc.inline_shapes)==15
for i in range(12):
    assert [[c.text for c in r.cells] for r in doc.tables[i].rows]==[[c.text for c in r.cells] for r in old.tables[i].rows],i
for i,(m,size) in enumerate([(m,s) for m in ['m2','m4'] for s in [640,960,1280]],1):
    v=summary['conditions'][f'{m}_s{size}_n60']
    for column,key in [(2,'AP50'),(3,'AP-small')]:
        assert doc.tables[12].cell(i,column).text==f'{v[key]*100:.2f}'
    assert doc.tables[12].cell(i,4).text==f"{v['timing']['amortised_ms_image']:.2f}"
for i,m in enumerate(['m2','m3','p2-only','m4'],1):
    v=summary['thresholds'][m]
    assert doc.tables[13].cell(i,1).text==f"{v['original']['F1']*100:.2f}"
    assert doc.tables[13].cell(i,2).text==f"{v['selected']['threshold']:.2f}"
    assert doc.tables[13].cell(i,3).text==f"{v['selected']['F1']*100:.2f}"
    old_metrics=json.loads((RUN/m/'val_metrics.json').read_text())
    assert [v['original'][k] for k in ['TP','FP','FN']]==[old_metrics[k] for k in ['TP','FP','FN']]
for i,key in enumerate(['m4_s640_n60','m4_s640_n60_zero_offset'],1):
    v=summary['conditions'][key]
    for col,k in [(1,'AP50'),(2,'AP75'),(3,'AP-small')]:
        assert doc.tables[14].cell(i,col).text==f'{v[k]*100:.2f}'
    assert doc.tables[14].cell(i,4).text==str(v['FN'])
text='\n'.join(p.text for p in doc.paragraphs)
assert '作者及学号  ____' in text
assert all(f'[{i}]' in text for i in range(1,19))
assert '这些方案尚未执行' not in text
assert '不增加训练种子' in text
with (EX/'conditions.csv').open('w',newline='',encoding='utf-8-sig') as f:
    keys=['condition','AP50','AP50:95','AP75','AP-small','AR-small','precision','recall','F1','TP','FP','FN','ms_per_image']
    writer=csv.DictWriter(f,keys);writer.writeheader()
    for name,v in summary['conditions'].items():
        row={k:v[k] for k in keys[1:-1]}
        row.update(condition=name,ms_per_image=v.get('timing',{}).get('amortised_ms_image',''))
        writer.writerow(row)
print('Verified 20 validation conditions, 76 threshold points, 6 three-repeat timings, unchanged checkpoint SHA256, 12 unchanged main tables, 3 new tables, 15 figures and 18 references.')
