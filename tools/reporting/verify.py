from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import json
from decimal import Decimal, ROUND_HALF_UP
from docx import Document

here=Path(__file__).resolve().parent
out=OUTPUT_DIR
run=RUN_DIR
doc=Document(out/'COMP9517_Group104_中文项目报告_5090D评审强化版.docx')
methods=['m1','m2','m3','p2-only','m4']
assert len(doc.tables)==13
assert len(doc.inline_shapes)==13
assert json.loads((run/'status.json').read_text())['status']=='complete'
for i,m in enumerate(methods,1):
    result=json.loads((run/m/'test_metrics.json').read_text())
    for col,key in enumerate(['AP50','AP50:95','AP75'],1):
        assert doc.tables[2].cell(i,col).text==f'{result[key]*100:.2f}',(m,key)
    for col,key in enumerate(['precision','recall','F1','AP-small','AR-small'],1):
        assert doc.tables[3].cell(i,col).text==f'{result[key]*100:.2f}',(m,key)
    for col,key in enumerate(['TP','FP','FN'],1):
        assert doc.tables[4].cell(i,col).text==str(result[key]),(m,key)
    curve=json.loads((here/'pr_curves.json').read_text())[m]
    assert abs(curve['AP50']-result['AP50'])<1e-12
for i,m in enumerate(methods[1:],1):
    config=json.loads((run/m/'config.json').read_text())
    assert config['train_batch']==8 and config['epochs']==100 and config['seed']==42
    history=[json.loads(line) for line in (run/m/'metrics.jsonl').read_text().splitlines()]
    timing=[json.loads(line) for line in (run/m/'epoch_timing.jsonl').read_text().splitlines()]
    assert [r['epoch'] for r in history]==list(range(1,101))
    assert [r['epoch'] for r in timing]==list(range(1,101))
    best=max(history,key=lambda r:(r['AP50'],r['AP50:95'],-r['epoch']))
    assert best['epoch']=={'m2':75,'m3':7,'p2-only':98,'m4':82}[m]
    for col,key in enumerate(['training_s','validation_s','checkpoint_s','epoch_wall_s'],1):
        assert doc.tables[9].cell(i,col).text==f'{sum(r[key] for r in timing)/100:.2f}'
paragraphs='\n'.join(p.text for p in doc.paragraphs)
assert '作者及学号  ____' in paragraphs
assert 'RTX 5090 D' in paragraphs and 'RTX 3090' not in paragraphs
assert '87.88%' in paragraphs and '0.875' in paragraphs and '400' in paragraphs
assert '最高小目标 AP 来自 P2-only' in paragraphs
assert all(f'[{i}]' in paragraphs for i in range(1,19))
print('Verified primary metrics, detection counts, COCO PR parity, 400 epoch records, 18 references, 13 tables and 13 figures.')
