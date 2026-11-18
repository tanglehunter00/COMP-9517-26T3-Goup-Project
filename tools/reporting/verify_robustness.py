from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import hashlib
import json
import shutil
from docx import Document

HERE=Path(__file__).resolve().parent
RUN=RUN_DIR
EX=RUN/'robustness'
OUT=OUTPUT_DIR
summary=json.loads((EX/'summary.json').read_text(encoding='utf-8'))
plan=json.loads((EX/'plan.json').read_text(encoding='utf-8'))
assert summary['status']=='complete' and summary['seed']==42 and summary['split']=='val'
conditions=['gaussian_5','gaussian_15','gaussian_30','blur_1p5','dark_0p5']
assert plan['conditions']==conditions
assert len(summary['methods'])==5
for method,results in summary['methods'].items():
    assert set(results)=={'clean',*conditions}
    model_path=RUN/method/'best.pt' if method!='m1' else RUN/'m1/models/rf_final.joblib'
    assert hashlib.sha256(model_path.read_bytes()).hexdigest()==plan['checkpoint_sha256'][method]
    baseline=json.loads((RUN/method/'val_metrics.json').read_text())
    for k in ['AP50','AP50:95','AP-small','F1','TP','FP','FN']:
        assert results['clean'][k]==baseline[k]
    for condition in conditions:
        metrics=json.loads((EX/method/condition/'metrics.json').read_text())
        assert metrics==results[condition]
        assert metrics['input_hashes_verified'] and metrics['checkpoint_unchanged']
        assert metrics['source_split']=='val' and metrics['TP']+metrics['FN']==1843
        rows=json.loads((EX/method/condition/'predictions.json').read_text())
        assert {r['image_id'] for r in rows}.issubset(set(range(1,292)))
assert json.loads((EX/'m3_clean_parity.json').read_text())['passed']
assert json.loads((EX/'m1_clean_worker_parity.json').read_text())['passed']
doc=Document(OUT/'COMP9517_Group104_中文项目报告_5090D鲁棒性完整版.docx')
old=Document(OUT/'COMP9517_Group104_中文项目报告_5090D补充实验版.docx')
assert len(doc.tables)==17 and len(doc.inline_shapes)==17
for i in range(15):
    assert [[c.text for c in r.cells] for r in doc.tables[i].rows]==[[c.text for c in r.cells] for r in old.tables[i].rows],i
methods=['m1','m2','m3','p2-only','m4']
for i,method in enumerate(methods,1):
    for j,condition in enumerate(['clean']+conditions,1):
        assert doc.tables[15].cell(i,j).text==f"{summary['methods'][method][condition]['AP50:95']*100:.2f}"
aggregate=json.loads((HERE/'robustness_aggregates.json').read_text(encoding='utf-8'))
for i,method in enumerate(methods,1):
    for j,key in enumerate(['AP50','AP-small','F1'],1):
        assert doc.tables[16].cell(i,j).text==f'{aggregate[method][key]*100:.2f}'
    assert doc.tables[16].cell(i,4).text==f"{aggregate[method]['AP95_retention_percent']:.2f}"
text='\n'.join(p.text for p in doc.paragraphs)
assert '作者及学号  ____' in text and all(f'[{i}]' in text for i in range(1,19))
assert '25 个完整扰动验证条件' in text
for name in ['robustness_assets.py','robustness_aggregates.json','corruption_example.json','add_robustness.py','build_robustness_report.py','verify_robustness.py']:
    shutil.copy2(HERE/name,EX/name)
shutil.copy2(ROOT/'robustness_benchmark.py',EX/'robustness_benchmark.py')
for name in ['comparison.csv','summary.json']:
    shutil.copy2(EX/name,ROOT/'reports'/('ROBUSTNESS_'+name))
shutil.copy2(EX/'comparison.csv',OUT/'COMP9517_Group104_鲁棒性横评.csv')
lines=['# Robustness comparison / 鲁棒性横评','',
       'Completed validation-only inference using the frozen seed-42 checkpoints. No new training, seed sweep, test-set tuning, or changes to model implementations, original images or labels. / 使用冻结的 seed 42 权重，未重新训练、增加训练种子、调测试阈值或修改原方法及数据。','',
       'Protocol / 设置：291 validation images, 1,843 boxes; input 640; deep batch 8, FP16; NMS IoU 0.6; prefilter 0.001; max detections 300; fixed score 0.25 and matching IoU 0.5. Five methods run sequentially. RF alone uses 8 image workers with forest n_jobs=1; this prediction pass is not a latency benchmark.','',
       'Corruptions / 扰动：independent additive Gaussian pixel/channel noise, sigma 5/15/30 on 0–255 values; Gaussian blur kernel 9, sigma 1.5, reflect-101 border; brightness multiplied by 0.5. All applied before resizing, clipped/rounded to uint8 where appropriate. Image-keyed random streams derive from seed 42 and reuse the same standard-normal field across noise intensities. Per-image SHA256 equality is checked for every method.','',
       '## Absolute AP50:95 (%) / 绝对检测精度','',
       '| Method | Clean | Noise 5 | Noise 15 | Noise 30 | Blur | Dark |',
       '|---|---:|---:|---:|---:|---:|---:|']
for method in methods:
    values=[summary['methods'][method][c]['AP50:95']*100 for c in ['clean']+conditions]
    lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['','## Descriptive five-condition means / 五项扰动均值','',
        '| Method | Mean AP50 (%) | Mean AP50:95 (%) | Mean AP-small (%) | Mean F1 (%) | AP95 drop (pp) | AP95 retention (%) |',
        '|---|---:|---:|---:|---:|---:|---:|']
for method in methods:
    a=aggregate[method]
    values=[100*a[k] for k in ['AP50','AP50:95','AP-small','F1']]+[a['AP95_drop_pp'],a['AP95_retention_percent']]
    lines.append('| '+method+' | '+' | '.join(f'{v:.2f}' for v in values)+' |')
lines+=['',
        'Each specified corruption has equal weight, so Gaussian noise accounts for 60% of this custom mean. It is not standard mCE or a model of real-world occurrence probabilities. Low clean AP can make relative retention misleading. / 每项等权，高斯噪声共占60%；该均值不是标准mCE，也不代表真实扰动发生概率。低基线方法的保留率不能单独用于判断实用性。','',
        'Units / 单位：raw CSV AP and F1 fields use the 0–1 scale; drop_pp fields use percentage points; retention fields use percent. Report tables express AP and F1 as percentages. / 原始CSV的AP及F1为0–1，drop_pp为百分点，retention为百分比；报告表格的AP和F1为百分数。','',
        'Limitations / 局限：one synthetic noise realization, no combination corruptions or real adverse-weather data; validation data were already used for checkpoint selection. These are diagnostic results rather than independent generalization evidence. Cross-architecture differences also include pretrained weights, optimization and augmentation choices.','',
        '## Reproduction / 复现','',
        'From the repository root / 在项目根目录运行：','',
        '```powershell',
        "$env:OMP_NUM_THREADS='2'", "$env:MKL_NUM_THREADS='2'", "$env:OPENBLAS_NUM_THREADS='2'", "$env:PYTHONIOENCODING='utf-8'", "$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'",
        "& '.\\Yolo Improeved\\.venv\\Scripts\\python.exe' -u robustness_benchmark.py --root '.\\Yolo Improeved\\work\\beach-bird\\Beach Bird Detection Dataset v1.0'",'```','',
        'Existing completed condition files are reused on rerun. Raw plan, image hashes, predictions, parity checks, execution notes, metrics and source snapshots are under `runs/5090d_batch8_seed42_20261117/robustness/`. The script copy there is for inspection; launch the repository-root entry point. / 已完成条件可复用；完整计划、图像哈希、预测、核对记录、执行说明、指标和源码快照均保存在该目录。','',
        'The first RF verification attempt used tolerances tighter than the original CSV rounding and stopped; it was corrected before any corrupted RF metrics were completed. RF exports retain original 2-decimal coordinates and 5-decimal scores. Primary clean test outputs were never replaced. / 首次RF核对因容差小于原CSV舍入精度而停止；在任何RF扰动指标完成前修正。RF仍采用原来的坐标两位、分数五位小数，主测试结果未被替换。','']
(ROOT/'reports/ROBUSTNESS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
print('Verified 25 corruption conditions, 5 frozen clean baselines, identical corrupted pixels, unchanged weights, 15 unchanged main/control tables, 2 new robustness tables, 17 figures and 18 references.')
