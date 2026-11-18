"""Sequential, fixed-checkpoint validation corruption benchmark; no tuning or training."""
import argparse
import csv
from datetime import datetime
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from controlled_diagnostics import save_json, predict, read_predictions, operating_counts
from data import image_paths, read_image
from evaluate import ground_truth, evaluate
from model import load_model
from infer import sync

CONDITIONS=['gaussian_5','gaussian_15','gaussian_30','blur_1p5','dark_0p5']
RF_MODEL=None
RF_INFER=None


def now(): return datetime.now().astimezone().isoformat()


def corrupt(image, stem, condition):
    if condition=='clean': return image.copy()
    if condition.startswith('gaussian_'):
        # Same standard-normal field at every severity; stable across methods/processes.
        image_key=int.from_bytes(hashlib.sha256(stem.encode('utf-8')).digest()[:8],'little')
        rng=np.random.default_rng(np.random.SeedSequence([42,image_key]))
        sigma=float(condition.split('_')[1])
        result=np.rint(np.clip(image.astype(np.float32)+sigma*rng.standard_normal(image.shape,dtype=np.float32),0,255)).astype(np.uint8)
    elif condition=='blur_1p5':
        result=cv2.GaussianBlur(image,(9,9),1.5,borderType=cv2.BORDER_REFLECT_101)
    elif condition=='dark_0p5':
        result=np.rint(image.astype(np.float32)*.5).astype(np.uint8)
    else: raise ValueError(condition)
    assert result.shape==image.shape and result.dtype==np.uint8
    return result


def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def rf_init(checkpoint):
    global RF_MODEL,RF_INFER
    from threadpoolctl import threadpool_limits
    import joblib
    threadpool_limits(1)
    cv2.setNumThreads(1)
    sys.path.insert(0,str(ROOT/'Random Forest/Submit'))
    RF_INFER=module_from_file('robust_rf_infer',ROOT/'Random Forest/Submit/infer.py')
    RF_MODEL=joblib.load(checkpoint)
    RF_MODEL.n_jobs=1


def rf_one(job):
    index,path,condition=job
    image=corrupt(read_image(path),Path(path).stem,condition)
    result=RF_INFER.run_one(RF_MODEL,image,0)
    # Match the original RF export precision, including its confidence tie behavior.
    rows=[dict(image_id=index,category_id=0,bbox=[float(f'{v:.2f}') for v in [x,y,w,h]],score=float(f'{s:.5f}')) for x,y,w,h,s in result]
    return index,hashlib.sha256(image.tobytes()).hexdigest(),rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--run',default=str(ROOT/'runs/5090d_batch8_seed42_20261117'))
    parser.add_argument('--rf-workers',type=int,default=8)
    args=parser.parse_args()
    run=Path(args.run); out=run/'robustness';out.mkdir(exist_ok=True)
    root=Path(args.root)
    checkpoints={m:run/m/'best.pt' for m in ['m2','m3','p2-only','m4']}
    checkpoints['m1']=run/'m1/models/rf_final.joblib'
    hashes={m:hashlib.sha256(p.read_bytes()).hexdigest() for m,p in checkpoints.items()}
    paths=image_paths(root,'val');images=[read_image(p) for p in paths]
    gt=ground_truth(root,'val');assert len(images)==291 and len(gt['annotations'])==1843
    assert [p.stem for p in paths]==[i['stem'] for i in gt['images']]
    mapping={im['stem']:im['id'] for im in gt['images']}
    image_hashes={p.stem:hashlib.sha256(im.tobytes()).hexdigest() for p,im in zip(paths,images)}
    input_hashes={condition:{p.stem:hashlib.sha256(corrupt(im,p.stem,condition).tobytes()).hexdigest() for p,im in zip(paths,images)} for condition in CONDITIONS}
    plan=dict(recorded_at=now(),split='val',images=291,objects=1843,seed=42,
        methods=['m2','m3','p2-only','m4','m1'],conditions=CONDITIONS,
        gaussian='Independent additive standard-normal BGR channel noise before resizing, sigma 5/15/30 on uint8 scale; clip [0,255], round nearest. Stable SHA256 image key plus root seed 42; same field across intensities.',
        blur='9x9 Gaussian, sigma=1.5, BORDER_REFLECT_101, before resizing',
        dark='Pixel value times 0.5, round nearest, before resizing',
        inference=dict(canvas=640,deep_batch=8,precision='FP16',nms_iou=.6,prefilter=.001,max_dets=300,operating_score=.25,matching_iou=.5),
        rf_execution=dict(workers=args.rf_workers,forest_n_jobs_per_worker=1,scope='One RF condition at a time; workers parallelize images only; no latency comparison.'),
        clean='Reuse frozen full-validation predictions; fresh M3 parity check and first-three-image RF worker parity check.',
        checkpoint_sha256=hashes,original_image_sha256=image_hashes,corrupted_image_sha256=input_hashes,
        prohibitions=['No training','No additional seeds','No test-set evaluation','No threshold or architecture tuning','No changes to labels or original images'],
        summaries='Paired AP percentage-point drop and AP retention from clean; arithmetic mean over the five specified corruptions is descriptive, not standard mCE or significance evidence.')
    if (out/'plan.json').exists():
        previous=json.loads((out/'plan.json').read_text(encoding='utf-8'))
        for key in ['checkpoint_sha256','original_image_sha256','corrupted_image_sha256','conditions','inference']:
            assert previous[key]==plan[key],key
    else: save_json(out/'plan.json',plan)
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    results={}
    for m in checkpoints:
        rows=read_predictions(run/m/'preds_val.csv',mapping)
        metrics=json.loads((run/m/'val_metrics.json').read_text())
        metrics.update(operating=operating_counts(gt,rows,.25),source_split='val',source='frozen validation predictions')
        results[m]={'clean':metrics}
    device=torch.device('cuda:0')
    started=time.perf_counter()
    def status(**extra):
        save_json(out/'status.json',dict(updated_at=now(),elapsed_s=time.perf_counter()-started,**extra))
    def record(method,condition,rows,wall):
        dest=out/method/condition;dest.mkdir(parents=True,exist_ok=True)
        metrics=evaluate(gt,rows)
        metrics.update(operating=operating_counts(gt,rows,.25),source_split='val',condition=condition,
                       execution_wall_s=wall,checkpoint_unchanged=True,input_hashes_verified=True)
        save_json(dest/'predictions.json',rows);save_json(dest/'metrics.json',metrics)
        results[method][condition]=metrics
        save_json(out/'partial_summary.json',results)
        print(method,condition,'AP50',metrics['AP50'],'AP95',metrics['AP50:95'],'F1',metrics['F1'],flush=True)
    for method in ['m2','m3','p2-only','m4']:
        status(status='running',method=method,condition='load')
        if method=='m3':
            pipe=module_from_file('robust_m3',ROOT/'Resnet/pipeline.py')
            model=pipe.build_model(False).to(device)
            model.load_state_dict(torch.load(checkpoints[method],map_location='cpu',weights_only=False)['model']);model.eval()
            predict_fn=pipe.predict_batch
        else:
            model,_=load_model(checkpoints[method],device)
            predict_fn=lambda model,batch,device:predict(model,batch,device,640,.6)
        for condition in (['clean'] if method=='m3' else [])+CONDITIONS:
            cache=out/method/condition/'metrics.json'
            if cache.exists(): results[method][condition]=json.loads(cache.read_text());continue
            status(status='running',method=method,condition=condition,images_done=0)
            rows=[];tick=time.perf_counter()
            for i in range(0,len(images),8):
                batch=[corrupt(im,p.stem,condition) for im,p in zip(images[i:i+8],paths[i:i+8])]
                for p,im in zip(paths[i:i+8],batch):
                    expected=image_hashes[p.stem] if condition=='clean' else input_hashes[condition][p.stem]
                    assert hashlib.sha256(im.tobytes()).hexdigest()==expected
                predictions=predict_fn(model,batch,device)
                for info,boxes in zip(gt['images'][i:i+8],predictions):
                    for x1,y1,x2,y2,score,_ in boxes:
                        rows.append(dict(image_id=info['id'],category_id=0,bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
            sync(device)
            if condition=='clean':
                check=evaluate(gt,rows)
                assert abs(check['AP50']-results[method]['clean']['AP50'])<1e-5
                save_json(out/'m3_clean_parity.json',dict(AP50=check['AP50'],stored_AP50=results[method]['clean']['AP50'],passed=True))
            else: record(method,condition,rows,time.perf_counter()-tick)
        del model;torch.cuda.empty_cache()
    # No GPU method is active during the CPU-only RF image worker pool.
    status(status='running',method='m1',condition='clean_worker_parity',images_done=0)
    with mp.get_context('spawn').Pool(args.rf_workers,initializer=rf_init,initargs=(str(checkpoints['m1']),)) as pool:
        parity_jobs=[(i+1,str(paths[i]),'clean') for i in range(3)]
        checked=list(pool.imap(rf_one,parity_jobs,chunksize=1))
        frozen=read_predictions(run/'m1/preds_val.csv',mapping)
        for index,digest,rows in checked:
            expected=[p for p in frozen if p['image_id']==index]
            assert len(rows)==len(expected)
            for a,b in zip(rows,expected):
                # The original RF CSV serializes coordinates to 2 decimals and scores to 5.
                assert np.allclose(a['bbox'],b['bbox'],atol=.00501,rtol=0),(a,b)
                assert abs(a['score']-b['score'])<.00000501,(a,b)
            assert digest==image_hashes[paths[index-1].stem]
        save_json(out/'m1_clean_worker_parity.json',dict(passed=True,images=3,forest_n_jobs=1,original_forest_n_jobs=2,
                  bbox_tolerance=.00501,score_tolerance=.00000501,reason='Original CSV decimal serialization precision'))
        for condition in CONDITIONS:
            cache=out/'m1'/condition/'metrics.json'
            if cache.exists(): results['m1'][condition]=json.loads(cache.read_text());continue
            status(status='running',method='m1',condition=condition,images_done=0)
            tick=time.perf_counter();rows=[]
            jobs=[(i+1,str(p),condition) for i,p in enumerate(paths)]
            for done,(index,digest,predictions) in enumerate(pool.imap(rf_one,jobs,chunksize=1),1):
                assert digest==input_hashes[condition][paths[index-1].stem]
                rows.extend(predictions)
                if done%8==0 or done==len(paths):
                    status(status='running',method='m1',condition=condition,images_done=done,images_total=len(paths),condition_elapsed_s=time.perf_counter()-tick)
                    print('RF progress',condition,done,'/',len(paths),'seconds',round(time.perf_counter()-tick,1),flush=True)
            record('m1',condition,rows,time.perf_counter()-tick)
    for method,path in checkpoints.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==hashes[method]
    for path in paths:
        assert hashlib.sha256(read_image(path).tobytes()).hexdigest()==image_hashes[path.stem]
    summary=dict(status='complete',completed_at=now(),seed=42,split='val',methods=results)
    save_json(out/'summary.json',summary)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=['method','condition','AP50','AP50:95','AP-small','F1','AP50_drop_pp','AP95_drop_pp','AP95_retention_percent','count_mae'])
        writer.writeheader()
        for method,conditions in results.items():
            clean=conditions['clean']
            for condition,v in conditions.items():
                writer.writerow(dict(method=method,condition=condition,**{k:v[k] for k in ['AP50','AP50:95','AP-small','F1']},
                    AP50_drop_pp=100*(clean['AP50']-v['AP50']),AP95_drop_pp=100*(clean['AP50:95']-v['AP50:95']),
                    AP95_retention_percent=100*v['AP50:95']/clean['AP50:95'],count_mae=v['operating']['count_mae']))
    status(status='complete',completed_conditions=25)
    print('All 25 full-validation corruption conditions complete.',flush=True)


if __name__=='__main__':
    mp.freeze_support()
    main()
