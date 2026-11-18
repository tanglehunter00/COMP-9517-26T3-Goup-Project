"""Validation-only single-factor diagnostics of frozen seed-42 checkpoints.

No model fitting, checkpoint selection, test-set evaluation, or seed sweep.
The immutable plan is written before computing validation outcomes.
"""
import argparse
import contextlib
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torchvision
from ultralytics.utils.nms import non_max_suppression

HERE = Path(__file__).resolve().parent
SHARED = HERE/'Yolo Improeved'
sys.path.insert(0, str(SHARED))
from data import image_paths, read_image
from model import load_model
from evaluate import ground_truth, evaluate
from infer import sync


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def xywh_iou(box, targets):
    if not targets:
        return np.empty(0)
    a, b = np.asarray(box), np.asarray(targets)
    intersection = np.maximum(0, np.minimum(a[:2]+a[2:], b[:, :2]+b[:, 2:])-np.maximum(a[:2], b[:, :2])).prod(1)
    return intersection/np.maximum(a[2:].prod()+b[:, 2:].prod(1)-intersection, 1e-12)


def operating_counts(gt, rows, threshold):
    """Greedy one-to-one IoU50 matching identical to the fixed main evaluator."""
    grouped_gt = {image['id']: [] for image in gt['images']}
    grouped_predictions = {key: [] for key in grouped_gt}
    for annotation in gt['annotations']:
        grouped_gt[annotation['image_id']].append(annotation['bbox'])
    for row in rows:
        if row['score'] >= threshold:
            grouped_predictions[row['image_id']].append(row)
    tp = fp = fn = negative_fp = negatives = small_tp = small_total = 0
    count_error = []
    for image_id, boxes in grouped_gt.items():
        selected = sorted(grouped_predictions[image_id], key=lambda row: -row['score'])
        used = set()
        small = [box[2]*box[3] < 1024 for box in boxes]
        small_total += sum(small)
        for row in selected:
            overlaps = xywh_iou(row['bbox'], boxes)
            candidates = [i for i in range(len(boxes)) if i not in used and overlaps[i] >= .5]
            if candidates:
                match = max(candidates, key=lambda i: overlaps[i])
                used.add(match)
                tp += 1
                small_tp += small[match]
            else:
                fp += 1
        fn += len(boxes)-len(used)
        count_error.append(abs(len(selected)-len(boxes)))
        if not boxes:
            negatives += 1
            negative_fp += len(selected)
    precision = tp/max(tp+fp, 1)
    recall = tp/max(tp+fn, 1)
    return dict(threshold=threshold, TP=tp, FP=fp, FN=fn, precision=precision, recall=recall,
                F1=2*tp/max(2*tp+fp+fn, 1), count_mae=float(np.mean(count_error)),
                negative_fp_per_image=negative_fp/negatives if negatives else None,
                small_recall_iou50=small_tp/max(small_total, 1))


def read_predictions(path, mapping):
    rows = []
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            rows.append(dict(image_id=mapping[row['image_id']], category_id=0,
                             bbox=[float(row[key]) for key in ['x', 'y', 'width', 'height']], score=float(row['score'])))
    return rows


@torch.inference_mode()
def predict(model, images, device, size, nms_iou):
    processed = []
    metadata = []
    for image in images:
        h, w = image.shape[:2]
        scale = min(size/w, size/h)
        nw, nh = round(w*scale), round(h*scale)
        px, py = (size-nw)//2, (size-nh)//2
        canvas = np.full((size, size, 3), 114, dtype=np.uint8)
        canvas[py:py+nh, px:px+nw] = cv2.resize(image, (nw, nh))
        processed.append(canvas[:, :, ::-1].transpose(2, 0, 1))
        metadata.append((scale, px, py, w, h))
    tensor = torch.from_numpy(np.ascontiguousarray(np.stack(processed))).to(device).float()/255
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(tensor)
    detections = non_max_suppression(outputs, conf_thres=.001, iou_thres=nms_iou,
        max_det=300, max_nms=30000, nc=1, max_time_img=60)
    result = []
    for detection, (scale, px, py, w, h) in zip(detections, metadata):
        boxes = detection.float().cpu().numpy()
        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]]-px)/scale, 0, w)
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]]-py)/scale, 0, h)
        result.append(boxes[(boxes[:, 2]>boxes[:, 0]) & (boxes[:, 3]>boxes[:, 1])])
    return result


def condition(model, images, gt, out, device, name, size=640, nms_iou=.6):
    target = out/name
    target.mkdir(exist_ok=True)
    result_path = target/'metrics.json'
    if result_path.exists():
        return json.loads(result_path.read_text(encoding='utf-8'))
    rows = []
    started = time.perf_counter()
    for start in range(0, len(images), 8):
        for image_info, boxes in zip(gt['images'][start:start+8], predict(model, images[start:start+8], device, size, nms_iou)):
            for x1, y1, x2, y2, score, _ in boxes:
                rows.append(dict(image_id=image_info['id'], category_id=0,
                    bbox=[float(x1), float(y1), float(x2-x1), float(y2-y1)], score=float(score)))
    sync(device)
    inference_s = time.perf_counter()-started
    result = evaluate(gt, rows)
    result.update(operating=operating_counts(gt, rows, .25), size=size, nms_iou=nms_iou,
                  prediction_pass_s=inference_s, source_split='val', checkpoint_unchanged=True,
                  inference_parameters_changed=False)
    save_json(target/'predictions.json', rows)
    save_json(result_path, result)
    print(name, 'AP50', result['AP50'], 'AP75', result['AP75'], 'F1', result['F1'], flush=True)
    return result


def benchmark(model, images, device, size):
    batches = [images[i:i+8] for i in range(0, len(images), 8)]
    for _ in range(20):
        predict(model, batches[0], device, size, .6)
    sync(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for _ in range(3):
        sync(device)
        started = time.perf_counter()
        for batch in batches:
            predict(model, batch, device, size, .6)
        sync(device)
        elapsed.append(time.perf_counter()-started)
    return dict(repeat_s=elapsed, repeats=3, warmup_batches=20, batch=8, images=len(images),
                amortised_ms_image=sum(elapsed)*1000/(3*len(images)),
                images_per_s=3*len(images)/sum(elapsed), peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                split='val')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--run', default=str(HERE/'runs/5090d_batch8_seed42_20261117'))
    args = parser.parse_args()
    root, run = Path(args.root), Path(args.run)
    out = run/'controlled_diagnostics'
    out.mkdir(exist_ok=True)
    checkpoints = {method: hashlib.sha256((run/method/'best.pt').read_bytes()).hexdigest()
                   for method in ['m2', 'm3', 'p2-only', 'm4']}
    plan = dict(recorded_at=datetime.now().astimezone().isoformat(), split='val', images=291, seed=42,
        checkpoint_sha256=checkpoints, batch=8, precision='FP16', score_prefilter=.001, max_dets=300, max_nms=30000,
        resolution=dict(methods=['m2','m4'], sizes=[640,960,1280], nms_iou=.6),
        nms=dict(methods=['m2','p2-only','m4'], iou=[.3,.5,.6,.7,.9], size=640),
        thresholds=dict(methods=['m2','m3','p2-only','m4'], values=[round(i*.05,2) for i in range(1,20)],
                        selection='Highest validation F1; break ties by higher precision, then higher threshold'),
        offset=dict(method='m4', size=640, nms_iou=.6, intervention='Set DySample offset weights and bias to zero in memory only'),
        prohibitions=['No training','No new random seeds','No test-set tuning or evaluation','Do not replace the frozen main results'],
        timing='Resolution only: preloaded validation images, 20 warmup batches, 3 full repetitions, CUDA sync')
    plan_path = out/'plan.json'
    if plan_path.exists():
        previous = json.loads(plan_path.read_text(encoding='utf-8'))
        assert previous['checkpoint_sha256']==checkpoints
    else:
        save_json(plan_path, plan)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device('cuda:0')
    paths = image_paths(root, 'val')
    images = [read_image(path) for path in paths]
    gt = ground_truth(root, 'val')
    mapping = {image['stem']: image['id'] for image in gt['images']}
    thresholds = {}
    for method in ['m2','m3','p2-only','m4']:
        rows = read_predictions(run/method/'preds_val.csv', mapping)
        grid = [operating_counts(gt, rows, t) for t in plan['thresholds']['values']]
        old = json.loads((run/method/'val_metrics.json').read_text())
        original = next(row for row in grid if row['threshold']==.25)
        assert [original[k] for k in ['TP','FP','FN']]==[old[k] for k in ['TP','FP','FN']], method
        selected = max(grid,key=lambda row:(row['F1'],row['precision'],row['threshold']))
        thresholds[method]=dict(grid=grid, original=original, selected=selected)
    save_json(out/'thresholds.json',thresholds)
    conditions = {}
    for method in ['m2','p2-only','m4']:
        model, _ = load_model(run/method/'best.pt',device)
        base = condition(model,images,gt,out,device,f'{method}_s640_n60')
        reference = json.loads((run/method/'val_metrics.json').read_text())
        if abs(base['AP50']-reference['AP50'])>1e-5:
            raise RuntimeError(f'{method}: fresh baseline differs from stored validation AP50')
        conditions[f'{method}_s640_n60']=base
        for nms_iou in [.3,.5,.7,.9]:
            name=f'{method}_s640_n{round(nms_iou*100)}'
            conditions[name]=condition(model,images,gt,out,device,name,nms_iou=nms_iou)
        if method in ['m2','m4']:
            for size in [640,960,1280]:
                name=f'{method}_s{size}_n60'
                if size!=640:
                    conditions[name]=condition(model,images,gt,out,device,name,size=size)
                timing=benchmark(model,images,device,size)
                conditions[name]['timing']=timing
                save_json(out/name/'metrics.json',conditions[name])
            for module in model.modules():
                if module.__class__.__name__=='DySample':
                    with torch.no_grad():
                        module.offset.weight.zero_()
                        module.offset.bias.zero_()
            if method=='m4':
                name='m4_s640_n60_zero_offset'
                conditions[name]=condition(model,images,gt,out,device,name)
                conditions[name]['checkpoint_unchanged']=True
                conditions[name]['inference_parameters_changed']=True
                conditions[name]['intervention']='In-memory zero offsets; checkpoint file unchanged'
                save_json(out/name/'metrics.json',conditions[name])
        del model
        torch.cuda.empty_cache()
    for method, digest in checkpoints.items():
        assert hashlib.sha256((run/method/'best.pt').read_bytes()).hexdigest()==digest
    save_json(out/'summary.json',dict(status='complete',completed_at=datetime.now().astimezone().isoformat(),
                                    conditions=conditions,thresholds=thresholds,split='val',seed=42))
    print('All validation-only controlled diagnostics complete.',flush=True)


if __name__=='__main__':
    main()
