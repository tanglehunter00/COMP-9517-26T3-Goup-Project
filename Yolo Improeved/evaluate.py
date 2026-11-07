"""Common original-coordinate COCO metrics, maxDets=300, plus fixed-point metrics."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from data import image_paths, read_image, labels


def ground_truth(root, split):
    images, annotations = [], []
    for i, p in enumerate(image_paths(root, split), 1):
        h, w = read_image(p).shape[:2]
        images.append(dict(id=i, file_name=p.name, width=w, height=h, stem=p.stem))
        for _, xc, yc, bw, bh in labels(root, split, p):
            box = [float((xc - bw/2)*w), float((yc - bh/2)*h), float(bw*w), float(bh*h)]
            annotations.append(dict(id=len(annotations)+1, image_id=i, category_id=0,
                                    bbox=box, area=box[2]*box[3], iscrowd=0))
    return dict(images=images, annotations=annotations, categories=[dict(id=0, name='bird')], info={})


def mean_valid(a):
    a = a[a > -1]
    return float(a.mean()) if a.size else None


def coco_metrics(gt, rows, ids):
    if not ids or not any(a['image_id'] in ids for a in gt['annotations']):
        return {k: None for k in ['AP50', 'AP50:95', 'AP75', 'AP-small', 'AR-small']}
    with contextlib.redirect_stdout(io.StringIO()):
        g = COCO(); g.dataset = gt; g.createIndex()
        if rows:
            d = g.loadRes(rows)
        else:
            d = COCO(); d.dataset = {**gt, 'annotations': []}; d.createIndex()
        e = COCOeval(g, d, 'bbox'); e.params.imgIds = list(ids); e.params.maxDets = [1, 10, 300]
        e.evaluate(); e.accumulate()
    p, r = e.eval['precision'], e.eval['recall']
    # Explicit array indexing avoids COCO summarize()'s AP default maxDets=100.
    return {'AP50:95': mean_valid(p[:, :, :, 0, 2]),
            'AP50': mean_valid(p[np.isclose(e.params.iouThrs, .5), :, :, 0, 2]),
            'AP75': mean_valid(p[np.isclose(e.params.iouThrs, .75), :, :, 0, 2]),
            'AP-small': mean_valid(p[:, :, :, 1, 2]), 'AR-small': mean_valid(r[:, :, 1, 2])}


def box_iou(a, b):
    if not len(b):
        return np.empty(0)
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    inter = np.maximum(0, np.minimum(a[:2]+a[2:], b[:, :2]+b[:, 2:])-np.maximum(a[:2], b[:, :2])).prod(1)
    return inter / np.maximum(a[2]*a[3]+b[:, 2]*b[:, 3]-inter, 1e-9)


def evaluate(gt, rows):
    ids = [x['id'] for x in gt['images']]
    result = coco_metrics(gt, rows, ids)
    tp = fp = fn = fpneg = nneg = 0
    groups = {'negative': [], 'sparse': [], 'medium': [], 'dense': []}
    for i in ids:
        boxes = [a['bbox'] for a in gt['annotations'] if a['image_id'] == i]
        n = len(boxes)
        groups['negative' if n == 0 else 'sparse' if n <= 10 else 'medium' if n <= 50 else 'dense'].append(i)
        dets = sorted((d for d in rows if d['image_id'] == i and d['score'] >= .25), key=lambda d: -d['score'])
        used = set()
        for d in dets:
            ious = box_iou(d['bbox'], boxes)
            candidates = [j for j in range(n) if j not in used and ious[j] >= .5]
            if candidates:
                used.add(max(candidates, key=lambda j: ious[j])); tp += 1
            else:
                fp += 1
        fn += n - len(used)
        if n == 0:
            nneg += 1; fpneg += len(dets)
    precision, recall = tp/max(tp+fp, 1), tp/max(tp+fn, 1)
    result.update(precision=precision, recall=recall, F1=2*precision*recall/max(precision+recall, 1e-12),
                  false_positives_per_negative_image=fpneg/nneg if nneg else None,
                  TP=tp, FP=fp, FN=fn, maxDets=[1, 10, 300])
    result['density'] = {name: {'images': len(g),
        'objects': sum(a['image_id'] in g for a in gt['annotations']),
        **coco_metrics(gt, rows, g)} for name, g in groups.items()}
    return result


def csv_rows(path, gt):
    mapping = {i['stem']: i['id'] for i in gt['images']}
    out, counts = [], {}
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            if r['image_id'] not in mapping or int(r['category_id']) != 0:
                raise ValueError('Prediction image/category ID does not match the dataset')
            i = mapping[r['image_id']]
            b = [float(r[k]) for k in ['x', 'y', 'width', 'height']]
            s = float(r['score'])
            if not np.isfinite(b+[s]).all() or min(b[2:]) <= 0 or not 0 <= s <= 1:
                raise ValueError('Invalid prediction')
            counts[i] = counts.get(i, 0)+1
            if counts[i] > 300:
                raise ValueError('More than 300 detections per image')
            if s >= .001:
                out.append(dict(image_id=i, category_id=0, bbox=b, score=s))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val'); p.add_argument('--preds', required=True)
    p.add_argument('--out', required=True); a = p.parse_args()
    gt = ground_truth(a.root, a.split)
    result = evaluate(gt, csv_rows(a.preds, gt))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result, indent=2))
