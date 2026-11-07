"""Square-640 inference, one NMS, CSV export and optional full-set benchmark."""
import argparse
import csv
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from ultralytics.utils.nms import non_max_suppression
from data import image_paths, read_image, letterbox
from model import load_model


@torch.inference_mode()
def predict_batch(model, images, device):
    processed = [letterbox(im) for im in images]
    array = np.stack([x[0][:, :, ::-1].transpose(2, 0, 1) for x in processed])
    tensor = torch.from_numpy(np.ascontiguousarray(array)).to(device).float().div_(255)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        pred = model(tensor)
    boxes = non_max_suppression(pred, conf_thres=.001, iou_thres=.6, max_det=300, nc=1, max_time_img=60)
    result = []
    for b, (_, (s, px, py, w, h)) in zip(boxes, processed):
        b = b.float().cpu().numpy()
        b[:, [0, 2]] = np.clip((b[:, [0, 2]]-px)/s, 0, w)
        b[:, [1, 3]] = np.clip((b[:, [1, 3]]-py)/s, 0, h)
        result.append(b[(b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])])
    return result


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def predict_split(model, root, split, device):
    paths = image_paths(root, split)
    rows = []
    for start in range(0, len(paths), 8):
        batch = paths[start:start+8]
        for p, boxes in zip(batch, predict_batch(model, [read_image(p) for p in batch], device)):
            for x1, y1, x2, y2, score, _ in boxes:
                rows.append(dict(image_id=p.stem, category_id=0, x=float(x1), y=float(y1),
                                 width=float(x2-x1), height=float(y2-y1), score=float(score)))
    return rows


def benchmark(model, images, device):
    if device.type != 'cuda':
        raise ValueError('Official deep-model timing requires CUDA FP16')
    batches = [images[i:i+8] for i in range(0, len(images), 8)]
    if not any(len(b) == 8 for b in batches):
        raise ValueError('Benchmark needs at least 8 images')
    for i in range(20):
        predict_batch(model, batches[0], device)
    sync(device); torch.cuda.reset_peak_memory_stats(device)
    runs, times = [], []
    for _ in range(3):
        sync(device); start = time.perf_counter()
        for batch in batches:
            sync(device); t = time.perf_counter()
            predict_batch(model, batch, device)
            sync(device)
            if len(batch) == 8:
                times.append(time.perf_counter()-t)
        sync(device); runs.append(time.perf_counter()-start)
    elapsed = sum(runs)
    return dict(device=torch.cuda.get_device_name(device), official_hardware='T4' in torch.cuda.get_device_name(device),
                images=len(images), repeats=3, warmup_batches=20, batch=8, repeat_seconds=runs,
                images_per_second=3*len(images)/elapsed, amortised_ms_per_image=1000*elapsed/(3*len(images)),
                mean_full_batch_ms=1000*float(np.mean(times)), p95_full_batch_ms=1000*float(np.percentile(times, 95)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--weights', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val'); p.add_argument('--out', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--benchmark', action='store_true')
    p.add_argument('--allow-smoke', action='store_true', help='Explicitly allow synthetic diagnostic checkpoints')
    a = p.parse_args()
    device = torch.device(a.device); model, ckpt = load_model(a.weights, device)
    if ckpt.get('smoke') and not a.allow_smoke:
        raise ValueError('Diagnostic checkpoint: pass --allow-smoke for synthetic verification only')
    if a.split == 'test' and ckpt['epoch'] != ckpt['epochs']-1 and not ckpt.get('training_complete', False):
        raise ValueError('Use best checkpoint marked training_complete after the full run')
    rows = predict_split(model, a.root, a.split, device)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['image_id','category_id','x','y','width','height','score'])
        writer.writeheader(); writer.writerows(rows)
    if a.benchmark:
        if a.split != 'test':
            raise ValueError('Runtime benchmark is defined on the test split')
        timing = benchmark(model, [read_image(p) for p in image_paths(a.root, a.split)], device)
        out.with_suffix('.timing.json').write_text(json.dumps(timing, indent=2))
    print(f'Wrote {len(rows)} predictions to {out}')
