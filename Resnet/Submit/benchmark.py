import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image

from dataset import BeachBirdDataset
from model import get_faster_rcnn_model

BATCH, WARMUP_BATCHES, REPEATS = 8, 20, 3


def load_images(data, split, limit=None):
    ds = BeachBirdDataset(os.path.join(data, "images", split), os.path.join(data, "labels", split))
    files = ds.image_files[:limit] if limit else ds.image_files
    images = []
    for f in files:
        with Image.open(os.path.join(ds.img_dir, f)) as im:
            images.append(im.convert("RGB"))
    return ds, images


@torch.inference_mode()
def run_batch(model, ds, images, device):
    canvases, metas = [], []
    for im in images:
        canvas, scale, pad_w, pad_h = ds._resize_and_pad(im)
        canvases.append(torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1))
        metas.append((scale, pad_w, pad_h, im.width, im.height))
    batch = torch.stack(canvases).to(device, non_blocking=True).float().div_(255)
    with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        outputs = model(list(batch))
    n = 0
    for out, (scale, pad_w, pad_h, width, height) in zip(outputs, metas):
        boxes = out["boxes"].float()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_w) / scale).clamp(0, width)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_h) / scale).clamp(0, height)
        boxes, scores = boxes.cpu(), out["scores"].float().cpu()
        n += len(scores)
    return n


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def batch_stats(times_s):
    ms = np.asarray(times_s) * 1000
    return dict(mean_batch_ms=round(float(ms.mean()), 2), p95_batch_ms=round(float(np.percentile(ms, 95)), 2))


def main():
    p = argparse.ArgumentParser(description="M3 runtime benchmark (protocol section 7)")
    p.add_argument("--data", required=True, help="dataset root with images/{train,val,test} and labels/{...}")
    p.add_argument("--ckpt", required=True, help="best.pt from train.py")
    p.add_argument("--out", required=True, help="output JSON file")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, help="smoke tests only: first N images")
    p.add_argument("--warmup", type=int, default=WARMUP_BATCHES, help="smoke tests only; the protocol is 20")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not a.limit:
        raise SystemExit("The protocol benchmark needs a CUDA GPU (Colab T4). Use --limit for CPU smoke tests.")
    torch.backends.cudnn.benchmark = True

    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = get_faster_rcnn_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    ds, images = load_images(a.data, a.split, a.limit)
    batches = [images[i:i + BATCH] for i in range(0, len(images), BATCH)]

    for i in range(a.warmup):
        run_batch(model, ds, batches[i % len(batches)], device)
    sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    repeats, full_batch_times, detections = [], [], None
    for _ in range(REPEATS):
        times, n_det = [], 0
        sync(device)
        start = time.perf_counter()
        for b in batches:
            t0 = time.perf_counter()
            n_det += run_batch(model, ds, b, device)
            sync(device)
            dt = time.perf_counter() - t0
            if len(b) == BATCH:
                times.append(dt)
        total = time.perf_counter() - start
        detections = n_det if detections is None else detections
        full_batch_times += times
        repeats.append(dict(total_s=round(total, 3), images_per_s=round(len(images) / total, 2),
                            ms_per_image=round(1000 * total / len(images), 2), **batch_stats(times)))

    total_s = sum(r["total_s"] for r in repeats)
    config = Path(a.ckpt).with_name("config.json")
    method = json.loads(config.read_text())["method"] if config.exists() else "M3 Faster R-CNN ResNet50-FPN V2"
    result = dict(
        method=method, checkpoint=str(a.ckpt), checkpoint_epoch=ckpt.get("epoch"),
        split=a.split, n_images=len(images), batch=BATCH, full_batches_per_repeat=len(images) // BATCH,
        warmup_batches=a.warmup, repeats=repeats,
        aggregate=dict(images=REPEATS * len(images), total_s=round(total_s, 3),
                       images_per_s=round(REPEATS * len(images) / total_s, 2),
                       ms_per_image=round(1000 * total_s / (REPEATS * len(images)), 2), **batch_stats(full_batch_times)),
        peak_gpu_mb=round(torch.cuda.max_memory_allocated(device) / 2**20) if device.type == "cuda" else None,
        detections_per_repeat=detections,
        timed=("letterbox preprocessing, host-to-device transfer, forward pass incl. RPN/RoI NMS, "
               "coordinate restoration to original pixels, device-to-host copy; images preloaded in RAM"),
        precision="AMP FP16" if device.type == "cuda" else "FP32", inference_mode=True, cudnn_benchmark=True,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        torch=torch.__version__, torchvision=torchvision.__version__, python=platform.python_version(),
    )
    Path(a.out).write_text(json.dumps(result, indent=1))
    print("BENCHMARK", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
