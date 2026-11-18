import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision

from dataset import make_loader
from evaluate import coco_ground_truth, coco_metrics, predict, write_csv
from model import PROTOCOL_KWARGS, get_faster_rcnn_model

SEED, EPOCHS = 42, 100
BATCH, ACCUM, VAL_BATCH = 2, 2, 8
PEAK_LR, MOMENTUM, WEIGHT_DECAY, WARMUP_EPOCHS = 0.0025, 0.9, 0.0001, 1


def lr_at(update, total, warmup):
    if update < warmup:
        return PEAK_LR * (0.1 + 0.9 * update / warmup)
    progress = (update - warmup) / max(total - warmup - 1, 1)
    return PEAK_LR * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * progress)))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_atomic(path, obj):
    tmp = Path(str(path) + ".tmp")
    if isinstance(obj, str):
        tmp.write_text(obj)
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().to("cpu", copy=True)
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(to_cpu(v) for v in obj)
    return obj


class Writer:

    def __init__(self):
        from concurrent.futures import ThreadPoolExecutor
        self.pool, self.pending = ThreadPoolExecutor(max_workers=1), []

    def wait(self):
        for f in self.pending:
            f.result()
        self.pending = []

    def save(self, path, obj):
        self.pending.append(self.pool.submit(save_atomic, path, obj))


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True, help="dataset root with images/{train,val,test} and labels/{...}")
    p.add_argument("--out", required=True, help="run folder (Google Drive on Colab); re-run to resume")
    p.add_argument("--epochs", type=int, default=EPOCHS, help="smoke tests only; the protocol is 100")
    p.add_argument("--limit", type=int, help="smoke tests only: first N images per split")
    p.add_argument("--robust", action="store_true",
                   help="NON-PROTOCOL variant: add colour jitter, blur, noise and JPEG distortion to training")
    a = p.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "DONE").exists():
        print("Run already complete:", (out / "DONE").read_text())
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and a.epochs == EPOCHS:
        raise SystemExit("The protocol run needs a CUDA GPU (Colab T4). Use --epochs/--limit for CPU smoke tests.")

    seed_all(SEED)

    torch.backends.cudnn.benchmark = True

    def loader(split, batch, train=False, generator=None, corrupt=False):
        dl = make_loader(a.data, split, batch, train=train, generator=generator, robust=a.robust, corrupt=corrupt)
        if a.limit:
            dl.dataset.image_files = dl.dataset.image_files[:a.limit]
        return dl

    generator = torch.Generator()
    train_loader, val_loader = loader("train", BATCH, True, generator), loader("val", VAL_BATCH)
    val_gt = coco_ground_truth(val_loader.dataset)
    n_train = len(train_loader.dataset)
    batches = math.ceil(n_train / BATCH)
    updates_per_epoch = math.ceil(batches / ACCUM)
    total, warmup = a.epochs * updates_per_epoch, WARMUP_EPOCHS * updates_per_epoch

    model = get_faster_rcnn_model(pretrained=True).to(device)
    params = [q for q in model.parameters() if q.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr_at(0, total, warmup), momentum=MOMENTUM, nesterov=False,
                                weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    start, updates, best, history = 0, 0, (-1.0, -1.0), []
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        if (ck["epochs"], ck["n_train"], ck.get("robust", False)) != (a.epochs, n_train, a.robust):
            raise SystemExit(f"{last} is from a different run configuration; use a new --out")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        start, updates, best, history = ck["epoch"], ck["updates"], tuple(ck["best"]), ck["history"]
        print(f"Resumed after epoch {start}/{a.epochs} ({updates} updates, best AP50/AP50:95 {best})", flush=True)

    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    config = dict(
        method="M3-robust (NON-PROTOCOL: distortion augmentation)" if a.robust
        else "M3 Faster R-CNN ResNet50-FPN V2", protocol=not a.robust, seed=SEED, epochs=a.epochs, limit=a.limit, data=a.data,
        train_images=n_train, train_batch=BATCH, accumulation=ACCUM, effective_batch=BATCH * ACCUM,
        validation_batch=VAL_BATCH, optimizer="SGD", peak_lr=PEAK_LR, final_lr=PEAK_LR * 0.01,
        momentum=MOMENTUM, nesterov=False, weight_decay=WEIGHT_DECAY, warmup_epochs=WARMUP_EPOCHS,
        updates_per_epoch=updates_per_epoch, total_updates=total, warmup_updates=warmup,
        precision="AMP FP16" if device.type == "cuda" else "FP32", augmentation="horizontal flip 0.5" + (
            "; p0.8 brightness/contrast/saturation x[0.6,1.4]; p0.25 each: gaussian blur r[0.5,2],"
            " gaussian noise sd[3,15], JPEG q[20,70]" if a.robust else ""),
        trainable_backbone_layers=3, frozen_backbone_batchnorm=True, weights="COCO_V1 (dd69338a)",
        model_kwargs=PROTOCOL_KWARGS, trainable_params=sum(q.numel() for q in params),
        device=gpu, torch=torch.__version__, torchvision=torchvision.__version__, python=platform.python_version(),
        nondeterminism="cudnn.benchmark=True; RoIAlign/scatter backward on CUDA are nondeterministic",
    )
    save_atomic(out / "config.json", json.dumps(config, indent=2))
    save_atomic(out / "model.txt", str(model))

    writer = Writer()
    run_start = time.perf_counter()
    for epoch in range(start, a.epochs):

        seed_all(SEED + epoch)
        generator.manual_seed(SEED + epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.perf_counter()
        loss_sums, seen = {}, 0
        for i, (images, targets) in enumerate(train_loader):
            images = [im.to(device, non_blocking=True).float().div_(255) for im in images]
            targets = [{"boxes": t["boxes"].to(device, non_blocking=True),
                        "labels": t["labels"].to(device, non_blocking=True)} for t in targets]

            group_images = min(BATCH * ACCUM, n_train - (i // ACCUM) * BATCH * ACCUM)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
            loss = sum(loss_dict.values())


            scaler.scale(loss * len(images) / group_images).backward()
            for k, v in loss_dict.items():
                loss_sums[k] = loss_sums.get(k, 0.0) + v.detach() * len(images)
            seen += len(images)
            if i % ACCUM == ACCUM - 1 or i == len(train_loader) - 1:
                lr = lr_at(updates, total, warmup)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        losses = {k: float(v) / seen for k, v in loss_sums.items()}
        if not all(math.isfinite(v) for v in losses.values()):
            raise RuntimeError(f"Non-finite training loss in epoch {epoch + 1}: {losses}")
        sync(device)
        train_s = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0

        t0 = time.perf_counter()
        metrics = coco_metrics(val_loader.dataset, predict(model, val_loader, device), gt=val_gt)
        sync(device)
        val_s = time.perf_counter() - t0


        key = (metrics["AP50"] or 0.0, metrics["AP50:95"] or 0.0)
        improved = key > best
        best = max(best, key)
        writer.wait()
        state = to_cpu(model.state_dict())
        log = dict(epoch=epoch + 1, loss=sum(losses.values()), **losses, lr=lr, updates=updates,
                   loss_scale=scaler.get_scale(), train_s=round(train_s, 1), val_s=round(val_s, 1),
                   peak_gpu_mb=round(peak_mb), **metrics, best=improved)
        history.append(log)


        if improved:
            writer.save(out / "best.pt", dict(model=state, epoch=epoch + 1, metrics=metrics))
        if (epoch + 1) % 10 == 0:
            writer.save(out / f"epoch{epoch + 1:03d}.pt", dict(model=state, epoch=epoch + 1, metrics=metrics))
        writer.save(last, dict(model=state, optimizer=to_cpu(optimizer.state_dict()), scaler=scaler.state_dict(),
                               epoch=epoch + 1, epochs=a.epochs, n_train=n_train, robust=a.robust, updates=updates,
                               best=best, history=list(history)))
        writer.save(out / "metrics.jsonl", "".join(json.dumps(h) + "\n" for h in history))
        print(json.dumps(log), flush=True)

    writer.wait()

    best_ck = torch.load(out / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(best_ck["model"])
    final = {"best_epoch": best_ck["epoch"], "train_wall_s": round(time.perf_counter() - run_start, 1)}


    for split in ("val", "test"):
        for corrupt in (False, True):
            name = split + ("_distorted" if corrupt else "")
            dl = loader(split, VAL_BATCH, corrupt=corrupt)
            rows = predict(model, dl, device)
            write_csv(out / f"preds_{name}.csv", rows)
            final[name] = coco_metrics(dl.dataset, rows)
    save_atomic(out / "final_metrics.json", json.dumps(final, indent=2))
    save_atomic(out / "DONE", json.dumps(final))
    print("DONE", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()


    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
