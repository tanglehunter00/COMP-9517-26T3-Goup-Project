import argparse
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import box_iou

from dataset import BeachBirdDataset, collate_fn, warp_boxes, warp_image
from evaluate import coco_ground_truth, coco_metrics, predict, write_csv
from model import get_faster_rcnn_model

DISTANCES = (None, 0, 0.5, 1, 2, 4)
TARGETS_PER_IMAGE, SIGMA, ANGLE = 3, 0.5, np.pi / 2
SCORE, IOU, BATCH = 0.25, 0.5, 8


def plan_warps(gt):
    by_image = {}
    for ann in gt["annotations"]:
        by_image.setdefault(ann["image_id"], []).append(ann)
    far = max(d for d in DISTANCES if d is not None)
    plans = {}
    for im in gt["images"]:
        anns = by_image.get(im["id"], [])
        rng = np.random.default_rng(20_000 + im["id"])
        chosen = []
        for i in rng.permutation(len(anns))[:TARGETS_PER_IMAGE]:
            x, y, w, h = anns[i]["bbox"]
            size, centre = max(w, h), np.array([x + w / 2, y + h / 2])
            phi = rng.uniform(0, 2 * np.pi)
            directions = [np.array([np.cos(phi + k * np.pi / 8), np.sin(phi + k * np.pi / 8)]) for k in range(16)]
            inside = [u for u in directions
                      if 0 <= (centre + far * size * u)[0] < im["width"] and 0 <= (centre + far * size * u)[1] < im["height"]]
            chosen.append(dict(ann=anns[i]["id"], centre=centre, size=size, direction=(inside or directions)[0],
                               angle=ANGLE * float(rng.choice([-1, 1]))))
        plans[im["id"]] = chosen
    return plans


def swirls_for(plan, distance):
    if distance is None:
        return []
    return [(p["centre"] + distance * p["size"] * p["direction"], SIGMA * p["size"], p["angle"]) for p in plan]


class WarpedSplit(Dataset):

    def __init__(self, base, plans, distance):
        self.base, self.plans, self.distance = base, plans, distance
        self.image_files = base.image_files

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.base.img_dir, self.image_files[idx])).convert("RGB")
        width, height = image.size
        image = warp_image(image, swirls_for(self.plans[idx], self.distance))
        canvas, scale, pad_w, pad_h = self.base._resize_and_pad(image)
        tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1)
        return tensor, {"image_id": torch.tensor([idx]),
                        "meta": torch.tensor([scale, pad_w, pad_h, width, height], dtype=torch.float64)}


def warped_ground_truth(gt, plans, distance):
    sizes = {im["id"]: (im["width"], im["height"]) for im in gt["images"]}
    annotations = []
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        (x1, y1, x2, y2), = warp_boxes([(x, y, x + w, y + h)], swirls_for(plans[ann["image_id"]], distance),
                                       *sizes[ann["image_id"]])
        annotations.append({**ann, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1)})
    return {**gt, "annotations": annotations}


def matched_birds(rows, gt):
    preds, birds = {}, {}
    for r in rows:
        if r["score"] >= SCORE:
            preds.setdefault(r["idx"], []).append((r["score"], [r["x"], r["y"], r["x"] + r["width"], r["y"] + r["height"]]))
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        birds.setdefault(ann["image_id"], []).append((ann["id"], [x, y, x + w, y + h]))
    matched = set()
    for image_id, gts in birds.items():
        dets = sorted(preds.get(image_id, []), key=lambda d: -d[0])
        if not dets:
            continue
        iou = box_iou(torch.tensor([d[1] for d in dets]), torch.tensor([g[1] for g in gts]))
        free = torch.ones(len(gts), dtype=torch.bool)
        for row in iou:
            row = torch.where(free, row, torch.full_like(row, -1.0))
            best = int(row.argmax())
            if row[best] >= IOU:
                free[best] = False
                matched.add(gts[best][0])
    return matched


def main():
    p = argparse.ArgumentParser(description="M3 local-warp robustness check (not a protocol result)")
    p.add_argument("--data", required=True, help="dataset root with images/{train,val,test} and labels/{...}")
    p.add_argument("--ckpt", required=True, help="best.pt from train.py")
    p.add_argument("--out", required=True, help="output folder")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, help="smoke tests only: first N images")
    a = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = get_faster_rcnn_model(pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    base = BeachBirdDataset(os.path.join(a.data, "images", a.split), os.path.join(a.data, "labels", a.split))
    if a.limit:
        base.image_files = base.image_files[:a.limit]
    gt = coco_ground_truth(base)
    plans = plan_warps(gt)
    targets = {p["ann"] for plan in plans.values() for p in plan}

    results = {}
    for distance in DISTANCES:
        name = "none" if distance is None else f"d{distance:g}"
        ds = WarpedSplit(base, plans, distance)
        loader = DataLoader(ds, batch_size=BATCH, num_workers=2, collate_fn=collate_fn)
        rows = predict(model, loader, device)
        write_csv(out / f"preds_{a.split}_warp_{name}.csv", rows)
        warped_gt = warped_ground_truth(gt, plans, distance)
        hit = matched_birds(rows, warped_gt)
        n_birds = len(warped_gt["annotations"])
        results[name] = dict(distance=distance, **coco_metrics(ds, rows, gt=warped_gt),
                             target_recall=len(hit & targets) / max(len(targets), 1),
                             other_recall=len(hit - targets) / max(n_birds - len(targets), 1))
        print(name, json.dumps(results[name]), flush=True)

    config = Path(a.ckpt).with_name("config.json")
    summary = dict(
        method=json.loads(config.read_text())["method"] if config.exists() else "M3 Faster R-CNN ResNet50-FPN V2",
        checkpoint_epoch=ckpt.get("epoch"), split=a.split, images=len(base.image_files),
        birds=len(gt["annotations"]), target_birds=len(targets), targets_per_image=TARGETS_PER_IMAGE,
        warp=dict(type="swirl", sigma_x_bird_size=SIGMA, peak_angle_rad=round(ANGLE, 4),
                  distances_x_bird_size=[d for d in DISTANCES if d is not None]),
        recall_score_threshold=SCORE, recall_iou=IOU, results=results,
        device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        precision="AMP FP16" if device.type == "cuda" else "FP32",
        torch=torch.__version__, torchvision=torchvision.__version__,
    )
    (out / "warp_metrics.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
