import numpy as np
import torch
import torchvision


import contextlib
import csv
import io
import os

from PIL import Image


@torch.inference_mode()
def predict(model, data_loader, device):
    """Detections for every image, mapped from the 640 letterbox canvas back to original-image pixels.
    The model already applies the protocol post-processing (score >= 0.001, one greedy NMS at IoU 0.6,
    max 300 per image), so no second suppression is run here."""
    model.eval()
    dataset = data_loader.dataset
    rows = []
    for images, targets in data_loader:
        images = [img.to(device, non_blocking=True).float().div_(255) for img in images]
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            outputs = model(images)
        for output, target in zip(outputs, targets):
            scale, pad_w, pad_h, width, height = target["meta"].tolist()
            idx = int(target["image_id"])
            boxes = output["boxes"].float().cpu()
            boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - pad_w) / scale).clamp(0, width)
            boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - pad_h) / scale).clamp(0, height)
            for (x1, y1, x2, y2), score in zip(boxes.tolist(), output["scores"].float().cpu().tolist()):
                rows.append({"idx": idx, "image_id": os.path.splitext(dataset.image_files[idx])[0],
                             "category_id": 0, "x": x1, "y": y1, "width": x2 - x1, "height": y2 - y1,
                             "score": score})
    return rows


def write_csv(path, rows):
    """Protocol export format: image_id (file stem), category_id (0 = bird), x, y, width, height, score."""
    fields = ["image_id", "category_id", "x", "y", "width", "height", "score"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def coco_ground_truth(dataset):
    """COCO-format ground truth in original-image pixels, read straight from the YOLO label files."""
    images, annotations = [], []
    for idx, name in enumerate(dataset.image_files):
        with Image.open(os.path.join(dataset.img_dir, name)) as im:
            width, height = im.size
        images.append({"id": idx, "width": width, "height": height, "file_name": name})
        label_path = os.path.join(dataset.label_dir, os.path.splitext(name)[0] + ".txt")
        if not os.path.exists(label_path):
            continue
        with open(label_path) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                xc, yc, w, h = (float(v) for v in parts[1:5])
                x1, y1 = max(0.0, (xc - w / 2) * width), max(0.0, (yc - h / 2) * height)
                x2, y2 = min(width, (xc + w / 2) * width), min(height, (yc + h / 2) * height)
                if x2 > x1 and y2 > y1:
                    annotations.append({"id": len(annotations) + 1, "image_id": idx, "category_id": 1,
                                        "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1),
                                        "iscrowd": 0})
    return {"images": images, "annotations": annotations, "categories": [{"id": 1, "name": "bird"}]}


def coco_metrics(dataset, rows, gt=None):
    """AP50, AP50:95, AP75, AP-small and AR@300 with maxDets [1, 10, 300]. Reads the 300-cap arrays
    directly because COCOeval.summarize() hard-codes a 100 cap."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    keys = ["AP50", "AP50:95", "AP75", "AP_small", "AR300", "AR_small"]
    if not rows:
        return dict.fromkeys(keys, 0.0)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt or coco_ground_truth(dataset)
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes([{"image_id": r["idx"], "category_id": 1, "score": r["score"],
                                    "bbox": [r["x"], r["y"], r["width"], r["height"]]} for r in rows])
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
    precision, recall = ev.eval["precision"], ev.eval["recall"]  # [T,R,K,A,M], [T,K,A,M]

    def mean(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else None  # None = N/A (e.g. no small objects)

    return {"AP50": mean(precision[0, :, 0, 0, 2]), "AP50:95": mean(precision[:, :, 0, 0, 2]),
            "AP75": mean(precision[5, :, 0, 0, 2]), "AP_small": mean(precision[:, :, 0, 1, 2]),
            "AR300": mean(recall[:, 0, 0, 2]), "AR_small": mean(recall[:, 0, 1, 2])}


def compute_operating_point_metrics(
    results, gt_annotations, iou_thresh=0.5, score_thresh=0.25
):
    """
    Computes Precision, Recall, F1, and False Positives per Negative Image at score threshold 0.25 and IoU 0.5.
    """
    # Filter predictions by score threshold 0.25
    filtered_results = [r for r in results if r["score"] >= score_thresh]

    # Group predictions and ground truths by image_id
    preds_by_img = {}
    for p in filtered_results:
        preds_by_img.setdefault(p["image_id"], []).append(p)

    gts_by_img = {}
    for gt in gt_annotations:
        gts_by_img.setdefault(gt["image_id"], []).append(gt)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    fp_negative_images = 0
    negative_image_count = 0

    for img_id, gts in gts_by_img.items():
        preds = preds_by_img.get(img_id, [])
        is_negative = len(gts) == 0

        if is_negative:
            negative_image_count += 1
            if len(preds) > 0:
                fp_negative_images += len(preds)
                false_positives += len(preds)
            continue

        gt_boxes = np.array([gt["bbox"] for gt in gts])  # [x, y, w, h]
        # Convert [x, y, w, h] to [x1, y1, x2, y2] for IoU calculation
        gt_boxes_xyxy = np.copy(gt_boxes)
        gt_boxes_xyxy[:, 2] += gt_boxes_xyxy[:, 0]
        gt_boxes_xyxy[:, 3] += gt_boxes_xyxy[:, 1]

        matched_gts = set()

        # Sort predictions by score descending
        preds = sorted(preds, key=lambda x: x["score"], reverse=True)

        for pred in preds:
            pred_box = pred["bbox"]
            pred_box_xyxy = [
                pred_box[0],
                pred_box[1],
                pred_box[0] + pred_box[2],
                pred_box[1] + pred_box[3],
            ]

            # Compute IoUs with all ground truths in the image
            best_iou = 0.0
            best_gt_idx = -1
            for idx, gt_box in enumerate(gt_boxes_xyxy):
                # Calculate IoU
                iou = box_iou_single(pred_box_xyxy, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou >= iou_thresh and best_gt_idx not in matched_gts:
                true_positives += 1
                matched_gts.add(best_gt_idx)
            else:
                false_positives += 1

        false_negatives += len(gts) - len(matched_gts)

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    fp_per_neg_image = (
        fp_negative_images / negative_image_count if negative_image_count > 0 else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fp_per_negative_image": fp_per_neg_image,
    }


def box_iou_single(box1, box2):
    """Helper to calculate IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0
