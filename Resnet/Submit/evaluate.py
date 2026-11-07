import numpy as np
import torch
import torchvision


def evaluate_model(model, data_loader, device, score_thresh=0.001, nms_iou=0.6):
    """
    Evaluates the model on the validation/test set according to the COMP9517 protocol.
    """
    model.eval()
    cpu_device = torch.device("cpu")
    results = []

    # Ground truth annotations in COCO format for pycocotools
    # (Assuming data_loader provides coco-style annotations or we format predictions accordingly)

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                image_id = targets[i]["image_id"].item()

                boxes = output["boxes"].to(cpu_device)
                scores = output["scores"].to(cpu_device)
                labels = output["labels"].to(cpu_device)

                # Filter by score prefilter (0.001 for COCO metrics)
                keep_prefilter = scores >= score_thresh
                boxes = boxes[keep_prefilter]
                scores = scores[keep_prefilter]
                labels = labels[keep_prefilter]

                # Apply greedy hard NMS with IoU 0.6 to final candidates
                if len(boxes) > 0:
                    keep_nms = torchvision.ops.nms(boxes, scores, nms_iou)
                    boxes = boxes[keep_nms]
                    scores = scores[keep_nms]
                    labels = labels[keep_nms]

                # Enforce 300 detection cap per image
                if len(boxes) > 300:
                    topk_indices = scores.argsort(descending=True)[:300]
                    boxes = boxes[topk_indices]
                    scores = scores[topk_indices]
                    labels = labels[topk_indices]

                # Format predictions for COCO evaluation
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box.tolist()
                    w = x2 - x1
                    h = y2 - y1
                    # Convert back to original image coordinates if needed,
                    # or keep in 640x640 canvas coordinates matching ground truth.
                    results.append(
                        {
                            "image_id": int(image_id),
                            "category_id": int(label),
                            "bbox": [x1, y1, w, h],
                            "score": float(score),
                        }
                    )

    return results


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
