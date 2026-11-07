"""Quick self-check of AP50 (COCO-style 101-point interpolation, IoU 0.5, greedy matching).
NOT the group's official evaluator; only for sanity-checking M1 output.
Usage: python quick_eval.py --root <dataset> --split val --preds preds_val.csv [--limit N]"""
import argparse, csv
from collections import defaultdict
import cv2, numpy as np
from data_utils import list_images, read_yolo_labels, imread
from config import split_dirs
from sliding import iou_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--split", default="val")
    ap.add_argument("--preds", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    paths = list_images(a.root, a.split)[:a.limit or None]
    _, lab_dir = split_dirs(a.root, a.split)
    gt = {}
    for p in paths:
        h, w = imread(p).shape[:2]
        l = read_yolo_labels(lab_dir / (p.stem + ".txt"))
        gt[p.stem] = np.stack([(l[:, 1] - l[:, 3] / 2) * w, (l[:, 2] - l[:, 4] / 2) * h,
                               (l[:, 1] + l[:, 3] / 2) * w, (l[:, 2] + l[:, 4] / 2) * h], 1) if len(l) else np.zeros((0, 4))
    dets = defaultdict(list)
    for r in csv.DictReader(open(a.preds)):
        if r["image_id"] in gt:
            x, y, w, h = (float(r[k]) for k in ("x", "y", "width", "height"))
            dets[r["image_id"]].append((float(r["score"]), x, y, x + w, y + h))
    recs, n_gt = [], sum(len(v) for v in gt.values())
    for img, g in gt.items():
        d = sorted(dets.get(img, []), reverse=True); used = np.zeros(len(g), bool)
        if not d:
            continue
        ious = iou_matrix(np.array([x[1:] for x in d]), g)
        for k, (s, *_b) in enumerate(d):
            j, best = -1, 0.5
            for gi in range(len(g)):
                if not used[gi] and ious[k, gi] >= best:
                    j, best = gi, ious[k, gi]
            if j >= 0:
                used[j] = True
            recs.append((s, j >= 0))
    recs.sort(key=lambda t: -t[0])
    tp = np.cumsum([r[1] for r in recs]); fp = np.cumsum([not r[1] for r in recs])
    rec = tp / max(n_gt, 1); prec = tp / np.maximum(tp + fp, 1)
    ap101 = np.mean([prec[rec >= t].max() if (rec >= t).any() else 0 for t in np.linspace(0, 1, 101)]) if len(recs) else 0
    print(f"images={len(gt)} gt_boxes={n_gt} detections={len(recs)} final_recall={rec[-1] if len(recs) else 0:.3f} AP50={ap101:.4f}")


if __name__ == "__main__":
    main()
