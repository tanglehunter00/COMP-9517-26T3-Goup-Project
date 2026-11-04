"""Step 2: data sanity check. Only REPORTS problems; never modifies the official split.
Usage: python check_data.py --root "<dataset folder>" [--out data_check.json]"""
import argparse, hashlib, json
from collections import defaultdict
import cv2, numpy as np
from config import SPLITS, split_dirs
from data_utils import list_images, read_yolo_labels, imread


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="data_check.json")
    a = ap.parse_args()

    report = {"splits": {}, "problems": [], "cross_split_duplicates": []}
    hashes = defaultdict(list)                       # md5 of file bytes -> [(split, name)]
    for split in SPLITS:
        _, lab_dir = split_dirs(a.root, split)
        n_img = n_box = n_neg = 0
        for p in list_images(a.root, split):
            n_img += 1
            hashes[hashlib.md5(p.read_bytes()).hexdigest()].append((split, p.name))
            img = imread(p)
            if img is None:
                report["problems"].append(f"{split}/{p.name}: unreadable image"); continue
            if img.shape[:2] != (512, 512):
                report["problems"].append(f"{split}/{p.name}: size {img.shape[:2]}")
            lp = lab_dir / (p.stem + ".txt")
            if not lp.exists():
                report["problems"].append(f"{split}/{p.name}: missing label file"); continue
            lab = read_yolo_labels(lp)
            n_box += len(lab); n_neg += int(len(lab) == 0)
            for row in lab:
                c, xc, yc, w, h = row
                if c != 0:
                    report["problems"].append(f"{split}/{p.stem}: class id {c}")
                if w <= 0 or h <= 0:
                    report["problems"].append(f"{split}/{p.stem}: non-positive box size")
                if xc - w/2 < -1e-6 or yc - h/2 < -1e-6 or xc + w/2 > 1+1e-6 or yc + h/2 > 1+1e-6:
                    report["problems"].append(f"{split}/{p.stem}: box outside image")
        report["splits"][split] = {"images": n_img, "boxes": n_box, "empty_label_images": n_neg}
    for h, items in hashes.items():
        if len(items) > 1:
            report["cross_split_duplicates"].append(items)
    report["n_cross_split_exact_duplicate_groups"] = sum(
        1 for it in report["cross_split_duplicates"] if len({s for s, _ in it}) > 1)
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in report.items() if k != "cross_split_duplicates"},
                     indent=1, ensure_ascii=False)[:3000])


if __name__ == "__main__":
    main()
