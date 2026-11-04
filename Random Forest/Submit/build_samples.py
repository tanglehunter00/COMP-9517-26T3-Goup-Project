"""Steps 3-5: build the training set (positives + random negatives) and extract features.
Only the TRAIN split is used. Output: samples_round1.npz (X, y, meta).
Usage: python build_samples.py --root "<dataset folder>" [--workers 4] [--limit_images 0]

Assumption (confirm with the group): negative windows take their size from the same
window shapes x pyramid scales used at inference, expressed in canvas pixels
(w/scale, h/scale), so the classifier sees backgrounds at the sizes it will be tested on."""
import argparse, time
from multiprocessing import Pool
import numpy as np
from config import (SEED, CONTEXT, N_NEGATIVES, NEG_IOU_MAX, CANVAS, PYRAMID_SCALES,
                    WINDOW_SHAPES)
from data_utils import list_images, load_canvas_and_boxes, iou_one_to_many
from features import patch_features, crop_window


def expand(box, ctx=CONTEXT):
    w, h = box[2] - box[0], box[3] - box[1]
    return np.array([box[0] - ctx * w, box[1] - ctx * h, box[2] + ctx * w, box[3] + ctx * h])


def sample_negatives(gt_per_image, n, rng):
    """Draw n windows (img_idx, x1, y1, x2, y2) with IoU < NEG_IOU_MAX vs ALL GT boxes
    of that image. Images are drawn uniformly (empty images included)."""
    sizes = [(w / s, h / s) for s in PYRAMID_SCALES for (w, h) in WINDOW_SHAPES]
    out, tries = [], 0
    while len(out) < n:
        tries += 1
        if tries > n * 200:
            raise RuntimeError("could not sample enough negatives")
        i = int(rng.integers(len(gt_per_image)))
        w, h = sizes[int(rng.integers(len(sizes)))]
        x1 = rng.uniform(0, CANVAS - w); y1 = rng.uniform(0, CANVAS - h)
        win = np.array([x1, y1, x1 + w, y1 + h])
        if np.all(iou_one_to_many(win, gt_per_image[i]) < NEG_IOU_MAX):
            out.append((i, *win))
    return out


def extract_job(job):
    """job = (img_path, root, split, [(label, x1, y1, x2, y2), ...]) -> list of (label, feat)."""
    img_path, root, split, wins = job
    canvas, _ = load_canvas_and_boxes(img_path, root, split)
    res = []
    for lab, x1, y1, x2, y2 in wins:
        crop = crop_window(canvas, x1, y1, x2, y2)
        if crop is not None:
            res.append((lab, patch_features(crop)))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit_images", type=int, default=0, help="debug only: use first N train images")
    ap.add_argument("--n_neg", type=int, default=N_NEGATIVES)
    ap.add_argument("--out", default="samples_round1.npz")
    a = ap.parse_args()

    t0 = time.time()
    rng = np.random.default_rng(SEED)
    paths = list_images(a.root, "train")
    if a.limit_images:
        paths = paths[:a.limit_images]
    gts = [load_canvas_and_boxes(p, a.root, "train")[1] for p in paths]

    jobs = {i: [] for i in range(len(paths))}
    n_pos = 0
    for i, g in enumerate(gts):                               # positives: GT box + 10 % context
        for b in g:
            jobs[i].append((1, *expand(b))); n_pos += 1
    for i, x1, y1, x2, y2 in sample_negatives(gts, a.n_neg, rng):   # negatives
        jobs[i].append((0, x1, y1, x2, y2))

    job_list = [(str(paths[i]), a.root, "train", w) for i, w in jobs.items() if w]
    with Pool(a.workers) as pool:
        results = pool.map(extract_job, job_list, chunksize=8)
    ys, Xs = [], []
    for r in results:
        for lab, f in r:
            ys.append(lab); Xs.append(f)
    X, y = np.stack(Xs), np.array(ys, dtype=np.int8)
    np.savez_compressed(a.out, X=X, y=y)
    print(f"images={len(paths)} positives={int(y.sum())} negatives={int((y == 0).sum())} "
          f"dim={X.shape[1]} time={time.time() - t0:.1f}s -> {a.out}")


if __name__ == "__main__":
    main()
