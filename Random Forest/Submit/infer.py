"""Step 7: sliding-window inference on a split + timing.
Usage: python infer.py --root <dataset> --model models/rf_final.joblib --split val --out preds_val.csv
Output CSV columns: image_id,category_id,x,y,width,height,score  (boxes in ORIGINAL image pixels).
Timing follows the protocol: images pre-read into RAM, warm-up, full pass repeated --repeats times,
each image processed alone (letterbox + windows + features + RF + NMS + coordinate restore).
Use --workers >1 only for a fast prediction pass: timing is then NOT recorded."""
import argparse, json, os, platform, time
from multiprocessing import Pool
import cv2, joblib, numpy as np
from config import SPLITS
from data_utils import list_images, letterbox, imread
from detector import detect

_M = {}


def cpu_name():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        try:
            import resource
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except ImportError:
            return float("nan")


def run_one(model, img, category_id):
    """BGR original image -> list of (x, y, w, h, score) in original pixel coordinates."""
    h, w = img.shape[:2]
    canvas, s, (px, py) = letterbox(img)
    boxes, scores = detect(model, canvas)
    rows = []
    for (x1, y1, x2, y2), sc in zip(boxes, scores):
        x1, x2 = np.clip([(x1 - px) / s, (x2 - px) / s], 0, w)
        y1, y2 = np.clip([(y1 - py) / s, (y2 - py) / s], 0, h)
        if x2 > x1 and y2 > y1:
            rows.append((x1, y1, x2 - x1, y2 - y1, float(sc)))
    return rows


def _init(path):
    _M["m"] = joblib.load(path); _M["m"].n_jobs = 1


def _work(arg):
    return run_one(_M["m"], arg, 0)        # category id is only written when saving the CSV


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", default="models/rf_final.joblib")
    ap.add_argument("--split", default="val", choices=SPLITS)
    ap.add_argument("--out", default="preds.csv")
    ap.add_argument("--category_id", type=int, default=0, help="confirm with the evaluator owner")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=20, help="images processed once and discarded")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="debug only: first N images")
    a = ap.parse_args()

    paths = list_images(a.root, a.split)
    if a.limit:
        paths = paths[:a.limit]
    imgs = [imread(p) for p in paths]          # pre-read: no disk I/O in timing
    t_start = time.time()

    if a.workers > 1:
        with Pool(a.workers, initializer=_init, initargs=(a.model,)) as pool:
            preds = pool.map(_work, imgs, chunksize=1)
        timing = None
        print("workers>1: predictions only, timing not recorded")
    else:
        model = joblib.load(a.model)                                       # protocol: n_jobs=2 (saved in model)
        for im in imgs[:min(a.warmup, len(imgs))]:
            run_one(model, im, a.category_id)
        runs, preds, peak = [], None, 0.0
        for rep in range(a.repeats):
            per, out = [], []
            t_rep = time.perf_counter()
            for im in imgs:
                t = time.perf_counter(); out.append(run_one(model, im, a.category_id))
                per.append(time.perf_counter() - t); peak = max(peak, rss_mb())
            tot = time.perf_counter() - t_rep
            runs.append({"total_s": tot, "mean_per_image_s": float(np.mean(per)),
                         "p95_per_image_s": float(np.percentile(per, 95)), "images_per_s": len(imgs) / tot})
            preds = preds or out
            print(f"repeat {rep + 1}/{a.repeats}: {tot:.1f}s total, {np.mean(per):.2f}s/img")
        timing = {"cpu": cpu_name(), "cpu_count": os.cpu_count(), "rf_n_jobs": model.n_jobs,
                  "n_images": len(imgs), "warmup_images": min(a.warmup, len(imgs)),
                  "repeats": runs, "peak_rss_mb": peak}

    with open(a.out, "w") as f:
        f.write("image_id,category_id,x,y,width,height,score\n")
        for p, rows in zip(paths, preds):
            for x, y, w, h, s in rows:
                f.write(f"{p.stem},{a.category_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{s:.5f}\n")
    if timing:
        json.dump(timing, open(os.path.splitext(a.out)[0] + "_timing.json", "w"), indent=1)
    print(f"{sum(len(r) for r in preds)} detections on {len(paths)} images -> {a.out} "
          f"({time.time() - t_start:.0f}s wall)")


if __name__ == "__main__":
    main()
