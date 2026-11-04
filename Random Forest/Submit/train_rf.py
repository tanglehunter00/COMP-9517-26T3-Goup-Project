"""Step 6: train RF round 1, mine hard negatives on the TRAIN images, retrain round 2.
Usage: python train_rf.py --root <dataset> --samples samples_round1.npz --workers 4
The second forest (rf_final.joblib) is the final model. No hyper-parameter search."""
import argparse, hashlib, json, os, time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from config import RF_PARAMS, N_HARD_NEG, CANVAS
from data_utils import list_images, load_canvas_and_boxes
from detector import mine_image
from sliding import patches_for_ids, batched_features

_M = {}


def _init(path):
    _M["model"] = joblib.load(path)
    _M["model"].n_jobs = 1                       # parallelism comes from the worker pool


def _cache_file(cache_dir, path):
    return os.path.join(cache_dir, os.path.basename(path) + ".npz")


def _mine(job):
    """Mine one image and store its candidates on disk (so an interrupted run can resume)."""
    path, root, cache_dir = job
    f = _cache_file(cache_dir, path)
    if os.path.exists(f):
        return 0
    canvas, gts = load_canvas_and_boxes(path, root, "train")
    sc, ids = mine_image(_M["model"], canvas, gts)
    np.savez(f + ".tmp.npz", sc=sc, ids=ids)
    os.replace(f + ".tmp.npz", f)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--samples", default="samples_round1.npz")
    ap.add_argument("--out_dir", default="models")
    ap.add_argument("--workers", type=int, default=2, help="parallel images during mining only (each worker holds a copy of the forest in RAM)")
    ap.add_argument("--mine_images", type=int, default=0, help="debug only: mine on the first N train images")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    log, T0 = {}, time.time()

    d = np.load(a.samples); X, y = d["X"], d["y"]
    t = time.time(); rf1 = RandomForestClassifier(**RF_PARAMS).fit(X, y); log["fit_round1_s"] = time.time() - t
    p1 = os.path.join(a.out_dir, "rf_round1.joblib"); joblib.dump(rf1, p1, compress=3)
    print(f"round 1 trained on {len(y)} windows in {log['fit_round1_s']:.1f}s")

    t = time.time()
    paths = list_images(a.root, "train")
    if a.mine_images:
        paths = paths[:a.mine_images]
    # cache folder is keyed by the training samples, so a different sample set never reuses old results
    key = hashlib.md5(X[::max(1, len(X) // 500)].tobytes()).hexdigest()[:8]
    cache_dir = os.path.join(a.out_dir, f"mine_cache_{key}_{len(paths)}")
    os.makedirs(cache_dir, exist_ok=True)
    workers = a.workers
    while True:
        todo = [str(p) for p in paths if not os.path.exists(_cache_file(cache_dir, str(p)))]
        if not todo:
            break
        print(f"mining {len(todo)} images with {workers} worker(s) ({len(paths) - len(todo)} already cached)", flush=True)
        try:
            with ProcessPoolExecutor(workers, initializer=_init, initargs=(p1,)) as ex:
                for k, _ in enumerate(ex.map(_mine, [(p, a.root, cache_dir) for p in todo]), 1):
                    if k % 25 == 0 or k == len(todo):
                        print(f"  {k}/{len(todo)} images, {time.time() - t:.0f}s elapsed", flush=True)
            break
        except BrokenProcessPool:
            if workers == 1:
                raise
            workers = max(1, workers // 2)                  # a worker died (probably memory): retry with fewer
            print(f"a worker process died; retrying the unfinished images with {workers} worker(s)", flush=True)
    res = []
    for p in paths:
        z = np.load(_cache_file(cache_dir, str(p))); res.append((z["sc"], z["ids"]))
    sc = np.concatenate([r[0] for r in res]); ids = np.concatenate([r[1] for r in res])
    img = np.concatenate([np.full(len(r[0]), i) for i, r in enumerate(res)])
    top = np.argsort(-sc, kind="stable")[:N_HARD_NEG]                # global top-5000 by round-1 score
    Xh = np.zeros((len(top), X.shape[1]), np.float32)
    for i in np.unique(img[top]):                                    # rebuild features of chosen windows
        sel = np.where(img[top] == i)[0]
        canvas, _ = load_canvas_and_boxes(paths[i], a.root, "train")
        Xh[sel] = batched_features(patches_for_ids(canvas, ids[top][sel]))
    log["mining_s"] = time.time() - t; log["n_hard_neg"] = int(len(top))
    log["hard_neg_score_range"] = [float(sc[top].min()), float(sc[top].max())] if len(top) else None
    print(f"mined {len(top)} hard negatives from {len(paths)} images in {log['mining_s']:.1f}s")

    X2, y2 = np.concatenate([X, Xh]), np.concatenate([y, np.zeros(len(top), np.int8)])
    t = time.time(); rf2 = RandomForestClassifier(**RF_PARAMS).fit(X2, y2); log["fit_round2_s"] = time.time() - t
    joblib.dump(rf2, os.path.join(a.out_dir, "rf_final.joblib"), compress=3)
    log["total_train_s"] = time.time() - T0; log["mined_on_images"] = len(paths)
    json.dump(log, open(os.path.join(a.out_dir, "train_log.json"), "w"), indent=1)
    print(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
