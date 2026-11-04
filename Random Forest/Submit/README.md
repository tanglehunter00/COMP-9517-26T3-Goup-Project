# M1: Multi-scale sliding window + HOG/colour features + Random Forest

**Owner:** [your name] | **Group:** Group 104 | **Course:** COMP9517 2026 T3

## Summary of work completed

- **Data preparation:** Verified the official Beach Bird Detection Dataset split (train 1,369 / val 291 / test 240 images; 8,663 / 1,843 / 1,518 boxes). No unreadable images, out-of-range boxes or exact cross-split duplicates were found. Images are 512x512 with YOLO-format labels; they are letterboxed to a 640x640 canvas (pad value 114).
- **Training samples (train split only):** 8,663 positives (each GT box expanded by 10% context) and 20,000 random negatives with IoU < 0.1 to every GT box. Each window is resized to 64x64 and described by HOG (1,764-d) plus an 8-bin-per-channel HSV histogram (24-d), 1,788-d in total.
- **Model:** Random Forest (300 trees, max depth 20, min_samples_leaf 2, sqrt features, balanced class weights, seed 42) trained in two rounds: round 1, then 5,000 hard negatives mined on all training images (highest-scoring windows with IoU < 0.1), then retraining with identical parameters.
- **Inference:** Sliding windows over a 4-level pyramid (scales 1.0, 0.8, 0.64, 0.512), window shapes 32x32, 48x32, 32x48 and 64x64, stride 8 (about 50,968 windows per image). Score threshold 0.001, greedy hard NMS at IoU 0.6, at most 300 boxes per image. Boxes are written in original 512x512 image pixels.
- **Predictions generated** for the validation and test sets, and **timing measured** on the full test set.

## Training and inference time

Hardware: Intel Core i7-12700H laptop CPU (20 logical threads). The random forest runs on CPU only.

| Stage | Time |
|---|---|
| Sample construction (4 worker processes) | 24.0 s |
| Forest round 1 (n_jobs=2) | 115.6 s |
| Hard-negative mining, 1,369 images (4 worker processes) | 7,894.9 s |
| Forest round 2 (n_jobs=2) | 132.4 s |
| **Total training (rounds + mining)** | **8,145.6 s** |

Test-set inference (240 images, single process, RF n_jobs=2, images pre-loaded in RAM, 20 warm-up images, 3 full repeats; timing includes letterbox, windowing, feature extraction, RF scoring, NMS and coordinate restoration):

| Metric | Value |
|---|---|
| Total time per repeat | 4,370.9 s / 4,337.1 s / 4,322.6 s (mean 4,343.5 s) |
| Mean time per image | about 18.1 s |
| P95 time per image | about 18.8 s |
| Throughput | about 0.055 images/s |
| Peak process memory (RSS) | 533 MB |

Official metrics (AP50, AP50:95, AP75, etc.) are computed by the shared group evaluator from the prediction files listed below.

## Files submitted

**Code**
- `config.py`: protocol constants (seed, pyramid, windows, RF parameters, NMS)
- `data_utils.py`: label parsing, 640x640 letterbox, IoU
- `features.py`: reference HOG + HSV feature extraction for one patch
- `sliding.py`: image pyramid, window generation, batched feature extraction, NMS
- `detector.py`: detection and hard-negative mining for one image
- `check_data.py`: dataset sanity check
- `build_samples.py`: positive/negative sample construction and features
- `train_rf.py`: two-round Random Forest training with hard-negative mining
- `infer.py`: sliding-window inference and timing
- `quick_eval.py`: quick AP50 self-check (not the official evaluator)
- `requirements.txt`

**Results (`results/`)**
- `preds_val.csv`, `preds_test.csv`: predictions (`image_id,category_id,x,y,width,height,score`; `image_id` is the file name without extension, `category_id` is 0)
- `preds_test_timing.json`: test-set timing (CPU, per-repeat total/mean/P95, throughput, peak memory)
- `train_log.json`: training times and hard-negative statistics

## Setup

```
pip install -r requirements.txt
```

`--root` is the dataset folder that directly contains `images/` and `labels/` (e.g. `Beach Bird Detection Dataset v1.0`).

## Pipeline (run in this order)

1. `python check_data.py --root <root>`: sanity check; only reports, never changes the official split.
2. `python build_samples.py --root <root> --workers 4`: TRAIN split only. Positives = GT boxes +10 % context; 20,000 random negatives with IoU < 0.1 to every GT box. Each window is resized to 64x64 and described by HOG (1764-d) + HSV histogram (24-d) = 1788-d. Output `samples_round1.npz`.
3. `python train_rf.py --root <root> --workers 4`: trains forest 1, mines up to 5,000 hard negatives (highest-scoring windows with IoU < 0.1) on the train images, trains forest 2 with the same parameters. Output `models/rf_final.joblib` and `models/train_log.json` (training times). `--workers` only parallelises mining across images; an interrupted run resumes from per-image cached results.
4. `python infer.py --root <root> --split test --out preds_test.csv`: sliding window (scales 1.0/0.8/0.64/0.512; windows 32x32, 48x32, 32x48, 64x64; stride 8), score >= 0.001, greedy hard NMS (IoU 0.6), max 300 boxes/image. Writes `image_id,category_id,x,y,width,height,score` (boxes in ORIGINAL 512x512 image pixels) and `preds_test_timing.json` (CPU model, per-repeat total/mean/P95 time, images/s, peak RAM). The default (`--workers 1 --repeats 3 --warmup 20`) is the timed run with the RF on 2 threads, as required by the protocol (timing on the full test set); the first repeat's predictions are saved. Use `--split val` for validation predictions, and `--workers N --repeats 1` for a fast prediction-only pass (timing is then not recorded). Run the test split only after training is finished and the configuration is frozen.
5. `python quick_eval.py --root <root> --split val --preds preds_val.csv`: quick AP50 self-check (not the official evaluator).

## Modules

`config.py` (protocol constants), `data_utils.py` (labels, 640 canvas letterbox, IoU), `features.py` (reference HOG+colour for one patch), `sliding.py` (pyramid, windows, batched feature extraction numerically equal to `features.py` (max diff ~1e-7), NMS), `detector.py` (detection, mining).

## Third-party libraries

NumPy, OpenCV (I/O, colour conversion, resize), scikit-image (`skimage.feature.hog`, reference implementation), scikit-learn (Random Forest), joblib, psutil (optional, peak RAM).

## Implementation notes

- Images are 512x512; letterboxing to 640 is a plain x1.25 resize (no padding).
- Negative window sizes are drawn from the inference window shapes x pyramid scales (not specified by the protocol).
- `category_id` is 0 (dataset class id) and `image_id` is the file stem.
- Warm-up is 20 images (the RF processes one image at a time, no batching).
- Hard negatives are mined on all 1,369 train images (use `--mine_images N` only for debugging).
- Sample construction, hard-negative mining and the validation-set prediction pass used 4 worker processes for speed, so those wall-clock times are not 2-thread measurements. The window batch size is 512 to limit memory; it does not change any feature or prediction.
