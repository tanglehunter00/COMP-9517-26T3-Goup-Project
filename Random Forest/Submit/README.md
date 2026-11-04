# M1: multi-scale sliding window + HOG/colour + Random Forest (COMP9517 Group 104)

Implements protocol method M1. All constants live in `config.py` and follow the fixed group
protocol (seed 42, no hyper-parameter search).

## Setup
    pip install -r requirements.txt
`--root` is the dataset folder that contains `images/` and `labels/`
(e.g. `Beach Bird Detection Dataset v1.0`).

## Pipeline (run in this order)
1. `python check_data.py --root <root>`: sanity check; only reports, never changes the official split.
2. `python build_samples.py --root <root> --workers 4`: TRAIN split only. Positives = GT boxes +10 %
   context; 20,000 random negatives with IoU < 0.1 to every GT box. Each window is resized to 64x64 and
   described by HOG (1764-d) + HSV histogram (24-d) = 1788-d. Output `samples_round1.npz`.
3. `python train_rf.py --root <root> --workers 4`: trains forest 1, mines up to 5,000 hard negatives
   (highest-scoring windows with IoU < 0.1) on the train images, trains forest 2 with the same
   parameters. Output `models/rf_final.joblib` and `models/train_log.json` (training times).
   `--workers` only parallelises mining across images.
4. `python infer.py --root <root> --split val --out preds_val.csv`
   Sliding window (scales 1.0/0.8/0.64/0.512; windows 32x32, 48x32, 32x48, 64x64; stride 8),
   score >= 0.001, greedy hard NMS (IoU 0.6), max 300 boxes/image. Writes
   `image_id,category_id,x,y,width,height,score` (boxes in ORIGINAL 512x512 image pixels) and
   `preds_val_timing.json` (CPU model, per-repeat total/mean/P95 time, images/s, peak RAM).
   The default (`--workers 1 --repeats 3 --warmup 20`) is the timed run with the RF on 2 threads.
   For a fast prediction-only pass use `--workers N --repeats 1` (timing is then not recorded).
   Run the validation split only after training is finished; the test split is for the final evaluation.
5. `python quick_eval.py --root <root> --split val --preds preds_val.csv`: quick AP50 self-check
   (not the official evaluator).

## Modules
`config.py` (protocol constants), `data_utils.py` (labels, 640 canvas letterbox, IoU),
`features.py` (reference HOG+colour for one patch), `sliding.py` (pyramid, windows, batched feature
extraction numerically equal to `features.py` (max diff ~1e-7), NMS), `detector.py` (detection, mining).

## Third-party libraries
NumPy, OpenCV (I/O, colour conversion, resize), scikit-image (`skimage.feature.hog`, reference
implementation), scikit-learn (Random Forest), joblib, psutil (optional, peak RAM).

## Assumptions to confirm with the group
- Images are 512x512; letterboxing to 640 is a plain x1.25 resize (no padding).
- Negative window sizes are drawn from the inference window shapes x pyramid scales.
- `category_id` defaults to 0 (dataset class id) and `image_id` is the file stem; adapt to the evaluator.
- Warm-up is 20 images (RF runs one image at a time, no batching).
- Hard negatives are mined on all 1,369 train images (use `--mine_images N` only for debugging).
