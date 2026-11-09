# M3: Faster R-CNN ResNet50-FPN V2

**Owner:** [your name] | **Group:** Group 104 | **Course:** COMP9517 2026 T3

## Summary of work completed

- **Data preparation:** Official Beach Bird Detection Dataset split (train 1,369 / val 291 / test 240 images), YOLO-format labels mapped to one foreground class (`bird` = 1, background = 0). Images are letterboxed to a 640x640 canvas (aspect-preserving, pad value 114), and Faster R-CNN's internal resize is disabled (`min_size = max_size = 640`).
- **Model:** Faster R-CNN ResNet50-FPN V2 from the official COCO weights (`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`), box predictor replaced with background + bird. `trainable_backbone_layers = 3`; backbone BatchNorm frozen (statistics and affine parameters).
- **Training:** 100 epochs, no early stopping. Batch 2, gradient accumulation 2 (effective batch 4, 343 optimiser updates per epoch), the final partial accumulation group flushed and normalised by its image count. SGD (momentum 0.9, Nesterov off, weight decay 0.0001), peak LR 0.0025 with a 1-epoch linear warm-up from 10% of peak, then cosine decay to 1% of peak, advanced per optimiser update. AMP FP16. Horizontal flip p = 0.5. Seed 42 for Python, NumPy, PyTorch, CUDA and data-loader workers.
- **Model selection:** validation on all 291 images every epoch. `best.pt` = highest val AP50, ties broken by AP50:95, then the earlier epoch. Best epoch: **7**. `last.pt` after every epoch, archive every 10 epochs.
- **Inference:** RPN NMS 0.7; RoI output score threshold 0.001, greedy hard NMS at IoU 0.6, at most 300 boxes per image, no second suppression. Boxes are written in original image pixels.
- **Predictions generated** for the validation and test sets from `best.pt`, after training finished. The test split was not used before then.

## Results

COCO-style evaluation, maxDets `[1, 10, 300]`, score prefilter 0.001.

| Split | AP50 | AP50:95 | AP75 | AP-small | AR-small | AR@300 |
|---|---|---|---|---|---|---|
| Val | 0.877 | 0.480 | 0.456 | 0.297 | 0.447 | 0.578 |
| **Test** | **0.881** | **0.488** | **0.473** | **0.335** | **0.456** | **0.583** |

Official metrics (precision, recall, F1, false positives per negative image, density groups) are computed by the shared group evaluator from the prediction files listed below.

Robustness check (not a protocol result): the same `best.pt` scored on fixed, seeded distorted copies of val and test (blur, noise, JPEG, dark, low contrast, cycled per image).

| Split | AP50 | AP50:95 | AP75 | AP-small | AR-small | AR@300 |
|---|---|---|---|---|---|---|
| Val (distorted) | 0.840 | 0.452 | 0.427 | 0.262 | 0.401 | 0.550 |
| Test (distorted) | 0.812 | 0.440 | 0.417 | 0.261 | 0.376 | 0.539 |

## Training time

Hardware: Google Colab, NVIDIA Tesla T4. Python 3.13.16, torch 2.14.1+cu130, torchvision 0.29.1+cu130.

| Stage | Time |
|---|---|
| Pure training (100 epochs) | 15,657.9 s (about 157 s per epoch) |
| Validation (100 x 291 images) | 1,781.3 s (about 18 s per epoch) |
| **Total run (training + validation)** | **17,469.6 s (4.85 h)** |
| Peak allocated GPU memory | 1,896 MB |

The protocol runtime benchmark (section 7: batch 8, 20 warm-up batches, 3 timed passes over the test set) has not been run yet.

## Files submitted

**Code**
- `train.py`: training, per-epoch validation, checkpointing, resume, final val/test export
- `model.py`: Faster R-CNN ResNet50-FPN V2 with the protocol settings and frozen backbone BatchNorm
- `dataset.py`: YOLO-label dataset, 640x640 letterbox, flip augmentation, distorted evaluation copies
- `evaluate.py`: prediction export in original-image pixels and COCO metrics (AP50, AP50:95, AP75, AP-small, AR@300)
- `requirements.txt`

**Results (`results/`)**
- `preds_val.csv`, `preds_test.csv`: predictions (`image_id,category_id,x,y,width,height,score`; `image_id` is the file name without extension, `category_id` is 0)
- `preds_val_distorted.csv`, `preds_test_distorted.csv`: the same on the distorted copies (robustness check only)
- `metrics.jsonl`: one line per epoch: losses, LR, optimiser updates, AMP loss scale, training/validation time, peak GPU memory, val AP50, AP50:95, AP75, AP-small, AR-small, AR@300
- `final_metrics.json`: best epoch, total run time and final metrics for every split
- `config.json`: full run configuration, model settings and library versions

Checkpoints (`best.pt`, `last.pt`, `epochNNN.pt`) are not included because of their size.

## Setup

```
pip install -r requirements.txt
```

`--data` is the dataset folder that directly contains `images/` and `labels/` (e.g. `Beach Bird Detection Dataset v1.0`). On Colab, `kagglehub.dataset_download('liugalhigh/beach-bird')` provides it.

## Pipeline

1. `python train.py --data <root> --out <run folder>`: trains for 100 epochs, validating every epoch, then exports `preds_{val,test}[_distorted].csv`, `final_metrics.json` and `DONE` from `best.pt`. On Colab, put `--out` on Google Drive.
2. Re-running the same command resumes from `last.pt` (model, optimiser, AMP scaler, schedule position, best score and history); a finished run (`DONE` present) is not retrained.

`--epochs` and `--limit N` (first N images per split) exist for smoke tests only. `--robust` trains a separate **non-protocol** variant with colour jitter, blur, noise and JPEG augmentation; write it to a different `--out`.

## Third-party libraries

PyTorch, torchvision (Faster R-CNN ResNet50-FPN V2 and COCO weights), NumPy, Pillow, pycocotools (COCO evaluation).

## Implementation notes

- Validation AP50 peaked at epoch 7 and no later epoch beat it. Per the protocol, the selected checkpoint is used regardless.
- AR@300 and the AP-small/AR-small values are read from COCOeval's arrays with a 300-detection cap, because `COCOeval.summarize()` hard-codes 100.
- Checkpoints are written atomically from a background thread, so Drive I/O does not stall training.
- Nondeterminism: `cudnn.benchmark = True`, and CUDA RoIAlign/scatter backward passes are nondeterministic.
