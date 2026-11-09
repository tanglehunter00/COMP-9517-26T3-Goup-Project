# M3: Faster R-CNN ResNet50-FPN V2 Pipeline

This directory contains the implementation and training pipeline for **Method 3 (M3)** as defined in the COMP9517 Fixed Training and Evaluation Protocol.

## Files

| File | Purpose |
| --- | --- |
| `Submit/train.py` | Training, per-epoch validation, checkpointing, resume and final val/test export |
| `Submit/model.py` | Faster R-CNN ResNet50-FPN V2 with the protocol settings and frozen backbone BatchNorm |
| `Submit/dataset.py` | YOLO-label dataset, 640 × 640 letterbox, augmentation and fixed distorted evaluation copies |
| `Submit/evaluate.py` | Prediction export in original-image pixels and COCO metrics (AP50, AP50:95, AP75, AP-small, AR@300) |

## Configuration & Hyperparameters
- **Model:** Faster R-CNN ResNet50-FPN V2 with official COCO weights (`fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth`).
- **Classes:** 2 classes (Background = 0, Bird = 1).
- **Training Length:** 100 epochs (early stopping disabled).
- **Batch Size:** Actual batch size = 2, Gradient accumulation = 2 (Effective batch size = 4).
- **Optimizer:** SGD (Momentum = 0.9, Nesterov = off).
- **Learning Rate:** Peak LR = 0.0025, Weight Decay = 0.0001, 1 warmup epoch (linear from 10% of peak), then cosine decay to 1% of peak.
- **Precision:** AMP FP16.
- **Backbone:** `trainable_backbone_layers = 3` with frozen BatchNorm stats and affine parameters.
- **Input:** Aspect-preserving resize to fit 640 × 640, padded with RGB 114; internal rescaling disabled (`min_size = max_size = 640`).
- **Augmentation:** Horizontal flip, p = 0.5.
- **Detection head:** Score threshold 0.001, NMS IoU 0.6, up to 300 detections per image.
- **Seed:** 42 for Python, NumPy, PyTorch, CUDA and data-loader workers. `cudnn.benchmark` and CUDA RoIAlign backward are nondeterministic.

## Running

```bash
cd Resnet/Submit
python train.py --data "<dataset root>/Beach Bird Detection Dataset v1.0" --out <run folder>
```

The dataset root must contain `images/{train,val,test}` and `labels/{train,val,test}`. On Colab, point `--out` at Google Drive.

- **Resume:** re-running with the same `--out` continues from `last.pt` (model, optimiser, AMP scaler, schedule position, best score and history). A finished run (`DONE` present) is not retrained.
- **Model selection:** `best.pt` is the epoch with the highest clean val AP50 (ties broken by AP50:95, then the earlier epoch).
- **Testing:** the test split is evaluated once, after all 100 epochs, using `best.pt`.
- `--epochs` and `--limit` exist for smoke tests only.

### Outputs (in `--out`)

| File | Contents |
| --- | --- |
| `config.json`, `model.txt` | Full run configuration, library versions and model structure |
| `metrics.jsonl` | One line per epoch: training losses, learning rate, timing and val metrics |
| `last.pt`, `best.pt` | Resume checkpoint and selected checkpoint |
| `preds_{val,test}.csv` | Protocol export: `image_id, category_id, x, y, width, height, score` in original-image pixels |
| `preds_{val,test}_distorted.csv` | Same, on the distorted copies |
| `final_metrics.json`, `DONE` | Final metrics for every split |

### Robustness evaluation and `--robust` (non-protocol)

Every run also scores fixed, seeded distorted copies of val and test (blur, noise, JPEG, dark and low contrast, cycled per image). These results are reported separately as `*_distorted` and are not protocol results.

`--robust` trains a separate **non-protocol** variant that adds colour jitter, Gaussian blur, sensor noise and JPEG artefacts to the training images. Everything else is unchanged. Write it to a different `--out` folder.

## Results (M3, protocol run)

Colab NVIDIA T4, 100 epochs, 4.85 h training. Best checkpoint: **epoch 7**.

| Split | AP50 | AP50:95 | AP75 | AP-small | AR@300 |
| --- | --- | --- | --- | --- | --- |
| Val | 0.877 | 0.480 | 0.456 | 0.297 | 0.578 |
| **Test** | **0.881** | **0.488** | **0.473** | **0.335** | **0.583** |
| Val (distorted) | 0.840 | 0.452 | 0.427 | 0.262 | 0.550 |
| Test (distorted) | 0.812 | 0.440 | 0.417 | 0.261 | 0.539 |

Validation AP50 peaked at epoch 7, and no later epoch beat it.
