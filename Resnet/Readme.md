# Faster R-CNN implementations

## Matched-comparison M3

`pipeline.py` implements Faster R-CNN ResNet50-FPN V2 with a two-class
background/bird predictor, three trainable ResNet stages and frozen backbone
BatchNorm. Input uses a 640 letterbox canvas. The common entry point selects
actual train batch 8 explicitly; the standalone historical default is batch 2.

```text
python project.py train --method m3 --root DATASET --run RUN
python project.py train --method m3 --root DATASET --run RUN --resume
python project.py infer --method m3 --root DATASET --run RUN --split val
python project.py evaluate --method m3 --root DATASET --run RUN --split val
```

The matched preset uses 100 epochs, seed 42, accumulation 2, SGD peak LR 0.0025,
momentum 0.9, decay 0.0001, one-epoch warmup, cosine decay and horizontal flip.
AMP overflow handling replays the same accumulation group with restored state.
Best ranking is validation AP50, AP50:95, then the earlier epoch.
Inference uses batch 8, FP16, RPN NMS 0.7, RoI NMS 0.6 and at most 300 outputs.

## Original feature/resnet-model contribution

`Submit/` is the original team's independent implementation, merged with history.
It includes dataset/model/train/evaluate helpers, runtime benchmarking,
training-time distortion augmentation and local swirl-warp evaluation.

```text
python project.py original-resnet train --root DATASET --run RUN
python project.py original-resnet train --root DATASET --run RUN --robust
python project.py original-resnet benchmark --root DATASET --run RUN --split val
python project.py original-resnet warp --root DATASET --run RUN --split val
```

Original M3 uses actual batch 2, accumulation 2 and a different checkpoint schema.
It resumes automatically from its `last.pt` and skips a completed `DONE` run.
After its own training finishes it automatically exports clean/distorted
validation and test predictions. The robust version adds extra augmentation
and is stored separately; it is not a matched-comparison configuration.

Outputs stay under `RUN/original-resnet/` or
`RUN/original-resnet-robust/`. `--weights` selects a trusted original checkpoint
for benchmark/warp. The original `Submit/README.md` and research records describe
its T4 experiment; they must not be pooled with the matched 5090 D results.
No original GPU experiment was rerun during integration.
