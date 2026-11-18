# Shared YOLO pipeline: M2, P2-only and P2 + DySample

This directory implements the matched three YOLO configurations.
The historical folder spelling is retained so existing imports remain valid.

## Architecture

Standard YOLO11s detects at strides 8, 16 and 32. The added P2 branch upsamples
neck P3 (layer 16), concatenates backbone P2 (layer 2), then uses C3k2 fusion.
The four-scale detection inputs are layers 25, 16, 19 and 22, at strides
4, 8, 16 and 32. `configs/yolo11s-p2.yaml` uses nearest-neighbour upsampling;
`configs/yolo11s-p2-dysample.yaml` replaces that added upsampler with DySample.

Compatible COCO tensors are transferred by shape. The old detection head at
layer 23 maps to layer 26 and its P3/P4/P5 branches to new branches 1/2/3.
New/incompatible tensors retain initialization; `transfer.json` records the
actual tensors loaded/skipped. Tensor counts are not layer counts.

## Common commands from repository root

```text
python project.py assets
python project.py audit --root DATASET
python project.py train --method p2-only --root DATASET --run RUN
python project.py train --method m4 --root DATASET --run RUN
python project.py train --method m4 --root DATASET --run RUN --resume
python project.py infer --method m4 --root DATASET --run RUN --split val
python project.py evaluate --method m4 --root DATASET --run RUN --split val
```

Use a fresh output directory for a new experiment. Seed 42, 100 epochs,
640 canvas, batch 8, accumulation 2 and validation batch 8 are fixed.
AdamW uses grouped weight decay, three-epoch update-based warmup and cosine
decay. Mosaic closes for the last ten epochs. No EMA or early stopping is used.
Last/best/periodic checkpoints retain model, optimizer, scaler, RNG, schedule
and dataset fingerprint state. Best selection is validation AP50, then
AP50:95, then earlier epoch. Do not use test results for model selection.

Inference exports original-coordinate class-0 CSV. The shared evaluator uses
COCO maxDets 300 and reports AP50/AP50:95/AP75/AP-small/AR-small plus operating
metrics at confidence 0.25 and matching IoU 0.5. The full benchmark warms up
20 batches, repeats three complete passes and synchronizes CUDA.

`project.py diagnostics`, `robustness` and `p2-analysis` expose the supplemental
analysis. Frozen-offset removal is not a retrained DySample baseline.
One training seed and a two-image dense test subset limit stronger claims.

`tests/` includes architecture/transfer, gradient, metrics and backend checks.
Tests need the ML environment and downloaded initialization. `--smoke` on the
underlying train script is a synthetic two-epoch diagnostic, not project evidence.
DySample grid_sample backward can be nondeterministic on CUDA; warnings and
actual hardware are recorded. T4 is not a course-wide hardware requirement.

Upstream source pins and licenses are under `vendor/`; retain those notices.
See the root INTEGRATION.md for the original feature branches and the root
README for the current environment, result provenance and submission ZIP.
