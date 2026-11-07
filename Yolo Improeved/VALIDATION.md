# Implementation verification — 2026-11-08

Environment: Windows, Python 3.13.2, torch 2.12.0+cu130,
torchvision 0.27.0+cu130, Ultralytics 8.3.221, RTX 5090 D.

Passed:

- Four unittest cases: DySample rectangular forward/backward; both P2 configurations,
  output shape and compatible checkpoint transfer; COCO perfect/empty predictions,
  negative-image false positives and empty density groups; LR schedule endpoints.
- Official COCO checkpoint, pinned wheel and vendored DySample SHA256 verification.
- M4 synthetic CUDA FP16 training: 17 train images, 8 validation images, 2 epochs.
  Three microbatches per epoch exercise the final one-image accumulation group.
  Completed four optimizer updates; final LR 0.00001; checkpoint/metrics saved.
- Separate run stopped after epoch 1 and resumed from last.pt to epoch 2.
  Optimizer, scaler and scheduler progress restored; four total updates.
- Reloaded selected best checkpoint; original-coordinate CSV inference on eight
  synthetic test images, followed by standalone COCO/operating-point evaluation.
- GPU benchmark exercised 20 warm-up batches and three full-set repeats, batch 8;
  output explicitly labels local GPU as non-official hardware.

Diagnostic artifacts remain ignored under runs/smoke-m4 and runs/smoke-resume.
Recreate fixture with `python tests/make_smoke_data.py`; use a new output directory
when rerunning training to avoid overwriting an existing run. Example:

```powershell
python train.py --root work/synthetic --out runs/check-resume --smoke --stop-after-epoch 1
python train.py --root work/synthetic --out runs/check-resume --smoke --resume runs/check-resume/last.pt
python infer.py --root work/synthetic --weights runs/check-resume/best.pt --split test --out runs/check-resume/preds.csv --allow-smoke --benchmark
python evaluate.py --root work/synthetic --split test --preds runs/check-resume/preds.csv --out runs/check-resume/metrics.json
```

Observed and handled: default AMP initial scale 65536 overflowed on the synthetic
run; initial scale 128 completed. CUDA grid_sample backward emits the expected
nondeterminism warning. Exact bitwise agreement across runs is not claimed.

Not measured: real Beach Bird accuracy, full 100-epoch execution, T4 memory or
runtime, or improvement over M2. Synthetic metrics are not project results.
