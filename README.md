# COMP9517 Group 104: Complete Bird Detection Project

This repository integrates the team's Random Forest, YOLO11s, Faster R-CNN,
P2-only and P2 + DySample work, including the original YOLO and Faster R-CNN
feature branches. Use `project.py` as the common command line.

[Integration and branch provenance](INTEGRATION.md) |
[Submission scope](SUBMISSION.md) |
[Third-party sources](THIRD_PARTY.md) |
[Source ZIP](submission/COMP9517_Group104_Code.zip)

## Setup

The latest matched comparison was recorded on Windows / Python 3.13.2 /
torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / RTX 5090 D.
Install a compatible matched CUDA torch/torchvision pair for your machine first;
the recorded CUDA build is not a universal installation requirement.

```powershell
python -m venv .venv
$python = (Resolve-Path ".venv/Scripts/python.exe").Path
# Install a matched CUDA-enabled torch/torchvision pair first.
& $python -m pip install -r requirements.txt
& $python project.py assets
```

On Linux use `.venv/bin/python` and equivalent shell syntax.
`requirements.txt` now selects the integrated dependency profile. The old
torch 2.6.0 profile is retained as `requirements-legacy-3090.txt`.
The asset step verifies the pinned YOLO checkpoint, Ultralytics wheel and
vendored DySample hash. Faster R-CNN downloads its official COCO initialization
through torchvision on first use. Setup therefore needs network access.
The source ZIP includes no model weights or dataset images.

## Dataset

Use the course-provided Beach Bird Detection Dataset v1.0, with the official
1,369 / 291 / 240 train/validation/test images. Set the root to the directory
directly containing `images/{train,val,test}` and `labels/{train,val,test}`.
Labels use normalized YOLO `0 xc yc w h` coordinates for the single bird class.
Empty label files are valid; missing labels are errors.

```powershell
$datasetRoot = "D:/path/to/Beach Bird Detection Dataset v1.0"
$runDir = Join-Path (Get-Location) "runs/integrated_seed42"
& $python project.py audit --root $datasetRoot
```

The audit checks readable images, label validity, official counts and exact
cross-split duplicates without changing the data. Keep the official split.
Training checkpoints record dataset fingerprints for matched-run resume.

## One entry point for all five comparison configurations

| Method ID | Model | Implementation guide |
| --- | --- | --- |
| `m1` | Sliding-window HOG/HSV + Random Forest | [RF](Random%20Forest/Submit/README.md) |
| `m2` | Standard YOLO11s | [YOLO](Yolo/Readme.md) |
| `m3` | Faster R-CNN ResNet50-FPN V2 | [Faster R-CNN](Resnet/Readme.md) |
| `p2-only` | YOLO11s + stride-4 P2 branch, nearest upsampling | [P2 and DySample](Yolo%20Improeved/Readme.md) |
| `m4` | YOLO11s + P2 + DySample | [P2 and DySample](Yolo%20Improeved/Readme.md) |

Train all five serially, then infer/evaluate/time validation and test splits:

```powershell
& $python project.py --dry-run run --root $datasetRoot --run $runDir
& $python project.py run --root $datasetRoot --run $runDir
```

The queue prevents overlapping owners, resumes deep checkpoints and skips
completed stages. It trains every configuration before final test evaluation.
It records the actual GPU name and accepts CUDA GPUs other than the recorded
RTX 5090 D. The deep workflow uses CUDA FP16; CPU-only training is not the
recorded protocol. RF runs on CPU and may take substantially longer.

For individual stages (replace the method with any ID in the table):

```powershell
& $python project.py train --method p2-only --root $datasetRoot --run $runDir
& $python project.py train --method p2-only --root $datasetRoot --run $runDir --resume
& $python project.py infer --method p2-only --root $datasetRoot --run $runDir --split val
& $python project.py evaluate --method p2-only --root $datasetRoot --run $runDir --split val
```

Choose a new run directory for new experiments. `--resume` applies to deep
checkpoints; RF's interrupted mining resumes through its existing cache.
Individual training does not run final test inference automatically.
After all training and choices are frozen, use `infer --split test --benchmark`
then `evaluate --split test` for final test results.

All matched deep configurations use 100 epochs, seed 42, input 640, actual
batch 8, accumulation 2, validation/inference batch 8. The M3 route explicitly
overrides its historical batch-2 default. YOLO variants share their AdamW and
augmentation presets; M3 retains its SGD and horizontal-flip preset.
Best selection uses validation AP50, then AP50:95, then the earlier epoch.
Per-run `config.json`, loss/timing logs and source snapshots record the details.

## Predictions and metrics

Output locations use `<run>/<method>/`: `best.pt` and `last.pt` for deep
models, `models/rf_final.joblib` for RF, and `preds_<split>.csv` /
`<split>_metrics.json` after inference/evaluation.
Native checkpoints from the original feature branches use separate folders.

The shared CSV schema is
`image_id,category_id,x,y,width,height,score`, using image filename stems,
dataset class 0 and boxes in original-image pixels.
Evaluation reports AP50, AP50:95, AP75, AP-small, AR-small and fixed-confidence
precision/recall/F1. Small-object COCO metrics use original-image area and
COCO area/ignore rules; fixed-confidence small recall is not AP-small.

Deep inference uses score prefilter 0.001, NMS IoU 0.6 and at most 300 detections.
Timing preloads images, warms up 20 batches, performs three full passes and
synchronizes CUDA. RF timing uses one image worker and CPU execution, so CPU/GPU
throughput comparisons must identify hardware. `--benchmark` is explicit;
ordinary RF prediction uses a single pass.

## Original team contributions

The complete original branches are merged with their commit histories:

- `Yolo/training/` and `Yolo/evaluation/`: original baseline, custom protocol
  trainer, native resume and evaluation scripts.
- `Resnet/Submit/`: original two-stage detector, dataset, train/evaluate helpers,
  benchmark, distortion augmentation and local swirl-warp experiments.

Portable entry points use the same dataset/run arguments:

```powershell
& $python project.py original-yolo train --root $datasetRoot --run $runDir --variant protocol
& $python project.py original-yolo resume --root $datasetRoot --run $runDir --variant protocol
& $python project.py original-yolo evaluate --root $datasetRoot --run $runDir --split val
& $python project.py original-resnet train --root $datasetRoot --run $runDir
& $python project.py original-resnet train --root $datasetRoot --run $runDir --robust
& $python project.py original-resnet benchmark --root $datasetRoot --run $runDir --split val
& $python project.py original-resnet warp --root $datasetRoot --run $runDir --split val
```

Original YOLO framework checkpoints differ from the shared comparison
checkpoints. Original Faster R-CNN uses batch 2 and its own checkpoint schema;
its training script automatically exports clean/distorted validation and test
predictions after its own training completes. The robust variant is an extra
augmentation experiment, not a matched-comparison configuration. Use separate
output folders and preserve these distinctions in the report.

Original scripts remain available for provenance; some contain their author's
historical local paths. Use the adapter/common entry point for portable paths.
Original training logic has been preserved, not newly certified as equivalent
to the matched loop. This integration does not rerun any recorded GPU experiment.

## P2, DySample and robustness analysis

```powershell
& $python project.py diagnostics --root $datasetRoot --run $runDir
& $python project.py robustness --root $datasetRoot --run $runDir
& $python project.py p2-analysis --root $datasetRoot --run $runDir --split val --out outputs/p2_val --label "Integrated seed-42 validation"
```

The first two commands use frozen checkpoints on the validation set.
They vary resolution/NMS/confidence, remove M4's learned offsets in memory and
compare noise/blur/darkening. Offset removal is an inference intervention, not a
retrained baseline. Bootstrap intervals condition on the trained models and
do not measure training-seed variability.

`reports/MAIN_comparison.csv` and JSON record the existing RTX 5090 D comparison.
Historical root `results/` are from the separate RTX 3090 experiment;
`Resnet/Submit/results*` are the original branch's T4 experiments.
Do not pool these runs or present their hardware timings as matched.
There is one training seed per configuration and only two dense test images.

## Tests and hand-in archive

```powershell
& $python -m unittest discover -s tests -v
& $python -m unittest discover -s tools/reporting/tests -p test_p2_ablation.py -v
& $python -m unittest discover -s "Yolo Improeved/tests" -v
& $python -m unittest discover -s Resnet -p test_pipeline.py -v
python tools/package_submission.py
```

The first two test groups run on CPU; the integration tests exercise real CLI
subprocesses, path isolation and archive coverage of both original branches.
Model tests need the installed ML stack and downloaded initialization.
The builder produces a deterministic source-only ZIP below 25 MB, with a
per-file SHA256 manifest and archive checksum. It includes all integrated model
source, configurations, tests, English execution guides and compact reference
records. It excludes weights, input/result images, environments, raw prediction
dumps, empty historical placeholders and document-authoring artefacts.

The video is not yet recorded and final report readiness remains to be checked.
GitHub and the source ZIP cover the code deliverable; the designated group member
must submit the required report/video/code through the course system separately.
