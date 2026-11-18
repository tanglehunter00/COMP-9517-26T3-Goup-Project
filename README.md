# COMP9517 Group 104: Beach Bird Detection

English source-code submission for COMP9517, 2026 T3. The project compares a
classical detector, standard YOLO11s, Faster R-CNN, a P2-only ablation, and P2 with
DySample. This guide is the canonical entry point for the integrated submission.

**Ready-to-submit source archive:** [COMP9517_Group104_Code.zip](submission/COMP9517_Group104_Code.zip).
The archive contains the English guide, runnable source, tests, configuration,
third-party notices and compact reference tables. It excludes model weights,
dataset images, result images, virtual environments and Git metadata.
See [SUBMISSION.md](SUBMISSION.md) for scope and remaining submission work.

## Methods and entry points

| Configuration | Implementation | Purpose |
| --- | --- | --- |
| M1: RF | `Random Forest/Submit/` | Sliding windows, HOG/HSV features, two-round Random Forest with hard-negative mining |
| M2: YOLO11s | `Yolo/train.py` | Standard three-scale detector using the shared YOLO training loop |
| M3: Faster R-CNN | `Resnet/pipeline.py` | ResNet50-FPN V2 two-stage detector |
| A: P2-only | `Yolo Improeved/train.py --p2-only` | Added stride-4 branch with nearest-neighbour upsampling |
| M4: P2 + DySample | `Yolo Improeved/train.py` | Same added branch with learned DySample upsampling |
| P2 analysis | `tools/reporting/analyze_p2_ablation.py` | CPU analysis of saved predictions and paired image bootstrap |
| Diagnostics | `controlled_diagnostics.py`, `robustness_benchmark.py` | Frozen-checkpoint, validation-only interventions and corruption comparisons |

The existing folder spelling `Yolo Improeved` is retained because imports and
scripts use it. Run commands below from the repository root unless stated otherwise.

## Environment and assets

The latest recorded experiment used Windows, Python 3.13.2, an RTX 5090 D,
torch 2.12.0+cu130 and torchvision 0.27.0+cu130. These are recorded versions, not a
claim that every machine supports that CUDA build. Install a compatible matched
CUDA-enabled torch/torchvision pair first. A GPU is needed for the full training
and timing workflow; the P2 analysis and archive builder run on CPU.

```powershell
python -m venv .venv
$python = (Resolve-Path ".venv/Scripts/python.exe").Path
# Install the matched CUDA torch/torchvision pair for your machine first.
& $python -m pip install -r requirements-submission.txt
& $python "Yolo Improeved/download_assets.py"
```

On Linux, use `.venv/bin/python` and equivalent shell quoting. The assets script
downloads the pinned Ultralytics wheel and official COCO YOLO11s initialization,
then checks SHA256 values. Faster R-CNN's COCO initialization downloads through
torchvision on first use. Network access is therefore required during setup.
Weights are not included in the code ZIP. See [THIRD_PARTY.md](THIRD_PARTY.md).

The historical root `requirements.txt` describes the older torch 2.6.0 experiment.
Use `requirements-submission.txt` for this integrated guide. The training modules
support the project's recorded old/new torchvision APIs; architecture and CUDA
compatibility still depend on the installed matched pair.

## Dataset and audit

Obtain the official **Beach Bird Detection Dataset v1.0** supplied for the project
(Kaggle source: `liugalhigh/beach-bird`). Set the root to the folder directly
containing the following tree:

```text
dataset/
  images/train/   images/val/   images/test/
  labels/train/   labels/val/   labels/test/
```

Labels use `0 x_center y_center width height` in normalized YOLO coordinates.
The official image counts are 1,369 / 291 / 240. Empty label files are valid;
missing label files are errors. Do not change the official splits.

```powershell
$datasetRoot = "D:/path/to/Beach Bird Detection Dataset v1.0"
$runDir = Join-Path (Get-Location) "runs/reproduction_seed42"
& $python "Yolo Improeved/data.py" --root $datasetRoot
```

The audit checks readable images, label validity, expected counts and exact
cross-split duplicates. It reports errors without rewriting the dataset.
Training records dataset fingerprints, and resume checks require matching data.

## Train the deep configurations

Use separate output folders. Each command below trains 100 epochs, seed 42.
For the latest equal-batch comparison, all deep models use actual batch 8,
accumulation 2 (effective batch 16), validation batch 8, and a 640 input canvas.

```powershell
& $python Yolo/train.py --root $datasetRoot --out "$runDir/m2"
& $python Resnet/pipeline.py train --root $datasetRoot --out "$runDir/m3" --batch 8
& $python "Yolo Improeved/train.py" --root $datasetRoot --out "$runDir/p2-only" --p2-only
& $python "Yolo Improeved/train.py" --root $datasetRoot --out "$runDir/m4"
```

M3's standalone historical default batch is 2, so `--batch 8` is explicit here.
YOLO variants share AdamW, learning-rate schedule, augmentations and best-epoch
selection; M3 uses its documented SGD preset and horizontal-flip augmentation.
Equal batches do not mean identical optimizers or augmentations across architectures.
Full settings and environment details are written to each run's `config.json`.

Add `--resume "$runDir/m4/last.pt"` to the matching M4 command to resume its own
run; use the corresponding command and path for the other configurations.
Checkpoints include optimizer, scaler, RNG, schedule and fingerprint state.
Selection uses validation AP50, then AP50:95, then the earlier epoch.
Never select checkpoints using test results.

**Recorded-hardware queue:** `run_serial_benchmark.py` trains M2, M3, P2-only,
M4 and RF sequentially, then exports/evaluates all configurations. It enforces an
RTX 5090-family check and is intentionally specific to the recorded experiment:

```powershell
& $python -u run_serial_benchmark.py --root $datasetRoot --out $runDir
```

Use the individual entry points on other CUDA hardware. Do not run the manual
training commands and queue against the same output directory concurrently.
T4 is not a course-wide hardware requirement; legacy `official_hardware`
metadata reflects an earlier internal protocol. Report the actual hardware.

## Train RF

The RF runs on CPU. Its data construction and mining use only training images.
Use absolute dataset/output paths when changing the working directory.

```powershell
$datasetRoot = (Resolve-Path $datasetRoot).Path
$rfDir = Join-Path $runDir "m1"
New-Item -ItemType Directory -Force -Path $rfDir | Out-Null
& $python "Random Forest/Submit/check_data.py" --root $datasetRoot
& $python "Random Forest/Submit/build_samples.py" --root $datasetRoot --workers 2 --out "$rfDir/samples_round1.npz"
& $python "Random Forest/Submit/train_rf.py" --root $datasetRoot --workers 2 --samples "$rfDir/samples_round1.npz" --out_dir "$rfDir/models"
```

## Inference, evaluation and timing

Finish training all configurations and freeze their choices before final test
evaluation. Example for M4 (replace the checkpoint/output folder for M2 or P2):

```powershell
& $python "Yolo Improeved/infer.py" --root $datasetRoot --weights "$runDir/m4/best.pt" --split test --out "$runDir/m4/preds_test.csv" --benchmark
& $python "Yolo Improeved/evaluate.py" --root $datasetRoot --split test --preds "$runDir/m4/preds_test.csv" --out "$runDir/m4/test_metrics.json"
& $python Resnet/pipeline.py infer --root $datasetRoot --weights "$runDir/m3/best.pt" --split test --out "$runDir/m3/preds_test.csv" --benchmark
& $python "Random Forest/Submit/infer.py" --root $datasetRoot --model "$rfDir/models/rf_final.joblib" --split test --out "$rfDir/preds_test.csv"
```

Run the same shared evaluator on the M3 and RF CSVs. Use `--split val` and
separate filenames for validation. CSV fields are
`image_id,category_id,x,y,width,height,score`, with file-stem IDs, class 0 and boxes
in original-image pixels. Deep inference uses batch 8, FP16, score prefilter
0.001, NMS IoU 0.6 and at most 300 detections. M3 uses its RoI NMS once.

The shared evaluator reports AP50, AP50:95, AP75, AP-small, AR-small and
precision/recall/F1 at confidence 0.25 and matching IoU 0.5. COCO small-object
metrics use original-image box area and the COCO area/ignore rules. Do not
interpret fixed-confidence small-object recall as AP-small.

Deep timing preloads images on CPU, warms up 20 batches and repeats the complete
set three times with CUDA synchronization. It includes preprocessing, transfer,
forward pass, NMS and coordinate restoration. RF timing is CPU-only, warms up
20 images and uses three complete repeats. Throughput comparisons must identify
hardware and measurement scope; GPU and CPU timings are not hardware-matched.

## Additional analysis

After producing the four deep checkpoint folders and RF model under `$runDir`:

```powershell
& $python controlled_diagnostics.py --root $datasetRoot --run $runDir
& $python robustness_benchmark.py --root $datasetRoot --run $runDir --rf-workers 8
& $python tools/reporting/analyze_p2_ablation.py --root $datasetRoot --results $runDir --split val --out outputs/p2_val --label "Reproduced seed-42 validation" --samples 5000 --seed 20261118
```

Diagnostics vary input size/NMS/confidence and set learned M4 offsets to zero in
memory. The offset intervention does not retrain a bilinear baseline. Robustness
uses Gaussian noise, blur and darkening on the validation set only. These
experiments explain frozen-model behavior; they are not independent test evidence
or multiple training seeds. Paired image-bootstrap intervals condition on the
trained models and do not measure training-seed variability.

## Recorded results and provenance

The compact `reports/MAIN_comparison.csv` and JSON contain the latest recorded
RTX 5090 D test comparison. Values below are percentages except throughput:

| Method | AP50 (%) | AP50:95 (%) | F1 (%) | Images/s |
| --- | ---: | ---: | ---: | ---: |
| RF (CPU) | 10.14 | 2.53 | 2.73 | 0.082 |
| YOLO11s | 87.08 | 47.40 | 83.54 | 252.28 |
| Faster R-CNN | 87.38 | 48.03 | 78.15 | 108.32 |
| P2-only | 87.35 | 48.35 | 83.39 | 246.00 |
| P2 + DySample | 87.88 | 47.90 | 83.86 | 247.21 |

These are existing team measurements, not experiments rerun during packaging.
There is one training seed per configuration. The dense test group has only two
images; small differences do not establish stable superiority.

The full repository also retains the historical RTX 3090 `results/` and
`EXPERIMENT_REPORT.md`, and older method READMEs. They describe earlier runs.
Do not combine those predictions with the newer summary table.
`reports/P2_ONLY_ANALYSIS.md` explains this distinction. The clean ZIP excludes
legacy reports, raw predictions and historical installation instructions.

## Verification and rebuilding the code archive

```powershell
& $python -m unittest discover -s tools/reporting/tests -p test_p2_ablation.py -v
& $python -m unittest discover -s "Yolo Improeved/tests" -v
& $python -m unittest discover -s Resnet -p test_pipeline.py -v
python tools/package_submission.py
```

The model tests require the installed deep dependencies and downloaded
initialization. CPU analysis tests require NumPy and Pillow. The archive builder
uses the Python standard library only. It validates source syntax, English text,
third-party DySample integrity and the file allowlist before writing the ZIP and
SHA256/manifest sidecars. It is deterministic for identical source contents.
The code ZIP is only one part of the course submission; the final report and
recorded presentation must be supplied separately.
