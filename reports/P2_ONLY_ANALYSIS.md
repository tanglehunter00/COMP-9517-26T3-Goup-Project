# P2 only ablation analysis

This contribution adds a CPU analysis entry point for the standard YOLO11s, P2-only and P2 + DySample configurations. It reproduces fixed-confidence recall by target size and IoU threshold and adds paired image-bootstrap intervals from saved predictions.

## Scope of the P2 control

The existing `Yolo Improeved/configs/yolo11s-p2.yaml` upsamples the neck P3 feature with nearest-neighbour interpolation, concatenates it with backbone P2 and fuses it with C3k2. Its Detect inputs are layers 25, 16, 19 and 22, giving strides 4, 8, 16 and 32. The new branch is compared with standard YOLO11s using the shared training protocol. M4 changes the added upsampling path to DySample.

The analysis reports three ordered comparisons. P2-only minus M2 describes the added branch. M4 minus P2-only describes the upsampler choice in the matched setup. M4 minus M2 describes the combined modification. Every difference is computed on the same images.

## Experiment provenance

The repository contains results from separate experiments. Select one result directory per analysis and label it explicitly.

| Record | Published RTX 3090 experiment | Newer RTX 5090 D experiment |
| --- | --- | --- |
| Prediction location | `results/<method>/preds_test.csv.gz` | `runs/5090d_batch8_seed42_20261117/<method>/preds_test.csv` |
| Summary source | `EXPERIMENT_REPORT.md` | `reports/MAIN_comparison.json` |
| P2-only AP50 | 86.59% | 87.35% |
| P2-only precision | 82.79% | 82.79% |
| P2-only recall | 83.33% | 83.99% |
| P2-only F1 | 83.06% | 83.39% |

The newer run directory and its complete predictions are local experiment artefacts. Obtain that run from the team to analyse the newer results. The public summary table alone cannot support image-level bootstrap calculations. The checked-in `results/` predictions reproduce the RTX 3090 experiment.

## Running the analysis

The script requires NumPy and Pillow. Both are included in the reporting environment described by `tools/reporting/requirements.txt`. It reads image dimensions and YOLO labels and processes CSV or gzip-compressed CSV predictions.

From the repository root in PowerShell, analyse the checked-in RTX 3090 predictions with the following command. Set the dataset path to the local directory containing `images/` and `labels/`.

```powershell
$birdDataset = 'C:/path/to/Beach Bird Detection Dataset v1.0'
python tools/reporting/analyze_p2_ablation.py --root $birdDataset --results results --label 'Published RTX 3090 seed 42' --out outputs/p2_ablation_3090
```

For the newer experiment, supply its complete saved run instead.

```powershell
python tools/reporting/analyze_p2_ablation.py --root $birdDataset --results runs/5090d_batch8_seed42_20261117 --label 'RTX 5090 D seed 42 batch 8' --out outputs/p2_ablation_5090d
```

Both commands default to the test split, confidence 0.25, 5,000 resamples and seed 20261118. `--split val` selects validation predictions. Each method directory must contain `preds_<split>.csv.gz` or `preds_<split>.csv`. Compressed predictions take precedence when both exist. A missing image entry in a valid CSV represents zero predictions. Prediction image IDs and categories are validated, and missing label files raise an error. Keep training and evaluation settings fixed when interpreting these post hoc analyses.

## Outputs and interpretation

- `analysis.md` contains report-ready recall and interval tables.
- `per_image.csv` contains TP, FP, FN, small TP and small GT for every model, image and IoU threshold.
- `summary.json` records numerical results, input SHA256 values, the label-and-image-geometry fingerprint, run label and reference checks.

At the default confidence, the script checks overall counts at IoU 0.50 against each available `<split>_metrics.json`. A mismatch stops the analysis. Without a reference file, the JSON records that the reference check is unavailable.

Matching uses descending confidence and assigns each detection to the highest-IoU unmatched annotation that passes the IoU threshold, following `Yolo Improeved/evaluate.py`. Small targets have original-image box area below 32 squared pixels. Recall here is evaluated at a fixed confidence and either IoU 0.50 or 0.75. COCO AP-small and AR-small retain their existing definitions and results.

Bootstrap resampling uses entire images, including negative images, with the same draws for every model. Reported intervals describe image-sampling uncertainty conditional on the supplied trained models. Training-seed variability requires repeated training. Samples containing no targets in a group have undefined recall and are excluded from that group's interval. The JSON records the valid replicate count.

## Verification

```powershell
python -m unittest discover -s tools/reporting/tests -p test_p2_ablation.py -v
```

Checks cover duplicate detections, negative images, stricter IoU matching, paired resampling, repeatability and undefined small-target intervals. The full RTX 3090 test run also checks the recorded TP, FP and FN of M2, P2-only and M4.

Model configuration sources and third-party acknowledgements remain documented in `Yolo Improeved/vendor/SOURCES.md` and the repository README.
