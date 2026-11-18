# Fixed-checkpoint validation diagnostics / 固定权重验证诊断

Completed on 18 November 2026 using the frozen RTX 5090 D seed-42 run. No additional training seeds or model fitting were performed. All GPU inference conditions ran sequentially. Existing model implementations, dataset labels, checkpoint files, and primary test results were retained.

补充实验于 2026 年 11 月 18 日完成，使用已经训练完成的 seed 42 权重。没有新增种子或重新训练，GPU 推理条件串行运行。各方法实现、数据标签、权重文件和主测试结果保持原样。

## Experiments / 实验

- Input canvas 640, 960, 1280 for M2 and M4, fixed NMS IoU 0.6. Six timing conditions each use 20 warmup batches and three complete validation passes, batch 8, FP16.
- NMS IoU 0.3, 0.5, 0.6, 0.7, 0.9 for M2, P2-only and M4 at input 640.
- Confidence thresholds 0.05–0.95 in increments of 0.05 for four deep configurations, using saved validation predictions. Select highest validation F1, then precision, then threshold.
- M4's learned DySample offsets set to zero in memory only, keeping the regular initial sampling grid and all other parameters. This is an inference intervention, not a retrained bilinear baseline.

共 20 个完整验证推理条件、76 个阈值工作点、6 组各三次的计时。评价范围仅为官方验证集的 291 张图像及 1,843 个真值框。既有测试表不被补充实验替换。

## Findings / 主要发现

- At input 960, M4 validation AP-small improves from 30.47% to 33.26%, but AP50 falls from 87.77% to 85.45% and processing cost grows 1.77 times. Input 1280 further reduces overall accuracy. The original images are 512 square; interpolation adds no new scene information.
- Raising M4 NMS IoU from 0.6 to 0.9 recovers 45 true positives but increases false positives from 333 to 2,374. Geometrically defined duplicate errors increase from 34 to 1,645.
- M3 validation F1 improves from 78.13% at confidence 0.25 to 84.93% at 0.80, with a substantial recall tradeoff. This is validation selection, not independent performance evidence or formal probability calibration.
- Removing learned offsets reduces M4 AP50/AP75/AP-small by 0.13/0.40/0.45 percentage points. This supports a small contribution inside the trained model but does not identify a training-time causal effect.

结论及局限见中文报告第 VII 节。全部实验使用已经参与选模的验证集，因此用于解释行为，不能作为独立泛化验证，也不证明统计显著性。

## Reproduction and records / 复现与记录

Run from the repository root, with no other GPU experiment active:

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& '.\Yolo Improeved\.venv\Scripts\python.exe' -u controlled_diagnostics.py --root '.\Yolo Improeved\work\beach-bird\Beach Bird Detection Dataset v1.0'
```

Results are stored under `runs/5090d_batch8_seed42_20261117/controlled_diagnostics/`:

- `plan.json`: settings recorded before supplementary outcomes and checkpoint SHA256 values.
- `summary.json`, `conditions.csv`, `thresholds.json`: full numeric results.
- Each condition directory: complete validation predictions and metrics.
- `nms_error_types.json`: post hoc geometric error counts using the report's existing IoU 0.1/0.5 diagnostic categories.
- `controlled_diagnostics.py`, `supplement_assets.py`: source snapshots. Run the repository-root entry point above; the archived copy is retained for inspection.
- `metadata_correction.json`: records clarification that checkpoint files remain unchanged even when inference parameters are changed in memory. Predictions and numerical metrics were not altered.

Existing per-condition predictions/metrics are reused when present; resolution timing passes are repeated on rerun. For a fresh independent rerun, supply a separate copy of the original run using `--run` and preserve the original diagnostic folder.

报告的最新版本为本目录的 `COMP9517_Group104_中文项目报告.docx` 和同名 PDF；带 `5090D补充实验版` 后缀的文件保存同一版本。
