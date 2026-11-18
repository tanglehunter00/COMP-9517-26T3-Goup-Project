# Robustness comparison / 鲁棒性横评

Completed validation-only inference using the frozen seed-42 checkpoints. No new training, seed sweep, test-set tuning, or changes to model implementations, original images or labels. / 使用冻结的 seed 42 权重，未重新训练、增加训练种子、调测试阈值或修改原方法及数据。

Protocol / 设置：291 validation images, 1,843 boxes; input 640; deep batch 8, FP16; NMS IoU 0.6; prefilter 0.001; max detections 300; fixed score 0.25 and matching IoU 0.5. Five methods run sequentially. RF alone uses 8 image workers with forest n_jobs=1; this prediction pass is not a latency benchmark.

Corruptions / 扰动：independent additive Gaussian pixel/channel noise, sigma 5/15/30 on 0–255 values; Gaussian blur kernel 9, sigma 1.5, reflect-101 border; brightness multiplied by 0.5. All applied before resizing, clipped/rounded to uint8 where appropriate. Image-keyed random streams derive from seed 42 and reuse the same standard-normal field across noise intensities. Per-image SHA256 equality is checked for every method.

## Absolute AP50:95 (%) / 绝对检测精度

| Method | Clean | Noise 5 | Noise 15 | Noise 30 | Blur | Dark |
|---|---:|---:|---:|---:|---:|---:|
| m1 | 3.78 | 3.47 | 2.61 | 1.64 | 2.34 | 3.18 |
| m2 | 46.48 | 46.58 | 42.25 | 22.51 | 38.48 | 45.71 |
| m3 | 47.62 | 47.45 | 44.20 | 36.74 | 40.97 | 46.65 |
| p2-only | 46.18 | 46.13 | 40.63 | 20.37 | 38.82 | 45.67 |
| m4 | 46.58 | 46.35 | 42.61 | 25.54 | 38.44 | 45.73 |

## Descriptive five-condition means / 五项扰动均值

| Method | Mean AP50 (%) | Mean AP50:95 (%) | Mean AP-small (%) | Mean F1 (%) | AP95 drop (pp) | AP95 retention (%) |
|---|---:|---:|---:|---:|---:|---:|
| m1 | 9.89 | 2.65 | 0.66 | 2.54 | 1.13 | 70.01 |
| m2 | 76.25 | 39.11 | 23.80 | 75.75 | 7.38 | 84.13 |
| m3 | 82.06 | 43.20 | 24.49 | 75.58 | 4.41 | 90.73 |
| p2-only | 75.74 | 38.32 | 23.01 | 75.04 | 7.85 | 82.99 |
| m4 | 77.35 | 39.73 | 23.15 | 76.83 | 6.84 | 85.31 |

Each specified corruption has equal weight, so Gaussian noise accounts for 60% of this custom mean. It is not standard mCE or a model of real-world occurrence probabilities. Low clean AP can make relative retention misleading. / 每项等权，高斯噪声共占60%；该均值不是标准mCE，也不代表真实扰动发生概率。低基线方法的保留率不能单独用于判断实用性。

Units / 单位：raw CSV AP and F1 fields use the 0–1 scale; drop_pp fields use percentage points; retention fields use percent. Report tables express AP and F1 as percentages. / 原始CSV的AP及F1为0–1，drop_pp为百分点，retention为百分比；报告表格的AP和F1为百分数。

Limitations / 局限：one synthetic noise realization, no combination corruptions or real adverse-weather data; validation data were already used for checkpoint selection. These are diagnostic results rather than independent generalization evidence. Cross-architecture differences also include pretrained weights, optimization and augmentation choices.

## Reproduction / 复现

From the repository root / 在项目根目录运行：

```powershell
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& '.\Yolo Improeved\.venv\Scripts\python.exe' -u robustness_benchmark.py --root '.\Yolo Improeved\work\beach-bird\Beach Bird Detection Dataset v1.0'
```

Existing completed condition files are reused on rerun. Raw plan, image hashes, predictions, parity checks, execution notes, metrics and source snapshots are under `runs/5090d_batch8_seed42_20261117/robustness/`. The script copy there is for inspection; launch the repository-root entry point. / 已完成条件可复用；完整计划、图像哈希、预测、核对记录、执行说明、指标和源码快照均保存在该目录。

The first RF verification attempt used tolerances tighter than the original CSV rounding and stopped; it was corrected before any corrupted RF metrics were completed. RF exports retain original 2-decimal coordinates and 5-decimal scores. Primary clean test outputs were never replaced. / 首次RF核对因容差小于原CSV舍入精度而停止；在任何RF扰动指标完成前修正。RF仍采用原来的坐标两位、分数五位小数，主测试结果未被替换。
