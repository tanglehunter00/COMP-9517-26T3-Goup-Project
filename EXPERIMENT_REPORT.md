# 实验结果 / Experiment results

本报告来自真实 Beach Bird 官方划分、seed 42 的固定配置实验。每个深度配置完成 100 轮训练；没有超参数搜索或提前停止。

This report presents fixed-configuration experiments on the official Beach Bird dataset splits with seed 42. Each deep-learning configuration completed 100 epochs, with no hyperparameter search or early stopping.

**硬件限制：** 本次使用 RTX 3090 24 GB，并非协议指定的 Colab T4。以下精度为本机结果，不能标成 T4 实验。

**Hardware limitation:** These experiments used an RTX 3090 with 24 GB of GPU memory, rather than the Colab T4 specified in the protocol. The accuracy results below are local measurements and must not be presented as T4 results.

RF 的训练和预测继承自原仓库，本次使用同一参考评估器重新计算指标，未重训 RF；原仓库未包含 RF 模型权重。

The RF training results and predictions were inherited from the original repository. Its metrics were recomputed with the same reference evaluator; RF was not retrained, and the original repository did not include its trained model weights.

## 测试集指标 / Test-set metrics

采用 pycocotools 2.0.11、maxDets=[1,10,300]；AP 预过滤 0.001。Precision/Recall/F1 使用置信度 0.25 和 IoU 0.5。

Evaluation uses pycocotools 2.0.11 with maxDets=[1,10,300] and a score prefilter of 0.001 for AP. Precision, recall and F1 use confidence 0.25 and IoU 0.5. AP, AR, precision, recall and F1 are reported as fractions from 0 to 1.

AP、AR、Precision、Recall 和 F1 均以 0–1 比例报告。FP/negative image 表示每张负样本图像的平均误检数。

FP/negative image is the average number of false detections per negative image.

| 方法 / Method | 最佳轮次 / Best epoch | AP50 | AP50:95 | AP75 | AP-small | AR-small | 精确率 / Precision | 召回率 / Recall | F1 | 负样本每图误检 / FP per negative image |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| RF | N/A | 0.1080 | 0.0284 | 0.0058 | 0.0121 | 0.1936 | 0.0142 | 0.6680 | 0.0277 | 296.6667 |
| YOLO11s | 93 | 0.8778 | 0.4821 | 0.4653 | 0.3134 | 0.4341 | 0.8360 | 0.8564 | 0.8461 | 0.1852 |
| Faster R-CNN V2 | 18 | 0.8714 | 0.4692 | 0.4514 | 0.3085 | 0.4154 | 0.8313 | 0.8669 | 0.8488 | 0.2222 |
| YOLO11s + P2 | 71 | 0.8659 | 0.4747 | 0.4663 | 0.3163 | 0.4558 | 0.8279 | 0.8333 | 0.8306 | 0.2593 |
| YOLO11s + P2 + DySample | 75 | 0.8752 | 0.4777 | 0.4515 | 0.3246 | 0.4508 | 0.8370 | 0.8426 | 0.8398 | 0.3333 |

## 配对 YOLO 比较 / Matched YOLO comparisons

- AP50: M4 − M2 = -0.0026; M4 − P2-only = +0.0093.
- AP50:95: M4 − M2 = -0.0044; M4 − P2-only = +0.0030.
- AP-small: M4 − M2 = +0.0112; M4 − P2-only = +0.0083.

单个种子的差值只描述本次实验，不构成统计显著性结论。M3 仅水平翻转；跨架构结果比较完整流水线，不能单独归因于某个模块。

These single-seed differences describe this experiment only and do not establish statistical significance. M3 uses horizontal flipping as its only augmentation. Comparisons across architectures therefore compare complete pipelines and cannot isolate the contribution of an individual module.

## 推理计时 / Inference timing

串行运行；图像预载入 CPU，FP16，batch 8，预热 20 批，完整测试集重复 3 次。计时覆盖预处理、传输、前向、NMS 和坐标恢复。每图时间为批次摊销值。

Benchmarks ran sequentially with images preloaded into CPU memory, FP16, batch size 8, 20 warm-up batches and three repetitions of the complete test set. Timing includes preprocessing, transfer, the forward pass, NMS and restoration of original-image coordinates. Per-image time is amortised over a batch, rather than single-request latency.

| 方法 / Method | 吞吐 / images/s | 摊销 / Amortised ms/image | 完整批均值 / Mean full-batch ms | 完整批 P95 / P95 full-batch ms | 峰值显存 / Peak GPU MiB |
|---|---:|---:|---:|---:|---:|
| YOLO11s | 218.016 | 4.587 | 36.691 | 39.597 | 296.880 |
| Faster R-CNN V2 | 60.463 | 16.539 | 132.308 | 138.448 | 1427.403 |
| YOLO11s + P2 | 193.513 | 5.168 | 41.337 | 43.672 | 537.015 |
| YOLO11s + P2 + DySample | 198.913 | 5.027 | 40.215 | 41.524 | 537.040 |

RF 的原 CPU 计时保存在 `Random Forest/Submit/results/preds_test_timing.json`；与 GPU 吞吐不可直接作硬件公平比较。

The original RF CPU timing is stored in `Random Forest/Submit/results/preds_test_timing.json`. Direct comparison with GPU throughput would not be a hardware-fair comparison.

## 训练记录与可复现性 / Training records and reproducibility

| 方法 / Method | 纯训练累计秒数 / Training seconds | 验证累计秒数 / Validation seconds | 更新次数 / Updates | 最终学习率 / Final LR |
|---|---:|---:|---:|---:|
| YOLO11s | 1979.3 | 503.2 | 8600 | 0.00001000 |
| Faster R-CNN V2 | 6591.3 | 593.1 | 34300 | 0.00002500 |
| YOLO11s + P2 | 2351.7 | 246.2 | 8600 | 0.00001000 |
| YOLO11s + P2 + DySample | 2369.2 | 258.9 | 8600 | 0.00001000 |

早期若干轮曾并发运行，随后改为顺序续训，因此上述训练耗时包含资源争用，不是隔离性能基准。中断轮从上一个完整 epoch 重放，表格仅累计最终保留日志，未计入失败/中断的未完成轮及启动开销。

Some early epochs ran concurrently before training resumed sequentially. The training times above therefore include resource contention and are not isolated performance benchmarks. Interrupted epochs were replayed from the last complete epoch. The table sums only the retained epoch logs and excludes incomplete failed/interrupted epochs and startup overhead.

M2 曾发生 CUDA illegal-memory-access；保存的断点独立验证通过后续训。早期空间不足导致中断，用户清理空间后恢复。没有改变 batch、分辨率、训练轮数或归档频率。

M2 encountered a CUDA illegal-memory-access error. Its saved checkpoint passed standalone validation before training resumed. An earlier storage shortage also interrupted execution; training resumed after the user freed space. Batch size, input resolution, epoch count and checkpoint archive frequency were unchanged.

M3 在第 69 轮曾因 AMP 缩放溢出中断。修复后对相同累积组降低 scale 并重算，恢复随机状态和模型缓冲区，不跳过优化器更新。第 69–100 轮共记录 4 次重算。强制 CUDA 溢出测试验证了 BatchNorm、RNG 和参数更新的一致性。

M3 stopped during epoch 69 because of AMP loss-scaling overflow. After the fix, the same accumulation group is recomputed at a lower scale, with random states and model buffers restored and no optimizer update skipped. Epochs 69–100 recorded 4 retries in total. A forced CUDA-overflow test verified consistent BatchNorm state, RNG state and parameter updates.

跨越所有中断的完整总墙钟耗时未精确记录；上述训练与验证秒数为保留 epoch 的实测之和。

Complete wall-clock time across all interruptions was not measured precisely. The training and validation times above are sums of measurements for the retained epochs.

训练中的 faster-coco-eval 1.7.0 已与参考后端验证数值一致；最终指标统一由参考 pycocotools 重算。DySample grid_sample 与 Faster R-CNN roi_align 的 CUDA 反向存在已记录的非确定性，未宣称逐位复现。

The faster-coco-eval 1.7.0 training backend was checked for numerical agreement with the reference backend. All final metrics were recomputed using reference pycocotools. CUDA backward passes for DySample grid_sample and Faster R-CNN roi_align have documented nondeterminism; bitwise reproducibility is not claimed.

![验证曲线 / Validation curves](results/validation_curves.png)

## 产物 / Artifacts

- `results/dataset.json`：官方图像数、框数量和数据指纹。 / Official image counts, box counts and dataset fingerprint.
- `results/<method>/`：逐轮日志、配置、完整验证/测试指标、密度组统计、压缩预测 CSV、计时和权重 SHA256。 / Per-epoch logs, configurations, full validation/test metrics, density-group statistics, compressed prediction CSVs, timing and weight SHA256 hashes.
- `trained/*-best.pt`：原 FP32 模型张量的推理权重，通过 Git LFS 分发，不是续训检查点。 / Inference weights containing the original FP32 model tensors, distributed through Git LFS; these are not resumable training checkpoints.
- 完整 optimizer/AMP/RNG 断点及每十轮归档保留在本机忽略的 `runs/` 目录。 / Complete optimizer/AMP/RNG checkpoints and ten-epoch archives remain in the local, Git-ignored `runs/` directories.
- `results/progress/` 是曾经因空间阻塞时的历史快照；当前完成状态见 `results/status.json`。 / `results/progress/` contains historical snapshots from the storage blockage; current completion status is in `results/status.json`.
