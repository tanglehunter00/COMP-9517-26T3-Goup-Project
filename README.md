# COMP-9517-26T3-Goup-Project

## Reproducing the completed pipelines

M1 code and previous training/predictions are in `Random Forest/Submit`. M2 now
shares the matched YOLO training loop with M4. M3 is implemented in
`Resnet/pipeline.py`. The original fixed protocol below remains authoritative.

```powershell
py -3.13 -m venv --system-site-packages .venv
# First install matched torch 2.6.0 / torchvision 0.21.0 CUDA wheels.
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python "Yolo Improeved/download_assets.py"
.venv/Scripts/python run_experiments.py --root <official-dataset-root>
```

Download the public dataset with `kagglehub.dataset_download('liugalhigh/beach-bird')`.
Pass its `Beach Bird Detection Dataset v1.0` subdirectory, containing `images/`
and `labels/`. Never alter the official split. The runner resumes matching
`last.pt` checkpoints, runs 100 epochs per deep configuration, and defers all test
evaluation until M2, M3, M4 and P2-only training has completed.

Training can use `COCO_BACKEND=faster` (faster-coco-eval 1.7.0); final published
metrics use reference pycocotools 2.0.11. Backend parity is tested. Loading
torchvision explicitly enables compiled greedy hard NMS without changing its
IoU threshold. Set the environment variable only when running individual training
commands; `run_experiments.py` sets the backend for each subprocess.

Full resumable checkpoints remain in ignored `runs/` folders. `trained/` contains
inference-only selected checkpoints with original FP32 tensors, distributed via
Git LFS (`git lfs pull` after cloning). `results/` contains data audit, per-epoch
logs, final metrics, gzip-compressed CSV predictions and hardware-labelled timing
when the runner completes. Inference-only weights cannot resume training.

Local experiments on RTX 3090 are explicitly non-T4 results. Concurrent local
training jobs share resources, so their training wall times are not isolated
benchmarks. Final inference benchmarks must run serially with other training jobs
stopped. See `EXECUTION.md` for measured execution status and limitations.

Tests:

```powershell
.venv/Scripts/python -m unittest discover -s "Yolo Improeved/tests" -v
.venv/Scripts/python -m unittest discover -s Resnet -p test_pipeline.py -v
```

Yolo repo:
https://github.com/ultralytics/ultralytics

Yolo Pretrained checkpoint:
https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt

Resnet repo:
https://github.com/pytorch/vision

Resnet Pretrained checkpoint:
https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth


# COMP9517 — Fixed Training and Evaluation Protocol

**Group:** Group104  
**Hardware:** Google Colab, NVIDIA T4  
**Random seed:** `42` only  
**Execution:** One training run per configuration. No hyperparameter search, additional seeds or early stopping.

## 1. Methods

| ID | Method | Initialisation |
| --- | --- | --- |
| M1 | Multi-scale sliding-window detector with HOG + colour features, Random Forest and NMS | Train RF on the project training split |
| M2 | Standard YOLO11s | COCO-pretrained `yolo11s.pt` |
| M3 | Faster R-CNN ResNet50-FPN V2 | Official COCO weights; replace predictor with background + bird |
| M4 | YOLO11s with P2 detection head and DySample on the added P3-to-P2 upsampling path | Transfer compatible M2 COCO layers; initialise new layers using their module initialisers |

Checkpoints:
- [YOLO11s COCO weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
- [Faster R-CNN V2 COCO weights](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth)

## 2. Dataset

Use the official [Beach Bird Detection Dataset](https://www.kaggle.com/datasets/liugalhigh/beach-bird): **1,369 training, 291 validation and 240 test images**. Preserve the official split. Use one foreground class, `bird`, after checking the source labels.

- Train only on the training split. All negative mining and sampling use this split.
- Validate only on the validation split. Test only after all training is complete.
- Do not move crops, augmentations or duplicate images between splits.
- Audit invalid boxes, unreadable images and cross-split duplicates; record problems rather than silently changing the official split.
- Seed Python, NumPy, PyTorch, CUDA, data-loader workers and RF with 42. Record nondeterministic operations.

## 3. Fixed deep-learning training settings

| Setting | M2, M4 and P2-only ablation | M3 |
| --- | --- | --- |
| Training length | **100 epochs** | **100 epochs** |
| Validation | **Every epoch, all 291 images** | **Every epoch, all 291 images** |
| Early stopping | Disabled | Disabled |
| Input | Aspect-preserving resize to fit 640 × 640; pad with RGB 114 | Same canvas; prevent internal resizing to a different scale |
| Actual training batch | **8** | **2** |
| Accumulation steps | **2** | **2** |
| Effective batch | **16** | **4** |
| Optimiser | AdamW | SGD |
| Peak learning rate | 0.001 | 0.0025 |
| Betas / momentum | Betas (0.9, 0.999) | Momentum 0.9, Nesterov off |
| Weight decay | 0.0005 | 0.0001 |
| Warm-up | 3 epochs | 1 epoch |
| Final learning rate | 0.00001 | 0.000025 |
| Precision | AMP FP16 | AMP FP16 |
| Validation batch | **8** | **8** |
| Training data order | Shuffled, seed 42 | Shuffled, seed 42 |
| Data-loader workers | 2 | 2 |
| Drop final training batch | No | No |

Use linear warm-up from 10% of the peak learning rate to the peak, followed by cosine decay to 1% of the peak. Advance the schedule by optimiser updates, covering exactly the scheduled epochs. Apply the same learning-rate multiplier to all parameter groups. Set warm-up bias LR consistently rather than retaining a conflicting framework default. Flush the final partial accumulation group each epoch and normalise its loss by the number of images actually accumulated.

Use explicit optimiser settings, not `optimizer=auto`. Configure Ultralytics accumulation to match this table; do not combine its built-in accumulation with a second manual implementation. Record actual LR, accumulation and weight decay to verify framework scaling. Do not change LR automatically with batch size.

For M2/M4, train all layers and evaluate non-EMA model weights; disable EMA-based checkpoint selection for both. For M3, set `trainable_backbone_layers=3` and freeze backbone BatchNorm running statistics and affine parameters. Use RPN pre/post-NMS limits of 2000/1000 during training and 1000/1000 during evaluation, RPN NMS IoU 0.7, RoI sampling 512 proposals per image with positive fraction 0.25. Keep other architecture internals at the pinned library defaults and save the resolved model configuration.

### Augmentation

M2, M4 and the P2-only ablation use identical settings:
- Horizontal flip 0.5; vertical flip 0.
- Translation 0.1; scale 0.3; rotation, shear and perspective 0.
- HSV h=0.015, s=0.4, v=0.3.
- Mosaic probability 0.5 for epochs 1–90; off for 91–100.
- MixUp, Copy-Paste, CutMix and multi-scale training off.

M3 uses horizontal flip 0.5 only. This difference must appear in the paper: cross-architecture results compare complete pipelines; module claims use matched YOLO runs.

## 4. Fixed RF configuration

- Resize/pad images to the same 640 canvas before constructing a pyramid with factors `[1.0, 0.8, 0.64, 0.512]`.
- Scan windows `(width, height) = (32,32), (48,32), (32,48), (64,64)` at stride 8; resize each window to 64 × 64 for features.
- HOG: 9 orientations, 8 × 8 pixels per cell, 2 × 2 cells per block, L2-Hys normalisation, grayscale input.
- Colour: HSV histogram, 8 bins per channel, each channel histogram L1-normalised; concatenate with HOG.
- Extract positives from every training ground-truth box with 10% context on each side, clipped to image bounds.
- Randomly sample 20,000 training background windows with maximum ground-truth IoU < 0.1, seed 42.
- RF: `n_estimators=300`, `max_depth=20`, `min_samples_leaf=2`, `max_features="sqrt"`, `class_weight="balanced"`, `bootstrap=True`, `random_state=42`, `n_jobs=2`; all other settings use pinned scikit-learn defaults.
- Train once, mine up to 5,000 highest-scoring training background windows with IoU < 0.1, then refit a fresh forest using the same parameters on the combined samples. This is one prescribed two-pass pipeline, not a hyperparameter search.
- Use predicted bird probability as the detection score. RF has no epochs. Record feature extraction, mining and both fitting times.

## 5. Checkpoints and validation

Save `last` after every epoch, archive every 10 epochs, and save `best` using validation AP50. Break ties by AP50:95, then earlier epoch. Verify this rule explicitly rather than relying on framework-default fitness.

Every epoch, log losses, AP50, AP50:95, AP75, recall, LR, optimiser updates, training/validation time and peak allocated GPU memory. Resume from complete optimiser, scheduler and AMP states after interruptions.

Use each configuration's selected best checkpoint for final evaluation, regardless of whether the results are strong or weak.

## 6. Common evaluation

Export original-image coordinates as `image_id, category_id, x, y, width, height, score`. Use one COCO-style evaluator for all methods, with maxDets `[1, 10, 300]`; label results as AP/AR with a 300-detection cap. Use the same cap internally for final outputs.

- AP50, AP50:95 and AP75: use score prefilter **0.001**.
- Precision, recall, F1 and false positives per negative image: use score threshold **0.25**, with one-to-one matching at IoU **0.5**.
- Apply greedy hard NMS with IoU **0.6** to final candidates for all methods. For Faster R-CNN this is the RoI output NMS; RPN NMS remains 0.7. Do not run a second suppression after outputs have already been filtered.
- Use original-image COCO area groups: small < 32² pixels, medium 32²–96², large ≥ 96². Apply correct ignore rules.
- Density groups: negative=0 birds, sparse=1–10, medium=11–50, dense=51 or more. Report image/object counts and mark empty groups N/A.

Report **AP50, AP50:95, AP75, AP-small, AR-small, precision, recall, F1 and false positives per negative image**. Report one seed-42 result per configuration, not mean ± SD. A fixed confidence threshold makes this a common operating point, not necessarily each model's best F1.

## 7. Runtime benchmark

**Validation and inference batch size is fixed at 8 for all deep models.** Use T4, FP16 autocast, evaluation mode and inference mode.

1. Preload images into CPU memory, excluding disk I/O from timing.
2. Warm up with 20 batches.
3. Measure the full test set exactly 3 times, synchronising CUDA at timing boundaries.
4. Include preprocessing, host-to-device transfer, forward pass, coordinate restoration and NMS.
5. Report aggregate throughput (total images / total elapsed time), mean and P95 batch time, amortised ms/image, and peak allocated GPU memory. Retain all images; compute batch-time statistics on full batches only.

Amortised ms/image is not single-request latency. RF runs on CPU with two threads; report its CPU model and runtime separately. Timing repeats do not require retraining. Separate pure training time, validation time and total run time.

If a run cannot execute the fixed configuration, record it as blocked/failed rather than silently changing batch, input size or epochs. Do not present an unexecuted setup as measured.

## References

- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Torchvision Faster R-CNN V2](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)
- [P2 architecture example (YOLOv8; adapt to YOLO11)](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)
- [DySample paper](https://arxiv.org/abs/2308.15085) and [author implementation](https://github.com/tiny-smart/dysample)

---

# COMP9517 — 固定训练与评测协议（中文翻译）

**小组：** Group104  
**硬件：** Google Colab，NVIDIA T4  
**随机种子：** 仅使用 `42`  
**执行方式：** 每个配置训练一次；不进行超参数搜索、多种子重复或提前停止。

## 1. 方法

| 编号 | 方法 | 初始化 |
| --- | --- | --- |
| M1 | 多尺度滑窗＋HOG与颜色特征＋Random Forest＋NMS | 在项目训练集上训练RF |
| M2 | 标准YOLO11s | COCO预训练 `yolo11s.pt` |
| M3 | Faster R-CNN ResNet50-FPN V2 | 官方COCO权重，预测头改为背景＋鸟 |
| M4 | YOLO11s＋P2检测头，在新增P3到P2上采样路径使用DySample | 迁移与M2相同来源的兼容COCO权重，新增层采用各模块初始化 |


权重地址：
- [YOLO11s COCO权重](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt)
- [Faster R-CNN V2 COCO权重](https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth)


## 2. 数据集

采用官方[Beach Bird Detection Dataset](https://www.kaggle.com/datasets/liugalhigh/beach-bird)：**训练1,369张、验证291张、测试240张**。保留官方划分；核实原始标签后使用单类bird。

- 训练、负样本采样与挖掘仅使用训练集。
- 验证仅使用验证集，全部训练完成后执行测试。
- 切块、增强或重复图像不得跨集合转移。
- 检查无效框、损坏图像及跨集合重复，记录问题，不擅自改变官方划分。
- Python、NumPy、PyTorch、CUDA、DataLoader worker及RF全部设种子42，记录无法保证确定性的操作。

## 3. 固定深度模型训练配置

| 参数 | M2、M4及仅P2消融 | M3 |
| --- | --- | --- |
| 训练轮数 | **100 epochs** | **100 epochs** |
| 验证频率 | **每epoch完整验证291张** | **每epoch完整验证291张** |
| 提前停止 | 关闭 | 关闭 |
| 输入 | 等比例缩放至640×640以内，RGB 114填充 | 相同画布，禁止内部改到其他尺度 |
| 实际训练batch | **8** | **2** |
| 梯度累积 | **2** | **2** |
| 有效batch | **16** | **4** |
| 优化器 | AdamW | SGD |
| 目标学习率 | 0.001 | 0.0025 |
| Betas／动量 | (0.9, 0.999) | 动量0.9，关闭Nesterov |
| 权重衰减 | 0.0005 | 0.0001 |
| 预热 | 3轮 | 1轮 |
| 最终学习率 | 0.00001 | 0.000025 |
| 精度 | AMP FP16 | AMP FP16 |
| 验证batch | **8** | **8** |
| 数据顺序 | 打乱，种子42 | 打乱，种子42 |
| DataLoader workers | 2 | 2 |
| 丢弃训练尾批 | 否 | 否 |

学习率从目标值的10%线性预热到目标值，再余弦下降至目标值的1%。按优化器更新次数推进，覆盖规定epoch。所有参数组使用相同学习率倍率，明确设置bias预热学习率，避免保留冲突默认值。每轮尾部不足一次累积的样本也执行更新，按实际累积图像数归一化损失。

显式指定优化器，禁用auto。配置Ultralytics累积以符合表格，不再叠加手工累积。记录实际学习率、累积和权重衰减，核对框架缩放。不得随batch自动改学习率。

M2/M4训练全部层，统一评价非EMA权重，关闭基于EMA的checkpoint选择。M3设`trainable_backbone_layers=3`，冻结骨干BatchNorm运行统计及仿射参数。RPN训练前／后NMS候选数为2000/1000，评价为1000/1000，RPN NMS IoU为0.7，RoI每图采样512、正样本比例0.25。其他结构内部参数使用固定库版本默认值，并保存解析后的完整模型配置。

### 数据增强

M2、M4与仅P2消融完全相同：
- 水平翻转0.5，垂直翻转0。
- 平移0.1，缩放0.3，旋转、剪切、透视为0。
- HSV：h=0.015、s=0.4、v=0.3。
- 第1–90轮Mosaic概率0.5，第91–100轮关闭。
- 关闭MixUp、Copy-Paste、CutMix与多尺度训练。

M3仅水平翻转0.5。报告明确该差异：跨架构比较完整流程，模块贡献以严格配对的YOLO实验为依据。

## 4. 固定RF配置

- 图像先缩放填充为相同640画布，再构建比例为`[1.0, 0.8, 0.64, 0.512]`的金字塔。
- 窗口宽高为`(32,32)、(48,32)、(32,48)、(64,64)`，步长8；每个窗口缩放为64×64提取特征。
- HOG：灰度输入、9方向、8×8像素cell、2×2 cells/block、L2-Hys归一化。
- 颜色：HSV每通道8-bin直方图，各通道L1归一化，与HOG拼接。
- 每个训练标注框四周增加10%上下文，裁到图像边界，作为正样本。
- 随机采样20,000个与所有标注框IoU均小于0.1的背景窗口，种子42。
- RF：`n_estimators=300`、`max_depth=20`、`min_samples_leaf=2`、`max_features="sqrt"`、`class_weight="balanced"`、`bootstrap=True`、`random_state=42`、`n_jobs=2`；其他参数使用固定scikit-learn版本默认值。
- 首次训练后，从训练集挖掘最多5,000个得分最高且IoU<0.1的背景窗口，合并样本后以相同参数重新训练一个森林。这是规定的两遍流程，不是调参。
- 使用鸟类预测概率作为检测分数。RF无epoch；记录特征、挖掘及两次拟合耗时。

## 5. Checkpoint与验证

每轮保存last，每10轮归档，按验证AP50最高保存best；并列时依次选AP50:95更高、epoch更早的模型。明确实现该规则，不依赖框架默认fitness。

每轮记录loss、AP50、AP50:95、AP75、Recall、学习率、更新次数、训练／验证时间及峰值已分配显存。中断续训恢复优化器、scheduler及AMP状态。

各配置使用对应best进行最终评价,成果好坏不重要

## 6. 统一评价

全部输出原图坐标`image_id, category_id, x, y, width, height, score`。使用同一COCO式评价器，maxDets为`[1,10,300]`，注明AP/AR检测上限为300，模型最终输出上限同步设300。

- AP50、AP50:95、AP75：预过滤分数固定**0.001**。
- Precision、Recall、F1和负样本每图误检：分数阈值固定**0.25**，IoU **0.5**下执行一对一匹配。
- 所有方法最终候选统一使用IoU **0.6**的贪心hard NMS。Faster R-CNN在RoI输出阶段设置此值，RPN仍为0.7；不得在已经过滤的输出后重复抑制。
- 原图框面积分组：小于32²为小，32²–96²为中，不小于96²为大，正确使用忽略规则。
- 密度：0只为负样本，1–10只稀疏，11–50只中等，51只及以上密集；报告样本量，空组标N/A。

报告**AP50、AP50:95、AP75、AP-small、AR-small、Precision、Recall、F1、负样本每图误检数**。每配置只报告种子42的单次结果，不写均值±标准差。统一置信度阈值对应共同工作点，不代表每个模型各自的最佳F1。

## 7. 运行时间评测

**所有深度模型验证和推理评测batch固定8。** T4、FP16 autocast、eval和inference模式保持一致。

1. 图像预读CPU内存，排除磁盘读取。
2. 预热20批。
3. 完整测试集计时恰好重复3次，计时边界CUDA同步。
4. 包含预处理、数据传输、前向、坐标恢复和NMS。
5. 报告总体吞吐（总图像数／总耗时）、平均与P95批次时间、摊销每图时间、峰值已分配显存。保留全部图像，批次时间分布仅统计完整批。

摊销每图时间不是单张请求延迟。RF使用CPU双线程，单独报告CPU型号与时间。三次计时不需要重新训练。分别报告纯训练、验证与总耗时。

若固定配置无法运行，记录为阻塞／失败，不擅自改batch、分辨率或epoch，不将未执行的配置写成实测结果。


## 参考

- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [Torchvision Faster R-CNN V2](https://docs.pytorch.org/vision/master/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html)
- [P2结构示例（YOLOv8，需适配YOLO11）](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8-p2.yaml)
- [DySample论文](https://arxiv.org/abs/2308.15085)及[作者实现](https://github.com/tiny-smart/dysample)
