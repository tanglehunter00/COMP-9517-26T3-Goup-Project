# M4：YOLO11s + P2 + DySample

所有实现均在本目录，不修改基础 YOLO、RF、Faster R-CNN 或上层协议。
当前已实现并通过模拟数据诊断，尚未在 Beach Bird 上完成正式训练，没有正式实验成绩。

## 结构

保留 YOLO11s backbone 和 P3/P4/P5 neck（层 0–22）。从 neck P3（层 16，128 通道）新增 DySample ×2 路径，与 backbone P2（层 2）拼接，经 C3k2 得到 P2 特征。检测头输入 `[P2,P3,P4,P5]`，stride `[4,8,16,32]`，单类 bird。

`configs/yolo11s-p2.yaml` 提供 P2-only 消融，只把新增路径的 DySample 换为 nearest 上采样。

从官方 `yolo11s.pt` 迁移形状兼容参数。旧检测头层 23 映射至新层 26，P3/P4/P5 分支映射至新分支 1/2/3；新增及不兼容参数保留初始化。检查时成功迁移 448 个 state_dict 张量，51 个源张量未迁移；该计数包含缓冲区，并非层数。每次训练保存 `transfer.json`。

## 安装与下载

推荐正式实验使用 Colab T4。已在 Windows / Python 3.13.2 / torch 2.12.0+cu130 / torchvision 0.27.0+cu130 / RTX 5090 D 完成诊断，不能作为 T4 正式结果。

在本目录打开 PowerShell。本次已经创建 `.venv`（复用现有 torch），并下载权重和固定版本源码包。重建命令：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python download_assets.py
```

新环境先安装适配目标 CUDA 的成对 torch/torchvision。Ultralytics 固定为 8.3.221。上游来源、许可证与 SHA256 见 `vendor/SOURCES.md`、`vendor/SHA256.json`。权重、环境、wheel、解包源码、运行输出被本目录 `.gitignore` 排除。

## 数据检查

数据根目录直接包含 `images/train`、`images/val`、`images/test` 及对应的 `labels/` 目录。标签格式为 `0 xc yc w h`，归一化坐标；空文件代表负样本，缺失文件会报错。

```powershell
.\.venv\Scripts\python data.py --root "D:\path\to\dataset"
```

检查图像可读性、有限坐标、有效框、单类标签、跨 split 完全相同文件及官方数量 1369/291/240，发现问题阻止正式训练，不修改数据。完全重复检测不等于近似图像去重。训练记录数据指纹，续训必须匹配；不会往原数据目录写标签缓存。

## 正式训练

```powershell
.\.venv\Scripts\python train.py --root "D:\path\to\dataset" --out runs/m4
.\.venv\Scripts\python train.py --root "D:\path\to\dataset" --out runs/m4 --resume runs/m4/last.pt
```

消融使用独立输出目录：

```powershell
.\.venv\Scripts\python train.py --root "D:\path\to\dataset" --p2-only --out runs/p2-only
```

- 100 epochs、seed 42、640 方形画布、RGB 114 填充。训练 batch 8、累积 2、验证 batch 8、workers 2，保留尾批并按累积组实际图像数归一化。
- AdamW，betas `(0.9,0.999)`；普通权重 decay 0.0005，bias/normalization 不 decay，分组记录在配置中。不自动缩放学习率或权重衰减。
- 按优化器更新数执行 3 epoch 预热 `0.0001 → 0.001`，随后余弦衰减至 `0.00001`，各参数组学习率一致。
- FP16 autocast、GradScaler 初始 scale 128、梯度裁剪范数 10（沿用框架训练惯例）。非有限损失或梯度明确报失败，不静默跳过固定配置。
- 仅此循环实施梯度累积，不调用框架 trainer 的累积机制。
- 增强按协议设置水平翻转、平移、缩放、HSV；mosaic 0.5，第 91 epoch 起关闭。额外 Albumentations、MixUp、CutMix、Copy-Paste、旋转、垂直翻转关闭。
- 不使用 EMA、提前停止。每轮完整验证，按 AP50、再 AP50:95 保存 best；完全并列保留较早 epoch。
- 每轮保存 `last.pt`，每 10 轮归档，保存 `best.pt`。包含 FP32 模型、优化器、AMP、调度进度、随机状态、数据指纹，保留续训状态。
- `--stop-after-epoch N` 在完整 epoch 后保存并中断，原计划保持不变；从同目录 `last.pt` 续训。进程在 epoch 中途停止时，从上一个完整 epoch 重跑。
- `config.json` 保存结构、环境、实际硬件与参数；`metrics.jsonl` 记录损失、AP/recall、学习率、更新数、训练/验证时间和峰值训练显存。

DySample 的 CUDA grid_sample 反向传播可能非确定性，已启用 deterministic warn-only 并记录限制。每 epoch 重建数据管线并按 seed 42 加 epoch 派生随机流，避免续训沿用不同 mosaic 缓冲状态。

## 推理、评估与计时

全部训练完成后再测试，使用 best：

```powershell
.\.venv\Scripts\python infer.py --root "D:\path\to\dataset" --weights runs/m4/best.pt --split val --out runs/m4/preds_val.csv
.\.venv\Scripts\python evaluate.py --root "D:\path\to\dataset" --split val --preds runs/m4/preds_val.csv --out runs/m4/val_metrics.json
.\.venv\Scripts\python infer.py --root "D:\path\to\dataset" --weights runs/m4/best.pt --split test --out runs/m4/preds_test.csv --benchmark
.\.venv\Scripts\python evaluate.py --root "D:\path\to\dataset" --split test --preds runs/m4/preds_test.csv --out runs/m4/test_metrics.json
```

- 统一 square-640 letterbox、FP16、batch 8。CSV 为原图坐标，字段 `image_id,category_id,x,y,width,height,score`；image_id 是文件名主干，category_id=0。
- score 预过滤 0.001，只执行一次 greedy hard NMS（IoU 0.6），最多 300 框/图。
- 使用 pycocotools，maxDets `[1,10,300]`，显式读取 300 cap 数组，避免默认 summarize 中 100 cap 混用。输出 AP50、AP50:95、AP75、AP-small、AR-small。
- score 0.25、IoU 0.5 一对一匹配，计算 Precision/Recall/F1 和负样本每图误检。按原图面积使用 COCO ignore 规则。报告负样本/稀疏/中等/密集组样本量与指标，空组或无 GT 的 AP 为 null（N/A）。
- 计时预读整个测试集到 CPU，20 批预热、完整集合 3 次重复、CUDA 同步；覆盖预处理、传输、前向、NMS、坐标恢复，报告吞吐、摊销 ms/image、完整批均值/P95、峰值 GPU 内存。
- 非 T4 本地计时记录 `official_hardware=false`。其他方法需使用相同评估器或核验映射和阈值后再比较；本实现未修改其他方法。

## 验证和限制

```powershell
.\.venv\Scripts\python -m unittest discover -s tests -v
```

覆盖 DySample 非方形反向梯度、两个配置的四尺度输出与权重迁移、完美/空检测 COCO 指标、负样本误检和学习率端点。

`--smoke` 使用 2 epoch、允许非官方数量，checkpoint 记录 `smoke=true`；推理默认拒绝，需显式 `--allow-smoke` 才能用于诊断。模拟数据和输出在忽略目录 `work/`、`runs/`，不能作为项目成绩。

尚未验证真实 Beach Bird 全量训练、T4 固定 batch 显存可行性、正式精度/耗时、与基础 YOLO 的配对实验。固定配置若失败，应记录失败，不改 batch/分辨率/epochs 来冒充协议结果。
