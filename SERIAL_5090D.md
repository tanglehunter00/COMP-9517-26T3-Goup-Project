# RTX 5090 D 顺序横评

本轮使用独立目录 `runs/5090d_batch8_seed42_20261117/`。深度配置统一 100 epoch、训练 batch 8、累积 2 步、有效 batch 16、验证与推理 batch 8、输入 640、FP16、seed 42。Faster R-CNN 的训练 batch 从历史配置的 2 显式改为 8，学习率保持其 SGD 预设，不随 batch 自动缩放。

基础 YOLO、Faster R-CNN、P2-only、P2 + DySample 依次训练。之后在 CPU 上重新构建 RF 样本、执行两遍拟合和困难负样本挖掘。RF 本身无 epoch 和 batch，其训练耗时及推理指标单独记录。

每个深度模型每轮验证完整 291 张验证图像，按 AP50、AP50:95、较早 epoch 选择 best。全部训练完成后，队列依次用 best 导出验证集与 240 张测试集预测，使用同一个 pycocotools 评估器计算 mAP@0.5 等指标。测试集不参与选模。深度推理计时使用 20 批预热、全测试集 3 次重复、batch 8、CUDA 同步。RF 使用 CPU n_jobs=2，测试计时也是完整集合重复 3 次。

## 启动与恢复

在项目根目录执行下面的 PowerShell 命令。重复执行会恢复未完成阶段和训练断点，文件锁阻止两个队列实例同时工作。

```powershell
& '.\Yolo Improeved\.venv\Scripts\python.exe' -u run_serial_benchmark.py --root '.\Yolo Improeved\work\beach-bird\Beach Bird Detection Dataset v1.0'
```

后台启动记录保存在输出目录的 `queue.pid`、`queue.stdout.log` 与 `queue.stderr.log`。每次只启动并等待一个训练或推理子进程。DataLoader worker 和 RF 采样 worker 属于当前方法的数据处理流程。

## 日志

- `plan.json` 保存数据指纹、统一参数、实际环境和计划顺序。
- `status.json` 保存队列状态、当前阶段、子进程 PID 和各配置完成轮数。
- `stages.json` 保存每阶段命令、起止时间、耗时和退出码。
- `logs/<阶段>.log` 保存各训练、推理和评估进程的完整输出。
- `<方法>/metrics.jsonl` 保存每轮训练 loss 分量、学习率、更新数和验证指标。
- `<方法>/epoch_timing.jsonl` 保存每轮 `training_s`、`validation_s`、`checkpoint_s` 和 `epoch_wall_s`。总耗时覆盖该轮数据管线准备、训练、验证及保存，队列阶段日志另记录完整进程耗时。
- `<方法>/epochs.csv` 自动合并每轮 loss、loss 总和、耗时和验证指标。YOLO 记录 box/cls/DFL，Faster R-CNN 记录分类、框回归、RPN objectness 和 RPN 框回归。架构间 loss 定义不同，绝对大小不能直接比较。
- `<方法>/best.pt`、`last.pt` 和每 10 轮归档保留在本地。
- `<方法>/val_metrics.json`、`test_metrics.json`、预测 CSV 与计时 JSON 保存最终结果。
- 全部完成后自动生成 `comparison.csv` 和 `comparison.json`。

每个配置的随机流由统一种子 42 派生，逐轮采用 42 + epoch 的相同规则。优化器和数据增强保留各方法的预设。YOLO 三配置共享完全相同的训练设置，Faster R-CNN 使用其 SGD 与水平翻转设置。模型均从官方 COCO 初始化开始，RF 从新构建的训练样本拟合。

本机使用 torch 2.12.0+cu130、torchvision 0.27.0+cu130，支持 5090 D。M3 增加了对匹配的 0.27.0 版本支持，历史 0.21.0 支持保留。7 个 YOLO/评估测试、3 个 Faster R-CNN 测试及 batch 8 实际 AMP 训练检查已通过。

任何阶段失败会把队列标记为 failed 并停止，不跳过模型或自动降低 batch。修复原因后可使用同一启动命令恢复。此前 3090 和 5090 D 的旧结果保留在原目录，本轮结果需等 `status.json` 标记 complete 后作为完整横评使用。
