"""Build the final report only from complete, evaluated fixed-protocol runs."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results'
METHODS = {'m1':'RF', 'm2':'YOLO11s', 'm3':'Faster R-CNN V2',
           'p2-only':'YOLO11s + P2', 'm4':'YOLO11s + P2 + DySample'}


def load(path):
    return json.loads(path.read_text())


def number(value):
    return 'N/A' if value is None else f'{value:.4f}'


def main():
    logs, configs, metrics = {}, {}, {}
    for method in METHODS:
        metrics[method] = load(RESULTS/method/'test_metrics.json')
        if method != 'm1':
            logs[method] = [json.loads(line) for line in (RESULTS/method/'metrics.jsonl').read_text().splitlines()]
            if [row['epoch'] for row in logs[method]] != list(range(1,101)):
                raise ValueError(f'{method}: expected exactly 100 unique consecutive epoch records')
            configs[method] = load(RESULTS/method/'config.json')
            if configs[method]['smoke']:
                raise ValueError('Synthetic training cannot appear in the final report')
    selected = {method:load(RESULTS/method/'weights.json')['selected_epoch'] for method in logs}
    for method in logs:
        best = max(logs[method], key=lambda row:(row['AP50'],row['AP50:95'],-row['epoch']))
        if best['epoch'] != selected[method]:
            raise ValueError(f'{method}: selected checkpoint does not match the specified ranking')
    fig, axes = plt.subplots(1, 2, figsize=(12,4), constrained_layout=True)
    for method, rows in logs.items():
        for axis, metric in zip(axes, ['AP50','AP50:95']):
            axis.plot([r['epoch'] for r in rows], [r[metric] for r in rows], label=METHODS[method], linewidth=1.3)
            axis.set(xlabel='Epoch', ylabel=metric, ylim=(0,1), xlim=(1,100))
            axis.grid(alpha=.2)
    axes[1].legend(fontsize=8)
    fig.savefig(RESULTS/'validation_curves.png', dpi=180); plt.close(fig)
    lines = ['# 实验结果 / Experiment results', '',
        '本报告来自真实 Beach Bird 官方划分、seed 42 的固定配置实验。每个深度配置完成 100 轮训练；没有超参数搜索或提前停止。', '',
        'This report presents fixed-configuration experiments on the official Beach Bird dataset splits with seed 42. Each deep-learning configuration completed 100 epochs, with no hyperparameter search or early stopping.', '',
        '**硬件限制：** 本次使用 RTX 3090 24 GB，并非协议指定的 Colab T4。以下精度为本机结果，不能标成 T4 实验。', '',
        '**Hardware limitation:** These experiments used an RTX 3090 with 24 GB of GPU memory, rather than the Colab T4 specified in the protocol. The accuracy results below are local measurements and must not be presented as T4 results.', '',
        'RF 的训练和预测继承自原仓库，本次使用同一参考评估器重新计算指标，未重训 RF；原仓库未包含 RF 模型权重。', '',
        'The RF training results and predictions were inherited from the original repository. Its metrics were recomputed with the same reference evaluator; RF was not retrained, and the original repository did not include its trained model weights.', '',
        '## 测试集指标 / Test-set metrics', '',
        '采用 pycocotools 2.0.11、maxDets=[1,10,300]；AP 预过滤 0.001。Precision/Recall/F1 使用置信度 0.25 和 IoU 0.5。', '',
        'Evaluation uses pycocotools 2.0.11 with maxDets=[1,10,300] and a score prefilter of 0.001 for AP. Precision, recall and F1 use confidence 0.25 and IoU 0.5. AP, AR, precision, recall and F1 are reported as fractions from 0 to 1.', '',
        'AP、AR、Precision、Recall 和 F1 均以 0–1 比例报告。FP/negative image 表示每张负样本图像的平均误检数。', '',
        'FP/negative image is the average number of false detections per negative image.', '',
        '| 方法 / Method | 最佳轮次 / Best epoch | AP50 | AP50:95 | AP75 | AP-small | AR-small | 精确率 / Precision | 召回率 / Recall | F1 | 负样本每图误检 / FP per negative image |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    keys = ['AP50','AP50:95','AP75','AP-small','AR-small','precision','recall','F1','false_positives_per_negative_image']
    for method, name in METHODS.items():
        lines.append('| '+name+' | '+str(selected.get(method,'N/A'))+' | '+' | '.join(number(metrics[method][k]) for k in keys)+' |')
    lines += ['', '## 配对 YOLO 比较 / Matched YOLO comparisons', '']
    for metric in ['AP50','AP50:95','AP-small']:
        lines.append(f"- {metric}: M4 − M2 = {metrics['m4'][metric]-metrics['m2'][metric]:+.4f}; M4 − P2-only = {metrics['m4'][metric]-metrics['p2-only'][metric]:+.4f}.")
    lines += ['', '单个种子的差值只描述本次实验，不构成统计显著性结论。M3 仅水平翻转；跨架构结果比较完整流水线，不能单独归因于某个模块。', '',
        'These single-seed differences describe this experiment only and do not establish statistical significance. M3 uses horizontal flipping as its only augmentation. Comparisons across architectures therefore compare complete pipelines and cannot isolate the contribution of an individual module.', '',
        '## 推理计时 / Inference timing', '',
        '串行运行；图像预载入 CPU，FP16，batch 8，预热 20 批，完整测试集重复 3 次。计时覆盖预处理、传输、前向、NMS 和坐标恢复。每图时间为批次摊销值。', '',
        'Benchmarks ran sequentially with images preloaded into CPU memory, FP16, batch size 8, 20 warm-up batches and three repetitions of the complete test set. Timing includes preprocessing, transfer, the forward pass, NMS and restoration of original-image coordinates. Per-image time is amortised over a batch, rather than single-request latency.', '',
        '| 方法 / Method | 吞吐 / images/s | 摊销 / Amortised ms/image | 完整批均值 / Mean full-batch ms | 完整批 P95 / P95 full-batch ms | 峰值显存 / Peak GPU MiB |',
        '|---|---:|---:|---:|---:|---:|']
    for method in logs:
        timing = load(RESULTS/method/'timing.json')
        values = [timing[k] for k in ['images_per_second','amortised_ms_per_image','mean_full_batch_ms','p95_full_batch_ms']]
        values.append(timing['peak_allocated_bytes']/2**20)
        lines.append('| '+METHODS[method]+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', 'RF 的原 CPU 计时保存在 `Random Forest/Submit/results/preds_test_timing.json`；与 GPU 吞吐不可直接作硬件公平比较。', '',
        'The original RF CPU timing is stored in `Random Forest/Submit/results/preds_test_timing.json`. Direct comparison with GPU throughput would not be a hardware-fair comparison.', '',
        '## 训练记录与可复现性 / Training records and reproducibility', '',
        '| 方法 / Method | 纯训练累计秒数 / Training seconds | 验证累计秒数 / Validation seconds | 更新次数 / Updates | 最终学习率 / Final LR |',
        '|---|---:|---:|---:|---:|']
    for method, rows in logs.items():
        lines.append(f"| {METHODS[method]} | {sum(r['training_s'] for r in rows):.1f} | {sum(r['validation_s'] for r in rows):.1f} | {rows[-1]['updates']} | {rows[-1]['lr']:.8f} |")
    lines += ['', '早期若干轮曾并发运行，随后改为顺序续训，因此上述训练耗时包含资源争用，不是隔离性能基准。中断轮从上一个完整 epoch 重放，表格仅累计最终保留日志，未计入失败/中断的未完成轮及启动开销。', '',
        'Some early epochs ran concurrently before training resumed sequentially. The training times above therefore include resource contention and are not isolated performance benchmarks. Interrupted epochs were replayed from the last complete epoch. The table sums only the retained epoch logs and excludes incomplete failed/interrupted epochs and startup overhead.', '',
        'M2 曾发生 CUDA illegal-memory-access；保存的断点独立验证通过后续训。早期空间不足导致中断，用户清理空间后恢复。没有改变 batch、分辨率、训练轮数或归档频率。', '',
        'M2 encountered a CUDA illegal-memory-access error. Its saved checkpoint passed standalone validation before training resumed. An earlier storage shortage also interrupted execution; training resumed after the user freed space. Batch size, input resolution, epoch count and checkpoint archive frequency were unchanged.', '',
        f"M3 在第 69 轮曾因 AMP 缩放溢出中断。修复后对相同累积组降低 scale 并重算，恢复随机状态和模型缓冲区，不跳过优化器更新。第 69–100 轮共记录 {sum(r.get('amp_retries',0) for r in logs['m3'])} 次重算。强制 CUDA 溢出测试验证了 BatchNorm、RNG 和参数更新的一致性。", '',
        f"M3 stopped during epoch 69 because of AMP loss-scaling overflow. After the fix, the same accumulation group is recomputed at a lower scale, with random states and model buffers restored and no optimizer update skipped. Epochs 69–100 recorded {sum(r.get('amp_retries',0) for r in logs['m3'])} retries in total. A forced CUDA-overflow test verified consistent BatchNorm state, RNG state and parameter updates.", '',
        '跨越所有中断的完整总墙钟耗时未精确记录；上述训练与验证秒数为保留 epoch 的实测之和。', '',
        'Complete wall-clock time across all interruptions was not measured precisely. The training and validation times above are sums of measurements for the retained epochs.', '',
        '训练中的 faster-coco-eval 1.7.0 已与参考后端验证数值一致；最终指标统一由参考 pycocotools 重算。DySample grid_sample 与 Faster R-CNN roi_align 的 CUDA 反向存在已记录的非确定性，未宣称逐位复现。', '',
        'The faster-coco-eval 1.7.0 training backend was checked for numerical agreement with the reference backend. All final metrics were recomputed using reference pycocotools. CUDA backward passes for DySample grid_sample and Faster R-CNN roi_align have documented nondeterminism; bitwise reproducibility is not claimed.', '',
        '![验证曲线 / Validation curves](results/validation_curves.png)', '',
        '## 产物 / Artifacts', '',
        '- `results/dataset.json`：官方图像数、框数量和数据指纹。 / Official image counts, box counts and dataset fingerprint.',
        '- `results/<method>/`：逐轮日志、配置、完整验证/测试指标、密度组统计、压缩预测 CSV、计时和权重 SHA256。 / Per-epoch logs, configurations, full validation/test metrics, density-group statistics, compressed prediction CSVs, timing and weight SHA256 hashes.',
        '- `trained/*-best.pt`：原 FP32 模型张量的推理权重，通过 Git LFS 分发，不是续训检查点。 / Inference weights containing the original FP32 model tensors, distributed through Git LFS; these are not resumable training checkpoints.',
        '- 完整 optimizer/AMP/RNG 断点及每十轮归档保留在本机忽略的 `runs/` 目录。 / Complete optimizer/AMP/RNG checkpoints and ten-epoch archives remain in the local, Git-ignored `runs/` directories.',
        '- `results/progress/` 是曾经因空间阻塞时的历史快照；当前完成状态见 `results/status.json`。 / `results/progress/` contains historical snapshots from the storage blockage; current completion status is in `results/status.json`.', '']
    (HERE/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines), encoding='utf-8')
    (RESULTS/'status.json').write_text(json.dumps(dict(status='complete_local_hardware', epochs=100,
        official_T4_experiment=False, selected_epochs=selected, methods=list(logs),
        final_evaluator='pycocotools==2.0.11', rf_retrained=False), indent=2))
    print('Verified 100-epoch logs, checkpoint selection and final report.')


if __name__=='__main__':
    main()
