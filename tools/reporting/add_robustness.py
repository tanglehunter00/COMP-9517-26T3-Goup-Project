from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR

HERE=Path(__file__).resolve().parent
source=(HERE/'build_supplement_report.py').read_text(encoding='utf-8')
source=source.replace("controlled=supplement['conditions']", "controlled=supplement['conditions']\nrobust=json.loads((RUNS/'robustness/summary.json').read_text(encoding='utf-8'))['methods']\nrobust_aggregate=json.loads((HERE/'robustness_aggregates.json').read_text(encoding='utf-8'))\nrobust_example=json.loads((HERE/'corruption_example.json').read_text(encoding='utf-8'))")
source=source.replace('和学习偏移干预检验失败解释，保留单种子限制。','、学习偏移干预和人工扰动横评检验失败解释，保留单种子限制。')
source=source.replace('本文用完整逐轮 loss、耗时、密度分组和错误案例解释差异，并以验证集上的输入尺寸、NMS、置信度阈值、学习偏移干预和人工扰动横评检验失败解释，保留单种子限制。','逐轮 loss、耗时、密度分组和案例用于解释差异，验证集受控实验及人工扰动横评进一步检验机制。强噪声下 M3 更稳定，说明干净精度与抗噪能力需分别评价。结论保留单种子和验证集诊断限制。')
source=source.replace('doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)\npara(\'只将 DySample', "para('只将 DySample")
section=r'''
h('E 人工扰动下的鲁棒性横评',2)
para('在同一完整验证集上，对全部五种方法施加五个固定扰动：加性高斯噪声 σ=5、15、30（0–255 像素尺度），9×9、σ=1.5 的高斯模糊，以及像素亮度乘 0.5。扰动在原图上、缩放前施加，不移动目标，真值框保持不变。高斯噪声对每个像素及 BGR 通道独立采样，以种子 42 和图像名称 SHA256 派生固定随机流；三个强度共享同一标准正态噪声场，裁剪后四舍五入为 uint8。')
para('扰动规则、权重与逐图像哈希均在计算结果前固定；逐方法校验收到完全相同的扰动像素。输入统一 640、NMS IoU=0.6、预筛选 0.001、最多 300 框，固定点指标用 score≥0.25。深度模型 batch 8、FP16，逐方法顺序运行。RF 单独使用 8 个 CPU 图像工作进程，每个森林 n_jobs=1，不与 GPU 方法重叠；该并行方式只用于预测，不能与主表的单进程耗时比较。')
figure('corruption_examples.jpg',f"扰动示例 {robust_example['stem']}。按排序选择首张含 5–15 个真值的验证图像，共 {robust_example['objects']} 个目标，不依据预测挑选。图像来自数据集 [1]。")
short_names={'m1':'M1 RF','m2':'M2','m3':'M3','p2-only':'A P2','m4':'M4'}
conditions=['gaussian_5','gaussian_15','gaussian_30','blur_1p5','dark_0p5']
table('验证集扰动横评  AP50:95 为百分比',
      ['方法','干净','噪声5','噪声15','噪声30','模糊','变暗'],
      [[short_names[m]]+[pct(robust[m][c]['AP50:95']) for c in ['clean']+conditions] for m in methods],
      [.8,.45,.45,.45,.45,.45,.45])
figure('noise_robustness.png','高斯噪声强度与完整验证集 AP50:95。σ=0 为干净图像；纵轴采用绝对 AP，避免低基线保留率掩盖性能差距。')
para(f"强噪声 σ=30 下，M3 的 AP50 为 {pct(robust['m3']['gaussian_30']['AP50'])}%，高于 M2 的 {pct(robust['m2']['gaussian_30']['AP50'])}% 和 M4 的 {pct(robust['m4']['gaussian_30']['AP50'])}%。对应 AP50:95 相对干净图像分别下降 {100*(robust['m3']['clean']['AP50:95']-robust['m3']['gaussian_30']['AP50:95']):.2f}、{100*(robust['m2']['clean']['AP50:95']-robust['m2']['gaussian_30']['AP50:95']):.2f} 和 {100*(robust['m4']['clean']['AP50:95']-robust['m4']['gaussian_30']['AP50:95']):.2f} 个百分点。M4 的强噪声 AP50:95 高于 M2 与 P2-only，但 AP-small 为 12.42%，仍低于 M2 的 12.92%。不能把干净测试中的优势等同于所有指标上的鲁棒性。")
para(f"在统一阈值下，σ=30 时 M4 的 FN 从 {robust['m4']['clean']['FN']} 增至 {robust['m4']['gaussian_30']['FN']}，FP 从 {robust['m4']['clean']['FP']} 变为 {robust['m4']['gaussian_30']['FP']}。M2 的 FN 增至 {robust['m2']['gaussian_30']['FN']}；M3 为 {robust['m3']['gaussian_30']['FN']}。因此本条件中的主要退化表现为大量漏检，而非仅有误检增加。M4 的 AP-small 从 {pct(robust['m4']['clean']['AP-small'])}% 降至 {pct(robust['m4']['gaussian_30']['AP-small'])}%，与小目标弱信号更易受扰动影响的解释相容，但不是对噪声内部作用机制的证明。")
table('五种扰动的等权描述性汇总  单位为百分比',
      ['方法','均AP50','均AP小','均F1','AP保留率'],
      [[short_names[m],pct(robust_aggregate[m]['AP50']),pct(robust_aggregate[m]['AP-small']),pct(robust_aggregate[m]['F1']),f"{robust_aggregate[m]['AP95_retention_percent']:.2f}"] for m in methods],
      [.8,.675,.675,.675,.675])
para('均值对上述五种扰动等权平均；三档高斯噪声合占 60% 权重，排名受扰动集合选择影响。AP 保留率为扰动均值 AP50:95 除以该方法干净 AP50:95。它是本实验的描述性汇总，不是标准 mCE，也不代表现实扰动发生概率。RF 的干净 AP 已很低，较小绝对下降或较高保留率不代表其优于深度模型。轻度扰动偶有微小提升时也保留原值，不把下降强制截为零。')
para('M3 的扰动均值 AP50:95 最高，为 43.20%；M4 的均值 F1 最高，为 76.83%，再次体现分数工作点与 AP 的区别。跨架构比较包含预训练、优化器和增强的差异，不能把抗噪能力归因于两阶段结构本身。四个深度配置在亮度减半下的退化均小于强噪声，不能据此推广到夜间成像。实验仅覆盖三类合成扰动及一个噪声实现，无组合扰动、真实恶劣天气或独立数据。验证集已经用于选模，本文不据此调参、替换测试成绩或主张统计显著性。')

'''
source=source.replace("h('VIII 结论与后续工作')",section+"h('VIII 结论与后续工作')")
source=source.replace("para('M3 的扰动均值", "doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)\npara('M3 的扰动均值")
source=source.replace('补充验证表明，输入尺度、候选抑制和工作点具有不同的成本与收益，未发现一个后处理改动能全面改善所有指标。','补充验证表明，输入尺度、候选抑制、工作点与人工扰动下的排名均存在取舍。M3 抗强噪声更稳定，M4 的强噪声 AP 高于其他 YOLO 配置，但没有单一方法在精度、效率和鲁棒性上全面占优。')
source=source.replace('包含 20 个推理条件、76 个阈值工作点和 6 组重复计时。','包含 20 个受控推理条件、76 个阈值工作点和 6 组重复计时；robustness 另保存五种方法的 25 个完整扰动验证条件、干净基线及像素哈希。')
source=source.replace('5090D补充实验版.docx','5090D鲁棒性完整版.docx')
(HERE/'build_robustness_report.py').write_text(source,encoding='utf-8')
print('Created robustness report builder.')
