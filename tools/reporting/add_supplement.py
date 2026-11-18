from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR

HERE=Path(__file__).resolve().parent
source=(HERE/'build_report.py').read_text(encoding='utf-8')
source=source.replace("extra=json.loads((HERE/'diagnostics.json').read_text(encoding='utf-8'))", "extra=json.loads((HERE/'diagnostics.json').read_text(encoding='utf-8'))\nsupplement=json.loads((RUNS/'controlled_diagnostics/summary.json').read_text(encoding='utf-8'))\ncontrolled=supplement['conditions']")
source=source.replace('并记录 RF 重新训练及单种子限制。','并以验证集上的输入尺寸、NMS、置信度阈值和学习偏移干预检验失败解释，保留单种子限制。')
source=source.replace('更高输入分辨率或切片策略需通过新的验证实验确认。','第 VII 节进一步检验放大输入的影响；切片策略仍未执行。')
source=source.replace('该证据不足以证明特征对齐是主因，进一步检验需要可控的上采样或损失对照。','该证据不足以证明特征对齐是主因，第 VII 节以固定权重的偏移干预进一步检验其推理贡献。')
section=r'''
h('VII 验证集受控补充实验')
para('为区分输入尺度、候选抑制、工作点与学习偏移的影响，固定本轮最佳权重，仅在 291 张验证图像及 1,843 个真值框上做单因素干预。所有实验保留 seed 42、batch 8、FP16、分数预筛选 0.001 和最多 300 个检测，不训练新模型、不增加种子。设置与权重 SHA256 在结果计算前写入 plan.json，结束后重新校验四份权重未改动。')
para('原测试结果全部冻结。补充数据用于诊断，不能视为独立泛化验证，因为相同验证集已经用于选最佳轮次。输入尺寸实验固定 NMS IoU=0.6；NMS 实验固定输入 640；阈值实验复用保存的验证预测；偏移实验只修改内存中的 M4 副本。下面除特别说明外，P/R/F1、TP/FP/FN 仍在 score≥0.25、IoU≥0.5 下计算。')
h('A 输入尺寸与计算成本',2)
table('验证集输入尺寸对照  AP 为百分比',
      ['方法','尺寸','AP50','AP-small','ms/图'],
      [[names[m].replace(' YOLO11s','').replace(' P2+DySample',''),str(size),pct(controlled[f'{m}_s{size}_n60']['AP50']),pct(controlled[f'{m}_s{size}_n60']['AP-small']),f"{controlled[f'{m}_s{size}_n60']['timing']['amortised_ms_image']:.2f}"] for m in ['m2','m4'] for size in [640,960,1280]],
      [.55,.5,.75,.9,.8])
para('M4 从 640 放大到 960 后，AP-small 从 30.47% 升至 33.26%，密集组 AP50 从 85.57% 升至 90.07%；但总体 AP50 从 87.77% 降至 85.45%，FN 从 266 增至 296。基础 YOLO 的 AP-small 从 31.20% 升至 32.20%，总体 AP50 同样下降。放大可能改善部分小目标尺度匹配，却不能消除全部检测错误。密集组仅 3 张、211 个目标，此分组收益不宜推广。')
para('M4 的 960 与 1280 输入耗时分别为 640 的 1.77 和 3.20 倍；1280 的 AP50 进一步降至 79.38%，AP-small 也回落至 30.44%。原图均为 512×512，放大属于插值，不会增加新的场景细节；未按新尺度微调也可能造成分布差异。本实验否定“单纯放大即可全面改善”的假设，不能否定高分辨率训练或切片。计时在预加载验证图像上预热 20 批并重复完整推理 3 次，包含预处理与 NMS，不与测试集计时直接混比。')
h('B NMS 与检出误报的取舍',2)
para('对 M2、P2-only、M4 依次取 NMS IoU∈{0.3,0.5,0.6,0.7,0.9}。三者 AP50 都在当前网格的 0.6 最高，但 AP50:95 在 0.7 更高，说明抑制强度与评价重叠标准有关。此处只报告网格现象，不将选出的设置用于重算测试集。')
figure('nms_sweep.png','验证集 NMS 单因素对照。左为 AP50，右为统一置信度工作点的误检数；虚线为主实验 IoU=0.6。')
para('以 M4 为例，NMS 从 0.6 放宽到 0.9 时，TP 从 1,577 增至 1,622，FN 少 45 个，但 FP 从 333 增至 2,374，F1 从 84.04% 降至 55.56%。降低到 0.3 则使 FP 降至 261，同时 FN 增至 315。因而 NMS 确实影响候选保留，但放宽抑制不足以解决漏检，且代价很大。')
nms_errors=json.loads((RUNS/'controlled_diagnostics/nms_error_types.json').read_text(encoding='utf-8'))
para(f"沿用第 VI 节的几何错误分类，M4 在 NMS 0.6 与 0.9 下的重复候选分别为 {nms_errors['60']['duplicate']} 和 {nms_errors['90']['duplicate']} 个，定位相关错误为 {nms_errors['60']['localization']} 和 {nms_errors['90']['localization']} 个，背景相关错误为 {nms_errors['60']['background']} 和 {nms_errors['90']['background']} 个。分类由预测与真值重叠定义，支持候选冗余随抑制放宽增加的解释，但不等同于人工确认的语义错误原因。")
h('C 置信度工作点敏感性',2)
table('验证集阈值网格诊断  F1 为百分比',
      ['方法','F1@0.25','阈值','网格最高F1'],
      [[names[m],pct(supplement['thresholds'][m]['original']['F1']),f"{supplement['thresholds'][m]['selected']['threshold']:.2f}",pct(supplement['thresholds'][m]['selected']['F1'])] for m in ['m2','m3','p2-only','m4']],
      [1.35,.75,.5,.9])
figure('threshold_sweep.png','验证集阈值从 0.05 到 0.95，步长 0.05。圆点为各模型网格最高 F1，虚线为统一阈值 0.25。')
para('选择规则事先固定为验证 F1 最大，并列时取精确率更高、再取阈值更高者。M3 从 0.25 调到 0.80 后，F1 从 78.13% 升至 84.93%，与其他模型网格最高 F1 的差距缩至 0.56 个百分点以内。FP 从 729 降至 157，但 FN 从 194 增至 367，小目标固定点召回从 82.03% 降至 64.67%。因此 M3 的低固定点 F1 部分源于分数工作点，而非全部来自表示能力；这也不是正式的概率校准实验。')
para('最优 F1 不等于最优计数。M2 在 0.40 阈值的计数 MAE 为 1.038，高于 0.25 时的 0.918；M4 在 0.35 的 MAE 为 0.883，低于 0.25 时的 0.931。应用应先确定漏检、误报或计数的目标。该表是同一验证集上选择并报告的诊断上界，存在选择偏差，不替代统一阈值的测试比较。')
h('D 学习偏移的推理干预',2)
table('M4 验证集学习偏移干预  AP 为百分比',
      ['偏移状态','AP50','AP75','AP-small','FN'],
      [[label,pct(controlled[key]['AP50']),pct(controlled[key]['AP75']),pct(controlled[key]['AP-small']),str(controlled[key]['FN'])] for label,key in [('原学习偏移','m4_s640_n60'),('偏移置零','m4_s640_n60_zero_offset')]],
      [1.05,.65,.65,.75,.4])
para('只将 DySample 的 offset 卷积权重及偏置在内存中置零，保留初始规则采样网格和其余网络参数。在当前 lp、scale=2 设置下，采样退化为规则双线性上采样。原学习偏移相对置零时的 AP50、AP75、AP-small 分别高 0.13、0.40 和 0.45 个百分点，FN 少 9 个，表明已学偏移对当前模型的推理有小幅贡献。')
para('该干预没有重新训练，零偏移会改变下游特征分布；它既不是重新训练的双线性基线，也不能把 M4 与 P2-only 的训练差异全部归因于偏移。结合主消融中 AP75 下降的现象，只能说学习偏移在当前权重内部有作用，尚不足以证明其训练机制或跨数据泛化优势。')

'''
source=source.replace("h('VII 结论与后续工作')",section+"h('VIII 结论与后续工作')")
start=source.index("para('后续工作应围绕已观察到的失败模式设计验证实验。")
end=source.index("columns(2,WD_SECTION_START.NEW_PAGE)",start)
source=source[:start]+'''para('补充验证表明，输入尺度、候选抑制和工作点具有不同的成本与收益，未发现一个后处理改动能全面改善所有指标。下一步可在训练与验证集上检验重叠切片 [18]、新尺度微调和困难负样本挖掘，并保留未参与方案设计的独立数据评价泛化。当前不增加训练种子，所有结论保留单次训练限制。')
para('代码、冻结权重、逐轮记录、原始预测及补充实验计划共同构成复现依据。补充数据保存于同一运行目录的 controlled_diagnostics，包含 20 个推理条件、76 个阈值工作点和 6 组重复计时。本文区分观察、干预证据和未验证假设，不通过更改测试标签或后验选择工作点替换主实验成绩。')

'''+source[end:]
body,refs=source.split('columns(2,WD_SECTION_START.NEW_PAGE)',1)
body=body.replace('doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)\n','')
source=body+'columns(2,WD_SECTION_START.NEW_PAGE)'+refs
source=source.replace("para('只将 DySample 的 offset", "doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)\npara('只将 DySample 的 offset")
source=source.replace('5090D评审强化版.docx','5090D补充实验版.docx')
(HERE/'build_supplement_report.py').write_text(source,encoding='utf-8')
print('Created supplement report builder.')
