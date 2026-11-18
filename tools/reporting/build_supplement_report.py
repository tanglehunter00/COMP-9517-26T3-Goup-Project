from pathlib import Path
from reporting_paths import ROOT, RUN_DIR, DATA_ROOT, OUTPUT_DIR
import json
from decimal import Decimal, ROUND_HALF_UP
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.enum.section import WD_SECTION_START
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT

HERE=Path(__file__).resolve().parent
RUNS=RUN_DIR
OUT=OUTPUT_DIR
OUT.mkdir(exist_ok=True)
methods=['m1','m2','m3','p2-only','m4']
names={'m1':'M1 RF','m2':'M2 YOLO11s','m3':'M3 Faster R-CNN','p2-only':'A P2-only','m4':'M4 P2+DySample'}
metrics={m:json.loads((RUNS/m/'test_metrics.json').read_text()) for m in methods}
logs={m:[json.loads(x) for x in (RUNS/m/'metrics.jsonl').read_text().splitlines()] for m in methods[1:]}
timings={m:json.loads((RUNS/m/'preds_test.timing.json').read_text()) for m in methods[1:]}
extra=json.loads((HERE/'diagnostics.json').read_text(encoding='utf-8'))
supplement=json.loads((RUNS/'controlled_diagnostics/summary.json').read_text(encoding='utf-8'))
controlled=supplement['conditions']
doc=Document()
section=doc.sections[0]
section.page_width=Inches(8.5); section.page_height=Inches(11)
section.top_margin=Inches(.75); section.bottom_margin=Inches(1)
section.left_margin=section.right_margin=Inches(.625)
section.header_distance=section.footer_distance=Inches(.35)

for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Caption']:
    s=doc.styles[name]; s.font.name='Times New Roman'; s.font.color.rgb=RGBColor(0,0,0)
    s._element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'宋体')
    s.font.size=Pt({'Normal':10,'Title':21,'Subtitle':10,'Heading 1':11,'Heading 2':10,'Caption':8.5}[name])
    s.font.italic=False
    props=s._element.find(qn('w:pPr'))
    if props is not None:
        for border in list(props.findall(qn('w:pBdr'))): props.remove(border)
    f=s.paragraph_format; f.line_spacing=Pt(12); f.space_after=Pt(4)
    if name in ['Heading 1','Heading 2']:
        s.font.bold=True; f.space_before=Pt(8); f.keep_with_next=True
    if name=='Normal': f.first_line_indent=Pt(10)
    if name=='Caption': f.first_line_indent=Pt(0)

def para(text,style=None):
    p=doc.add_paragraph(text,style)
    if style is None: p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    return p

def h(text,level=1):
    return doc.add_heading(text,level)

def columns(n,start=WD_SECTION_START.CONTINUOUS):
    sec=doc.add_section(start)
    cols=sec._sectPr.find(qn('w:cols'))
    cols.set(qn('w:num'),str(n)); cols.set(qn('w:space'),'360')
    return sec

table_number=0
def table(title,headers,rows,widths=None):
    global table_number
    table_number+=1
    p=para(f'表 {table_number}  {title}','Caption'); p.paragraph_format.keep_with_next=True
    t=doc.add_table(rows=1,cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.autofit=False
    widths=widths or [3.5/len(headers)]*len(headers)
    for col,w in zip(t.columns,widths): col.width=Inches(w)
    for c,x in zip(t.rows[0].cells,headers): c.text=x
    for row in rows:
        for c,x in zip(t.add_row().cells,row): c.text=str(x)
    repeat=OxmlElement('w:tblHeader'); t.rows[0]._tr.get_or_add_trPr().append(repeat)
    for i,row in enumerate(t.rows):
        cant=OxmlElement('w:cantSplit'); row._tr.get_or_add_trPr().append(cant)
        for j,(c,w) in enumerate(zip(row.cells,widths)):
            c.width=Inches(w); c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            props=c._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge)
                for k,v in [('val','single'),('sz','4'),('color','D9D9D9')]: e.set(qn('w:'+k),v)
                borders.append(e)
            props.append(borders)
            margin=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge); e.set(qn('w:w'),'55'); e.set(qn('w:type'),'dxa'); margin.append(e)
            props.append(margin)
            if i==0:
                shade=OxmlElement('w:shd'); shade.set(qn('w:fill'),'EEEEEE'); props.append(shade)
            for p in c.paragraphs:
                p.paragraph_format.first_line_indent=Pt(0); p.paragraph_format.line_spacing=Pt(10)
                p.paragraph_format.space_after=Pt(1)
                p.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.keep_with_next=i<len(t.rows)-1
                for r in p.runs: r.font.size=Pt(8); r.bold=i==0
    para('').paragraph_format.space_after=Pt(1)

figure_number=0
def figure(file,caption):
    global figure_number
    figure_number+=1
    p=doc.add_paragraph(); p.paragraph_format.first_line_indent=Pt(0)
    p.paragraph_format.keep_with_next=True; p.paragraph_format.space_after=Pt(2)
    p.paragraph_format.line_spacing=1.0
    p.add_run().add_picture(str(HERE/file),width=Inches(3.5))
    para(f'图 {figure_number}  {caption}','Caption')

def pct(v): return f'{v*100:.2f}'
def rounded(v): return str(Decimal(str(v)).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP))
def signed_rounded(v): return f"{Decimal(str(v)).quantize(Decimal('0.001'),rounding=ROUND_HALF_UP):+.3f}"

footer=section.footer.paragraphs[0]
footer.alignment=WD_ALIGN_PARAGRAPH.CENTER; footer.paragraph_format.first_line_indent=Pt(0)
f=OxmlElement('w:fldSimple'); f.set(qn('w:instr'),'PAGE'); footer._p.append(f)
title=para('自然环境中的鸟类检测','Title'); title.alignment=WD_ALIGN_PARAGRAPH.CENTER
sub=para('统一训练配置下的传统特征与深度模型横向评估','Subtitle'); sub.alignment=WD_ALIGN_PARAGRAPH.CENTER
authors=para('Group104    COMP9517 2026 T3\n作者及学号  ____________________________________________________','Subtitle'); authors.alignment=WD_ALIGN_PARAGRAPH.CENTER
columns(2)

p=para('摘要  本文在 Beach Bird 官方划分上比较 HOG 与颜色随机森林、YOLO11s、Faster R-CNN ResNet50-FPN V2，以及增加 P2 与 DySample 的 YOLO11s，并设置 P2-only 消融。四个深度配置在 RTX 5090 D 上各训练 100 轮，统一实际 batch 8、累积 2 步、seed 42，全程串行运行。M4 的测试 AP50 与 F1 分别为 87.88% 和 83.86%，计数 MAE 为 0.875 只/图。P2-only 取得最高 AP50:95 48.35% 和小目标 AP 32.20%，基础 YOLO 的吞吐最高，为 252.28 张/秒。DySample 相对 P2-only 提高 AP50，但小目标 AP 和高重叠定位指标下降，显示增益依赖评价目标。本文用完整逐轮 loss、耗时、密度分组和错误案例解释差异，并以验证集上的输入尺寸、NMS、置信度阈值和学习偏移干预检验失败解释，保留单种子限制。')
p.paragraph_format.first_line_indent=Pt(0)
para('关键词  鸟类检测  随机森林  YOLO11  Faster R-CNN  DySample  小目标','Caption')

h('I 引言')
para('自动鸟类检测可为种群监测和生态调查提供候选位置及数量信息。然而，自然环境中的鸟类在尺度、姿态、光照和背景上变化明显。远距离鸟体像素少，密集鸟群相互遮挡，浪花、枝干和岩石纹理还可能触发误检。因此，一个实际系统需要同时考虑检出、定位、背景误报和处理成本。')
para('本项目采用 Beach Bird Detection Dataset v1.0 [1]，将全部鸟类合并为单一前景类别。研究围绕三个问题展开。首先，手工特征与学习式检测器的差距体现在哪些错误上。其次，一阶段与两阶段检测器在精度、计数和效率上如何取舍。最后，给 YOLO11s 增加 P2 分支与动态上采样，能否改善小目标检测，并在总体指标上保留收益。')
para('为回答这些问题，我们在同一机器上从初始化重新运行全部方法，深度配置统一训练轮数、实际 batch、累积步数和数据划分。M1 在同机 CPU 上重新拟合并评估。最终结果使用相同坐标定义、检测上限和参考评估器。本文重点是已有模块的任务适配、配对实验及细粒度分析，采用的 HOG、随机森林、FPN 和 DySample 分别引用原始来源 [2]–[6]、[9]。')

h('II 文献综述')
h('A 手工特征与滑窗检测',2)
para('HOG 通过局部梯度方向直方图及块归一化描述轮廓，适合将候选窗口转换为固定维度特征 [2]。颜色直方图补充外观信息，但不保留精确空间结构。随机森林结合多棵随机化决策树，能够学习非线性分类边界 [3]。这类流程便于理解特征、采样和分类器各环节，代价是需要显式枚举位置和尺度。')
para('将窗口分类概率直接用于检测还存在两点困难。训练时的裁剪框与测试滑窗并不完全同分布，分类概率也不保证反映框的定位质量。鸟体姿态和尺度变化较大时，固定窗口形状尤其容易产生高分低重叠候选。本文保留该方法作为可解释的传统基线，并用困难负样本挖掘加强背景学习。')
h('B 一阶段与两阶段深度检测',2)
para('YOLO 将检测组织为整图上的联合预测，减少逐区域重复计算 [7]。本文使用的是 Ultralytics YOLO11s 软件实现 [8]，其架构和训练配置以固定版本代码为准，原始 YOLO 文献用于解释一阶段范式。Faster R-CNN 则通过共享卷积特征上的区域建议网络产生候选，再完成区域分类和回归 [4]。')
para('ResNet 的残差连接帮助训练深层特征提取网络 [5]。FPN 通过自顶向下路径及横向融合，将语义信息传递到不同分辨率特征图 [6]。本文采用 Torchvision 的 Faster R-CNN ResNet50-FPN V2 [11]，而非仅用 ResNet 分类器。两种检测范式的比较包含各自优化器和增强流程的影响。')
h('C 高分辨率特征与动态上采样',2)
para('较小的特征步长可以保留更多空间位置，但会增加特征图和候选数量。增加 P2 头有机会改善小鸟定位，也可能放大背景纹理响应。DySample 将上采样视为带学习偏移的采样过程，以规则网格为基础调整采样位置 [9]。这为高分辨率融合提供了灵活性，但其论文结果不能直接替代本任务验证。')
para('因此，本文设置基础 YOLO、P2-only、P2 与 DySample 三个配置，依次检查增加分辨率和改变新增上采样算子的影响。使用相同增强、训练长度和权重来源，尽量减少与模块无关的差异。最终仍通过完整测试集和固定阈值错误分析判断收益。')
h('D 鸟类监测与小目标研究的联系',2)
para('鸟类监测的评价目标取决于数据与应用。Visual WetlandBirds 关注视频中的物种识别和行为识别 [17]，提供与静态框检测不同的研究视角。本文只评价 Beach Bird 的单类框检测和衍生计数，不把物种分类正确率或行为识别结果混入当前指标。这样的任务边界使方法、数据和结论能够一一对应。')
para('小目标改进也可发生在输入层面。SAHI 通过切片推理与微调处理小目标 [18]。与之相比，新增 P2 主要增加特征图的空间采样密度，不会恢复原图缩放后已经丢失的细节。切片有机会保留局部像素，但需要额外前向及跨片合并。本项目选择 P2 路径以维持单画布流程，将切片作为后续验证方向。')
para('上述研究提示三个需要分别验证的因素。手工描述符决定候选的外观表达，学习式检测器联合优化分类与定位，多尺度融合及上采样改变细节传递。本文通过 M1 与深度模型比较表示能力，再通过 M2、P2-only 和 M4 比较多尺度路径。由这些对照得到的贡献结论只限于当前实现及数据，不能代替跨数据集泛化实验。')

h('III 方法')
h('A M1 手工特征随机森林',2)
para('图像先等比例缩放并填充至 640×640，以 1、0.8、0.64 和 0.512 构建金字塔。每层扫描 32×32、48×32、32×48 和 64×64 窗口，步长为 8，窗口再缩放到 64×64。每个窗口提取 1,764 维 HOG 与 24 维 HSV 直方图，拼接为 1,788 维特征。HOG 使用 9 个方向、8×8 像素单元、2×2 单元块及 L2-Hys 归一化。HSV 每通道 8 个 bin，分别做 L1 归一化。')
para('训练正样本由全部训练框向外扩展 10% 上下文获得，共 8,663 个。随机采样 20,000 个与所有真值框 IoU 均小于 0.1 的背景窗口。森林使用 300 棵树、最大深度 20、叶节点最少 2 个样本、平方根特征采样及平衡类别权重，随机种子为 42。首次拟合后，在全部训练图像中选取得分最高的最多 5,000 个困难负样本，再以相同参数重新拟合。')
para('推理使用鸟类预测概率作为分数，经过统一 NMS 输出检测框。由于不包含框回归器，定位受窗口尺寸与网格约束。本轮从重新生成的样本开始拟合两次森林，保存模型、训练日志及完整验证、测试预测。方法的固定采样和结构参数沿用既定配置，指标使用统一参考评估器计算。')
h('B M2 与 M3 深度检测基线',2)
para('M2 从 COCO 预训练 YOLO11s 初始化，预测单类 bird，保留 P3、P4、P5 三个检测尺度，步长为 8、16、32。全部层参与训练，验证和选模使用原始模型参数，关闭 EMA 选模。')
para('M3 从官方 COCO 权重初始化 Faster R-CNN ResNet50-FPN V2，将分类头替换为背景与鸟两类。设置 trainable_backbone_layers=3，冻结骨干 BatchNorm 的统计和仿射参数。RPN 训练前后候选上限为 2000/1000，推理为 1000/1000，RPN NMS IoU 为 0.7。RoI 每图采样 512 个区域，正样本比例为 0.25，最终 RoI NMS IoU 为 0.6。输入已统一为 640 画布，避免内部改变图像尺度。')
h('C M4 的 P2 与 DySample 路径',2)
para('保留基础 YOLO 的主干和原 P3/P4/P5 颈部。在颈部 P3 特征上增加两倍上采样路径，与骨干 P2 特征拼接，经 C3k2 融合后，形成步长为 4 的 P2 输出。检测头同时接收 P2、P3、P4 和 P5。640 输入下，P2 的空间尺寸为 160×160，高于 P3 的 80×80。')
para('新增路径使用 DySample 的 lp 形式，scale=2、groups=4、dyscope=False。对输入特征 X，1×1 卷积预测偏移，再将其乘以 0.25 并叠加规则子像素网格，经双线性采样生成高分辨率特征 [9]。原有 P5→P4 与 P4→P3 路径保持各自定义。')
para('新增层保留模块初始化，兼容层从同一 COCO YOLO11s 权重迁移。原检测头的 P3/P4/P5 分支映射至四尺度头相应分支，只复制名称映射后形状一致的张量。P2-only 配置使用完全相同的四尺度拓扑，将新增 DySample 换成 nearest 上采样。该消融隔离新增路径上的上采样算子，仍受单种子训练差异影响。')
figure('architecture.png','新增 P2 路径与检测头连接，方括号为代码层索引。M4 在新增路径使用 DySample，P2-only 使用 nearest。原 P3/P4/P5 输出直接接入四尺度头。作者依据配置文件及来源 [8]、[9] 绘制。')
para('本项目的独立实现部分包括样本与困难负样本构建、批量特征提取、模型结构拼装与兼容权重映射、统一坐标导出、逐轮计时和顺序实验调度。HOG、森林、基础检测器及 DySample 核心算子来自引用的文献与库。我们将工程适配及实验分析作为贡献，不将已有采样算子表述为新提出的算法。')

h('IV 实验设置')
h('A 数据与泄漏控制',2)
table('官方划分与标注规模',['划分','图像','标注框'],[['训练','1,369','8,663'],['验证','291','1,843'],['测试','240','1,518']],[1.2,1.0,1.3])
para('保留原始训练、验证和测试划分，负样本采样与挖掘仅用训练集。图像读取、类别、框范围及跨划分完全重复文件经过审计，数据指纹与实验配置一同记录。精确文件校验无法发现经过重编码的近重复图像，本文未宣称消除所有视觉相似性。验证集负责选模，测试集仅在配置训练完成后用于最终评估与事后分析。')
para('测试集共 545 个小目标，约占 1,518 个标注的 35.9%。按每图鸟数分为负样本 0、稀疏 1–10、中等 11–50 和密集 51 以上。测试集各组分别有 27、170、41 和 2 张图像。密集组规模很小，讨论中同时给出组样本量。')
h('B 固定训练设置',2)
table('深度配置训练参数',['设置','YOLO 三配置','Faster R-CNN'],[
['训练轮数','100','100'],['实际 batch','8','8'],['累积步数','2','2'],['有效 batch','16','16'],['优化器','AdamW','SGD'],['峰值 LR','0.001','0.0025'],['最终 LR','0.00001','0.000025'],['预热轮数','3','1'],['权重衰减','0.0005¹','0.0001'],['动量设置','β=(0.9,0.999)','0.9'],['验证 batch','8','8']],[1.1,1.2,1.2])
para('¹ YOLO 的 bias 与 normalization 参数不施加权重衰减。四个深度配置统一 seed 42、实际 batch 8、累积 2 步、有效 batch 16、FP16、640 输入和 2 个加载 worker。每轮完整验证并保留尾批，按实际累积图像数归一化损失。各配置均执行 8,600 次优化器更新。学习率按更新数线性预热，再余弦降至峰值的 1%。')
para('YOLO 三配置共享水平翻转 0.5、平移 0.1、缩放 0.3，以及 HSV 扰动 0.015/0.4/0.3。前 90 轮 Mosaic 概率为 0.5，最后 10 轮关闭。旋转、剪切、透视、垂直翻转、MixUp、Copy-Paste、CutMix 和多尺度训练关闭。M3 仅使用水平翻转 0.5。')
para('每轮保存 last，每十轮归档，以验证 AP50 最高保存 best，依次用 AP50:95 和较早轮次打破并列。100 轮固定训练且关闭提前停止。训练代码显式控制学习率与梯度累积，记录优化器、调度、AMP 和 RNG 状态以支持中断恢复。')
h('C 评估与时间测量',2)
para('全部输出恢复为原图坐标，以统一 pycocotools 2.0.11 评估。本文为单类任务，AP50 即 mAP@0.5。AP50:95 对 IoU 0.50–0.95、步长 0.05 的十个阈值求平均。maxDets 显式设为 [1,10,300]，AP/AR 均报告 300 检测上限版本 [10]、[12]。面积分组使用原图面积与 COCO 的 ignore 规则。')
para('AP 预过滤分数为 0.001，最终检测使用 IoU 0.6 的 greedy hard NMS。Precision、Recall、F1 及负样本误检采用 score≥0.25、IoU≥0.5 的按分数贪心一对一匹配。F1 为精确率与召回率的调和平均。对没有真值的组，AP 记为 N/A。固定阈值表示相同工作点，不保证各模型达到自身最高 F1。')
para('IoU 衡量预测框与真值框的交并面积比，AP 汇总不同置信度工作点的精确率与召回率。本文的 PR 图采用 COCO 在 IoU=0.5 下的 101 个召回采样点及插值精确率，AP50 为这些有效采样点精确率的均值。本文没有像素分割标签，因此不计算 Dice，也不把框 IoU 当作 AP 或分类 accuracy。所有 AP、P、R、F1 表格以百分比展示，计数和耗时保留各自单位。')
para('深度实验在 RTX 5090 D 32 GB 上执行，使用 Windows、Python 3.13.2、PyTorch 2.12.0+cu130、Torchvision 0.27.0+cu130 及 Ultralytics 8.3.221。队列以 M2、M3、P2-only、M4、M1 的顺序完成训练，随后串行进行最终验证与测试。每次只运行一项训练或推理任务，数值库线程上限统一设为 2。')
para('图像处理、随机森林与深度实现分别依赖 OpenCV、scikit-learn 和 PyTorch [13]–[15]。HOG 参考实现来自 scikit-image [16]，项目提供批量特征提取及滑窗代码。上游库、代码仓库和许可证在项目来源文件中记录。')
para('M1 在同机 AMD Ryzen 7 9800X3D CPU 上执行，森林 n_jobs=2，样本构建与困难负样本挖掘使用 2 个进程。RF 不含 epoch 与训练 batch，其完整两遍拟合流程单独记录。GPU 与 CPU 时间反映各自部署流程，不能将速度差全部归因于算法。深度推理预读 CPU 内存，batch 8、FP16，预热 20 批后对全部 240 张测试图像重复 3 次，CUDA 同步并包含预处理、传输、前向、NMS 和坐标恢复。')

h('V 实验结果')
h('A 整体检测性能',2)
table('测试集 AP 指标  单位为百分比',['方法','AP50','AP50:95','AP75'],[[names[m],pct(metrics[m]['AP50']),pct(metrics[m]['AP50:95']),pct(metrics[m]['AP75'])] for m in methods],[1.45,.65,.75,.65])
para('M4 取得最高测试 AP50 87.88%，较基础 YOLO 的 87.08% 提高 0.80 个百分点。P2-only 的 AP50:95 为 48.35%、AP75 为 47.45%，均为本轮最高。M3 的 AP50 为 87.38%，与其他深度配置接近。RF 的 AP50 为 10.14%，AP75 仅 0.26%，固定窗口流程仍存在明显定位与候选排序困难。')
table('固定工作点和小目标指标  百分比',['方法','P','R','F1','AP小','AR小'],[[names[m],pct(metrics[m]['precision']),pct(metrics[m]['recall']),pct(metrics[m]['F1']),pct(metrics[m]['AP-small']),pct(metrics[m]['AR-small'])] for m in methods],[1.1,.48,.48,.48,.48,.48])
para('M4 的 F1 最高，为 83.86%，基础 YOLO 为 83.54%。M3 召回率达到 89.20%，但精确率只有 69.54%，使 F1 降至 78.15%。最高小目标 AP 来自 P2-only，为 32.20%，高于 M4 的 31.89%。RF 的召回为 65.28%，但精确率仅 1.39%，大量高分候选造成严重误报。')
figure('pr_curve.png','五种配置在 IoU=0.5 下的测试 PR 曲线，score 预过滤 0.001、每图上限 300。所有曲线的 101 点均值已与保存的 AP50 逐项核对。插值包络并不等于固定 score=0.25 时的精确率。')
table('固定工作点的对象匹配计数',['方法','TP','FP','FN'],[[names[m],metrics[m]['TP'],metrics[m]['FP'],metrics[m]['FN']] for m in methods],[1.4,.7,.7,.7])
figure('detection_confusion.png','M4 的检测混淆计数，score≥0.25、IoU≥0.5。背景列统计未匹配预测，背景行统计漏检。背景到背景的 TN 没有固定对象计数，留为未定义。')
para('以 M4 为例，1,283 个真值被匹配，235 个真值漏检，共计 1,518 个标注。259 个预测未匹配，故 Precision=1283/(1283+259)=83.20%，Recall=1283/1518=84.52%。这些数量与 PR 工作点指标一致。目标检测没有预先固定的背景对象集合，故不人为补出 TN 或据此构造 overall accuracy。')
h('B 配对消融与训练轨迹',2)
table('配对消融差值  单位为百分点',['比较','ΔAP50','ΔAP50:95','ΔAP小'],[
['P2-only − M2',f"{100*(metrics['p2-only']['AP50']-metrics['m2']['AP50']):+.2f}",f"{100*(metrics['p2-only']['AP50:95']-metrics['m2']['AP50:95']):+.2f}",f"{100*(metrics['p2-only']['AP-small']-metrics['m2']['AP-small']):+.2f}"],
['M4 − P2-only','+0.53','−0.45','−0.32'],['M4 − M2','+0.80','+0.50','+0.46']],[1.3,.7,.8,.7])
para('增加 P2 后，相对 M2 的 AP50、AP50:95 和 AP-small 分别提高 0.27、0.95 和 0.78 个百分点。在同一四尺度拓扑上，将 nearest 换成 DySample 后，AP50 再提高 0.53 个百分点，但 AP50:95 降低 0.45、AP-small 降低 0.32 个百分点。结果不支持“动态上采样在所有指标上均更好”的假设。')
figure('curve_ap50.png','验证集 AP50 轨迹。四个深度配置均完整训练 100 轮，曲线用于观察优化过程与选模依据。')
figure('curve_ap.png','验证集 AP50:95 轨迹。M3 与 YOLO 优化器和增强不同，曲线差异同时包含流水线差异。')
figure('loss_curves.png','各配置总训练 loss 相对第 1 轮的比例。每项 loss 原值和分量均保存，架构间 loss 的绝对大小不具可比性。')
figure('epoch_times.png','每个完整 epoch 的实际耗时，包含管线准备、训练、验证和保存。全部深度任务顺序执行，原始记录共 400 行。')
para('按验证 AP50 选出的最佳轮次为 M2 第 75 轮、M3 第 7 轮、P2-only 第 98 轮和 M4 第 82 轮。M3 后期训练 loss 继续下降，但验证 AP50 未持续改善，第 100 轮验证 AP50 为 83.21%，低于其最佳轮次。此现象与过拟合或训练设置适配不足相容，但单次运行不能区分这些原因。')
h('C 密度与计数分析',2)
table('测试密度组 AP50:95  单位为百分比',['方法','稀疏','中等','密集'],[[names[m],pct(metrics[m]['density']['sparse']['AP50:95']),pct(metrics[m]['density']['medium']['AP50:95']),pct(metrics[m]['density']['dense']['AP50:95'])] for m in methods],[1.45,.68,.68,.69])
para('稀疏组有 170 张图像和 588 个目标，中等组有 41 张和 808 个目标，密集组仅 2 张和 122 个目标。M2 的 AP50:95 从稀疏组 57.67% 降至中等组 41.27%、密集组 32.82%。M4 密集组为 32.71%，与 M2 接近。M3 在稀疏组达到 58.92%，但密集组为 29.18%。密集组只有两张，不能据此确定普遍的架构优劣。')
table('固定阈值计数及负图误检',['方法','计数 MAE','计数偏差','负图 FP'],[[names[m],rounded(extra['aggregates'][m]['count_mae']),signed_rounded(extra['aggregates'][m]['count_bias']),f"{metrics[m]['false_positives_per_negative_image']:.3f}"] for m in methods],[1.35,.75,.75,.65])
para('计数 MAE 在全部 240 张图像上，对 score≥0.25 后预测数量与真值数量的绝对差求平均。M4 的 MAE 最低，为 0.875 只/图，139 张图像数量完全正确。M2 与 P2-only 的 MAE 分别为 0.975 和 0.992。M3 的 MAE 为 1.963，平均偏差 +1.788，反映该工作点的过计数倾向。数量正确不保证目标逐一匹配，误检与漏检仍可能抵消。')
h('D 训练和推理成本',2)
table('训练与验证累计时间  单位为分钟',['方法','训练','验证','更新数'],[[names[m],f"{sum(x['training_s'] for x in logs[m])/60:.2f}",f"{sum(x['validation_s'] for x in logs[m])/60:.2f}",str(logs[m][-1]['updates'])] for m in methods[1:]],[1.45,.7,.65,.7])
timing_logs={m:[json.loads(x) for x in (RUNS/m/'epoch_timing.jsonl').read_text().splitlines()] for m in methods[1:]}
table('逐轮耗时均值  单位为秒',['方法','训练','验证','保存','整轮'],[[names[m],f"{sum(x['training_s'] for x in timing_logs[m])/100:.2f}",f"{sum(x['validation_s'] for x in timing_logs[m])/100:.2f}",f"{sum(x['checkpoint_s'] for x in timing_logs[m])/100:.2f}",f"{sum(x['epoch_wall_s'] for x in timing_logs[m])/100:.2f}"] for m in methods[1:]],[1.1,.6,.6,.6,.6])
para('整轮时间还包括数据管线准备等开销，因此通常高于训练、验证和保存三项之和。单轮记录能够区分初始化、周期性归档与阶段耗时，比较时不再依赖仅由首轮外推的 ETA。')
para('RF 两次拟合与挖掘共 152.97 分钟，其中挖掘为 147.54 分钟，首次与第二次拟合分别为 2.42 和 2.99 分钟。测试集最终三次完整推理用时 2928.96、2925.50 和 2938.12 秒，合计吞吐 0.0819 图/秒，摊销约 12.21 秒/图。测试阶段峰值进程内存为 491.02 MiB。')
table('RTX 5090 D 串行推理计时',['方法','图/秒','ms/图','P95 ms/批','显存 MiB'],[[names[m],f"{timings[m]['images_per_second']:.2f}",f"{timings[m]['amortised_ms_per_image']:.3f}",f"{timings[m]['p95_full_batch_ms']:.2f}",f"{timings[m]['peak_allocated_bytes']/2**20:.1f}"] for m in methods[1:]],[1.15,.6,.6,.65,.5])
para('M2 的吞吐为 252.28 图/秒，约为 M3 的 2.33 倍。M4 达到 247.21 图/秒，相对 M2 下降约 2.0%，推理峰值分配显存约为 M2 的 1.76 倍。P2-only 为 246.00 图/秒。M4 与 P2-only 的速度差不足 1%，三次计时不足以证明稳定的速度优势。allocated 指标不包含全部 CUDA 保留内存和系统显存占用。')
figure('tradeoff.png','小目标 AP 与摊销推理时间。P2-only 具有最高 AP-small，基础 YOLO 最快，M4 与 P2-only 耗时接近。')
para('四个深度训练均由单实例队列顺序执行。逐轮总耗时覆盖数据管线准备、训练、验证与模型保存，训练阶段和验证阶段另做 CUDA 同步计时。各配置 100 轮的总耗时分别约为 34.27、58.15、36.32 和 36.63 分钟。这里只统计模型各自完整 epoch，初始化、进程启动及最终测试另见阶段日志。系统其他后台负载没有进行硬件隔离。')

h('VI 讨论与失败分析')
h('A AP 增益与固定工作点的区别',2)
para('P2-only 的 AP-small 最高，却在 score≥0.25、IoU≥0.5 下只检出 545 个小目标中的 394 个，召回为 72.29%。M3 检出 449 个，召回 82.39%，但其 AP-small 为 31.75%。M4 检出 399 个，召回 73.21%，基础 YOLO 为 404 个、74.13%。AP 汇总候选排序和多个工作点，AR-small 还对多个 IoU 平均，不能与单一工作点召回互换。')
para('M3 的高召回伴随 593 个误检，显著多于 M2 的 258 个和 M4 的 259 个。固定阈值下置信度校准差异影响 F1 与计数。部署时可依据验证集的误报或漏检约束选择工作点，但本文未在测试集上寻找更优阈值。当前测试结果保留事先规定的统一 0.25 阈值。')
h('B 错误类型分解',2)
para('在相同一对一匹配之后，进一步把未匹配预测按其与任意真值的最大 IoU 分类。最大 IoU≥0.5 记为重复候选，0.1≤IoU<0.5 记为定位相关错误，IoU<0.1 记为背景相关错误。这是事后诊断规则，阈值固定为 0.1/0.5，不能等同于经过人工确认的因果分类。')
table('固定点未匹配预测的诊断分类',['方法','重复','定位','背景'],[[names[m],extra['aggregates'][m]['error_types']['duplicate'],extra['aggregates'][m]['error_types']['localization'],extra['aggregates'][m]['error_types']['background']] for m in methods],[1.4,.7,.7,.7])
para('RF 的 70,100 个误检中，43,246 个与任意真值几乎不重叠，25,154 个为低重叠定位候选，另有 1,700 个重复候选。重新挖掘 5,000 个困难负样本后仍出现严重误报，说明本次固定特征、采样和分类设置难以适应全图候选分布。滑窗缺少连续框回归也限制定位。当前流程未进一步做特征替换或概率校准实验，不能隔离各因素的独立贡献。')
para('相对 P2-only，M4 的 FP 从 265 降至 259，FN 从 243 降至 235，F1 从 83.39% 升至 83.86%。相对 M2，M4 多 1 个 FP，但少 9 个 FN，整体 F1 同样改善。M3 的 593 个 FP 中，357 个是定位相关、121 个是重复候选，提示在统一阈值下其较高候选响应未全部转化为精确定位。')
h('C 成功案例与双向证据',2)
para('以下案例图由本轮已保存的测试预测生成，均使用 score≥0.25、IoU≥0.5。绿色表示匹配预测，红色表示未匹配预测，青色表示漏检真值。左侧为 M2，右侧为 M4。展示例按明确规则事后选取，仅用于机制讨论。成功例从 3–10 只且 M4 无错误的列表取中点，局部收益例取 M4 相对 M2 的 TP 增量最大图像。')
figure('case_success.jpg','成功例 beach_bird_1777。两模型均正确检测 4 只鸟，FP=0、FN=0。图像来自数据集 [1]，框来自本轮保存的预测。')
para('该图中目标轮廓较完整，两模型均正确匹配 4 只鸟，FP 和 FN 均为 0。图像展示了本轮模型对较清晰场景的检测能力，同时并不提供区分新增模块贡献的证据。')
figure('case_better.jpg','局部检出收益例 beach_bird_1710。真值 12 个。M2 为 TP=8、FP=5、FN=4，M4 为 TP=11、FP=5、FN=1。')
para('该例存在拥挤候选与相邻目标。M4 比 M2 多匹配 3 个真值，同时仍保留 5 个误检。其预测数为 16，真值为 12，计数误差反而高于 M2。该图按最大 TP 增益选取，不能代表平均收益，也说明检出提升与计数准确性可能分离。')
h('D 小目标漏检与密集遮挡',2)
figure('case_miss.jpg','远距离失败例 beach_bird_1818。16 个真值全部为小目标。M2 漏检 12 个，M4 漏检 9 个。')
para('该图从不超过 20 只的图像中按 M4 漏检数最多选取。16 个真值全部为小目标，M4 比 M2 多检出 3 个，但仍漏检 9 个。低对比度和有限像素可能限制特征辨别，P2 路径只能部分缓解。第 VII 节进一步检验放大输入的影响；切片策略仍未执行。')
figure('case_dense.jpg','密集失败例 beach_bird_1790。真值 66 个，其中小目标 57 个。M2 为 TP=41、FP=14、FN=25，M4 为 TP=37、FP=12、FN=29。')
para('该图按测试集最高鸟数选择。M4 在 66 个目标中检出 37 个，M2 检出 41 个，密集拥挤仍可能造成特征混淆和定位偏差。M3 本例匹配 54 个、FP=12、FN=12，预测数恰好为 66。准确计数掩盖了对象级错误，这也是同时报告检测指标和数量误差的原因。')
h('E 负图误报与标注边界',2)
figure('case_negative.jpg','负样本标签例 beach_bird_1829。标注没有鸟类真值，M2 与 M4 分别产生 2 和 5 个高于阈值的预测。')
para('该图按 M4 在负图上的最大误检数选择。树洞和树干包含拟鸟结构，也可能存在伪装目标的解释空间。按提供标注这些预测均计为 FP，但单凭图像无法区分纯背景误报与潜在漏标。本文保留官方标注完成评价，建议后续由独立标注者复核此类样例。不能在看到测试预测后更改标签来提高当前成绩。')
h('F 研究有效性与可复现性',2)
para('四个深度配置使用相同硬件、训练轮数、实际 batch、有效 batch、种子和原始划分，但并非所有优化条件相同。M3 保留 SGD、1 轮预热和仅水平翻转，YOLO 使用 AdamW、3 轮预热和更丰富增强，因此跨架构差异仍评价完整流水线。M1 仅使用项目训练样本，深度模型额外受益于 COCO 预训练。')
para('消融结果同时支持和限制了设计假设。P2-only 相对基础 YOLO 的 AP-small 增加 0.78 个百分点，符合高分辨率表征可能保留小目标细节的动机。DySample 相对 P2-only 的 AP50 增加 0.53，但 AP75 从 47.45% 降至 45.55%，提示低 IoU 检出收益未转化为更严格的定位收益。该证据不足以证明特征对齐是主因，第 VII 节以固定权重的偏移干预进一步检验其推理贡献。')
para('每个配置只有 seed 42 的一次训练，本文不把小幅差值解释为统计显著性。随机流由统一种子派生，各轮均按 42 + epoch 生成，四个深度配置各执行 8,600 次更新。DySample grid_sample 与 Faster R-CNN roi_align 的 CUDA 反向可能非确定，不能宣称逐位复现。记录中的 400 个 epoch 及耗时行完整，M3 本轮没有 AMP 重算。')
para('训练验证使用 faster-coco-eval 1.7.0，最终验证和测试统一由 pycocotools 2.0.11 重算。结果位于 runs/5090d_batch8_seed42_20261117，每个模型保存 metrics.jsonl、epoch_timing.jsonl、epochs.csv、配置、权重和预测。RF 测试曾在两次计时后中断，恢复后重新完整执行三次，本文只统计最终成功尝试的三次计时。此前 3090 结果和旧 5090 单模型结果未混入本报告。')
para('复现应同时核对数据和运行入口。plan.json 固定数据指纹及统一设置，source_snapshot 保存本轮训练和评估源文件，stages.json 保存命令、进程和退出码。由 run_serial_benchmark.py 接收数据根目录后依次调用各方法，避免人工串联时混用配置。每项主结果均可追溯至相应 test_metrics.json，训练曲线由逐轮日志生成，案例图由保存的预测与官方标注生成。本文没有重新推理后择优替换样例分数。')


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
doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)
para('只将 DySample 的 offset 卷积权重及偏置在内存中置零，保留初始规则采样网格和其余网络参数。在当前 lp、scale=2 设置下，采样退化为规则双线性上采样。原学习偏移相对置零时的 AP50、AP75、AP-small 分别高 0.13、0.40 和 0.45 个百分点，FN 少 9 个，表明已学偏移对当前模型的推理有小幅贡献。')
para('该干预没有重新训练，零偏移会改变下游特征分布；它既不是重新训练的双线性基线，也不能把 M4 与 P2-only 的训练差异全部归因于偏移。结合主消融中 AP75 下降的现象，只能说学习偏移在当前权重内部有作用，尚不足以证明其训练机制或跨数据泛化优势。')

h('VIII 结论与后续工作')
para('在统一 100 轮、batch 8、有效 batch 16 和 seed 42 的 5090 D 横评中，深度模型明显优于重新拟合的手工特征 RF。M4 在 AP50、F1 和计数 MAE 上表现最好，P2-only 在 AP50:95 与 AP-small 上表现最好，基础 YOLO 的吞吐最高。M3 提供最高固定点召回，但精确率与计数误差代价较大。动态上采样的收益受指标与工作点影响，不能仅用一项 AP 概括模型价值。')
para('补充验证表明，输入尺度、候选抑制和工作点具有不同的成本与收益，未发现一个后处理改动能全面改善所有指标。下一步可在训练与验证集上检验重叠切片 [18]、新尺度微调和困难负样本挖掘，并保留未参与方案设计的独立数据评价泛化。当前不增加训练种子，所有结论保留单次训练限制。')
para('代码、冻结权重、逐轮记录、原始预测及补充实验计划共同构成复现依据。补充数据保存于同一运行目录的 controlled_diagnostics，包含 20 个推理条件、76 个阈值工作点和 6 组重复计时。本文区分观察、干预证据和未验证假设，不通过更改测试标签或后验选择工作点替换主实验成绩。')

columns(2,WD_SECTION_START.NEW_PAGE)
h('参考文献')
refs=[
'[1] K. Liu, “Beach Bird Detection Dataset v1.0,” Kaggle, 2026. https://www.kaggle.com/datasets/liugalhigh/beach-bird （访问日期 2026-11-17）.',
'[2] N. Dalal and B. Triggs, “Histograms of Oriented Gradients for Human Detection,” in CVPR, 2005. https://lear.inrialpes.fr/people/triggs/pubs/Dalal-cvpr05.pdf',
'[3] L. Breiman, “Random Forests,” Machine Learning, vol. 45, pp. 5–32, 2001. doi:10.1023/A:1010933404324.',
'[4] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks,” 2015. https://arxiv.org/abs/1506.01497',
'[5] K. He, X. Zhang, S. Ren, and J. Sun, “Deep Residual Learning for Image Recognition,” 2015. https://arxiv.org/abs/1512.03385',
'[6] T.-Y. Lin et al., “Feature Pyramid Networks for Object Detection,” 2016. https://arxiv.org/abs/1612.03144',
'[7] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” 2015. https://arxiv.org/abs/1506.02640',
'[8] Ultralytics, “YOLO11,” documentation and ultralytics v8.3.221 source code. https://docs.ultralytics.com/models/yolo11/ ; https://github.com/ultralytics/ultralytics/tree/v8.3.221 （访问日期 2026-11-17）.',
'[9] W. Liu, H. Lu, H. Fu, and Z. Cao, “Learning to Upsample by Learning to Sample,” ICCV, 2023. https://arxiv.org/abs/2308.15085 ; code: https://github.com/tiny-smart/dysample',
'[10] T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” 2014. https://arxiv.org/abs/1405.0312',
'[11] Torchvision contributors, “fasterrcnn_resnet50_fpn_v2,” online API documentation. https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html （访问日期 2026-11-18，实验实际使用 torchvision 0.27.0）.',
'[12] COCO contributors, “COCO API and evaluation code,” pycocotools 2.0.11. https://github.com/cocodataset/cocoapi （访问日期 2026-11-17）.',
'[13] G. Bradski, “The OpenCV Library,” Dr. Dobb’s Journal of Software Tools, 2000. OpenCV image IO and resize. https://opencv.org/',
'[14] F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011. RF implementation. https://jmlr.org/papers/v12/pedregosa11a.html',
'[15] A. Paszke et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” NeurIPS, 2019. https://arxiv.org/abs/1912.01703',
'[16] S. van der Walt et al., “scikit-image: Image Processing in Python,” PeerJ, vol. 2, e453, 2014. https://doi.org/10.7717/peerj.453',
'[17] J. Rodriguez-Juan et al., “Visual WetlandBirds Dataset: Bird Species Identification and Behavior Recognition in Videos,” Scientific Data, vol. 12, art. 1200, 2025. doi:10.1038/s41597-025-05516-5.',
'[18] F. C. Akyon, S. O. Altinuc, and A. Temizel, “Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection,” ICIP, 2022. https://arxiv.org/abs/2202.06934',
]
for i,ref in enumerate(refs):
    if i==9: doc.add_paragraph().add_run().add_break(WD_BREAK.COLUMN)
    p=para(ref); p.paragraph_format.first_line_indent=Pt(-10); p.paragraph_format.left_indent=Pt(10)
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT
    p.paragraph_format.line_spacing=Pt(10); p.paragraph_format.space_after=Pt(6)
    for r in p.runs: r.font.size=Pt(8.5)

doc.core_properties.title='自然环境中的鸟类检测 传统特征与深度模型比较及小目标检测消融研究'
doc.core_properties.author=''
doc.core_properties.subject='COMP9517 2026 T3 Group104 中文 IEEE 双栏报告'
doc.save(OUT/'COMP9517_Group104_中文项目报告_5090D补充实验版.docx')
print(OUT/'COMP9517_Group104_中文项目报告_5090D补充实验版.docx')
