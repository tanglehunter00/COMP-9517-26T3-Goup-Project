# Experiment code and reproduction / 实验代码与复现

## Scope / 范围

This revision contains the full 5090 D serial training queue, per-epoch loss and
wall-time recording, controlled inference diagnostics, corruption robustness
comparison, and analysis/report generation code. Existing model architectures,
RF features, and primary test scores are preserved. All experiment random streams
derive from seed 42; there is no additional training-seed sweep.

本次包含全部训练及每轮计时改动、受控补充实验、人工扰动横评、结果分析和报告生成代码。
深度方法各训练100轮，实际batch 8、累积2步、有效batch 16，验证/推理batch 8。
RF不具有epoch或训练batch；单独在CPU拟合。模型架构和既有测试结果未被补充实验替换。

## Environment / 环境

Measured hardware: RTX 5090 D 32 GB, Ryzen 7 9800X3D, Windows, Python 3.13.2.
Install a matched CUDA-enabled torch/torchvision pair before other dependencies.
The measured pair is `torch==2.12.0+cu130`, `torchvision==0.27.0+cu130`.
Use `requirements-5090d.txt` for this run; the root `requirements.txt` retains the
historical 3090 torch/torchvision pins. Do not install the historical pins over
the 5090 environment.

```powershell
$python = '.\Yolo Improeved\.venv\Scripts\python.exe'
& $python -m pip install -r requirements-5090d.txt
& $python -m pip install -r tools/reporting/requirements.txt
$datasetRoot = '.\Yolo Improeved\work\beach-bird\Beach Bird Detection Dataset v1.0'
$runDir = '.\runs\5090d_batch8_seed42_20261117'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
$env:PYTHONIOENCODING='utf-8'
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
```

The dataset root must contain official `images/{train,val,test}` and
`labels/{train,val,test}` directories. Dataset and checkpoint download entry
points remain documented in the method READMEs. Selected 5090 checkpoint files
remain local under the run directory; historical `trained/` weights are not
interchangeable with the new SHA256-checked checkpoints.

## Sequential experiments / 串行实验

Run each command only after the previous one exits successfully. Do not launch
the queue and diagnostics simultaneously. Queue locking prevents multiple queue
owners; diagnostic runners rely on this explicit sequencing. RF image workers
belong to a single active method, not concurrent model experiments.

```powershell
& $python -u run_serial_benchmark.py --root $datasetRoot --out $runDir
& $python -u controlled_diagnostics.py --root $datasetRoot --run $runDir
& $python -u robustness_benchmark.py --root $datasetRoot --run $runDir --rf-workers 8
```

- `run_serial_benchmark.py`: M2 → M3 → P2-only → M4 → RF; complete training
  before final serial validation/test evaluation. Records loss components,
  selected epochs, every epoch's training/validation/save/total time and final timing.
- `controlled_diagnostics.py`: 20 validation inference conditions for input
  640/960/1280, NMS IoU 0.3/0.5/0.6/0.7/0.9, in-memory M4 offset removal;
  additionally 76 fixed confidence-grid points and six three-repeat timings.
- `robustness_benchmark.py`: M2 → M3 → P2-only → M4 → RF, each on all 291 validation
  images under Gaussian noise sigma 5/15/30, Gaussian blur and brightness ×0.5.
  Records input hashes, parity checks, predictions and metrics for all 25 conditions.
  RF uses eight image workers, one forest thread each; this is not a latency benchmark.

深度训练和两类补充实验均按方法顺序执行。补充实验仅使用验证集的291张图像、1843个框，
不重新训练、不调测试阈值。高斯噪声三个强度共享按图像名称和seed 42派生的标准正态场；
全部方法逐像素输入一致。详情见 [controlled diagnostics](reports/CONTROLLED_DIAGNOSTICS.md)
及 [robustness report](reports/ROBUSTNESS_REPORT.md)。

Completed conditions are reused. Resolution timing in the controlled runner is
repeated on rerun. Interrupted, incomplete RF conditions restart from the beginning;
completed conditions are retained. Failures do not silently lower batch or skip methods.

## Analysis and report sources / 分析及报告源码

`tools/reporting/reporting_paths.py` resolves the repository root without local
drive assumptions. These environment variables optionally override locations:

```powershell
$env:COMP9517_RUN_DIR = (Resolve-Path $runDir).Path
$env:COMP9517_DATA_ROOT = (Resolve-Path $datasetRoot).Path
# Optional: $env:COMP9517_REPORT_OUTPUT = '<absolute-output-directory>'
```

With a completed run and original predictions available, execute in this order:

```powershell
& $python tools/reporting/analyze.py
& $python tools/reporting/enhance_assets.py
& $python tools/reporting/supplement_assets.py
& $python tools/reporting/robustness_assets.py
& $python tools/reporting/build_report.py
& $python tools/reporting/verify.py
& $python tools/reporting/add_supplement.py
& $python tools/reporting/build_supplement_report.py
& $python tools/reporting/verify_supplement.py
& $python tools/reporting/add_robustness.py
& $python tools/reporting/build_robustness_report.py
& $python tools/reporting/verify_robustness.py
```

The three builders produce the successive report versions under `outputs/`.
The two `add_*.py` scripts regenerate the later builders from earlier versions.
Verifiers cross-check source metrics, tables, checkpoints, epoch logs and counts;
supplement verifiers also export compact result records and source snapshots.
Analysis assets next to the scripts and outputs are ignored by Git. The report
contains frozen numeric discussion of this specific run, not a general template
for substituting new experiments without revising the interpretation.

DOCX generation uses python-docx. PDF export was done with Microsoft Word and
all pages were visually inspected; a matching Word/LibreOffice installation is
needed to export a regenerated DOCX to PDF. The report uses Chinese fonts.
The measured analysis libraries were matplotlib 3.10.5 and Pillow 11.3.0.

## Published records / 提交记录

`reports/` contains the latest canonical DOCX/PDF, main comparison, 400 epoch rows,
controlled-condition summary and robustness CSV/JSON. Historical report copies,
datasets, environments, checkpoints and large raw prediction arrays remain local.
Report AP/F1 tables use percent; raw CSV AP/F1 use 0–1, `drop_pp` uses percentage
points, and retention uses percent. The original 3090 results remain separately
labelled in `results/` and the historical report.
