# Whole-project integration record

## Branch audit

The integration starts from main commit
`d7c17115e4ea742e0740c38cc4ca516080038ca7` and audits every remote branch
present on 18 November 2026.

| Branch | Audited tip | Action |
| --- | --- | --- |
| main | d7c17115e4ea742e0740c38cc4ca516080038ca7 | Keep matched RF / M2 / M3 / P2 / M4 pipelines, diagnostics and records |
| feature/Yolo | 4b663a15d617cc9f057d568a4d77256e7ba717a5 | Merge original YOLO training, protocol trainer, resume and evaluation source with history |
| feature/resnet-model | 83f6d9feed0717a5ab0915c6f7e350082164f34c | Merge original Faster R-CNN source, robustness/warp modules and branch records with history |
| jwz55/p2-ablation-analysis | bde861526ee6bd87829357c840811077d910b004 | Already contained in main through PR #1; retain |
| jwz55/english-submission-integration | 1a40be4edff24a5d97021d8de7d38616689c9428 | Already contained in main through PR #2; extend its archive coverage |

The first packaging PR covered main's implementations but did not integrate the
two older feature branches. This whole-project change closes that gap.
The feature merges are additive and have no file conflicts against the audited
main. Original authorship stays in Git history; integration is not attribution
of the original model implementations to the integrator.

## Active source map

- `project.py`: common subprocess entry point, with dependency-free help/dry-run.
- `Random Forest/Submit/`: RF features, sample construction, hard-negative
  mining, training, inference and AP50 helper.
- `Yolo/train.py`: matched standard YOLO wrapper.
- `Yolo Improeved/`: shared YOLO train/data/model/inference/evaluator,
  P2 / P2+DySample YAMLs, upstream source pin and licenses.
- `Resnet/pipeline.py`: matched Faster R-CNN, including the explicit batch-8
  route used by the latest comparison.
- `Yolo/training/`, `Yolo/evaluation/`: original framework-based YOLO work;
  `Yolo/original_pipeline.py` adds portable dataset/checkpoint paths.
- `Resnet/Submit/`: original Faster R-CNN and its robust/warp extensions.
- `run_serial_benchmark.py`: resumable, locked, serial all-method workflow;
  the common entry point permits other CUDA GPUs and records their actual name.
- `controlled_diagnostics.py`, `robustness_benchmark.py`: matched frozen-model
  validation diagnostics.
- `tools/reporting/analyze_p2_ablation.py`: saved-prediction CPU analysis.
- `tests/`, method tests and `tools/package_submission.py`: integration
  verification, existing method checks and complete source packaging.

## Integration choices

Existing method folder names and core import paths are retained to avoid
changing recorded implementations. Subprocess isolation prevents collisions
between the original and matched modules named `model`, `data`, `dataset`
and `evaluate`. Commands use absolute script/dataset/output paths, including
paths with spaces.

`requirements.txt` is the common installation entry.
`requirements-legacy-3090.txt` preserves the old pinned environment.
Original branch requirements remain as provenance; install the common profile
for the integrated project and a compatible matched CUDA torch/torchvision pair.

Original YOLO native framework checkpoints and original Faster R-CNN checkpoints
are not interchangeable with matched-run checkpoints. The common entry point
routes them to `original-yolo/<variant>`, `original-resnet` and
`original-resnet-robust`. It never silently substitutes their metrics into the
main comparison.

The original YOLO trainer is preserved as authored, including its framework
hooks and schedule behavior. It is available for examining/reproducing the
original contribution; the matched shared loop remains the primary comparison.
The portable adapter makes final test evaluation explicit. The old standalone
scripts remain unchanged and may retain historical machine-specific paths.

## Repository versus source submission

The full Git repository keeps branch figures, raw predictions and compact
results as original research records. The clean English code ZIP includes every
model/analysis source module from both feature branches, plus the unified
interface and matched implementation. It excludes images, weights and raw
predictions as required by the code-submission specification.

`Submit/*.py` contains four zero-byte historical scaffolds from an early branch.
These are not active implementations and are excluded from the source ZIP.
The implemented modules are under the method directories listed above.

Root historical reports and `tools/reporting/` document-rendering helpers are
retained in Git but are not needed to execute the computational pipelines.
The active guides and code submission are English; historical bilingual report
artefacts are not presented as the final English report.

## Verification limits

Integration checks verify actual CLI routing, separate output/checkpoint
formats, method source coverage, Python syntax, package hashes and deterministic
packaging. P2 regression checks verify matching/resampling behavior. Full model
tests require PyTorch, torchvision, Ultralytics and initialization assets.
The consolidation environment has no PyTorch installed; no GPU training,
inference or original-branch robustness experiment was rerun for this merge.
Existing results are team records, not new integration measurements.
