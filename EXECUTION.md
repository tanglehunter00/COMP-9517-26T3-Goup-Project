# Execution record — 2026-11-12 to 2026-11-13

**Completed:** M2, M3, M4 and P2-only each finished 100 epochs on the official Beach Bird splits, seed 42. Final validation/test predictions, reference COCO metrics, serial inference benchmarks and selected inference weights are available.

See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for measured results and [results/status.json](results/status.json) for machine-readable completion status.

| Method | Epochs | Updates | Selected epoch | Test AP50 |
|---|---:|---:|---:|---:|
| m2 | 100 | 8600 | 93 | 0.877782 |
| m3 | 100 | 34300 | 18 | 0.871429 |
| p2-only | 100 | 8600 | 71 | 0.865865 |
| m4 | 100 | 8600 | 75 | 0.875160 |

## Verification

- Seven YOLO/evaluator/NMS tests and three Faster R-CNN tests passed (10 total).
- Forced CUDA AMP overflow test verifies replay preserves accepted parameter updates, BatchNorm counters and RNG state.
- All 40 ten-epoch archives were loaded and checked for correct epoch/update counts, data fingerprint, optimizer, AMP, scheduler and RNG states.
- All 400 epoch records are unique and consecutive; checkpoint selection follows AP50, AP50:95, then earlier epoch.
- Every exported model tensor equals its selected FP32 checkpoint tensor. SHA256 manifests match.
- Compressed predictions, density-group counts, 300-detection caps, benchmark batch/repeat/warmup settings and finite weights were verified.
- Reports: [final artifacts](results/verification/final_artifacts.json), [weight identity](results/verification/exported_weights.json), [backend parity](results/verification/evaluator-parity.json).

## Hardware and limits

Actual hardware: RTX 3090 24 GB, Windows, Python 3.13, torch 2.6.0+cu124, torchvision 0.21.0+cu124, Ultralytics 8.3.221. These are local results; prescribed Colab T4 execution remains unexecuted. Final benchmarks ran serially after training ended. Early training wall times include resource contention. Complete elapsed wall time across interruptions was not precisely measured.

Final metrics use pycocotools 2.0.11, maxDets=[1,10,300]. Training used the numerically checked faster-coco-eval 1.7.0 backend. CUDA grid_sample/roi_align backward nondeterminism is recorded; bitwise reproduction is not claimed.

M1 training/predictions are inherited from the repository and were not retrained; its prediction CSVs were re-evaluated with the common reference evaluator. The original repository did not include its trained forest.

## Interruptions and recovery

- Storage initially blocked execution with about 2.9 GB free. After the user cleared space, about 241 GB was available. All prescribed archives were retained; batch, input resolution, LR schedule and epoch count were unchanged.
- M2 encountered an illegal-memory-access error during concurrent Windows GPU execution. Its durable checkpoint passed standalone validation and completed sequential training. The underlying CUDA cause remains unconfirmed.
- M3 stopped during epoch69 when AMP scale262144 produced nonfinite gradients. The repaired loop restores buffers and CPU/CUDA RNG and replays the same accumulation group after scale backoff. Epoch69–100 logged four successful overflow recoveries and retained all scheduled updates.
- Historical storage-block snapshots remain in results/progress; they are superseded by results/status.json.

## Artifacts

trained/*.pt are inference-only selected FP32 weights distributed with Git LFS. Run git lfs pull after cloning. Full resumable checkpoints and all ten-epoch archives remain in ignored local runs/ directories. No synthetic output is presented as real-data results.

The pre-existing November10 M4/RTX5090D report is preserved as historical material; it is separate from the new matched RTX3090 experiments in the root report.
