# Execution record — 2026-11-12

**Latest update:** The user cleared storage (about 241 GB free), and sequential
training resumed. M2 has completed all 100 epochs / 8,600 optimizer updates.
The selected checkpoint is epoch 93 (validation AP50 0.8771966083). Original-FP32
inference weights are in `trained/m2-best.pt` via Git LFS. M4, P2-only and M3
remain in progress; final test evaluation is deferred until all finish.
The storage-block record below is historical.

Implementation and validation:

- M2 standard YOLO11s now uses the same training and inference code as M4, with
  the original three-scale head and direct compatible COCO weight transfer.
- M3 Faster R-CNN V2 training, complete-state resume, original-coordinate
  inference, common evaluation and runtime benchmark are implemented.
- Seven YOLO/evaluation/NMS tests and two Faster R-CNN protocol tests passed.
- M3 synthetic two-epoch CUDA FP16 training, interruption after epoch one,
  complete-state resume, inference and benchmark passed. Synthetic measurements
  in `results/verification/` are diagnostics, not Beach Bird experiment results.
- Reference and accelerated COCO evaluation were compared on all existing RF
  validation predictions. Largest observed metric difference was approximately
  1.74e-18 (floating-point rounding); AP50 and AP50:95 matched exactly.
- Windows CRLF checkout is normalised when verifying the vendored Python source
  hash. Binary checkpoint and wheel verification still hashes exact bytes.

Real-data execution is blocked by storage. Full dataset audit passed: official split
1369/291/240, one class, valid readable images/labels and no exact cross-split
duplicates. Fixed 100-epoch runs started for M2, M3, M4 and P2-only.

Hardware is **RTX 3090 24 GB**, Windows, Python 3.13, torch 2.6.0+cu124,
torchvision 0.21.0+cu124, Ultralytics 8.3.221. These are local experiments;
the prescribed **Colab T4 experiments remain unexecuted**. Training jobs share
GPU/CPU resources, so logged training wall times are contended measurements.
Final inference timing must run serially after all training ends.

Storage limitation discovered during execution: C: had approximately 2.9 GB
free, insufficient for all prescribed ten-epoch archives plus final checkpoints.
Disposable synthetic weights were removed after retaining diagnostic logs. A
request for additional storage was sent before exhausting space. No official
archive retention, batch size, epoch count or input resolution was reduced.

No final real-data test accuracy or completed 100-epoch result is claimed here
until execution finishes. Existing M1 results are inherited from the repository;
M1 has not been retrained in this execution, and its trained forest was not
included in the original repository.

Preserved complete epochs: M2 **15/100**, M3 **1/100**, M4 **7/100**, P2-only
**7/100**. Each last checkpoint was reloaded successfully and contains optimizer,
scaler, scheduler progress and RNG states. Logs/configurations and a machine-readable
status are in `results/progress/`. The actual resumable checkpoints remain local
under the ignored run directories; unfinished weights have not been published as
final models. No real-data test evaluation or final runtime benchmark was run.

M2 encountered a CUDA illegal-memory-access error during validation of the next
epoch while multiple jobs shared the Windows GPU. The root cause is unconfirmed.
Its saved epoch-15 checkpoint subsequently completed standalone validation of all
291 images successfully (30,256 exported boxes). No error was hidden as success;
future execution uses the sequential runner. Remaining jobs were stopped for the
storage block and preserve their previous complete epochs.

After making at least 8 GB available, resume with the root `run_experiments.py`
command. Alternatively pass `--runs-root <larger-drive-directory>`; existing local
run directories are copied there if the destination is absent, then resumed.
Original local checkpoints are retained. The runner checks remaining archive
space before training and raises an explicit error if insufficient.
