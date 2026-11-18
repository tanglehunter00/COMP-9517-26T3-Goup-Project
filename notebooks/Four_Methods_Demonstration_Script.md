# Four-Method Software Demonstration Script

**Companion notebook:** `Four_Methods_Demonstration.ipynb`  
**Presenter:** [Your Name]  
**Target length:** approximately two minutes, including short screen transitions.  
**Scope:** three validation images, four frozen seed-42 matched-run checkpoints, sequential inference and common evaluation. This demonstration is separate from the full validation/test benchmark.

## Preparation — not spoken

- Open the notebook in VS Code or Jupyter and select the existing project virtual environment as the Python kernel. Confirm the repository, dataset, checkpoint paths and device in the first configuration cell.
- Keep `FULL_VALIDATION=False` for this short recording. The examples are `beach_bird_1371`, `beach_bird_1374` and `beach_bird_1658`; they illustrate different flock densities and are not a representative sample for ranking methods.
- Start a fresh execution from the first cell. Record the live command and output for all four method cells, in order. The RF stage took about one minute in the verified run; edit waiting periods to fit the final video. Do not show only previously saved cell outputs as evidence of a live run.
- Keep your webcam visible in a corner without covering commands or results. Say your name when beginning your part. The whole group video must remain within ten minutes, with approximately two minutes allocated to the software demonstration.
- Use the notebook's actual cell outputs and generated contact sheets. Do not replace them with report figures or suggest that screen-recording duration measures inference throughput.

## 0:00–0:20 — Introduce the inputs and implementations

**On screen:** notebook title, configuration cell, checkpoint preflight and validation subset selection.

**Say:**

“Hi, I'm [Your Name]. This notebook demonstrates all four bird detectors: Random Forest, baseline YOLO, Faster R-CNN, and our improved YOLO with P2 and DySample. We use three validation images and frozen seed-42 checkpoints. These examples demonstrate the workflow; they do not replace our full benchmark.”

## 0:20–0:55 — Show real code execution

**On screen:** briefly expand `demonstrate` to show the `infer` and `evaluate` calls. Run the four method cells sequentially. Show each printed command and its completion message; cut only waiting periods. Point to `--method m1`, `m2`, `m3` and `m4`.

**Say:**

“The shared function calls project.py for inference, exports original-coordinate bounding boxes to CSV, and then invokes the common evaluator. Each method finishes before the next starts. The implementations run in separate processes, avoiding conflicting module names. No training is started. Here are the actual commands and completion messages for all four methods.”

## 0:55–1:30 — Explain a success and a failure

**On screen:** the four-method contact sheet for `beach_bird_1371`. Point to the RF panel, then the baseline YOLO and M4 panels. Keep the image header and legend visible.

**Say:**

“On this rocky scene, green boxes are matched predictions, red boxes are false positives, and cyan boxes mark missed birds. Random Forest produces many false positives on the textured background. Baseline YOLO detects twelve of thirteen birds, while M4 detects ten. M4 therefore does not win on every image. Small birds and background similarity remain challenges; this example alone cannot establish their causes.”

## 1:30–2:00 — Show evaluation and close

**On screen:** current evaluation output and the final comparison table, including its demo-subset warning. Point to AP50, precision, recall and the timing disclaimer. Show the saved-output folder path.

**Say:**

“The table reports AP50, AP50-to-95, small-object AP, precision, recall, and F1, all recomputed on the same inputs. These three-image scores are illustrative. Final comparisons use our complete held-out results. The displayed wall times include startup and model loading, so they are not benchmark throughput. Predictions, overlays, metrics, logs, and checkpoint provenance are saved for inspection. Thank you.”

## Accuracy checks for the presenter — not spoken

- The example counts above come from the verified three-image run. If rerunning produces different counts, update the corresponding sentence to match the visible results.
- Overlay matching uses score ≥ 0.25 and IoU ≥ 0.5 with one-to-one matching. AP is computed from the full exported low-threshold predictions, with COCO `maxDets=300`; the overlay display threshold does not filter the AP input.
- Do not call the demo-subset table “test accuracy.” Its inputs come from validation. Validation diagnostics, final test performance and warmed-up timing are different measurements.
- Do not claim that a P2 head or DySample guarantees better predictions. Explain observed differences and limitations using the visible evidence.
- Use the complete report for final performance rankings. This short demonstration is designed to show how the code runs and what it produces.
