# YOLO implementations

## Matched-comparison M2

`train.py` calls the shared YOLO loop with the standard YOLO11s architecture.
M2, P2-only and M4 share data handling, non-EMA training, augmentation, optimizer,
checkpoint ranking, inference and the common evaluator. Only the architecture
differs. Use the repository root as the working directory:

```text
python project.py train --method m2 --root DATASET --run RUN
python project.py train --method m2 --root DATASET --run RUN --resume
python project.py infer --method m2 --root DATASET --run RUN --split val
python project.py evaluate --method m2 --root DATASET --run RUN --split val
```

Full runs use 100 epochs, seed 42, actual batch 8 and accumulation 2.
Test inference is explicit after all model choices are frozen.

## Original feature/Yolo contribution

The original `training/` and `evaluation/` files are merged with their
authorship history. They implement native Ultralytics training, the custom
`YOLOProtocolTrainer`, checkpoint resume and framework evaluation.
`original_pipeline.py` provides portable paths around the original trainer:

```text
python project.py original-yolo train --root DATASET --run RUN --variant protocol
python project.py original-yolo resume --root DATASET --run RUN --variant protocol
python project.py original-yolo evaluate --root DATASET --run RUN --split val
```

Use `--variant baseline` for the original framework-default baseline preset.
The adapter writes an absolute-path dataset YAML. Native resume restores the
checkpoint's saved training/data configuration. Outputs are under
`RUN/original-yolo/VARIANT/`, and checkpoints under its `weights/` directory.
Supply `--weights` for another trusted native checkpoint.

These native checkpoints and trainer hooks differ from the matched shared loop.
The original logic is preserved and has not been newly GPU-tested by the
integration. Original standalone scripts retain historical paths; prefer the
adapter. This contribution does not replace the recorded matched M2 results.
