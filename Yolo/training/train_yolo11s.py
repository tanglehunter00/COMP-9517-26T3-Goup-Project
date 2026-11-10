from ultralytics import YOLO
import torch
from pathlib import Path


def main():

    # Check GPU availability
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    PROJECT = Path("runs/beach_bird_detection").resolve()

    # Initialise YOLO11s using COCO-pretrained weights
    model = YOLO("yolo11s.pt")

    results = model.train(
        data="data/Beach Bird Detection Dataset v1.0/dataset.yaml",
        # Training settings
        epochs=100,
        imgsz=640,
        batch=8,
        nbs=16,  # Gradient accumulation: 2
        # Optimiser settings
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,  # Final LR: 0.00001
        momentum=0.9,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.9,
        # Hardware settings
        device=0,
        workers=2,
        amp=True,
        # Reproducibility
        seed=42,
        deterministic=True,
        # Other settings
        patience=0,  # Disable early stopping
        rect=False,
        plots=True,
        project=str(PROJECT),
        name="yolo11s_baseline",
    )

    print("\nTraining completed!")

    # Validate best model
    best_path = results.save_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_path))

    metrics = best_model.val(
        data="data/Beach Bird Detection Dataset v1.0/dataset.yaml",
        split="val",
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
    )

    print("\nValidation results:")
    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()
