from ultralytics import YOLO
import torch
from pathlib import Path


def main():

    # Check GPU availability
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    # Locate project and checkpoint
    ROOT = Path(__file__).resolve().parents[2]

    LAST = (
        ROOT
        / "runs"
        / "beach_bird_detection"
        / "yolo11s_baseline"
        / "weights"
        / "last.pt"
    )

    # Check checkpoint availability
    if not LAST.exists():
        raise FileNotFoundError(f"Checkpoint not found: {LAST}")

    print("\nResuming from:", LAST)

    # Load previous training checkpoint
    model = YOLO(str(LAST))

    # Resume training
    results = model.train(resume=True)

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
