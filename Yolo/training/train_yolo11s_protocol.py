from ultralytics import YOLO
from trainers.yolo_protocol_trainer import YOLOProtocolTrainer
from pathlib import Path
import torch


def main():

    # Check GPU availability
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    PROJECT = Path("runs/beach_bird_detection").resolve()

    # Initialise COCO-pretrained YOLO11s
    model = YOLO("yolo11s.pt")

    results = model.train(
        trainer=YOLOProtocolTrainer,
        # Dataset
        data="data/Beach Bird Detection Dataset v1.0/dataset.yaml",
        # Training
        epochs=100,
        imgsz=640,
        batch=8,
        workers=2,
        device=0,
        # Optimiser
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        weight_decay=0.0005,
        momentum=0.9,
        nbs=16,
        # Warm-up (will be overridden by our custom schedule)
        warmup_epochs=0,
        # Reproducibility
        seed=42,
        deterministic=True,
        # Augmentation
        fliplr=0.5,
        flipud=0.0,
        translate=0.1,
        scale=0.3,
        degrees=0.0,
        shear=0.0,
        perspective=0.0,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.3,
        mosaic=0.5,
        close_mosaic=10,
        mixup=0.0,
        copy_paste=0.0,
        cutmix=0.0,
        multi_scale=0.0,
        auto_augment=None,
        erasing=0.0,
        # Hardware
        amp=True,
        # Checkpointing
        save=True,
        save_period=10,
        patience=0,
        # Output
        plots=True,
        project=str(PROJECT),
        name="yolo11s_protocol_final",
        exist_ok=False,
    )

    print("\nTraining completed!")

    # Validate best model
    best_path = results.save_dir / "weights" / "best.pt"
    best_model = YOLO(str(best_path))

    metrics = best_model.val(
        data="data/Beach Bird Detection Dataset v1.0/dataset.yaml",
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        conf=0.001,
        iou=0.6,
        max_det=300,
    )

    print("\nEvaluation results:")
    print("AP50:", metrics.box.map50)
    print("AP50-95:", metrics.box.map)
    print("AP75:", metrics.box.map75)
    print("Precision:", metrics.box.mp)
    print("Recall:", metrics.box.mr)


if __name__ == "__main__":
    main()
