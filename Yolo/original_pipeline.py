"""Portable adapter for the original feature/Yolo contribution.

Keeps native Ultralytics checkpoints separate from the shared comparison loop.
The original scripts and custom trainer remain in training/ for provenance.
Test-set evaluation is explicit; training does not automatically run model.val(test).
"""
import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def dataset_config(root, out):
    root = Path(root).resolve()
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            if not (root / kind / split).is_dir():
                raise ValueError(f"Missing official dataset directory: {root / kind / split}")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "dataset.yaml"
    # JSON is valid YAML and quotes Windows paths without hand-built escaping.
    content = dict(path=root.as_posix(), train="images/train", val="images/val",
                   test="images/test", names={0: "bird"})
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("train", "resume", "evaluate"))
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--variant", choices=("protocol", "baseline"), default="protocol")
    p.add_argument("--weights", type=Path)
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--device", default="0")
    a = p.parse_args()
    out = a.out.resolve()
    if a.action == "train" and out.exists() and any(out.iterdir()):
        p.error("Training output already contains files; choose a new folder or resume")
    # Imported only after argument parsing so --help works without the ML stack.
    from ultralytics import YOLO
    sys.path.insert(0, str(HERE / "training"))
    trainer = {}
    if a.variant == "protocol":
        from trainers.yolo_protocol_trainer import YOLOProtocolTrainer
        trainer = dict(trainer=YOLOProtocolTrainer)
    if a.action == "train":
        config = dataset_config(a.root, out)
        weights = a.weights or HERE.parent / "Yolo Improeved/checkpoints/yolo11s.pt"
        if not weights.is_file():
            p.error("Missing initialization; run project.py assets or supply --weights")
        model = YOLO(str(weights))
        options = dict(data=str(config), epochs=100, imgsz=640, batch=8,
                       nbs=16, workers=2, device=a.device, optimizer="AdamW",
                       lr0=.001, lrf=.01, weight_decay=.0005, momentum=.9,
                       seed=42, deterministic=True, amp=True, patience=0,
                       save=True, plots=True, project=str(out.parent),
                       name=out.name, exist_ok=True)
        if a.variant == "protocol":
            options.update(warmup_epochs=0, fliplr=.5, flipud=0., translate=.1,
                           scale=.3, degrees=0., shear=0., perspective=0.,
                           hsv_h=.015, hsv_s=.4, hsv_v=.3, mosaic=.5,
                           close_mosaic=10, mixup=0., copy_paste=0., cutmix=0.,
                           multi_scale=0., auto_augment=None, erasing=0.,
                           save_period=10)
        else:
            options.update(warmup_epochs=3, warmup_momentum=.9, rect=False)
        model.train(**trainer, **options)
    elif a.action == "resume":
        checkpoint = a.weights or out / "weights/last.pt"
        if not checkpoint.is_file():
            p.error(f"Missing native Ultralytics checkpoint: {checkpoint}")
        # Native resume restores the saved dataset/optimizer/schedule configuration.
        YOLO(str(checkpoint)).train(**trainer, resume=str(checkpoint))
    else:
        checkpoint = a.weights or out / "weights/best.pt"
        if not checkpoint.is_file():
            p.error(f"Missing native Ultralytics checkpoint: {checkpoint}")
        config = dataset_config(a.root, out)
        metrics = YOLO(str(checkpoint)).val(
            data=str(config), split=a.split, imgsz=640, batch=8,
            device=a.device, workers=2, conf=.001, iou=.6, max_det=300)
        report = dict(provenance="Original feature/Yolo implementation",
                      split=a.split, variant=a.variant, checkpoint=str(checkpoint),
                      AP50=float(metrics.box.map50), AP50_95=float(metrics.box.map),
                      AP75=float(metrics.box.map75))
        (out / f"native_{a.split}_metrics.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
