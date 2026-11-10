from ultralytics import YOLO


def main():

    model = YOLO("runs/beach_bird_detection/yolo11s_logging_smoke_test/weights/best.pt")

    metrics = model.val(
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
