from ultralytics import YOLO
from trainers.yolo_protocol_trainer import YOLOProtocolTrainer

CHECKPOINT = (
    "runs/beach_bird_detection/" "yolo11s_protocol_smoke_test/weights/epoch0.pt"
)


def main():
    print("Resuming from:", CHECKPOINT)

    model = YOLO(CHECKPOINT)

    results = model.train(
        trainer=YOLOProtocolTrainer,
        resume=CHECKPOINT,
    )

    print("\nResume test completed!")
    print("Results directory:", results.save_dir)


if __name__ == "__main__":
    main()
