from ultralytics import YOLO


def main() -> None:
    model = YOLO("yolov8n.pt")

    model.train(
        data="data/sponsor_dataset/yolo_dataset/data.yaml",
        epochs=100,
        imgsz=960,
        batch=8,
        name="sponsor_detector",
    )


if __name__ == "__main__":
    main()
