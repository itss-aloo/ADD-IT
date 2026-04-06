import cv2
from pathlib import Path

from layers.detection.detection_layer import (
    detect_and_filter_players,
    draw_detections,
    load_detection_model,
)


if __name__ == "__main__":
    input_dir = Path("data/images")
    output_dir = Path("test_images")
    output_dir.mkdir(exist_ok=True)

    model = load_detection_model("yolov8n.pt")

    image_paths = sorted(
        [
            path
            for path in input_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        detections = detect_and_filter_players(
            model,
            image,
            min_confidence=0.7,
            min_area=50000,
        )
        debug_image = draw_detections(image, detections)

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), debug_image)
        print(f"{image_path.name}: {len(detections)} detecciones")
