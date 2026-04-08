from pathlib import Path

import cv2
from ultralytics import YOLO


def draw_sponsor_detections(image, results, min_confidence: float):
    debug_image = image.copy()

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf.item())
            if confidence < min_confidence:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(debug_image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                debug_image,
                f"sponsor {confidence:.2f}",
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

    return debug_image


def process_image(
    image_path: Path,
    model,
    output_dir: Path,
    min_confidence: float,
) -> None:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"{image_path.name}: no se pudo leer")
        return

    results = model(image, imgsz=960, conf=min_confidence, verbose=False)
    debug_image = draw_sponsor_detections(image, results, min_confidence)
    output_path = output_dir / f"sponsor_{image_path.name}"
    cv2.imwrite(str(output_path), debug_image)

    detections_count = sum(
        1
        for result in results
        for box in result.boxes
        if float(box.conf.item()) >= min_confidence
    )
    print(f"{image_path.name}: {detections_count} sponsors")


if __name__ == "__main__":
    model = YOLO("models/sponsor_detector/best_colab.pt")
    image_dir = Path("data/testing")
    output_dir = Path("test_images/sponsor")
    output_dir.mkdir(parents=True, exist_ok=True)

    single_image_name = "04_image.jpg"  # Cambia a None para procesar todas las imágenes"
    min_confidence = 0.25

    if single_image_name:
        image_paths = [image_dir / single_image_name]
    else:
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    for image_path in image_paths:
        process_image(image_path, model, output_dir, min_confidence)
