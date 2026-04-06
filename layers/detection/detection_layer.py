import cv2
import numpy as np
from ultralytics import YOLO


def load_detection_model(model_path: str):
    """
    Carga el modelo de detección de jugadores.
    Modelos: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

    Entrada:
        model_path (str): Ruta del modelo (YOLO u otro).

    Salida:
        model: Modelo cargado listo para inferencia.
    """
    try:
        model = YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el modelo de deteccion: {model_path}") from exc
    return model

def detect_players(model, image: np.ndarray) -> list:
    """
    Ejecuta el modelo de detección sobre una imagen.

    Entrada:
        model: Modelo de detección cargado.
        image (np.ndarray): Imagen (H, W, 3).

    Salida:
        detections (list): Lista de detecciones con formato:
            [
                {
                    "bbox": (x, y, w, h),
                    "confidence": float
                },
                ...
            ]
    """
    results = model(image, verbose=False)
    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls.item())
            if class_id != 0:
                continue

            raw_bbox = box.xyxy[0].tolist()
            bbox = convert_bbox_format(raw_bbox)
            confidence = float(box.conf.item())

            detections.append(
                {
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

    return detections

def filter_detections(detections: list, min_confidence: float, min_area: int) -> list:
    """
    Filtra detecciones irrelevantes por confianza y tamaño.

    Entrada:
        detections (list): Lista de detecciones.
        min_confidence (float): Umbral mínimo de confianza.
        min_area (int): Área mínima del bounding box.

    Salida:
        filtered_detections (list): Lista filtrada.
    """
    filtered_detections = []

    for detection in detections:
        x, y, w, h = detection["bbox"]
        area = w * h
        confidence = detection["confidence"]

        if confidence >= min_confidence and area >= min_area:
            filtered_detections.append(detection)

    return filtered_detections

def convert_bbox_format(raw_bbox) -> tuple:
    """
    Convierte bounding boxes al formato estándar (x, y, w, h).

    Entrada:
        raw_bbox: Bounding box en formato del modelo.

    Salida:
        bbox (tuple): (x, y, w, h)
    """
    x1, y1, x2, y2 = raw_bbox
    x = int(x1)
    y = int(y1)
    w = int(x2 - x1)
    h = int(y2 - y1)
    return (x, y, w, h)

def detect_and_filter_players(model, image: np.ndarray, min_confidence: float, min_area: int) -> list:
    """
    Pipeline completo de detección y filtrado de jugadores.

    Entrada:
        model: Modelo de detección.
        image (np.ndarray): Imagen (H, W, 3).
        min_confidence (float): Umbral de confianza.
        min_area (int): Área mínima.

    Salida:
        players (list): Lista final de jugadores detectados:
            [
                {
                    "bbox": (x, y, w, h),
                    "confidence": float
                },
                ...
            ]
    """
    detections = detect_players(model, image)
    players = filter_detections(detections, min_confidence, min_area)
    return players

def draw_detections(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Dibuja bounding boxes sobre la imagen para visualización.

    Entrada:
        image (np.ndarray): Imagen base.
        detections (list): Lista de detecciones.

    Salida:
        debug_image (np.ndarray): Imagen con bounding boxes dibujados.
    """
    debug_image = image.copy()

    for detection in detections:
        x, y, w, h = detection["bbox"]
        confidence = detection["confidence"]

        cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            debug_image,
            f"{confidence:.2f}",
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    return debug_image











