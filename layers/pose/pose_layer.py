import cv2
import numpy as np
from ultralytics import YOLO


def load_pose_model(model_name: str):
    """
    Carga el modelo de estimación de pose humana.
    Modelos: yolov8n-pose.pt yolov8s-pose.pt yolov8m-pose.pt yolov8l-pose.pt yolov8x-pose.pt

    Entrada:
        model_name (str): Nombre o ruta del modelo de pose (OpenPose, etc.).

    Salida:
        model: Modelo de pose cargado listo para inferencia.
    """
    try:
        model = YOLO(model_name)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el modelo de pose: {model_name}") from exc
    return model

def extract_person_crop(image: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Extrae la región de la imagen correspondiente a un jugador.

    Entrada:
        image (np.ndarray): Imagen completa (H, W, 3).
        bbox (tuple): Bounding box (x, y, w, h).

    Salida:
        crop (np.ndarray): Imagen recortada del jugador.
    """
    x, y, w, h = bbox
    image_height, image_width = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(image_width, x + w)
    y2 = min(image_height, y + h)

    crop = image[y1:y2, x1:x2]
    return crop

def estimate_keypoints(model, person_crop: np.ndarray) -> dict:
    """
    Estima los keypoints del cuerpo humano en un recorte de imagen.

    Entrada:
        model: Modelo de pose.
        person_crop (np.ndarray): Imagen del jugador recortada.

    Salida:
        keypoints (dict): Diccionario de puntos clave, por ejemplo:
            {
                "left_shoulder": (x, y),
                "right_shoulder": (x, y)
            }
    """
    results = model(person_crop, verbose=False)
    crop_height, crop_width = person_crop.shape[:2]
    min_keypoint_confidence = 0.30

    for result in results:
        if result.keypoints is None or result.keypoints.xy is None:
            continue
        if len(result.keypoints.xy) == 0:
            continue

        points_list = result.keypoints.xy.cpu().numpy()
        confidence_list = (
            result.keypoints.conf.cpu().numpy()
            if result.keypoints.conf is not None
            else None
        )
        boxes = result.boxes.xyxy.cpu().numpy() if result.boxes is not None else None
        best_points = select_primary_pose(points_list, boxes, crop_width, crop_height)
        if best_points is None:
            continue
        best_index = find_pose_index(points_list, best_points)
        point_confidence = (
            confidence_list[best_index]
            if confidence_list is not None and 0 <= best_index < len(confidence_list)
            else None
        )

        keypoints = {}
        for name, point_index in (
            ("left_shoulder", 5),
            ("right_shoulder", 6),
            ("left_hip", 11),
            ("right_hip", 12),
        ):
            point = extract_pose_point(best_points, point_confidence, point_index, min_keypoint_confidence)
            if point is not None:
                keypoints[name] = point

        return keypoints

    return {}

def find_pose_index(points_list: np.ndarray, selected_points: np.ndarray) -> int:
    for index, points in enumerate(points_list):
        if np.array_equal(points, selected_points):
            return index
    return 0

def extract_pose_point(
    points: np.ndarray,
    confidence: np.ndarray,
    point_index: int,
    min_confidence: float,
):
    if point_index >= len(points):
        return None

    if confidence is not None:
        if point_index >= len(confidence) or float(confidence[point_index]) < min_confidence:
            return None

    px, py = points[point_index]
    if px <= 0 and py <= 0:
        return None

    return (int(px), int(py))

def select_primary_pose(points_list: np.ndarray, boxes: np.ndarray, crop_width: int, crop_height: int):
    """
    Selecciona la pose principal dentro del crop.

    Prioriza la persona más grande y más centrada para evitar quedarse
    con jugadores secundarios que aparezcan dentro del mismo recorte.
    """
    if len(points_list) == 0:
        return None

    if boxes is None or len(boxes) != len(points_list):
        return points_list[0]

    crop_center_x = crop_width / 2.0
    crop_center_y = crop_height / 2.0
    crop_area = max(crop_width * crop_height, 1)
    best_index = 0
    best_score = float("-inf")

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        area_score = (width * height) / crop_area

        box_center_x = x1 + width / 2.0
        box_center_y = y1 + height / 2.0
        dx = (box_center_x - crop_center_x) / max(crop_width, 1)
        dy = (box_center_y - crop_center_y) / max(crop_height, 1)
        center_penalty = dx * dx + dy * dy

        score = area_score - center_penalty
        if score > best_score:
            best_score = score
            best_index = index

    return points_list[best_index]

def map_keypoints_to_image(keypoints: dict, bbox: tuple) -> dict:
    """
    Convierte keypoints del espacio local (crop) al espacio de la imagen original.

    Entrada:
        keypoints (dict): Keypoints en coordenadas del crop.
        bbox (tuple): Bounding box original (x, y, w, h).

    Salida:
        mapped_keypoints (dict): Keypoints en coordenadas globales.
    """
    x, y, _, _ = bbox
    mapped_keypoints = {}

    for name, point in keypoints.items():
        px, py = point
        mapped_keypoints[name] = (x + px, y + py)

    return mapped_keypoints

def compute_torso_region(keypoints: dict, bbox: tuple) -> dict:
    """
    Calcula la región del torso a partir de los keypoints.

    Entrada:
        keypoints (dict): Keypoints del jugador.

    Salida:
        torso (dict):
            {
                "center": (x, y),
                "width": float,
                "height": float
            }
    """
    if "left_shoulder" not in keypoints or "right_shoulder" not in keypoints:
        return {}

    left_shoulder = np.array(keypoints["left_shoulder"], dtype=np.float32)
    right_shoulder = np.array(keypoints["right_shoulder"], dtype=np.float32)
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    shoulder_vector = right_shoulder - left_shoulder
    shoulder_width = float(np.linalg.norm(shoulder_vector))
    if shoulder_width < 1.0:
        return {}

    x, y, w, h = bbox
    if "left_hip" in keypoints and "right_hip" in keypoints:
        left_hip = np.array(keypoints["left_hip"], dtype=np.float32)
        right_hip = np.array(keypoints["right_hip"], dtype=np.float32)
        torso = build_rotated_torso(
            shoulder_center=shoulder_center,
            shoulder_vector=shoulder_vector,
            shoulder_width=shoulder_width,
            hip_center=(left_hip + right_hip) / 2.0,
            bbox=bbox,
        )
        if torso:
            return torso

    return build_fallback_torso(
        shoulder_center=shoulder_center,
        shoulder_width=shoulder_width,
        bbox=bbox,
    )

def build_rotated_torso(
    shoulder_center: np.ndarray,
    shoulder_vector: np.ndarray,
    shoulder_width: float,
    hip_center: np.ndarray,
    bbox: tuple,
) -> dict:
    axis_vector = hip_center - shoulder_center
    axis_length = float(np.linalg.norm(axis_vector))
    if axis_length < max(shoulder_width * 0.35, 20.0):
        return {}

    axis_unit = axis_vector / axis_length
    normal_unit = shoulder_vector / max(shoulder_width, 1.0)
    width = shoulder_width * 1.15
    height = max(axis_length * 1.15, shoulder_width * 1.05)
    center = (shoulder_center + hip_center) / 2.0

    half_axis = axis_unit * (height / 2.0)
    half_normal = normal_unit * (width / 2.0)
    corners = np.array(
        [
            center - half_axis - half_normal,
            center - half_axis + half_normal,
            center + half_axis + half_normal,
            center + half_axis - half_normal,
        ],
        dtype=np.float32,
    )
    corners = clamp_points_to_bbox(corners, bbox)

    return {
        "mode": "rotated",
        "center": tuple(np.mean(corners, axis=0)),
        "width": width,
        "height": height,
        "corners": [tuple(point) for point in corners],
    }

def build_fallback_torso(shoulder_center: np.ndarray, shoulder_width: float, bbox: tuple) -> dict:
    x, y, w, h = bbox
    bbox_x2 = x + w
    bbox_y2 = y + h

    torso_width = max(shoulder_width * 1.2, w * 0.18)
    torso_height = max(shoulder_width * 1.45, h * 0.22)
    torso_height = min(torso_height, h * 0.65)
    torso_center_y = shoulder_center[1] + torso_height * 0.35

    x1 = max(x, shoulder_center[0] - torso_width / 2.0)
    y1 = max(y, torso_center_y - torso_height / 2.0)
    x2 = min(bbox_x2, shoulder_center[0] + torso_width / 2.0)
    y2 = min(bbox_y2, torso_center_y + torso_height / 2.0)

    if x2 <= x1 or y2 <= y1:
        return {}

    corners = np.array(
        [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2],
        ],
        dtype=np.float32,
    )
    return {
        "mode": "fallback",
        "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
        "width": x2 - x1,
        "height": y2 - y1,
        "corners": [tuple(point) for point in corners],
    }

def clamp_points_to_bbox(points: np.ndarray, bbox: tuple) -> np.ndarray:
    x, y, w, h = bbox
    points[:, 0] = np.clip(points[:, 0], x, x + w)
    points[:, 1] = np.clip(points[:, 1], y, y + h)
    return points

def process_player_pose(model, image: np.ndarray, detection: dict) -> dict:
    """
    Procesa la pose de un jugador individual.

    Entrada:
        model: Modelo de pose.
        image (np.ndarray): Imagen completa.
        detection (dict): Detección con bbox.

    Salida:
        player_pose (dict):
            {
                "bbox": (x, y, w, h),
                "keypoints": dict,
                "torso": dict
            }
    """
    bbox = detection["bbox"]
    person_crop = extract_person_crop(image, bbox)
    local_keypoints = estimate_keypoints(model, person_crop)
    keypoints = map_keypoints_to_image(local_keypoints, bbox)
    torso = compute_torso_region(keypoints, bbox)

    player_pose = {
        "bbox": bbox,
        "keypoints": keypoints,
        "torso": torso,
    }
    return player_pose

def estimate_poses(model, image: np.ndarray, detections: list) -> list:
    """
    Ejecuta estimación de pose para todos los jugadores detectados.

    Entrada:
        model: Modelo de pose.
        image (np.ndarray): Imagen completa.
        detections (list): Lista de jugadores detectados.

    Salida:
        players_pose (list): Lista con información de pose:
            [
                {
                    "bbox": (x, y, w, h),
                    "keypoints": dict,
                    "torso": dict
                },
                ...
            ]
    """
    players_pose = []

    for detection in detections:
        player_pose = process_player_pose(model, image, detection)
        players_pose.append(player_pose)

    return players_pose

def draw_torso_regions(image: np.ndarray, players_pose: list) -> np.ndarray:
    """
    Dibuja la region del torso estimada sobre la imagen.

    Entrada:
        image (np.ndarray): Imagen base.
        players_pose (list): Lista con informacion de pose de jugadores.

    Salida:
        debug_image (np.ndarray): Imagen con torsos dibujados.
    """
    debug_image = image.copy()

    for player_pose in players_pose:
        keypoints = player_pose["keypoints"]
        torso = player_pose["torso"]
        if not torso:
            continue
        center_x, center_y = torso["center"]
        corners = torso.get("corners", [])
        if corners:
            polygon = np.array(
                [[int(round(px)), int(round(py))] for px, py in corners],
                dtype=np.int32,
            ).reshape((-1, 1, 2))
            cv2.polylines(debug_image, [polygon], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.circle(debug_image, (int(center_x), int(center_y)), 4, (0, 0, 255), -1)

        for point_name in ("left_shoulder", "right_shoulder", "left_hip", "right_hip"):
            if point_name not in keypoints:
                continue
            sx, sy = keypoints[point_name]
            cv2.drawMarker(
                debug_image,
                (int(sx), int(sy)),
                (255, 0, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=14,
                thickness=2,
            )

    return debug_image


