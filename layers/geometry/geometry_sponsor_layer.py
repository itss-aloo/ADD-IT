import math

import cv2
import numpy as np


def compute_shoulder_angle(keypoints: dict) -> float:
    """
    Calcula el ángulo del torso usando los hombros.

    Entrada:
        keypoints (dict):
            {
                "left_shoulder": (x, y),
                "right_shoulder": (x, y)
            }

    Salida:
        angle (float): ángulo en radianes del eje de los hombros.
    """
    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")

    if left_shoulder is None or right_shoulder is None:
        raise ValueError("left_shoulder and right_shoulder keypoints are required")

    left_x, left_y = left_shoulder
    right_x, right_y = right_shoulder

    return math.atan2(right_y - left_y, right_x - left_x)


def compute_torso_angle(keypoints: dict) -> float:
    """
    Calcula el Ã¡ngulo del sponsor usando la columna aproximada del torso.

    Usa el eje hombros-centro -> caderas-centro como referencia principal.
    Si no hay caderas, usa la lÃ­nea de hombros como fallback.
    """
    left_hip = keypoints.get("left_hip")
    right_hip = keypoints.get("right_hip")

    if left_hip is None or right_hip is None:
        return normalize_sponsor_angle(compute_shoulder_angle(keypoints))

    left_shoulder = keypoints.get("left_shoulder")
    right_shoulder = keypoints.get("right_shoulder")

    if left_shoulder is None or right_shoulder is None:
        raise ValueError("left_shoulder and right_shoulder keypoints are required")

    shoulder_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_center_x = (left_hip[0] + right_hip[0]) / 2
    hip_center_y = (left_hip[1] + right_hip[1]) / 2

    spine_dx = hip_center_x - shoulder_center_x
    spine_dy = hip_center_y - shoulder_center_y

    if math.hypot(spine_dx, spine_dy) < 1.0:
        return normalize_sponsor_angle(compute_shoulder_angle(keypoints))

    spine_angle = math.atan2(spine_dy, spine_dx)
    return normalize_sponsor_angle(spine_angle - math.pi / 2)


def normalize_sponsor_angle(angle: float) -> float:
    """
    Normaliza el Ã¡ngulo para evitar que el sponsor se renderice boca abajo.
    """
    while angle > math.pi / 2:
        angle -= math.pi
    while angle < -math.pi / 2:
        angle += math.pi
    return angle


def bbox_to_center(bbox: tuple) -> tuple:
    """
    Convierte un bounding box en centro y dimensiones.

    Entrada:
        bbox (tuple): (x, y, w, h)

    Salida:
        center (tuple): (cx, cy)
        width (float)
        height (float)
    """
    x, y, width, height = bbox
    center = (x + width / 2, y + height / 2)

    return center, width, height

def create_local_rectangle(width: float, height: float) -> np.ndarray:
    """
    Crea un rectángulo centrado en el origen.

    Entrada:
        width (float)
        height (float)

    Salida:
        rect (np.ndarray): shape (4,2) con puntos:
            [(-w/2,-h/2), (w/2,-h/2), (w/2,h/2), (-w/2,h/2)]
    """
    half_width = width / 2
    half_height = height / 2

    return np.array(
        [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ],
        dtype=np.float32,
    )


def rotate_points(points: np.ndarray, angle: float) -> np.ndarray:
    """
    Rota un conjunto de puntos 2D alrededor del origen.

    Entrada:
        points (np.ndarray): shape (N,2)
        angle (float): en radianes

    Salida:
        rotated (np.ndarray): puntos rotados
    """
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    rotation_matrix = np.array(
        [
            [cos_angle, -sin_angle],
            [sin_angle, cos_angle],
        ],
        dtype=np.float32,
    )

    return points @ rotation_matrix.T

def translate_points(points: np.ndarray, center: tuple) -> np.ndarray:
    """
    Traslada puntos al centro real en la imagen.

    Entrada:
        points (np.ndarray): shape (N,2)
        center (tuple): (cx, cy)

    Salida:
        translated (np.ndarray): puntos en coordenadas de imagen
    """
    return points + np.array(center, dtype=np.float32)

def build_oriented_sponsor_quad(
    bbox: tuple,
    keypoints: dict
) -> list:
    """
    Genera el cuadrilátero del sponsor alineado con el torso.

    Entrada:
        bbox (tuple): (x, y, w, h) del sponsor detectado
        keypoints (dict): keypoints del pose

    Salida:
        quad (list):
            [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            en orden consistente para warpPerspective
    """
    center, width, height = bbox_to_center(bbox)
    angle = normalize_sponsor_angle(compute_torso_angle(keypoints))
    local_rectangle = create_local_rectangle(width, height)
    rotated_rectangle = rotate_points(local_rectangle, angle)
    translated_rectangle = translate_points(rotated_rectangle, center)

    return [tuple(point) for point in translated_rectangle.tolist()]


def fallback_axis_aligned_quad(bbox: tuple) -> list:
    """
    Genera un cuadrilátero sin rotación (fallback si falla pose).

    Entrada:
        bbox (tuple): (x, y, w, h)

    Salida:
        quad (list):
            [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    """
    x, y, width, height = bbox

    return [
        (x, y),
        (x + width, y),
        (x + width, y + height),
        (x, y + height),
    ]

def get_sponsor_quad(
    bbox: tuple,
    keypoints: dict
) -> list:
    """
    Función principal que decide si usar geometría orientada o fallback.

    Entrada:
        bbox (tuple): bbox del sponsor
        keypoints (dict): keypoints del jugador

    Salida:
        quad (list): cuadrilátero final listo para rendering
    """
    try:
        return build_oriented_sponsor_quad(bbox, keypoints)
    except (KeyError, TypeError, ValueError):
        return fallback_axis_aligned_quad(bbox)


def bbox_area(bbox: tuple) -> float:
    """
    Calcula el area de una bbox en formato (x, y, w, h).
    """
    _, _, width, height = bbox
    return max(0, width) * max(0, height)


def bbox_intersection_area(first_bbox: tuple, second_bbox: tuple) -> float:
    """
    Calcula el area de solape entre dos bboxes en formato (x, y, w, h).
    """
    first_x, first_y, first_w, first_h = first_bbox
    second_x, second_y, second_w, second_h = second_bbox

    x1 = max(first_x, second_x)
    y1 = max(first_y, second_y)
    x2 = min(first_x + first_w, second_x + second_w)
    y2 = min(first_y + first_h, second_y + second_h)

    return max(0, x2 - x1) * max(0, y2 - y1)


def torso_to_bbox(torso: dict) -> tuple | None:
    """
    Convierte un torso con corners a su bbox envolvente (x, y, w, h).
    """
    corners = torso.get("corners", [])
    if not corners:
        return None

    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    x1 = min(xs)
    y1 = min(ys)
    x2 = max(xs)
    y2 = max(ys)

    return (x1, y1, x2 - x1, y2 - y1)


def sponsor_torso_overlap_ratio(sponsor_bbox: tuple, player_pose: dict) -> float:
    """
    Calcula quÃ© porcentaje del sponsor cae dentro del torso del jugador.
    """
    torso_bbox = torso_to_bbox(player_pose.get("torso", {}))
    if torso_bbox is None:
        return 0.0

    sponsor_area = max(bbox_area(sponsor_bbox), 1)
    return bbox_intersection_area(sponsor_bbox, torso_bbox) / sponsor_area


def bbox_center_point(bbox: tuple) -> tuple:
    """
    Devuelve el centro de una bbox en formato (x, y, w, h).
    """
    center, _, _ = bbox_to_center(bbox)
    return center


def point_inside_bbox(point: tuple, bbox: tuple) -> bool:
    """
    Indica si un punto (x, y) cae dentro de una bbox (x, y, w, h).
    """
    px, py = point
    x, y, width, height = bbox
    return x <= px <= x + width and y <= py <= y + height


def distance_squared(first_point: tuple, second_point: tuple) -> float:
    """
    Distancia euclÃ­dea al cuadrado entre dos puntos.
    """
    dx = first_point[0] - second_point[0]
    dy = first_point[1] - second_point[1]
    return dx * dx + dy * dy


def match_sponsor_to_player(sponsor: dict, players_pose: list) -> dict:
    """
    Asocia un sponsor detectado con el jugador mas probable.

    Prioridad:
        1. Mayor porcentaje del sponsor dentro del torso.
        2. Centro del sponsor dentro de la bbox del jugador.
        3. Jugador mas cercano al centro del sponsor.
    """
    if not players_pose:
        return {}

    sponsor_bbox = sponsor["bbox"]
    sponsor_center = bbox_center_point(sponsor_bbox)
    torso_candidates = [
        (player_pose, sponsor_torso_overlap_ratio(sponsor_bbox, player_pose))
        for player_pose in players_pose
    ]
    torso_candidates = [
        (player_pose, overlap_ratio)
        for player_pose, overlap_ratio in torso_candidates
        if overlap_ratio > 0.0
    ]

    if torso_candidates:
        return max(torso_candidates, key=lambda candidate: candidate[1])[0]

    bbox_candidates = [
        player_pose
        for player_pose in players_pose
        if point_inside_bbox(sponsor_center, player_pose["bbox"])
    ]

    if bbox_candidates:
        return max(
            bbox_candidates,
            key=lambda player_pose: bbox_intersection_area(sponsor_bbox, player_pose["bbox"]),
        )

    return min(
        players_pose,
        key=lambda player_pose: distance_squared(
            sponsor_center,
            bbox_center_point(player_pose["bbox"]),
        ),
    )


def match_sponsors_to_players(sponsors: list, players_pose: list) -> list:
    """
    Asocia una lista de sponsors con sus jugadores correspondientes.
    """
    return [
        {
            "sponsor": sponsor,
            "player": match_sponsor_to_player(sponsor, players_pose),
        }
        for sponsor in sponsors
    ]

