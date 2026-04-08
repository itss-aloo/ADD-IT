import cv2
import numpy as np
from ultralytics import YOLO

def load_segmentation_model(model_path: str):
    """
    Carga el modelo de segmentación de personas.

    Entrada:
        model_path (str): Ruta del modelo (ej: yolov8n-seg.pt).

    Salida:
        model: Modelo de segmentación listo para inferencia.
    """
    try:
        model = YOLO(model_path)
    except Exception as exc:
        raise RuntimeError(f"No se pudo cargar el modelo de segmentacion: {model_path}") from exc
    return model

def segment_players(model, image: np.ndarray) -> list:
    """
    Ejecuta segmentación sobre la imagen para obtener máscaras de jugadores.

    Entrada:
        model: Modelo de segmentación.
        image (np.ndarray): Imagen (H, W, 3).

    Salida:
        segments (list):
            [
                {
                    "mask": np.ndarray (H, W) binaria,
                    "confidence": float
                },
                ...
            ]
    """
    results = model(image, verbose=False)
    image_height, image_width = image.shape[:2]
    segments = []

    for result in results:
        if result.masks is None or result.masks.data is None:
            continue

        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes
        if boxes is None:
            continue

        class_ids = boxes.cls.cpu().numpy().astype(int)
        confidences = boxes.conf.cpu().numpy()

        for mask, class_id, confidence in zip(masks, class_ids, confidences):
            if class_id != 0:
                continue

            resized_mask = resize_mask_to_image(mask, image_width, image_height)
            binary_mask = (resized_mask > 0.5).astype(np.uint8)
            segments.append(
                {
                    "mask": binary_mask,
                    "confidence": float(confidence),
                }
            )

    return segments


def resize_mask_to_image(mask: np.ndarray, image_width: int, image_height: int) -> np.ndarray:
    return cv2.resize(mask, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

def extract_mask_from_segment(segment: dict) -> np.ndarray:
    """
    Extrae la máscara binaria de un segmento individual.

    Entrada:
        segment (dict): Resultado de segmentación.

    Salida:
        mask (np.ndarray): Máscara (H, W) con valores 0 o 1.
    """
    if "mask" not in segment:
        raise KeyError("El segmento no contiene una mascara en la clave 'mask'")

    mask = segment["mask"]
    return (mask > 0).astype(np.uint8)


def clip_mask_to_bbox(mask: np.ndarray, bbox: tuple) -> np.ndarray:
    """
    Limita una mascara a la bbox de un jugador para evitar mascaras globales.
    """
    clipped_mask = np.zeros_like(mask, dtype=np.uint8)
    mask_height, mask_width = mask.shape[:2]
    x, y, width, height = bbox
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(mask_width, x + width)
    y2 = min(mask_height, y + height)

    if x2 <= x1 or y2 <= y1:
        return clipped_mask

    clipped_mask[y1:y2, x1:x2] = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    return clipped_mask


def match_mask_to_player(segment_masks: list, bbox: tuple) -> np.ndarray:
    """
    Asocia la máscara correcta a un jugador usando su bounding box.

    Entrada:
        segment_masks (list): Lista de máscaras detectadas.
        bbox (tuple): Bounding box del jugador (x, y, w, h).

    Salida:
        selected_mask (np.ndarray): Máscara del jugador correspondiente.
    """
    if not segment_masks:
        return np.array([], dtype=np.uint8)

    x, y, width, height = bbox
    best_mask = None
    best_overlap = 0

    for segment in segment_masks:
        mask = extract_mask_from_segment(segment)
        mask_height, mask_width = mask.shape[:2]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(mask_width, x + width)
        y2 = min(mask_height, y + height)

        if x2 <= x1 or y2 <= y1:
            continue

        overlap = int(mask[y1:y2, x1:x2].sum())
        if overlap > best_overlap:
            best_overlap = overlap
            best_mask = mask

    if best_mask is None:
        first_mask = extract_mask_from_segment(segment_masks[0])
        return np.zeros_like(first_mask, dtype=np.uint8)

    return clip_mask_to_bbox(best_mask, bbox)

def combine_masks(masks: list) -> np.ndarray:
    """
    Combina múltiples máscaras en una sola máscara global.

    Entrada:
        masks (list): Lista de máscaras (H, W).

    Salida:
        combined_mask (np.ndarray): Máscara unificada (H, W).
    """
    if not masks:
        return np.array([], dtype=np.uint8)

    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, (mask > 0).astype(np.uint8))

    return combined_mask

def apply_occlusion(
    original_image: np.ndarray,
    rendered_image: np.ndarray,
    player_mask: np.ndarray
) -> np.ndarray:
    """
    Aplica oclusión restaurando el jugador original por encima del logo.

    Entrada:
        original_image (np.ndarray): Imagen original (H, W, 3).
        rendered_image (np.ndarray): Imagen con logo renderizado (H, W, 3).
        player_mask (np.ndarray): Máscara del jugador (H, W) binaria.

    Salida:
        result (np.ndarray): Imagen final con oclusión aplicada.
    """
    if original_image.shape != rendered_image.shape:
        raise ValueError("original_image y rendered_image deben tener el mismo shape")

    if player_mask.shape[:2] != original_image.shape[:2]:
        raise ValueError("player_mask debe tener el mismo alto y ancho que las imagenes")

    mask = (player_mask > 0).astype(np.float32)
    mask = mask[:, :, np.newaxis]

    original = original_image.astype(np.float32)
    rendered = rendered_image.astype(np.float32)
    result = mask * original + (1.0 - mask) * rendered

    return result.astype(np.uint8)


def clip_render_to_player_mask(
    original_image: np.ndarray,
    rendered_image: np.ndarray,
    player_mask: np.ndarray,
) -> np.ndarray:
    outside_player_mask = 1 - (player_mask > 0).astype(np.uint8)
    return apply_occlusion(original_image, rendered_image, outside_player_mask)


def draw_player_mask_outline(
    image: np.ndarray,
    player_mask: np.ndarray,
    color: tuple = (255, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    debug_image = image.copy()
    binary_mask = (player_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(debug_image, contours, -1, color, thickness)

    return debug_image


def draw_player_mask_outlines(
    image: np.ndarray,
    player_masks: list,
    thickness: int = 2,
) -> np.ndarray:
    debug_image = image.copy()
    colors = [
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 255, 0),
        (255, 128, 0),
        (0, 128, 255),
    ]

    for index, player_mask in enumerate(player_masks):
        color = colors[index % len(colors)]
        debug_image = draw_player_mask_outline(debug_image, player_mask, color, thickness)

    return debug_image

def process_occlusion(
    model,
    image: np.ndarray,
    rendered_image: np.ndarray,
    detections: list,
    protected_mask: np.ndarray | None = None,
    clip_to_player_mask: bool = True,
    draw_mask_outline: bool = False,
) -> np.ndarray:
    """
    Pipeline completo de oclusión para una imagen.

    Entrada:
        model: Modelo de segmentación.
        image (np.ndarray): Imagen original.
        rendered_image (np.ndarray): Imagen con logo.
        detections (list): Lista de jugadores detectados (bbox).
        protected_mask (np.ndarray): Zona donde no se debe restaurar original.

    Salida:
        result (np.ndarray): Imagen final con oclusión.
    """
    segments = segment_players(model, image)
    matched_masks = []

    for detection in detections:
        if "bbox" not in detection:
            continue

        player_mask = match_mask_to_player(segments, detection["bbox"])
        if player_mask.size == 0:
            continue

        matched_masks.append(player_mask)

    if not matched_masks:
        return rendered_image

    combined_mask = combine_masks(matched_masks)
    player_mask = combined_mask.copy()
    if clip_to_player_mask:
        rendered_image = clip_render_to_player_mask(image, rendered_image, player_mask)

    if protected_mask is not None:
        if protected_mask.shape[:2] != combined_mask.shape[:2]:
            raise ValueError("protected_mask debe tener el mismo alto y ancho que las mascaras")
        protected_mask = (protected_mask > 0).astype(np.uint8)
        combined_mask = combined_mask * (1 - protected_mask)

    result = apply_occlusion(image, rendered_image, combined_mask)
    if draw_mask_outline:
        result = draw_player_mask_outlines(result, matched_masks)

    return result
