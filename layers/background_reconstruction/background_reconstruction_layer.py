import numpy as np
import cv2

try:
    from simple_lama_inpainting import SimpleLama
except ImportError:  # pragma: no cover - dependencia opcional
    SimpleLama = None


def clip_bbox_to_frame(frame_shape, bbox):
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = bbox

    x1 = max(0, int(round(x)))
    y1 = max(0, int(round(y)))
    x2 = min(frame_width, int(round(x + width)))
    y2 = min(frame_height, int(round(y + height)))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2 - x1, y2 - y1)


def merge_bboxes(first_bbox, second_bbox):
    if first_bbox is None:
        return second_bbox
    if second_bbox is None:
        return first_bbox

    first_x, first_y, first_w, first_h = first_bbox
    second_x, second_y, second_w, second_h = second_bbox
    x1 = min(first_x, second_x)
    y1 = min(first_y, second_y)
    x2 = max(first_x + first_w, second_x + second_w)
    y2 = max(first_y + first_h, second_y + second_h)
    return (x1, y1, x2 - x1, y2 - y1)


def crop_to_bbox(image, bbox):
    x, y, width, height = bbox
    return image[y:y + height, x:x + width]


def mask_to_bbox(mask):
    if mask is None or mask.size == 0:
        return None

    binary_mask = (mask > 0).astype(np.uint8)
    if not np.any(binary_mask):
        return None

    ys, xs = np.where(binary_mask > 0)
    x1 = int(xs.min())
    y1 = int(ys.min())
    x2 = int(xs.max()) + 1
    y2 = int(ys.max()) + 1
    return (x1, y1, x2 - x1, y2 - y1)


def generate_logo_mask_from_bbox(frame_shape, logo_bbox):
    """
    Genera una máscara binaria del logo.

    A partir del bounding box detectado del sponsor, crea una imagen máscara
    del mismo tamaño que el frame donde:
    - blanco (255) → zona del logo a eliminar
    - negro (0) → resto de la imagen

    Esta máscara es la entrada directa para el inpainting.
    """
    frame_height, frame_width = frame_shape[:2]
    x, y, width, height = logo_bbox

    logo_mask = np.zeros((frame_height, frame_width), dtype=np.uint8)

    x1 = max(0, int(x))
    y1 = max(0, int(y))
    x2 = min(frame_width, int(x + width))
    y2 = min(frame_height, int(y + height))

    if x2 <= x1 or y2 <= y1:
        return logo_mask

    logo_mask[y1:y2, x1:x2] = 255
    return logo_mask

def expand_logo_mask(logo_mask, expansion_pixels):
    """
    Expande (dilata) la máscara del logo.

    Añade un margen alrededor del bounding box para asegurarse de que
    se eliminan completamente los bordes del logo original y evitar
    artefactos tras el inpainting.

    expansion_pixels define cuántos píxeles se expande en cada dirección.
    """
    if expansion_pixels <= 0:
        return logo_mask.copy()

    kernel_size = 2 * int(expansion_pixels) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded_mask = cv2.dilate(logo_mask, kernel, iterations=1)
    return expanded_mask

def refine_logo_mask_with_segmentation(logo_mask, player_mask):
    """
    Refina la máscara del logo usando la segmentación del jugador.

    Asegura que la máscara solo afecta a la zona de la camiseta,
    evitando borrar píxeles fuera del cuerpo del jugador.

    Interseca la máscara del logo con la máscara del jugador.
    """
    if player_mask is None:
        return logo_mask.copy()

    player_mask_uint8 = np.where(player_mask > 0, 255, 0).astype(np.uint8)
    refined_mask = cv2.bitwise_and(logo_mask, player_mask_uint8)
    return refined_mask

def apply_inpainting(frame, logo_mask, inpaint_radius, method):
    """
    Aplica inpainting sobre la región del logo.

    Usa algoritmos clásicos (OpenCV) para rellenar la zona del logo
    a partir de los píxeles circundantes, reconstruyendo la textura
    de la camiseta.

    method puede ser:
    - cv2.INPAINT_TELEA (rápido, recomendado para MVP)
    - cv2.INPAINT_NS (más suave, más lento)
    """
    if method not in (cv2.INPAINT_TELEA, cv2.INPAINT_NS):
        method = cv2.INPAINT_TELEA

    inpaint_mask = np.where(logo_mask > 0, 255, 0).astype(np.uint8)

    if not np.any(inpaint_mask):
        return frame.copy()

    return cv2.inpaint(frame, inpaint_mask, float(inpaint_radius), method)


def apply_inpainting_on_crop(frame, logo_mask, roi_bbox, inpaint_radius, method):
    """
    Aplica inpainting solo dentro de un crop local de la imagen.

    El resultado se pega de vuelta al frame original para reducir el
    contexto visible al algoritmo y evitar que tome demasiado fondo lejano.
    """
    clipped_roi = clip_bbox_to_frame(frame.shape, roi_bbox)
    if clipped_roi is None:
        return frame.copy()

    x, y, width, height = clipped_roi
    crop = crop_to_bbox(frame, clipped_roi)
    crop_mask = crop_to_bbox(logo_mask, clipped_roi)

    if crop.size == 0 or not np.any(crop_mask):
        return frame.copy()

    inpainted_crop = apply_inpainting(crop, crop_mask, inpaint_radius, method)
    inpainted_frame = frame.copy()
    inpainted_frame[y:y + height, x:x + width] = inpainted_crop
    return inpainted_frame


def load_lama_model(device=None):
    """
    Carga el modelo LaMa para inpainting.

    Requiere el paquete `simple-lama-inpainting`. En la primera carga,
    el paquete descargara el modelo TorchScript si aun no existe en cache.
    """
    if SimpleLama is None:
        raise ImportError(
            "No se pudo importar simple_lama_inpainting. "
            "Instala el paquete `simple-lama-inpainting` para usar LaMa."
        )

    if device is None:
        return SimpleLama()

    return SimpleLama(device=device)


def prepare_mask_for_lama(logo_mask):
    """
    Convierte la mascara del sponsor al formato esperado por LaMa.

    LaMa espera una mascara uint8 donde:
    - 255 representa la region a reconstruir
    - 0 representa la region preservada
    """
    return np.where(logo_mask > 0, 255, 0).astype(np.uint8)


def create_context_ring_mask(logo_mask, player_mask=None, ring_pixels=12):
    """
    Crea una corona alrededor de la mascara del sponsor para muestrear
    los colores reales de la camiseta cercana.
    """
    base_mask = np.where(logo_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(base_mask):
        return np.zeros_like(base_mask, dtype=np.uint8)

    kernel_size = max(3, 2 * int(ring_pixels) + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(base_mask, kernel, iterations=1)
    ring_mask = cv2.subtract(dilated_mask, base_mask)

    if player_mask is not None and player_mask.size != 0:
        player_mask_uint8 = np.where(player_mask > 0, 255, 0).astype(np.uint8)
        ring_mask = cv2.bitwise_and(ring_mask, player_mask_uint8)

    return ring_mask


def color_correct_reconstructed_region(
    original_frame,
    reconstructed_frame,
    logo_mask,
    player_mask=None,
    ring_pixels=12,
):
    """
    Ajusta el color de la region reconstruida para acercarlo a la camiseta.

    Usa una corona alrededor del sponsor como referencia cromatica en espacio
    LAB y aplica un desplazamiento suave de medias sobre la zona reconstruida.
    """
    region_mask = np.where(logo_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(region_mask):
        return reconstructed_frame.copy()

    original_lab = cv2.cvtColor(original_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    reconstructed_lab = cv2.cvtColor(reconstructed_frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    corrected_lab = reconstructed_lab.copy()

    height, width = region_mask.shape
    global_ring_mask = create_context_ring_mask(region_mask, player_mask, ring_pixels=ring_pixels)
    global_reference_pixels = original_lab[global_ring_mask > 0]
    if len(global_reference_pixels) == 0:
        return reconstructed_frame.copy()

    global_palette = build_local_color_palette(global_reference_pixels)
    if len(global_palette) == 0:
        return reconstructed_frame.copy()

    for x in range(width):
        column_region = region_mask[:, x] > 0
        if not np.any(column_region):
            continue

        x1 = max(0, x - 2)
        x2 = min(width, x + 3)
        column_ring = np.zeros((height, width), dtype=np.uint8)
        column_ring[:, x1:x2] = global_ring_mask[:, x1:x2]
        column_reference_pixels = original_lab[column_ring > 0]
        column_palette = (
            build_local_color_palette(column_reference_pixels)
            if len(column_reference_pixels) > 0
            else global_palette
        )
        if len(column_palette) == 0:
            column_palette = global_palette

        corrected_column = project_pixels_to_palette(
            corrected_lab[column_region, x],
            column_palette,
        )
        corrected_lab[column_region, x] = corrected_column

    corrected_bgr = cv2.cvtColor(corrected_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    composed_frame = reconstructed_frame.copy()
    composed_frame[region_mask > 0] = corrected_bgr[region_mask > 0]
    return composed_frame


def build_local_color_palette(reference_pixels, max_colors=4):
    """
    Construye una paleta compacta de colores reales de la camiseta.

    La paleta se obtiene con k-means en espacio LAB para capturar
    los tonos dominantes de la corona alrededor del sponsor.
    """
    if reference_pixels is None or len(reference_pixels) == 0:
        return np.empty((0, 3), dtype=np.float32)

    pixels = np.asarray(reference_pixels, dtype=np.float32)
    if pixels.ndim == 1:
        pixels = pixels.reshape(1, -1)
    elif pixels.ndim > 2:
        pixels = pixels.reshape(-1, pixels.shape[-1])

    unique_count = len(pixels)
    if unique_count == 1:
        return pixels.astype(np.float32)

    cluster_count = max(1, min(max_colors, unique_count))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        20,
        1.0,
    )

    compactness, labels, centers = cv2.kmeans(
        pixels,
        cluster_count,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    _ = compactness
    labels = labels.reshape(-1)

    ordered_centers = []
    for cluster_index in range(cluster_count):
        members = pixels[labels == cluster_index]
        if len(members) == 0:
            continue
        center = centers[cluster_index]
        ordered_centers.append((len(members), center))

    ordered_centers.sort(key=lambda item: item[0], reverse=True)
    return np.array([center for _, center in ordered_centers], dtype=np.float32)


def project_pixels_to_palette(pixels_lab, palette_lab):
    """
    Proyecta los pixeles reconstruidos a la paleta local de la camiseta.

    Mantiene la luminancia original del parche, pero fuerza los canales
    cromaticos A/B a coincidir con alguno de los colores reales vecinos.
    """
    if len(pixels_lab) == 0 or len(palette_lab) == 0:
        return pixels_lab

    pixels = np.asarray(pixels_lab, dtype=np.float32)
    palette = np.asarray(palette_lab, dtype=np.float32)

    pixel_ab = pixels[:, 1:3][:, np.newaxis, :]
    palette_ab = palette[np.newaxis, :, 1:3]
    distances = np.sum((pixel_ab - palette_ab) ** 2, axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    nearest_palette = palette[nearest_indices]

    projected = pixels.copy()
    projected[:, 1] = nearest_palette[:, 1]
    projected[:, 2] = nearest_palette[:, 2]
    projected[:, 0] = np.clip(projected[:, 0], 0, 255)
    return projected


def apply_lama_inpainting(frame, logo_mask, lama_model):
    """
    Aplica LaMa inpainting sobre una imagen completa.
    """
    lama_mask = prepare_mask_for_lama(logo_mask)
    if not np.any(lama_mask):
        return frame.copy()

    inpainted_image = lama_model(frame, lama_mask)
    return cv2.cvtColor(np.array(inpainted_image), cv2.COLOR_RGB2BGR)


def apply_lama_inpainting_on_crop(frame, logo_mask, roi_bbox, lama_model):
    """
    Aplica LaMa solo dentro de un crop local y pega el resultado al frame.
    """
    clipped_roi = clip_bbox_to_frame(frame.shape, roi_bbox)
    if clipped_roi is None:
        return frame.copy()

    x, y, width, height = clipped_roi
    crop = crop_to_bbox(frame, clipped_roi)
    crop_mask = crop_to_bbox(logo_mask, clipped_roi)

    if crop.size == 0 or not np.any(crop_mask):
        return frame.copy()

    inpainted_crop = apply_lama_inpainting(crop, crop_mask, lama_model)
    if inpainted_crop.shape[:2] != crop.shape[:2]:
        inpainted_crop = cv2.resize(
            inpainted_crop,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    crop_mask_binary = np.where(crop_mask > 0, 255, 0).astype(np.uint8)
    composed_crop = crop.copy()
    composed_crop[crop_mask_binary > 0] = inpainted_crop[crop_mask_binary > 0]
    inpainted_frame = frame.copy()
    inpainted_frame[y:y + height, x:x + width] = composed_crop
    return inpainted_frame

def smooth_reconstructed_region(inpainted_frame, logo_mask, blur_kernel_size):
    """
    Aplica un suavizado ligero sobre la zona reconstruida.

    Reduce artefactos visuales del inpainting y mejora la continuidad
    con el resto de la camiseta.

    Solo se aplica dentro de la región del logo.
    """
    if blur_kernel_size <= 1:
        return inpainted_frame.copy()

    if blur_kernel_size % 2 == 0:
        blur_kernel_size += 1

    mask = np.where(logo_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return inpainted_frame.copy()

    blurred_frame = cv2.GaussianBlur(
        inpainted_frame,
        (blur_kernel_size, blur_kernel_size),
        0,
    )
    smoothed_frame = inpainted_frame.copy()
    smoothed_frame[mask > 0] = blurred_frame[mask > 0]
    return smoothed_frame

def compose_reconstructed_frame(original_frame, inpainted_frame, logo_mask):
    """
    Combina el frame original con la región reconstruida.

    Sustituye únicamente los píxeles de la máscara del logo por los
    del resultado del inpainting, manteniendo intacto el resto del frame.
    """
    mask = np.where(logo_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask):
        return original_frame.copy()

    composed_frame = original_frame.copy()
    composed_frame[mask > 0] = inpainted_frame[mask > 0]
    return composed_frame

def background_reconstruction_pipeline(frame, logo_bbox, player_mask, reconstruction_bbox=None):
    """
    Pipeline completa de reconstrucción de fondo.

    1. Genera máscara del logo a partir del bounding box
    2. Expande la máscara para eliminar bordes
    3. Refina con segmentación del jugador (opcional)
    4. Aplica inpainting para reconstruir la camiseta
    5. Suaviza la región reconstruida
    6. Devuelve el frame limpio sin logo original

    Este resultado se usa como base para proyectar el nuevo logo.
    """
    logo_mask = generate_logo_mask_from_bbox(frame.shape, logo_bbox)
    expanded_mask = expand_logo_mask(logo_mask, expansion_pixels=4)
    refined_mask = refine_logo_mask_with_segmentation(expanded_mask, player_mask)
    logo_region_bbox = clip_bbox_to_frame(frame.shape, logo_bbox)
    player_region_bbox = mask_to_bbox(player_mask)
    reconstruction_region = merge_bboxes(
        clip_bbox_to_frame(frame.shape, reconstruction_bbox) if reconstruction_bbox is not None else player_region_bbox,
        logo_region_bbox,
    )
    if reconstruction_region is None:
        return frame.copy()

    inpainted_frame = apply_inpainting_on_crop(
        frame,
        refined_mask,
        reconstruction_region,
        inpaint_radius=3,
        method=cv2.INPAINT_TELEA,
    )
    smoothed_frame = smooth_reconstructed_region(
        inpainted_frame,
        refined_mask,
        blur_kernel_size=5,
    )
    reconstructed_frame = compose_reconstructed_frame(
        frame,
        smoothed_frame,
        refined_mask,
    )
    return reconstructed_frame


def background_reconstruction_lama_pipeline(
    frame,
    logo_bbox,
    player_mask,
    lama_model,
    reconstruction_bbox=None,
):
    """
    Pipeline de reconstruccion usando LaMa en lugar de OpenCV inpaint.

    Flujo:
    1. Genera mascara del sponsor detectado
    2. Expande ligeramente la mascara
    3. La recorta al jugador si hay segmentacion
    4. Ejecuta LaMa sobre un crop local
    5. Devuelve el frame reconstruido
    """
    logo_mask = generate_logo_mask_from_bbox(frame.shape, logo_bbox)
    expanded_mask = expand_logo_mask(logo_mask, expansion_pixels=4)
    refined_mask = refine_logo_mask_with_segmentation(expanded_mask, player_mask)

    logo_region_bbox = clip_bbox_to_frame(frame.shape, logo_bbox)
    player_region_bbox = mask_to_bbox(player_mask)
    reconstruction_region = merge_bboxes(
        clip_bbox_to_frame(frame.shape, reconstruction_bbox)
        if reconstruction_bbox is not None
        else player_region_bbox,
        logo_region_bbox,
    )
    if reconstruction_region is None:
        return frame.copy()

    reconstructed_frame = apply_lama_inpainting_on_crop(
        frame,
        refined_mask,
        reconstruction_region,
        lama_model,
    )
    reconstructed_frame = color_correct_reconstructed_region(
        frame,
        reconstructed_frame,
        refined_mask,
        player_mask=player_mask,
        ring_pixels=12,
    )
    return reconstructed_frame

