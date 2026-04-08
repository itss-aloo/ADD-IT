
import cv2
import numpy as np


def extract_luminance_from_frame_region(frame_region):
    """
    Extrae la luminancia de la región de la camiseta.

    Convierte la imagen a un espacio de color adecuado (HSV o LAB)
    y devuelve un mapa de intensidad que representa la iluminación
    real del tejido (sombras, luz, arrugas).
    """
    if frame_region is None or frame_region.size == 0:
        return np.array([], dtype=np.uint8)

    frame_region_uint8 = np.asarray(frame_region, dtype=np.uint8)
    lab_region = cv2.cvtColor(frame_region_uint8, cv2.COLOR_BGR2LAB)
    luminance_map = lab_region[:, :, 0]
    return luminance_map

def smooth_luminance_map(luminance_map):
    """
    Suaviza el mapa de luminancia mediante blur.

    Elimina detalles de alta frecuencia como bordes del logo original
    o patrones de la camiseta, conservando únicamente la iluminación
    general (sombras y gradientes suaves).
    """
    if luminance_map is None or luminance_map.size == 0:
        return np.array([], dtype=np.uint8)

    smoothed_map = cv2.GaussianBlur(luminance_map, (9, 9), 0)
    return smoothed_map

def normalize_luminance_map(luminance_map):
    """
    Normaliza la luminancia a un rango [0, 1].

    Esto permite usarla como factor multiplicativo sobre el logo,
    evitando saturaciones o valores extremos.
    """
    if luminance_map is None or luminance_map.size == 0:
        return np.array([], dtype=np.float32)

    luminance = np.asarray(luminance_map, dtype=np.float32)
    min_value = float(luminance.min())
    max_value = float(luminance.max())

    if max_value <= min_value:
        return np.ones_like(luminance, dtype=np.float32)

    normalized_map = (luminance - min_value) / (max_value - min_value)
    return normalized_map.astype(np.float32)

def apply_luminance_to_logo(logo_image, normalized_luminance, intensity_factor):
    """
    Aplica el mapa de luminancia al logo.

    Multiplica el logo por la luminancia para que herede la iluminación
    del pecho. El parámetro intensity_factor controla cuánto influye
    la iluminación (mezcla entre plano y realista).
    """
    if logo_image is None or logo_image.size == 0:
        return np.array([], dtype=np.uint8)

    if normalized_luminance is None or normalized_luminance.size == 0:
        return logo_image.copy()

    logo = np.asarray(logo_image, dtype=np.float32)
    luminance = np.asarray(normalized_luminance, dtype=np.float32)
    intensity_factor = float(np.clip(intensity_factor, 0.0, 1.0))

    if logo.shape[:2] != luminance.shape[:2]:
        luminance = cv2.resize(
            luminance,
            (logo.shape[1], logo.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    centered_luminance = 0.7 + (0.6 * luminance)
    shading_factor = (1.0 - intensity_factor) + (centered_luminance * intensity_factor)
    shading_factor = shading_factor[:, :, np.newaxis]

    if logo.shape[2] == 4:
        logo_rgb = logo[:, :, :3].astype(np.uint8)
        shaded_alpha = logo[:, :, 3:4]
    else:
        logo_rgb = logo.astype(np.uint8)
        shaded_alpha = None

    logo_hsv = cv2.cvtColor(logo_rgb, cv2.COLOR_BGR2HSV).astype(np.float32)
    logo_hsv[:, :, 2] = np.clip(logo_hsv[:, :, 2] * shading_factor[:, :, 0], 0, 255)
    shaded_rgb = cv2.cvtColor(logo_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

    if shaded_alpha is not None:
        shaded_logo = np.concatenate((shaded_rgb, shaded_alpha), axis=2)
    else:
        shaded_logo = shaded_rgb

    return np.clip(shaded_logo, 0, 255).astype(np.uint8)

def adjust_logo_contrast(logo_image, alpha, beta):
    """
    Ajusta el contraste y brillo del logo.

    Permite recuperar visibilidad tras aplicar shading,
    controlando intensidad (alpha) y desplazamiento (beta).
    """
    if logo_image is None or logo_image.size == 0:
        return np.array([], dtype=np.uint8)

    logo = np.asarray(logo_image, dtype=np.float32)
    alpha = float(alpha)
    beta = float(beta)

    if logo.shape[2] == 4:
        adjusted_rgb = (logo[:, :, :3] * alpha) + beta
        adjusted_alpha = logo[:, :, 3:4]
        adjusted_logo = np.concatenate((adjusted_rgb, adjusted_alpha), axis=2)
    else:
        adjusted_logo = (logo * alpha) + beta

    return np.clip(adjusted_logo, 0, 255).astype(np.uint8)

def compose_shaded_logo(frame_region, projected_logo, intensity_factor):
    """
    Pipeline completa de shading.

    1. Extrae luminancia del frame original
    2. Suaviza para eliminar detalles del logo antiguo
    3. Normaliza el mapa de iluminación
    4. Aplica la luminancia al logo proyectado

    Devuelve el logo con shading listo para composición final.
    """
    if frame_region is None or frame_region.size == 0:
        return projected_logo.copy()

    if projected_logo is None or projected_logo.size == 0:
        return np.array([], dtype=np.uint8)

    luminance_map = extract_luminance_from_frame_region(frame_region)
    smoothed_luminance = smooth_luminance_map(luminance_map)
    normalized_luminance = normalize_luminance_map(smoothed_luminance)
    shaded_logo = apply_luminance_to_logo(
        projected_logo,
        normalized_luminance,
        intensity_factor,
    )
    return shaded_logo





