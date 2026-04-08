import cv2
import numpy as np


def create_feathered_logo_mask(binary_logo_mask, feather_kernel_size):
    """
    Crea una versión suavizada (feathered) de la máscara del logo.

    Convierte una máscara binaria (0/1) en una máscara con transición gradual
    en los bordes. Esto permite que el logo se mezcle suavemente con la camiseta,
    evitando bordes artificiales o recortados.

    feather_kernel_size controla cuánto se difuminan los bordes.
    """
    if binary_logo_mask is None or binary_logo_mask.size == 0:
        return np.array([], dtype=np.float32)

    if feather_kernel_size <= 1:
        return (binary_logo_mask > 0).astype(np.float32)

    if feather_kernel_size % 2 == 0:
        feather_kernel_size += 1

    base_mask = (np.asarray(binary_logo_mask) > 0).astype(np.float32)
    feathered_mask = cv2.GaussianBlur(
        base_mask,
        (feather_kernel_size, feather_kernel_size),
        0,
    )
    feathered_mask *= base_mask
    return np.clip(feathered_mask, 0.0, 1.0).astype(np.float32)

def blend_logo_with_background(background_region, shaded_logo, feathered_mask):
    """
    Mezcla el logo con el fondo usando alpha blending.

    Para cada píxel:
    resultado = logo * máscara + fondo * (1 - máscara)

    La máscara suavizada asegura una transición progresiva en los bordes,
    evitando que el logo parezca pegado encima.
    """
    if background_region is None or background_region.size == 0:
        return np.array([], dtype=np.uint8)

    if shaded_logo is None or shaded_logo.size == 0:
        return background_region.copy()

    if feathered_mask is None or feathered_mask.size == 0:
        return shaded_logo.copy()

    background = np.asarray(background_region, dtype=np.float32)
    logo = np.asarray(shaded_logo, dtype=np.float32)
    mask = np.asarray(feathered_mask, dtype=np.float32)

    if background.shape[:2] != logo.shape[:2]:
        raise ValueError("background_region y shaded_logo deben tener el mismo tamaño")

    if mask.shape[:2] != background.shape[:2]:
        mask = cv2.resize(
            mask,
            (background.shape[1], background.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    mask = np.clip(mask, 0.0, 1.0)
    mask = mask[:, :, np.newaxis]

    blended_region = (logo * mask) + (background * (1.0 - mask))
    return np.clip(blended_region, 0, 255).astype(np.uint8)

def match_logo_color_to_environment(shaded_logo, background_region, logo_mask):
    """
    Ajusta el color del logo para integrarlo mejor con la camiseta.

    Analiza el color medio del entorno (zona alrededor del logo)
    y corrige ligeramente:
    - brillo
    - saturación
    - tono

    Evita que el logo destaque de forma artificial respecto al tejido.
    """
    if shaded_logo is None or shaded_logo.size == 0:
        return np.array([], dtype=np.uint8)

    if background_region is None or background_region.size == 0:
        return shaded_logo.copy()

    if logo_mask is None or logo_mask.size == 0:
        return shaded_logo.copy()

    logo = np.asarray(shaded_logo, dtype=np.uint8)
    background = np.asarray(background_region, dtype=np.uint8)
    mask = (np.asarray(logo_mask) > 0).astype(np.uint8)

    if logo.shape[:2] != background.shape[:2]:
        raise ValueError("shaded_logo y background_region deben tener el mismo tamaño")

    if mask.shape[:2] != logo.shape[:2]:
        mask = cv2.resize(
            mask,
            (logo.shape[1], logo.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0).astype(np.uint8)

    if not np.any(mask):
        return logo.copy()

    logo_hsv = cv2.cvtColor(logo[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)
    background_hsv = cv2.cvtColor(background[:, :, :3], cv2.COLOR_BGR2HSV).astype(np.float32)

    ring_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    dilated_mask = cv2.dilate(mask, ring_kernel, iterations=1)
    env_mask = np.clip(dilated_mask - mask, 0, 1).astype(bool)

    if not np.any(env_mask):
        env_mask = mask.astype(bool)

    logo_pixels = mask.astype(bool)

    logo_mean_v = float(np.mean(logo_hsv[:, :, 2][logo_pixels]))
    env_mean_v = float(np.mean(background_hsv[:, :, 2][env_mask]))
    logo_mean_s = float(np.mean(logo_hsv[:, :, 1][logo_pixels]))
    env_mean_s = float(np.mean(background_hsv[:, :, 1][env_mask]))
    env_mean_h = float(np.mean(background_hsv[:, :, 0][env_mask]))

    adjusted_hsv = logo_hsv.copy()
    adjusted_hsv[:, :, 2][logo_pixels] = np.clip(
        adjusted_hsv[:, :, 2][logo_pixels] + ((env_mean_v - logo_mean_v) * 0.2),
        0,
        255,
    )
    adjusted_hsv[:, :, 1][logo_pixels] = np.clip(
        adjusted_hsv[:, :, 1][logo_pixels] + ((env_mean_s - logo_mean_s) * 0.15),
        0,
        255,
    )
    adjusted_hsv[:, :, 0][logo_pixels] = np.clip(
        (adjusted_hsv[:, :, 0][logo_pixels] * 0.9) + (env_mean_h * 0.1),
        0,
        179,
    )

    adjusted_logo = cv2.cvtColor(adjusted_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return adjusted_logo

def apply_global_alpha_adjustment(logo_image, alpha_value):
    """
    Reduce ligeramente la opacidad del logo.

    Un alpha_value típico (0.9–0.98) ayuda a que el logo se integre mejor
    con la textura de la camiseta, evitando un aspecto completamente plano.
    """
    if logo_image is None or logo_image.size == 0:
        return np.array([], dtype=np.uint8)

    logo = np.asarray(logo_image, dtype=np.float32)
    alpha_value = float(np.clip(alpha_value, 0.0, 1.0))

    if len(logo.shape) != 3 or logo.shape[2] != 4:
        return np.clip(logo, 0, 255).astype(np.uint8)

    adjusted_logo = logo.copy()
    adjusted_logo[:, :, 3] = np.clip(adjusted_logo[:, :, 3] * alpha_value, 0, 255)
    return adjusted_logo.astype(np.uint8)

def remove_edge_artifacts(refined_region, logo_mask):
    """
    Elimina pequeños artefactos visuales en los bordes del logo.

    Corrige píxeles inconsistentes generados por:
    - warping
    - oclusión
    - shading

    Mejora la continuidad visual entre logo y fondo.
    """
    if refined_region is None or refined_region.size == 0:
        return np.array([], dtype=np.uint8)

    if logo_mask is None or logo_mask.size == 0:
        return refined_region.copy()

    region = np.asarray(refined_region, dtype=np.uint8)
    mask = (np.asarray(logo_mask) > 0).astype(np.uint8)

    if mask.shape[:2] != region.shape[:2]:
        mask = cv2.resize(
            mask,
            (region.shape[1], region.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask > 0).astype(np.uint8)

    if not np.any(mask):
        return region.copy()

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    edge_mask = cv2.subtract(dilated_mask, eroded_mask)

    if not np.any(edge_mask):
        return region.copy()

    blurred_region = cv2.GaussianBlur(region, (3, 3), 0)
    cleaned_region = region.copy()
    cleaned_region[edge_mask > 0] = blurred_region[edge_mask > 0]
    return cleaned_region
    
def apply_micro_texture_noise(refined_region, intensity):
    """
    Añade una ligera variación de textura (ruido suave).

    Esto rompe la apariencia demasiado limpia o digital del logo,
    ayudando a integrarlo con la textura real de la camiseta.

    intensity controla la cantidad de ruido aplicado.
    """
    if refined_region is None or refined_region.size == 0:
        return np.array([], dtype=np.uint8)

    intensity = float(max(0.0, intensity))
    if intensity == 0.0:
        return refined_region.copy()

    region = np.asarray(refined_region, dtype=np.float32)
    noise_map = np.random.normal(
        loc=0.0,
        scale=intensity,
        size=region.shape[:2],
    ).astype(np.float32)
    noise = noise_map[:, :, np.newaxis]
    noisy_region = region + noise
    return np.clip(noisy_region, 0, 255).astype(np.uint8)

def refinement_pipeline(background_region, shaded_logo, binary_logo_mask):
    """
    Pipeline completa de refinamiento final.

    1. Genera máscara suavizada (feather)
    2. Ajusta color del logo al entorno
    3. Aplica ligera transparencia global
    4. Mezcla logo y fondo
    5. Elimina artefactos de bordes
    6. Añade micro-textura opcional

    Devuelve la región final lista para integrarse en el frame.
    """
    if background_region is None or background_region.size == 0:
        return np.array([], dtype=np.uint8)

    if shaded_logo is None or shaded_logo.size == 0:
        return background_region.copy()

    feathered_mask = create_feathered_logo_mask(
        binary_logo_mask,
        feather_kernel_size=7,
    )
    adjusted_logo = apply_global_alpha_adjustment(
        shaded_logo,
        alpha_value=1.0,
    )

    if len(adjusted_logo.shape) == 3 and adjusted_logo.shape[2] == 4:
        adjusted_logo_rgb = adjusted_logo[:, :, :3]
    else:
        adjusted_logo_rgb = adjusted_logo

    blended_region = blend_logo_with_background(
        background_region,
        adjusted_logo_rgb,
        feathered_mask,
    )
    cleaned_region = remove_edge_artifacts(
        blended_region,
        binary_logo_mask,
    )
    refined_region = apply_micro_texture_noise(
        cleaned_region,
        intensity=2.0,
    )
    return refined_region

