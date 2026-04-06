import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    Carga una imagen base desde disco en formato BGR.

    Entrada:
        image_path (str): Ruta de la imagen.

    Salida:
        image (np.ndarray): Imagen cargada (H, W, 3).
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    return image

def load_logo(logo_path: str) -> np.ndarray:
    """
    Carga un logo con canal alpha (RGBA).

    Entrada:
        logo_path (str): Ruta del logo PNG.

    Salida:
        logo (np.ndarray): Imagen RGBA (H, W, 4).
    """
    logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise FileNotFoundError(f"No se pudo cargar el logo: {logo_path}")
    if len(logo.shape) != 3 or logo.shape[2] != 4:
        raise ValueError(f"El logo debe tener 4 canales (RGBA): {logo_path}")
    return logo

def prepare_logo_patch(logo: np.ndarray, background_color: tuple) -> np.ndarray:
    """
    Convierte un logo con transparencia en un patch opaco con fondo de color.

    Entrada:
        logo (np.ndarray): Imagen RGBA (H, W, 4).
        background_color (tuple): Color de fondo (B, G, R).

    Salida:
        patch (np.ndarray): Imagen RGBA opaca (H, W, 4).
    """
    alpha = logo[:, :, 3].astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=1.5, sigmaY=1.5)
    alpha = alpha[:, :, np.newaxis]

    logo_rgb = logo[:, :, :3].astype(np.float32)
    background = np.full(logo.shape[:2] + (3,), background_color, dtype=np.float32)

    patch_rgb = alpha * logo_rgb + (1.0 - alpha) * background
    patch_alpha = np.full(logo.shape[:2] + (1,), 255, dtype=np.uint8)

    patch = np.concatenate((patch_rgb.astype(np.uint8), patch_alpha), axis=2)
    return patch

def compute_perspective_transform(logo: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz de transformación perspectiva del logo al cuadrilátero destino.

    Entrada:
        logo (np.ndarray): Imagen del logo (H, W, 4).
        dst_points (np.ndarray): Array de puntos destino (4,2).

    Salida:
        matrix (np.ndarray): Matriz de transformación 3x3.
    """
    height, width = logo.shape[:2]
    src_points = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )
    dst_points = np.asarray(dst_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix

def warp_logo(logo_patch: np.ndarray, matrix: np.ndarray, output_shape: tuple) -> np.ndarray:
    """
    Aplica transformación perspectiva al logo.

    Entrada:
        logo_patch (np.ndarray): Patch RGBA del logo.
        matrix (np.ndarray): Matriz de transformación 3x3.
        output_shape (tuple): Dimensiones de la imagen base (height, width).

    Salida:
        warped_logo (np.ndarray): Logo deformado (H, W, 4).
    """
    height, width = output_shape
    warped_logo = cv2.warpPerspective(logo_patch, matrix, (width, height))
    return warped_logo

def alpha_blend(base_image: np.ndarray, overlay: np.ndarray) -> np.ndarray:
    """
    Mezcla el logo deformado sobre la imagen base usando canal alpha.

    Entrada:
        base_image (np.ndarray): Imagen base (H, W, 3).
        overlay (np.ndarray): Logo RGBA deformado (H, W, 4).

    Salida:
        result (np.ndarray): Imagen final (H, W, 3).
    """
    alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    overlay_rgb = overlay[:, :, :3].astype(np.float32)
    base_rgb = base_image.astype(np.float32)

    result = alpha * overlay_rgb + (1.0 - alpha) * base_rgb
    return result.astype(np.uint8)

def render_logo_on_image(image_path: str, logo_path: str, dst_points: np.ndarray) -> np.ndarray:
    """
    Pipeline completo para proyectar un logo sobre una imagen usando transformación perspectiva.

    Entrada:
        image_path (str): Ruta de la imagen base.
        logo_path (str): Ruta del logo.
        dst_points (np.ndarray): Puntos destino (4,2).

    Salida:
        result (np.ndarray): Imagen final renderizada.
    """
    image = load_image(image_path)
    logo = load_logo(logo_path)
    logo_patch = prepare_logo_patch(logo, (255, 255, 255))
    matrix = compute_perspective_transform(logo, dst_points)
    warped_logo = warp_logo(logo_patch, matrix, image.shape[:2])
    result = alpha_blend(image, warped_logo)
    return result
