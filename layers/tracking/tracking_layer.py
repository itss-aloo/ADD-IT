import numpy as np


def initialize_tracking_state(detection_bbox, torso_geometry):
    """
    Inicializa el estado de tracking para un jugador.

    Guarda:
    - posición del bbox
    - tamaño (ancho, alto)
    - orientación del torso

    Este estado se usará como referencia para los siguientes frames.
    """
    return {
        "bbox": tuple(detection_bbox) if detection_bbox is not None else None,
        "torso_geometry": torso_geometry.copy() if isinstance(torso_geometry, dict) else torso_geometry,
        "predicted": False,
        "confidence": 1.0,
        "lost_frames": 0,
    }

def update_tracking_state(previous_state, current_detection_bbox, current_torso_geometry):
    """
    Actualiza el estado del jugador combinando:

    - detección actual (frame t)
    - estado previo (frame t-1)

    Permite mantener continuidad aunque la detección fluctúe ligeramente.
    """
    if previous_state is None:
        return initialize_tracking_state(current_detection_bbox, current_torso_geometry)

    previous_bbox = previous_state.get("bbox")
    previous_geometry = previous_state.get("torso_geometry")
    updated_bbox = smooth_bbox(previous_bbox, current_detection_bbox, smoothing_factor=0.65)
    updated_geometry = smooth_torso_geometry(
        previous_geometry,
        current_torso_geometry,
        smoothing_factor=0.65,
    )

    updated_state = previous_state.copy()
    updated_state["bbox"] = updated_bbox
    updated_state["torso_geometry"] = updated_geometry
    updated_state["predicted"] = False
    updated_state["lost_frames"] = 0
    return updated_state

def smooth_bbox(previous_bbox, current_bbox, smoothing_factor):
    """
    Suaviza el bounding box para evitar saltos bruscos.

    Mezcla el bbox actual con el anterior:
    bbox_final = alpha * actual + (1 - alpha) * anterior

    smoothing_factor (alpha) controla la estabilidad vs. reactividad.
    """
    if current_bbox is None:
        return tuple(previous_bbox) if previous_bbox is not None else None
    if previous_bbox is None:
        return tuple(current_bbox)

    alpha = float(np.clip(smoothing_factor, 0.0, 1.0))
    previous = np.asarray(previous_bbox, dtype=np.float32)
    current = np.asarray(current_bbox, dtype=np.float32)
    smoothed = (alpha * current) + ((1.0 - alpha) * previous)
    return tuple(int(round(value)) for value in smoothed.tolist())

def smooth_torso_geometry(previous_geometry, current_geometry, smoothing_factor):
    """
    Suaviza la geometría del torso (rotación, escala).

    Evita cambios bruscos en la orientación del plano del pecho,
    lo que estabiliza la proyección del logo.
    """
    if current_geometry is None or current_geometry == {}:
        return previous_geometry.copy() if isinstance(previous_geometry, dict) else previous_geometry
    if previous_geometry is None or previous_geometry == {}:
        return current_geometry.copy() if isinstance(current_geometry, dict) else current_geometry

    alpha = float(np.clip(smoothing_factor, 0.0, 1.0))
    smoothed_geometry = current_geometry.copy()

    for key in ("center",):
        if key in previous_geometry and key in current_geometry:
            previous_value = np.asarray(previous_geometry[key], dtype=np.float32)
            current_value = np.asarray(current_geometry[key], dtype=np.float32)
            smoothed_value = (alpha * current_value) + ((1.0 - alpha) * previous_value)
            smoothed_geometry[key] = tuple(smoothed_value.tolist())

    for key in ("width", "height"):
        if key in previous_geometry and key in current_geometry:
            previous_value = float(previous_geometry[key])
            current_value = float(current_geometry[key])
            smoothed_geometry[key] = (alpha * current_value) + ((1.0 - alpha) * previous_value)

    if "corners" in previous_geometry and "corners" in current_geometry:
        previous_corners = np.asarray(previous_geometry["corners"], dtype=np.float32)
        current_corners = np.asarray(current_geometry["corners"], dtype=np.float32)
        if previous_corners.shape == current_corners.shape:
            smoothed_corners = (alpha * current_corners) + ((1.0 - alpha) * previous_corners)
            smoothed_geometry["corners"] = [tuple(point.tolist()) for point in smoothed_corners]

    if "mode" in current_geometry:
        smoothed_geometry["mode"] = current_geometry["mode"]

    return smoothed_geometry

def predict_next_state(previous_state):
    """
    Predice el estado del jugador cuando la detección falla.

    Usa el estado anterior para mantener continuidad temporal,
    evitando que el logo desaparezca o salte.
    """
    if previous_state is None:
        return None

    predicted_state = previous_state.copy()
    predicted_state["predicted"] = True
    predicted_state["lost_frames"] = int(previous_state.get("lost_frames", 0)) + 1
    return predicted_state

def handle_detection_confidence(detection_confidence, threshold, previous_state, current_detection):
    """
    Evalúa si la detección actual es fiable.

    - Si confidence >= threshold → usar detección
    - Si no → usar estado previo (o predicción)

    Evita errores por detecciones parciales o inestables.
    """
    confidence = 0.0 if detection_confidence is None else float(detection_confidence)
    threshold = float(threshold)

    if current_detection is not None and confidence >= threshold:
        return current_detection

    if previous_state is None:
        return current_detection

    return predict_next_state(previous_state)

def stabilize_luminance(previous_luminance, current_luminance, smoothing_factor):
    """
    Suaviza cambios de luminancia entre frames.

    Evita que el shading del logo fluctúe de forma brusca,
    manteniendo una iluminación coherente en el tiempo.
    """
    if current_luminance is None or getattr(current_luminance, "size", 0) == 0:
        return previous_luminance.copy() if previous_luminance is not None else current_luminance
    if previous_luminance is None or getattr(previous_luminance, "size", 0) == 0:
        return current_luminance.copy()

    alpha = float(np.clip(smoothing_factor, 0.0, 1.0))
    previous = np.asarray(previous_luminance, dtype=np.float32)
    current = np.asarray(current_luminance, dtype=np.float32)

    if previous.shape != current.shape:
        previous = np.resize(previous, current.shape)

    stabilized = (alpha * current) + ((1.0 - alpha) * previous)
    return stabilized.astype(np.float32)

def temporal_consistency_pipeline(previous_state, detection_bbox, torso_geometry, detection_confidence):
    """
    Pipeline completa de consistencia temporal.

    1. Evalúa la fiabilidad de la detección
    2. Decide usar detección o estado previo
    3. Actualiza el estado del jugador
    4. Aplica suavizado a bbox y geometría
    5. Predice estado si hay fallo de detección

    Devuelve el estado estabilizado para el frame actual.
    """
    detection_threshold = 0.5
    selected_state = handle_detection_confidence(
        detection_confidence,
        detection_threshold,
        previous_state,
        {
            "bbox": detection_bbox,
            "torso_geometry": torso_geometry,
            "confidence": detection_confidence,
        } if detection_bbox is not None else None,
    )

    if selected_state is None:
        return None

    if selected_state.get("predicted"):
        return selected_state

    if previous_state is None:
        initialized_state = initialize_tracking_state(
            selected_state.get("bbox"),
            selected_state.get("torso_geometry"),
        )
        initialized_state["confidence"] = float(detection_confidence or 0.0)
        return initialized_state

    updated_state = update_tracking_state(
        previous_state,
        selected_state.get("bbox"),
        selected_state.get("torso_geometry"),
    )
    updated_state["confidence"] = float(detection_confidence or 0.0)
    return updated_state







