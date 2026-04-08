from layers.tracking.tracking_layer import (
    initialize_tracking_state,
    temporal_consistency_pipeline,
)


def bbox_center(bbox):
    if bbox is None:
        return None

    x, y, width, height = bbox
    return (x + (width / 2.0), y + (height / 2.0))


def bbox_distance(first_bbox, second_bbox):
    first_center = bbox_center(first_bbox)
    second_center = bbox_center(second_bbox)
    if first_center is None or second_center is None:
        return float("inf")

    dx = first_center[0] - second_center[0]
    dy = first_center[1] - second_center[1]
    return (dx * dx + dy * dy) ** 0.5


def initialize_global_state():
    """
    Inicializa el estado global del sistema.

    Contiene:
    - frame_index = 0
    - diccionario de jugadores activos (player_id → estado)
    - configuración del sistema (flags, modos)
    
    Se ejecuta una sola vez al iniciar el procesamiento de vídeo.
    """
    return {
        "frame_index": 0,
        "players": {},
        "next_player_id": 1,
        "config": {
            "confidence_threshold": 0.5,
            "max_missing_frames": 5,
            "matching_distance_threshold": 150.0,
        },
    }

def increment_frame_index(global_state):
    """
    Incrementa el índice de frame.

    Permite llevar control temporal del vídeo y gestionar caducidad
    de estados de jugadores.
    """
    global_state["frame_index"] = int(global_state.get("frame_index", 0)) + 1
    return global_state["frame_index"]

def assign_player_ids(detections, global_state):
    """
    Asigna un player_id a cada detección en el frame actual.

    Compara las detecciones actuales con los jugadores ya existentes
    (por proximidad espacial) y mantiene IDs consistentes entre frames.

    Si una detección no coincide con ningún jugador previo,
    se crea un nuevo player_id.
    """
    active_players = global_state.get("players", {})
    next_player_id = int(global_state.get("next_player_id", 1))
    distance_threshold = float(
        global_state.get("config", {}).get("matching_distance_threshold", 150.0)
    )

    assignments = []
    used_player_ids = set()

    for detection in detections:
        detection_bbox = detection.get("bbox")
        best_player_id = None
        best_distance = float("inf")

        for player_id, player_state in active_players.items():
            if player_id in used_player_ids:
                continue

            player_bbox = player_state.get("bbox")
            distance = bbox_distance(detection_bbox, player_bbox)
            if distance < best_distance and distance <= distance_threshold:
                best_distance = distance
                best_player_id = player_id

        if best_player_id is None:
            best_player_id = next_player_id
            next_player_id += 1

        used_player_ids.add(best_player_id)
        assignments.append(
            {
                "player_id": best_player_id,
                "detection": detection,
            }
        )

    global_state["next_player_id"] = next_player_id
    return assignments

def initialize_player_state(player_id, detection_bbox, torso_geometry, frame_index):
    """
    Inicializa el estado de un jugador nuevo.

    Guarda:
    - bbox inicial
    - geometría del torso
    - último frame donde fue visto
    - estado de tracking
    """
    return {
        "player_id": player_id,
        "bbox": tuple(detection_bbox) if detection_bbox is not None else None,
        "torso_geometry": torso_geometry.copy() if isinstance(torso_geometry, dict) else torso_geometry,
        "last_seen_frame": frame_index,
        "tracking_state": initialize_tracking_state(detection_bbox, torso_geometry),
        "last_valid_logo_bbox": None,
        "missing_frames": 0,
    }

def update_player_state(player_state, detection_bbox, torso_geometry, frame_index):
    """
    Actualiza el estado de un jugador existente.

    - actualiza bbox
    - actualiza geometría del torso
    - actualiza último frame visto

    Mantiene coherencia temporal del jugador.
    """
    confidence = player_state.get("detection_confidence", 1.0)
    tracking_state = temporal_consistency_pipeline(
        player_state.get("tracking_state"),
        detection_bbox,
        torso_geometry,
        confidence,
    )

    updated_state = player_state.copy()
    updated_state["bbox"] = tracking_state.get("bbox") if tracking_state is not None else detection_bbox
    updated_state["torso_geometry"] = (
        tracking_state.get("torso_geometry")
        if tracking_state is not None
        else torso_geometry
    )
    updated_state["last_seen_frame"] = frame_index
    updated_state["tracking_state"] = tracking_state
    updated_state["missing_frames"] = 0
    return updated_state

def handle_missing_players(global_state, max_missing_frames):
    """
    Gestiona jugadores que no han sido detectados.

    - si desaparecen pocos frames → mantener estado (posible oclusión)
    - si superan max_missing_frames → eliminar jugador

    Evita perder jugadores por fallos puntuales de detección.
    """
    frame_index = int(global_state.get("frame_index", 0))
    players = global_state.get("players", {})
    player_ids_to_remove = []

    for player_id, player_state in players.items():
        last_seen_frame = int(player_state.get("last_seen_frame", frame_index))
        missing_frames = frame_index - last_seen_frame
        player_state["missing_frames"] = missing_frames
        if missing_frames > max_missing_frames:
            player_ids_to_remove.append(player_id)

    for player_id in player_ids_to_remove:
        players.pop(player_id, None)

    return global_state

def select_valid_detection_or_fallback(detection_bbox, detection_confidence, player_state, confidence_threshold):
    """
    Decide si usar la detección actual o fallback.

    - si la detección es fiable → usarla
    - si no → usar bbox previo del estado

    Evita usar detecciones parciales o erróneas.
    """
    if detection_bbox is not None and float(detection_confidence or 0.0) >= float(confidence_threshold):
        return detection_bbox, True

    if player_state is None:
        return detection_bbox, False

    return player_state.get("bbox"), False

def select_logo_region(current_logo_bbox, player_state):
    """
    Selecciona la región del logo a usar.

    - si el bbox actual es válido → usarlo
    - si no → usar el último bbox válido guardado

    Evita errores cuando el logo está parcialmente ocluido.
    """
    if current_logo_bbox is not None:
        return current_logo_bbox

    if player_state is None:
        return None

    return player_state.get("last_valid_logo_bbox")

def update_last_valid_logo(player_state, logo_bbox, is_valid):
    """
    Actualiza el último bbox válido del logo.

    Solo se guarda cuando la detección es fiable.
    Permite reutilizarlo en frames con oclusión.
    """
    if player_state is None:
        return None

    if is_valid and logo_bbox is not None:
        player_state["last_valid_logo_bbox"] = tuple(logo_bbox)

    return player_state

def prepare_player_pipeline_inputs(player_state):
    """
    Prepara inputs necesarios para el pipeline de render.

    Devuelve:
    - bbox estabilizado
    - geometría del torso
    - región del logo

    Se usa como entrada para las siguientes layers.
    """
    if player_state is None:
        return {}

    return {
        "player_id": player_state.get("player_id"),
        "bbox": player_state.get("bbox"),
        "torso_geometry": player_state.get("torso_geometry"),
        "logo_region": player_state.get("last_valid_logo_bbox"),
        "tracking_state": player_state.get("tracking_state"),
    }


def control_state_pipeline(global_state, detections, torso_geometries, logo_detections):
    """
    Pipeline principal de control y estado por frame.

    1. Incrementa frame_index
    2. Asigna IDs a jugadores
    3. Inicializa nuevos jugadores
    4. Actualiza estados existentes
    5. Gestiona jugadores desaparecidos
    6. Selecciona detecciones válidas o fallback
    7. Prepara inputs para el pipeline visual

    Devuelve el estado actualizado y los inputs listos para las siguientes capas.
    """
    increment_frame_index(global_state)
    frame_index = global_state["frame_index"]
    players = global_state["players"]
    confidence_threshold = float(global_state["config"].get("confidence_threshold", 0.5))
    max_missing_frames = int(global_state["config"].get("max_missing_frames", 5))

    enriched_detections = []
    for index, detection in enumerate(detections):
        enriched_detection = detection.copy()
        enriched_detection["torso_geometry"] = (
            torso_geometries[index] if index < len(torso_geometries) else None
        )
        enriched_detection["logo_bbox"] = (
            logo_detections[index] if index < len(logo_detections) else None
        )
        enriched_detections.append(enriched_detection)

    assignments = assign_player_ids(enriched_detections, global_state)
    seen_player_ids = set()

    for assignment in assignments:
        player_id = assignment["player_id"]
        detection = assignment["detection"]
        detection_bbox = detection.get("bbox")
        torso_geometry = detection.get("torso_geometry")
        detection_confidence = float(detection.get("confidence", 0.0))
        logo_bbox = detection.get("logo_bbox")

        previous_state = players.get(player_id)
        selected_bbox, is_valid_detection = select_valid_detection_or_fallback(
            detection_bbox,
            detection_confidence,
            previous_state,
            confidence_threshold,
        )

        if previous_state is None:
            player_state = initialize_player_state(
                player_id,
                selected_bbox,
                torso_geometry,
                frame_index,
            )
        else:
            previous_state["detection_confidence"] = detection_confidence
            player_state = update_player_state(
                previous_state,
                selected_bbox,
                torso_geometry,
                frame_index,
            )

        player_state["player_id"] = player_id
        player_state["detection_confidence"] = detection_confidence
        selected_logo_bbox = select_logo_region(logo_bbox, previous_state)
        player_state = update_last_valid_logo(
            player_state,
            selected_logo_bbox,
            is_valid_detection and selected_logo_bbox is not None,
        )
        players[player_id] = player_state
        seen_player_ids.add(player_id)

    handle_missing_players(global_state, max_missing_frames)

    pipeline_inputs = []
    for player_id, player_state in players.items():
        if player_id not in seen_player_ids and player_state.get("missing_frames", 0) > max_missing_frames:
            continue
        pipeline_inputs.append(prepare_player_pipeline_inputs(player_state))

    return global_state, pipeline_inputs
