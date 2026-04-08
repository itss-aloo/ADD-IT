import argparse
from pathlib import Path
import time

import cv2
import numpy as np

from layers.background_reconstruction.background_reconstruction_layer import (
    background_reconstruction_lama_pipeline,
    load_lama_model,
)
from layers.control.control_layer import control_state_pipeline, initialize_global_state
from layers.detection.detection_layer import (
    convert_bbox_format,
    detect_and_filter_players,
    load_detection_model,
)
from layers.geometry.geometry_sponsor_layer import get_sponsor_quad, match_sponsor_to_player
from layers.occlusion.occlusion_layer import (
    combine_masks,
    load_segmentation_model,
    match_mask_to_player,
    segment_players,
)
from layers.pose.pose_layer import estimate_poses, load_pose_model
from layers.refinement.refinement_layer import refinement_pipeline
from layers.rendering.rendering_layer import (
    compute_perspective_transform,
    load_logo,
    prepare_logo_patch,
    warp_logo,
)
from layers.shading.shading_layer import compose_shaded_logo
from layers.video_io.video_io_layer import (
    display_frame,
    initialize_video_writer,
    open_video,
    read_next_frame,
    release_video_resources,
    write_frame,
)


def detect_sponsors(model, image, min_confidence=0.25):
    results = model(image, imgsz=960, conf=min_confidence, verbose=False)
    sponsors = []

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf.item())
            if confidence < min_confidence:
                continue

            sponsors.append(
                {
                    "bbox": convert_bbox_format(box.xyxy[0].tolist()),
                    "confidence": confidence,
                }
            )

    return sponsors


def quad_width_height(quad):
    points = np.array(quad, dtype=np.float32)
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    left_height = np.linalg.norm(points[3] - points[0])
    right_height = np.linalg.norm(points[2] - points[1])
    return (top_width + bottom_width) / 2.0, (left_height + right_height) / 2.0


def find_logo_files(logo_dir):
    return sorted(
        path
        for path in Path(logo_dir).iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def get_fixed_logo_path(logo_dir, logo_filename="05_logo.png"):
    logo_path = Path(logo_dir) / logo_filename
    if not logo_path.exists():
        raise FileNotFoundError(f"No existe el logo fijo solicitado: {logo_path}")
    return logo_path


def choose_logo_for_quad(logo_paths, quad):
    quad_width, quad_height = quad_width_height(quad)
    if quad_height <= 0 or not logo_paths:
        return None

    target_ratio = quad_width / quad_height
    best_logo_path = None
    best_score = float("inf")

    for logo_path in logo_paths:
        logo = cv2.imread(str(logo_path), cv2.IMREAD_UNCHANGED)
        if logo is None:
            continue

        logo_height, logo_width = logo.shape[:2]
        if logo_height <= 0:
            continue

        logo_ratio = logo_width / logo_height
        score = abs(np.log(max(logo_ratio, 1e-6) / max(target_ratio, 1e-6)))
        if score < best_score:
            best_score = score
            best_logo_path = logo_path

    return best_logo_path


def estimate_jersey_background_color(image, bbox, margin=10):
    x, y, width, height = bbox
    image_height, image_width = image.shape[:2]

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(image_width, x + width + margin)
    y2 = min(image_height, y + height + margin)
    if x2 <= x1 or y2 <= y1:
        return (255, 255, 255)

    patch = image[y1:y2, x1:x2]
    ring_mask = np.ones(patch.shape[:2], dtype=bool)
    inner_x1 = max(0, x - x1)
    inner_y1 = max(0, y - y1)
    inner_x2 = min(patch.shape[1], x + width - x1)
    inner_y2 = min(patch.shape[0], y + height - y1)
    ring_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

    samples = patch[ring_mask]
    if len(samples) == 0:
        samples = patch.reshape(-1, 3)

    color = np.median(samples, axis=0)
    return tuple(int(channel) for channel in color)


def render_logo_overlay_on_quad(image, logo_path, quad, background_color, use_background_patch=True):
    logo = load_logo(str(logo_path))
    if use_background_patch:
        logo_overlay = prepare_logo_patch(logo, background_color)
    else:
        logo_overlay = logo.copy()

    matrix = compute_perspective_transform(logo_overlay, np.array(quad, dtype=np.float32))
    return warp_logo(logo_overlay, matrix, image.shape[:2])


def render_logo_ink_mask_on_quad(image, logo_path, quad):
    logo = load_logo(str(logo_path))
    logo_alpha = logo[:, :, 3]
    ink_overlay = np.zeros(logo.shape[:2] + (4,), dtype=np.uint8)
    ink_overlay[:, :, 3] = logo_alpha
    matrix = compute_perspective_transform(ink_overlay, np.array(quad, dtype=np.float32))
    warped_ink = warp_logo(ink_overlay, matrix, image.shape[:2])
    return warped_ink[:, :, 3]


def apply_masks_to_overlay_alpha(overlay, player_mask, occluder_mask=None):
    masked_overlay = overlay.copy()
    alpha = masked_overlay[:, :, 3].astype(np.float32)
    alpha *= (player_mask > 0).astype(np.float32)

    if occluder_mask is not None and occluder_mask.size != 0:
        alpha *= 1.0 - (occluder_mask > 0).astype(np.float32)

    visible_pixels = alpha > 0
    alpha[visible_pixels] = 255.0
    masked_overlay[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return masked_overlay


def apply_shading_to_ink_only(frame_region, logo_overlay, ink_mask, intensity_factor):
    shaded_overlay = logo_overlay.copy()
    fully_shaded = compose_shaded_logo(frame_region, logo_overlay, intensity_factor)

    ink_region = ink_mask > 32
    if not np.any(ink_region):
        return shaded_overlay

    shaded_overlay[ink_region, :3] = fully_shaded[ink_region, :3]
    shaded_overlay[:, :, 3] = logo_overlay[:, :, 3]
    return shaded_overlay


def resize_frame_for_processing(frame, max_side=1600):
    height, width = frame.shape[:2]
    largest_side = max(height, width)
    if max_side <= 0 or largest_side <= max_side:
        return frame, 1.0

    scale = float(max_side) / float(largest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_frame = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    return resized_frame, scale


def torso_to_bbox(torso):
    corners = torso.get("corners", []) if isinstance(torso, dict) else []
    if not corners:
        return None

    xs = [point[0] for point in corners]
    ys = [point[1] for point in corners]
    x1 = int(round(min(xs)))
    y1 = int(round(min(ys)))
    x2 = int(round(max(xs)))
    y2 = int(round(max(ys)))
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2 - x1, y2 - y1)


def reconstruct_sponsors_background_with_lama(image, sponsors, players_pose, segmentation_model, lama_model):
    reconstructed_image = image.copy()
    segments = segment_players(segmentation_model, image)
    player_masks = [
        match_mask_to_player(segments, player_pose["bbox"])
        for player_pose in players_pose
    ]

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        player_index = next(
            (index for index, player_pose in enumerate(players_pose) if player_pose is matched_player),
            None,
        )
        player_mask = None
        if player_index is not None and player_masks[player_index].size != 0:
            player_mask = player_masks[player_index]
        reconstruction_bbox = torso_to_bbox(matched_player.get("torso", {}))
        reconstructed_image = background_reconstruction_lama_pipeline(
            reconstructed_image,
            sponsor["bbox"],
            player_mask,
            lama_model=lama_model,
            reconstruction_bbox=reconstruction_bbox,
        )

    return reconstructed_image


def build_logo_detections_for_players(players_pose, sponsors):
    logo_detections = [None] * len(players_pose)
    if not players_pose:
        return logo_detections

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        for index, player_pose in enumerate(players_pose):
            if player_pose is matched_player or player_pose.get("bbox") == matched_player.get("bbox"):
                current_logo = logo_detections[index]
                if current_logo is None or sponsor.get("confidence", 0.0) > current_logo.get("confidence", 0.0):
                    logo_detections[index] = sponsor
                break

    return [logo["bbox"] if logo is not None else None for logo in logo_detections]


def find_best_current_player(player_input, players_pose):
    if not players_pose:
        return None, None

    input_bbox = player_input.get("bbox")
    if input_bbox is None:
        return None, None

    best_index = None
    best_distance = float("inf")
    for index, player_pose in enumerate(players_pose):
        player_bbox = player_pose.get("bbox")
        if player_bbox is None:
            continue
        distance = ((player_bbox[0] - input_bbox[0]) ** 2 + (player_bbox[1] - input_bbox[1]) ** 2) ** 0.5
        if distance < best_distance:
            best_distance = distance
            best_index = index

    if best_index is None:
        return None, None
    return best_index, players_pose[best_index]


def render_frame_pipeline(
    frame,
    detection_model,
    pose_model,
    sponsor_model,
    segmentation_model,
    logo_paths,
    global_state,
    use_lama_reconstruction=False,
    lama_model=None,
    shading_intensity=0.85,
):
    detections = detect_and_filter_players(
        detection_model,
        frame,
        min_confidence=0.56,
        min_area=50000,
    )
    sponsors = detect_sponsors(sponsor_model, frame, min_confidence=0.25) if sponsor_model else []
    has_tracked_logo = any(
        player_state.get("last_valid_logo_bbox") is not None
        for player_state in global_state.get("players", {}).values()
    )
    if not sponsors and not has_tracked_logo:
        control_state_pipeline(global_state, [], [], [])
        return frame.copy(), global_state

    players_pose = estimate_poses(pose_model, frame, detections) if detections else []

    torso_geometries = [player_pose.get("torso") for player_pose in players_pose]
    logo_detections = build_logo_detections_for_players(players_pose, sponsors)
    global_state, player_inputs = control_state_pipeline(
        global_state,
        detections,
        torso_geometries,
        logo_detections,
    )

    working_frame = frame.copy()
    if use_lama_reconstruction and sponsors and lama_model is not None:
        working_frame = reconstruct_sponsors_background_with_lama(
            frame,
            sponsors,
            players_pose,
            segmentation_model,
            lama_model,
        )

    segments = segment_players(segmentation_model, frame) if players_pose else []
    player_masks = [
        match_mask_to_player(segments, player_pose["bbox"])
        for player_pose in players_pose
    ]
    output_frame = working_frame.copy()

    for player_input in player_inputs:
        logo_region = player_input.get("logo_region")
        if logo_region is None:
            continue

        player_index, current_player = find_best_current_player(player_input, players_pose)
        if current_player is None:
            continue

        keypoints = current_player.get("keypoints", {})
        quad = get_sponsor_quad(logo_region, keypoints)
        logo_path = choose_logo_for_quad(logo_paths, quad)
        if logo_path is None:
            continue

        player_mask = player_masks[player_index]
        if player_mask is None or player_mask.size == 0:
            continue

        background_color = estimate_jersey_background_color(working_frame, logo_region)
        warped_logo = render_logo_overlay_on_quad(
            working_frame,
            logo_path,
            quad,
            background_color,
            use_background_patch=not use_lama_reconstruction,
        )
        warped_ink_mask = render_logo_ink_mask_on_quad(
            working_frame,
            logo_path,
            quad,
        )

        other_player_masks = [
            mask for index, mask in enumerate(player_masks)
            if index != player_index and mask.size != 0
        ]
        other_players_mask = combine_masks(other_player_masks) if other_player_masks else None

        visible_logo = apply_masks_to_overlay_alpha(
            warped_logo,
            player_mask,
            other_players_mask,
        )
        visible_ink_mask = warped_ink_mask.copy()
        visible_ink_mask *= (player_mask > 0).astype(np.uint8)
        if other_players_mask is not None and other_players_mask.size != 0:
            visible_ink_mask *= (other_players_mask == 0).astype(np.uint8)

        if not np.any(visible_logo[:, :, 3] > 0):
            continue

        shaded_logo = apply_shading_to_ink_only(
            working_frame,
            visible_logo,
            visible_ink_mask,
            shading_intensity,
        )
        refinement_mask = (
            (visible_logo[:, :, 3] > 0).astype(np.uint8)
            if not use_lama_reconstruction
            else (visible_ink_mask > 0).astype(np.uint8)
        )
        refinement_mask *= (player_mask > 0).astype(np.uint8)

        shaded_rgb = shaded_logo[:, :, :3]
        output_frame = refinement_pipeline(
            output_frame,
            shaded_rgb,
            refinement_mask,
        )

    return output_frame, global_state


def process_video(
    video_path,
    output_dir,
    detection_model_path="yolov8n.pt",
    pose_model_path="yolov8n-pose.pt",
    sponsor_model_path="models/sponsor_detector/best_colab.pt",
    segmentation_model_path="yolov8n-seg.pt",
    logo_dir="data/logos",
    fixed_logo_filename="05_logo.png",
    use_lama_reconstruction=True,
    max_frame_side=1600,
    display=False,
):
    start_time = time.perf_counter()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(video_path).stem}_edited.mp4"
    print(f"[START] Iniciando procesamiento de video: {video_path}")
    print(f"[OUTPUT] El resultado se guardara en: {output_path}")

    detection_model = load_detection_model(detection_model_path)
    pose_model = load_pose_model(pose_model_path)
    sponsor_model = load_detection_model(sponsor_model_path)
    segmentation_model = load_segmentation_model(segmentation_model_path)
    lama_model = load_lama_model() if use_lama_reconstruction else None
    fixed_logo_path = get_fixed_logo_path(logo_dir, fixed_logo_filename)
    global_state = initialize_global_state()

    video_capture, fps, frame_size = open_video(video_path)
    writer = None
    fps_display = float(fps) if fps else 0.0
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[VIDEO] fps={fps_display:.2f}, size={frame_size[0]}x{frame_size[1]}, "
        f"lama={'on' if use_lama_reconstruction else 'off'}"
    )
    print(f"[VIDEO] Total de frames: {total_frames}")

    try:
        safe_fps = fps if fps and fps > 0 else 25.0
        frame_counter = 0

        while True:
            frame, end_of_video = read_next_frame(video_capture)
            if end_of_video:
                break

            frame_counter += 1
            if frame_counter == 1 or frame_counter % 10 == 0:
                if total_frames > 0:
                    print(f"[FRAME] Procesando frame {frame_counter}/{total_frames}...")
                else:
                    print(f"[FRAME] Procesando frame {frame_counter}...")

            processing_frame, scale = resize_frame_for_processing(frame, max_side=max_frame_side)
            output_frame, global_state = render_frame_pipeline(
                processing_frame,
                detection_model,
                pose_model,
                sponsor_model,
                segmentation_model,
                [fixed_logo_path],
                global_state,
                use_lama_reconstruction=use_lama_reconstruction,
                lama_model=lama_model,
            )

            if scale != 1.0:
                output_frame = cv2.resize(
                    output_frame,
                    (frame.shape[1], frame.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            if writer is None:
                writer = initialize_video_writer(str(output_path), safe_fps, (frame.shape[1], frame.shape[0]))

            write_frame(writer, output_frame)
            if display:
                display_frame(output_frame)

        elapsed_seconds = time.perf_counter() - start_time
        print(f"[DONE] Video completado. Frames procesados: {frame_counter}")
        print(f"[TIME] Tiempo total: {elapsed_seconds:.2f} segundos")
        return output_path
    finally:
        release_video_resources(video_capture, writer)


def build_argument_parser():
    parser = argparse.ArgumentParser(description="Pipeline de video para sustitucion de sponsor.")
    parser.add_argument("video_name", help="Nombre del video dentro de data/videos")
    parser.add_argument("--video-dir", default="data/videos", help="Carpeta de videos de entrada")
    parser.add_argument("--output-dir", default="test_videos", help="Carpeta para el video resultante")
    parser.add_argument("--logo-dir", default="data/logos", help="Carpeta de logos PNG")
    parser.add_argument("--detection-model", default="yolov8n.pt", help="Modelo de deteccion de jugadores")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="Modelo de pose")
    parser.add_argument("--sponsor-model", default="models/sponsor_detector/best_colab.pt", help="Modelo de deteccion de sponsor")
    parser.add_argument("--segmentation-model", default="yolov8n-seg.pt", help="Modelo de segmentacion")
    parser.add_argument("--no-lama", action="store_true", help="Desactiva reconstruccion con LaMa")
    parser.add_argument("--max-frame-side", type=int, default=1600, help="Lado maximo del frame para procesado")
    parser.add_argument("--display", action="store_true", help="Muestra el video procesado en una ventana")
    return parser


def main():
    hardcoded_video_path = "data/videos/01_video.mov"
    hardcoded_output_dir = "test_videos"

    if hardcoded_video_path:
        video_path = Path(hardcoded_video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"No existe el video solicitado: {video_path}")

        output_path = process_video(
            video_path=str(video_path),
            output_dir=hardcoded_output_dir,
            detection_model_path="yolov8n.pt",
            pose_model_path="yolov8n-pose.pt",
            sponsor_model_path="models/sponsor_detector/best_colab.pt",
            segmentation_model_path="yolov8n-seg.pt",
            logo_dir="data/logos",
            fixed_logo_filename="05_logo.png",
            use_lama_reconstruction=True,
            max_frame_side=1600,
            display=False,
        )
    else:
        parser = build_argument_parser()
        args = parser.parse_args()
        video_path = Path(args.video_dir) / args.video_name

        if not video_path.exists():
            raise FileNotFoundError(f"No existe el video solicitado: {video_path}")

        output_path = process_video(
            video_path=str(video_path),
            output_dir=args.output_dir,
            detection_model_path=args.detection_model,
            pose_model_path=args.pose_model,
            sponsor_model_path=args.sponsor_model,
            segmentation_model_path=args.segmentation_model,
            logo_dir=args.logo_dir,
            fixed_logo_filename="05_logo.png",
            use_lama_reconstruction=not args.no_lama,
            max_frame_side=args.max_frame_side,
            display=args.display,
        )
    print(f"Video procesado guardado en: {output_path}")


if __name__ == "__main__":
    main()
