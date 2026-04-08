import cv2
import numpy as np
from pathlib import Path

from layers.background_reconstruction.background_reconstruction_layer import (
    background_reconstruction_lama_pipeline,
    load_lama_model,
)
from layers.detection.detection_layer import (
    convert_bbox_format,
    detect_and_filter_players,
    draw_detections,
    load_detection_model,
)
from layers.geometry.geometry_sponsor_layer import get_sponsor_quad, match_sponsor_to_player
from layers.occlusion.occlusion_layer import (
    apply_occlusion,
    clip_render_to_player_mask,
    combine_masks,
    draw_player_mask_outlines,
    load_segmentation_model,
    match_mask_to_player,
    segment_players,
)
from layers.pose.pose_layer import draw_torso_regions, estimate_poses, load_pose_model
from layers.rendering.rendering_layer import (
    alpha_blend,
    compute_perspective_transform,
    load_logo,
    prepare_logo_patch,
    warp_logo,
)
from layers.refinement.refinement_layer import refinement_pipeline
from layers.shading.shading_layer import compose_shaded_logo


def detect_sponsors(model, image, min_confidence: float = 0.25) -> list:
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


def draw_cross(image, point: tuple, color: tuple, size: int = 18, thickness: int = 2) -> None:
    cv2.drawMarker(
        image,
        (int(point[0]), int(point[1])),
        color,
        markerType=cv2.MARKER_CROSS,
        markerSize=size,
        thickness=thickness,
    )


def draw_torso_spine(image, keypoints: dict) -> None:
    required_points = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(point_name not in keypoints for point_name in required_points):
        return

    left_shoulder = keypoints["left_shoulder"]
    right_shoulder = keypoints["right_shoulder"]
    left_hip = keypoints["left_hip"]
    right_hip = keypoints["right_hip"]
    shoulder_center = (
        int(round((left_shoulder[0] + right_shoulder[0]) / 2)),
        int(round((left_shoulder[1] + right_shoulder[1]) / 2)),
    )
    hip_center = (
        int(round((left_hip[0] + right_hip[0]) / 2)),
        int(round((left_hip[1] + right_hip[1]) / 2)),
    )

    cv2.line(image, shoulder_center, hip_center, (255, 0, 255), 2)
    cv2.circle(image, shoulder_center, 4, (255, 0, 255), -1)
    cv2.circle(image, hip_center, 4, (255, 0, 255), -1)


def draw_geometry_debug(image, sponsors: list, players_pose: list) -> np.ndarray:
    debug_image = image.copy()

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        keypoints = matched_player.get("keypoints", {})
        quad = get_sponsor_quad(sponsor["bbox"], keypoints)
        quad_points = np.array(
            [[int(round(px)), int(round(py))] for px, py in quad],
            dtype=np.int32,
        ).reshape((-1, 1, 2))

        cv2.polylines(debug_image, [quad_points], isClosed=True, color=(0, 255, 255), thickness=2)

        x, y, _, _ = sponsor["bbox"]
        cv2.putText(
            debug_image,
            f"sponsor {sponsor['confidence']:.2f}",
            (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for shoulder_name in ("left_shoulder", "right_shoulder"):
            if shoulder_name in keypoints:
                draw_cross(debug_image, keypoints[shoulder_name], color=(0, 255, 0))

        for hip_name in ("left_hip", "right_hip"):
            if hip_name in keypoints:
                draw_cross(debug_image, keypoints[hip_name], color=(0, 255, 0))

        draw_torso_spine(debug_image, keypoints)

    return debug_image


def quad_width_height(quad: list) -> tuple:
    points = np.array(quad, dtype=np.float32)
    top_width = np.linalg.norm(points[1] - points[0])
    bottom_width = np.linalg.norm(points[2] - points[3])
    left_height = np.linalg.norm(points[3] - points[0])
    right_height = np.linalg.norm(points[2] - points[1])

    return (top_width + bottom_width) / 2, (left_height + right_height) / 2


def find_logo_files(logo_dir: Path) -> list:
    return sorted(
        path
        for path in logo_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    )


def choose_logo_for_quad(logo_paths: list, quad: list) -> Path | None:
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


def estimate_jersey_background_color(image: np.ndarray, bbox: tuple, margin: int = 10) -> tuple:
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


def resize_image_for_testing(image: np.ndarray, max_side: int = 1920) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    largest_side = max(height, width)
    if max_side <= 0 or largest_side <= max_side:
        return image.copy(), 1.0

    scale = float(max_side) / float(largest_side)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized_image = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )
    return resized_image, scale


def render_logo_on_quad(
    image: np.ndarray,
    logo_path: Path,
    quad: list,
    background_color: tuple,
) -> np.ndarray:
    logo = load_logo(str(logo_path))
    logo_patch = prepare_logo_patch(logo, background_color)
    matrix = compute_perspective_transform(logo_patch, np.array(quad, dtype=np.float32))
    warped_logo = warp_logo(logo_patch, matrix, image.shape[:2])

    return alpha_blend(image, warped_logo)


def render_logo_overlay_on_quad(
    image: np.ndarray,
    logo_path: Path,
    quad: list,
    background_color: tuple,
    use_background_patch: bool = True,
) -> np.ndarray:
    logo = load_logo(str(logo_path))
    if use_background_patch:
        logo_overlay = prepare_logo_patch(logo, background_color)
    else:
        logo_overlay = logo.copy()

    matrix = compute_perspective_transform(logo_overlay, np.array(quad, dtype=np.float32))
    return warp_logo(logo_overlay, matrix, image.shape[:2])


def render_logo_ink_mask_on_quad(
    image: np.ndarray,
    logo_path: Path,
    quad: list,
) -> np.ndarray:
    logo = load_logo(str(logo_path))
    logo_alpha = logo[:, :, 3]
    ink_overlay = np.zeros(logo.shape[:2] + (4,), dtype=np.uint8)
    ink_overlay[:, :, 3] = logo_alpha
    matrix = compute_perspective_transform(ink_overlay, np.array(quad, dtype=np.float32))
    warped_ink = warp_logo(ink_overlay, matrix, image.shape[:2])
    return warped_ink[:, :, 3]


def apply_masks_to_overlay_alpha(
    overlay: np.ndarray,
    player_mask: np.ndarray,
    occluder_mask: np.ndarray | None = None,
) -> np.ndarray:
    masked_overlay = overlay.copy()
    alpha = masked_overlay[:, :, 3].astype(np.float32)
    alpha *= (player_mask > 0).astype(np.float32)

    if occluder_mask is not None and occluder_mask.size != 0:
        alpha *= 1.0 - (occluder_mask > 0).astype(np.float32)

    visible_pixels = alpha > 0
    alpha[visible_pixels] = 255.0
    masked_overlay[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return masked_overlay


def apply_shading_to_ink_only(
    frame_region: np.ndarray,
    logo_overlay: np.ndarray,
    ink_mask: np.ndarray,
    intensity_factor: float,
) -> np.ndarray:
    shaded_overlay = logo_overlay.copy()
    fully_shaded = compose_shaded_logo(frame_region, logo_overlay, intensity_factor)

    ink_region = ink_mask > 32
    if not np.any(ink_region):
        return shaded_overlay

    shaded_overlay[ink_region, :3] = fully_shaded[ink_region, :3]
    shaded_overlay[:, :, 3] = logo_overlay[:, :, 3]
    return shaded_overlay


def render_sponsors(
    image: np.ndarray,
    sponsors: list,
    players_pose: list,
    logo_paths: list,
) -> np.ndarray:
    rendered_image = image.copy()

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        keypoints = matched_player.get("keypoints", {})
        quad = get_sponsor_quad(sponsor["bbox"], keypoints)
        logo_path = choose_logo_for_quad(logo_paths, quad)
        if logo_path is None:
            continue

        background_color = estimate_jersey_background_color(image, sponsor["bbox"])
        rendered_image = render_logo_on_quad(
            rendered_image,
            logo_path,
            quad,
            background_color,
        )

    return rendered_image


def find_player_index(matched_player: dict, players_pose: list) -> int | None:
    for index, player_pose in enumerate(players_pose):
        if player_pose is matched_player:
            return index
        if player_pose.get("bbox") == matched_player.get("bbox"):
            return index

    return None


def torso_to_bbox(torso: dict) -> tuple | None:
    corners = torso.get("corners", [])
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


def render_sponsors_with_occlusion(
    image: np.ndarray,
    sponsors: list,
    players_pose: list,
    logo_paths: list,
    segmentation_model,
    draw_mask_outline: bool = True,
) -> np.ndarray:
    segments = segment_players(segmentation_model, image)
    player_masks = [
        match_mask_to_player(segments, player_pose["bbox"])
        for player_pose in players_pose
    ]
    rendered_image = image.copy()

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        player_index = find_player_index(matched_player, players_pose)
        if player_index is None or player_masks[player_index].size == 0:
            continue

        keypoints = matched_player.get("keypoints", {})
        quad = get_sponsor_quad(sponsor["bbox"], keypoints)
        logo_path = choose_logo_for_quad(logo_paths, quad)
        if logo_path is None:
            continue

        background_color = estimate_jersey_background_color(image, sponsor["bbox"])
        candidate_image = render_logo_on_quad(
            rendered_image,
            logo_path,
            quad,
            background_color,
        )

        player_mask = player_masks[player_index]
        candidate_image = clip_render_to_player_mask(
            rendered_image,
            candidate_image,
            player_mask,
        )

        other_player_masks = [
            mask
            for index, mask in enumerate(player_masks)
            if index != player_index and mask.size != 0
        ]
        if other_player_masks:
            other_players_mask = combine_masks(other_player_masks)
            candidate_image = apply_occlusion(
                rendered_image,
                candidate_image,
                other_players_mask,
            )

        rendered_image = candidate_image

    if draw_mask_outline and player_masks:
        rendered_image = draw_player_mask_outlines(
            rendered_image,
            [mask for mask in player_masks if mask.size != 0],
        )

    return rendered_image


def render_sponsors_with_occlusion_and_shading(
    image: np.ndarray,
    sponsors: list,
    players_pose: list,
    logo_paths: list,
    segmentation_model,
    intensity_factor: float = 0.85,
    draw_mask_outline: bool = False,
    use_background_patch: bool = True,
) -> np.ndarray:
    segments = segment_players(segmentation_model, image)
    player_masks = [
        match_mask_to_player(segments, player_pose["bbox"])
        for player_pose in players_pose
    ]
    rendered_image = image.copy()

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        player_index = find_player_index(matched_player, players_pose)
        if player_index is None or player_masks[player_index].size == 0:
            continue

        keypoints = matched_player.get("keypoints", {})
        quad = get_sponsor_quad(sponsor["bbox"], keypoints)
        logo_path = choose_logo_for_quad(logo_paths, quad)
        if logo_path is None:
            continue

        background_color = estimate_jersey_background_color(image, sponsor["bbox"])
        warped_logo = render_logo_overlay_on_quad(
            image,
            logo_path,
            quad,
            background_color,
            use_background_patch=use_background_patch,
        )
        warped_ink_mask = render_logo_ink_mask_on_quad(
            image,
            logo_path,
            quad,
        )

        other_player_masks = [
            mask
            for index, mask in enumerate(player_masks)
            if index != player_index and mask.size != 0
        ]
        other_players_mask = combine_masks(other_player_masks) if other_player_masks else None
        visible_logo = apply_masks_to_overlay_alpha(
            warped_logo,
            player_masks[player_index],
            other_players_mask,
        )
        visible_ink_mask = warped_ink_mask.copy()
        visible_ink_mask *= (player_masks[player_index] > 0).astype(np.uint8)
        if other_players_mask is not None and other_players_mask.size != 0:
            visible_ink_mask *= (other_players_mask == 0).astype(np.uint8)

        if not np.any(visible_logo[:, :, 3] > 0):
            continue

        shaded_logo = apply_shading_to_ink_only(
            image,
            visible_logo,
            visible_ink_mask,
            intensity_factor,
        )
        rendered_image = alpha_blend(rendered_image, shaded_logo)

    if draw_mask_outline and player_masks:
        rendered_image = draw_player_mask_outlines(
            rendered_image,
            [mask for mask in player_masks if mask.size != 0],
        )

    return rendered_image


def render_sponsors_with_occlusion_shading_and_refinement(
    image: np.ndarray,
    sponsors: list,
    players_pose: list,
    logo_paths: list,
    segmentation_model,
    intensity_factor: float = 0.85,
    draw_mask_outline: bool = False,
    use_background_patch: bool = True,
) -> np.ndarray:
    segments = segment_players(segmentation_model, image)
    player_masks = [
        match_mask_to_player(segments, player_pose["bbox"])
        for player_pose in players_pose
    ]
    rendered_image = image.copy()

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        player_index = find_player_index(matched_player, players_pose)
        if player_index is None or player_masks[player_index].size == 0:
            continue

        keypoints = matched_player.get("keypoints", {})
        quad = get_sponsor_quad(sponsor["bbox"], keypoints)
        logo_path = choose_logo_for_quad(logo_paths, quad)
        if logo_path is None:
            continue

        background_color = estimate_jersey_background_color(image, sponsor["bbox"])
        warped_logo = render_logo_overlay_on_quad(
            image,
            logo_path,
            quad,
            background_color,
            use_background_patch=use_background_patch,
        )
        warped_ink_mask = render_logo_ink_mask_on_quad(
            image,
            logo_path,
            quad,
        )

        other_player_masks = [
            mask
            for index, mask in enumerate(player_masks)
            if index != player_index and mask.size != 0
        ]
        other_players_mask = combine_masks(other_player_masks) if other_player_masks else None
        visible_logo = apply_masks_to_overlay_alpha(
            warped_logo,
            player_masks[player_index],
            other_players_mask,
        )
        visible_ink_mask = warped_ink_mask.copy()
        visible_ink_mask *= (player_masks[player_index] > 0).astype(np.uint8)
        if other_players_mask is not None and other_players_mask.size != 0:
            visible_ink_mask *= (other_players_mask == 0).astype(np.uint8)

        if not np.any(visible_logo[:, :, 3] > 0):
            continue

        shaded_logo = apply_shading_to_ink_only(
            image,
            visible_logo,
            visible_ink_mask,
            intensity_factor,
        )
        if use_background_patch:
            refinement_mask = (visible_logo[:, :, 3] > 0).astype(np.uint8)
        else:
            refinement_mask = (visible_ink_mask > 0).astype(np.uint8)

        if len(shaded_logo.shape) == 3 and shaded_logo.shape[2] == 4:
            shaded_logo_rgb = shaded_logo[:, :, :3]
        else:
            shaded_logo_rgb = shaded_logo

        rendered_image = refinement_pipeline(
            rendered_image,
            shaded_logo_rgb,
            refinement_mask,
        )

    if draw_mask_outline and player_masks:
        rendered_image = draw_player_mask_outlines(
            rendered_image,
            [mask for mask in player_masks if mask.size != 0],
        )

    return rendered_image


def reconstruct_sponsors_background_with_lama(
    image: np.ndarray,
    sponsors: list,
    players_pose: list,
    segmentation_model,
    lama_model,
) -> np.ndarray:
    reconstructed_image = image.copy()
    segments = segment_players(segmentation_model, image)
    player_masks = [
        match_mask_to_player(segments, player_pose["bbox"])
        for player_pose in players_pose
    ]

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        player_index = find_player_index(matched_player, players_pose)
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


def create_sponsor_protected_mask(
    image_shape: tuple,
    sponsors: list,
    players_pose: list,
    dilation_pixels: int = 6,
) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for sponsor in sponsors:
        matched_player = match_sponsor_to_player(sponsor, players_pose)
        keypoints = matched_player.get("keypoints", {})
        quad = get_sponsor_quad(sponsor["bbox"], keypoints)
        quad_points = np.array(
            [[int(round(px)), int(round(py))] for px, py in quad],
            dtype=np.int32,
        )
        cv2.fillConvexPoly(mask, quad_points, 1)

    if dilation_pixels > 0 and mask.any():
        kernel = np.ones((dilation_pixels, dilation_pixels), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)

    return mask


def process_image(
    image_path: str,
    detection_model,
    pose_model,
    sponsor_model=None,
    segmentation_model=None,
    logo_paths: list | None = None,
    output_dir: str = "test_images",
    min_confidence: float = 0.56,
    min_area: int = 50000,
    sponsor_min_confidence: float = 0.25,
    save_detection: bool = False,
    save_pose: bool = True,
    save_geometry: bool = False,
    save_full_process: bool = False,
    save_occlusion: bool = False,
    save_shading: bool = False,
    save_refinement: bool = False,
    shading_intensity: float = 0.85,
    use_lama_reconstruction: bool = False,
    lama_model=None,
    max_image_side: int = 1920,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(image_path)
    if image is None:
        print(f"{Path(image_path).name}: no se pudo leer")
        return

    original_shape = image.shape[:2]
    image, resize_scale = resize_image_for_testing(image, max_side=max_image_side)
    if resize_scale < 1.0:
        print(
            f"{Path(image_path).name}: redimensionada para test "
            f"de {original_shape[1]}x{original_shape[0]} "
            f"a {image.shape[1]}x{image.shape[0]}"
        )

    detections = detect_and_filter_players(
        detection_model,
        image,
        min_confidence=min_confidence,
        min_area=min_area,
    )

    players_pose = estimate_poses(pose_model, image, detections) if detections else []
    sponsors = detect_sponsors(sponsor_model, image, sponsor_min_confidence) if sponsor_model else []
    working_image = image.copy()

    if use_lama_reconstruction and segmentation_model is not None and lama_model is not None:
        working_image = reconstruct_sponsors_background_with_lama(
            image,
            sponsors,
            players_pose,
            segmentation_model,
            lama_model,
        )

    image_name = Path(image_path).name

    if save_detection:
        detection_debug = draw_detections(image, detections)
        cv2.imwrite(str(output_path / f"detection_{image_name}"), detection_debug)

    if save_pose:
        pose_debug = draw_torso_regions(image, players_pose)
        cv2.imwrite(str(output_path / f"pose_{image_name}"), pose_debug)

    if save_geometry:
        geometry_debug = draw_geometry_debug(image, sponsors, players_pose)
        cv2.imwrite(str(output_path / f"geometry_{image_name}"), geometry_debug)

    if save_full_process:
        full_process_image = render_sponsors(working_image, sponsors, players_pose, logo_paths or [])
        cv2.imwrite(str(output_path / f"full_{image_name}"), full_process_image)

    if save_occlusion:
        occlusion_image = render_sponsors_with_occlusion(
            working_image,
            sponsors,
            players_pose,
            logo_paths or [],
            segmentation_model,
            draw_mask_outline=True,
        )
        cv2.imwrite(str(output_path / f"occlusion_{image_name}"), occlusion_image)

    if save_shading:
        shading_image = render_sponsors_with_occlusion_and_shading(
            working_image,
            sponsors,
            players_pose,
            logo_paths or [],
            segmentation_model,
            intensity_factor=shading_intensity,
            draw_mask_outline=False,
            use_background_patch=not use_lama_reconstruction,
        )
        cv2.imwrite(
            str(output_path / f"shading_{image_name}"),
            shading_image,
        )

    if save_refinement:
        refinement_image = render_sponsors_with_occlusion_shading_and_refinement(
            working_image,
            sponsors,
            players_pose,
            logo_paths or [],
            segmentation_model,
            intensity_factor=shading_intensity,
            draw_mask_outline=False,
            use_background_patch=not use_lama_reconstruction,
        )
        cv2.imwrite(
            str(output_path / f"refinement_{image_name}"),
            refinement_image,
        )

    print(
        f"{image_name}: "
        f"{len(detections)} jugadores, "
        f"{len(players_pose)} poses, "
        f"{len(sponsors)} sponsors"
    )
    for index, player_pose in enumerate(players_pose, start=1):
        keypoints = player_pose["keypoints"]
        print(
            f"Jugador {index}: "
            f"left_shoulder={keypoints.get('left_shoulder')}, "
            f"right_shoulder={keypoints.get('right_shoulder')}"
        )


if __name__ == "__main__":
    detection_model = load_detection_model("yolov8n.pt")
    pose_model = load_pose_model("yolov8n-pose.pt")
    sponsor_model = load_detection_model("models/sponsor_detector/best_colab.pt")
    segmentation_model = load_segmentation_model("yolov8n-seg.pt")
    save_detection = False
    save_pose = False
    save_geometry = False
    save_full_process = False
    save_occlusion = False
    save_shading = False
    save_refinement = True
    image_dir = Path("data/testing")
    logo_paths = find_logo_files(Path("data/logos"))
    single_image_name = None
    lama_model = load_lama_model()
    max_image_side = 1920
    output_configs = [
        {
            "output_dir": "test_images/refinement",
            "use_lama_reconstruction": False,
        },
        {
            "output_dir": "test_images/refinement_lama",
            "use_lama_reconstruction": True,
        },
    ]

    if single_image_name:
        image_paths = [image_dir / single_image_name]
    else:
        image_paths = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    for config in output_configs:
        for image_path in image_paths:
            process_image(
                str(image_path),
                detection_model,
                pose_model,
                sponsor_model=sponsor_model,
                segmentation_model=segmentation_model,
                logo_paths=logo_paths,
                output_dir=config["output_dir"],
                save_detection=save_detection,
                save_pose=save_pose,
                save_geometry=save_geometry,
                save_full_process=save_full_process,
                save_occlusion=save_occlusion,
                save_shading=save_shading,
                save_refinement=save_refinement,
                use_lama_reconstruction=config["use_lama_reconstruction"],
                lama_model=lama_model,
                max_image_side=max_image_side,
            )
