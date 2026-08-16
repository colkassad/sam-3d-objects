from __future__ import annotations

import json
import math
import shutil
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image

from .artifacts import (
    artifact_path,
    atomic_write_json,
    config_digest,
    read_route_manifest,
    read_tracks,
    relative_artifact,
    software_versions,
    stage_is_current,
    update_stage,
)
from .geometry import (
    model_pose_to_glb_transform,
    points_from_range,
    r3_to_pytorch3d_points,
    tangent_camera,
    transform_pointmap_per_column,
    transform_points,
)
from .lidar_fit import (
    LidarFitView,
    make_fit_view,
    refine_mesh_with_lidar_rays,
    select_diverse_views,
    world_rays_from_frame,
)


@dataclass(frozen=True)
class ReconstructConfig:
    sam3d_config: str
    seed: int = 42
    mesh_target_faces: int = 10_000
    stage1_inference_steps: int = 15
    stage2_inference_steps: int = 15
    flat_shading: bool = True
    memory_profile: str = "low_vram"
    compile_model: bool = False
    fit_mode: str = "raycast"
    fit_max_axis_scale_change: float = 0.25
    fit_max_rays_per_view: int = 2_000
    fit_max_views: int = 5
    fit_max_evaluations: int = 160
    fit_max_rotation_deg: float = 20.0
    fit_grounded: bool = True
    fit_align_long_axis: bool = True
    fit_max_up_tilt_deg: float = 20.0

    def __post_init__(self) -> None:
        if self.mesh_target_faces <= 0:
            raise ValueError("mesh_target_faces must be greater than zero")
        if self.stage1_inference_steps <= 0 or self.stage2_inference_steps <= 0:
            raise ValueError("inference steps must be greater than zero")
        if self.memory_profile not in {"auto", "low_vram", "resident"}:
            raise ValueError("invalid memory profile")
        if self.fit_mode not in {"raycast", "none"}:
            raise ValueError("fit_mode must be 'raycast' or 'none'")
        if not 0 < self.fit_max_axis_scale_change < 1:
            raise ValueError("fit_max_axis_scale_change must be in (0, 1)")
        if self.fit_max_rays_per_view < 50:
            raise ValueError("fit_max_rays_per_view must be at least 50")
        if self.fit_max_views <= 0:
            raise ValueError("fit_max_views must be greater than zero")
        if self.fit_max_evaluations <= 0:
            raise ValueError("fit_max_evaluations must be greater than zero")
        if not 0 < self.fit_max_rotation_deg <= 180:
            raise ValueError("fit_max_rotation_deg must be in (0, 180]")
        if not 0 < self.fit_max_up_tilt_deg < 90:
            raise ValueError("fit_max_up_tilt_deg must be in (0, 90)")

    def manifest_value(self) -> dict[str, Any]:
        return asdict(self)


def _load_inference_api() -> tuple[Callable[..., Any], Callable[..., Any]]:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    notebook_dir = repo_root / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    from inference import Inference, load_image
    return Inference, load_image


def _prediction_for_track(track: dict[str, Any]) -> dict[str, Any]:
    selected = track["selected_observation_id"]
    try:
        return next(value for value in track["observations"] if value["id"] == selected)
    except StopIteration as exc:
        raise ValueError(f"track {track['id']} has no selected observation") from exc


def _circular_mask_center(mask: np.ndarray) -> int:
    _, columns = np.nonzero(mask)
    if not len(columns):
        raise ValueError("selected mask is empty")
    width = mask.shape[1]
    angles = columns.astype(np.float64) / width * 2.0 * np.pi
    angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    if angle < 0:
        angle += 2.0 * np.pi
    return int(round(angle / (2.0 * np.pi) * width)) % width


def _square_crop(
    rgb: np.ndarray,
    mask: np.ndarray,
    pointmap: np.ndarray,
    *,
    box_factor: float = 1.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        raise ValueError("selected mask is empty")
    center_row = (int(rows.min()) + int(rows.max())) / 2.0
    center_col = (int(columns.min()) + int(columns.max())) / 2.0
    size = max(int(rows.max() - rows.min() + 1), int(columns.max() - columns.min() + 1), 2)
    size = max(2, int(math.ceil(size * box_factor)))
    top = int(math.floor(center_row - size / 2))
    left = int(math.floor(center_col - size / 2))
    bottom, right = top + size, left + size
    output_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    output_mask = np.zeros((size, size), dtype=bool)
    output_points = np.full((size, size, 3), np.nan, dtype=np.float32)
    source_top, source_left = max(0, top), max(0, left)
    source_bottom, source_right = min(rgb.shape[0], bottom), min(rgb.shape[1], right)
    target_top, target_left = source_top - top, source_left - left
    target_bottom = target_top + source_bottom - source_top
    target_right = target_left + source_right - source_left
    source = np.s_[source_top:source_bottom, source_left:source_right]
    target = np.s_[target_top:target_bottom, target_left:target_right]
    output_rgb[target] = rgb[source]
    output_mask[target] = mask[source]
    output_points[target] = pointmap[source]
    return output_rgb, output_mask, output_points


def prepare_sam3d_inputs(
    run_dir: Path,
    manifest: dict[str, Any],
    track: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, Any, np.ndarray, dict[str, Any]]:
    """Build an object-centered RGB/mask and metric PyTorch3D pointmap."""

    observation = _prediction_for_track(track)
    frame = next(
        value for value in manifest["keyframes"] if value["id"] == observation["frame_id"]
    )
    rgb = np.asarray(Image.open(artifact_path(run_dir, frame["rgb"])).convert("RGB"))
    mask = np.asarray(
        Image.open(artifact_path(run_dir, observation["mask_path"])).convert("L")
    ) > 0
    calibration_document = json.loads(
        artifact_path(run_dir, manifest["calibration"]).read_text(encoding="utf-8")
    )
    with np.load(artifact_path(run_dir, calibration_document["arrays"])) as calibration:
        ray_direction = calibration["ray_direction"]
        ray_origin = calibration["ray_origin"]
        sensor_to_body = calibration["sensor_to_body"].astype(np.float64)
    with np.load(artifact_path(run_dir, frame["geometry"])) as geometry:
        range_mm = geometry["range_mm"]
        poses = geometry["body_to_world"].astype(np.float64)
        reference_column = int(geometry["reference_column"])
    points_sensor = points_from_range(range_mm, ray_direction, ray_origin)
    points_world = transform_pointmap_per_column(points_sensor, poses, sensor_to_body)
    world_from_reference_sensor = poses[reference_column] @ sensor_to_body
    reference_sensor_from_world = np.linalg.inv(world_from_reference_sensor)
    points_reference_sensor = transform_points(points_world, reference_sensor_from_world)
    selected_reference_points = points_reference_sensor[
        mask & np.all(np.isfinite(points_reference_sensor), axis=-1)
    ]
    if len(selected_reference_points):
        forward = np.median(selected_reference_points, axis=0)
        forward /= np.linalg.norm(forward)
    else:
        azimuth, elevation = observation["azimuth_rad"], observation["elevation_rad"]
        forward = np.asarray(
            [
                math.cos(elevation) * math.cos(azimuth),
                math.cos(elevation) * math.sin(azimuth),
                math.sin(elevation),
            ],
            dtype=np.float64,
        )
    camera = tangent_camera(forward)
    points_r3 = transform_points(points_reference_sensor, camera.camera_from_sensor)
    points_p3d = r3_to_pytorch3d_points(points_r3).astype(np.float32)

    shift = rgb.shape[1] // 2 - _circular_mask_center(mask)
    rgb = np.roll(rgb, shift, axis=1)
    mask = np.roll(mask, shift, axis=1)
    points_p3d = np.roll(points_p3d, shift, axis=1)
    rgb, mask, points_p3d = _square_crop(rgb, mask, points_p3d)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("SAM3D reconstruction requires PyTorch") from exc
    pointmap_tensor = torch.from_numpy(points_p3d)
    p3d_from_r3 = np.eye(4, dtype=np.float64)
    p3d_from_r3[0, 0] = -1.0
    p3d_from_r3[1, 1] = -1.0
    world_from_p3d = (
        world_from_reference_sensor
        @ camera.sensor_from_camera
        @ np.linalg.inv(p3d_from_r3)
    )
    context = {
        "frame_id": frame["id"],
        "roll_columns": shift,
        "world_from_pytorch3d_camera": world_from_p3d,
        "world_from_reference_sensor": world_from_reference_sensor,
    }
    return rgb, mask, pointmap_tensor, world_from_p3d, context




def _unit_vector(value: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(vector))
    if not math.isfinite(length) or length < 1e-8:
        vector = np.asarray(fallback, dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(vector))
    return vector / length


def _rotation_difference_degrees(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).T @ np.asarray(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def stabilize_mesh_orientation(
    vertices_local: np.ndarray,
    target_world: np.ndarray,
    initial_transform: np.ndarray,
    *,
    grounded: bool = True,
    align_long_axis: bool = True,
    max_up_tilt_deg: float = 20.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a world-up sanity prior and a robust horizontal LiDAR heading."""

    vertices = np.asarray(vertices_local, dtype=np.float64)
    target = np.asarray(target_world, dtype=np.float64)
    transform = np.asarray(initial_transform, dtype=np.float64).copy()
    scales = np.linalg.norm(transform[:3, :3], axis=0)
    if not np.all(np.isfinite(scales)) or np.any(scales <= 1e-8):
        raise ValueError("initial mesh transform has invalid scale")
    current_axes = transform[:3, :3] / scales
    world_up = np.asarray([0.0, 0.0, 1.0])
    current_up = _unit_vector(current_axes[:, 1], world_up)
    up_dot_before = float(np.dot(current_up, world_up))
    maximum_tilt_cosine = math.cos(math.radians(max_up_tilt_deg))
    tilt_corrected = bool(grounded and up_dot_before < maximum_tilt_cosine)
    desired_up = world_up if tilt_corrected else current_up

    local_extents = np.quantile(vertices, 0.90, axis=0) - np.quantile(
        vertices, 0.10, axis=0
    )
    primary_axis = 0 if local_extents[0] >= local_extents[2] else 2
    secondary_axis = 2 if primary_axis == 0 else 0
    local_anisotropy = float(
        local_extents[primary_axis] / max(local_extents[secondary_axis], 1e-8)
    )

    centered_xy = target[:, :2] - np.median(target[:, :2], axis=0)
    covariance = centered_xy.T @ centered_xy / max(len(centered_xy) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    target_anisotropy = float(eigenvalues[0] / max(eigenvalues[1], 1e-9))
    current_primary = _unit_vector(current_axes[:, primary_axis], [1.0, 0.0, 0.0])
    use_lidar_heading = bool(
        align_long_axis and local_anisotropy >= 1.25 and target_anisotropy >= 1.25
    )
    if use_lidar_heading:
        desired_primary = np.asarray(
            [eigenvectors[0, 0], eigenvectors[1, 0], 0.0], dtype=np.float64
        )
    else:
        desired_primary = current_primary.copy()
    desired_primary -= desired_up * float(np.dot(desired_primary, desired_up))
    desired_primary = _unit_vector(desired_primary, [1.0, 0.0, 0.0])
    current_projected = current_primary - desired_up * float(
        np.dot(current_primary, desired_up)
    )
    if np.linalg.norm(current_projected) > 1e-8 and np.dot(
        desired_primary, current_projected
    ) < 0:
        desired_primary *= -1.0

    if primary_axis == 2:
        z_axis = desired_primary
        y_axis = desired_up
        x_axis = _unit_vector(np.cross(y_axis, z_axis), [1.0, 0.0, 0.0])
        z_axis = _unit_vector(np.cross(x_axis, y_axis), z_axis)
    else:
        x_axis = desired_primary
        y_axis = desired_up
        z_axis = _unit_vector(np.cross(x_axis, y_axis), [0.0, 1.0, 0.0])
        x_axis = _unit_vector(np.cross(y_axis, z_axis), x_axis)
    stabilized_axes = np.column_stack((x_axis, y_axis, z_axis))
    local_center = np.median(vertices, axis=0)
    world_center = transform_points(local_center[None], transform)[0]
    transform[:3, :3] = stabilized_axes @ np.diag(scales)
    # Rotate about the mesh center. Keeping the old translation here can move an
    # off-origin SAM3D mesh by meters while merely trying to make it upright.
    transform[:3, 3] = world_center - transform[:3, :3] @ local_center
    return transform, {
        "grounded": grounded,
        "tilt_corrected": tilt_corrected,
        "up_dot_world_before": up_dot_before,
        "up_dot_world_after": float(np.dot(stabilized_axes[:, 1], world_up)),
        "orientation_change_deg": _rotation_difference_degrees(
            current_axes, stabilized_axes
        ),
        "long_axis_aligned": use_lidar_heading,
        "local_primary_axis": "x" if primary_axis == 0 else "z",
        "local_horizontal_anisotropy": local_anisotropy,
        "lidar_horizontal_anisotropy": target_anisotropy,
    }




def _tensor_values(value: Any, count: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size < count:
        raise ValueError(f"SAM3D pose output has {array.size} values; expected {count}")
    return array[:count]


def _as_trimesh(value: Any) -> Any:
    import trimesh

    if isinstance(value, trimesh.Trimesh):
        return value.copy()
    if isinstance(value, trimesh.Scene):
        geometries = tuple(value.geometry.values())
        if len(geometries) != 1:
            raise ValueError("SAM3D GLB scene must contain exactly one geometry")
        return geometries[0].copy()
    raise TypeError(f"unsupported SAM3D GLB type {type(value).__name__}")


def _export_positioned_glb(mesh: Any, transform: np.ndarray, path: Path, node_name: str) -> None:
    import trimesh

    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name=node_name, geom_name=node_name, transform=transform)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path, file_type="glb")


def _write_empty_glb(path: Path) -> None:
    document = json.dumps(
        {
            "asset": {"version": "2.0", "generator": "sam3d-ouster-route"},
            "scene": 0,
            "scenes": [{"nodes": []}],
            "nodes": [],
        },
        separators=(",", ":"),
    ).encode("utf-8")
    document += b" " * ((4 - len(document) % 4) % 4)
    total_length = 12 + 8 + len(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
        stream.write(struct.pack("<I4s", len(document), b"JSON"))
        stream.write(document)


def _load_positioned_geometry(mesh_path: Path) -> tuple[Any, np.ndarray]:
    import trimesh

    scene = trimesh.load(mesh_path, force="scene")
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) != 1:
        raise ValueError(f"positioned GLB must contain one mesh node: {mesh_path}")
    transform, geometry_name = scene.graph[nodes[0]]
    return scene.geometry[geometry_name].copy(), np.asarray(transform)


def _reconstruction_eligible(track: dict[str, Any]) -> bool:
    """Only confirmed-static, in-range tracks reach SAM3D."""

    motion_state = track.get("motion_state")
    if motion_state is None:
        return not bool(track.get("dynamic"))
    if motion_state != "confirmed_static":
        return False
    range_gate = track.get("range_gate")
    return range_gate is None or bool(range_gate.get("eligible", True))


def load_lidar_fit_views(
    run_dir: Path,
    manifest: dict[str, Any],
    track: dict[str, Any],
    *,
    maximum_rays: int,
    maximum_views: int,
) -> list[LidarFitView]:
    """Load reliable range-qualified observations in their original viewpoints."""

    calibration_document = json.loads(
        artifact_path(run_dir, manifest["calibration"]).read_text(encoding="utf-8")
    )
    with np.load(artifact_path(run_dir, calibration_document["arrays"])) as calibration:
        ray_direction = calibration["ray_direction"].astype(np.float64)
        ray_origin = calibration["ray_origin"].astype(np.float64)
        sensor_to_body = calibration["sensor_to_body"].astype(np.float64)
    frames = {value["id"]: value for value in manifest["keyframes"]}
    range_gate = track.get("range_gate") or {}
    eligible_ids = range_gate.get("eligible_observation_ids")
    eligible = set(eligible_ids) if eligible_ids is not None else None
    views: list[LidarFitView] = []
    for observation in track["observations"]:
        if eligible is not None and observation["id"] not in eligible:
            continue
        if float(observation.get("inlier_fraction", 0.0)) < 0.20:
            continue
        frame = frames.get(observation["frame_id"])
        if frame is None:
            continue
        cleaned_path = observation.get("cleaned_mask_path") or observation.get("mask_path")
        raw_path = observation.get("mask_path") or cleaned_path
        if not cleaned_path or not raw_path:
            continue
        cleaned = np.asarray(
            Image.open(artifact_path(run_dir, cleaned_path)).convert("L")
        ) > 0
        raw = np.asarray(Image.open(artifact_path(run_dir, raw_path)).convert("L")) > 0
        with np.load(artifact_path(run_dir, frame["geometry"])) as geometry:
            origins, directions, ranges = world_rays_from_frame(
                geometry["range_mm"],
                ray_direction,
                ray_origin,
                geometry["body_to_world"],
                sensor_to_body,
            )
        quality = max(0.05, float(observation.get("quality", observation.get("score", 0.5))))
        quality *= max(0.20, float(observation.get("inlier_fraction", 1.0)))
        view = make_fit_view(
            observation_id=observation["id"],
            cleaned_mask=cleaned,
            raw_mask=raw,
            origins_world=origins,
            directions_world=directions,
            ranges_m=ranges,
            maximum_rays=maximum_rays,
            weight=quality,
        )
        if view is not None:
            views.append(view)
    selected = select_diverse_views(views, maximum_views)
    if selected:
        largest = max(value.weight for value in selected)
        selected = [
            LidarFitView(
                **{
                    **value.__dict__,
                    "weight": float(np.clip(value.weight / largest, 0.25, 1.0)),
                }
            )
            for value in selected
        ]
    return selected


def reconstruct_route(
    run_dir: Path,
    config: ReconstructConfig,
    *,
    overwrite: bool = False,
    inference_factory: Optional[Callable[..., Any]] = None,
    image_loader: Optional[Callable[[Path], Any]] = None,
) -> tuple[Path, int]:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    if manifest.get("stages", {}).get("segment", {}).get("status") != "complete":
        raise RuntimeError("segment stage must complete before reconstruction")
    tracks_path = artifact_path(run_dir, manifest["tracks"])
    tracks_document = read_tracks(tracks_path)
    unfinished = [
        track
        for track in tracks_document["tracks"]
        if _reconstruction_eligible(track) and track.get("status") != "ok"
    ]
    if (
        stage_is_current(manifest, "reconstruct", config.manifest_value())
        and not overwrite
        and not unfinished
    ):
        return artifact_path(run_dir, manifest["outputs"]["scene_glb"]), 0
    current = manifest.get("stages", {}).get("reconstruct")
    if (
        current
        and current.get("config_sha256") != config_digest(config.manifest_value())
        and not overwrite
    ):
        raise RuntimeError("reconstruct configuration changed; pass --overwrite")
    if overwrite:
        for path in (run_dir / "meshes", run_dir / "scene.glb", run_dir / "scene.json"):
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        for track in tracks_document["tracks"]:
            if _reconstruction_eligible(track):
                track["status"] = "pending"
                track["mesh"] = None
    update_stage(manifest, "reconstruct", config.manifest_value(), status="running")
    atomic_write_json(manifest_path, manifest)

    pending = [
        track
        for track in tracks_document["tracks"]
        if _reconstruction_eligible(track) and track.get("status") != "ok"
    ]
    if pending and (inference_factory is None or image_loader is None):
        Inference, load_image = _load_inference_api()
        inference_factory = inference_factory or Inference
        image_loader = image_loader or load_image
    inference = None
    if pending:
        inference = inference_factory(
            str(Path(config.sam3d_config).expanduser().resolve()),
            compile=config.compile_model,
            memory_profile=config.memory_profile,
        )
    failures = 0
    try:
        for track in pending:
            try:
                rgb, mask, pointmap, world_from_p3d, context = prepare_sam3d_inputs(
                    run_dir, manifest, track
                )
                track_dir = run_dir / "tracks" / track["id"]
                crop_path = track_dir / "sam3d_crop.png"
                crop_mask_path = track_dir / "sam3d_mask.png"
                Image.fromarray(rgb).save(crop_path, format="PNG")
                Image.fromarray(mask.astype(np.uint8) * 255).save(
                    crop_mask_path, format="PNG"
                )
                image = image_loader(crop_path)
                output = inference(
                    image,
                    mask,
                    seed=config.seed,
                    pointmap=pointmap,
                    mesh_target_faces=config.mesh_target_faces,
                    flat_shading=config.flat_shading,
                    stage1_inference_steps=config.stage1_inference_steps,
                    stage2_inference_steps=config.stage2_inference_steps,
                )
                if output.get("glb") is None:
                    raise RuntimeError("SAM3D output did not contain a GLB mesh")
                rotation = _tensor_values(output["rotation"], 4)
                translation = _tensor_values(output["translation"], 3)
                scale = _tensor_values(output["scale"], 3)
                camera_from_glb = model_pose_to_glb_transform(
                    rotation, translation, scale
                )
                initial_world_from_glb = world_from_p3d @ camera_from_glb
                mesh = _as_trimesh(output["glb"])
                point_artifact = track.get("reconstruction_points") or track["points"]
                with np.load(artifact_path(run_dir, point_artifact)) as point_file:
                    track_points = point_file["points_world"]
                stabilized_transform, orientation = stabilize_mesh_orientation(
                    np.asarray(mesh.vertices),
                    track_points,
                    initial_world_from_glb,
                    grounded=config.fit_grounded,
                    align_long_axis=config.fit_align_long_axis,
                    max_up_tilt_deg=config.fit_max_up_tilt_deg,
                )
                if config.fit_mode == "none":
                    final_transform = stabilized_transform
                    fit = {
                        "method": "none",
                        "accepted": False,
                        "reason": "sensor-ray refinement disabled",
                        "orientation": orientation,
                    }
                else:
                    views = load_lidar_fit_views(
                        run_dir,
                        manifest,
                        track,
                        maximum_rays=config.fit_max_rays_per_view,
                        maximum_views=config.fit_max_views,
                    )
                    final_transform, fit = refine_mesh_with_lidar_rays(
                        np.asarray(mesh.vertices),
                        np.asarray(mesh.faces),
                        stabilized_transform,
                        views,
                        track_points,
                        max_axis_scale_change=config.fit_max_axis_scale_change,
                        max_rotation_deg=config.fit_max_rotation_deg,
                        max_evaluations=config.fit_max_evaluations,
                        grounded=config.fit_grounded,
                    )
                    fit["orientation"] = orientation
                mesh_path = run_dir / "meshes" / f"{track['id']}.glb"
                _export_positioned_glb(mesh, final_transform, mesh_path, track["id"])
                track["status"] = "ok"
                track["sam3d_crop"] = relative_artifact(run_dir, crop_path)
                track["sam3d_mask"] = relative_artifact(run_dir, crop_mask_path)
                track["mesh"] = {
                    "path": relative_artifact(run_dir, mesh_path),
                    "world_from_glb": final_transform.tolist(),
                    "initial_world_from_glb": initial_world_from_glb.tolist(),
                    "sam3d_pose": {
                        "rotation_wxyz": rotation.tolist(),
                        "translation": translation.tolist(),
                        "scale": scale.tolist(),
                        "convention": "PyTorch3D row-vector local-to-camera",
                    },
                    "pointmap_world_from_camera": world_from_p3d.tolist(),
                    "fit": fit,
                    "source_frame_id": context["frame_id"],
                }
            except Exception as exc:
                failures += 1
                track["status"] = "failed"
                track["mesh"] = {
                    "error": f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:500]}"
                }
            atomic_write_json(tracks_path, tracks_document)

        import trimesh

        scene = trimesh.Scene()
        scene_records = []
        for track in tracks_document["tracks"]:
            mesh_record = track.get("mesh") or {}
            if (
                not _reconstruction_eligible(track)
                or track.get("status") != "ok"
                or not mesh_record.get("path")
            ):
                continue
            mesh_path = artifact_path(run_dir, mesh_record["path"])
            mesh, transform = _load_positioned_geometry(mesh_path)
            scene.add_geometry(
                mesh,
                node_name=track["id"],
                geom_name=track["id"],
                transform=transform,
            )
            selected = _prediction_for_track(track)
            scene_records.append(
                {
                    "track_id": track["id"],
                    "prompt": track["prompt"],
                    "mesh": mesh_record["path"],
                    "world_from_glb": mesh_record["world_from_glb"],
                    "source_rgb": track["best_rgb"],
                    "source_mask": track["best_mask"],
                    "sam3d_crop": track.get("sam3d_crop"),
                    "sam3_score": selected["score"],
                    "dimensions_m": selected["extents_world"],
                    "median_range_m": selected["median_range_m"],
                    "centroid_world": track.get("reconstruction_centroid_world")
                    or track["centroid_world"],
                    "track_centroid_world": track["centroid_world"],
                    "range_gate": track.get("range_gate"),
                    "fit": mesh_record["fit"],
                }
            )
        scene_glb = run_dir / "scene.glb"
        if scene_records:
            scene.export(scene_glb, file_type="glb")
        else:
            _write_empty_glb(scene_glb)
        scene_json = run_dir / "scene.json"
        atomic_write_json(
            scene_json,
            {
                "schema": "ouster-mesh-scene/v1",
                "coordinate_system": manifest["coordinate_system"],
                "meshes": scene_records,
                "point_cloud": manifest.get("outputs", {}).get("point_cloud"),
            },
        )
        manifest["outputs"]["scene_glb"] = relative_artifact(run_dir, scene_glb)
        manifest["outputs"]["scene_json"] = relative_artifact(run_dir, scene_json)
        manifest.setdefault("software", {}).update(
            software_versions(
                {
                    "sam3d_objects": "sam-3d-objects",
                    "torch": "torch",
                    "trimesh": "trimesh",
                    "open3d": "open3d",
                }
            )
        )
        update_stage(manifest, "reconstruct", config.manifest_value(), status="complete")
        atomic_write_json(tracks_path, tracks_document)
        atomic_write_json(manifest_path, manifest)
        return scene_glb, failures
    except Exception as exc:
        update_stage(
            manifest,
            "reconstruct",
            config.manifest_value(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(tracks_path, tracks_document)
        raise
