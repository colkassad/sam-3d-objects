from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import Delaunay, QhullError, cKDTree

from sam3_masking.artifacts import load_mask_manifest

from .artifacts import (
    artifact_path,
    atomic_write_json,
    config_digest,
    read_route_manifest,
    relative_artifact,
    software_versions,
    stage_is_current,
    update_stage,
)
from .extract import ExtractConfig, extract_route, write_binary_ply
from .geometry import points_from_range, transform_pointmap_per_column


SURFACE_SCHEMA = "ouster-surface-tin/v1"


@dataclass(frozen=True)
class SurfaceSegmentConfig:
    prompts: tuple[str, ...]
    sam3_model_dir: str
    sam3_executable: str = "sam3-mask-route"
    sam3_device: str = "auto"
    sam3_dtype: str = "auto"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5

    def __post_init__(self) -> None:
        if not self.prompts or any(not value.strip() for value in self.prompts):
            raise ValueError("at least one nonempty surface prompt is required")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")
        if not 0.0 <= self.mask_threshold <= 1.0:
            raise ValueError("mask_threshold must be between 0 and 1")
        if self.sam3_dtype not in {"auto", "bf16", "fp16", "fp32"}:
            raise ValueError("unsupported SAM3 dtype")

    def manifest_value(self) -> dict[str, Any]:
        value = asdict(self)
        value["prompts"] = list(self.prompts)
        return value


@dataclass(frozen=True)
class TinConfig:
    surface_resolution_m: float = 0.20
    max_surface_range_m: Optional[float] = 30.0
    max_triangle_edge_m: float = 1.0
    max_slope_deg: float = 45.0
    tin_tile_size_m: float = 50.0

    def __post_init__(self) -> None:
        for name in (
            "surface_resolution_m",
            "max_triangle_edge_m",
            "tin_tile_size_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and greater than zero")
        if self.max_surface_range_m is not None and (
            not math.isfinite(self.max_surface_range_m) or self.max_surface_range_m <= 0
        ):
            raise ValueError("max_surface_range_m must be finite and greater than zero")
        if not math.isfinite(self.max_slope_deg) or not 0 < self.max_slope_deg < 90:
            raise ValueError("max_slope_deg must be finite and between 0 and 90")
        if self.tin_tile_size_m <= 2.0 * self.max_triangle_edge_m:
            raise ValueError(
                "tin_tile_size_m must be greater than twice max_triangle_edge_m"
            )

    def manifest_value(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SurfaceOutputs:
    point_cloud: Path
    mesh: Path
    metadata: Path


@dataclass(frozen=True)
class SurfacePointSet:
    points: np.ndarray
    colors: np.ndarray
    statistics: dict[str, Any]


class _XYPointAccumulator:
    """Fuse repeated surface returns into deterministic XY cells."""

    def __init__(self, resolution_m: float) -> None:
        self.resolution_m = float(resolution_m)
        self._values: dict[tuple[int, int], list[Any]] = {}

    def add(self, points: np.ndarray, colors: np.ndarray) -> None:
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        if len(xyz) != len(rgb):
            raise ValueError("surface point and RGB counts differ")
        valid = np.all(np.isfinite(xyz), axis=1)
        xyz, rgb = xyz[valid], rgb[valid]
        if not len(xyz):
            return
        keys = np.floor(xyz[:, :2] / self.resolution_m).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        sums = np.zeros((len(unique), 6), dtype=np.float64)
        counts = np.bincount(inverse)
        np.add.at(sums[:, :3], inverse, xyz)
        np.add.at(sums[:, 3:], inverse, rgb)
        for key, value, count in zip(unique, sums, counts):
            item = self._values.setdefault(
                (int(key[0]), int(key[1])), [np.zeros(6, dtype=np.float64), 0]
            )
            item[0] += value
            item[1] += int(count)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        points: list[np.ndarray] = []
        colors: list[np.ndarray] = []
        for key in sorted(self._values):
            sums, count = self._values[key]
            points.append(sums[:3] / count)
            colors.append(np.rint(sums[3:] / count))
        return (
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        )


def _surface_outputs(run_dir: Path) -> SurfaceOutputs:
    directory = run_dir / "surface"
    return SurfaceOutputs(
        point_cloud=directory / "surface-points.ply",
        mesh=directory / "surface.glb",
        metadata=directory / "surface.json",
    )


def _clear_surface_build(run_dir: Path, manifest: dict[str, Any]) -> None:
    directory = run_dir / "surface"
    if directory.exists():
        shutil.rmtree(directory)
    outputs = manifest.setdefault("outputs", {})
    for name in ("surface_point_cloud", "surface_glb", "surface_metadata"):
        outputs.pop(name, None)
    manifest.get("stages", {}).pop("surface_tin", None)


def _clear_surface_segmentation(run_dir: Path, manifest: dict[str, Any]) -> None:
    for frame in manifest["keyframes"]:
        directory = run_dir / "frames" / frame["id"] / "surface-segmentation"
        if directory.exists():
            shutil.rmtree(directory)
        frame["surface_mask_manifest"] = None
    manifest["surface_prompts"] = []
    _clear_surface_build(run_dir, manifest)


def segment_surface_route(
    run_dir: Path,
    config: SurfaceSegmentConfig,
    *,
    overwrite: bool = False,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> Path:
    """Run SAM 3 into a surface-only mask artifact set."""

    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    if manifest.get("stages", {}).get("extract", {}).get("status") != "complete":
        raise RuntimeError("extract stage must complete before surface segmentation")
    masks_exist = all(
        frame.get("surface_mask_manifest")
        and artifact_path(run_dir, frame["surface_mask_manifest"]).is_file()
        for frame in manifest["keyframes"]
    )
    if (
        stage_is_current(manifest, "surface_segment", config.manifest_value())
        and masks_exist
        and not overwrite
    ):
        return manifest_path
    stage = manifest.get("stages", {}).get("surface_segment")
    if (
        stage
        and stage.get("config_sha256") != config_digest(config.manifest_value())
        and not overwrite
    ):
        raise RuntimeError(
            "surface segmentation configuration changed; pass --overwrite"
        )
    if overwrite:
        _clear_surface_segmentation(run_dir, manifest)
    else:
        _clear_surface_build(run_dir, manifest)
    update_stage(manifest, "surface_segment", config.manifest_value(), status="running")
    atomic_write_json(manifest_path, manifest)
    try:
        command = [
            config.sam3_executable,
            "--run-dir",
            str(run_dir),
            "--model-dir",
            str(Path(config.sam3_model_dir).expanduser().resolve()),
            "--score-threshold",
            str(config.score_threshold),
            "--mask-threshold",
            str(config.mask_threshold),
            "--device",
            config.sam3_device,
            "--dtype",
            config.sam3_dtype,
            "--artifact-set",
            "surface",
        ]
        for prompt in config.prompts:
            command.extend(("--prompt", prompt))
        if overwrite:
            command.append("--overwrite")
        completed = subprocess_run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(
                f"SAM3 surface batch exited with status {completed.returncode}"
            )
        manifest = read_route_manifest(manifest_path)
        if not all(
            frame.get("surface_mask_manifest") for frame in manifest["keyframes"]
        ):
            raise RuntimeError(
                "SAM3 surface batch did not produce every frame manifest"
            )
        manifest["surface_prompts"] = list(config.prompts)
        update_stage(
            manifest, "surface_segment", config.manifest_value(), status="complete"
        )
        atomic_write_json(manifest_path, manifest)
        return manifest_path
    except Exception as exc:
        manifest = read_route_manifest(manifest_path)
        update_stage(
            manifest,
            "surface_segment",
            config.manifest_value(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(manifest_path, manifest)
        raise


def collect_surface_points(
    run_dir: Path, manifest: dict[str, Any], config: TinConfig
) -> SurfacePointSet:
    """Fuse all surface-mask LiDAR returns into RGB XY cells."""

    calibration_document = json.loads(
        artifact_path(run_dir, manifest["calibration"]).read_text(encoding="utf-8")
    )
    with np.load(artifact_path(run_dir, calibration_document["arrays"])) as values:
        ray_direction = values["ray_direction"].astype(np.float64)
        ray_origin = values["ray_origin"].astype(np.float64)
        sensor_to_body = values["sensor_to_body"].astype(np.float64)
    accumulator = _XYPointAccumulator(config.surface_resolution_m)
    frame_statistics: list[dict[str, Any]] = []
    selected_pixels = 0
    valid_returns = 0
    prediction_count = 0
    for frame in manifest["keyframes"]:
        mask_record = frame.get("surface_mask_manifest")
        if not mask_record:
            raise RuntimeError(f"keyframe {frame['id']} has no surface mask manifest")
        mask_frame = load_mask_manifest(artifact_path(run_dir, mask_record))
        union = np.zeros((mask_frame.height, mask_frame.width), dtype=bool)
        for prediction in mask_frame.predictions:
            union |= prediction.mask
        with np.load(artifact_path(run_dir, frame["geometry"])) as geometry:
            range_mm = geometry["range_mm"].astype(np.uint32)
            poses = geometry["body_to_world"].astype(np.float64)
        if union.shape != range_mm.shape:
            raise RuntimeError(
                f"surface mask shape {union.shape} does not match range shape "
                f"{range_mm.shape} for {frame['id']}"
            )
        rgb = np.asarray(
            Image.open(artifact_path(run_dir, frame["rgb"])).convert("RGB")
        )
        if rgb.shape[:2] != range_mm.shape:
            raise RuntimeError(
                f"RGB shape {rgb.shape[:2]} does not match range shape "
                f"{range_mm.shape} for {frame['id']}"
            )
        valid = union & (range_mm != 0)
        if config.max_surface_range_m is not None:
            valid &= range_mm <= config.max_surface_range_m * 1000.0
        points_sensor = points_from_range(range_mm, ray_direction, ray_origin)
        points_world = transform_pointmap_per_column(
            points_sensor, poses, sensor_to_body
        )
        valid &= np.all(np.isfinite(points_world), axis=-1)
        accumulator.add(points_world[valid], rgb[valid])
        mask_pixels = int(np.count_nonzero(union))
        frame_valid = int(np.count_nonzero(valid))
        selected_pixels += mask_pixels
        valid_returns += frame_valid
        prediction_count += len(mask_frame.predictions)
        frame_statistics.append(
            {
                "frame_id": frame["id"],
                "prediction_count": len(mask_frame.predictions),
                "union_mask_pixels": mask_pixels,
                "valid_surface_returns": frame_valid,
            }
        )
    points, colors = accumulator.arrays()
    return SurfacePointSet(
        points=points,
        colors=colors,
        statistics={
            "keyframe_count": len(manifest["keyframes"]),
            "prediction_count": prediction_count,
            "union_mask_pixels": selected_pixels,
            "valid_surface_returns": valid_returns,
            "fused_point_count": len(points),
            "frames": frame_statistics,
        },
    )


def _local_spacing(xy: np.ndarray, resolution_m: float) -> np.ndarray:
    count = len(xy)
    neighbors = min(7, count)
    distances, _ = cKDTree(xy).query(xy, k=neighbors)
    if distances.ndim == 1:
        distances = distances[:, np.newaxis]
    divisor = math.sqrt(max(neighbors - 1, 1))
    spacing = distances[:, -1] / divisor
    return np.maximum(spacing, resolution_m)


def _component_statistics(vertices: np.ndarray, faces: np.ndarray) -> dict[str, Any]:
    used = np.unique(faces)
    remap = np.full(len(vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    local_faces = remap[faces]
    edges = np.vstack(
        (local_faces[:, [0, 1]], local_faces[:, [1, 2]], local_faces[:, [2, 0]])
    )
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    graph = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(used), len(used)),
    ).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    face_labels = labels[local_faces[:, 0]]
    face_counts = np.bincount(face_labels, minlength=component_count)
    triangles = vertices[faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    component_areas = np.bincount(face_labels, weights=areas, minlength=component_count)
    order = np.argsort(component_areas)[::-1]
    return {
        "count": int(component_count),
        "components": [
            {
                "face_count": int(face_counts[index]),
                "area_m2": float(component_areas[index]),
            }
            for index in order
        ],
    }


def triangulate_surface(
    points: np.ndarray, config: TinConfig
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a conservatively supported, tiled 2.5D Delaunay TIN."""

    vertices = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(vertices) < 3:
        raise ValueError("at least three fused surface points are required")
    xy = vertices[:, :2]
    centered = xy - np.mean(xy, axis=0)
    if np.linalg.matrix_rank(centered, tol=config.surface_resolution_m * 1e-6) < 2:
        raise ValueError("surface points are collinear in XY and cannot form a TIN")
    support_tree = cKDTree(xy)
    spacing = _local_spacing(xy, config.surface_resolution_m)
    origin = np.min(xy, axis=0)
    tile_indices = np.floor((xy - origin) / config.tin_tile_size_m).astype(np.int64)
    bins: dict[tuple[int, int], np.ndarray] = {}
    unique_tiles, inverse_tiles = np.unique(tile_indices, axis=0, return_inverse=True)
    tile_order = np.argsort(inverse_tiles, kind="stable")
    tile_counts = np.bincount(inverse_tiles, minlength=len(unique_tiles))
    tile_offsets = np.concatenate(([0], np.cumsum(tile_counts)))
    for index, key in enumerate(unique_tiles):
        item = (int(key[0]), int(key[1]))
        bins[item] = tile_order[tile_offsets[index] : tile_offsets[index + 1]]
    rejection = {
        "degenerate": 0,
        "slope": 0,
        "edge": 0,
        "circumradius": 0,
        "support": 0,
    }
    accepted: list[np.ndarray] = []
    raw_simplices = 0
    candidate_faces = 0
    qhull_failures = 0
    overlap = config.max_triangle_edge_m
    area_epsilon = max(1e-12, config.surface_resolution_m**2 * 1e-6)

    for tile_key in sorted(bins):
        neighbors = [
            bins.get((tile_key[0] + dx, tile_key[1] + dy), np.empty(0, dtype=np.int64))
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]
        candidate_indices = np.concatenate(neighbors)
        low = origin + np.asarray(tile_key) * config.tin_tile_size_m
        high = low + config.tin_tile_size_m
        values = xy[candidate_indices]
        in_overlap = np.all(values >= low - overlap, axis=1) & np.all(
            values <= high + overlap, axis=1
        )
        candidate_indices = candidate_indices[in_overlap]
        if len(candidate_indices) < 3:
            continue
        tile_xy = xy[candidate_indices]
        if np.linalg.matrix_rank(tile_xy - np.mean(tile_xy, axis=0)) < 2:
            qhull_failures += 1
            continue
        try:
            triangulation = Delaunay(tile_xy, qhull_options="Qbb Qc Qz Q12")
        except QhullError:
            qhull_failures += 1
            continue
        mapped = candidate_indices[np.asarray(triangulation.simplices, dtype=np.int64)]
        raw_simplices += len(mapped)
        centroids = np.mean(xy[mapped], axis=1)
        owners = np.floor((centroids - origin) / config.tin_tile_size_m).astype(
            np.int64
        )
        mapped = mapped[np.all(owners == np.asarray(tile_key), axis=1)]
        if not len(mapped):
            continue
        candidate_faces += len(mapped)
        triangle_xy = xy[mapped]
        twice_area = np.abs(
            (triangle_xy[:, 1, 0] - triangle_xy[:, 0, 0])
            * (triangle_xy[:, 2, 1] - triangle_xy[:, 0, 1])
            - (triangle_xy[:, 1, 1] - triangle_xy[:, 0, 1])
            * (triangle_xy[:, 2, 0] - triangle_xy[:, 0, 0])
        )
        valid = twice_area > 2.0 * area_epsilon
        rejection["degenerate"] += int(np.count_nonzero(~valid))

        triangle_xyz = vertices[mapped]
        normals = np.cross(
            triangle_xyz[:, 1] - triangle_xyz[:, 0],
            triangle_xyz[:, 2] - triangle_xyz[:, 0],
        )
        slopes = np.degrees(
            np.arctan2(np.linalg.norm(normals[:, :2], axis=1), np.abs(normals[:, 2]))
        )
        keep = slopes <= config.max_slope_deg
        rejection["slope"] += int(np.count_nonzero(valid & ~keep))
        valid &= keep

        edges_xy = np.stack(
            (
                triangle_xy[:, 1] - triangle_xy[:, 0],
                triangle_xy[:, 2] - triangle_xy[:, 1],
                triangle_xy[:, 0] - triangle_xy[:, 2],
            ),
            axis=1,
        )
        edge_lengths = np.linalg.norm(edges_xy, axis=2)
        local = np.max(spacing[mapped], axis=1)
        edge_limit = np.minimum(config.max_triangle_edge_m, 2.5 * local)
        keep = np.max(edge_lengths, axis=1) <= edge_limit
        rejection["edge"] += int(np.count_nonzero(valid & ~keep))
        valid &= keep

        edge_a, edge_b, edge_c = edge_lengths.T
        circumradius = edge_a * edge_b * edge_c / np.maximum(2.0 * twice_area, 1e-15)
        keep = circumradius <= 1.5 * local
        rejection["circumradius"] += int(np.count_nonzero(valid & ~keep))
        valid &= keep

        support_samples = np.stack(
            (
                np.mean(triangle_xy, axis=1),
                (triangle_xy[:, 0] + triangle_xy[:, 1]) * 0.5,
                (triangle_xy[:, 1] + triangle_xy[:, 2]) * 0.5,
                (triangle_xy[:, 2] + triangle_xy[:, 0]) * 0.5,
            ),
            axis=1,
        )
        support_distances, _ = support_tree.query(support_samples.reshape(-1, 2), k=1)
        support_distances = support_distances.reshape(-1, 4)
        keep = np.max(support_distances, axis=1) <= 1.5 * local
        rejection["support"] += int(np.count_nonzero(valid & ~keep))
        valid &= keep
        selected = mapped[valid].copy()
        selected_normals = normals[valid]
        downward = selected_normals[:, 2] < 0
        selected[downward, 1], selected[downward, 2] = (
            selected[downward, 2].copy(),
            selected[downward, 1].copy(),
        )
        if len(selected):
            accepted.append(selected)

    if not accepted:
        raise RuntimeError("no triangles survived conservative surface clipping")
    faces = np.concatenate(accepted).astype(np.int64, copy=False)
    canonical = np.sort(faces, axis=1)
    _, unique_indices = np.unique(canonical, axis=0, return_index=True)
    duplicate_faces = len(faces) - len(unique_indices)
    faces = faces[np.sort(unique_indices)]
    if not len(faces):
        raise RuntimeError("surface triangulation produced no unique faces")
    components = _component_statistics(vertices, faces)
    return faces, {
        "tile_count": len(bins),
        "qhull_failure_count": qhull_failures,
        "raw_simplex_count": raw_simplices,
        "candidate_face_count": candidate_faces,
        "face_count": len(faces),
        "duplicate_face_count": duplicate_faces,
        "rejected_faces": rejection,
        "local_spacing_m": {
            "minimum": float(np.min(spacing)),
            "median": float(np.median(spacing)),
            "maximum": float(np.max(spacing)),
        },
        "components": components,
    }


def _export_surface_glb(
    path: Path, points: np.ndarray, colors: np.ndarray, faces: np.ndarray
) -> None:
    import trimesh

    mesh = trimesh.Trimesh(
        vertices=np.asarray(points, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    alpha = np.full((len(colors), 1), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = np.column_stack((colors, alpha))
    _ = mesh.vertex_normals
    scene = trimesh.Scene()
    scene.add_geometry(mesh, node_name="surface", geom_name="surface")
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path, file_type="glb")


def build_surface_tin(
    run_dir: Path, config: Optional[TinConfig] = None, *, overwrite: bool = False
) -> SurfaceOutputs:
    """Build a fused point cloud and conservative TIN from saved surface masks."""

    config = config or TinConfig()
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    if (
        manifest.get("stages", {}).get("surface_segment", {}).get("status")
        != "complete"
    ):
        raise RuntimeError("surface segment stage must complete before TIN generation")
    outputs = _surface_outputs(run_dir)
    artifacts_exist = all(
        path.is_file() for path in (outputs.point_cloud, outputs.mesh, outputs.metadata)
    )
    if (
        stage_is_current(manifest, "surface_tin", config.manifest_value())
        and artifacts_exist
        and not overwrite
    ):
        return outputs
    stage = manifest.get("stages", {}).get("surface_tin")
    if (
        stage
        and stage.get("config_sha256") != config_digest(config.manifest_value())
        and not overwrite
    ):
        raise RuntimeError("surface TIN configuration changed; pass --overwrite")
    if overwrite:
        _clear_surface_build(run_dir, manifest)
    outputs = _surface_outputs(run_dir)
    outputs.mesh.parent.mkdir(parents=True, exist_ok=True)
    if outputs.mesh.exists():
        outputs.mesh.unlink()
    update_stage(manifest, "surface_tin", config.manifest_value(), status="running")
    atomic_write_json(manifest_path, manifest)
    report: dict[str, Any] = {
        "schema": SURFACE_SCHEMA,
        "status": "running",
        "prompts": list(manifest.get("surface_prompts", [])),
        "coordinate_system": manifest["coordinate_system"],
        "config": config.manifest_value(),
    }
    try:
        point_set = collect_surface_points(run_dir, manifest, config)
        report["point_collection"] = point_set.statistics
        write_binary_ply(outputs.point_cloud, point_set.points, point_set.colors)
        report["point_cloud"] = relative_artifact(run_dir, outputs.point_cloud)
        if len(point_set.points):
            report["bounds_world"] = {
                "minimum": np.min(point_set.points, axis=0).astype(float).tolist(),
                "maximum": np.max(point_set.points, axis=0).astype(float).tolist(),
            }
        faces, triangulation = triangulate_surface(point_set.points, config)
        report["triangulation"] = triangulation
        _export_surface_glb(outputs.mesh, point_set.points, point_set.colors, faces)
        report.update(
            {
                "status": "complete",
                "mesh": relative_artifact(run_dir, outputs.mesh),
                "world_from_glb": np.eye(4, dtype=float).tolist(),
            }
        )
        atomic_write_json(outputs.metadata, report)
        manifest = read_route_manifest(manifest_path)
        manifest.setdefault("outputs", {}).update(
            {
                "surface_point_cloud": relative_artifact(run_dir, outputs.point_cloud),
                "surface_glb": relative_artifact(run_dir, outputs.mesh),
                "surface_metadata": relative_artifact(run_dir, outputs.metadata),
            }
        )
        manifest.setdefault("software", {}).update(
            software_versions(
                {
                    "numpy": "numpy",
                    "scipy": "scipy",
                    "trimesh": "trimesh",
                }
            )
        )
        update_stage(
            manifest, "surface_tin", config.manifest_value(), status="complete"
        )
        atomic_write_json(manifest_path, manifest)
        return outputs
    except Exception as exc:
        if outputs.mesh.exists():
            outputs.mesh.unlink()
        report["status"] = "failed"
        report["error"] = (
            f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:1000]}"
        )
        atomic_write_json(outputs.metadata, report)
        manifest = read_route_manifest(manifest_path)
        destination = manifest.setdefault("outputs", {})
        destination.pop("surface_glb", None)
        if outputs.point_cloud.exists():
            destination["surface_point_cloud"] = relative_artifact(
                run_dir, outputs.point_cloud
            )
        destination["surface_metadata"] = relative_artifact(run_dir, outputs.metadata)
        update_stage(
            manifest,
            "surface_tin",
            config.manifest_value(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(manifest_path, manifest)
        raise


def generate_surface_route(
    source: Path,
    run_dir: Path,
    *,
    segment_config: SurfaceSegmentConfig,
    extract_config: Optional[ExtractConfig] = None,
    tin_config: Optional[TinConfig] = None,
    metadata: Optional[Path] = None,
    overwrite: bool = False,
) -> SurfaceOutputs:
    """Run extraction, surface segmentation, and TIN generation end to end."""

    extract_route(
        source,
        run_dir,
        metadata=metadata,
        config=extract_config or ExtractConfig(keyframe_distance_m=1.0),
        overwrite=overwrite,
    )
    segment_surface_route(run_dir, segment_config, overwrite=overwrite)
    return build_surface_tin(run_dir, tin_config, overwrite=overwrite)
