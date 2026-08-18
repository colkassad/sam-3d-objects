from __future__ import annotations

import json
import math
import os
import struct
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial import ConvexHull, QhullError

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
from .tracking import (
    DEFAULT_DUPLICATE_TRACK_MAX_CENTROID_M,
    DEFAULT_DUPLICATE_TRACK_MIN_CONTAINMENT,
    DEFAULT_DUPLICATE_TRACK_MIN_SHARED_FRACTION,
    duplicate_track_evidence,
    track_evidence_summary,
)


SCENE_SCHEMA = "ouster-mesh-scene/v1"


@dataclass(frozen=True)
class SceneConfig:
    suppress_overlapping_meshes: bool = True
    mesh_overlap_min_iou: float = 0.35
    mesh_overlap_min_containment: float = 0.75
    mesh_vertical_overlap_min: float = 0.50
    mesh_overlap_resolution_m: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "mesh_overlap_min_iou",
            "mesh_overlap_min_containment",
            "mesh_vertical_overlap_min",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if (
            not math.isfinite(self.mesh_overlap_resolution_m)
            or self.mesh_overlap_resolution_m <= 0
        ):
            raise ValueError(
                "mesh_overlap_resolution_m must be finite and greater than zero"
            )

    def manifest_value(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _SceneCandidate:
    track: dict[str, Any]
    mesh: Any
    transform: np.ndarray
    world_vertices: np.ndarray
    footprint: np.ndarray
    low: np.ndarray
    high: np.ndarray
    quality: dict[str, Any]


def _prediction_for_track(track: dict[str, Any]) -> dict[str, Any]:
    selected = track["selected_observation_id"]
    try:
        return next(value for value in track["observations"] if value["id"] == selected)
    except StopIteration as exc:
        raise ValueError(f"track {track['id']} has no selected observation") from exc


def _load_positioned_geometry(mesh_path: Path) -> tuple[Any, np.ndarray]:
    import trimesh

    scene = trimesh.load_scene(mesh_path)
    nodes = list(scene.graph.nodes_geometry)
    if len(nodes) != 1:
        raise ValueError(f"positioned GLB must contain one mesh node: {mesh_path}")
    transform, geometry_name = scene.graph[nodes[0]]
    mesh = scene.geometry[geometry_name]
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.vertices) == 0:
        raise ValueError(f"positioned GLB has no triangle mesh vertices: {mesh_path}")
    return mesh.copy(), np.asarray(transform, dtype=np.float64)


def _convex_footprint(vertices: np.ndarray) -> np.ndarray:
    xy = np.unique(np.asarray(vertices, dtype=np.float64)[:, :2], axis=0)
    if len(xy) >= 3 and np.linalg.matrix_rank(xy - np.mean(xy, axis=0)) >= 2:
        try:
            polygon = xy[ConvexHull(xy).vertices]
        except QhullError:
            polygon = np.empty((0, 2), dtype=np.float64)
    else:
        polygon = np.empty((0, 2), dtype=np.float64)
    if len(polygon) < 3:
        low = np.min(xy, axis=0)
        high = np.max(xy, axis=0)
        epsilon = 1e-6
        high = np.maximum(high, low + epsilon)
        polygon = np.asarray(
            [low, [high[0], low[1]], high, [low[0], high[1]]], dtype=np.float64
        )
    signed_area = 0.5 * np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    return polygon if signed_area >= 0 else polygon[::-1]


def _inside_convex_polygon(samples: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    result = np.ones(samples.shape[:-1], dtype=bool)
    epsilon = 1e-10
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        cross = edge[0] * (samples[..., 1] - first[1]) - edge[1] * (
            samples[..., 0] - first[0]
        )
        result &= cross >= -epsilon
    return result


def _mesh_overlap_metrics(
    first: _SceneCandidate,
    second: _SceneCandidate,
    *,
    resolution_m: float,
) -> dict[str, Any]:
    overlap = np.maximum(
        0.0,
        np.minimum(first.high, second.high) - np.maximum(first.low, second.low),
    )
    first_height = max(float(first.high[2] - first.low[2]), 1e-9)
    second_height = max(float(second.high[2] - second.low[2]), 1e-9)
    vertical_containment = float(overlap[2] / min(first_height, second_height))
    if overlap[0] <= 0 or overlap[1] <= 0 or vertical_containment <= 0:
        return {
            "footprint_iou": 0.0,
            "footprint_containment": 0.0,
            "vertical_containment": max(0.0, vertical_containment),
            "raster_limited": False,
        }
    low = np.minimum(first.low[:2], second.low[:2]) - resolution_m
    high = np.maximum(first.high[:2], second.high[:2]) + resolution_m
    shape = np.ceil((high - low) / resolution_m).astype(np.int64) + 1
    if int(shape[0]) * int(shape[1]) > 4_000_000:
        return {
            "footprint_iou": 0.0,
            "footprint_containment": 0.0,
            "vertical_containment": vertical_containment,
            "raster_limited": True,
        }
    columns = np.arange(int(shape[0]))
    rows = np.arange(int(shape[1]))
    grid_x, grid_y = np.meshgrid(columns, rows)
    samples = np.stack(
        (
            low[0] + (grid_x + 0.5) * resolution_m,
            low[1] + (grid_y + 0.5) * resolution_m,
        ),
        axis=-1,
    )
    first_mask = _inside_convex_polygon(samples, first.footprint)
    second_mask = _inside_convex_polygon(samples, second.footprint)
    first_area = int(np.count_nonzero(first_mask))
    second_area = int(np.count_nonzero(second_mask))
    intersection = int(np.count_nonzero(first_mask & second_mask))
    union = first_area + second_area - intersection
    return {
        "footprint_iou": float(intersection / max(union, 1)),
        "footprint_containment": float(
            intersection / max(min(first_area, second_area), 1)
        ),
        "vertical_containment": vertical_containment,
        "raster_limited": False,
    }


def _scene_quality(track: dict[str, Any]) -> dict[str, Any]:
    mesh_record = track.get("mesh") or {}
    fit = mesh_record.get("fit") or {}
    accepted = bool(fit.get("accepted"))
    views = fit.get("candidate_views" if accepted else "baseline_views") or []
    score = fit.get("candidate_score" if accepted else "baseline_score")
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = math.inf
    if not math.isfinite(score_value):
        score_value = math.inf
    residuals = [
        float(value["median_depth_residual_m"])
        for value in views
        if math.isfinite(float(value.get("median_depth_residual_m", math.inf)))
    ]
    track_quality = track_evidence_summary(track)
    return {
        "final_ray_objective": None if math.isinf(score_value) else score_value,
        "reliable_view_count": len(views),
        "median_depth_residual_m": (
            float(np.median(residuals)) if residuals else None
        ),
        **track_quality,
    }


def _scene_quality_sort_key(candidate: _SceneCandidate) -> tuple[Any, ...]:
    quality = candidate.quality
    score = quality["final_ray_objective"]
    residual = quality["median_depth_residual_m"]
    return (
        math.inf if score is None else score,
        -quality["reliable_view_count"],
        math.inf if residual is None else residual,
        -quality["median_inlier_fraction"],
        -quality["primary_component_fraction"],
        -quality["observation_count"],
        candidate.track["id"],
    )


def _scene_record(track: dict[str, Any]) -> dict[str, Any]:
    mesh_record = track["mesh"]
    selected = _prediction_for_track(track)
    return {
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
    with path.open("wb") as stream:
        stream.write(struct.pack("<4sII", b"glTF", 2, total_length))
        stream.write(struct.pack("<I4s", len(document), b"JSON"))
        stream.write(document)


def _export_scene_atomic(scene: Any, path: Path, *, empty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp.glb", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        if empty:
            _write_empty_glb(temporary)
        else:
            scene.export(temporary, file_type="glb")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _duplicate_thresholds(tracks_document: dict[str, Any]) -> dict[str, float]:
    values = (tracks_document.get("duplicate_suppression") or {}).get("config") or {}
    return {
        "max_centroid_m": float(
            values.get("max_centroid_m", DEFAULT_DUPLICATE_TRACK_MAX_CENTROID_M)
        ),
        "min_shared_fraction": float(
            values.get(
                "min_shared_fraction", DEFAULT_DUPLICATE_TRACK_MIN_SHARED_FRACTION
            )
        ),
        "min_containment": float(
            values.get("min_containment", DEFAULT_DUPLICATE_TRACK_MIN_CONTAINMENT)
        ),
    }


def compose_scene(
    run_dir: Path,
    config: Optional[SceneConfig] = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Compose positioned object GLBs with conservative overlap suppression."""

    import trimesh

    config = config or SceneConfig()
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    if manifest.get("stages", {}).get("reconstruct", {}).get("status") != "complete":
        raise RuntimeError("reconstruct stage must complete before scene composition")
    scene_glb = run_dir / "scene.glb"
    scene_json = run_dir / "scene.json"
    artifacts_exist = scene_glb.is_file() and scene_json.is_file()
    if (
        stage_is_current(manifest, "scene_compose", config.manifest_value())
        and artifacts_exist
        and not overwrite
    ):
        return scene_glb
    current = manifest.get("stages", {}).get("scene_compose")
    if (
        current
        and current.get("config_sha256") != config_digest(config.manifest_value())
        and not overwrite
    ):
        raise RuntimeError("scene configuration changed; pass --overwrite")
    if not current and artifacts_exist and not overwrite:
        raise RuntimeError("scene artifacts already exist; pass --overwrite to replace them")
    scene_glb.unlink(missing_ok=True)
    scene_json.unlink(missing_ok=True)
    manifest.setdefault("outputs", {}).pop("scene_glb", None)
    manifest.setdefault("outputs", {}).pop("scene_json", None)
    update_stage(manifest, "scene_compose", config.manifest_value(), status="running")
    atomic_write_json(manifest_path, manifest)
    try:
        tracks_path = artifact_path(run_dir, manifest["tracks"])
        tracks_document = read_tracks(tracks_path)
        candidates: list[_SceneCandidate] = []
        for track in tracks_document["tracks"]:
            mesh_record = track.get("mesh") or {}
            if track.get("status") != "ok" or not mesh_record.get("path"):
                continue
            mesh_path = artifact_path(run_dir, mesh_record["path"])
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f"positioned mesh for {track['id']} does not exist: {mesh_path}"
                )
            mesh, transform = _load_positioned_geometry(mesh_path)
            world_vertices = trimesh.transform_points(mesh.vertices, transform)
            if not np.all(np.isfinite(world_vertices)):
                raise ValueError(
                    f"positioned mesh for {track['id']} has non-finite world vertices"
                )
            candidates.append(
                _SceneCandidate(
                    track=track,
                    mesh=mesh,
                    transform=transform,
                    world_vertices=world_vertices,
                    footprint=_convex_footprint(world_vertices),
                    low=np.min(world_vertices, axis=0),
                    high=np.max(world_vertices, axis=0),
                    quality=_scene_quality(track),
                )
            )

        kept: list[_SceneCandidate] = []
        suppressed: list[dict[str, Any]] = []
        duplicate_thresholds = _duplicate_thresholds(tracks_document)
        for candidate in sorted(candidates, key=_scene_quality_sort_key):
            suppression: Optional[dict[str, Any]] = None
            if config.suppress_overlapping_meshes:
                for winner in kept:
                    if candidate.track["prompt"].casefold() != winner.track[
                        "prompt"
                    ].casefold():
                        continue
                    support = duplicate_track_evidence(
                        candidate.track,
                        winner.track,
                        **duplicate_thresholds,
                    )
                    overlap = _mesh_overlap_metrics(
                        candidate,
                        winner,
                        resolution_m=config.mesh_overlap_resolution_m,
                    )
                    mesh_conflict = (
                        (
                            overlap["footprint_iou"] >= config.mesh_overlap_min_iou
                            or overlap["footprint_containment"]
                            >= config.mesh_overlap_min_containment
                        )
                        and overlap["vertical_containment"]
                        >= config.mesh_vertical_overlap_min
                    )
                    reasons = []
                    if support is not None:
                        reasons.append("duplicate_track_support")
                    if mesh_conflict:
                        reasons.append("world_mesh_overlap")
                    if reasons:
                        suppression = {
                            "track_id": candidate.track["id"],
                            "loser_track_id": candidate.track["id"],
                            "winner_track_id": winner.track["id"],
                            "reasons": reasons,
                            "track_support": support,
                            "mesh_overlap": overlap,
                            "winner_quality": winner.quality,
                            "loser_quality": candidate.quality,
                        }
                        break
            if suppression is None:
                kept.append(candidate)
            else:
                suppressed.append(suppression)

        scene = trimesh.Scene()
        records: list[dict[str, Any]] = []
        for candidate in sorted(kept, key=lambda value: value.track["id"]):
            scene.add_geometry(
                candidate.mesh,
                node_name=candidate.track["id"],
                geom_name=candidate.track["id"],
                transform=candidate.transform,
            )
            records.append(_scene_record(candidate.track))
        _export_scene_atomic(scene, scene_glb, empty=not records)
        atomic_write_json(
            scene_json,
            {
                "schema": SCENE_SCHEMA,
                "coordinate_system": manifest["coordinate_system"],
                "meshes": records,
                "suppressed_meshes": suppressed,
                "composition": {
                    "config": config.manifest_value(),
                    "input_mesh_count": len(candidates),
                    "kept_mesh_count": len(records),
                    "suppressed_mesh_count": len(suppressed),
                    "duplicate_support_suppressed_count": sum(
                        "duplicate_track_support" in value["reasons"]
                        for value in suppressed
                    ),
                    "mesh_overlap_suppressed_count": sum(
                        "world_mesh_overlap" in value["reasons"]
                        for value in suppressed
                    ),
                },
                "point_cloud": manifest.get("outputs", {}).get("point_cloud"),
            },
        )
        manifest = read_route_manifest(manifest_path)
        manifest.setdefault("outputs", {})["scene_glb"] = relative_artifact(
            run_dir, scene_glb
        )
        manifest.setdefault("outputs", {})["scene_json"] = relative_artifact(
            run_dir, scene_json
        )
        manifest.setdefault("software", {}).update(
            software_versions({"trimesh": "trimesh", "scipy": "scipy"})
        )
        update_stage(
            manifest, "scene_compose", config.manifest_value(), status="complete"
        )
        atomic_write_json(manifest_path, manifest)
        return scene_glb
    except Exception as exc:
        scene_glb.unlink(missing_ok=True)
        scene_json.unlink(missing_ok=True)
        manifest = read_route_manifest(manifest_path)
        manifest.setdefault("outputs", {}).pop("scene_glb", None)
        manifest.setdefault("outputs", {}).pop("scene_json", None)
        update_stage(
            manifest,
            "scene_compose",
            config.manifest_value(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(manifest_path, manifest)
        raise
