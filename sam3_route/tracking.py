from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from sam3_masking.artifacts import load_mask_manifest, read_manifest_document

from .artifacts import (
    TRACKS_SCHEMA,
    artifact_path,
    atomic_write_json,
    config_digest,
    read_route_manifest,
    relative_artifact,
    stage_is_current,
    update_stage,
)
from .geometry import points_from_range, transform_pointmap_per_column


@dataclass(frozen=True)
class SegmentConfig:
    prompts: tuple[str, ...]
    sam3_model_dir: str
    sam3_executable: str = "sam3-mask-route"
    sam3_device: str = "auto"
    sam3_dtype: str = "auto"
    score_threshold: float = 0.5
    mask_threshold: float = 0.5
    min_range_points: int = 10
    dynamic_min_speed_mps: float = 0.5
    max_mesh_range_m: Optional[float] = 30.0

    def __post_init__(self) -> None:
        if not self.prompts or any(not value.strip() for value in self.prompts):
            raise ValueError("at least one nonempty prompt is required")
        if self.min_range_points < 3:
            raise ValueError("min_range_points must be at least 3")
        if self.dynamic_min_speed_mps <= 0:
            raise ValueError("dynamic_min_speed_mps must be greater than zero")
        if self.max_mesh_range_m is not None and (
            not math.isfinite(self.max_mesh_range_m) or self.max_mesh_range_m <= 0
        ):
            raise ValueError("max_mesh_range_m must be finite and greater than zero")
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


@dataclass
class Observation:
    id: str
    frame_id: str
    scan_index: int
    timestamp_ns: Optional[int]
    prediction_id: str
    prompt: str
    score: float
    mask_path: str
    cleaned_mask_path: str
    points_path: str
    mask_area_px: int
    image_area_px: int
    valid_depth_fraction: float
    inlier_fraction: float
    border_touch: bool
    median_range_m: float
    azimuth_rad: float
    elevation_rad: float
    centroid_world: list[float]
    bbox_min_world: list[float]
    bbox_max_world: list[float]
    extents_world: list[float]
    quality: float = 0.0
    depth_candidate_rank: int = 0
    depth_candidate_count: int = 1

    @property
    def centroid(self) -> np.ndarray:
        return np.asarray(self.centroid_world, dtype=np.float64)

    @property
    def extents(self) -> np.ndarray:
        return np.asarray(self.extents_world, dtype=np.float64)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int64)
        self.sizes = np.ones(size, dtype=np.int64)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != int(self.parent[parent]):
            parent = int(self.parent[parent])
        while value != parent:
            next_value = int(self.parent[value])
            self.parent[value] = parent
            value = next_value
        return parent

    def union(self, first: int, second: int) -> None:
        a, b = self.find(first), self.find(second)
        if a == b:
            return
        if self.sizes[a] < self.sizes[b]:
            a, b = b, a
        self.parent[b] = a
        self.sizes[a] += self.sizes[b]


@dataclass(frozen=True)
class MotionRecord:
    state: str
    estimated_speed_mps: float
    maximum_displacement_m: float
    position_uncertainty_m: float
    reason: str

    @property
    def dynamic(self) -> bool:
        return self.state == "dynamic"


@dataclass(frozen=True)
class _DepthComponent:
    mask: np.ndarray
    points: np.ndarray
    radius: float


def _depth_components(
    mask: np.ndarray,
    points_world: np.ndarray,
    *,
    min_points: int,
) -> list[_DepthComponent]:
    """Return all coherent masked range surfaces, largest first."""

    mask = np.asarray(mask, dtype=bool)
    points = np.asarray(points_world, dtype=np.float64)
    if points.shape != (*mask.shape, 3):
        raise ValueError("points_world must align with the mask")
    valid = mask & np.all(np.isfinite(points), axis=-1)
    flat_indices = np.flatnonzero(valid)
    if flat_indices.size < min_points:
        return []
    label_for_flat = np.full(mask.size, -1, dtype=np.int64)
    label_for_flat[flat_indices] = np.arange(flat_indices.size)
    height, width = mask.shape
    candidate_pairs: list[tuple[int, int, float]] = []
    spacings: list[float] = []

    for row_offset, col_offset in ((1, 0), (0, 1)):
        if row_offset:
            pairs = valid[:-1] & valid[1:]
            rows, cols = np.nonzero(pairs)
            next_rows, next_cols = rows + 1, cols
        else:
            pairs = valid & np.roll(valid, -1, axis=1)
            rows, cols = np.nonzero(pairs)
            next_rows, next_cols = rows, (cols + 1) % width
        for row, col, next_row, next_col in zip(rows, cols, next_rows, next_cols):
            distance = float(np.linalg.norm(points[row, col] - points[next_row, next_col]))
            if math.isfinite(distance) and distance > 0:
                spacings.append(distance)
                candidate_pairs.append(
                    (
                        int(label_for_flat[row * width + col]),
                        int(label_for_flat[next_row * width + next_col]),
                        distance,
                    )
                )
    spacing = float(np.median(spacings)) if spacings else 0.0625
    radius = float(np.clip(spacing * 4.0, 0.05, 0.75))
    groups = _UnionFind(flat_indices.size)
    for first, second, distance in candidate_pairs:
        if first >= 0 and second >= 0 and distance <= radius:
            groups.union(first, second)
    roots = np.asarray([groups.find(index) for index in range(flat_indices.size)])
    components: list[_DepthComponent] = []
    for root in np.unique(roots):
        selected = roots == root
        count = int(np.count_nonzero(selected))
        if count < min_points:
            continue
        clean = np.zeros(mask.size, dtype=bool)
        clean[flat_indices[selected]] = True
        clean = clean.reshape(mask.shape)
        components.append(_DepthComponent(clean, points[clean], radius))
    components.sort(
        key=lambda value: (
            -len(value.points),
            float(np.median(np.linalg.norm(value.points, axis=1))),
            int(np.flatnonzero(value.mask)[0]),
        )
    )
    return components


def dominant_depth_component(
    mask: np.ndarray,
    points_world: np.ndarray,
    *,
    min_points: int = 10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Keep the largest range-coherent 4-connected component, including seam wrap."""

    components = _depth_components(mask, points_world, min_points=min_points)
    if not components:
        return np.zeros_like(mask, dtype=bool), np.empty((0, 3), dtype=np.float64), 0.0
    selected = components[0]
    return selected.mask, selected.points, selected.radius


def _circular_mean(values: np.ndarray) -> float:
    return float(math.atan2(np.mean(np.sin(values)), np.mean(np.cos(values))))


def _observations_from_prediction(
    run_dir: Path,
    frame: dict[str, Any],
    prediction: Any,
    points_sensor: np.ndarray,
    points_world: np.ndarray,
    *,
    min_points: int,
) -> list[Observation]:
    components = _depth_components(prediction.mask, points_world, min_points=min_points)
    if not components:
        return []
    mask_manifest = artifact_path(run_dir, frame["mask_manifest"])
    document = read_manifest_document(mask_manifest)
    record = next(item for item in document["predictions"] if item["id"] == prediction.id)
    mask_path = (mask_manifest.parent / record["mask"]).resolve()
    mask_area = int(np.count_nonzero(prediction.mask))
    valid_depth = int(np.count_nonzero(prediction.mask & np.all(np.isfinite(points_world), axis=-1)))
    border = bool(
        np.any(prediction.mask[:2])
        or np.any(prediction.mask[-2:])
        or np.any(prediction.mask[:, :2])
        or np.any(prediction.mask[:, -2:])
    )
    observations: list[Observation] = []
    for rank, component in enumerate(components):
        clean_mask = component.mask
        selected_world = component.points
        selected_sensor = points_sensor[clean_mask]
        observation_id = f"{frame['id']}-{prediction.id}-c{rank:03d}"
        points_path = run_dir / "observations" / f"{observation_id}.npz"
        points_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            points_path,
            points_world=selected_world.astype(np.float32),
            points_sensor=selected_sensor.astype(np.float32),
        )
        cleaned_mask_path = run_dir / "observations" / f"{observation_id}-cleaned.png"
        Image.fromarray(clean_mask.astype(np.uint8) * 255).save(
            cleaned_mask_path, format="PNG"
        )
        low, high = np.quantile(selected_world, [0.10, 0.90], axis=0)
        sensor_norm = np.linalg.norm(selected_sensor, axis=1)
        horizontal = np.linalg.norm(selected_sensor[:, :2], axis=1)
        azimuths = np.arctan2(selected_sensor[:, 1], selected_sensor[:, 0])
        elevations = np.arctan2(selected_sensor[:, 2], horizontal)
        observations.append(
            Observation(
                id=observation_id,
                frame_id=frame["id"],
                scan_index=int(frame["scan_index"]),
                timestamp_ns=frame.get("timestamp_ns"),
                prediction_id=prediction.id,
                prompt=prediction.prompt.strip(),
                score=float(prediction.score),
                mask_path=relative_artifact(run_dir, mask_path),
                cleaned_mask_path=relative_artifact(run_dir, cleaned_mask_path),
                points_path=relative_artifact(run_dir, points_path),
                mask_area_px=mask_area,
                image_area_px=int(prediction.mask.size),
                valid_depth_fraction=float(valid_depth / max(mask_area, 1)),
                inlier_fraction=float(len(selected_world) / max(valid_depth, 1)),
                border_touch=border,
                median_range_m=float(np.median(sensor_norm)),
                azimuth_rad=_circular_mean(azimuths),
                elevation_rad=float(np.median(elevations)),
                centroid_world=np.median(selected_world, axis=0).tolist(),
                bbox_min_world=low.tolist(),
                bbox_max_world=high.tolist(),
                extents_world=(high - low).tolist(),
                depth_candidate_rank=rank,
                depth_candidate_count=len(components),
            )
        )
    return observations


def collect_observations(
    run_dir: Path,
    manifest: dict[str, Any],
    *,
    min_points: int,
    dynamic_min_speed_mps: float = 0.5,
) -> tuple[list[Observation], list[dict[str, Any]]]:
    calibration_document = json.loads(
        artifact_path(run_dir, manifest["calibration"]).read_text(encoding="utf-8")
    )
    calibration_path = artifact_path(run_dir, calibration_document["arrays"])
    with np.load(calibration_path) as calibration:
        ray_direction = calibration["ray_direction"]
        ray_origin = calibration["ray_origin"]
        sensor_to_body_matrix = calibration["sensor_to_body"]
    candidate_groups: list[list[Observation]] = []
    rejected: list[dict[str, Any]] = []
    for frame in manifest["keyframes"]:
        if not frame.get("mask_manifest"):
            raise RuntimeError(f"keyframe {frame['id']} has no SAM3 mask manifest")
        with np.load(artifact_path(run_dir, frame["geometry"])) as geometry:
            range_mm = geometry["range_mm"]
            poses = geometry["body_to_world"]
        points_sensor = points_from_range(range_mm, ray_direction, ray_origin)
        points_world = transform_pointmap_per_column(
            points_sensor, poses, sensor_to_body_matrix
        )
        mask_frame = load_mask_manifest(artifact_path(run_dir, frame["mask_manifest"]))
        for prediction in mask_frame.predictions:
            candidates = _observations_from_prediction(
                run_dir,
                frame,
                prediction,
                points_sensor,
                points_world,
                min_points=min_points,
            )
            if not candidates:
                rejected.append(
                    {
                        "frame_id": frame["id"],
                        "prediction_id": prediction.id,
                        "prompt": prediction.prompt,
                        "reason": "insufficient coherent range points",
                    }
                )
            else:
                candidate_groups.append(candidates)
    observations = _select_consistent_depth_candidates(
        candidate_groups,
        dynamic_min_speed_mps=dynamic_min_speed_mps,
    )
    return observations, rejected


def _aabb_iou(first: Observation, second: Observation, expansion: float = 0.75) -> float:
    first_low = np.asarray(first.bbox_min_world) - expansion
    first_high = np.asarray(first.bbox_max_world) + expansion
    second_low = np.asarray(second.bbox_min_world) - expansion
    second_high = np.asarray(second.bbox_max_world) + expansion
    overlap = np.maximum(0.0, np.minimum(first_high, second_high) - np.maximum(first_low, second_low))
    intersection = float(np.prod(overlap))
    first_volume = float(np.prod(np.maximum(first_high - first_low, 1e-6)))
    second_volume = float(np.prod(np.maximum(second_high - second_low, 1e-6)))
    return intersection / max(first_volume + second_volume - intersection, 1e-6)


def _track_centroid(track: list[Observation]) -> np.ndarray:
    return np.median(np.asarray([value.centroid for value in track]), axis=0)


def _association_cost(track: list[Observation], observation: Observation) -> float:
    previous = track[-1]
    distance = float(np.linalg.norm(_track_centroid(track) - observation.centroid))
    first_diagonal = float(np.linalg.norm(np.median([item.extents for item in track], axis=0)))
    second_diagonal = float(np.linalg.norm(observation.extents))
    gate = max(1.5, 0.75 * (first_diagonal + second_diagonal))
    if distance > gate:
        return math.inf
    iou = _aabb_iou(previous, observation)
    size_penalty = min(
        1.5,
        abs(math.log(max(second_diagonal, 1e-3) / max(first_diagonal, 1e-3))),
    )
    return distance / gate + 0.5 * (1.0 - iou) + 0.25 * abs(
        size_penalty
    )


def associate_observations(observations: Sequence[Observation]) -> list[list[Observation]]:
    tracks: list[list[Observation]] = []
    frames = sorted({value.scan_index for value in observations})
    for scan_index in frames:
        frame_values = sorted(
            (value for value in observations if value.scan_index == scan_index),
            key=lambda value: (value.prompt.casefold(), value.id),
        )
        for prompt in sorted({value.prompt.casefold() for value in frame_values}):
            current = [value for value in frame_values if value.prompt.casefold() == prompt]
            candidates = [
                index
                for index, track in enumerate(tracks)
                if track[0].prompt.casefold() == prompt
            ]
            assigned_observations: set[int] = set()
            if candidates:
                costs = np.full((len(candidates), len(current)), 1e6, dtype=np.float64)
                for row, track_index in enumerate(candidates):
                    for column, observation in enumerate(current):
                        cost = _association_cost(tracks[track_index], observation)
                        if math.isfinite(cost):
                            costs[row, column] = cost
                rows, columns = linear_sum_assignment(costs)
                for row, column in zip(rows, columns):
                    if costs[row, column] <= 1.75:
                        tracks[candidates[int(row)]].append(current[int(column)])
                        assigned_observations.add(int(column))
            for index, observation in enumerate(current):
                if index not in assigned_observations:
                    tracks.append([observation])
    return _merge_fragmented_tracks(tracks)


def _merge_fragmented_tracks(
    tracks: Sequence[Sequence[Observation]],
) -> list[list[Observation]]:
    """Merge spatially coincident fragments while preserving same-frame objects."""

    merged = [list(track) for track in tracks]
    while True:
        best: Optional[tuple[float, int, int]] = None
        for first_index, first in enumerate(merged):
            first_scans = {value.scan_index for value in first}
            first_centroid = _track_centroid(first)
            first_diagonal = float(
                np.linalg.norm(np.median([value.extents for value in first], axis=0))
            )
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                if first[0].prompt.casefold() != second[0].prompt.casefold():
                    continue
                if first_scans & {value.scan_index for value in second}:
                    continue
                second_centroid = _track_centroid(second)
                second_diagonal = float(
                    np.linalg.norm(np.median([value.extents for value in second], axis=0))
                )
                distance = float(np.linalg.norm(first_centroid - second_centroid))
                gate = max(1.5, 0.5 * (first_diagonal + second_diagonal))
                if distance > gate:
                    continue
                vertical_gate = max(
                    0.75,
                    0.5
                    * (
                        float(np.median([value.extents[2] for value in first]))
                        + float(np.median([value.extents[2] for value in second]))
                    ),
                )
                if abs(first_centroid[2] - second_centroid[2]) > vertical_gate:
                    continue
                cost = distance / gate
                candidate = (cost, first_index, second_index)
                if best is None or candidate < best:
                    best = candidate
        if best is None:
            break
        _, first_index, second_index = best
        merged[first_index].extend(merged.pop(second_index))
        merged[first_index].sort(key=lambda value: (value.scan_index, value.id))
    return merged


def _observation_evidence(value: Observation) -> tuple[float, float, int, str]:
    return (
        0.5 * value.score
        + 0.25 * value.valid_depth_fraction
        + 0.25 * value.inlier_fraction,
        value.score,
        value.mask_area_px,
        value.id,
    )


def deduplicate_frame_observations(
    run_dir: Path, observations: Sequence[Observation]
) -> list[Observation]:
    """Suppress nearly identical same-prompt SAM masks within one keyframe."""

    kept: list[Observation] = []
    mask_cache: dict[str, np.ndarray] = {}

    def load_mask(value: Observation) -> np.ndarray:
        cached = mask_cache.get(value.mask_path)
        if cached is None:
            with Image.open(artifact_path(run_dir, value.mask_path)) as image:
                cached = np.asarray(image) > 0
            mask_cache[value.mask_path] = cached
        return cached

    groups: dict[tuple[int, str], list[Observation]] = {}
    for value in observations:
        groups.setdefault((value.scan_index, value.prompt.casefold()), []).append(value)
    for key in sorted(groups):
        accepted: list[Observation] = []
        for value in sorted(groups[key], key=_observation_evidence, reverse=True):
            mask = load_mask(value)
            duplicate = False
            for prior in accepted:
                prior_mask = load_mask(prior)
                intersection = int(np.count_nonzero(mask & prior_mask))
                if intersection == 0:
                    continue
                first_area = int(np.count_nonzero(mask))
                second_area = int(np.count_nonzero(prior_mask))
                union = first_area + second_area - intersection
                iou = intersection / max(union, 1)
                containment = intersection / max(min(first_area, second_area), 1)
                if iou >= 0.80 or containment >= 0.92:
                    duplicate = True
                    break
            if not duplicate:
                accepted.append(value)
        kept.extend(accepted)
    return sorted(kept, key=lambda value: (value.scan_index, value.id))


def _motion_record(track: Sequence[Observation], minimum_speed: float) -> MotionRecord:
    values = sorted(track, key=lambda value: (value.timestamp_ns or -1, value.id))
    ranges = np.asarray([value.median_range_m for value in values], dtype=np.float64)
    diagonals = np.asarray([np.linalg.norm(value.extents) for value in values])
    uncertainty = float(
        max(
            0.5,
            0.015 * float(np.median(ranges)),
            0.15 * float(np.median(diagonals)),
        )
    )
    if len(values) < 2:
        return MotionRecord(
            "unconfirmed",
            0.0,
            0.0,
            uncertainty,
            "one observation cannot establish a static world position",
        )
    if any(value.timestamp_ns is None for value in values):
        return MotionRecord(
            "unconfirmed",
            0.0,
            0.0,
            uncertainty,
            "timestamps are required to distinguish motion from measurement noise",
        )
    times = np.asarray([value.timestamp_ns for value in values], dtype=np.float64) / 1e9
    times -= times[0]
    if times[-1] < 0.5:
        return MotionRecord(
            "unconfirmed",
            0.0,
            0.0,
            uncertainty,
            "observation baseline is shorter than 0.5 seconds",
        )
    points = np.asarray([value.centroid for value in values])
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    maximum_displacement = float(np.max(pairwise))
    design = np.column_stack((times, np.ones_like(times)))
    coefficients, _, _, _ = np.linalg.lstsq(design, points, rcond=None)
    fitted_speed = float(np.linalg.norm(coefficients[0]))
    maximum_pair_speed = 0.0
    for first in range(len(values)):
        for second in range(first + 1, len(values)):
            elapsed = float(times[second] - times[first])
            if elapsed >= 0.5:
                maximum_pair_speed = max(
                    maximum_pair_speed,
                    float(pairwise[first, second]) / elapsed,
                )
    estimated_speed = max(fitted_speed, maximum_pair_speed)
    displacement_gate = 2.0 * uncertainty
    if maximum_displacement > displacement_gate and estimated_speed >= minimum_speed:
        return MotionRecord(
            "dynamic",
            estimated_speed,
            maximum_displacement,
            uncertainty,
            "world-centroid displacement exceeds range-dependent static uncertainty",
        )
    if maximum_displacement <= displacement_gate:
        return MotionRecord(
            "confirmed_static",
            fitted_speed,
            maximum_displacement,
            uncertainty,
            "world centroids remain within range-dependent static uncertainty",
        )
    return MotionRecord(
        "unconfirmed",
        estimated_speed,
        maximum_displacement,
        uncertainty,
        "displacement is significant but observed speed is below the motion threshold",
    )


def _dynamic_record(track: Sequence[Observation], minimum_speed: float) -> tuple[bool, float]:
    """Backward-compatible summary used by callers and older tests."""

    motion = _motion_record(track, minimum_speed)
    return motion.dynamic, motion.estimated_speed_mps


def _select_consistent_depth_candidates(
    candidate_groups: Sequence[Sequence[Observation]],
    *,
    dynamic_min_speed_mps: float,
) -> list[Observation]:
    """Use stable multi-frame tracks to disambiguate foreground range leakage."""

    if not candidate_groups:
        return []
    primary = [group[0] for group in candidate_groups]
    initial_tracks = _merge_fragmented_tracks(associate_observations(primary))
    anchors = [
        track
        for track in initial_tracks
        if _motion_record(track, dynamic_min_speed_mps).state == "confirmed_static"
    ]
    selected: list[Observation] = []
    for group in candidate_groups:
        best = group[0]
        best_match: Optional[tuple[float, int, int]] = None
        for candidate_index, candidate in enumerate(group):
            candidate_diagonal = float(np.linalg.norm(candidate.extents))
            for anchor_index, anchor in enumerate(anchors):
                if candidate.prompt.casefold() != anchor[0].prompt.casefold():
                    continue
                anchor_centroid = _track_centroid(anchor)
                anchor_diagonal = float(
                    np.linalg.norm(np.median([value.extents for value in anchor], axis=0))
                )
                distance = float(np.linalg.norm(candidate.centroid - anchor_centroid))
                gate = max(
                    1.5,
                    0.5 * (candidate_diagonal + anchor_diagonal),
                    0.025 * candidate.median_range_m,
                )
                if distance > gate:
                    continue
                match = (distance / gate, candidate_index, anchor_index)
                if best_match is None or match < best_match:
                    best_match = match
                    best = candidate
        selected.append(best)
    return selected


def _select_observation(values: Sequence[Observation]) -> Observation:
    """Score track views while penalizing incomplete or depth-incoherent masks."""

    max_area = max(value.mask_area_px for value in values)
    for value in values:
        value.quality = (
            0.45 * value.score
            + 0.20 * value.mask_area_px / max(max_area, 1)
            + 0.20 * value.valid_depth_fraction
            + 0.15 * value.inlier_fraction
            - (0.25 if value.border_touch else 0.0)
        )
    return max(
        values,
        key=lambda value: (
            value.quality,
            value.score,
            value.mask_area_px,
            -value.median_range_m,
        ),
    )


def _combine_observation_points(
    run_dir: Path, values: Sequence[Observation]
) -> np.ndarray:
    points = []
    for value in values:
        with np.load(artifact_path(run_dir, value.points_path)) as item:
            points.append(item["points_world"])
    combined = np.concatenate(points, axis=0).astype(np.float32)
    voxel_keys = np.floor(combined / 0.05).astype(np.int64)
    _, unique_indices = np.unique(voxel_keys, axis=0, return_index=True)
    return combined[np.sort(unique_indices)]


def _range_eligible_observations(
    values: Sequence[Observation], max_mesh_range_m: Optional[float]
) -> list[Observation]:
    if max_mesh_range_m is None:
        return list(values)
    return [value for value in values if value.median_range_m <= max_mesh_range_m]


def _track_status(motion_state: str, has_range_eligible_observation: bool) -> str:
    if motion_state == "dynamic":
        return "dynamic_skipped"
    if motion_state == "unconfirmed":
        return "unconfirmed_skipped"
    if motion_state == "confirmed_static":
        return "pending" if has_range_eligible_observation else "range_skipped"
    raise ValueError(f"unsupported motion state {motion_state!r}")


def _track_document(
    run_dir: Path,
    track_id: str,
    values: Sequence[Observation],
    *,
    dynamic_min_speed_mps: float,
    max_mesh_range_m: Optional[float],
) -> dict[str, Any]:
    overall_selected = _select_observation(values)
    range_eligible = _range_eligible_observations(values, max_mesh_range_m)
    selected = _select_observation(range_eligible) if range_eligible else overall_selected
    motion = _motion_record(values, dynamic_min_speed_mps)
    combined = _combine_observation_points(run_dir, values)
    track_dir = run_dir / "tracks" / track_id
    track_dir.mkdir(parents=True, exist_ok=True)
    points_path = track_dir / "points.npz"
    np.savez_compressed(points_path, points_world=combined)
    reconstruction_points_path: Optional[Path] = None
    reconstruction_centroid: Optional[np.ndarray] = None
    if range_eligible:
        reconstruction_points = _combine_observation_points(run_dir, range_eligible)
        reconstruction_points_path = track_dir / "reconstruction-points.npz"
        np.savez_compressed(
            reconstruction_points_path, points_world=reconstruction_points
        )
        reconstruction_centroid = np.median(
            np.asarray([value.centroid for value in range_eligible]), axis=0
        )
    source_frame = next(frame for frame in read_route_manifest(run_dir)["keyframes"] if frame["id"] == selected.frame_id)
    rgb_source = artifact_path(run_dir, source_frame["rgb"])
    mask_source = artifact_path(run_dir, selected.mask_path)
    rgb_target = track_dir / "best_rgb.png"
    mask_target = track_dir / "best_mask.png"
    shutil.copy2(rgb_source, rgb_target)
    shutil.copy2(mask_source, mask_target)
    fused_centroid = np.median(np.asarray([value.centroid for value in values]), axis=0)
    status = _track_status(motion.state, bool(range_eligible))
    minimum_range = min(value.median_range_m for value in values)
    if max_mesh_range_m is None:
        range_reason = "mesh range limit is disabled"
    elif range_eligible:
        range_reason = (
            f"{len(range_eligible)} observation(s) are at or below "
            f"{max_mesh_range_m:g} m"
        )
    else:
        range_reason = (
            f"no observation is at or below {max_mesh_range_m:g} m; "
            f"nearest is {minimum_range:g} m"
        )
    return {
        "id": track_id,
        "prompt": values[0].prompt,
        "status": status,
        "motion_state": motion.state,
        "motion": asdict(motion),
        "dynamic": motion.dynamic,
        "estimated_speed_mps": motion.estimated_speed_mps,
        "centroid_world": fused_centroid.tolist(),
        "observations": [asdict(value) for value in values],
        "selected_observation_id": selected.id,
        "points": relative_artifact(run_dir, points_path),
        "reconstruction_points": (
            relative_artifact(run_dir, reconstruction_points_path)
            if reconstruction_points_path is not None
            else None
        ),
        "reconstruction_centroid_world": (
            reconstruction_centroid.tolist()
            if reconstruction_centroid is not None
            else None
        ),
        "range_gate": {
            "max_mesh_range_m": max_mesh_range_m,
            "minimum_observation_range_m": minimum_range,
            "eligible": bool(range_eligible),
            "eligible_observation_count": len(range_eligible),
            "eligible_observation_ids": [value.id for value in range_eligible],
            "reason": range_reason,
        },
        "best_rgb": relative_artifact(run_dir, rgb_target),
        "best_mask": relative_artifact(run_dir, mask_target),
        "mesh": None,
    }


def build_tracks(
    run_dir: Path, manifest: dict[str, Any], config: SegmentConfig
) -> Path:
    observations, rejected = collect_observations(
        run_dir,
        manifest,
        min_points=config.min_range_points,
        dynamic_min_speed_mps=config.dynamic_min_speed_mps,
    )
    observations = deduplicate_frame_observations(run_dir, observations)
    raw_tracks = associate_observations(observations)
    raw_tracks.sort(
        key=lambda values: (
            values[0].prompt.casefold(),
            *_track_centroid(values).tolist(),
        )
    )
    if (run_dir / "tracks").exists():
        shutil.rmtree(run_dir / "tracks")
    documents = [
        _track_document(
            run_dir,
            f"track-{index:06d}",
            values,
            dynamic_min_speed_mps=config.dynamic_min_speed_mps,
            max_mesh_range_m=config.max_mesh_range_m,
        )
        for index, values in enumerate(raw_tracks, start=1)
    ]
    output = run_dir / "tracks.json"
    atomic_write_json(
        output,
        {
            "schema": TRACKS_SCHEMA,
            "tracks": documents,
            "rejected_observations": rejected,
        },
    )
    return output


def retrack_route(
    run_dir: Path,
    *,
    min_range_points: Optional[int] = None,
    dynamic_min_speed_mps: float = 0.5,
    max_mesh_range_m: Optional[float] = 30.0,
    overwrite: bool = False,
) -> Path:
    """Rebuild geometry/tracks from saved SAM masks without loading either model."""

    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    segment_stage = manifest.get("stages", {}).get("segment", {})
    if segment_stage.get("status") != "complete":
        raise RuntimeError("segment stage must complete before model-free retracking")
    if manifest.get("tracks") and not overwrite:
        raise RuntimeError("tracking artifacts already exist; pass --overwrite to replace them")
    values = dict(segment_stage.get("config") or {})
    if not values:
        raise RuntimeError("segment stage does not contain its resolved configuration")
    values["prompts"] = tuple(values["prompts"])
    values["dynamic_min_speed_mps"] = dynamic_min_speed_mps
    values["max_mesh_range_m"] = max_mesh_range_m
    if min_range_points is not None:
        values["min_range_points"] = min_range_points
    config = SegmentConfig(**values)
    observations_dir = run_dir / "observations"
    if observations_dir.exists():
        shutil.rmtree(observations_dir)
    tracks_path = build_tracks(run_dir, manifest, config)
    for path in (run_dir / "meshes", run_dir / "scene.glb", run_dir / "scene.json"):
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    manifest["tracks"] = relative_artifact(run_dir, tracks_path)
    manifest["prompts"] = list(config.prompts)
    manifest.get("stages", {}).pop("reconstruct", None)
    manifest.setdefault("outputs", {}).pop("scene_glb", None)
    manifest.setdefault("outputs", {}).pop("scene_json", None)
    update_stage(manifest, "segment", config.manifest_value(), status="complete")
    atomic_write_json(manifest_path, manifest)
    return tracks_path


def segment_route(
    run_dir: Path,
    config: SegmentConfig,
    *,
    overwrite: bool = False,
    subprocess_run: Callable[..., Any] = subprocess.run,
) -> Path:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    if manifest.get("stages", {}).get("extract", {}).get("status") != "complete":
        raise RuntimeError("extract stage must complete before segmentation")
    if stage_is_current(manifest, "segment", config.manifest_value()) and not overwrite:
        return artifact_path(run_dir, manifest["tracks"])
    stage = manifest.get("stages", {}).get("segment")
    if (
        stage
        and stage.get("config_sha256") != config_digest(config.manifest_value())
        and not overwrite
    ):
        raise RuntimeError("segment configuration changed; pass --overwrite to replace artifacts")
    for name in ("observations", "tracks", "meshes"):
        directory = run_dir / name
        if directory.exists():
            shutil.rmtree(directory)
    for name in ("tracks.json", "scene.glb", "scene.json"):
        path = run_dir / name
        if path.exists():
            path.unlink()
    manifest["tracks"] = None
    manifest.get("stages", {}).pop("reconstruct", None)
    manifest["outputs"].pop("scene_glb", None)
    manifest["outputs"].pop("scene_json", None)
    update_stage(manifest, "segment", config.manifest_value(), status="running")
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
        ]
        for prompt in config.prompts:
            command.extend(("--prompt", prompt))
        if overwrite:
            command.append("--overwrite")
        completed = subprocess_run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"SAM3 route batch exited with status {completed.returncode}")
        manifest = read_route_manifest(manifest_path)
        tracks_path = build_tracks(run_dir, manifest, config)
        manifest["prompts"] = list(config.prompts)
        manifest["tracks"] = relative_artifact(run_dir, tracks_path)
        update_stage(manifest, "segment", config.manifest_value(), status="complete")
        atomic_write_json(manifest_path, manifest)
        return tracks_path
    except Exception as exc:
        manifest = read_route_manifest(manifest_path)
        update_stage(
            manifest,
            "segment",
            config.manifest_value(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(manifest_path, manifest)
        raise
