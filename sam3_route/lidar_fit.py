from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.optimize import minimize
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class LidarFitView:
    observation_id: str
    origins_world: np.ndarray
    directions_world: np.ndarray
    ranges_m: np.ndarray
    background_origins_world: np.ndarray
    background_directions_world: np.ndarray
    background_ranges_m: np.ndarray
    ground_points_world: np.ndarray
    sensor_position_world: np.ndarray
    weight: float


def _unit_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.linalg.norm(values, axis=-1)
    valid = np.isfinite(lengths) & (lengths > 1e-8)
    output = np.zeros_like(values, dtype=np.float64)
    output[valid] = values[valid] / lengths[valid, None]
    return output, lengths


def world_rays_from_frame(
    range_mm: np.ndarray,
    ray_direction: np.ndarray,
    ray_origin: np.ndarray,
    body_to_world: np.ndarray,
    sensor_to_body: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pixel world origins, unit directions, and metric ray distances."""

    combined = np.asarray(body_to_world, dtype=np.float64) @ np.asarray(
        sensor_to_body, dtype=np.float64
    )
    rotations = combined[:, :3, :3]
    translations = combined[:, :3, 3]
    origins = (
        np.einsum("hwj,wij->hwi", np.asarray(ray_origin), rotations)
        + translations[None, :, :]
    )
    directions_unscaled = np.einsum(
        "hwj,wij->hwi", np.asarray(ray_direction), rotations
    )
    directions, direction_lengths = _unit_rows(directions_unscaled)
    ranges = np.asarray(range_mm, dtype=np.float64) * 0.001 * direction_lengths
    invalid = (range_mm <= 0) | ~np.isfinite(ranges) | (direction_lengths <= 1e-8)
    ranges[invalid] = np.nan
    return origins, directions, ranges


def _deterministic_pixels(
    mask: np.ndarray, maximum: int
) -> tuple[np.ndarray, np.ndarray]:
    rows, columns = np.nonzero(mask)
    if len(rows) <= maximum:
        return rows, columns
    interior = distance_transform_edt(mask)[rows, columns]
    order = np.lexsort((columns, rows, -interior))
    selected = order[np.linspace(0, len(order) - 1, maximum, dtype=np.int64)]
    return rows[selected], columns[selected]


def make_fit_view(
    *,
    observation_id: str,
    cleaned_mask: np.ndarray,
    raw_mask: np.ndarray,
    origins_world: np.ndarray,
    directions_world: np.ndarray,
    ranges_m: np.ndarray,
    maximum_rays: int,
    weight: float,
) -> LidarFitView | None:
    valid = np.isfinite(ranges_m) & np.all(np.isfinite(origins_world), axis=-1)
    valid &= np.all(np.isfinite(directions_world), axis=-1)
    inside = np.asarray(cleaned_mask, dtype=bool) & valid
    if int(np.count_nonzero(inside)) < 10:
        return None
    rows, columns = _deterministic_pixels(inside, maximum_rays)

    ring_width = max(2, round(min(raw_mask.shape) * 0.015))
    ring = binary_dilation(np.asarray(raw_mask, dtype=bool), iterations=ring_width)
    ring &= ~np.asarray(raw_mask, dtype=bool) & valid
    bg_rows, bg_columns = _deterministic_pixels(ring, max(64, maximum_rays // 4))
    mask_rows = np.nonzero(raw_mask)[0]
    ground_region = ring.copy()
    if len(mask_rows):
        row_grid = np.arange(raw_mask.shape[0])[:, None]
        ground_region &= row_grid >= max(0, int(mask_rows.max()) - 1)
    ground_rows, ground_columns = _deterministic_pixels(
        ground_region, max(100, maximum_rays // 4)
    )
    ground_points = (
        origins_world[ground_rows, ground_columns]
        + directions_world[ground_rows, ground_columns]
        * ranges_m[ground_rows, ground_columns, None]
    )
    sensor_position = np.median(origins_world[rows, columns], axis=0)
    return LidarFitView(
        observation_id=observation_id,
        origins_world=origins_world[rows, columns].astype(np.float64),
        directions_world=directions_world[rows, columns].astype(np.float64),
        ranges_m=ranges_m[rows, columns].astype(np.float64),
        background_origins_world=origins_world[bg_rows, bg_columns].astype(np.float64),
        background_directions_world=directions_world[bg_rows, bg_columns].astype(
            np.float64
        ),
        background_ranges_m=ranges_m[bg_rows, bg_columns].astype(np.float64),
        ground_points_world=ground_points.astype(np.float64),
        sensor_position_world=sensor_position.astype(np.float64),
        weight=float(weight),
    )


def select_diverse_views(
    views: Iterable[LidarFitView], maximum: int
) -> list[LidarFitView]:
    candidates = sorted(views, key=lambda value: (-value.weight, value.observation_id))
    if len(candidates) <= maximum:
        return candidates
    selected = [candidates.pop(0)]
    center = np.median(
        np.concatenate(
            [
                v.origins_world + v.directions_world * v.ranges_m[:, None]
                for v in selected
            ],
            axis=0,
        ),
        axis=0,
    )
    while candidates and len(selected) < maximum:
        selected_directions = [
            (center - item.sensor_position_world)
            / max(np.linalg.norm(center - item.sensor_position_world), 1e-8)
            for item in selected
        ]

        def score(
            item: LidarFitView,
            directions: tuple[np.ndarray, ...] = tuple(selected_directions),
        ) -> tuple[float, str]:
            direction = center - item.sensor_position_world
            direction /= max(np.linalg.norm(direction), 1e-8)
            separation = min(
                math.acos(float(np.clip(direction @ other, -1.0, 1.0)))
                for other in directions
            )
            return item.weight + separation / math.pi, item.observation_id

        choice = max(candidates, key=score)
        candidates.remove(choice)
        selected.append(choice)
    return selected


def _huber(values: np.ndarray, delta: float = 0.15) -> np.ndarray:
    absolute = np.abs(values)
    return np.where(
        absolute <= delta, 0.5 * absolute**2 / delta, absolute - 0.5 * delta
    )


def _raycast(
    vertices_world: np.ndarray, faces: np.ndarray, rays: np.ndarray
) -> np.ndarray:
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(
        o3d.core.Tensor(vertices_world.astype(np.float32)),
        o3d.core.Tensor(np.asarray(faces, dtype=np.uint32)),
    )
    answer = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)))
    return np.asarray(answer["t_hit"].numpy(), dtype=np.float64)


def _translation_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = value
    return matrix


def candidate_transform(
    baseline: np.ndarray,
    local_center: np.ndarray,
    parameters: np.ndarray,
) -> np.ndarray:
    translation = parameters[:3]
    yaw = parameters[3]
    scales = np.exp(parameters[4:7])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    yaw_matrix = np.eye(4, dtype=np.float64)
    yaw_matrix[:3, :3] = np.asarray(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )
    local_scale = np.eye(4, dtype=np.float64)
    local_scale[:3, :3] = np.diag(scales)
    local_scale = (
        _translation_matrix(local_center)
        @ local_scale
        @ _translation_matrix(-local_center)
    )
    center_h = np.append(local_center, 1.0)
    world_center = (baseline @ center_h)[:3]
    world_adjustment = (
        _translation_matrix(translation)
        @ _translation_matrix(world_center)
        @ yaw_matrix
        @ _translation_matrix(-world_center)
    )
    return world_adjustment @ baseline @ local_scale


def _horizontal_support(points: np.ndarray, sensors: np.ndarray) -> dict[str, Any]:
    xy = points[:, :2]
    centered = xy - np.median(xy, axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = np.asarray([vh[0, 0], vh[0, 1], 0.0])
    axis /= max(np.linalg.norm(axis), 1e-8)
    projections = points @ axis
    low, high = (
        float(np.quantile(projections, 0.02)),
        float(np.quantile(projections, 0.98)),
    )
    # Point counts are viewpoint biased. Occupancy makes every supported interval equal.
    bins = np.arange(low, high + 0.1001, 0.10)
    occupied = np.histogram(projections, bins=bins)[0] > 0
    if len(occupied) >= 3:
        occupied = binary_dilation(occupied, iterations=1)
    indices = np.flatnonzero(occupied)
    if len(indices):
        low, high = (
            float(bins[indices[0]]),
            float(bins[min(indices[-1] + 1, len(bins) - 1)]),
        )
    sensor_projection = float(np.median(sensors @ axis))
    near_is_low = abs(sensor_projection - low) <= abs(sensor_projection - high)
    return {
        "axis_world": axis,
        "low_m": low,
        "high_m": high,
        "near_m": low if near_is_low else high,
        "far_m": high if near_is_low else low,
        "near_is_low": near_is_low,
    }


def _fit_ground_plane(points: np.ndarray) -> dict[str, Any]:
    if len(points) < 30:
        return {"accepted": False, "reason": "insufficient points"}
    cutoff = np.quantile(points[:, 2], 0.30)
    candidates = points[points[:, 2] <= cutoff]
    if len(candidates) < 20:
        return {"accepted": False, "reason": "insufficient lower support"}
    center = np.median(candidates, axis=0)
    _, _, vh = np.linalg.svd(candidates - center, full_matrices=False)
    normal = vh[-1]
    if normal[2] < 0:
        normal = -normal
    tilt = math.degrees(math.acos(float(np.clip(normal[2], -1.0, 1.0))))
    residuals = np.abs((candidates - center) @ normal)
    inliers = residuals <= 0.12
    accepted = bool(tilt <= 20.0 and np.count_nonzero(inliers) >= 20)
    return {
        "accepted": accepted,
        "normal_world": normal.tolist(),
        "offset": float(-normal @ center),
        "tilt_deg": tilt,
        "inlier_count": int(np.count_nonzero(inliers)),
        "residual_median_m": float(np.median(residuals[inliers]))
        if np.any(inliers)
        else None,
        "reason": None if accepted else "plane is sparse or inconsistent with world-up",
    }


def _heading_information(
    views: list[LidarFitView],
    support: dict[str, Any],
) -> tuple[np.ndarray, list[LidarFitView], list[dict[str, Any]]]:
    """Find views that expose enough longitudinal support to constrain yaw."""

    support_axis = np.asarray(support["axis_world"], dtype=np.float64)
    track_span = max(abs(float(support["high_m"]) - float(support["low_m"])), 1e-6)
    informative: list[LidarFitView] = []
    records: list[dict[str, Any]] = []
    heading_points: list[np.ndarray] = []
    for view in views:
        points = view.origins_world + view.directions_world * view.ranges_m[:, None]
        projection = points @ support_axis
        span = float(np.quantile(projection, 0.98) - np.quantile(projection, 0.02))
        ratio = span / track_span
        qualifies = bool(len(points) >= 50 and ratio >= 0.45)
        records.append(
            {
                "observation_id": view.observation_id,
                "longitudinal_span_m": span,
                "track_span_fraction": ratio,
                "yaw_informative": qualifies,
            }
        )
        if not qualifies:
            continue
        informative.append(view)
        voxels = np.unique(np.floor(points / 0.05).astype(np.int64), axis=0)
        if len(voxels) > 500:
            voxels = voxels[np.linspace(0, len(voxels) - 1, 500, dtype=np.int64)]
        heading_points.append(voxels.astype(np.float64) * 0.05)
    if not heading_points:
        return support_axis, [], records
    combined = np.concatenate(heading_points, axis=0)
    xy = combined[:, :2]
    tree = cKDTree(xy)
    neighborhood_radius = float(np.clip(track_span * 0.075, 0.15, 0.50))
    local_angles: list[float] = []
    local_weights: list[float] = []
    for point in xy:
        neighbors = tree.query_ball_point(point, neighborhood_radius)
        if len(neighbors) < 5:
            continue
        local = xy[neighbors] - np.mean(xy[neighbors], axis=0)
        eigenvalues, eigenvectors = np.linalg.eigh(local.T @ local)
        anisotropy = float(eigenvalues[1] / max(eigenvalues[0], 1e-9))
        if anisotropy < 2.0:
            continue
        direction = eigenvectors[:, 1]
        local_angles.append(math.atan2(direction[1], direction[0]))
        local_weights.append(min(anisotropy, 20.0))
    if len(local_angles) >= 20:
        # Fourfold angles merge perpendicular faces of the same rectangular
        # footprint without letting a densely sampled face choose the sign.
        moment = sum(
            weight * np.exp(4j * angle)
            for angle, weight in zip(local_angles, local_weights, strict=False)
        )
        tangent_strength = float(abs(moment) / max(sum(local_weights), 1e-9))
        tangent_angle = float(np.angle(moment) / 4.0)
        candidates = (
            np.asarray([math.cos(tangent_angle), math.sin(tangent_angle)]),
            np.asarray([-math.sin(tangent_angle), math.cos(tangent_angle)]),
        )
        axis_xy = max(candidates, key=lambda value: abs(value @ support_axis[:2]))
        estimator = "local_surface_tangents"
    else:
        centered = xy - np.mean(xy, axis=0)
        covariance = centered.T @ centered / max(len(centered) - 1, 1)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        axis_xy = eigenvectors[:, int(np.argmax(eigenvalues))]
        tangent_strength = 0.0
        estimator = "global_pca_fallback"
    axis = np.asarray([axis_xy[0], axis_xy[1], 0.0], dtype=np.float64)
    axis /= max(np.linalg.norm(axis), 1e-8)
    if axis @ support_axis < 0:
        axis = -axis
    for record in records:
        if record["yaw_informative"]:
            record["heading_estimator"] = estimator
            record["local_tangent_strength"] = tangent_strength
            record["neighborhood_radius_m"] = neighborhood_radius
    return axis, informative, records


def _view_metrics(
    vertices_world: np.ndarray,
    faces: np.ndarray,
    view: LidarFitView,
) -> dict[str, float]:
    rays = np.column_stack((view.origins_world, view.directions_world))
    hits = _raycast(vertices_world, faces, rays)
    finite = np.isfinite(hits)
    residuals = hits[finite] - view.ranges_m[finite]
    depth_loss = float(np.mean(_huber(residuals))) if len(residuals) else 1.0
    missing_fraction = float(1.0 - np.mean(finite))
    median_residual = (
        float(np.median(np.abs(residuals))) if len(residuals) else math.inf
    )
    background_penalty = 0.0
    false_background_fraction = 0.0
    if len(view.background_ranges_m):
        background_rays = np.column_stack(
            (view.background_origins_world, view.background_directions_world)
        )
        background_hits = _raycast(vertices_world, faces, background_rays)
        false = np.isfinite(background_hits) & (
            background_hits < view.background_ranges_m - 0.15
        )
        false_background_fraction = float(np.mean(false))
        if np.any(false):
            background_penalty = float(
                np.mean(
                    np.minimum(
                        view.background_ranges_m[false] - background_hits[false] - 0.15,
                        1.0,
                    )
                )
            )
    return {
        "depth_loss": depth_loss,
        "median_depth_residual_m": median_residual,
        "hit_fraction": 1.0 - missing_fraction,
        "missing_fraction": missing_fraction,
        "background_penalty": background_penalty,
        "false_background_fraction": false_background_fraction,
        "ray_count": float(len(hits)),
        "background_ray_count": float(len(view.background_ranges_m)),
    }


def evaluate_mesh_views(
    vertices_local: np.ndarray,
    faces: np.ndarray,
    transform: np.ndarray,
    views: list[LidarFitView],
) -> dict[str, Any]:
    """Evaluate an already-positioned mesh against the supplied sensor views."""

    vertices = np.asarray(vertices_local, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    world = vertices @ matrix[:3, :3].T + matrix[:3, 3]
    metrics = []
    for view in views:
        metric = _view_metrics(world, faces, view)
        metric["observation_id"] = view.observation_id
        metrics.append(metric)
    ray_count = sum(value["ray_count"] for value in metrics)
    background_count = sum(value["background_ray_count"] for value in metrics)
    return {
        "views": metrics,
        "view_count": len(metrics),
        "median_depth_residual_m": (
            float(np.median([value["median_depth_residual_m"] for value in metrics]))
            if metrics
            else math.inf
        ),
        "hit_fraction": (
            float(
                sum(value["hit_fraction"] * value["ray_count"] for value in metrics)
                / ray_count
            )
            if ray_count
            else 0.0
        ),
        "false_background_fraction": (
            float(
                sum(
                    value["false_background_fraction"] * value["background_ray_count"]
                    for value in metrics
                )
                / background_count
            )
            if background_count
            else 0.0
        ),
        "median_range_m": (
            float(np.median(np.concatenate([view.ranges_m for view in views])))
            if views
            else math.inf
        ),
    }


def refine_mesh_with_lidar_rays(
    vertices_local: np.ndarray,
    faces: np.ndarray,
    initial_transform: np.ndarray,
    views: list[LidarFitView],
    target_points_world: np.ndarray,
    *,
    max_axis_scale_change: float = 0.25,
    max_rotation_deg: float = 20.0,
    max_evaluations: int = 160,
    grounded: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    vertices = np.asarray(vertices_local, dtype=np.float64)
    points = np.asarray(target_points_world, dtype=np.float64)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(vertices) < 3 or len(faces) < 1 or len(points) < 3 or not views:
        return np.asarray(initial_transform), {
            "method": "multi_view_lidar_raycast",
            "accepted": False,
            "reason": "insufficient mesh, LiDAR points, or reliable views",
        }

    local_center = np.median(vertices, axis=0)
    sensors = np.stack([view.sensor_position_world for view in views])
    support = _horizontal_support(points, sensors)
    heading_axis, heading_views, heading_records = _heading_information(views, support)
    ground_points = (
        np.concatenate(
            [
                view.ground_points_world
                for view in views
                if len(view.ground_points_world)
            ],
            axis=0,
        )
        if any(len(view.ground_points_world) for view in views)
        else np.empty((0, 3))
    )
    ground = (
        _fit_ground_plane(ground_points)
        if grounded
        else {"accepted": False, "reason": "disabled"}
    )

    baseline = np.asarray(initial_transform, dtype=np.float64)
    max_translation = min(
        2.0,
        max(
            0.5,
            0.25
            * float(
                np.linalg.norm(
                    np.quantile(points, 0.9, axis=0) - np.quantile(points, 0.1, axis=0)
                )
            ),
        ),
    )
    scale_bound = (
        math.log(1.0 - max_axis_scale_change),
        math.log(1.0 + max_axis_scale_change),
    )
    bounds = [
        (-max_translation, max_translation),
        (-max_translation, max_translation),
        (-min(0.5, max_translation), min(0.5, max_translation)),
        (-math.radians(max_rotation_deg), math.radians(max_rotation_deg)),
        scale_bound,
        scale_bound,
        scale_bound,
    ]

    def evaluate(
        parameters: np.ndarray,
        include_metrics: bool = False,
        active_views: list[LidarFitView] | None = None,
        heading_target: float | None = None,
        global_constraints: bool = True,
    ) -> tuple[float, list[dict[str, Any]]]:
        transform = candidate_transform(baseline, local_center, parameters)
        world = vertices @ transform[:3, :3].T + transform[:3, 3]
        metrics: list[dict[str, Any]] = []
        weighted = 0.0
        total_weight = 0.0
        for view in active_views or views:
            metric = _view_metrics(world, faces, view)
            metric["observation_id"] = view.observation_id
            metric["weight"] = view.weight
            metrics.append(metric)
            view_loss = (
                metric["depth_loss"]
                + 0.50 * metric["missing_fraction"]
                + 0.50 * metric["background_penalty"]
            )
            weighted += view.weight * view_loss
            total_weight += view.weight
        loss = weighted / max(total_weight, 1e-8)

        if global_constraints:
            mesh_projection = world @ support["axis_world"]
            mesh_low = float(np.quantile(mesh_projection, 0.02))
            mesh_high = float(np.quantile(mesh_projection, 0.98))
            mesh_near = mesh_low if support["near_is_low"] else mesh_high
            mesh_far = mesh_high if support["near_is_low"] else mesh_low
            loss += 0.50 * float(_huber(np.asarray([mesh_near - support["near_m"]]))[0])
            loss += 0.15 * float(
                _huber(np.asarray([mesh_far - support["far_m"]]), 0.30)[0]
            )
            if ground.get("accepted"):
                normal = np.asarray(ground["normal_world"])
                distances = world @ normal + float(ground["offset"])
                support_distance = float(np.quantile(distances, 0.02))
                loss += 0.50 * float(_huber(np.asarray([support_distance]))[0])
            loss += 0.02 * float(np.sum((parameters[:3] / max_translation) ** 2))
            loss += 0.02 * float(
                (parameters[3] / max(math.radians(max_rotation_deg), 1e-8)) ** 2
            )
            loss += 0.04 * float(np.sum(parameters[4:7] ** 2))
        if heading_target is not None:
            heading_error = float(parameters[3] - heading_target)
            # Depth alone often prefers an end surface of an imperfect mesh.
            # A long-span footprint is the stronger yaw measurement.
            loss += 0.25 * (heading_error / math.radians(5.0)) ** 2
        return float(loss), metrics if include_metrics else []

    zero = np.zeros(7, dtype=np.float64)
    baseline_score, baseline_metrics = evaluate(zero, include_metrics=True)
    # A PCA-heading seed and a near-surface translation seed improve convergence while
    # keeping the original SAM3D semantic front/back orientation.
    base_world = vertices @ baseline[:3, :3].T + baseline[:3, 3]
    footprint_xy = base_world[:, :2] - np.mean(base_world[:, :2], axis=0)
    footprint_values, footprint_vectors = np.linalg.eigh(footprint_xy.T @ footprint_xy)
    footprint_axis = footprint_vectors[:, int(np.argmax(footprint_values))]
    base_axis = np.asarray([footprint_axis[0], footprint_axis[1], 0.0])
    target_axis = np.asarray(heading_axis)
    if base_axis @ target_axis < 0:
        target_axis = -target_axis
    heading_delta = math.atan2(
        base_axis[0] * target_axis[1] - base_axis[1] * target_axis[0],
        base_axis[:2] @ target_axis[:2],
    )
    bounded_heading_delta = float(np.clip(heading_delta, bounds[3][0], bounds[3][1]))
    heading_seed = zero.copy()
    heading_seed[3] = bounded_heading_delta
    projected = base_world @ support["axis_world"]
    mesh_near = float(np.quantile(projected, 0.02 if support["near_is_low"] else 0.98))
    near_seed = heading_seed.copy()
    near_seed[:3] = np.clip(
        (support["near_m"] - mesh_near) * support["axis_world"],
        [item[0] for item in bounds[:3]],
        [item[1] for item in bounds[:3]],
    )
    seeds = [zero, heading_seed, near_seed]
    best_parameters = zero
    best_score = baseline_score
    evaluations = 1
    optimizer_messages: list[str] = []
    polish_grid_count = min(25, max(9, max_evaluations // 6)) if heading_views else 0
    polish_xy_budget = min(27, max(0, max_evaluations // 6)) if heading_views else 0
    general_budget = max(
        len(seeds) * 5,
        max_evaluations - 2 * polish_grid_count - polish_xy_budget - evaluations,
    )
    budget_per_seed = max(5, general_budget // len(seeds))
    for seed in seeds:
        result = minimize(
            lambda value: evaluate(value)[0],
            seed,
            method="Powell",
            bounds=bounds,
            options={"maxfev": budget_per_seed, "xtol": 1e-3, "ftol": 1e-3},
        )
        evaluations += int(result.nfev)
        optimizer_messages.append(str(result.message))
        if np.isfinite(result.fun) and float(result.fun) < best_score:
            best_score = float(result.fun)
            best_parameters = np.asarray(result.x, dtype=np.float64)

    yaw_polish: dict[str, Any] = {
        "performed": False,
        "target_delta_deg": math.degrees(heading_delta),
        "bounded_target_delta_deg": math.degrees(bounded_heading_delta),
        "informative_observation_ids": [
            value.observation_id for value in heading_views
        ],
    }
    if heading_views:
        yaw_before_polish = float(best_parameters[3])
        polish_before, _ = evaluate(
            best_parameters,
            active_views=heading_views,
            heading_target=bounded_heading_delta,
            global_constraints=False,
        )
        minimum_yaw = max(bounds[3][0], bounded_heading_delta - math.radians(5.0))
        maximum_yaw = min(bounds[3][1], bounded_heading_delta + math.radians(5.0))
        yaw_values = np.linspace(minimum_yaw, maximum_yaw, polish_grid_count)
        yaw_values = np.unique(
            np.clip(
                np.append(
                    yaw_values,
                    [best_parameters[3], bounded_heading_delta, 0.0],
                ),
                bounds[3][0],
                bounds[3][1],
            )
        )
        polished = best_parameters.copy()
        polished_score = math.inf
        for yaw in yaw_values:
            for reset_xy in (False, True):
                candidate_parameters = best_parameters.copy()
                if reset_xy:
                    candidate_parameters[:2] = 0.0
                candidate_parameters[3] = float(yaw)
                score, _ = evaluate(
                    candidate_parameters,
                    active_views=heading_views,
                    heading_target=bounded_heading_delta,
                    global_constraints=False,
                )
                evaluations += 1
                if score < polished_score:
                    polished_score = score
                    polished = candidate_parameters

        if polish_xy_budget:
            # Ray hits make the XY objective piecewise smooth, so a small,
            # deterministic grid is more reliable than a short Powell run.
            centers = (np.zeros(2), best_parameters[:2].copy())
            coarse_offsets = (-0.10, 0.0, 0.10)
            xy_candidates = {
                (
                    float(np.clip(center[0] + dx, *bounds[0])),
                    float(np.clip(center[1] + dy, *bounds[1])),
                )
                for center in centers
                for dx in coarse_offsets
                for dy in coarse_offsets
            }
            for x, y in sorted(xy_candidates)[:polish_xy_budget]:
                candidate_parameters = polished.copy()
                candidate_parameters[:2] = [x, y]
                score, _ = evaluate(
                    candidate_parameters,
                    active_views=heading_views,
                    heading_target=bounded_heading_delta,
                    global_constraints=False,
                )
                evaluations += 1
                if score < polished_score:
                    polished_score = score
                    polished = candidate_parameters
        best_parameters = polished
        yaw_polish = {
            **yaw_polish,
            "performed": True,
            "score_before": polish_before,
            "score_after": polished_score,
            "yaw_before_deg": math.degrees(yaw_before_polish),
            "yaw_after_deg": math.degrees(float(polished[3])),
            "grid_min_deg": math.degrees(minimum_yaw),
            "grid_max_deg": math.degrees(maximum_yaw),
            "grid_step_deg": math.degrees(maximum_yaw - minimum_yaw)
            / max(polish_grid_count - 1, 1),
        }

    candidate = candidate_transform(baseline, local_center, best_parameters)
    candidate_score, candidate_metrics = evaluate(best_parameters, include_metrics=True)
    informative_ids = {value.observation_id for value in heading_views}
    per_view_acceptance: list[dict[str, Any]] = []
    for baseline_metric, candidate_metric in zip(
        baseline_metrics, candidate_metrics, strict=False
    ):
        informative = baseline_metric["observation_id"] in informative_ids
        depth_tolerance = 0.03 if informative else 0.15
        background_tolerance = 0.03 if informative else 0.10
        depth_ok = bool(
            candidate_metric["median_depth_residual_m"]
            <= baseline_metric["median_depth_residual_m"] + depth_tolerance
        )
        background_ok = bool(
            candidate_metric["false_background_fraction"]
            <= baseline_metric["false_background_fraction"] + background_tolerance
        )
        per_view_acceptance.append(
            {
                "observation_id": baseline_metric["observation_id"],
                "yaw_informative": informative,
                "depth_ok": depth_ok,
                "background_ok": background_ok,
                "accepted": depth_ok and background_ok,
            }
        )
    per_view_ok = all(value["accepted"] for value in per_view_acceptance)
    improvement = (
        (baseline_score - candidate_score) / baseline_score
        if baseline_score > 1e-9
        else 0.0
    )
    bounds_satisfied = bool(
        all(
            lower - 1e-9 <= float(value) <= upper + 1e-9
            for value, (lower, upper) in zip(best_parameters, bounds, strict=False)
        )
    )
    accepted = bool(
        np.all(np.isfinite(candidate))
        and improvement >= 0.05
        and per_view_ok
        and bounds_satisfied
    )
    report = {
        "method": "multi_view_lidar_raycast",
        "accepted": accepted,
        "reason": None
        if accepted
        else "candidate did not improve the bounded sensor-ray objective",
        "baseline_score": baseline_score,
        "candidate_score": candidate_score,
        "relative_improvement": improvement,
        "parameters": {
            "translation_world_m": best_parameters[:3].tolist(),
            "yaw_delta_deg": math.degrees(float(best_parameters[3])),
            "axis_scale": np.exp(best_parameters[4:7]).tolist(),
        },
        "limits": {
            "max_translation_m": max_translation,
            "max_rotation_deg": max_rotation_deg,
            "max_axis_scale_change": max_axis_scale_change,
            "max_evaluations": max_evaluations,
            "satisfied": bounds_satisfied,
        },
        "evaluations": evaluations,
        "optimizer_messages": optimizer_messages,
        "heading": {
            "axis_world": heading_axis.tolist(),
            "observations": heading_records,
            "polish": yaw_polish,
            "per_view_acceptance": per_view_acceptance,
        },
        "support": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in support.items()
        },
        "ground_plane": ground,
        "baseline_views": baseline_metrics,
        "candidate_views": candidate_metrics,
    }
    return (candidate if accepted else baseline), report
