from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


MODEL_FROM_GLB = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)
GLB_FROM_MODEL = MODEL_FROM_GLB.T


def require_transform(value: Any, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains non-finite values")
    if not np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError(f"{name} has an invalid homogeneous row")
    return matrix


def column_poses(scan: Any) -> np.ndarray:
    value = getattr(scan, "body_to_world", None)
    if value is None:
        value = getattr(scan, "pose", None)
    if callable(value):
        value = value()
    poses = np.asarray(value, dtype=np.float64)
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(
            f"scan must expose per-column poses with shape (W, 4, 4), got {poses.shape}"
        )
    if not np.all(np.isfinite(poses)):
        raise ValueError("scan poses contain non-finite values")
    return poses


def sensor_to_body(sensor_info: Any) -> np.ndarray:
    value = getattr(sensor_info, "sensor_to_body", None)
    if value is None:
        value = getattr(sensor_info, "extrinsic", None)
    if callable(value):
        value = value()
    if value is None:
        value = np.eye(4, dtype=np.float64)
    return require_transform(value, "sensor_to_body")


def valid_column_bounds(scan: Any, range_staggered: Optional[np.ndarray] = None) -> tuple[int, int]:
    first = _call_int(scan, ("get_first_valid_column", "first_valid_column"))
    last = _call_int(scan, ("get_last_valid_column", "last_valid_column"))
    if first is not None and last is not None and first >= 0 and last >= first:
        return first, last
    if range_staggered is None:
        raise ValueError("scan has no valid column bounds")
    columns = np.flatnonzero(np.any(np.asarray(range_staggered) != 0, axis=0))
    if columns.size == 0:
        raise ValueError("scan has no valid range columns")
    return int(columns[0]), int(columns[-1])


def _call_int(value: Any, names: Sequence[str]) -> Optional[int]:
    for name in names:
        member = getattr(value, name, None)
        if member is None:
            continue
        try:
            return int(member() if callable(member) else member)
        except (TypeError, ValueError, RuntimeError):
            continue
    return None


def representative_column(scan: Any, range_staggered: Optional[np.ndarray] = None) -> int:
    first, last = valid_column_bounds(scan, range_staggered)
    return (first + last) // 2


def scan_timestamp_ns(scan: Any, column: int) -> Optional[int]:
    for name in ("timestamp", "packet_timestamp"):
        value = getattr(scan, name, None)
        if value is None:
            continue
        try:
            timestamp = int(np.asarray(value)[column])
            if timestamp > 0:
                return timestamp
        except (IndexError, TypeError, ValueError, OverflowError):
            pass
    fallback = getattr(scan, "get_first_valid_packet_timestamp", None)
    if callable(fallback):
        try:
            timestamp = int(fallback())
            return timestamp if timestamp > 0 else None
        except (RuntimeError, TypeError, ValueError, OverflowError):
            pass
    return None


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    matrix = require_transform(transform)
    if points.shape[-1] != 3:
        raise ValueError("points must have a final dimension of 3")
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def transform_pointmap_per_column(
    points_sensor: np.ndarray,
    body_to_world: np.ndarray,
    sensor_to_body_matrix: np.ndarray,
) -> np.ndarray:
    """Dewarp an HxWx3 sensor-frame pointmap into the SLAM world frame."""

    points = np.asarray(points_sensor, dtype=np.float64)
    poses = np.asarray(body_to_world, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("points_sensor must have shape (H, W, 3)")
    if poses.shape != (points.shape[1], 4, 4):
        raise ValueError(
            f"body_to_world must have shape ({points.shape[1]}, 4, 4), got {poses.shape}"
        )
    effective = poses @ require_transform(sensor_to_body_matrix, "sensor_to_body")
    rotated = np.einsum("wij,hwj->hwi", effective[:, :3, :3], points)
    return rotated + effective[np.newaxis, :, :3, 3]


def points_from_range(
    range_mm: np.ndarray, ray_direction: np.ndarray, ray_origin: np.ndarray
) -> np.ndarray:
    ranges = np.asarray(range_mm, dtype=np.float64)
    directions = np.asarray(ray_direction, dtype=np.float64)
    origins = np.asarray(ray_origin, dtype=np.float64)
    expected = (*ranges.shape, 3)
    if directions.shape != expected or origins.shape != expected:
        raise ValueError(
            f"ray calibration must have shape {expected}; got {directions.shape} and {origins.shape}"
        )
    points = origins + directions * (ranges[..., np.newaxis] / 1000.0)
    points[ranges == 0] = np.nan
    return points


def pose_distance(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    first = require_transform(first, "first pose")
    second = require_transform(second, "second pose")
    translation = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return translation, math.degrees(math.acos(cosine))


def rotation_matrix_to_quaternion_wxyz(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        values = (
            0.25 * scale,
            (matrix[2, 1] - matrix[1, 2]) / scale,
            (matrix[0, 2] - matrix[2, 0]) / scale,
            (matrix[1, 0] - matrix[0, 1]) / scale,
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(max(0.0, 1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])) * 2.0
            values = ((matrix[2, 1] - matrix[1, 2]) / scale, 0.25 * scale, (matrix[0, 1] + matrix[1, 0]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale)
        elif index == 1:
            scale = math.sqrt(max(0.0, 1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])) * 2.0
            values = ((matrix[0, 2] - matrix[2, 0]) / scale, (matrix[0, 1] + matrix[1, 0]) / scale, 0.25 * scale, (matrix[1, 2] + matrix[2, 1]) / scale)
        else:
            scale = math.sqrt(max(0.0, 1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])) * 2.0
            values = ((matrix[1, 0] - matrix[0, 1]) / scale, (matrix[0, 2] + matrix[2, 0]) / scale, (matrix[1, 2] + matrix[2, 1]) / scale, 0.25 * scale)
    quaternion = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("rotation produced an invalid quaternion")
    return tuple(float(value) for value in quaternion / norm)


def rotation_matrix_to_xyz_euler(rotation: np.ndarray) -> tuple[float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64)
    pitch = math.asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(matrix[2, 1], matrix[2, 2])
        yaw = math.atan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = math.atan2(-matrix[1, 2], matrix[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def quaternion_wxyz_to_matrix(quaternion: Any) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        raise ValueError("quaternion must contain four finite wxyz values")
    values /= np.linalg.norm(values)
    w, x, y, z = values
    return np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compose_similarity(rotation: np.ndarray, translation: Any, scale: Any) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64).reshape(3)
    scale_values = np.asarray(scale, dtype=np.float64).reshape(-1)
    if scale_values.size == 1:
        scale_values = np.repeat(scale_values, 3)
    if scale_values.size != 3:
        raise ValueError("scale must be scalar or contain three values")
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("rotation must be a finite 3x3 matrix")
    if not np.all(np.isfinite(translation)):
        raise ValueError("translation must contain three finite values")
    if not np.all(np.isfinite(scale_values)) or np.any(scale_values <= 0):
        raise ValueError("scale values must be finite and greater than zero")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ np.diag(scale_values)
    matrix[:3, 3] = translation
    return matrix


def model_pose_to_glb_transform(
    rotation_wxyz: Any, translation: Any, scale: Any
) -> np.ndarray:
    """Convert SAM3D's PyTorch3D row-vector pose to a GLB column transform.

    ``compose_transform().rotate(R)`` applies points as ``p @ R``.  The
    equivalent column-vector rotation is therefore ``R.T``.  The generated
    model is z-up before ``to_glb`` converts its vertices to y-up.
    """

    rotation = quaternion_wxyz_to_matrix(rotation_wxyz).T @ MODEL_FROM_GLB
    return compose_similarity(rotation, translation, scale)


@dataclass(frozen=True)
class TangentCamera:
    sensor_from_camera: np.ndarray
    camera_from_sensor: np.ndarray
    forward_sensor: np.ndarray


def tangent_camera(forward_sensor: Any, up_sensor: Any = (0.0, 0.0, 1.0)) -> TangentCamera:
    """Return an OpenCV-style tangent camera: x right, y down, z forward."""

    forward = np.asarray(forward_sensor, dtype=np.float64).reshape(3)
    forward /= np.linalg.norm(forward)
    up = np.asarray(up_sensor, dtype=np.float64).reshape(3)
    up /= np.linalg.norm(up)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        up = np.asarray([0.0, 1.0, 0.0])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    down /= np.linalg.norm(down)
    rotation = np.column_stack((right, down, forward))
    sensor_from_camera = np.eye(4, dtype=np.float64)
    sensor_from_camera[:3, :3] = rotation
    return TangentCamera(
        sensor_from_camera=sensor_from_camera,
        camera_from_sensor=np.linalg.inv(sensor_from_camera),
        forward_sensor=forward,
    )


def r3_to_pytorch3d_points(points: np.ndarray) -> np.ndarray:
    """Match camera_to_pytorch3d_camera(): OpenCV/R3 to PyTorch3D axes."""

    value = np.asarray(points, dtype=np.float64)
    converted = value.copy()
    converted[..., 0] *= -1.0
    converted[..., 1] *= -1.0
    return converted
