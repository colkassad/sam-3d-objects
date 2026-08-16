import math

import numpy as np

from sam3_route.geometry import (
    MODEL_FROM_GLB,
    model_pose_to_glb_transform,
    points_from_range,
    pose_distance,
    transform_pointmap_per_column,
    transform_points,
)
from sam3_route.extract import KeyframeSelector


def test_per_column_world_transform_applies_sensor_to_body_and_pose():
    points = np.zeros((2, 3, 3), dtype=np.float64)
    points[..., 0] = 1.0
    poses = np.repeat(np.eye(4)[None], 3, axis=0)
    poses[:, 0, 3] = [0.0, 1.0, 2.0]
    sensor_to_body = np.eye(4)
    sensor_to_body[1, 3] = 5.0

    world = transform_pointmap_per_column(points, poses, sensor_to_body)

    np.testing.assert_allclose(world[..., 0], [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_allclose(world[..., 1], 5.0)


def test_range_calibration_preserves_origin_and_invalid_pixels():
    ranges = np.asarray([[1000, 0]], dtype=np.uint32)
    direction = np.asarray([[[1, 0, 0], [0, 1, 0]]], dtype=np.float32)
    origin = np.asarray([[[0.1, 0.2, 0.3], [0, 0, 0]]], dtype=np.float32)

    points = points_from_range(ranges, direction, origin)

    np.testing.assert_allclose(points[0, 0], [1.1, 0.2, 0.3])
    assert np.isnan(points[0, 1]).all()


def test_keyframe_selector_uses_translation_or_rotation():
    selector = KeyframeSelector(distance_m=5.0, angle_deg=5.0)
    identity = np.eye(4)
    translated = identity.copy()
    translated[0, 3] = 4.9
    rotated = identity.copy()
    angle = math.radians(5.1)
    rotated[:2, :2] = [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]]

    assert selector.select(identity)
    assert not selector.select(translated)
    assert selector.select(rotated)
    distance, degrees = pose_distance(identity, rotated)
    assert distance == 0.0
    assert degrees > 5.0


def test_sam3d_model_pose_accounts_for_exported_glb_axis_conversion():
    transform = model_pose_to_glb_transform(
        [1.0, 0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [2.0, 2.0, 2.0]
    )
    glb_point = np.asarray([[1.0, 2.0, 3.0]])
    expected_model = (MODEL_FROM_GLB @ glb_point[0]) * 2.0 + [10.0, 20.0, 30.0]

    np.testing.assert_allclose(transform_points(glb_point, transform)[0], expected_model)


def test_sam3d_model_pose_transposes_pytorch3d_row_vector_rotation():
    half_angle = math.pi / 4.0
    quaternion = [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)]
    transform = model_pose_to_glb_transform(quaternion, [0.0, 0.0, 0.0], [1.0])
    glb_point = np.asarray([[1.0, 0.0, 0.0]])
    row_vector_rotation = np.asarray(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    model_point = MODEL_FROM_GLB @ glb_point[0]
    expected = row_vector_rotation.T @ model_point

    np.testing.assert_allclose(
        transform_points(glb_point, transform)[0], expected, atol=1e-7
    )
