import numpy as np
import trimesh

from sam3_route.lidar_fit import (
    LidarFitView,
    _heading_information,
    refine_mesh_with_lidar_rays,
    world_rays_from_frame,
)


def _view_from_points(observation_id, points):
    origins = np.zeros_like(points)
    ranges = np.linalg.norm(points, axis=1)
    return LidarFitView(
        observation_id=observation_id,
        origins_world=origins,
        directions_world=points / ranges[:, None],
        ranges_m=ranges,
        background_origins_world=np.empty((0, 3)),
        background_directions_world=np.empty((0, 3)),
        background_ranges_m=np.empty(0),
        ground_points_world=np.empty((0, 3)),
        sensor_position_world=np.zeros(3),
        weight=1.0,
    )


def test_world_rays_apply_sensor_extrinsic_and_per_column_pose():
    ranges = np.full((1, 2), 2_000, dtype=np.uint32)
    directions = np.asarray([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    origins = np.zeros_like(directions)
    sensor_to_body = np.eye(4)
    sensor_to_body[:3, 3] = [0.5, 0.0, 0.0]
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, 3] = [10.0, 1.0, 0.0]

    ray_origins, ray_directions, ray_ranges = world_rays_from_frame(
        ranges, directions, origins, poses, sensor_to_body
    )

    np.testing.assert_allclose(ray_origins[0, 0], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(ray_origins[0, 1], [10.5, 1.0, 0.0])
    np.testing.assert_allclose(ray_directions, directions)
    np.testing.assert_allclose(ray_ranges, 2.0)


def test_heading_uses_long_span_view_and_rejects_end_on_view():
    x = np.linspace(4.0, 8.0, 100)
    side_points = np.column_stack((x, 0.2 * x, np.ones_like(x)))
    width = np.linspace(-0.8, 0.8, 100)
    end_points = np.column_stack((np.full_like(width, 4.2), width, np.ones_like(width)))
    support = {
        "axis_world": np.asarray([1.0, 0.0, 0.0]),
        "low_m": 4.0,
        "high_m": 8.0,
    }

    axis, informative, records = _heading_information(
        [_view_from_points("side", side_points), _view_from_points("end", end_points)],
        support,
    )

    assert [view.observation_id for view in informative] == ["side"]
    assert records[0]["yaw_informative"] is True
    assert records[1]["yaw_informative"] is False
    assert abs(axis[1]) > 0.1


def test_raycast_fit_recovers_sensor_depth_without_density_recentering():
    mesh = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    y, z = np.meshgrid(np.linspace(-0.8, 0.8, 10), np.linspace(-0.8, 0.8, 10))
    points = np.column_stack((np.full(y.size, 4.0), y.ravel(), z.ravel()))
    origins = np.zeros_like(points)
    lengths = np.linalg.norm(points, axis=1)
    directions = points / lengths[:, None]
    view = LidarFitView(
        observation_id="view-1",
        origins_world=origins,
        directions_world=directions,
        ranges_m=lengths,
        background_origins_world=np.empty((0, 3)),
        background_directions_world=np.empty((0, 3)),
        background_ranges_m=np.empty(0),
        ground_points_world=np.empty((0, 3)),
        sensor_position_world=np.zeros(3),
        weight=1.0,
    )
    shifted = np.eye(4)
    shifted[:3, 3] = [5.5, 0.0, 0.0]

    transform, report = refine_mesh_with_lidar_rays(
        np.asarray(mesh.vertices),
        np.asarray(mesh.faces),
        shifted,
        [view],
        points,
        max_evaluations=120,
        grounded=False,
    )

    assert report["method"] == "multi_view_lidar_raycast"
    assert report["accepted"] is True
    assert transform[0, 3] < shifted[0, 3] - 0.2
    assert report["candidate_views"][0]["median_depth_residual_m"] < 0.15
