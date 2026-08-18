import json

import numpy as np
import pytest
import trimesh
from PIL import Image

from sam3_route.artifacts import ROUTE_MANIFEST_SCHEMA, TRACKS_SCHEMA, atomic_write_json
from sam3_route.reconstruct import (
    ReconstructConfig,
    _reconstruction_eligible,
    align_mesh_to_observation_tangent,
    reconstruct_route,
    stabilize_mesh_orientation,
    validate_mesh_placement,
)


def test_motion_is_diagnostic_for_evidence_first_tracks():
    track = {
        "status": "pending",
        "motion_state": "dynamic",
        "dynamic": True,
        "motion_gate_applied": False,
        "evidence_gate": {"eligible": True, "range_eligible": True},
    }

    assert _reconstruction_eligible(track) is True


def test_tangent_alignment_preserves_depth_and_corrects_lateral_offset():
    vertices = np.asarray(trimesh.creation.box(extents=(2, 2, 2)).vertices)
    transform = np.eye(4)
    transform[:3, 3] = [5.0, 3.0, 0.0]
    support = np.asarray([[5.0, 0.0, 0.0], [5.0, 0.2, 0.0]])

    aligned, report = align_mesh_to_observation_tangent(
        vertices, transform, support, [1.0, 0.0, 0.0]
    )

    assert report["applied"] is True
    assert aligned[0, 3] == pytest.approx(5.0)
    assert aligned[1, 3] == pytest.approx(0.1)


def test_placement_validation_rejects_bad_depth_and_neighbor_centroid(
    monkeypatch,
):
    mesh = trimesh.creation.box(extents=(4, 2, 2))
    good_metrics = {
        "views": [],
        "view_count": 1,
        "median_depth_residual_m": 0.20,
        "hit_fraction": 0.95,
        "false_background_fraction": 0.10,
        "median_range_m": 10.0,
    }
    monkeypatch.setattr(
        "sam3_route.reconstruct.evaluate_mesh_views",
        lambda *args, **kwargs: dict(good_metrics),
    )
    track = {"id": "track-a", "prompt": "car", "centroid_world": [0, 0, 0]}
    neighbor = {
        "id": "track-b",
        "prompt": "car",
        "status": "pending",
        "centroid_world": [0.5, 0, 0],
    }

    blocked = validate_mesh_placement(
        mesh.vertices,
        mesh.faces,
        np.eye(4),
        [object()],
        track,
        [track, neighbor],
    )
    assert blocked["accepted"] is False
    assert blocked["checks"]["neighbor_centroids_clear"] is False

    monkeypatch.setattr(
        "sam3_route.reconstruct.evaluate_mesh_views",
        lambda *args, **kwargs: {**good_metrics, "median_depth_residual_m": 0.80},
    )
    bad_depth = validate_mesh_placement(
        mesh.vertices, mesh.faces, np.eye(4), [object()], track, [track]
    )
    assert bad_depth["accepted"] is False
    assert bad_depth["checks"]["depth_residual"] is False


def test_orientation_prior_turns_object_upright_and_aligns_long_axis():
    vertices = np.asarray(trimesh.creation.box(extents=(2, 2, 6)).vertices)
    desired = np.eye(4)
    desired[:3, :3] = np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    desired[:3, 3] = [10.0, 20.0, 1.0]
    target = vertices @ desired[:3, :3].T + desired[:3, 3]
    upside_down = desired.copy()
    upside_down[:3, :3] = np.asarray(
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
    )

    stabilized, report = stabilize_mesh_orientation(vertices, target, upside_down)

    axes = stabilized[:3, :3] / np.linalg.norm(stabilized[:3, :3], axis=0)
    assert axes[:, 1] @ [0.0, 0.0, 1.0] > 0.999
    assert abs(axes[:, 2] @ [1.0, 0.0, 0.0]) > 0.999
    assert report["tilt_corrected"] is True
    assert report["long_axis_aligned"] is True


def test_orientation_prior_rotates_about_mesh_center():
    vertices = np.asarray(trimesh.creation.box(extents=(2, 2, 6)).vertices) + [4, 2, 1]
    target = np.asarray(trimesh.creation.box(extents=(6, 2, 2)).vertices) + [20, 5, 1]
    initial = np.eye(4)
    initial[:3, 3] = [10, 3, 2]
    before = np.median(vertices @ initial[:3, :3].T + initial[:3, 3], axis=0)

    stabilized, _ = stabilize_mesh_orientation(vertices, target, initial)

    after = np.median(vertices @ stabilized[:3, :3].T + stabilized[:3, 3], axis=0)
    np.testing.assert_allclose(after, before, atol=1e-8)


def make_route_run(tmp_path):
    run_dir = tmp_path / "run"
    frame_dir = run_dir / "frames" / "frame-000001"
    track_dir = run_dir / "tracks" / "track-000001"
    frame_dir.mkdir(parents=True)
    track_dir.mkdir(parents=True)
    rgb = np.full((8, 16, 3), 128, dtype=np.uint8)
    mask = np.zeros((8, 16), dtype=np.uint8)
    mask[2:6, 6:10] = 255
    Image.fromarray(rgb).save(frame_dir / "rgb.png")
    Image.fromarray(mask).save(track_dir / "best_mask.png")
    Image.fromarray(rgb).save(track_dir / "best_rgb.png")
    ray_direction = np.zeros((8, 16, 3), dtype=np.float32)
    for column in range(16):
        angle = (column - 8) * 0.02
        ray_direction[:, column, 0] = np.cos(angle)
        ray_direction[:, column, 1] = np.sin(angle)
    ray_direction[..., 2] = np.linspace(-0.05, 0.05, 8)[:, None]
    np.savez_compressed(
        run_dir / "calibration.npz",
        sensor_to_body=np.eye(4),
        ray_direction=ray_direction,
        ray_origin=np.zeros_like(ray_direction),
    )
    atomic_write_json(
        run_dir / "calibration.json",
        {"arrays": "calibration.npz", "layout": "destaggered HxW"},
    )
    poses = np.repeat(np.eye(4)[None], 16, axis=0)
    np.savez_compressed(
        frame_dir / "geometry.npz",
        range_mm=np.full((8, 16), 5000, dtype=np.uint32),
        body_to_world=poses,
        column_timestamp_ns=np.arange(16),
        reference_column=np.asarray(8, dtype=np.int32),
    )
    cube = trimesh.creation.box(extents=(1, 1, 1))
    points = np.asarray(cube.vertices, dtype=np.float32) + [5.0, 0.0, 0.0]
    np.savez_compressed(track_dir / "points.npz", points_world=points)
    route_manifest = {
        "schema": ROUTE_MANIFEST_SCHEMA,
        "source": {},
        "coordinate_system": {"type": "local_slam", "units": "meters"},
        "stages": {"segment": {"status": "complete"}},
        "calibration": "calibration.json",
        "trajectory": None,
        "keyframes": [
            {
                "id": "frame-000001",
                "scan_index": 1,
                "timestamp_ns": 1,
                "reference_column": 8,
                "rgb": "frames/frame-000001/rgb.png",
                "geometry": "frames/frame-000001/geometry.npz",
                "mask_manifest": "unused.json",
            }
        ],
        "prompts": ["car"],
        "tracks": "tracks.json",
        "outputs": {},
    }
    atomic_write_json(run_dir / "route-manifest.json", route_manifest)
    observation = {
        "id": "frame-000001-p000-i000",
        "frame_id": "frame-000001",
        "scan_index": 1,
        "timestamp_ns": 1,
        "prediction_id": "p000-i000",
        "prompt": "car",
        "score": 0.9,
        "mask_path": "tracks/track-000001/best_mask.png",
        "cleaned_mask_path": "tracks/track-000001/best_mask.png",
        "points_path": "tracks/track-000001/points.npz",
        "mask_area_px": 16,
        "image_area_px": 128,
        "valid_depth_fraction": 1.0,
        "inlier_fraction": 1.0,
        "border_touch": False,
        "median_range_m": 5.0,
        "azimuth_rad": 0.0,
        "elevation_rad": 0.0,
        "centroid_world": [5.0, 0.0, 0.0],
        "bbox_min_world": [4.5, -0.5, -0.5],
        "bbox_max_world": [5.5, 0.5, 0.5],
        "extents_world": [1.0, 1.0, 1.0],
        "quality": 1.0,
    }
    atomic_write_json(
        run_dir / "tracks.json",
        {
            "schema": TRACKS_SCHEMA,
            "tracks": [
                {
                    "id": "track-000001",
                    "prompt": "car",
                    "status": "pending",
                    "motion_state": "confirmed_static",
                    "dynamic": False,
                    "estimated_speed_mps": 0.0,
                    "centroid_world": [5.0, 0.0, 0.0],
                    "observations": [observation],
                    "selected_observation_id": observation["id"],
                    "points": "tracks/track-000001/points.npz",
                    "best_rgb": "tracks/track-000001/best_rgb.png",
                    "best_mask": "tracks/track-000001/best_mask.png",
                    "mesh": None,
                }
            ],
            "rejected_observations": [],
        },
    )
    return run_dir, cube


def test_reconstruct_writes_positioned_individual_and_aggregate_glbs(tmp_path):
    run_dir, cube = make_route_run(tmp_path)
    calls = []

    class FakeInference:
        def __init__(self, config, **kwargs):
            calls.append(("init", config, kwargs))

        def __call__(self, image, mask, **kwargs):
            calls.append(("infer", image.shape, mask.shape, kwargs))
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    scene_path, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert failures == 0
    assert scene_path.is_file()
    assert (run_dir / "meshes" / "track-000001.glb").is_file()
    assert (run_dir / "scene.json").is_file()
    assert [call[0] for call in calls] == ["init", "infer"]
    infer_kwargs = calls[1][3]
    assert infer_kwargs["mesh_target_faces"] == 10_000
    assert infer_kwargs["stage1_inference_steps"] == 15
    assert infer_kwargs["pointmap"].shape[-1] == 3
    tracks = json.loads((run_dir / "tracks.json").read_text())
    assert tracks["tracks"][0]["status"] == "ok"
    assert tracks["tracks"][0]["mesh"]["sam3d_pose"]["convention"] == (
        "PyTorch3D row-vector local-to-camera"
    )
    matrix = np.asarray(tracks["tracks"][0]["mesh"]["world_from_glb"])
    assert matrix.shape == (4, 4)

    individual = trimesh.load(run_dir / "meshes" / "track-000001.glb", force="scene")
    individual_node = list(individual.graph.nodes_geometry)[0]
    glb_matrix, _ = individual.graph[individual_node]
    np.testing.assert_allclose(glb_matrix, matrix, atol=1e-6)

    scene_document = json.loads((run_dir / "scene.json").read_text())
    assert scene_document["meshes"][0]["dimensions_m"] == [1.0, 1.0, 1.0]
    np.testing.assert_allclose(
        scene_document["meshes"][0]["world_from_glb"], glb_matrix, atol=1e-6
    )

    second_path, second_failures = reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )
    assert second_path == scene_path
    assert second_failures == 0
    assert [call[0] for call in calls] == ["init", "infer"]


def test_reconstruct_retries_ranked_views_and_records_attempts(tmp_path, monkeypatch):
    run_dir, cube = make_route_run(tmp_path)
    document = json.loads((run_dir / "tracks.json").read_text())
    track = document["tracks"][0]
    original = track["observations"][0]
    observations = []
    for index in range(3):
        value = json.loads(json.dumps(original))
        value["id"] = f"view-{index + 1}"
        value["selection_rank"] = index + 1
        observations.append(value)
    track["observations"] = observations
    track["selected_observation_id"] = "view-1"
    track["ranked_observation_ids"] = ["view-1", "view-2", "view-3"]
    atomic_write_json(run_dir / "tracks.json", document)
    calls = []
    monkeypatch.setattr(
        "sam3_route.reconstruct.validate_mesh_placement",
        lambda *args, **kwargs: {"accepted": True},
    )

    class FakeInference:
        def __init__(self, config, **kwargs):
            pass

        def __call__(self, image, mask, **kwargs):
            calls.append(len(calls) + 1)
            if len(calls) < 3:
                return {}
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    _, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(
            sam3d_config=str(tmp_path / "pipeline.yaml"), fit_mode="none"
        ),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert failures == 0
    assert calls == [1, 2, 3]
    result = json.loads((run_dir / "tracks.json").read_text())["tracks"][0]
    assert result["selected_observation_id"] == "view-3"
    assert [value["status"] for value in result["reconstruction_attempts"]] == [
        "failed",
        "failed",
        "ok",
    ]


def test_reconstruct_records_all_ranked_view_failures(tmp_path):
    run_dir, _ = make_route_run(tmp_path)
    document = json.loads((run_dir / "tracks.json").read_text())
    track = document["tracks"][0]
    original = track["observations"][0]
    track["observations"] = []
    for index in range(3):
        value = json.loads(json.dumps(original))
        value["id"] = f"view-{index + 1}"
        track["observations"].append(value)
    track["selected_observation_id"] = "view-1"
    track["ranked_observation_ids"] = ["view-1", "view-2", "view-3"]
    atomic_write_json(run_dir / "tracks.json", document)

    class FailingInference:
        def __init__(self, config, **kwargs):
            pass

        def __call__(self, image, mask, **kwargs):
            return {}

    _, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(
            sam3d_config=str(tmp_path / "pipeline.yaml"), fit_mode="none"
        ),
        inference_factory=FailingInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert failures == 1
    result = json.loads((run_dir / "tracks.json").read_text())["tracks"][0]
    assert result["status"] == "failed"
    assert result["selected_observation_id"] == "view-1"
    assert [value["status"] for value in result["reconstruction_attempts"]] == [
        "failed",
        "failed",
        "failed",
    ]


def test_reconstruct_retries_when_placement_validation_fails(tmp_path, monkeypatch):
    run_dir, cube = make_route_run(tmp_path)
    document = json.loads((run_dir / "tracks.json").read_text())
    track = document["tracks"][0]
    original = track["observations"][0]
    track["observations"] = []
    for index in range(3):
        value = json.loads(json.dumps(original))
        value["id"] = f"view-{index + 1}"
        track["observations"].append(value)
    track["selected_observation_id"] = "view-1"
    track["ranked_observation_ids"] = ["view-1", "view-2", "view-3"]
    atomic_write_json(run_dir / "tracks.json", document)
    inference_calls = []
    validation_calls = []

    def fake_validation(*args, **kwargs):
        validation_calls.append(len(validation_calls) + 1)
        return {"accepted": len(validation_calls) >= 5}

    monkeypatch.setattr(
        "sam3_route.reconstruct.validate_mesh_placement", fake_validation
    )

    class FakeInference:
        def __init__(self, config, **kwargs):
            pass

        def __call__(self, image, mask, **kwargs):
            inference_calls.append(1)
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    _, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(
            sam3d_config=str(tmp_path / "pipeline.yaml"), fit_mode="none"
        ),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert failures == 0
    assert len(inference_calls) == 3
    result = json.loads((run_dir / "tracks.json").read_text())["tracks"][0]
    assert result["selected_observation_id"] == "view-3"
    assert [value["status"] for value in result["reconstruction_attempts"]] == [
        "failed",
        "failed",
        "ok",
    ]
    assert result["placement_mode"] == "multi_view"


def test_reconstruct_falls_back_to_snapshot_when_multi_view_disagrees(
    tmp_path, monkeypatch
):
    run_dir, cube = make_route_run(tmp_path)
    validation_calls = []

    def fake_validation(*args, **kwargs):
        validation_calls.append(1)
        return {"accepted": len(validation_calls) == 2}

    monkeypatch.setattr(
        "sam3_route.reconstruct.validate_mesh_placement", fake_validation
    )
    selected = "frame-000001-p000-i000"
    monkeypatch.setattr(
        "sam3_route.reconstruct.load_lidar_fit_views",
        lambda *args, **kwargs: [
            type("View", (), {"observation_id": selected})(),
            type("View", (), {"observation_id": "other-view"})(),
        ],
    )

    class FakeInference:
        def __init__(self, config, **kwargs):
            pass

        def __call__(self, image, mask, **kwargs):
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    _, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(
            sam3d_config=str(tmp_path / "pipeline.yaml"), fit_mode="none"
        ),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert failures == 0
    result = json.loads((run_dir / "tracks.json").read_text())["tracks"][0]
    assert result["placement_mode"] == "snapshot"
    assert [
        value["placement_mode"]
        for value in result["reconstruction_attempts"][0]["placement_candidates"]
    ] == ["multi_view", "snapshot"]


def test_all_placement_attempts_fail_with_explicit_status(tmp_path, monkeypatch):
    run_dir, cube = make_route_run(tmp_path)
    monkeypatch.setattr(
        "sam3_route.reconstruct.validate_mesh_placement",
        lambda *args, **kwargs: {"accepted": False, "checks": {"depth": False}},
    )

    class FakeInference:
        def __init__(self, config, **kwargs):
            pass

        def __call__(self, image, mask, **kwargs):
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    _, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(
            sam3d_config=str(tmp_path / "pipeline.yaml"), fit_mode="none"
        ),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert failures == 1
    result = json.loads((run_dir / "tracks.json").read_text())["tracks"][0]
    assert result["status"] == "failed_placement"
    assert (
        result["reconstruction_attempts"][0]["placement_validation"]["accepted"]
        is False
    )


def test_reconstruct_skips_unconfirmed_tracks(tmp_path):
    run_dir, cube = make_route_run(tmp_path)
    document = json.loads((run_dir / "tracks.json").read_text())
    unconfirmed = json.loads(json.dumps(document["tracks"][0]))
    unconfirmed["id"] = "track-000002"
    unconfirmed["motion_state"] = "unconfirmed"
    unconfirmed["status"] = "unconfirmed_skipped"
    document["tracks"].append(unconfirmed)
    atomic_write_json(run_dir / "tracks.json", document)
    calls = []

    class FakeInference:
        def __init__(self, config, **kwargs):
            calls.append("init")

        def __call__(self, image, mask, **kwargs):
            calls.append("infer")
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert calls == ["init", "infer"]
    scene = json.loads((run_dir / "scene.json").read_text())
    assert [value["track_id"] for value in scene["meshes"]] == ["track-000001"]


def test_reconstruct_skips_tracks_marked_as_duplicates(tmp_path):
    run_dir, cube = make_route_run(tmp_path)
    document = json.loads((run_dir / "tracks.json").read_text())
    duplicate = json.loads(json.dumps(document["tracks"][0]))
    duplicate["id"] = "track-000002"
    duplicate["status"] = "duplicate_skipped"
    duplicate["duplicate_of"] = "track-000001"
    document["tracks"].append(duplicate)
    atomic_write_json(run_dir / "tracks.json", document)
    calls = []

    class FakeInference:
        def __init__(self, config, **kwargs):
            calls.append("init")

        def __call__(self, image, mask, **kwargs):
            calls.append("infer")
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    assert calls == ["init", "infer"]
    tracks = json.loads((run_dir / "tracks.json").read_text())["tracks"]
    assert tracks[1]["status"] == "duplicate_skipped"
    assert tracks[1]["mesh"] is None


def test_reconstruct_skips_confirmed_static_track_outside_range(tmp_path):
    run_dir, _ = make_route_run(tmp_path)
    document = json.loads((run_dir / "tracks.json").read_text())
    track = document["tracks"][0]
    track["status"] = "range_skipped"
    track["range_gate"] = {
        "max_mesh_range_m": 30.0,
        "minimum_observation_range_m": 50.0,
        "eligible": False,
        "eligible_observation_count": 0,
        "eligible_observation_ids": [],
        "reason": "no observation is at or below 30 m; nearest is 50 m",
    }
    track["reconstruction_points"] = None
    atomic_write_json(run_dir / "tracks.json", document)
    calls = []

    class UnexpectedInference:
        def __init__(self, config, **kwargs):
            calls.append("init")

    scene_path, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        inference_factory=UnexpectedInference,
        image_loader=lambda path: pytest.fail("image loader must not be called"),
    )

    assert failures == 0
    assert calls == []
    assert scene_path.read_bytes()[:4] == b"glTF"
    scene = json.loads((run_dir / "scene.json").read_text())
    assert scene["meshes"] == []


def test_reconstruct_prefers_range_filtered_points(tmp_path, monkeypatch):
    run_dir, cube = make_route_run(tmp_path)
    filtered_points = np.asarray(cube.vertices, dtype=np.float32) + [7.0, 0.0, 0.0]
    track_dir = run_dir / "tracks" / "track-000001"
    np.savez_compressed(
        track_dir / "reconstruction-points.npz", points_world=filtered_points
    )
    document = json.loads((run_dir / "tracks.json").read_text())
    track = document["tracks"][0]
    track["range_gate"] = {"eligible": True, "max_mesh_range_m": 30.0}
    track["reconstruction_points"] = "tracks/track-000001/reconstruction-points.npz"
    track["reconstruction_centroid_world"] = [7.0, 0.0, 0.0]
    atomic_write_json(run_dir / "tracks.json", document)
    captured = {}

    def fake_refine(vertices, faces, initial_transform, views, target_points, **kwargs):
        captured["points"] = np.asarray(target_points)
        return initial_transform, {"accepted": True}

    monkeypatch.setattr(
        "sam3_route.reconstruct.load_lidar_fit_views",
        lambda *args, **kwargs: [
            type("View", (), {"observation_id": "frame-000001-p000-i000"})(),
            type("View", (), {"observation_id": "other-view"})(),
        ],
    )
    monkeypatch.setattr(
        "sam3_route.reconstruct.refine_mesh_with_lidar_rays", fake_refine
    )
    monkeypatch.setattr(
        "sam3_route.reconstruct.validate_mesh_placement",
        lambda *args, **kwargs: {"accepted": True},
    )

    class FakeInference:
        def __init__(self, config, **kwargs):
            pass

        def __call__(self, image, mask, **kwargs):
            return {
                "glb": cube.copy(),
                "rotation": np.asarray([[1.0, 0.0, 0.0, 0.0]]),
                "translation": np.asarray([[0.0, 0.0, 5.0]]),
                "scale": np.asarray([[1.0, 1.0, 1.0]]),
            }

    reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        inference_factory=FakeInference,
        image_loader=lambda path: np.asarray(Image.open(path).convert("RGB")),
    )

    np.testing.assert_allclose(captured["points"], filtered_points)
    scene = json.loads((run_dir / "scene.json").read_text())
    assert scene["meshes"][0]["centroid_world"] == [7.0, 0.0, 0.0]
    assert scene["meshes"][0]["track_centroid_world"] == [5.0, 0.0, 0.0]


def test_reconstruct_exports_a_valid_empty_scene_for_zero_tracks(tmp_path):
    run_dir, _ = make_route_run(tmp_path)
    atomic_write_json(
        run_dir / "tracks.json",
        {
            "schema": TRACKS_SCHEMA,
            "tracks": [],
            "rejected_observations": [],
        },
    )

    scene_path, failures = reconstruct_route(
        run_dir,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
    )

    assert failures == 0
    assert scene_path.read_bytes()[:4] == b"glTF"
    scene_document = json.loads((run_dir / "scene.json").read_text())
    assert scene_document["meshes"] == []
