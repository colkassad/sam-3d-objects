import json
from pathlib import Path

import numpy as np
import pytest
import trimesh

from sam3_route.artifacts import ROUTE_MANIFEST_SCHEMA, TRACKS_SCHEMA, atomic_write_json
from sam3_route.scene import SceneConfig, compose_scene


def _observation(identifier, scan, centroid):
    center = np.asarray(centroid, dtype=float)
    return {
        "id": f"{identifier}-observation-{scan}",
        "frame_id": f"frame-{scan:06d}",
        "scan_index": scan,
        "timestamp_ns": scan,
        "prediction_id": identifier,
        "prompt": "car",
        "score": 0.9,
        "mask_path": f"tracks/{identifier}/mask.png",
        "cleaned_mask_path": f"tracks/{identifier}/mask.png",
        "points_path": f"tracks/{identifier}/points.npz",
        "mask_area_px": 100,
        "image_area_px": 1000,
        "valid_depth_fraction": 0.9,
        "inlier_fraction": 0.9,
        "border_touch": False,
        "median_range_m": 10.0,
        "azimuth_rad": 0.0,
        "elevation_rad": 0.0,
        "centroid_world": center.tolist(),
        "bbox_min_world": (center - [1.0, 0.5, 0.5]).tolist(),
        "bbox_max_world": (center + [1.0, 0.5, 0.5]).tolist(),
        "extents_world": [2.0, 1.0, 1.0],
        "quality": 0.9,
        "depth_candidate_rank": 0,
    }


def _write_track_mesh(
    run_dir: Path, identifier: str, mesh_center, *, yaw_deg: float = 0.0
) -> str:
    mesh = trimesh.creation.box(extents=(4.0, 2.0, 1.5))
    transform = trimesh.transformations.rotation_matrix(
        np.radians(yaw_deg), [0.0, 0.0, 1.0]
    )
    transform[:3, 3] = mesh_center
    scene = trimesh.Scene()
    scene.add_geometry(
        mesh, node_name=identifier, geom_name=identifier, transform=transform
    )
    path = run_dir / "meshes" / f"{identifier}.glb"
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(path, file_type="glb")
    return path.relative_to(run_dir).as_posix()


def _track(
    run_dir,
    identifier,
    mesh_center,
    support_center,
    score,
    *,
    prompt="car",
    scan=1,
    yaw_deg=0.0,
):
    observation = _observation(identifier, scan, support_center)
    observation["prompt"] = prompt
    mesh_path = _write_track_mesh(
        run_dir, identifier, mesh_center, yaw_deg=yaw_deg
    )
    transform = trimesh.transformations.rotation_matrix(
        np.radians(yaw_deg), [0.0, 0.0, 1.0]
    )
    transform[:3, 3] = mesh_center
    return {
        "id": identifier,
        "prompt": prompt,
        "status": "ok",
        "motion_state": "confirmed_static",
        "dynamic": False,
        "centroid_world": list(support_center),
        "observations": [observation],
        "selected_observation_id": observation["id"],
        "best_rgb": f"tracks/{identifier}/rgb.png",
        "best_mask": f"tracks/{identifier}/mask.png",
        "mesh": {
            "path": mesh_path,
            "world_from_glb": transform.tolist(),
            "fit": {
                "accepted": False,
                "baseline_score": score,
                "baseline_views": [{"median_depth_residual_m": score / 2}],
            },
        },
    }


def _write_run(run_dir, tracks):
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(
        run_dir / "tracks.json",
        {
            "schema": TRACKS_SCHEMA,
            "tracks": tracks,
            "rejected_observations": [],
            "duplicate_suppression": {
                "config": {
                    "max_centroid_m": 1.0,
                    "min_shared_fraction": 0.5,
                    "min_containment": 0.2,
                }
            },
        },
    )
    atomic_write_json(
        run_dir / "route-manifest.json",
        {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "coordinate_system": {"type": "local_slam", "units": "meters"},
            "stages": {"reconstruct": {"status": "complete"}},
            "keyframes": [],
            "tracks": "tracks.json",
            "outputs": {},
        },
    )


def test_scene_suppresses_bad_overlap_but_preserves_individual_glb(tmp_path):
    run_dir = tmp_path / "run"
    good = _track(run_dir, "track-good", [0, 0, 0], [0, 0, 0], 0.5)
    angled = _track(run_dir, "track-angled", [0.25, 0, 0], [4, 0, 0], 4.0)
    neighbor = _track(run_dir, "track-neighbor", [6, 0, 0], [8, 0, 0], 0.8)
    elevated = _track(run_dir, "track-elevated", [0, 0, 4], [12, 0, 0], 0.7)
    other_prompt = _track(
        run_dir, "track-bus", [0, 0, 0], [16, 0, 0], 0.6, prompt="bus"
    )
    _write_run(run_dir, [angled, elevated, good, neighbor, other_prompt])

    output = compose_scene(run_dir, SceneConfig(), overwrite=True)

    document = json.loads((run_dir / "scene.json").read_text())
    kept = {value["track_id"] for value in document["meshes"]}
    assert kept == {"track-good", "track-neighbor", "track-elevated", "track-bus"}
    assert document["suppressed_meshes"][0]["track_id"] == "track-angled"
    assert document["suppressed_meshes"][0]["winner_track_id"] == "track-good"
    assert document["suppressed_meshes"][0]["reasons"] == ["world_mesh_overlap"]
    assert (run_dir / "meshes" / "track-angled.glb").is_file()
    saved_tracks = json.loads((run_dir / "tracks.json").read_text())["tracks"]
    assert [value["status"] for value in saved_tracks] == [
        "ok",
        "ok",
        "ok",
        "ok",
        "ok",
    ]
    composed = trimesh.load_scene(output)
    assert len(composed.graph.nodes_geometry) == 4


def test_scene_suppresses_yosemite_like_iou_when_containment_is_below_threshold(
    tmp_path,
):
    run_dir = tmp_path / "yosemite-overlap"
    good = _track(run_dir, "track-000026", [0, 0, 0], [0, 10, 0], 0.7)
    angled = _track(
        run_dir,
        "track-000028",
        [1, 0, 0],
        [5, 10, 0],
        4.4,
        yaw_deg=45.0,
    )
    neighbor = _track(run_dir, "track-neighbor", [6, 0, 0], [10, 10, 0], 0.8)
    _write_run(run_dir, [angled, neighbor, good])

    compose_scene(run_dir, SceneConfig(), overwrite=True)

    document = json.loads((run_dir / "scene.json").read_text())
    assert {value["track_id"] for value in document["meshes"]} == {
        "track-000026",
        "track-neighbor",
    }
    suppression = document["suppressed_meshes"][0]
    assert suppression["loser_track_id"] == "track-000028"
    assert suppression["winner_track_id"] == "track-000026"
    assert suppression["reasons"] == ["world_mesh_overlap"]
    metrics = suppression["mesh_overlap"]
    assert metrics["footprint_iou"] >= 0.35
    assert metrics["footprint_containment"] < 0.75
    assert metrics["vertical_containment"] >= 0.50


def test_scene_uses_duplicate_support_when_final_meshes_do_not_overlap(tmp_path):
    run_dir = tmp_path / "support-run"
    strong = _track(run_dir, "track-strong", [0, 0, 0], [2, 2, 0], 0.4, scan=1)
    weak = _track(run_dir, "track-weak", [20, 0, 0], [2.2, 2, 0], 3.0, scan=1)
    _write_run(run_dir, [weak, strong])

    compose_scene(run_dir, overwrite=True)

    document = json.loads((run_dir / "scene.json").read_text())
    assert [value["track_id"] for value in document["meshes"]] == ["track-strong"]
    suppression = document["suppressed_meshes"][0]
    assert suppression["reasons"] == ["duplicate_track_support"]
    assert suppression["track_support"]["median_aabb_containment"] > 0.2


def test_yosemite_overlap_regression_omits_observed_lower_quality_pairs(tmp_path):
    run_dir = tmp_path / "yosemite-regression"
    track_000008 = _track(
        run_dir, "track-000008", [0, 0, 0], [10, 0, 0], 1.236
    )
    track_000006 = _track(
        run_dir, "track-000006", [0.2, 0, 0], [15, 0, 0], 8.269
    )
    track_000024 = _track(
        run_dir, "track-000024", [12, 0, 0], [20, 0, 0], 0.675
    )
    track_000026 = _track(
        run_dir, "track-000026", [12.25, 0, 0], [25, 0, 0], 0.942
    )
    track_000019 = _track(
        run_dir, "track-000019", [25, 0, 0], [30, 0, 0], 1.5
    )
    _write_run(
        run_dir,
        [
            track_000006,
            track_000008,
            track_000019,
            track_000024,
            track_000026,
        ],
    )

    compose_scene(run_dir, overwrite=True)

    document = json.loads((run_dir / "scene.json").read_text())
    assert {value["track_id"] for value in document["meshes"]} == {
        "track-000008",
        "track-000019",
        "track-000024",
    }
    assert {
        (value["track_id"], value["winner_track_id"])
        for value in document["suppressed_meshes"]
    } == {
        ("track-000006", "track-000008"),
        ("track-000026", "track-000024"),
    }


def test_scene_greedy_suppression_does_not_transitively_remove_neighbor(tmp_path):
    run_dir = tmp_path / "chain-run"
    first = _track(run_dir, "track-a", [0, 0, 0], [0, 10, 0], 0.2)
    middle = _track(run_dir, "track-b", [2, 0, 0], [4, 10, 0], 1.0)
    last = _track(run_dir, "track-c", [4, 0, 0], [8, 10, 0], 0.8)
    _write_run(run_dir, [middle, last, first])
    config = SceneConfig(mesh_overlap_min_iou=0.30, mesh_overlap_min_containment=0.45)

    compose_scene(run_dir, config, overwrite=True)

    document = json.loads((run_dir / "scene.json").read_text())
    assert {value["track_id"] for value in document["meshes"]} == {
        "track-a",
        "track-c",
    }
    assert [value["track_id"] for value in document["suppressed_meshes"]] == [
        "track-b"
    ]


def test_scene_suppression_can_be_disabled_and_is_resumable(tmp_path):
    run_dir = tmp_path / "disabled-run"
    first = _track(run_dir, "track-a", [0, 0, 0], [0, 0, 0], 0.2)
    second = _track(run_dir, "track-b", [0, 0, 0], [5, 0, 0], 2.0)
    _write_run(run_dir, [first, second])
    config = SceneConfig(suppress_overlapping_meshes=False)

    first_output = compose_scene(run_dir, config, overwrite=True)
    second_output = compose_scene(run_dir, config)

    assert first_output == second_output
    document = json.loads((run_dir / "scene.json").read_text())
    assert len(document["meshes"]) == 2
    assert document["suppressed_meshes"] == []
    with pytest.raises(RuntimeError, match="configuration changed"):
        compose_scene(run_dir, SceneConfig())


def test_scene_failure_removes_stale_outputs_and_marks_stage_failed(tmp_path):
    run_dir = tmp_path / "failure-run"
    track = _track(run_dir, "track-missing", [0, 0, 0], [0, 0, 0], 1.0)
    _write_run(run_dir, [track])
    (run_dir / track["mesh"]["path"]).unlink()
    (run_dir / "scene.glb").write_bytes(b"stale")
    (run_dir / "scene.json").write_text("{}")

    with pytest.raises(FileNotFoundError, match="does not exist"):
        compose_scene(run_dir, overwrite=True)

    assert not (run_dir / "scene.glb").exists()
    assert not (run_dir / "scene.json").exists()
    manifest = json.loads((run_dir / "route-manifest.json").read_text())
    assert manifest["stages"]["scene_compose"]["status"] == "failed"
    assert "scene_glb" not in manifest["outputs"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mesh_overlap_min_iou", 0.0),
        ("mesh_overlap_min_containment", 1.1),
        ("mesh_vertical_overlap_min", float("nan")),
        ("mesh_overlap_resolution_m", 0.0),
    ],
)
def test_scene_config_validates_thresholds(field, value):
    with pytest.raises(ValueError, match=field):
        SceneConfig(**{field: value})
