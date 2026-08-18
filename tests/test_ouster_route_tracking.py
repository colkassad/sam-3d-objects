import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam3_route.artifacts import (
    ROUTE_MANIFEST_SCHEMA,
    atomic_write_json,
    read_route_manifest,
)
from sam3_route.reconstruct import ReconstructConfig, reconstruct_route
from sam3_route.tracking import (
    Observation,
    SegmentConfig,
    _dynamic_record,
    _motion_record,
    _range_eligible_observations,
    _select_observation,
    _select_consistent_depth_candidates,
    _track_document,
    _track_status,
    associate_observations,
    build_tracks,
    dominant_depth_component,
    duplicate_track_evidence,
    suppress_duplicate_track_documents,
)


def make_observation(identifier, frame, x, *, prompt="car", timestamp=0):
    return Observation(
        id=identifier,
        frame_id=f"frame-{frame}",
        scan_index=frame,
        timestamp_ns=timestamp,
        prediction_id=identifier,
        prompt=prompt,
        score=0.9,
        mask_path="mask.png",
        cleaned_mask_path="clean.png",
        points_path="points.npz",
        mask_area_px=100,
        image_area_px=1000,
        valid_depth_fraction=1.0,
        inlier_fraction=1.0,
        border_touch=False,
        median_range_m=10.0,
        azimuth_rad=0.0,
        elevation_rad=0.0,
        centroid_world=[x, 0.0, 0.0],
        bbox_min_world=[x - 0.5, -0.5, -0.5],
        bbox_max_world=[x + 0.5, 0.5, 0.5],
        extents_world=[1.0, 1.0, 1.0],
    )


def test_depth_cleanup_keeps_coherent_component_and_wraps_panorama_seam():
    mask = np.zeros((3, 8), dtype=bool)
    mask[1, [7, 0, 1]] = True
    mask[1, 4] = True
    points = np.full((3, 8, 3), np.nan)
    points[1, 7] = [5.0, -0.1, 0]
    points[1, 0] = [5.0, 0.0, 0]
    points[1, 1] = [5.0, 0.1, 0]
    points[1, 4] = [20.0, 0, 0]

    clean, selected, radius = dominant_depth_component(mask, points, min_points=3)

    assert clean[1, 7] and clean[1, 0] and clean[1, 1]
    assert not clean[1, 4]
    assert selected.shape == (3, 3)
    assert radius >= 0.05


def test_association_keeps_neighboring_objects_separate():
    observations = [
        make_observation("a1", 1, 0.0),
        make_observation("b1", 1, 5.0),
        make_observation("a2", 2, 0.1),
        make_observation("b2", 2, 5.1),
    ]

    tracks = associate_observations(observations)

    assert sorted([len(track) for track in tracks]) == [2, 2]
    assert {tuple(value.id for value in track) for track in tracks} == {
        ("a1", "a2"),
        ("b1", "b2"),
    }


def test_constant_velocity_track_is_marked_dynamic():
    track = [
        make_observation("a", 1, 0.0, timestamp=1_000_000_000),
        make_observation("b", 2, 2.0, timestamp=2_000_000_000),
        make_observation("c", 3, 4.0, timestamp=3_000_000_000),
    ]

    dynamic, speed = _dynamic_record(track, 1.0)

    assert dynamic
    assert speed == 2.0


def test_no_reliable_observations_cannot_confirm_motion():
    motion = _motion_record([], 0.5)

    assert motion.state == "unconfirmed"
    assert motion.position_uncertainty_m == 0.5
    assert "no observations" in motion.reason


def test_two_observations_can_establish_motion():
    track = [
        make_observation("a", 1, 0.0, timestamp=1_000_000_000),
        make_observation("b", 2, 2.0, timestamp=3_000_000_000),
    ]

    motion = _motion_record(track, 0.5)

    assert motion.state == "dynamic"
    assert motion.estimated_speed_mps == 1.0


def test_single_observation_is_unconfirmed_instead_of_static():
    motion = _motion_record(
        [make_observation("a", 1, 0.0, timestamp=1_000_000_000)], 0.5
    )

    assert motion.state == "unconfirmed"
    assert not motion.dynamic


def test_stop_and_go_displacement_is_dynamic_without_constant_velocity_fit():
    track = [
        make_observation("a", 1, 0.0, timestamp=1_000_000_000),
        make_observation("b", 2, 3.0, timestamp=2_000_000_000),
        make_observation("c", 3, 0.2, timestamp=3_000_000_000),
    ]

    motion = _motion_record(track, 0.5)

    assert motion.state == "dynamic"
    assert motion.maximum_displacement_m == 3.0


def test_association_survives_temporary_occlusion():
    observations = [
        make_observation("a1", 1, 2.0),
        make_observation("a3", 3, 2.1),
    ]

    tracks = associate_observations(observations)

    assert [[value.id for value in track] for track in tracks] == [["a1", "a3"]]


def test_association_allows_large_view_dependent_extent_change():
    broadside = make_observation("broadside", 1, 20.0, timestamp=1_000_000_000)
    broadside.extents_world = [8.0, 3.0, 2.5]
    end_on = make_observation("end-on", 2, 20.8, timestamp=2_000_000_000)
    end_on.extents_world = [0.7, 2.5, 2.2]

    tracks = associate_observations([broadside, end_on])

    assert [[value.id for value in track] for track in tracks] == [
        ["broadside", "end-on"]
    ]


def test_stable_track_selects_matching_alternate_depth_layer():
    first = make_observation("bus-a", 1, 20.0, timestamp=1_000_000_000)
    first.median_range_m = 70.0
    second = make_observation("bus-b", 2, 20.2, timestamp=3_000_000_000)
    second.median_range_m = 70.0
    foreground = make_observation("ambiguous-near", 3, 3.0, timestamp=5_000_000_000)
    foreground.prediction_id = "ambiguous"
    background = make_observation("ambiguous-far", 3, 20.1, timestamp=5_000_000_000)
    background.prediction_id = "ambiguous"
    background.depth_candidate_rank = 1
    background.median_range_m = 70.0

    selected = _select_consistent_depth_candidates(
        [[first], [second], [foreground, background]],
        dynamic_min_speed_mps=0.5,
    )

    assert [value.id for value in selected] == ["bus-a", "bus-b", "ambiguous-far"]


def test_clean_candidate_beats_a_larger_truncated_depth_incoherent_mask():
    clean = make_observation("clean", 1, 0.0)
    clean.score = 0.95
    clean.mask_area_px = 100
    dirty = make_observation("dirty", 2, 0.0)
    dirty.score = 0.60
    dirty.mask_area_px = 1000
    dirty.valid_depth_fraction = 0.20
    dirty.inlier_fraction = 0.20
    dirty.border_touch = True

    selected = _select_observation([dirty, clean])

    assert selected is clean
    assert clean.quality > dirty.quality


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_mesh_range_must_be_positive_and_finite(value):
    with pytest.raises(ValueError, match="max_mesh_range_m"):
        SegmentConfig(
            prompts=("car",),
            sam3_model_dir="model",
            max_mesh_range_m=value,
        )


def test_mesh_range_defaults_to_thirty_metres_and_can_be_disabled():
    default = SegmentConfig(prompts=("car",), sam3_model_dir="model")
    unlimited = SegmentConfig(
        prompts=("car",), sam3_model_dir="model", max_mesh_range_m=None
    )

    assert default.max_mesh_range_m == 30.0
    assert unlimited.max_mesh_range_m is None


def _duplicate_track(identifier, observations, *, primary=True, prompt="car"):
    values = []
    for observation in observations:
        value = vars(observation).copy()
        value["depth_candidate_rank"] = 0 if primary else 1
        value["inlier_fraction"] = 0.95 if primary else 0.35
        value["valid_depth_fraction"] = 0.95 if primary else 0.45
        value["score"] = 0.90 if primary else 0.70
        values.append(value)
    return {
        "id": identifier,
        "prompt": prompt,
        "status": "pending",
        "motion_state": "confirmed_static",
        "range_gate": {"eligible": True},
        "centroid_world": np.median(
            [value["centroid_world"] for value in values], axis=0
        ).tolist(),
        "observations": values,
    }


def test_duplicate_track_suppression_keeps_stronger_depth_support():
    strong = _duplicate_track(
        "track-strong",
        [make_observation("a1", 1, 0.0), make_observation("a2", 2, 0.1)],
    )
    weak = _duplicate_track(
        "track-weak",
        [make_observation("b1", 1, 0.2), make_observation("b2", 2, 0.3)],
        primary=False,
    )
    config = SegmentConfig(prompts=("car",), sam3_model_dir="model")

    report = suppress_duplicate_track_documents([weak, strong], config)

    assert strong["status"] == "pending"
    assert weak["status"] == "duplicate_skipped"
    assert weak["duplicate_of"] == "track-strong"
    assert report["suppressed_count"] == 1
    assert report["suppressions"][0]["evidence"]["shared_scan_fraction"] == 1.0


def test_duplicate_track_detection_preserves_distinct_support_and_prompts():
    base = _duplicate_track(
        "track-base",
        [make_observation("a1", 1, 0.0), make_observation("a2", 2, 0.0)],
    )
    disjoint = _duplicate_track(
        "track-disjoint",
        [make_observation("b1", 1, 2.0), make_observation("b2", 2, 2.0)],
    )
    other_prompt = _duplicate_track(
        "track-bus",
        [make_observation("c1", 1, 0.0), make_observation("c2", 2, 0.0)],
        prompt="bus",
    )
    low_overlap = _duplicate_track(
        "track-late",
        [make_observation("d1", 2, 0.0), make_observation("d2", 3, 0.0)],
    )

    assert duplicate_track_evidence(base, disjoint) is None
    assert duplicate_track_evidence(base, other_prompt) is None
    assert duplicate_track_evidence(base, low_overlap, min_shared_fraction=0.75) is None


def test_duplicate_track_default_rejects_marginal_aabb_containment():
    first = _duplicate_track(
        "track-first",
        [make_observation("a1", 1, 0.0), make_observation("a2", 2, 0.0)],
    )
    second = _duplicate_track(
        "track-second",
        [make_observation("b1", 1, 0.38), make_observation("b2", 2, 0.38)],
    )
    for observation in first["observations"]:
        observation["bbox_min_world"] = [0.0, 0.0, 0.0]
        observation["bbox_max_world"] = [1.0, 1.0, 1.0]
    for observation in second["observations"]:
        observation["bbox_min_world"] = [0.78, 0.0, 0.0]
        observation["bbox_max_world"] = [1.78, 1.0, 1.0]

    assert duplicate_track_evidence(first, second) is None
    marginal = duplicate_track_evidence(first, second, min_containment=0.20)
    assert marginal is not None
    assert marginal["median_aabb_containment"] == pytest.approx(0.22)


def test_duplicate_track_suppression_can_be_disabled():
    first = _duplicate_track(
        "track-a",
        [make_observation("a1", 1, 0.0), make_observation("a2", 2, 0.0)],
    )
    second = _duplicate_track(
        "track-b",
        [make_observation("b1", 1, 0.1), make_observation("b2", 2, 0.1)],
        primary=False,
    )
    config = SegmentConfig(
        prompts=("car",), sam3_model_dir="model", suppress_duplicate_tracks=False
    )

    report = suppress_duplicate_track_documents([first, second], config)

    assert report["suppressed_count"] == 0
    assert [first["status"], second["status"]] == ["pending", "pending"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duplicate_track_max_centroid_m", 0.0),
        ("duplicate_track_min_shared_fraction", 0.0),
        ("duplicate_track_min_containment", 1.1),
    ],
)
def test_duplicate_track_thresholds_are_validated(field, value):
    with pytest.raises(ValueError, match=field):
        SegmentConfig(prompts=("car",), sam3_model_dir="model", **{field: value})


def test_mesh_range_is_inclusive():
    boundary = make_observation("boundary", 1, 0.0)
    boundary.median_range_m = 30.0
    outside = make_observation("outside", 2, 0.0)
    outside.median_range_m = 30.0001

    assert _range_eligible_observations([boundary, outside], 30.0) == [boundary]
    assert _range_eligible_observations([boundary, outside], None) == [
        boundary,
        outside,
    ]


@pytest.mark.parametrize(
    ("motion_state", "range_eligible", "expected"),
    [
        ("dynamic", False, "dynamic_skipped"),
        ("unconfirmed", False, "unconfirmed_skipped"),
        ("confirmed_static", False, "range_skipped"),
        ("confirmed_static", True, "pending"),
    ],
)
def test_motion_state_takes_precedence_over_range_gate(
    motion_state, range_eligible, expected
):
    assert _track_status(motion_state, range_eligible) == expected


def test_track_uses_only_qualifying_view_and_points_for_reconstruction(tmp_path):
    run_dir = tmp_path / "run"
    observations_dir = run_dir / "observations"
    observations_dir.mkdir(parents=True)
    near = make_observation("near", 1, 0.0, timestamp=1_000_000_000)
    near.median_range_m = 30.0
    near.score = 0.80
    far = make_observation("far", 2, 0.1, timestamp=3_000_000_000)
    far.median_range_m = 50.0
    far.score = 0.99
    far.mask_area_px = 1000
    keyframes = []
    for value, offset in ((near, 0.0), (far, 10.0)):
        frame_dir = run_dir / "frames" / value.frame_id
        frame_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(frame_dir / "rgb.png")
        Image.new("L", (4, 4), 255).save(observations_dir / f"{value.id}.png")
        points = np.asarray(
            [[offset, 0.0, 0.0], [offset, 0.1, 0.0], [offset, 0.0, 0.1]],
            dtype=np.float32,
        )
        np.savez_compressed(observations_dir / f"{value.id}.npz", points_world=points)
        value.mask_path = f"observations/{value.id}.png"
        value.points_path = f"observations/{value.id}.npz"
        keyframes.append({"id": value.frame_id, "rgb": f"frames/{value.frame_id}/rgb.png"})
    atomic_write_json(
        run_dir / "route-manifest.json",
        {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "stages": {},
            "keyframes": keyframes,
        },
    )

    track = _track_document(
        run_dir,
        "track-000001",
        [near, far],
        dynamic_min_speed_mps=0.5,
        max_mesh_range_m=30.0,
    )

    assert track["status"] == "pending"
    assert track["selected_observation_id"] == "near"
    assert track["quality_gate"] == {
        "min_inlier_fraction": 0.2,
        "accepted_observation_count": 2,
        "accepted_observation_ids": ["near", "far"],
        "rejected_observation_count": 0,
        "rejected_observation_ids": [],
    }
    assert track["range_gate"] == {
        "max_mesh_range_m": 30.0,
        "minimum_observation_range_m": 30.0,
        "eligible": True,
        "eligible_observation_count": 1,
        "eligible_observation_ids": ["near"],
        "reason": "1 observation(s) are at or below 30 m",
    }
    with np.load(run_dir / track["points"]) as values:
        assert len(values["points_world"]) == 6
    with np.load(run_dir / track["reconstruction_points"]) as values:
        np.testing.assert_allclose(values["points_world"][:, 0], 0.0)
    np.testing.assert_allclose(track["reconstruction_centroid_world"], near.centroid)


def test_low_inlier_depth_leaks_cannot_confirm_or_source_reconstruction(tmp_path):
    run_dir = tmp_path / "run"
    observations_dir = run_dir / "observations"
    observations_dir.mkdir(parents=True)
    first_leak = make_observation(
        "large-mask-leak-a", 1, 5.0, timestamp=1_000_000_000
    )
    second_leak = make_observation(
        "large-mask-leak-b", 2, 5.1, timestamp=3_000_000_000
    )
    reliable = make_observation(
        "credible-partial-view", 3, 5.0, timestamp=5_000_000_000
    )
    first_leak.score = 0.98
    first_leak.mask_area_px = 20_000
    first_leak.inlier_fraction = 0.0011
    second_leak.score = 0.97
    second_leak.mask_area_px = 20_000
    second_leak.inlier_fraction = 0.0022
    reliable.score = 0.75
    reliable.mask_area_px = 350
    reliable.inlier_fraction = 0.72
    keyframes = []
    for index, value in enumerate((first_leak, second_leak, reliable)):
        frame_dir = run_dir / "frames" / value.frame_id
        frame_dir.mkdir(parents=True)
        Image.new("RGB", (4, 4)).save(frame_dir / "rgb.png")
        Image.new("L", (4, 4), 255).save(observations_dir / f"{value.id}.png")
        points = np.asarray(
            [[5.0, index, 0.0], [5.1, index, 0.0], [5.0, index, 0.1]],
            dtype=np.float32,
        )
        np.savez_compressed(observations_dir / f"{value.id}.npz", points_world=points)
        value.mask_path = f"observations/{value.id}.png"
        value.points_path = f"observations/{value.id}.npz"
        keyframes.append(
            {"id": value.frame_id, "rgb": f"frames/{value.frame_id}/rgb.png"}
        )
    atomic_write_json(
        run_dir / "route-manifest.json",
        {
            "schema": ROUTE_MANIFEST_SCHEMA,
            "stages": {},
            "keyframes": keyframes,
        },
    )

    track = _track_document(
        run_dir,
        "track-000028",
        [first_leak, second_leak, reliable],
        dynamic_min_speed_mps=0.5,
        max_mesh_range_m=30.0,
    )

    assert track["motion_state"] == "unconfirmed"
    assert track["status"] == "unconfirmed_skipped"
    assert track["selected_observation_id"] == reliable.id
    assert len(track["observations"]) == 3
    assert track["quality_gate"] == {
        "min_inlier_fraction": 0.2,
        "accepted_observation_count": 1,
        "accepted_observation_ids": [reliable.id],
        "rejected_observation_count": 2,
        "rejected_observation_ids": [first_leak.id, second_leak.id],
    }
    with np.load(run_dir / track["reconstruction_points"]) as values:
        assert len(values["points_world"]) == 3
        np.testing.assert_allclose(values["points_world"][:, 1], 2.0)


@pytest.mark.skipif(
    not os.environ.get("SAM3_ROUTE_YOSEMITE_FIXTURE"),
    reason="set SAM3_ROUTE_YOSEMITE_FIXTURE for the model-free real-artifact regression",
)
def test_yosemite_vehicle_artifacts_yield_one_static_bus(tmp_path):
    source = Path(os.environ["SAM3_ROUTE_YOSEMITE_FIXTURE"]).resolve()
    for name in ("route-manifest.json", "calibration.json", "calibration.npz"):
        shutil.copy2(source / name, tmp_path / name)
    shutil.copytree(source / "frames", tmp_path / "frames")
    manifest = read_route_manifest(tmp_path)
    values = dict(manifest["stages"]["segment"]["config"])
    values["prompts"] = tuple(values["prompts"])
    values["dynamic_min_speed_mps"] = 0.5

    tracks_path = build_tracks(tmp_path, manifest, SegmentConfig(**values))

    document = json.loads(tracks_path.read_text())
    by_state = {}
    for track in document["tracks"]:
        by_state.setdefault(track["motion_state"], []).append(track)
    assert len(by_state["confirmed_static"]) == 1
    assert len(by_state["dynamic"]) == 3
    assert len(by_state["unconfirmed"]) == 2
    bus = by_state["confirmed_static"][0]
    assert len(bus["observations"]) == 7
    np.testing.assert_allclose(bus["centroid_world"], [21.3, -69.2, -1.3], atol=0.5)
    assert all(value["median_range_m"] > 70.0 for value in bus["observations"])
    assert bus["status"] == "range_skipped"
    assert bus["range_gate"]["eligible"] is False
    assert bus["range_gate"]["eligible_observation_count"] == 0
    assert bus["reconstruction_points"] is None

    def unexpected_model_load(*args, **kwargs):
        pytest.fail("SAM3D must not load when every track is skipped")

    scene_path, failures = reconstruct_route(
        tmp_path,
        ReconstructConfig(sam3d_config=str(tmp_path / "pipeline.yaml")),
        overwrite=True,
        inference_factory=unexpected_model_load,
        image_loader=unexpected_model_load,
    )

    assert failures == 0
    assert scene_path.read_bytes()[:4] == b"glTF"
    scene_document = json.loads((tmp_path / "scene.json").read_text())
    assert scene_document["meshes"] == []
