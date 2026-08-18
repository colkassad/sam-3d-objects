import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import trimesh
from PIL import Image

from sam3_masking.artifacts import write_mask_manifest
from sam3_masking.types import MaskFrame, MaskPrediction
from sam3_route.artifacts import ROUTE_MANIFEST_SCHEMA, atomic_write_json
from sam3_route.cli import build_parser, run_cli
from sam3_route.surface import (
    SurfacePointSet,
    SurfaceSegmentConfig,
    TinConfig,
    build_surface_tin,
    collect_surface_points,
    segment_surface_route,
    triangulate_surface,
)


def _base_manifest(keyframes, *, stages=None):
    return {
        "schema": ROUTE_MANIFEST_SCHEMA,
        "source": {},
        "coordinate_system": {
            "type": "local_slam",
            "units": "meters",
            "georeferenced": False,
        },
        "stages": stages or {"extract": {"status": "complete"}},
        "calibration": None,
        "trajectory": None,
        "keyframes": keyframes,
        "prompts": ["car"],
        "surface_prompts": [],
        "tracks": "tracks.json",
        "outputs": {"scene_glb": "scene.glb"},
        "software": {},
    }


def test_surface_segmentation_preserves_prompts_and_object_artifacts(tmp_path):
    run_dir = tmp_path / "run"
    frame_dir = run_dir / "frames" / "frame-000001"
    frame_dir.mkdir(parents=True)
    Image.new("RGB", (4, 3)).save(frame_dir / "rgb.png")
    manifest_path = run_dir / "route-manifest.json"
    atomic_write_json(
        manifest_path,
        _base_manifest(
            [
                {
                    "id": "frame-000001",
                    "rgb": "frames/frame-000001/rgb.png",
                    "geometry": "frames/frame-000001/geometry.npz",
                    "mask_manifest": "frames/frame-000001/segmentation/manifest.json",
                    "surface_mask_manifest": None,
                }
            ]
        ),
    )
    commands = []

    def fake_run(command, check):
        commands.append(command)
        surface_manifest = frame_dir / "surface-segmentation" / "manifest.json"
        atomic_write_json(surface_manifest, {"predictions": []})
        document = json.loads(manifest_path.read_text())
        document["keyframes"][0]["surface_mask_manifest"] = (
            "frames/frame-000001/surface-segmentation/manifest.json"
        )
        document["surface_prompts"] = ["dirt track", "gravel carriageway"]
        atomic_write_json(manifest_path, document)
        return SimpleNamespace(returncode=0)

    config = SurfaceSegmentConfig(
        prompts=("dirt track", "gravel carriageway"),
        sam3_model_dir="model",
        sam3_executable="sam3-mask-route",
    )
    segment_surface_route(run_dir, config, subprocess_run=fake_run)

    assert len(commands) == 1
    assert commands[0][commands[0].index("--prompts") + 1] == (
        "dirt track,gravel carriageway"
    )
    assert commands[0][commands[0].index("--artifact-set") + 1] == "surface"
    document = json.loads(manifest_path.read_text())
    assert document["surface_prompts"] == ["dirt track", "gravel carriageway"]
    assert document["prompts"] == ["car"]
    assert document["outputs"]["scene_glb"] == "scene.glb"
    assert document["stages"]["surface_segment"]["status"] == "complete"

    segment_surface_route(
        run_dir,
        config,
        subprocess_run=lambda *args, **kwargs: pytest.fail("current stage reran SAM 3"),
    )
    with pytest.raises(RuntimeError, match="configuration changed"):
        segment_surface_route(
            run_dir,
            SurfaceSegmentConfig(
                prompts=("paved carriageway",),
                sam3_model_dir="model",
                sam3_executable="sam3-mask-route",
            ),
            subprocess_run=fake_run,
        )


def test_collect_surface_points_unions_masks_filters_range_and_preserves_rgb(tmp_path):
    run_dir = tmp_path / "run"
    frame_dir = run_dir / "frames" / "frame-000001"
    frame_dir.mkdir(parents=True)
    height, width = 2, 4
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for row in range(height):
        for column in range(width):
            rgb[row, column] = [10 * column, 20 * row, 100]
    Image.fromarray(rgb).save(frame_dir / "rgb.png")
    ranges = np.full((height, width), 1000, dtype=np.uint32)
    ranges[0, 0] = 0
    ranges[1, 3] = 40_000
    poses = np.repeat(np.eye(4)[None], width, axis=0)
    poses[:, 0, 3] = np.arange(width) * 0.3
    np.savez_compressed(
        frame_dir / "geometry.npz",
        range_mm=ranges,
        body_to_world=poses,
        column_timestamp_ns=np.arange(width),
        reference_column=np.asarray(2, dtype=np.int32),
    )
    directions = np.zeros((height, width, 3), dtype=np.float64)
    directions[..., 0] = 1.0
    directions[..., 1] = (np.arange(width) - 1.5) * 0.2
    directions[1, :, 2] = 0.2
    np.savez_compressed(
        run_dir / "calibration.npz",
        sensor_to_body=np.eye(4),
        ray_direction=directions,
        ray_origin=np.zeros_like(directions),
    )
    atomic_write_json(run_dir / "calibration.json", {"arrays": "calibration.npz"})
    first = np.zeros((height, width), dtype=bool)
    first[0, :3] = True
    first[1, 1] = True
    second = np.zeros((height, width), dtype=bool)
    second[0, 2:] = True
    second[1, [1, 3]] = True
    mask_path = write_mask_manifest(
        MaskFrame(
            width,
            height,
            (
                MaskPrediction("p000-i000", "dirt track", 0.9, (0, 0, 4, 2), first),
                MaskPrediction(
                    "p001-i000",
                    "gravel carriageway",
                    0.8,
                    (0, 0, 4, 2),
                    second,
                ),
            ),
            source_id="frame-000001",
        ),
        frame_dir / "surface-segmentation",
        image_path=frame_dir / "rgb.png",
    )
    manifest = _base_manifest(
        [
            {
                "id": "frame-000001",
                "rgb": "frames/frame-000001/rgb.png",
                "geometry": "frames/frame-000001/geometry.npz",
                "mask_manifest": "object-manifest.json",
                "surface_mask_manifest": mask_path.relative_to(run_dir).as_posix(),
            }
        ]
    )
    manifest["calibration"] = "calibration.json"

    result = collect_surface_points(
        run_dir,
        manifest,
        TinConfig(surface_resolution_m=0.05, max_surface_range_m=30.0),
    )

    assert result.statistics["prediction_count"] == 2
    assert result.statistics["union_mask_pixels"] == 6
    assert result.statistics["valid_surface_returns"] == 4
    assert result.statistics["fused_point_count"] == 3
    index = np.argmin(np.linalg.norm(result.points[:, :2] - [1.3, -0.1], axis=1))
    np.testing.assert_allclose(result.points[index], [1.3, -0.1, 0.1], atol=1e-6)
    np.testing.assert_array_equal(result.colors[index], [10, 10, 100])


def _l_shaped_surface():
    coordinates = []
    for x in np.arange(0.0, 2.01, 0.2):
        for y in np.arange(0.0, 2.01, 0.2):
            if x <= 0.8 or y <= 0.8:
                coordinates.append([x, y, 0.02 * x + 0.01 * y])
    first = np.asarray(coordinates, dtype=np.float64)
    second = first + [4.0, 0.0, 0.2]
    return np.vstack((first, second))


def test_tiled_tin_clips_concavity_and_keeps_disconnected_components():
    points = _l_shaped_surface()
    faces, report = triangulate_surface(
        points,
        TinConfig(
            surface_resolution_m=0.20,
            max_surface_range_m=None,
            max_triangle_edge_m=0.45,
            max_slope_deg=45.0,
            tin_tile_size_m=1.0,
        ),
    )

    triangles = points[faces]
    centroids = np.mean(triangles, axis=1)
    first_component = centroids[:, 0] < 3.0
    assert not np.any(
        first_component & (centroids[:, 0] > 0.9) & (centroids[:, 1] > 0.9)
    )
    assert not np.any(
        (np.min(triangles[:, :, 0], axis=1) < 3.0)
        & (np.max(triangles[:, :, 0], axis=1) > 3.0)
    )
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    assert np.all(normals[:, 2] > 0)
    assert report["components"]["count"] == 2
    assert report["tile_count"] > 2


def _surface_with_internal_gap():
    rng = np.random.default_rng(4)
    coordinates = []
    for x in np.arange(0.0, 6.01, 0.2):
        for y in np.arange(0.0, 2.01, 0.2):
            if 2.7 < x < 3.3 and 0.4 < y < 1.6:
                continue
            jitter = rng.normal(0.0, 0.001, 2)
            coordinates.append([x + jitter[0], y + jitter[1], 0.01 * x])
    return np.asarray(coordinates, dtype=np.float64)


def _gap_tin_config(**overrides):
    values = {
        "surface_resolution_m": 0.2,
        "max_surface_range_m": None,
        "max_triangle_edge_m": 0.35,
        "max_slope_deg": 45.0,
        "tin_tile_size_m": 3.0,
    }
    values.update(overrides)
    return TinConfig(**values)


def test_hole_filling_is_opt_in_width_limited_and_crosses_tile_seams():
    points = _surface_with_internal_gap()
    default_faces, default_report = triangulate_surface(points, _gap_tin_config())
    disabled_faces, disabled_report = triangulate_surface(
        points, _gap_tin_config(fill_holes=False)
    )
    narrow_faces, narrow_report = triangulate_surface(
        points,
        _gap_tin_config(fill_holes=True, max_hole_width_m=0.5),
    )
    filled_faces, filled_report = triangulate_surface(
        points,
        _gap_tin_config(fill_holes=True, max_hole_width_m=1.0),
    )

    np.testing.assert_array_equal(default_faces, disabled_faces)
    assert not default_report["hole_fill"]["enabled"]
    assert not disabled_report["hole_fill"]["evaluated"]
    assert len(narrow_faces) == len(default_faces)
    assert narrow_report["hole_fill"]["skipped_too_wide_count"] == 1
    assert len(filled_faces) > len(default_faces)
    assert filled_report["hole_fill"]["filled_region_count"] == 1
    assert filled_report["hole_fill"]["filled_face_count"] == (
        len(filled_faces) - len(default_faces)
    )
    assert filled_report["hole_fill"]["filled_width_m"]["maximum"] <= 1.0
    assert filled_report["duplicate_face_count"] == 0
    canonical = np.sort(filled_faces, axis=1)
    assert len(np.unique(canonical, axis=0)) == len(filled_faces)


def test_hole_filling_uses_width_instead_of_total_gap_area():
    rng = np.random.default_rng(8)
    coordinates = []
    for x in np.arange(0.0, 6.01, 0.2):
        for y in np.arange(0.0, 2.01, 0.2):
            if 1.0 < x < 5.0 and 0.7 < y < 1.3:
                continue
            jitter = rng.normal(0.0, 0.001, 2)
            coordinates.append([x + jitter[0], y + jitter[1], 0.0])
    points = np.asarray(coordinates, dtype=np.float64)

    baseline, _ = triangulate_surface(points, _gap_tin_config())
    repaired, report = triangulate_surface(
        points,
        _gap_tin_config(fill_holes=True, max_hole_width_m=1.0),
    )

    assert len(repaired) > len(baseline)
    assert report["hole_fill"]["filled_region_count"] >= 1
    assert report["hole_fill"]["filled_width_m"]["maximum"] <= 1.0


def test_hole_filling_preserves_exterior_concavities_and_disconnected_surfaces():
    points = _l_shaped_surface()
    baseline, _ = triangulate_surface(
        points,
        TinConfig(
            surface_resolution_m=0.2,
            max_surface_range_m=None,
            max_triangle_edge_m=0.45,
            tin_tile_size_m=12.0,
        ),
    )
    repaired, report = triangulate_surface(
        points,
        TinConfig(
            surface_resolution_m=0.2,
            max_surface_range_m=None,
            max_triangle_edge_m=0.45,
            tin_tile_size_m=12.0,
            fill_holes=True,
            max_hole_width_m=5.0,
        ),
    )

    assert len(repaired) == len(baseline)
    assert report["hole_fill"]["filled_face_count"] == 0
    assert report["hole_fill"]["exterior_region_count"] > 0
    assert report["components"]["count"] == 2


def test_hole_filling_does_not_restore_steep_rejected_faces():
    points = np.asarray(
        [
            [x, y, 2.0 if x == 1.0 and y == 1.0 else 0.0]
            for x in np.arange(0.0, 2.01, 0.2)
            for y in np.arange(0.0, 2.01, 0.2)
        ],
        dtype=np.float64,
    )
    _, report = triangulate_surface(
        points,
        TinConfig(
            surface_resolution_m=0.2,
            max_surface_range_m=None,
            max_triangle_edge_m=0.35,
            tin_tile_size_m=5.0,
            fill_holes=True,
            max_hole_width_m=1.0,
        ),
    )

    assert report["rejected_faces"]["slope"] > 0
    assert report["hole_fill"]["enclosed_region_count"] == 1
    assert report["hole_fill"]["filled_face_count"] == 0
    assert report["hole_fill"]["skipped_unsafe_count"] == 1


def test_tin_rejects_collinear_and_excessively_steep_surfaces():
    with pytest.raises(ValueError, match="collinear"):
        triangulate_surface(
            np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float),
            TinConfig(),
        )
    steep = np.asarray(
        [[x, y, 2.0 * x] for x in (0.0, 0.2, 0.4) for y in (0.0, 0.2, 0.4)]
    )
    with pytest.raises(RuntimeError, match="no triangles"):
        triangulate_surface(
            steep,
            TinConfig(max_triangle_edge_m=0.5, tin_tile_size_m=2.0),
        )


def _write_build_manifest(run_dir):
    atomic_write_json(
        run_dir / "route-manifest.json",
        _base_manifest(
            [],
            stages={
                "extract": {"status": "complete"},
                "surface_segment": {"status": "complete"},
            },
        ),
    )
    document = json.loads((run_dir / "route-manifest.json").read_text())
    document["surface_prompts"] = ["paved carriageway"]
    atomic_write_json(run_dir / "route-manifest.json", document)


def test_build_exports_colored_glb_metadata_and_resumes_without_sam3(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_build_manifest(run_dir)
    points = _surface_with_internal_gap().astype(np.float32)
    colors = np.tile(np.asarray([[90, 80, 70]], dtype=np.uint8), (len(points), 1))
    point_set = SurfacePointSet(
        points,
        colors,
        {"fused_point_count": len(points), "frames": []},
    )
    monkeypatch.setattr(
        "sam3_route.surface.collect_surface_points",
        lambda *args, **kwargs: point_set,
    )
    config = TinConfig(
        surface_resolution_m=0.2,
        max_surface_range_m=None,
        max_triangle_edge_m=0.35,
        tin_tile_size_m=3.0,
        fill_holes=True,
        max_hole_width_m=1.0,
    )

    outputs = build_surface_tin(run_dir, config)

    assert outputs.point_cloud.read_bytes().startswith(
        b"ply\nformat binary_little_endian"
    )
    assert outputs.mesh.read_bytes()[:4] == b"glTF"
    scene = trimesh.load_scene(outputs.mesh)
    assert len(scene.geometry) == 1
    mesh = next(iter(scene.geometry.values()))
    assert len(mesh.faces) > 0
    assert not mesh.is_watertight
    assert np.all(mesh.face_normals[:, 2] > 0)
    np.testing.assert_array_equal(mesh.visual.vertex_colors[0, :3], [90, 80, 70])
    metadata = json.loads(outputs.metadata.read_text())
    assert metadata["status"] == "complete"
    assert metadata["prompts"] == ["paved carriageway"]
    assert metadata["world_from_glb"] == np.eye(4).tolist()
    assert metadata["triangulation"]["hole_fill"]["filled_region_count"] == 1
    assert metadata["triangulation"]["hole_fill"]["filled_face_count"] > 0
    assert len(mesh.faces) == metadata["triangulation"]["face_count"]
    manifest = json.loads((run_dir / "route-manifest.json").read_text())
    assert manifest["outputs"]["scene_glb"] == "scene.glb"
    assert manifest["outputs"]["surface_glb"] == "surface/surface.glb"

    monkeypatch.setattr(
        "sam3_route.surface.collect_surface_points",
        lambda *args, **kwargs: pytest.fail("current build recomputed surface points"),
    )
    assert build_surface_tin(run_dir, config) == outputs
    with pytest.raises(RuntimeError, match="configuration changed"):
        build_surface_tin(
            run_dir,
            TinConfig(
                surface_resolution_m=0.25,
                max_surface_range_m=None,
                max_triangle_edge_m=0.5,
                tin_tile_size_m=2.0,
            ),
        )


def test_failed_build_writes_diagnostics_without_stale_glb(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_build_manifest(run_dir)
    monkeypatch.setattr(
        "sam3_route.surface.collect_surface_points",
        lambda *args, **kwargs: SurfacePointSet(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint8),
            {"fused_point_count": 0, "frames": []},
        ),
    )

    with pytest.raises(ValueError, match="at least three"):
        build_surface_tin(run_dir, TinConfig())

    assert not (run_dir / "surface" / "surface.glb").exists()
    metadata = json.loads((run_dir / "surface" / "surface.json").read_text())
    assert metadata["status"] == "failed"
    manifest = json.loads((run_dir / "route-manifest.json").read_text())
    assert manifest["stages"]["surface_tin"]["status"] == "failed"
    assert "surface_glb" not in manifest["outputs"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"surface_resolution_m": 0},
        {"max_surface_range_m": 0},
        {"max_triangle_edge_m": -1},
        {"max_hole_width_m": 0},
        {"max_slope_deg": 90},
        {"tin_tile_size_m": 2, "max_triangle_edge_m": 1},
        {
            "fill_holes": True,
            "max_triangle_edge_m": 1,
            "max_hole_width_m": 2,
            "tin_tile_size_m": 6,
        },
    ],
)
def test_tin_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TinConfig(**kwargs)


def test_surface_cli_hole_fill_options_are_opt_in():
    parser = build_parser()
    defaults = parser.parse_args(["surface", "build", "run"])
    enabled = parser.parse_args(
        [
            "surface",
            "build",
            "run",
            "--fill-holes",
            "--max-hole-width-m",
            "1.5",
        ]
    )
    disabled = parser.parse_args(
        ["surface", "build", "run", "--no-fill-holes"]
    )

    assert defaults.fill_holes is False
    assert defaults.max_hole_width_m == 1.0
    assert enabled.fill_holes is True
    assert enabled.max_hole_width_m == 1.5
    assert disabled.fill_holes is False


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("RUN_SAM3_SURFACE_INTEGRATION") != "1",
    reason="set RUN_SAM3_SURFACE_INTEGRATION=1 to run",
)
def test_real_ouster_surface_masks_feed_tin_generation(tmp_path):
    source = os.environ.get("SAM3_ROUTE_OSF_FIXTURE")
    executable = os.environ.get("SAM3_MASK_ROUTE_EXECUTABLE")
    model_dir = os.environ.get("SAM3_MODEL_DIR", "checkpoints/sam3-hf")
    if not source or not executable:
        pytest.fail(
            "SAM3_ROUTE_OSF_FIXTURE and SAM3_MASK_ROUTE_EXECUTABLE are required"
        )
    args = build_parser().parse_args(
        [
            "surface",
            "run",
            source,
            "--output-dir",
            str(tmp_path / "surface-run"),
            "--max-scans",
            os.environ.get("SAM3_SURFACE_MAX_SCANS", "5"),
            "--keyframe-distance-m",
            "0.1",
            "--prompts",
            os.environ.get("SAM3_SURFACE_TEST_PROMPT", "drivable surface"),
            "--sam3-executable",
            executable,
            "--sam3-model-dir",
            model_dir,
            "--surface-resolution-m",
            "0.5",
        ]
    )

    assert run_cli(args) == 0
    output = tmp_path / "surface-run" / "surface"
    assert (output / "surface-points.ply").stat().st_size > 0
    assert (output / "surface.glb").read_bytes()[:4] == b"glTF"
    metadata = json.loads((output / "surface.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["triangulation"]["face_count"] > 0
