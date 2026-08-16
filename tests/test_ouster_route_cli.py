import pytest

from sam3_route.cli import build_parser


def test_route_parser_exposes_stages_and_aggressive_mesh_defaults():
    parser = build_parser()
    extract = parser.parse_args(["extract", "route.osf", "--output-dir", "output"])
    extract_window = parser.parse_args(
        [
            "extract",
            "route.osf",
            "--output-dir",
            "output",
            "--start-frame",
            "101",
            "--stop-frame",
            "200",
        ]
    )
    run_window = parser.parse_args(
        [
            "run",
            "route.osf",
            "--output-dir",
            "output",
            "--prompt",
            "vehicle",
            "--start-frame",
            "301",
            "--stop-frame",
            "400",
        ]
    )
    segment = parser.parse_args(["segment", "output", "--prompt", "vehicle"])
    track = parser.parse_args(["track", "output", "--overwrite"])
    segment_unlimited = parser.parse_args(
        ["segment", "output", "--prompt", "vehicle", "--no-max-mesh-range"]
    )
    track_unlimited = parser.parse_args(
        ["track", "output", "--no-max-mesh-range", "--overwrite"]
    )
    track_custom = parser.parse_args(
        ["track", "output", "--max-mesh-range-m", "12.5", "--overwrite"]
    )
    reconstruct = parser.parse_args(["reconstruct", "output"])
    scene_build = parser.parse_args(["scene", "build", "output"])
    surface_run = parser.parse_args(
        [
            "surface",
            "run",
            "route.osf",
            "--output-dir",
            "surface-output",
            "--prompt",
            "dirt track",
        ]
    )
    surface_segment = parser.parse_args(
        ["surface", "segment", "surface-output", "--prompt", "gravel carriageway"]
    )
    surface_build = parser.parse_args(["surface", "build", "surface-output"])

    assert extract.keyframe_distance_m == 5.0
    assert extract.keyframe_angle_deg == 5.0
    assert extract.start_frame is None
    assert extract.stop_frame is None
    assert extract_window.start_frame == 101
    assert extract_window.stop_frame == 200
    assert run_window.start_frame == 301
    assert run_window.stop_frame == 400
    assert segment.dynamic_min_speed_mps == 0.5
    assert segment.suppress_duplicate_tracks is True
    assert segment.duplicate_track_max_centroid_m == 1.0
    assert segment.duplicate_track_min_shared_fraction == 0.50
    assert segment.duplicate_track_min_containment == 0.30
    assert segment.max_mesh_range_m == 30.0
    assert track.dynamic_min_speed_mps == 0.5
    assert track.max_mesh_range_m == 30.0
    assert segment_unlimited.max_mesh_range_m is None
    assert track_unlimited.max_mesh_range_m is None
    assert track_custom.max_mesh_range_m == 12.5
    assert reconstruct.mesh_target_faces == 10_000
    assert reconstruct.stage1_inference_steps == 15
    assert reconstruct.stage2_inference_steps == 15
    assert reconstruct.flat_shading is True
    assert reconstruct.memory_profile == "low_vram"
    assert reconstruct.fit_mode == "raycast"
    assert reconstruct.fit_max_axis_scale_change == 0.25
    assert reconstruct.fit_max_rays_per_view == 2_000
    assert reconstruct.fit_max_views == 5
    assert reconstruct.fit_max_evaluations == 160
    assert reconstruct.fit_grounded is True
    assert reconstruct.fit_align_long_axis is True
    assert reconstruct.fit_max_up_tilt_deg == 20.0
    assert reconstruct.suppress_overlapping_meshes is True
    assert reconstruct.mesh_overlap_min_iou == 0.35
    assert reconstruct.mesh_overlap_min_containment == 0.75
    assert reconstruct.mesh_vertical_overlap_min == 0.50
    assert reconstruct.mesh_overlap_resolution_m == 0.10
    assert scene_build.scene_command == "build"
    assert scene_build.suppress_overlapping_meshes is True
    assert surface_run.surface_command == "run"
    assert surface_run.keyframe_distance_m == 1.0
    assert surface_run.prompt == ["dirt track"]
    assert surface_run.surface_resolution_m == 0.20
    assert surface_run.max_surface_range_m == 30.0
    assert surface_segment.prompt == ["gravel carriageway"]
    assert surface_build.max_triangle_edge_m == 1.0
    assert surface_build.max_slope_deg == 45.0
    assert surface_build.tin_tile_size_m == 50.0

    disabled = parser.parse_args(["reconstruct", "output", "--fit-mode", "none"])
    assert disabled.fit_mode == "none"
    duplicate_disabled = parser.parse_args(
        ["track", "output", "--no-suppress-duplicate-tracks"]
    )
    overlap_disabled = parser.parse_args(
        ["scene", "build", "output", "--no-suppress-overlapping-meshes"]
    )
    assert duplicate_disabled.suppress_duplicate_tracks is False
    assert overlap_disabled.suppress_overlapping_meshes is False
    with pytest.raises(SystemExit):
        parser.parse_args(["reconstruct", "output", "--fit-mode", "legacy"])
