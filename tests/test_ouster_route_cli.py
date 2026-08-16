import pytest

from sam3_route.cli import build_parser


def test_route_parser_exposes_stages_and_aggressive_mesh_defaults():
    parser = build_parser()
    extract = parser.parse_args(
        ["extract", "route.osf", "--output-dir", "output"]
    )
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

    assert extract.keyframe_distance_m == 5.0
    assert extract.keyframe_angle_deg == 5.0
    assert extract.start_frame is None
    assert extract.stop_frame is None
    assert extract_window.start_frame == 101
    assert extract_window.stop_frame == 200
    assert run_window.start_frame == 301
    assert run_window.stop_frame == 400
    assert segment.dynamic_min_speed_mps == 0.5
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

    disabled = parser.parse_args(["reconstruct", "output", "--fit-mode", "none"])
    assert disabled.fit_mode == "none"
    with pytest.raises(SystemExit):
        parser.parse_args(["reconstruct", "output", "--fit-mode", "legacy"])
