from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Sequence

from .extract import ExtractConfig, extract_route
from .reconstruct import ReconstructConfig, reconstruct_route
from .surface import (
    SurfaceSegmentConfig,
    TinConfig,
    build_surface_tin,
    generate_surface_route,
    segment_surface_route,
)
from .tracking import SegmentConfig, retrack_route, segment_route


REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_repo_path(value: Path) -> Path:
    path = value.expanduser()
    return (path if path.is_absolute() else REPO_ROOT / path).resolve()


def discover_batch_executable(explicit: Optional[str]) -> Path:
    candidates: list[tuple[str, Optional[str]]] = [
        ("--sam3-executable", explicit),
        ("SAM3_MASK_ROUTE_EXECUTABLE", os.environ.get("SAM3_MASK_ROUTE_EXECUTABLE")),
    ]
    executable_name = "sam3-mask-route.exe" if os.name == "nt" else "sam3-mask-route"
    executable_dir = "Scripts" if os.name == "nt" else "bin"
    sibling = (
        Path(sys.prefix).resolve().parent
        / "sam3-masking"
        / executable_dir
        / executable_name
    )
    candidates.append(("sibling sam3-masking environment", str(sibling)))
    candidates.append(("PATH", shutil.which("sam3-mask-route")))
    for _, value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if not path.is_absolute() and os.sep not in value:
            located = shutil.which(value)
            if located:
                path = Path(located)
        path = path.resolve()
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError(
        "could not find sam3-mask-route; install this project into the sibling "
        "sam3-masking environment, set SAM3_MASK_ROUTE_EXECUTABLE, or pass "
        "--sam3-executable"
    )


def _add_extract_options(
    parser: argparse.ArgumentParser, *, include_source: bool
) -> None:
    if include_source:
        parser.add_argument("source", type=Path)
        parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    parser.add_argument("--keyframe-distance-m", type=float, default=5.0)
    parser.add_argument("--keyframe-angle-deg", type=float, default=5.0)
    parser.add_argument("--slam-min-range-m", type=float, default=1.0)
    parser.add_argument("--slam-max-range-m", type=float, default=75.0)
    parser.add_argument("--slam-voxel-size-m", type=float, default=1.0)
    parser.add_argument(
        "--point-cloud",
        type=str,
        help="Optional run-relative binary PLY path, for example route.ply.",
    )
    parser.add_argument("--point-cloud-voxel-m", type=float, default=0.10)
    parser.add_argument("--max-scans", type=int)
    parser.add_argument(
        "--start-frame",
        type=int,
        help="First included OSF frame, using 1-based recording indices.",
    )
    parser.add_argument(
        "--stop-frame",
        type=int,
        help="Last included OSF frame, using inclusive 1-based indices.",
    )


def _add_segment_options(
    parser: argparse.ArgumentParser, *, require_prompt: bool
) -> None:
    parser.add_argument(
        "--prompt",
        action="append",
        required=require_prompt,
        help="Repeat for each object concept.",
    )
    parser.add_argument(
        "--sam3-model-dir", type=Path, default=Path("checkpoints/sam3-hf")
    )
    parser.add_argument("--sam3-executable")
    parser.add_argument("--sam3-device", default="auto")
    parser.add_argument(
        "--sam3-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--min-range-points", type=int, default=10)
    parser.add_argument("--dynamic-min-speed-mps", type=float, default=0.5)
    _add_mesh_range_options(parser)


def _add_mesh_range_options(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--max-mesh-range-m",
        type=float,
        help="Maximum cleaned median LiDAR range eligible for mesh reconstruction.",
    )
    group.add_argument(
        "--no-max-mesh-range",
        action="store_const",
        const=None,
        dest="max_mesh_range_m",
        help="Disable the mesh reconstruction range limit.",
    )
    parser.set_defaults(max_mesh_range_m=30.0)


def _add_surface_segment_options(
    parser: argparse.ArgumentParser, *, require_prompt: bool
) -> None:
    parser.add_argument(
        "--prompt",
        action="append",
        required=require_prompt,
        help="Repeat for every text description of the target surface.",
    )
    parser.add_argument(
        "--sam3-model-dir", type=Path, default=Path("checkpoints/sam3-hf")
    )
    parser.add_argument("--sam3-executable")
    parser.add_argument("--sam3-device", default="auto")
    parser.add_argument(
        "--sam3-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)


def _add_tin_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--surface-resolution-m", type=float, default=0.20)
    range_group = parser.add_mutually_exclusive_group()
    range_group.add_argument("--max-surface-range-m", type=float)
    range_group.add_argument(
        "--no-max-surface-range",
        action="store_const",
        const=None,
        dest="max_surface_range_m",
    )
    parser.set_defaults(max_surface_range_m=30.0)
    parser.add_argument("--max-triangle-edge-m", type=float, default=1.0)
    parser.add_argument("--max-slope-deg", type=float, default=45.0)
    parser.add_argument("--tin-tile-size-m", type=float, default=50.0)


def _add_reconstruct_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--sam3d-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml")
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh-target-faces", type=int, default=10_000)
    parser.add_argument("--stage1-inference-steps", type=int, default=15)
    parser.add_argument("--stage2-inference-steps", type=int, default=15)
    parser.add_argument(
        "--flat-shading", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--memory-profile", choices=("auto", "low_vram", "resident"), default="low_vram"
    )
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument("--fit-mode", choices=("raycast", "none"), default="raycast")
    parser.add_argument("--fit-max-axis-scale-change", type=float, default=0.25)
    parser.add_argument("--fit-max-rays-per-view", type=int, default=2_000)
    parser.add_argument("--fit-max-views", type=int, default=5)
    parser.add_argument("--fit-max-evaluations", type=int, default=160)
    parser.add_argument("--fit-max-rotation-deg", type=float, default=20.0)
    parser.add_argument(
        "--fit-grounded", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fit-align-long-axis", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--fit-max-up-tilt-deg", type=float, default=20.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sam3d-ouster-route",
        description=(
            "Extract, segment, and reconstruct Ouster route objects and surfaces."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="Run SLAM and export motion-selected keyframes."
    )
    _add_extract_options(extract, include_source=True)
    extract.add_argument("--overwrite", action="store_true")

    segment = subparsers.add_parser(
        "segment", help="Run batch SAM3 and associate observations."
    )
    segment.add_argument("run_dir", type=Path)
    _add_segment_options(segment, require_prompt=True)
    segment.add_argument("--overwrite", action="store_true")

    track = subparsers.add_parser(
        "track", help="Rebuild depth hypotheses and tracks from saved SAM3 masks."
    )
    track.add_argument("run_dir", type=Path)
    track.add_argument("--min-range-points", type=int)
    track.add_argument("--dynamic-min-speed-mps", type=float, default=0.5)
    _add_mesh_range_options(track)
    track.add_argument("--overwrite", action="store_true")

    reconstruct = subparsers.add_parser(
        "reconstruct", help="Generate and place one mesh per confirmed-static track."
    )
    reconstruct.add_argument("run_dir", type=Path)
    _add_reconstruct_options(reconstruct)
    reconstruct.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run", help="Run all stages with resumable artifacts.")
    _add_extract_options(run, include_source=True)
    _add_segment_options(run, require_prompt=True)
    _add_reconstruct_options(run)
    run.add_argument("--overwrite", action="store_true")

    surface = subparsers.add_parser(
        "surface",
        help="Segment and triangulate a large prompted route surface without SAM 3D.",
    )
    surface_commands = surface.add_subparsers(dest="surface_command", required=True)
    surface_run = surface_commands.add_parser(
        "run", help="Extract, segment, and build a surface TIN."
    )
    _add_extract_options(surface_run, include_source=True)
    surface_run.set_defaults(keyframe_distance_m=1.0)
    _add_surface_segment_options(surface_run, require_prompt=True)
    _add_tin_options(surface_run)
    surface_run.add_argument("--overwrite", action="store_true")

    surface_segment = surface_commands.add_parser(
        "segment", help="Run SAM 3 into independent surface mask artifacts."
    )
    surface_segment.add_argument("run_dir", type=Path)
    _add_surface_segment_options(surface_segment, require_prompt=True)
    surface_segment.add_argument("--overwrite", action="store_true")

    surface_build = surface_commands.add_parser(
        "build", help="Build a point cloud and TIN from saved surface masks."
    )
    surface_build.add_argument("run_dir", type=Path)
    _add_tin_options(surface_build)
    surface_build.add_argument("--overwrite", action="store_true")
    return parser


def _extract_config(args: argparse.Namespace) -> ExtractConfig:
    return ExtractConfig(
        keyframe_distance_m=args.keyframe_distance_m,
        keyframe_angle_deg=args.keyframe_angle_deg,
        slam_min_range_m=args.slam_min_range_m,
        slam_max_range_m=args.slam_max_range_m,
        slam_voxel_size_m=args.slam_voxel_size_m,
        point_cloud=args.point_cloud,
        point_cloud_voxel_m=args.point_cloud_voxel_m,
        max_scans=args.max_scans,
        start_frame=args.start_frame,
        stop_frame=args.stop_frame,
    )


def _segment_config(args: argparse.Namespace) -> SegmentConfig:
    executable = discover_batch_executable(args.sam3_executable)
    return SegmentConfig(
        prompts=tuple(value.strip() for value in args.prompt),
        sam3_model_dir=str(_resolve_repo_path(args.sam3_model_dir)),
        sam3_executable=str(executable),
        sam3_device=args.sam3_device,
        sam3_dtype=args.sam3_dtype,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        min_range_points=args.min_range_points,
        dynamic_min_speed_mps=args.dynamic_min_speed_mps,
        max_mesh_range_m=args.max_mesh_range_m,
    )


def _reconstruct_config(args: argparse.Namespace) -> ReconstructConfig:
    config_path = _resolve_repo_path(args.sam3d_config)
    if not config_path.is_file():
        raise FileNotFoundError(f"SAM3D configuration does not exist: {config_path}")
    return ReconstructConfig(
        sam3d_config=str(config_path),
        seed=args.seed,
        mesh_target_faces=args.mesh_target_faces,
        stage1_inference_steps=args.stage1_inference_steps,
        stage2_inference_steps=args.stage2_inference_steps,
        flat_shading=args.flat_shading,
        memory_profile=args.memory_profile,
        compile_model=args.compile_model,
        fit_mode=args.fit_mode,
        fit_max_axis_scale_change=args.fit_max_axis_scale_change,
        fit_max_rays_per_view=args.fit_max_rays_per_view,
        fit_max_views=args.fit_max_views,
        fit_max_evaluations=args.fit_max_evaluations,
        fit_max_rotation_deg=args.fit_max_rotation_deg,
        fit_grounded=args.fit_grounded,
        fit_align_long_axis=args.fit_align_long_axis,
        fit_max_up_tilt_deg=args.fit_max_up_tilt_deg,
    )


def _surface_segment_config(args: argparse.Namespace) -> SurfaceSegmentConfig:
    executable = discover_batch_executable(args.sam3_executable)
    return SurfaceSegmentConfig(
        prompts=tuple(value.strip() for value in args.prompt),
        sam3_model_dir=str(_resolve_repo_path(args.sam3_model_dir)),
        sam3_executable=str(executable),
        sam3_device=args.sam3_device,
        sam3_dtype=args.sam3_dtype,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
    )


def _tin_config(args: argparse.Namespace) -> TinConfig:
    return TinConfig(
        surface_resolution_m=args.surface_resolution_m,
        max_surface_range_m=args.max_surface_range_m,
        max_triangle_edge_m=args.max_triangle_edge_m,
        max_slope_deg=args.max_slope_deg,
        tin_tile_size_m=args.tin_tile_size_m,
    )


def run_cli(args: argparse.Namespace) -> int:
    if args.command == "surface":
        if args.surface_command == "segment":
            path = segment_surface_route(
                args.run_dir,
                _surface_segment_config(args),
                overwrite=args.overwrite,
            )
            print(f"Wrote surface mask artifacts: {path}")
            return 0
        if args.surface_command == "build":
            outputs = build_surface_tin(
                args.run_dir, _tin_config(args), overwrite=args.overwrite
            )
            print(f"Wrote surface TIN: {outputs.mesh}")
            print(f"Wrote surface point cloud: {outputs.point_cloud}")
            return 0
        if args.surface_command == "run":
            outputs = generate_surface_route(
                args.source,
                args.output_dir,
                metadata=args.meta,
                extract_config=_extract_config(args),
                segment_config=_surface_segment_config(args),
                tin_config=_tin_config(args),
                overwrite=args.overwrite,
            )
            print(f"Wrote surface TIN: {outputs.mesh}")
            print(f"Wrote surface point cloud: {outputs.point_cloud}")
            return 0
        raise AssertionError(f"unsupported surface command {args.surface_command}")
    if args.command == "extract":
        path = extract_route(
            args.source,
            args.output_dir,
            metadata=args.meta,
            config=_extract_config(args),
            overwrite=args.overwrite,
        )
        print(f"Extracted route artifacts: {path}")
        return 0
    if args.command == "segment":
        path = segment_route(
            args.run_dir,
            _segment_config(args),
            overwrite=args.overwrite,
        )
        print(f"Associated route tracks: {path}")
        return 0
    if args.command == "track":
        path = retrack_route(
            args.run_dir,
            min_range_points=args.min_range_points,
            dynamic_min_speed_mps=args.dynamic_min_speed_mps,
            max_mesh_range_m=args.max_mesh_range_m,
            overwrite=args.overwrite,
        )
        print(f"Rebuilt route tracks without model inference: {path}")
        return 0
    if args.command == "reconstruct":
        scene, failures = reconstruct_route(
            args.run_dir,
            _reconstruct_config(args),
            overwrite=args.overwrite,
        )
        print(f"Wrote positioned mesh scene: {scene}")
        return 1 if failures else 0
    if args.command == "run":
        extract_route(
            args.source,
            args.output_dir,
            metadata=args.meta,
            config=_extract_config(args),
            overwrite=args.overwrite,
        )
        segment_route(
            args.output_dir,
            _segment_config(args),
            overwrite=args.overwrite,
        )
        scene, failures = reconstruct_route(
            args.output_dir,
            _reconstruct_config(args),
            overwrite=args.overwrite,
        )
        print(f"Wrote positioned mesh scene: {scene}")
        return 1 if failures else 0
    raise AssertionError(f"unsupported command {args.command}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run_cli(build_parser().parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
