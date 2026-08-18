from __future__ import annotations

import csv
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image

from .artifacts import (
    ROUTE_MANIFEST_SCHEMA,
    atomic_write_json,
    config_digest,
    read_route_manifest,
    relative_artifact,
    software_versions,
    source_fingerprint,
    stage_is_current,
    update_stage,
)
from .geometry import (
    column_poses,
    pose_distance,
    representative_column,
    rotation_matrix_to_quaternion_wxyz,
    rotation_matrix_to_xyz_euler,
    scan_timestamp_ns,
    sensor_to_body,
    transform_pointmap_per_column,
)
from .ouster_adapter import OusterAdapter


@dataclass(frozen=True)
class ExtractConfig:
    keyframe_distance_m: float = 5.0
    keyframe_angle_deg: float = 5.0
    sam3_recovery_distance_m: float = 1.0
    slam_min_range_m: float = 1.0
    slam_max_range_m: float = 75.0
    slam_voxel_size_m: float = 1.0
    point_cloud: Optional[str] = None
    point_cloud_voxel_m: float = 0.10
    max_scans: Optional[int] = None
    start_frame: Optional[int] = None
    stop_frame: Optional[int] = None

    def __post_init__(self) -> None:
        if self.keyframe_distance_m <= 0:
            raise ValueError("keyframe_distance_m must be greater than zero")
        if self.keyframe_angle_deg <= 0:
            raise ValueError("keyframe_angle_deg must be greater than zero")
        if self.sam3_recovery_distance_m <= 0:
            raise ValueError("sam3_recovery_distance_m must be greater than zero")
        if self.slam_min_range_m < 0 or self.slam_max_range_m <= self.slam_min_range_m:
            raise ValueError("SLAM range limits are invalid")
        if self.slam_voxel_size_m <= 0 or self.point_cloud_voxel_m <= 0:
            raise ValueError("voxel sizes must be greater than zero")
        if self.max_scans is not None and self.max_scans <= 0:
            raise ValueError("max_scans must be greater than zero")
        if self.start_frame is not None and self.start_frame <= 0:
            raise ValueError("start_frame must be greater than zero")
        if self.stop_frame is not None and self.stop_frame <= 0:
            raise ValueError("stop_frame must be greater than zero")
        if (
            self.start_frame is not None
            and self.stop_frame is not None
            and self.stop_frame < self.start_frame
        ):
            raise ValueError("stop_frame must be greater than or equal to start_frame")

    def manifest_value(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_frame_window(config: ExtractConfig, recording_frames: int) -> tuple[int, int]:
    if recording_frames <= 0:
        raise RuntimeError("indexed OSF source contains no frames")
    start = config.start_frame or 1
    if start > recording_frames:
        raise ValueError(
            f"start_frame {start} exceeds OSF frame count {recording_frames}"
        )
    stop = min(config.stop_frame or recording_frames, recording_frames)
    if config.max_scans is not None:
        stop = min(stop, start + config.max_scans - 1)
    return start, stop


@dataclass
class ScanSnapshot:
    scan_index: int
    timestamp_ns: Optional[int]
    reference_column: int
    body_to_world: np.ndarray
    column_timestamp_ns: np.ndarray
    range_mm: np.ndarray
    rgb: np.ndarray


@dataclass(frozen=True)
class TrajectorySample:
    scan_index: int
    timestamp_ns: Optional[int]
    reference_column: int
    body_to_world: np.ndarray
    valid_points: int
    valid: bool = True
    validity: str = "valid"


class KeyframeSelector:
    def __init__(self, distance_m: float = 5.0, angle_deg: float = 5.0) -> None:
        self.distance_m = float(distance_m)
        self.angle_deg = float(angle_deg)
        self._last_pose: Optional[np.ndarray] = None

    def select(self, pose: np.ndarray) -> bool:
        if self._last_pose is None:
            self._last_pose = np.asarray(pose, dtype=np.float64).copy()
            return True
        distance, angle = pose_distance(self._last_pose, pose)
        if distance >= self.distance_m or angle >= self.angle_deg:
            self._last_pose = np.asarray(pose, dtype=np.float64).copy()
            return True
        return False


class VoxelPointCloud:
    def __init__(self, voxel_size_m: float) -> None:
        self.voxel_size_m = float(voxel_size_m)
        self._values: dict[tuple[int, int, int], list[Any]] = {}

    def add(self, points: np.ndarray, colors: np.ndarray) -> None:
        xyz = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        rgb = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        valid = np.all(np.isfinite(xyz), axis=1)
        xyz, rgb = xyz[valid], rgb[valid]
        if not len(xyz):
            return
        keys = np.floor(xyz / self.voxel_size_m).astype(np.int64)
        unique, inverse = np.unique(keys, axis=0, return_inverse=True)
        point_sums = np.zeros((len(unique), 3), dtype=np.float64)
        color_sums = np.zeros((len(unique), 3), dtype=np.float64)
        counts = np.bincount(inverse)
        np.add.at(point_sums, inverse, xyz)
        np.add.at(color_sums, inverse, rgb)
        for key, point_sum, color_sum, count in zip(
            unique, point_sums, color_sums, counts
        ):
            item = self._values.setdefault(tuple(int(v) for v in key), [np.zeros(3), np.zeros(3), 0])
            item[0] += point_sum
            item[1] += color_sum
            item[2] += int(count)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        points, colors = [], []
        for key in sorted(self._values):
            point_sum, color_sum, count = self._values[key]
            points.append(point_sum / count)
            colors.append(np.rint(color_sum / count))
        return (
            np.asarray(points, dtype=np.float32).reshape(-1, 3),
            np.asarray(colors, dtype=np.uint8).reshape(-1, 3),
        )


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if len(points) != len(colors):
        raise ValueError("point and RGB counts differ")
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    ).encode("ascii")
    records = np.empty(
        len(points),
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    records["x"], records["y"], records["z"] = points.T
    records["red"], records["green"], records["blue"] = colors.T
    with path.open("wb") as stream:
        stream.write(header)
        records.tofile(stream)


def _trajectory_row(snapshot: TrajectorySample) -> dict[str, Any]:
    pose = snapshot.body_to_world
    if snapshot.valid:
        qw, qx, qy, qz = rotation_matrix_to_quaternion_wxyz(pose[:3, :3])
        roll, pitch, yaw = rotation_matrix_to_xyz_euler(pose[:3, :3])
        position: list[Any] = [float(pose[0, 3]), float(pose[1, 3]), float(pose[2, 3])]
        orientation: list[Any] = [qw, qx, qy, qz, roll, pitch, yaw]
    else:
        position = ["", "", ""]
        orientation = ["", "", "", "", "", "", ""]
    return {
        "scan_index": snapshot.scan_index,
        "timestamp_ns": "" if snapshot.timestamp_ns is None else snapshot.timestamp_ns,
        "reference_column": snapshot.reference_column,
        "x": position[0],
        "y": position[1],
        "z": position[2],
        "qw": orientation[0],
        "qx": orientation[1],
        "qy": orientation[2],
        "qz": orientation[3],
        "roll_rad": orientation[4],
        "pitch_rad": orientation[5],
        "yaw_rad": orientation[6],
        "valid_points": snapshot.valid_points,
        "valid": snapshot.valid,
        "validity": snapshot.validity,
    }


def _write_trajectory(run_dir: Path, snapshots: Iterable[TrajectorySample]) -> tuple[Path, Path]:
    values = list(snapshots)
    csv_path = run_dir / "trajectory.csv"
    fields = list(_trajectory_row(values[0]).keys()) if values else [
        "scan_index", "timestamp_ns", "reference_column", "x", "y", "z",
        "qw", "qx", "qy", "qz", "roll_rad", "pitch_rad", "yaw_rad", "valid_points",
        "valid", "validity",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(_trajectory_row(value) for value in values)
    npz_path = run_dir / "trajectory.npz"
    np.savez_compressed(
        npz_path,
        scan_index=np.asarray([v.scan_index for v in values], dtype=np.int64),
        timestamp_ns=np.asarray([v.timestamp_ns or 0 for v in values], dtype=np.int64),
        reference_column=np.asarray([v.reference_column for v in values], dtype=np.int32),
        body_to_world=np.asarray(
            [v.body_to_world for v in values], dtype=np.float64
        ),
        valid_points=np.asarray([v.valid_points for v in values], dtype=np.int64),
        valid=np.asarray([v.valid for v in values], dtype=bool),
        validity=np.asarray([v.validity for v in values], dtype=np.str_),
    )
    return csv_path, npz_path


def _save_keyframe(run_dir: Path, snapshot: ScanSnapshot) -> dict[str, Any]:
    frame_id = f"frame-{snapshot.scan_index:06d}"
    directory = run_dir / "frames" / frame_id
    directory.mkdir(parents=True, exist_ok=True)
    rgb_path = directory / "rgb.png"
    geometry_path = directory / "geometry.npz"
    Image.fromarray(snapshot.rgb).save(rgb_path, format="PNG")
    np.savez_compressed(
        geometry_path,
        range_mm=snapshot.range_mm.astype(np.uint32, copy=False),
        body_to_world=snapshot.body_to_world.astype(np.float64, copy=False),
        column_timestamp_ns=snapshot.column_timestamp_ns.astype(np.int64, copy=False),
        reference_column=np.asarray(snapshot.reference_column, dtype=np.int32),
    )
    return {
        "id": frame_id,
        "scan_index": snapshot.scan_index,
        "timestamp_ns": snapshot.timestamp_ns,
        "reference_column": snapshot.reference_column,
        "rgb": relative_artifact(run_dir, rgb_path),
        "geometry": relative_artifact(run_dir, geometry_path),
        "mask_manifest": None,
        "surface_mask_manifest": None,
    }


def _snapshot_scan(
    scan: Any, scan_index: int, sensor_info: Any, adapter: OusterAdapter
) -> ScanSnapshot:
    range_staggered = adapter.range_staggered(scan)
    reference = representative_column(scan, range_staggered)
    poses = column_poses(scan)
    if poses.shape[0] != range_staggered.shape[1]:
        raise RuntimeError("per-column pose count does not match scan width")
    rgb = adapter.prepare_rgb(sensor_info, adapter.rgb_staggered(scan))
    range_destaggered = adapter.destagger(sensor_info, range_staggered)
    if rgb.shape[:2] != range_destaggered.shape:
        raise RuntimeError(
            f"registered RGB shape {rgb.shape[:2]} does not match range {range_destaggered.shape}"
        )
    return ScanSnapshot(
        scan_index=scan_index,
        timestamp_ns=scan_timestamp_ns(scan, reference),
        reference_column=reference,
        body_to_world=poses.copy(),
        column_timestamp_ns=adapter.column_timestamps(scan, poses.shape[0]),
        range_mm=range_destaggered.astype(np.uint32, copy=True),
        rgb=rgb.copy(),
    )


def _initial_manifest(source: Path, metadata: Optional[Path]) -> dict[str, Any]:
    return {
        "schema": ROUTE_MANIFEST_SCHEMA,
        "source": source_fingerprint(source),
        "metadata": None if metadata is None else source_fingerprint(metadata),
        "coordinate_system": {
            "type": "local_slam",
            "units": "meters",
            "georeferenced": False,
            "vector_convention": "column vectors; T_parent_child maps child into parent",
            "point_transform": "p_world = T_world_body[column] @ T_body_sensor @ p_sensor",
        },
        "stages": {},
        "calibration": None,
        "trajectory": None,
        "keyframes": [],
        "recovery_frames": [],
        "prompts": [],
        "synonyms": {},
        "prompt_categories": [],
        "surface_prompts": [],
        "tracks": None,
        "source_window": None,
        "outputs": {},
        "software": software_versions({"numpy": "numpy", "pillow": "Pillow"}),
    }


def _clean_extract_outputs(run_dir: Path, manifest: dict[str, Any]) -> None:
    for name in ("frames", "observations", "tracks", "meshes", "surface"):
        directory = run_dir / name
        if directory.exists():
            shutil.rmtree(directory)
    for name in (
        "trajectory.csv",
        "trajectory.npz",
        "calibration.npz",
        "calibration.json",
        "tracks.json",
        "scene.glb",
        "scene.json",
    ):
        path = run_dir / name
        if path.exists():
            path.unlink()
    point_cloud = manifest.get("outputs", {}).get("point_cloud")
    if point_cloud:
        path = run_dir / point_cloud
        if path.exists():
            path.unlink()


def extract_route(
    source: Path,
    run_dir: Path,
    *,
    metadata: Optional[Path] = None,
    config: Optional[ExtractConfig] = None,
    overwrite: bool = False,
    adapter: Optional[OusterAdapter] = None,
) -> Path:
    config = config or ExtractConfig()
    source = source.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not source.is_file() or source.suffix.lower() not in {".pcap", ".osf"}:
        raise ValueError("source must be an existing .pcap or .osf file")
    has_frame_window = config.start_frame is not None or config.stop_frame is not None
    if has_frame_window and source.suffix.lower() != ".osf":
        raise ValueError("start_frame and stop_frame are supported only for OSF sources")
    if metadata is not None and not metadata.expanduser().resolve().is_file():
        raise FileNotFoundError(f"PCAP metadata does not exist: {metadata}")
    run_dir.mkdir(parents=True, exist_ok=True)
    point_cloud_output: Optional[Path] = None
    if config.point_cloud:
        point_cloud_output = Path(config.point_cloud).expanduser()
        if not point_cloud_output.is_absolute():
            point_cloud_output = run_dir / point_cloud_output
        point_cloud_output = point_cloud_output.resolve()
        try:
            point_cloud_output.relative_to(run_dir)
        except ValueError as exc:
            raise ValueError("point cloud output must be inside the run directory") from exc
        if point_cloud_output == run_dir or point_cloud_output.suffix.lower() != ".ply":
            raise ValueError("point cloud output must be a .ply file inside the run directory")
    manifest_path = run_dir / "route-manifest.json"
    manifest = (
        read_route_manifest(manifest_path)
        if manifest_path.exists()
        else _initial_manifest(source, metadata)
    )
    fingerprint = source_fingerprint(source)
    metadata_fingerprint = (
        None if metadata is None else source_fingerprint(metadata.expanduser().resolve())
    )
    if (
        manifest.get("source") != fingerprint
        or manifest.get("metadata") != metadata_fingerprint
    ):
        if not overwrite:
            raise RuntimeError(
                "run directory belongs to a different or changed recording/metadata pair"
            )
        manifest = _initial_manifest(source, metadata)
    stage = manifest.get("stages", {}).get("extract")
    if stage_is_current(manifest, "extract", config.manifest_value()) and not overwrite:
        return manifest_path
    if (
        stage
        and stage.get("config_sha256") != config_digest(config.manifest_value())
        and not overwrite
    ):
        raise RuntimeError("extract configuration changed; pass --overwrite to replace artifacts")
    if overwrite or stage:
        _clean_extract_outputs(run_dir, manifest)
        manifest["keyframes"] = []
        manifest["recovery_frames"] = []
        manifest["trajectory"] = None
        manifest["calibration"] = None
        manifest["tracks"] = None
        manifest["prompts"] = []
        manifest["synonyms"] = {}
        manifest["prompt_categories"] = []
        manifest.pop("sam3_recovery", None)
        manifest["source_window"] = None
        manifest.get("stages", {}).pop("segment", None)
        manifest.get("stages", {}).pop("reconstruct", None)
        manifest.get("stages", {}).pop("scene_compose", None)
        manifest.get("stages", {}).pop("surface_segment", None)
        manifest.get("stages", {}).pop("surface_tin", None)
        manifest["surface_prompts"] = []
        manifest["outputs"] = {}

    update_stage(manifest, "extract", config.manifest_value(), status="running")
    atomic_write_json(manifest_path, manifest)
    adapter = adapter or OusterAdapter()
    source_handle = None
    try:
        source_handle = adapter.open(source, metadata, indexed=has_frame_window)
        frame_source = source_handle
        first_source_index = 1
        recording_frame_count: Optional[int] = None
        resolved_stop_frame: Optional[int] = None
        if has_frame_window:
            recording_frame_count = adapter.frame_count(source_handle)
            first_source_index, resolved_stop_frame = _resolve_frame_window(
                config, recording_frame_count
            )
            frame_source = adapter.frame_slice(
                source_handle,
                first_source_index - 1,
                resolved_stop_frame,
            )
        infos = adapter.sensor_infos(source_handle)
        if len(infos) != 1:
            raise RuntimeError(f"version one requires exactly one sensor, found {len(infos)}")
        info = infos[0]
        slam = adapter.make_slam(
            infos,
            min_range_m=config.slam_min_range_m,
            max_range_m=config.slam_max_range_m,
            voxel_size_m=config.slam_voxel_size_m,
        )
        selector = KeyframeSelector(
            config.keyframe_distance_m, config.keyframe_angle_deg
        )
        recovery_selector = KeyframeSelector(
            config.sam3_recovery_distance_m, config.keyframe_angle_deg
        )
        trajectory: list[TrajectorySample] = []
        keyframes: list[dict[str, Any]] = []
        recovery_frames: list[dict[str, Any]] = []
        last_snapshot: Optional[ScanSnapshot] = None
        last_selected_index: Optional[int] = None
        calibration_written = False
        sensor_to_body_matrix = sensor_to_body(info)
        cloud = VoxelPointCloud(config.point_cloud_voxel_m) if config.point_cloud else None
        processed_frame_count = 0
        last_source_index: Optional[int] = None

        for offset, frame_set in enumerate(frame_source):
            if (
                not has_frame_window
                and config.max_scans is not None
                and processed_frame_count >= config.max_scans
            ):
                break
            raw_index = first_source_index + offset
            processed_frame_count += 1
            last_source_index = raw_index
            updated = adapter.update_slam(slam, frame_set)
            scan = adapter.first_scan(updated)
            if scan is None:
                trajectory.append(
                    TrajectorySample(
                        scan_index=raw_index,
                        timestamp_ns=None,
                        reference_column=-1,
                        body_to_world=np.full((4, 4), np.nan, dtype=np.float64),
                        valid_points=0,
                        valid=False,
                        validity="no_slam_output",
                    )
                )
                continue
            range_staggered = adapter.range_staggered(scan)
            valid_points = int(np.count_nonzero(range_staggered))
            if not valid_points:
                trajectory.append(
                    TrajectorySample(
                        scan_index=raw_index,
                        timestamp_ns=scan_timestamp_ns(scan, 0),
                        reference_column=-1,
                        body_to_world=np.full((4, 4), np.nan, dtype=np.float64),
                        valid_points=0,
                        valid=False,
                        validity="no_valid_range",
                    )
                )
                continue
            snapshot = _snapshot_scan(scan, raw_index, info, adapter)
            pose = snapshot.body_to_world[snapshot.reference_column]
            trajectory.append(
                TrajectorySample(
                    scan_index=snapshot.scan_index,
                    timestamp_ns=snapshot.timestamp_ns,
                    reference_column=snapshot.reference_column,
                    body_to_world=pose.copy(),
                    valid_points=valid_points,
                )
            )
            last_snapshot = snapshot

            if not calibration_written:
                ray_direction, ray_origin = adapter.ray_calibration(
                    info, snapshot.range_mm.shape
                )
                calibration_npz = run_dir / "calibration.npz"
                altitude = np.asarray(
                    getattr(info, "beam_altitude_angles", ()), dtype=np.float64
                )
                azimuth = np.asarray(
                    getattr(info, "beam_azimuth_angles", ()), dtype=np.float64
                )
                np.savez_compressed(
                    calibration_npz,
                    sensor_to_body=sensor_to_body_matrix,
                    ray_direction=ray_direction,
                    ray_origin=ray_origin,
                    beam_altitude_angles_deg=altitude,
                    beam_azimuth_angles_deg=azimuth,
                )
                calibration_json = run_dir / "calibration.json"
                atomic_write_json(
                    calibration_json,
                    {
                        "metadata": adapter.metadata_document(info),
                        "arrays": relative_artifact(run_dir, calibration_npz),
                        "layout": "destaggered HxW",
                        "ray_formula": "origin_m + direction_unit * range_mm / 1000",
                    },
                )
                manifest["calibration"] = relative_artifact(run_dir, calibration_json)
                calibration_written = True

            baseline_selected = selector.select(pose)
            recovery_selected = recovery_selector.select(pose)
            if baseline_selected:
                keyframes.append(_save_keyframe(run_dir, snapshot))
                last_selected_index = snapshot.scan_index
            elif recovery_selected:
                recovery_frames.append(_save_keyframe(run_dir, snapshot))

            if cloud is not None:
                range_staggered = adapter.range_staggered(scan)
                xyz_staggered = adapter.xyz_sensor_staggered(info, range_staggered)
                xyz_destaggered = adapter.destagger(info, xyz_staggered)
                world = transform_pointmap_per_column(
                    xyz_destaggered, snapshot.body_to_world, sensor_to_body_matrix
                )
                valid = snapshot.range_mm != 0
                cloud.add(world[valid], snapshot.rgb[valid])

        if not trajectory or last_snapshot is None:
            raise RuntimeError("SLAM produced no valid scans")
        if last_selected_index != last_snapshot.scan_index:
            recovery_frames = [
                frame
                for frame in recovery_frames
                if frame["scan_index"] != last_snapshot.scan_index
            ]
            keyframes.append(_save_keyframe(run_dir, last_snapshot))
        trajectory_csv, trajectory_npz = _write_trajectory(run_dir, trajectory)
        manifest["trajectory"] = {
            "csv": relative_artifact(run_dir, trajectory_csv),
            "matrices": relative_artifact(run_dir, trajectory_npz),
            "rows": len(trajectory),
            "valid_rows": int(sum(value.valid for value in trajectory)),
        }
        manifest["source_window"] = {
            "requested_start_frame": config.start_frame,
            "requested_stop_frame": config.stop_frame,
            "effective_start_frame": first_source_index,
            "effective_stop_frame": last_source_index,
            "recording_frame_count": recording_frame_count,
            "processed_frame_count": processed_frame_count,
            "numbering": "1-based inclusive",
            "indexed": has_frame_window,
            "slam_origin": "reset_at_effective_start_frame",
        }
        manifest["keyframes"] = keyframes
        manifest["recovery_frames"] = recovery_frames
        manifest["software"]["ouster_sdk"] = adapter.sdk_version
        if cloud is not None:
            points, colors = cloud.arrays()
            assert point_cloud_output is not None
            write_binary_ply(point_cloud_output, points, colors)
            manifest["outputs"]["point_cloud"] = relative_artifact(
                run_dir, point_cloud_output
            )
        update_stage(manifest, "extract", config.manifest_value(), status="complete")
        atomic_write_json(manifest_path, manifest)
        return manifest_path
    except Exception as exc:
        update_stage(
            manifest,
            "extract",
            config.manifest_value(),
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )
        atomic_write_json(manifest_path, manifest)
        raise
    finally:
        if source_handle is not None:
            adapter.close(source_handle)
