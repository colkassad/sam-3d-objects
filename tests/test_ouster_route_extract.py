import csv
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sam3_route.extract import ExtractConfig, extract_route
from sam3_route.ouster_adapter import OusterAdapter


class FakeScan:
    def __init__(self, x, timestamp):
        self.pose = np.repeat(np.eye(4)[None], 4, axis=0)
        self.pose[:, 0, 3] = x
        self.timestamp = np.arange(timestamp, timestamp + 4, dtype=np.int64)
        self.range = np.full((2, 4), 1000, dtype=np.uint32)
        self.rgb = np.full((2, 4, 3), 0.5, dtype=np.float32)

    def get_first_valid_column(self):
        return 0

    def get_last_valid_column(self):
        return 3


class FakeAdapter:
    sdk_version = "0.16.2-test"

    def __init__(self):
        self.scans = [FakeScan(x, 1000 + index * 10) for index, x in enumerate((0, 2, 6, 7))]
        self.updated = []
        self.closed = False
        self.opened_indexed = False

    def open(self, source, metadata=None, *, indexed=False):
        self.opened_indexed = indexed
        return FakeSource(self.scans)

    def frame_count(self, source):
        return len(source)

    def frame_slice(self, source, start, stop):
        return source[start:stop]

    def sensor_infos(self, source):
        return tuple(source.sensor_info)

    def make_slam(self, *args, **kwargs):
        return object()

    def update_slam(self, slam, frame_set):
        self.updated.append(frame_set)
        return [frame_set]

    def first_scan(self, frame_set):
        return frame_set[0] if frame_set else None

    def range_staggered(self, scan):
        return scan.range

    def rgb_staggered(self, scan):
        return scan.rgb

    def destagger(self, sensor_info, value):
        return np.asarray(value)

    def prepare_rgb(self, sensor_info, value):
        return np.rint(value * 255).astype(np.uint8)

    def ray_calibration(self, sensor_info, shape):
        direction = np.zeros((*shape, 3), dtype=np.float32)
        direction[..., 0] = 1.0
        return direction, np.zeros_like(direction)

    def xyz_sensor_staggered(self, sensor_info, range_staggered):
        points = np.zeros((*range_staggered.shape, 3), dtype=np.float64)
        points[..., 0] = range_staggered / 1000.0
        return points

    def column_timestamps(self, scan, width):
        return scan.timestamp.copy()

    def metadata_document(self, info):
        return {"serial": "fake"}

    def close(self, source):
        self.closed = True


class FakeSource:
    def __init__(self, scans):
        self.sensor_info = [SimpleNamespace(extrinsic=np.eye(4))]
        self.scans = scans

    def __iter__(self):
        return iter(self.scans)

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, item):
        value = self.scans[item]
        return FakeSource(value) if isinstance(item, slice) else value


def test_first_scan_unwraps_sdk_scan_set_even_when_container_has_field_method():
    scan = FakeScan(0.0, 1000)

    class ScanSet:
        def __getitem__(self, index):
            return scan

        def field(self, field):
            raise IndexError(f"Field {field!r} not found in LidarScanSet")

    assert OusterAdapter.first_scan(ScanSet()) is scan


def test_extract_updates_slam_for_every_scan_and_forces_last_keyframe(tmp_path):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()
    source = FakeSource(adapter.scans)
    adapter.open = lambda *_, **__: source

    manifest_path = extract_route(
        source_path,
        tmp_path / "output",
        config=ExtractConfig(keyframe_distance_m=5.0, keyframe_angle_deg=5.0),
        adapter=adapter,
    )

    manifest = json.loads(manifest_path.read_text())
    assert len(adapter.updated) == 4
    assert [item["scan_index"] for item in manifest["keyframes"]] == [1, 3, 4]
    with (tmp_path / "output" / "trajectory.csv").open(newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    assert adapter.closed

    exploding = FakeAdapter()
    exploding.open = lambda *_, **__: (_ for _ in ()).throw(AssertionError("resume reopened source"))
    assert extract_route(source_path, tmp_path / "output", adapter=exploding) == manifest_path
    with pytest.raises(RuntimeError, match="configuration changed"):
        extract_route(
            source_path,
            tmp_path / "output",
            config=ExtractConfig(keyframe_distance_m=6.0),
            adapter=exploding,
        )


def test_trajectory_preserves_an_invalid_row_for_every_slam_input(tmp_path):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()
    source = FakeSource(adapter.scans)
    adapter.open = lambda *_, **__: source
    original_update = adapter.update_slam

    def update_with_gap(slam, frame_set):
        if frame_set is adapter.scans[1]:
            adapter.updated.append(frame_set)
            return None
        return original_update(slam, frame_set)

    adapter.update_slam = update_with_gap
    manifest_path = extract_route(
        source_path,
        tmp_path / "output",
        adapter=adapter,
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["trajectory"]["rows"] == 4
    assert manifest["trajectory"]["valid_rows"] == 3
    with np.load(tmp_path / "output" / "trajectory.npz") as trajectory:
        assert trajectory["valid"].tolist() == [True, False, True, True]
        assert np.isnan(trajectory["body_to_world"][1]).all()


def test_extract_streams_an_optional_binary_rgb_point_cloud(tmp_path):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()
    adapter.open = lambda *_, **__: FakeSource(adapter.scans)

    extract_route(
        source_path,
        tmp_path / "output",
        config=ExtractConfig(point_cloud="route.ply"),
        adapter=adapter,
    )

    payload = (tmp_path / "output" / "route.ply").read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"format binary_little_endian 1.0" in header
    assert b"element vertex 4" in header
    assert len(body) == 4 * 15


@pytest.mark.parametrize(
    ("config", "expected_indices"),
    [
        (ExtractConfig(start_frame=2, stop_frame=3), [2, 3]),
        (ExtractConfig(start_frame=2), [2, 3, 4]),
        (ExtractConfig(stop_frame=2), [1, 2]),
        (ExtractConfig(stop_frame=10), [1, 2, 3, 4]),
        (ExtractConfig(start_frame=2, stop_frame=4, max_scans=2), [2, 3]),
    ],
)
def test_extract_uses_an_indexed_inclusive_osf_window(
    tmp_path, config, expected_indices
):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()

    manifest_path = extract_route(
        source_path,
        tmp_path / "output",
        config=config,
        adapter=adapter,
    )

    assert adapter.opened_indexed is True
    assert adapter.updated == [adapter.scans[index - 1] for index in expected_indices]
    with (tmp_path / "output" / "trajectory.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["scan_index"]) for row in rows] == expected_indices
    manifest = json.loads(manifest_path.read_text())
    assert manifest["keyframes"][0]["scan_index"] == expected_indices[0]
    assert manifest["keyframes"][-1]["scan_index"] == expected_indices[-1]
    assert manifest["source_window"] == {
        "requested_start_frame": config.start_frame,
        "requested_stop_frame": config.stop_frame,
        "effective_start_frame": expected_indices[0],
        "effective_stop_frame": expected_indices[-1],
        "recording_frame_count": 4,
        "processed_frame_count": len(expected_indices),
        "numbering": "1-based inclusive",
        "indexed": True,
        "slam_origin": "reset_at_effective_start_frame",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_frame": 0},
        {"stop_frame": 0},
        {"start_frame": -1},
        {"stop_frame": -1},
        {"start_frame": 3, "stop_frame": 2},
    ],
)
def test_extract_rejects_invalid_frame_windows(kwargs):
    with pytest.raises(ValueError):
        ExtractConfig(**kwargs)


def test_extract_rejects_start_past_osf_end_and_closes_source(tmp_path):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()

    with pytest.raises(ValueError, match="exceeds OSF frame count"):
        extract_route(
            source_path,
            tmp_path / "output",
            config=ExtractConfig(start_frame=5),
            adapter=adapter,
        )

    assert adapter.closed


def test_extract_rejects_frame_window_for_pcap(tmp_path):
    source_path = tmp_path / "route.pcap"
    source_path.write_bytes(b"fake pcap")

    with pytest.raises(ValueError, match="only for OSF"):
        extract_route(
            source_path,
            tmp_path / "output",
            config=ExtractConfig(start_frame=2),
            adapter=FakeAdapter(),
        )


def test_extract_window_limits_optional_point_cloud(tmp_path):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()

    extract_route(
        source_path,
        tmp_path / "output",
        config=ExtractConfig(
            start_frame=2,
            stop_frame=3,
            point_cloud="route.ply",
        ),
        adapter=adapter,
    )

    payload = (tmp_path / "output" / "route.ply").read_bytes()
    header, body = payload.split(b"end_header\n", 1)
    assert b"element vertex 2" in header
    assert len(body) == 2 * 15


def test_extract_window_resume_uses_config_digest(tmp_path):
    source_path = tmp_path / "route.osf"
    source_path.write_bytes(b"fake osf")
    adapter = FakeAdapter()
    run_dir = tmp_path / "output"
    config = ExtractConfig(start_frame=2, stop_frame=3)

    manifest_path = extract_route(source_path, run_dir, config=config, adapter=adapter)
    exploding = FakeAdapter()
    exploding.open = lambda *_, **__: (_ for _ in ()).throw(
        AssertionError("matching window reopened source")
    )

    assert extract_route(source_path, run_dir, config=config, adapter=exploding) == manifest_path
    with pytest.raises(RuntimeError, match="configuration changed"):
        extract_route(
            source_path,
            run_dir,
            config=ExtractConfig(start_frame=1, stop_frame=3),
            adapter=exploding,
        )


def test_ouster_adapter_forwards_indexed_open_option(tmp_path):
    calls = []
    adapter = object.__new__(OusterAdapter)
    adapter._open_source = lambda *args, **kwargs: calls.append((args, kwargs)) or object()
    source_path = tmp_path / "route.osf"

    adapter.open(source_path, indexed=True)

    assert calls == [((str(source_path),), {"index": True})]


@pytest.mark.skipif(
    not os.environ.get("SAM3_ROUTE_OSF_FIXTURE"),
    reason="set SAM3_ROUTE_OSF_FIXTURE for the indexed real-OSF smoke test",
)
def test_real_osf_extracts_absolute_indexed_window(tmp_path):
    source_path = Path(os.environ["SAM3_ROUTE_OSF_FIXTURE"]).resolve()

    manifest_path = extract_route(
        source_path,
        tmp_path / "output",
        config=ExtractConfig(start_frame=10, stop_frame=12),
    )

    manifest = json.loads(manifest_path.read_text())
    assert manifest["trajectory"]["rows"] == 3
    assert manifest["source_window"]["effective_start_frame"] == 10
    assert manifest["source_window"]["effective_stop_frame"] == 12
    with np.load(tmp_path / "output" / "trajectory.npz") as trajectory:
        assert trajectory["scan_index"].tolist() == [10, 11, 12]
