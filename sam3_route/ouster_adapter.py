from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


class OusterAdapter:
    """The only module that directly depends on ouster-sdk.

    The route artifacts intentionally use stable names even though the SDK renamed
    ScanSource metadata and pose members after 0.16.
    """

    def __init__(self) -> None:
        try:
            from ouster.sdk import open_source
            from ouster.sdk.core import ChanField, XYZLut, destagger
            from ouster.sdk.core._utils import AutoExposure
            from ouster.sdk.mapping import SlamConfig, SlamEngine
        except ImportError as exc:
            raise RuntimeError(
                "Ouster route support is not installed; install "
                "requirements.ouster.txt in the SAM3D environment"
            ) from exc
        self._open_source = open_source
        self.ChanField = ChanField
        self.XYZLut = XYZLut
        self._destagger = destagger
        self.AutoExposure = AutoExposure
        self.SlamConfig = SlamConfig
        self.SlamEngine = SlamEngine
        self._rgb_corrector: Optional[Any] = None
        self._xyzlut: Optional[Any] = None

    @property
    def sdk_version(self) -> str:
        try:
            return version("ouster-sdk")
        except PackageNotFoundError:
            return "unknown"

    def open(
        self,
        source: Path,
        metadata: Optional[Path] = None,
        *,
        indexed: bool = False,
    ) -> Any:
        options = {"index": True} if indexed else {}
        if metadata is None:
            return self._open_source(str(source), **options)
        try:
            # Ouster SDK 0.16's PCAP source declares meta as List[str].
            return self._open_source(str(source), meta=[str(metadata)], **options)
        except TypeError:
            return self._open_source(str(source), meta=str(metadata), **options)

    @staticmethod
    def frame_count(source: Any) -> int:
        try:
            return len(source)
        except TypeError as exc:
            raise RuntimeError("indexed Ouster source did not expose its frame count") from exc

    @staticmethod
    def frame_slice(source: Any, start: int, stop: int) -> Any:
        try:
            return source[start:stop]
        except (IndexError, TypeError) as exc:
            raise RuntimeError("indexed Ouster source did not support frame slicing") from exc

    @staticmethod
    def sensor_infos(source: Any) -> tuple[Any, ...]:
        infos = getattr(source, "sensor_info", None)
        if infos is None:
            infos = getattr(source, "metadata", None)
        if infos is None:
            raise RuntimeError("Ouster source did not expose sensor metadata")
        if isinstance(infos, (list, tuple)):
            return tuple(infos)
        return (infos,)

    def make_slam(
        self,
        sensor_infos: Sequence[Any],
        *,
        min_range_m: float,
        max_range_m: float,
        voxel_size_m: float,
    ) -> Any:
        config = self.SlamConfig()
        for name, value in (
            ("min_range", min_range_m),
            ("max_range", max_range_m),
            ("voxel_size", voxel_size_m),
            ("backend", "kiss"),
        ):
            if hasattr(config, name):
                setattr(config, name, value)
        create = getattr(self.SlamEngine, "create", None)
        if callable(create):
            try:
                return create(list(sensor_infos), config)
            except TypeError:
                return create(sensor_infos[0], config)
        return self.SlamEngine(list(sensor_infos), config)

    @staticmethod
    def update_slam(slam: Any, frame_set: Any) -> Any:
        return slam.update(frame_set)

    @staticmethod
    def first_scan(frame_set: Any) -> Optional[Any]:
        if frame_set is None:
            return None
        # SDK 0.16's LidarScanSet is list-like but also exposes field(), so
        # checking only for a field method incorrectly treats the container as
        # a LidarScan. Prefer a contained object when it is itself scan-like.
        try:
            candidate = frame_set[0]
        except (IndexError, KeyError, TypeError):
            candidate = None
        if candidate is not None and candidate is not frame_set:
            if any(
                hasattr(candidate, name)
                for name in ("field", "pose", "body_to_world")
            ):
                return candidate
        return frame_set if hasattr(frame_set, "field") else None

    @staticmethod
    def _field(scan: Any, field: Any) -> np.ndarray:
        try:
            return np.asarray(scan.field(field))
        except Exception as exc:
            names = ", ".join(str(value) for value in getattr(scan, "fields", ()))
            raise RuntimeError(
                f"recording does not contain required Ouster field {field}; "
                f"available fields: {names or '<unknown>'}"
            ) from exc

    def range_staggered(self, scan: Any) -> np.ndarray:
        return self._field(scan, self.ChanField.RANGE)

    def rgb_staggered(self, scan: Any) -> np.ndarray:
        return self._field(scan, self.ChanField.RGB)

    def destagger(self, sensor_info: Any, value: np.ndarray) -> np.ndarray:
        return np.asarray(self._destagger(sensor_info, value))

    def prepare_rgb(self, sensor_info: Any, rgb_staggered: np.ndarray) -> np.ndarray:
        image = self.destagger(sensor_info, rgb_staggered)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(
                f"registered RGB field must have shape (H, W, 3), got {image.shape}"
            )
        if image.dtype == np.float16:
            if self._rgb_corrector is None:
                self._rgb_corrector = self.AutoExposure(0.05, 0.02, 3)
            update = getattr(self._rgb_corrector, "update", self._rgb_corrector)
            corrected = update(image)
            if corrected is None:
                raise RuntimeError("Ouster RGB autoexposure returned no image")
            image = np.asarray(corrected, dtype=np.float32)
        elif np.issubdtype(image.dtype, np.integer):
            maximum = float(np.iinfo(image.dtype).max)
            image = image.astype(np.float32) / maximum
        else:
            image = image.astype(np.float32, copy=False)
        return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)

    def xyz_sensor_staggered(self, sensor_info: Any, range_staggered: np.ndarray) -> np.ndarray:
        if self._xyzlut is None:
            try:
                self._xyzlut = self.XYZLut(sensor_info, use_extrinsics=False)
            except TypeError:
                self._xyzlut = self.XYZLut(sensor_info)
        return np.asarray(self._xyzlut(range_staggered), dtype=np.float64)

    def ray_calibration(
        self, sensor_info: Any, range_shape: tuple[int, int]
    ) -> tuple[np.ndarray, np.ndarray]:
        one_meter = np.full(range_shape, 1000, dtype=np.uint32)
        two_meters = np.full(range_shape, 2000, dtype=np.uint32)
        first = self.destagger(
            sensor_info, self.xyz_sensor_staggered(sensor_info, one_meter)
        )
        second = self.destagger(
            sensor_info, self.xyz_sensor_staggered(sensor_info, two_meters)
        )
        direction = second - first
        origin = first - direction
        return direction.astype(np.float32), origin.astype(np.float32)

    @staticmethod
    def column_timestamps(scan: Any, width: int) -> np.ndarray:
        for name in ("timestamp", "packet_timestamp"):
            value = getattr(scan, name, None)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.int64).reshape(-1)
            if array.size == width:
                return array.copy()
        return np.zeros(width, dtype=np.int64)

    @staticmethod
    def metadata_document(sensor_info: Any) -> dict[str, Any]:
        document: dict[str, Any] = {}
        for name in (
            "sn",
            "prod_line",
            "prod_sn",
            "fw_rev",
            "lidar_mode",
            "udp_profile_lidar",
        ):
            value = getattr(sensor_info, name, None)
            if value is not None:
                document[name] = str(value)
        raw: Optional[str] = None
        for name in ("to_json_string", "to_json"):
            method = getattr(sensor_info, name, None)
            if callable(method):
                try:
                    value = method()
                    raw = value if isinstance(value, str) else json.dumps(value)
                    break
                except Exception:
                    pass
        if raw:
            try:
                document["raw"] = json.loads(raw)
            except json.JSONDecodeError:
                document["raw_json"] = raw
        return document

    @staticmethod
    def close(source: Any) -> None:
        close = getattr(source, "close", None)
        if callable(close):
            close()
