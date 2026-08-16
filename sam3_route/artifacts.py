from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Union


ROUTE_MANIFEST_SCHEMA = "ouster-route-manifest/v1"
TRACKS_SCHEMA = "ouster-route-tracks/v1"


def software_versions(distributions: Mapping[str, str]) -> dict[str, str]:
    """Return stable runtime version metadata without importing heavy packages."""

    versions = {"python": platform.python_version()}
    for label, distribution in distributions.items():
        try:
            versions[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[label] = "not-installed"
    return versions


def atomic_write_json(path: Union[str, Path], document: Mapping[str, Any]) -> None:
    """Write deterministic JSON without exposing a partially-written artifact."""

    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, output)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_json(path: Union[str, Path]) -> dict[str, Any]:
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON document must be an object: {path}")
    return value


def read_route_manifest(path_or_dir: Union[str, Path]) -> dict[str, Any]:
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_dir():
        path = path / "route-manifest.json"
    document = read_json(path)
    if document.get("schema") != ROUTE_MANIFEST_SCHEMA:
        raise ValueError(f"unsupported route manifest schema {document.get('schema')!r}")
    if not isinstance(document.get("stages"), dict):
        raise ValueError("route manifest has no stages record")
    if not isinstance(document.get("keyframes"), list):
        raise ValueError("route manifest has no keyframes list")
    return document


def read_tracks(path_or_dir: Union[str, Path]) -> dict[str, Any]:
    path = Path(path_or_dir).expanduser().resolve()
    if path.is_dir():
        path = path / "tracks.json"
    document = read_json(path)
    if document.get("schema") != TRACKS_SCHEMA:
        raise ValueError(f"unsupported tracks schema {document.get('schema')!r}")
    if not isinstance(document.get("tracks"), list):
        raise ValueError("tracks document has no tracks list")
    return document


def artifact_path(run_dir: Union[str, Path], relative_path: str) -> Path:
    """Resolve a manifest path while preventing traversal outside the run."""

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact path must be a nonempty string")
    root = Path(run_dir).expanduser().resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes run directory: {relative_path!r}") from exc
    return candidate


def relative_artifact(run_dir: Union[str, Path], path: Union[str, Path]) -> str:
    root = Path(run_dir).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ValueError(f"artifact is outside run directory: {resolved}") from exc


def config_digest(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        config, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def source_fingerprint(path: Union[str, Path]) -> dict[str, Any]:
    """Fingerprint a recording without hashing an entire multi-gigabyte route."""

    source = Path(path).expanduser().resolve()
    stat = source.stat()
    digest = hashlib.sha256()
    chunk_size = 1024 * 1024
    with source.open("rb") as stream:
        digest.update(stream.read(chunk_size))
        if stat.st_size > chunk_size:
            stream.seek(max(0, stat.st_size - chunk_size))
            digest.update(stream.read(chunk_size))
    digest.update(str(stat.st_size).encode("ascii"))
    return {
        "path": str(source),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sample_sha256": digest.hexdigest(),
    }


def stage_is_current(
    manifest: Mapping[str, Any], stage: str, config: Mapping[str, Any]
) -> bool:
    record = manifest.get("stages", {}).get(stage, {})
    return record.get("status") == "complete" and record.get(
        "config_sha256"
    ) == config_digest(config)


def update_stage(
    manifest: MutableMapping[str, Any],
    stage: str,
    config: Mapping[str, Any],
    *,
    status: str,
    error: Optional[str] = None,
) -> None:
    if status not in {"pending", "running", "complete", "failed"}:
        raise ValueError(f"unsupported stage status {status!r}")
    stages = manifest.setdefault("stages", {})
    record = {
        "status": status,
        "config": dict(config),
        "config_sha256": config_digest(config),
    }
    if error:
        record["error"] = str(error).replace("\n", " ")[:1000]
    stages[stage] = record
