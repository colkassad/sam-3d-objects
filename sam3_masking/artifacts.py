from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np
from PIL import Image

from .types import MaskFrame, MaskPrediction

MANIFEST_SCHEMA = "sam3-mask-manifest/v1"


def prompt_slug(prompt: str, *, max_length: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")
    return (slug or "prompt")[:max_length].rstrip("-")


def _atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_mask_manifest(
    frame: MaskFrame,
    output_dir: Union[str, Path],
    *,
    image_path: Optional[Union[str, Path]] = None,
) -> Path:
    """Write lossless masks and their versioned JSON manifest."""

    output_dir = Path(output_dir).expanduser().resolve()
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for prediction in frame.predictions:
        filename = f"{prediction.id}-{prompt_slug(prediction.prompt)}.png"
        mask_path = masks_dir / filename
        Image.fromarray(prediction.mask.astype(np.uint8) * 255).save(
            mask_path, format="PNG"
        )
        records.append(
            {
                "id": prediction.id,
                "prompt": prediction.prompt,
                "query_prompt": prediction.query_prompt,
                "score": prediction.score,
                "box_xyxy": list(prediction.box_xyxy),
                "mask": mask_path.relative_to(output_dir).as_posix(),
            }
        )
    document = {
        "schema": MANIFEST_SCHEMA,
        "image": {
            "path": (
                str(Path(image_path).expanduser().resolve())
                if image_path is not None
                else None
            ),
            "width": frame.width,
            "height": frame.height,
            "source_id": frame.source_id,
        },
        "predictions": records,
    }
    manifest_path = output_dir / "manifest.json"
    _atomic_write_json(manifest_path, document)
    return manifest_path


def read_manifest_document(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as stream:
        document = json.load(stream)
    if document.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported mask manifest schema {document.get('schema')!r}")
    if not isinstance(document.get("image"), dict):
        raise ValueError("mask manifest has no image record")
    if not isinstance(document.get("predictions"), list):
        raise ValueError("mask manifest predictions must be a list")
    return document


def _resolve_artifact_path(manifest_path: Path, relative_path: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("artifact path must be a nonempty string")
    base = manifest_path.parent.resolve()
    candidate = (base / relative_path).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError("artifact path escapes the manifest directory") from exc
    return candidate


def load_mask_manifest(path: Union[str, Path]) -> MaskFrame:
    """Read and strictly validate a mask manifest and its PNG masks."""

    manifest_path = Path(path).expanduser().resolve()
    document = read_manifest_document(manifest_path)
    image = document["image"]
    width, height = int(image["width"]), int(image["height"])
    predictions = []
    for record in document["predictions"]:
        if not isinstance(record, dict):
            raise ValueError("prediction records must be objects")
        mask_path = _resolve_artifact_path(manifest_path, record.get("mask"))
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L")) > 0
        predictions.append(
            MaskPrediction(
                id=str(record["id"]),
                prompt=str(record["prompt"]),
                query_prompt=str(record.get("query_prompt", record["prompt"])),
                score=float(record["score"]),
                box_xyxy=tuple(float(value) for value in record["box_xyxy"]),
                mask=np.asarray(mask, dtype=np.bool_),
            )
        )
    return MaskFrame(
        width=width,
        height=height,
        predictions=tuple(predictions),
        source_id=image.get("source_id"),
    )


def update_mesh_records(
    path: Union[str, Path], records: Mapping[str, Mapping[str, Any]]
) -> None:
    """Atomically attach per-prediction mesh status records to a manifest."""

    manifest_path = Path(path).expanduser().resolve()
    document = read_manifest_document(manifest_path)
    known_ids = {record["id"] for record in document["predictions"]}
    unknown_ids = set(records).difference(known_ids)
    if unknown_ids:
        raise ValueError(f"unknown prediction ids: {sorted(unknown_ids)}")
    for prediction in document["predictions"]:
        if prediction["id"] in records:
            prediction["mesh"] = dict(records[prediction["id"]])
    _atomic_write_json(manifest_path, document)
