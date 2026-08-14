import json
import struct
from pathlib import Path

import pytest

from sam3_masking.checkpoint import (
    prepare_model_bundle,
    resolve_hf_token,
    validate_sam3_checkpoint,
)
from sam3_masking.cli_prepare import build_parser


def write_fake_checkpoint(path: Path):
    header = {
        "detector_model.layer.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [0, 4],
        },
        "tracker_model.layer.weight": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [4, 8],
        },
        "__metadata__": {"format": "pt"},
    }
    encoded = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + b"\0" * 8)


def test_real_checkpoint_has_transformers_sam3_layout():
    path = Path("/mnt/d/Data/models/sam3.safetensors")
    if not path.is_file():
        pytest.skip("local SAM 3 checkpoint is not mounted")
    summary = validate_sam3_checkpoint(path)
    assert summary["tensor_count"] == 1797
    assert summary["size_bytes"] > 3_000_000_000


def test_token_resolution_precedence(tmp_path):
    (tmp_path / ".env").write_text("HF_TOKEN=dotenv-secret\n")

    def dotenv_reader(_):
        return {"HF_TOKEN": "dotenv-secret"}

    def configured():
        return "configured-secret"

    assert (
        resolve_hf_token(
            tmp_path,
            environ={"HF_TOKEN": "environment-secret"},
            dotenv_values_fn=dotenv_reader,
            configured_token_getter=configured,
        )
        == "environment-secret"
    )
    assert (
        resolve_hf_token(
            tmp_path,
            environ={},
            dotenv_values_fn=dotenv_reader,
            configured_token_getter=configured,
        )
        == "dotenv-secret"
    )
    assert (
        resolve_hf_token(
            tmp_path,
            environ={},
            dotenv_values_fn=lambda _: {},
            configured_token_getter=configured,
        )
        == "configured-secret"
    )


def test_prepare_bundle_does_not_persist_token(tmp_path):
    weights = tmp_path / "sam3.safetensors"
    model_dir = tmp_path / "model"
    write_fake_checkpoint(weights)
    seen_tokens = []

    def fake_download(**kwargs):
        seen_tokens.append(kwargs["token"])
        destination = Path(kwargs["local_dir"]) / kwargs["filename"]
        destination.write_text("asset")
        return str(destination)

    prepare_model_bundle(
        weights,
        model_dir,
        repo_root=tmp_path,
        download_fn=fake_download,
        token="top-secret-value",
    )

    assert (model_dir / "model.safetensors").resolve() == weights.resolve()
    assert seen_tokens and set(seen_tokens) == {"top-secret-value"}
    metadata = (model_dir / "sam3_bundle.json").read_text()
    assert "top-secret-value" not in metadata
    assert "hf_token" not in metadata.lower()
    assert "--token" not in build_parser().format_help()
