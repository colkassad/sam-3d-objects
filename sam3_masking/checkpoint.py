from __future__ import annotations

import json
import os
import shutil
import struct
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

SAM3_ASSET_FILES = (
    "config.json",
    "processor_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)


def find_repo_root(start: Optional[Union[str, Path]] = None) -> Path:
    candidates = [Path(start or Path.cwd()).expanduser().resolve()]
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        for directory in (candidate, *candidate.parents):
            if (directory / "pyproject.toml").is_file() and (
                directory / "sam3d_objects"
            ).is_dir():
                return directory
    raise FileNotFoundError(
        "could not find the SAM 3D Objects repository root; pass --repo-root"
    )


def read_safetensors_header(path: Union[str, Path]) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        length_bytes = stream.read(8)
        if len(length_bytes) != 8:
            raise ValueError("checkpoint is too short to be a safetensors file")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length <= 0 or header_length > min(file_size - 8, 64 * 1024**2):
            raise ValueError("checkpoint has an invalid safetensors header length")
        try:
            header = json.loads(stream.read(header_length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("checkpoint has an invalid safetensors header") from exc
    if not isinstance(header, dict):
        raise ValueError("checkpoint safetensors header must be an object")
    return header


def validate_sam3_checkpoint(path: Union[str, Path]) -> dict[str, Any]:
    """Validate the lightweight structure of a Transformers-format SAM 3 file."""

    checkpoint = Path(path).expanduser().resolve()
    header = read_safetensors_header(checkpoint)
    keys = [key for key in header if key != "__metadata__"]
    required_prefixes = ("detector_model.", "tracker_model.")
    missing = [
        prefix
        for prefix in required_prefixes
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        raise ValueError(
            "checkpoint is not Transformers-format SAM 3; missing key prefixes: "
            + ", ".join(missing)
        )
    return {
        "path": str(checkpoint),
        "size_bytes": checkpoint.stat().st_size,
        "tensor_count": len(keys),
        "metadata": header.get("__metadata__", {}),
    }


def resolve_hf_token(
    repo_root: Union[str, Path],
    *,
    environ: Optional[Mapping[str, str]] = None,
    dotenv_values_fn: Optional[Callable[[Path], Mapping[str, Any]]] = None,
    configured_token_getter: Optional[Callable[[], Optional[str]]] = None,
) -> Optional[str]:
    """Resolve HF_TOKEN without logging or mutating the process environment."""

    environ = os.environ if environ is None else environ
    environment_token = environ.get("HF_TOKEN")
    if environment_token and environment_token.strip():
        return environment_token.strip()

    env_path = Path(repo_root).expanduser().resolve() / ".env"
    if env_path.is_file():
        if dotenv_values_fn is None:
            try:
                from dotenv import dotenv_values
            except ImportError as exc:
                raise RuntimeError(
                    "python-dotenv is required to read HF_TOKEN from .env"
                ) from exc
            dotenv_values_fn = dotenv_values
        dotenv_token = dotenv_values_fn(env_path).get("HF_TOKEN")
        if dotenv_token and str(dotenv_token).strip():
            return str(dotenv_token).strip()

    if configured_token_getter is None:
        try:
            from huggingface_hub import get_token
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required to prepare the SAM 3 model"
            ) from exc
        configured_token_getter = get_token
    configured_token = configured_token_getter()
    return configured_token.strip() if configured_token else None


def prepare_model_bundle(
    weights_path: Union[str, Path],
    model_dir: Union[str, Path],
    *,
    repo_root: Union[str, Path],
    model_id: str = "facebook/sam3",
    revision: str = "main",
    copy_weights: bool = False,
    force: bool = False,
    download_fn: Optional[Callable[..., str]] = None,
    token: Optional[str] = None,
) -> dict[str, Any]:
    """Prepare an offline local SAM 3 directory without downloading weights."""

    checkpoint_summary = validate_sam3_checkpoint(weights_path)
    weights_path = Path(weights_path).expanduser().resolve()
    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    token = token or resolve_hf_token(repo_root)
    if not token:
        raise RuntimeError(
            "no Hugging Face token is available in HF_TOKEN, the root .env, "
            "or the Hugging Face CLI configuration"
        )
    if download_fn is None:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface-hub is required to prepare the SAM 3 model"
            ) from exc
        download_fn = hf_hub_download

    for filename in SAM3_ASSET_FILES:
        try:
            download_fn(
                repo_id=model_id,
                filename=filename,
                revision=revision,
                local_dir=str(model_dir),
                token=token,
            )
        except Exception:
            raise RuntimeError(
                f"failed to download required SAM 3 asset {filename!r}; "
                "verify model access and the configured Hugging Face token"
            ) from None

    model_path = model_dir / "model.safetensors"
    if model_path.exists() or model_path.is_symlink():
        matches_requested = (
            not copy_weights
            and model_path.is_symlink()
            and model_path.resolve() == weights_path
        )
        if not matches_requested:
            if not force:
                raise FileExistsError(
                    f"model file already exists: {model_path}; "
                    "pass --force to replace it"
                )
            model_path.unlink()
    if not model_path.exists():
        if copy_weights:
            shutil.copy2(weights_path, model_path)
        else:
            model_path.symlink_to(weights_path)

    bundle = {
        "schema": "sam3-local-bundle/v1",
        "model_id": model_id,
        "revision": revision,
        "checkpoint": checkpoint_summary,
        "weights_mode": "copy" if copy_weights else "symlink",
        "assets": list(SAM3_ASSET_FILES),
    }
    with (model_dir / "sam3_bundle.json").open("w", encoding="utf-8") as stream:
        json.dump(bundle, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return bundle
