#!/usr/bin/env python3
"""Generate prompted SAM 3 masks and reconstruct each instance as a GLB."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sam3_masking.artifacts import prompt_slug, read_manifest_document  # noqa: E402
from sam3_masking.mesh_bridge import build_parser as build_bridge_parser  # noqa: E402
from sam3_masking.mesh_bridge import run as run_bridge  # noqa: E402
from sam3_masking.prompts import parse_prompt_catalog  # noqa: E402


def _resolve_executable(value: str, *, source: str) -> Path:
    expanded = Path(value).expanduser()
    is_path = (
        expanded.is_absolute()
        or os.sep in value
        or (os.altsep is not None and os.altsep in value)
    )
    candidate = expanded.resolve() if is_path else None
    if candidate is None:
        located = shutil.which(value)
        candidate = Path(located).resolve() if located else None
    if (
        candidate is None
        or not candidate.is_file()
        or not os.access(candidate, os.X_OK)
    ):
        raise FileNotFoundError(
            f"{source} does not identify an executable sam3-mask command: {value!r}"
        )
    return candidate


def discover_sam3_executable(
    explicit: Optional[str],
    *,
    environ: Optional[Mapping[str, str]] = None,
    prefix: Optional[Path] = None,
) -> Path:
    """Find sam3-mask using the documented, deterministic precedence."""

    environ = os.environ if environ is None else environ
    if explicit:
        return _resolve_executable(explicit, source="--sam3-executable")

    configured = environ.get("SAM3_MASK_EXECUTABLE")
    if configured:
        return _resolve_executable(configured, source="SAM3_MASK_EXECUTABLE")

    environment_prefix = Path(prefix or sys.prefix).expanduser().resolve()
    executable_name = "sam3-mask.exe" if os.name == "nt" else "sam3-mask"
    executable_dir = "Scripts" if os.name == "nt" else "bin"
    sibling = (
        environment_prefix.parent / "sam3-masking" / executable_dir / executable_name
    )
    if sibling.is_file() and os.access(sibling, os.X_OK):
        return sibling.resolve()

    located = shutil.which("sam3-mask")
    if located:
        return Path(located).resolve()

    raise FileNotFoundError(
        "could not find sam3-mask; create the sam3-masking environment as "
        "documented, set SAM3_MASK_EXECUTABLE, or pass --sam3-executable"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment an arbitrary image with SAM 3 text prompts, release SAM 3, "
            "and generate one SAM 3D GLB per detected instance."
        ),
        allow_abbrev=False,
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        required=True,
        help="Comma-separated text concept prompts.",
    )
    parser.add_argument("--synonyms", default="")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sam3-executable")
    parser.add_argument(
        "--sam3-model-dir", type=Path, default=Path("checkpoints/sam3-hf")
    )
    parser.add_argument(
        "--sam3d-config", type=Path, default=Path("checkpoints/hf/pipeline.yaml")
    )
    parser.add_argument("--sam3-device", default="auto")
    parser.add_argument(
        "--sam3-dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mesh-target-faces", type=int)
    parser.add_argument("--flat-shading", action="store_true")
    parser.add_argument("--stage1-inference-steps", type=int)
    parser.add_argument("--stage2-inference-steps", type=int)
    parser.add_argument("--profile-memory", action="store_true")
    return parser


def _resolve_repo_path(path: Path) -> Path:
    path = path.expanduser()
    return (REPO_ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    image_path = args.image.expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"source image does not exist: {image_path}")
    catalog = parse_prompt_catalog(args.prompts, args.synonyms)
    args.prompts = ",".join(catalog.prompts)

    model_dir = _resolve_repo_path(args.sam3_model_dir)
    if not model_dir.is_dir() or not (model_dir / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"prepared SAM 3 model bundle is missing at {model_dir}; run sam3-prepare"
        )
    sam3d_config = _resolve_repo_path(args.sam3d_config)
    if not sam3d_config.is_file():
        raise FileNotFoundError(f"SAM 3D configuration does not exist: {sam3d_config}")

    executable = discover_sam3_executable(args.sam3_executable)
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else REPO_ROOT / "outputs" / "sam3-demo" / prompt_slug(image_path.stem)
    )
    return image_path, model_dir, sam3d_config, executable, output_dir.resolve()


def _bridge_arguments(
    args: argparse.Namespace,
    *,
    image_path: Path,
    model_dir: Path,
    sam3d_config: Path,
    executable: Path,
    output_dir: Path,
) -> argparse.Namespace:
    argv = [
        "--image",
        str(image_path),
        "--output-dir",
        str(output_dir),
        "--sam3-executable",
        str(executable),
        "--sam3-model-dir",
        str(model_dir),
        "--sam3d-config",
        str(sam3d_config),
        "--sam3-device",
        args.sam3_device,
        "--sam3-dtype",
        args.sam3_dtype,
        "--score-threshold",
        str(args.score_threshold),
        "--mask-threshold",
        str(args.mask_threshold),
        "--seed",
        str(args.seed),
        "--memory-profile",
        "low_vram",
        "--repo-root",
        str(REPO_ROOT),
    ]
    argv.extend(("--prompts", args.prompts))
    if args.synonyms.strip():
        argv.extend(("--synonyms", args.synonyms))
    optional_values = (
        ("--mesh-target-faces", args.mesh_target_faces),
        ("--stage1-inference-steps", args.stage1_inference_steps),
        ("--stage2-inference-steps", args.stage2_inference_steps),
    )
    for option, value in optional_values:
        if value is not None:
            argv.extend((option, str(value)))
    if args.flat_shading:
        argv.append("--flat-shading")
    if args.profile_memory:
        argv.append("--profile-memory")
    return build_bridge_parser().parse_args(argv)


def _manifest_signature(path: Path) -> Optional[tuple[int, int]]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    return stat.st_mtime_ns, stat.st_size


def print_summary(manifest_path: Path) -> None:
    document = read_manifest_document(manifest_path)
    predictions = document["predictions"]
    successful = [
        prediction
        for prediction in predictions
        if prediction.get("mesh", {}).get("status") == "ok"
    ]
    failed = [
        prediction
        for prediction in predictions
        if prediction.get("mesh", {}).get("status") == "failed"
    ]
    print("\nDemo summary")
    print(f"  Detections: {len(predictions)}")
    print(f"  Meshes: {len(successful)} succeeded, {len(failed)} failed")
    print(f"  Manifest: {manifest_path}")
    if successful:
        print("  GLBs:")
        for prediction in successful:
            mesh_path = Path(prediction["mesh"]["path"])
            if not mesh_path.is_absolute():
                mesh_path = (manifest_path.parent / mesh_path).resolve()
            print(f"    {mesh_path}")


def run(args: argparse.Namespace) -> int:
    image, model, config, executable, output = _validate_inputs(args)
    bridge_args = _bridge_arguments(
        args,
        image_path=image,
        model_dir=model,
        sam3d_config=config,
        executable=executable,
        output_dir=output,
    )
    manifest_path = output / "segmentation" / "manifest.json"
    before = _manifest_signature(manifest_path)
    return_code = int(run_bridge(bridge_args))
    after = _manifest_signature(manifest_path)
    if after is not None and (return_code == 0 or after != before):
        print_summary(manifest_path)
    elif return_code != 0:
        print("The pipeline failed before producing a new manifest.", file=sys.stderr)
    else:
        raise RuntimeError("pipeline succeeded without producing a manifest")
    return return_code


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        return run(build_parser().parse_args(argv))
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
