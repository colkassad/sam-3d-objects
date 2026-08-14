from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from .artifacts import (
    load_mask_manifest,
    prompt_slug,
    update_mesh_records,
)
from .checkpoint import find_repo_root


def _load_sam3d_api(repo_root: Path) -> tuple[Callable[..., Any], Callable[..., Any]]:
    notebook_dir = repo_root / "notebook"
    if str(notebook_dir) not in sys.path:
        sys.path.insert(0, str(notebook_dir))
    try:
        from inference import Inference, load_image
    except ImportError as exc:
        raise RuntimeError(
            "could not import the SAM 3D inference API; run this command from "
            "the configured SAM 3D Objects environment"
        ) from exc
    return Inference, load_image


def reconstruct_manifest(
    manifest_path: Path,
    *,
    image_path: Path,
    output_dir: Path,
    sam3d_config: Path,
    seed: Optional[int] = 42,
    mesh_target_faces: Optional[int] = None,
    flat_shading: bool = False,
    stage1_inference_steps: Optional[int] = None,
    stage2_inference_steps: Optional[int] = None,
    compile_model: bool = False,
    memory_profile: str = "low_vram",
    profile_memory: bool = False,
    inference_factory: Optional[Callable[..., Any]] = None,
    image_loader: Optional[Callable[[Path], Any]] = None,
    repo_root: Optional[Path] = None,
) -> int:
    """Generate one mesh per manifest prediction, returning the failure count."""

    frame = load_mask_manifest(manifest_path)
    if not frame.predictions:
        return 0
    if inference_factory is None or image_loader is None:
        Inference, load_image = _load_sam3d_api(repo_root or find_repo_root())
        inference_factory = inference_factory or Inference
        image_loader = image_loader or load_image

    image = image_loader(image_path)
    if tuple(image.shape[:2]) != (frame.height, frame.width):
        raise ValueError(
            f"source image shape {tuple(image.shape[:2])} does not match "
            f"manifest shape {(frame.height, frame.width)}"
        )
    inference = inference_factory(
        str(sam3d_config),
        compile=compile_model,
        memory_profile=memory_profile,
        profile_memory=profile_memory,
    )
    meshes_dir = output_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    failure_count = 0
    manifest_parent = manifest_path.expanduser().resolve().parent

    for prediction in frame.predictions:
        mesh_name = f"{prediction.id}-{prompt_slug(prediction.prompt)}.glb"
        mesh_path = (meshes_dir / mesh_name).resolve()
        try:
            output = inference(
                image,
                prediction.mask,
                seed=seed,
                mesh_target_faces=mesh_target_faces,
                flat_shading=flat_shading,
                stage1_inference_steps=stage1_inference_steps,
                stage2_inference_steps=stage2_inference_steps,
            )
            if "glb" not in output:
                raise RuntimeError("SAM 3D output did not contain a GLB mesh")
            output["glb"].export(str(mesh_path))
            recorded_path = Path(
                os.path.relpath(mesh_path, start=manifest_parent)
            ).as_posix()
            records[prediction.id] = {
                "status": "ok",
                "path": recorded_path,
            }
        except Exception as exc:
            failure_count += 1
            message = str(exc).replace("\n", " ").strip()
            records[prediction.id] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {message}"[:500],
            }
    update_mesh_records(manifest_path, records)
    return failure_count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Segment prompted objects with SAM 3, then reconstruct each " "with SAM 3D."
        )
    )
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--sam3-executable", default="sam3-mask")
    parser.add_argument("--sam3-model-dir", type=Path, required=True)
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
    parser.add_argument("--compile", action="store_true", dest="compile_model")
    parser.add_argument(
        "--memory-profile",
        choices=("auto", "low_vram", "resident"),
        default="low_vram",
    )
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--repo-root", type=Path)
    return parser


def run(
    args: argparse.Namespace, *, subprocess_run: Callable[..., Any] = subprocess.run
) -> int:
    repo_root = find_repo_root(args.repo_root)
    output_dir = args.output_dir.expanduser().resolve()
    image_path = args.image.expanduser().resolve()
    sam3_model_dir = args.sam3_model_dir.expanduser()
    if not sam3_model_dir.is_absolute():
        sam3_model_dir = repo_root / sam3_model_dir
    sam3_model_dir = sam3_model_dir.resolve()
    segmentation_dir = output_dir / "segmentation"
    manifest_path = segmentation_dir / "manifest.json"
    command = [
        str(args.sam3_executable),
        "--model-dir",
        str(sam3_model_dir),
        "--image",
        str(image_path),
        "--output-dir",
        str(segmentation_dir),
        "--score-threshold",
        str(args.score_threshold),
        "--mask-threshold",
        str(args.mask_threshold),
        "--device",
        args.sam3_device,
        "--dtype",
        args.sam3_dtype,
    ]
    if args.source_id is not None:
        command.extend(("--source-id", args.source_id))
    if args.profile_memory:
        command.append("--profile-memory")
    for prompt in args.prompt:
        command.extend(("--prompt", prompt))

    completed = subprocess_run(command, check=False)
    if completed.returncode != 0:
        return int(completed.returncode)

    sam3d_config = args.sam3d_config
    if not sam3d_config.is_absolute():
        sam3d_config = repo_root / sam3d_config
    failures = reconstruct_manifest(
        manifest_path,
        image_path=image_path,
        output_dir=output_dir,
        sam3d_config=sam3d_config,
        seed=args.seed,
        mesh_target_faces=args.mesh_target_faces,
        flat_shading=args.flat_shading,
        stage1_inference_steps=args.stage1_inference_steps,
        stage2_inference_steps=args.stage2_inference_steps,
        compile_model=args.compile_model,
        memory_profile=args.memory_profile,
        profile_memory=args.profile_memory,
        repo_root=repo_root,
    )
    if failures:
        print(
            f"Mesh generation completed with {failures} failure(s); "
            f"see {manifest_path}.",
            file=sys.stderr,
        )
        return 1
    frame = load_mask_manifest(manifest_path)
    print(f"Generated {len(frame.predictions)} mesh(es); see {manifest_path}.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
