from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from sam3_masking.artifacts import write_mask_manifest
from sam3_masking.generator import Sam3MaskGenerator

from .artifacts import (
    artifact_path,
    atomic_write_json,
    read_route_manifest,
    relative_artifact,
    software_versions,
)


def batch_segment_route(
    run_dir: Path,
    *,
    model_dir: Path,
    prompts: Sequence[str],
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "auto",
    dtype: str = "auto",
    overwrite: bool = False,
    generator_factory: Callable[..., Any] = Sam3MaskGenerator.from_pretrained,
) -> int:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    normalized_prompts = tuple(value.strip() for value in prompts if value.strip())
    if not normalized_prompts:
        raise ValueError("at least one nonempty prompt is required")
    count = 0
    with generator_factory(model_dir, device=device, dtype=dtype) as generator:
        for frame in manifest["keyframes"]:
            image_path = artifact_path(run_dir, frame["rgb"])
            output_dir = run_dir / "frames" / frame["id"] / "segmentation"
            output_manifest = output_dir / "manifest.json"
            if output_manifest.exists() and not overwrite:
                frame["mask_manifest"] = relative_artifact(run_dir, output_manifest)
                continue
            if output_dir.exists() and overwrite:
                shutil.rmtree(output_dir)
            result = generator.segment(
                image_path,
                normalized_prompts,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                source_id=frame["id"],
            )
            output_manifest = write_mask_manifest(
                result, output_dir, image_path=image_path
            )
            frame["mask_manifest"] = relative_artifact(run_dir, output_manifest)
            count += len(result.predictions)
    manifest["prompts"] = list(normalized_prompts)
    manifest.setdefault("software", {}).update(
        software_versions(
            {
                "sam3": "sam3",
                "torch": "torch",
                "transformers": "transformers",
            }
        )
    )
    atomic_write_json(manifest_path, manifest)
    return count


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one persistent SAM 3 model over route keyframes."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    count = batch_segment_route(
        args.run_dir,
        model_dir=args.model_dir,
        prompts=args.prompt,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
        dtype=args.dtype,
        overwrite=args.overwrite,
    )
    print(f"Wrote {count} new SAM 3 route observation(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
