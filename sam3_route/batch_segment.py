from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

from sam3_masking.artifacts import write_mask_manifest
from sam3_masking.generator import Sam3MaskGenerator
from sam3_masking.prompts import build_prompt_catalog, parse_prompt_catalog

from .artifacts import (
    artifact_path,
    atomic_write_json,
    read_route_manifest,
    relative_artifact,
    software_versions,
)

_ARTIFACT_SETS = {
    "objects": ("segmentation", "mask_manifest", "prompts"),
    "surface": ("surface-segmentation", "surface_mask_manifest", "surface_prompts"),
}


def batch_segment_route(
    run_dir: Path,
    *,
    model_dir: Path,
    prompts: Sequence[str],
    synonyms: str = "",
    score_threshold: float = 0.5,
    mask_threshold: float = 0.5,
    device: str = "auto",
    dtype: str = "auto",
    overwrite: bool = False,
    artifact_set: str = "objects",
    frame_ids: Optional[Sequence[str]] = None,
    generator_factory: Callable[..., Any] = Sam3MaskGenerator.from_pretrained,
) -> int:
    run_dir = run_dir.expanduser().resolve()
    manifest_path = run_dir / "route-manifest.json"
    manifest = read_route_manifest(manifest_path)
    catalog = build_prompt_catalog(prompts, synonyms)
    normalized_prompts = catalog.prompts
    try:
        output_name, manifest_field, prompts_field = _ARTIFACT_SETS[artifact_set]
    except KeyError as exc:
        raise ValueError(f"unsupported artifact set {artifact_set!r}") from exc
    count = 0
    with generator_factory(model_dir, device=device, dtype=dtype) as generator:
        frames = list(manifest["keyframes"])
        if artifact_set == "objects":
            frames.extend(manifest.get("recovery_frames", []))
        selected_ids = None if frame_ids is None else set(frame_ids)
        for frame in frames:
            if selected_ids is not None and frame["id"] not in selected_ids:
                continue
            image_path = artifact_path(run_dir, frame["rgb"])
            output_dir = run_dir / "frames" / frame["id"] / output_name
            output_manifest = output_dir / "manifest.json"
            if output_manifest.exists() and not overwrite:
                frame[manifest_field] = relative_artifact(run_dir, output_manifest)
                continue
            if output_dir.exists() and overwrite:
                shutil.rmtree(output_dir)
            result = generator.segment(
                image_path,
                normalized_prompts,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                source_id=frame["id"],
                synonym_to_canonical=catalog.synonym_to_canonical,
            )
            output_manifest = write_mask_manifest(
                result, output_dir, image_path=image_path
            )
            frame[manifest_field] = relative_artifact(run_dir, output_manifest)
            count += len(result.predictions)
    manifest[prompts_field] = list(normalized_prompts)
    if artifact_set == "objects":
        manifest["synonyms"] = catalog.normalized_synonyms()
        manifest["prompt_categories"] = list(catalog.categories)
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
        description="Run one persistent SAM 3 model over route keyframes.",
        allow_abbrev=False,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--synonyms", default="")
    parser.add_argument("--frame-id", action="append", dest="frame_ids")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument(
        "--artifact-set", choices=tuple(_ARTIFACT_SETS), default="objects"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = parse_prompt_catalog(args.prompts, args.synonyms)
    count = batch_segment_route(
        args.run_dir,
        model_dir=args.model_dir,
        prompts=catalog.prompts,
        synonyms=args.synonyms,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        device=args.device,
        dtype=args.dtype,
        overwrite=args.overwrite,
        artifact_set=args.artifact_set,
        frame_ids=args.frame_ids,
    )
    print(f"Wrote {count} new SAM 3 route observation(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
