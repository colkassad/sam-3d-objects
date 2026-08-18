from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .artifacts import write_mask_manifest
from .generator import Sam3MaskGenerator
from .prompts import parse_prompt_catalog


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate SAM 3 instance masks for one image and text prompts.",
        allow_abbrev=False,
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument(
        "--prompts",
        required=True,
        help="Comma-separated text concept prompts.",
    )
    parser.add_argument(
        "--synonyms",
        default="",
        help="Canonical groups: 'vehicle:car,truck;sign:road sign,traffic sign'.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype", choices=("auto", "bf16", "fp16", "fp32"), default="auto"
    )
    parser.add_argument("--profile-memory", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    catalog = parse_prompt_catalog(args.prompts, args.synonyms)
    torch = None
    if args.profile_memory:
        import torch as torch_module

        torch = torch_module
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    with Sam3MaskGenerator.from_pretrained(
        args.model_dir, device=args.device, dtype=args.dtype
    ) as generator:
        frame = generator.segment(
            args.image,
            catalog.prompts,
            score_threshold=args.score_threshold,
            mask_threshold=args.mask_threshold,
            source_id=args.source_id,
            synonym_to_canonical=catalog.synonym_to_canonical,
        )
        manifest_path = write_mask_manifest(
            frame, args.output_dir, image_path=args.image
        )
        device_type = generator.device.type

    print(f"Wrote {len(frame.predictions)} instance mask(s) to {manifest_path}.")
    if args.profile_memory and torch is not None and device_type == "cuda":
        peak_mib = torch.cuda.max_memory_allocated() / 1024**2
        print(f"Peak CUDA memory allocated: {peak_mib:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
