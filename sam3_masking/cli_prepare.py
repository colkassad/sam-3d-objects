from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

from .checkpoint import find_repo_root, prepare_model_bundle


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare an offline Transformers SAM 3 model bundle."
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("/mnt/d/Data/models/sam3.safetensors"),
        help="Existing Transformers-format SAM 3 safetensors checkpoint.",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("checkpoints/sam3-hf"))
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--model-id", default="facebook/sam3")
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--copy-weights",
        action="store_true",
        help="Copy the 3.44 GB checkpoint instead of creating a symlink.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing model.safetensors in the destination.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = find_repo_root(args.repo_root)
    model_dir = args.model_dir
    if not model_dir.is_absolute():
        model_dir = repo_root / model_dir
    bundle = prepare_model_bundle(
        args.weights,
        model_dir,
        repo_root=repo_root,
        model_id=args.model_id,
        revision=args.revision,
        copy_weights=args.copy_weights,
        force=args.force,
    )
    checkpoint = bundle["checkpoint"]
    print(
        f"Prepared {model_dir.resolve()} with "
        f"{checkpoint['tensor_count']:,} checkpoint tensors."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
