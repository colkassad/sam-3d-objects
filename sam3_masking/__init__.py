"""Lightweight SAM 3 masking interfaces for SAM 3D Objects."""

from .artifacts import load_mask_manifest, write_mask_manifest
from .generator import Sam3MaskGenerator
from .prompts import PromptCatalog, build_prompt_catalog, parse_prompt_catalog
from .types import MaskFrame, MaskPrediction

__all__ = [
    "MaskFrame",
    "MaskPrediction",
    "PromptCatalog",
    "Sam3MaskGenerator",
    "build_prompt_catalog",
    "load_mask_manifest",
    "parse_prompt_catalog",
    "write_mask_manifest",
]
