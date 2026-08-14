from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class MaskPrediction:
    """One text-prompted SAM 3 instance at the source image resolution."""

    id: str
    prompt: str
    score: float
    box_xyxy: Tuple[float, float, float, float]
    mask: np.ndarray

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.id):
            raise ValueError(
                "id must start with an alphanumeric character and contain only "
                "letters, digits, dots, underscores, or hyphens"
            )
        mask = np.asarray(self.mask)
        if mask.ndim != 2:
            raise ValueError(f"mask must be two-dimensional, got shape {mask.shape}")
        if mask.dtype != np.bool_:
            raise TypeError(f"mask must have boolean dtype, got {mask.dtype}")
        if not mask.any():
            raise ValueError("mask must contain at least one foreground pixel")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if len(self.box_xyxy) != 4 or not np.isfinite(self.box_xyxy).all():
            raise ValueError("box_xyxy must contain four finite coordinates")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")


@dataclass(frozen=True)
class MaskFrame:
    """All SAM 3 instance masks produced for one image/frame."""

    width: int
    height: int
    predictions: Tuple[MaskPrediction, ...]
    source_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("frame width and height must be positive")
        expected_shape = (self.height, self.width)
        ids = set()
        for prediction in self.predictions:
            if prediction.mask.shape != expected_shape:
                raise ValueError(
                    f"mask {prediction.id!r} has shape {prediction.mask.shape}; "
                    f"expected {expected_shape}"
                )
            if prediction.id in ids:
                raise ValueError(f"duplicate prediction id {prediction.id!r}")
            ids.add(prediction.id)
