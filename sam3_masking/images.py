from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image, ImageOps

ImageInput = Union[str, PathLike[str], Image.Image, np.ndarray]


def normalize_image(image: ImageInput) -> Image.Image:
    """Return an EXIF-oriented RGB PIL image without mutating the input."""

    if isinstance(image, (str, PathLike)):
        with Image.open(Path(image)) as opened:
            return ImageOps.exif_transpose(opened).convert("RGB").copy()
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB").copy()
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.ndim not in (2, 3):
            raise ValueError(
                "NumPy image must have shape (H, W), (H, W, 1), "
                "(H, W, 3), or (H, W, 4)"
            )
        if array.ndim == 3 and array.shape[2] not in (1, 3, 4):
            raise ValueError(f"unsupported channel count {array.shape[2]}")
        if array.dtype == np.bool_:
            array = array.astype(np.uint8) * 255
        elif np.issubdtype(array.dtype, np.floating):
            if not np.isfinite(array).all():
                raise ValueError("NumPy image contains non-finite values")
            if array.size and array.min() >= 0 and array.max() <= 1:
                array = np.rint(array * 255)
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[..., 0]
        return Image.fromarray(array).convert("RGB")
    raise TypeError("image must be a path, PIL image, or NumPy array")
