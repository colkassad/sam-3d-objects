"""Resumable Ouster route ingestion, segmentation, and mesh reconstruction."""

import importlib
from typing import TYPE_CHECKING, Any

from .artifacts import ROUTE_MANIFEST_SCHEMA, TRACKS_SCHEMA


if TYPE_CHECKING:
    from .surface import (
        SURFACE_SCHEMA,
        SurfaceOutputs,
        SurfaceSegmentConfig,
        TinConfig,
        build_surface_tin,
        generate_surface_route,
        segment_surface_route,
    )


_SURFACE_EXPORTS = frozenset(
    {
        "SURFACE_SCHEMA",
        "SurfaceOutputs",
        "SurfaceSegmentConfig",
        "TinConfig",
        "build_surface_tin",
        "generate_surface_route",
        "segment_surface_route",
    }
)

__all__ = [
    "ROUTE_MANIFEST_SCHEMA",
    "SURFACE_SCHEMA",
    "TRACKS_SCHEMA",
    "SurfaceOutputs",
    "SurfaceSegmentConfig",
    "TinConfig",
    "build_surface_tin",
    "generate_surface_route",
    "segment_surface_route",
]


def __getattr__(name: str) -> Any:
    """Load SciPy-dependent surface exports only when they are requested."""

    if name not in _SURFACE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(".surface", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()).union(__all__))
