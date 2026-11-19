"""Terrain material splatting package."""

from .splat import (
    SplatError,
    blend,
    height_blend,
    limit_layers,
    mip_bias,
    normalize_weights,
    resolve_tiles,
    splat_batch,
    stitch_uv,
)

__all__ = [
    "SplatError",
    "blend",
    "height_blend",
    "limit_layers",
    "mip_bias",
    "normalize_weights",
    "resolve_tiles",
    "splat_batch",
    "stitch_uv",
]
