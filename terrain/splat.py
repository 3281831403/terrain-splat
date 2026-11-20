"""Contracts for layered terrain material splatting.

A layer is a plain dictionary built inline by the caller::

    {
        "id": "grass",          # str, material name
        "weight": 0.5,          # float in [0, 1]
        "uv_scale": 1.0,        # float > 0, texel density multiplier
        "tile": 0,              # int >= 0, tile index for UV stitching
    }

``blend`` additionally reads an optional ``"color"`` key containing an
``(r, g, b)`` triple whose components are floats in [0, 1].  No layer, list, or
dictionary supplied by a caller is mutated.

All weights are normalized values in [0, 1].  Helpers that return weight lists
make the values sum to 1 within an absolute tolerance of 1e-6 unless their
docstring says otherwise.  UV coordinates are wrapped component-wise into
[0, 1) before sampling, so negative and out-of-range UVs are valid input.  All
numeric assertions in the test suite use an absolute tolerance of 1e-6.

The implementation is deterministic, offline, and uses only the Python
standard library (``math`` is expected) plus pytest in the tests.  Every public
function below implements the contract stated in its docstring and reports
contract violations with :class:`SplatError`.
"""

import math

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


class SplatError(Exception):
    """Raised when a layer, UV, tile, or batch argument violates its contract."""


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _is_number(value):
    """True for real numeric values (bools are not treated as numbers)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value):
    """True for finite real numbers, without leaking overflow exceptions."""
    if not _is_number(value):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def _is_int(value):
    """True for integers (bools are not treated as integers)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _sequence(value, what):
    """Validate a sequence of items and return it unchanged."""
    if not isinstance(value, (list, tuple)):
        raise SplatError("%s must be a list or tuple" % what)
    return value


def _layer_dict(layer):
    if not isinstance(layer, dict):
        raise SplatError("each layer must be a dictionary")
    return layer


def _layer_weight(layer):
    """Return the validated ``weight`` of a layer as a float in [0, 1]."""
    data = _layer_dict(layer)
    if "weight" not in data:
        raise SplatError("layer is missing its 'weight'")
    weight = data["weight"]
    if not _is_finite_number(weight) or weight < 0.0 or weight > 1.0:
        raise SplatError("layer weight must be a finite number in [0, 1]")
    return float(weight)


def _layer_uv_scale(layer):
    """Return the validated ``uv_scale`` of a layer as a positive float."""
    data = _layer_dict(layer)
    if "uv_scale" not in data:
        raise SplatError("layer is missing its 'uv_scale'")
    scale = data["uv_scale"]
    if not _is_finite_number(scale) or scale <= 0.0:
        raise SplatError("layer uv_scale must be a finite number greater than 0")
    return float(scale)


def _layer_tile(layer):
    """Return the validated non-negative integer ``tile`` index of a layer."""
    data = _layer_dict(layer)
    if "tile" not in data:
        raise SplatError("layer is missing its 'tile'")
    tile = data["tile"]
    if not _is_int(tile) or tile < 0:
        raise SplatError("layer tile must be a non-negative integer")
    return tile


def _layer_id(layer):
    """Return the validated string ``id`` of a layer."""
    data = _layer_dict(layer)
    if "id" not in data:
        raise SplatError("layer is missing its 'id'")
    layer_id = data["id"]
    if not isinstance(layer_id, str):
        raise SplatError("layer id must be a string")
    return layer_id


def _layer_color(layer):
    """Return the validated ``color`` of a layer as an (r, g, b) tuple."""
    data = _layer_dict(layer)
    if "color" not in data:
        raise SplatError("layer is missing its 'color'")
    color = data["color"]
    if not isinstance(color, (list, tuple)) or len(color) != 3:
        raise SplatError("layer color must be an (r, g, b) triple")
    for component in color:
        if not _is_finite_number(component) or component < 0.0 or component > 1.0:
            raise SplatError("layer color components must be finite numbers in [0, 1]")
    return tuple(float(component) for component in color)


def _uv_pair(uv):
    """Return a validated pair of finite floats."""
    if not isinstance(uv, (list, tuple)) or len(uv) != 2:
        raise SplatError("uv must be a pair of finite floats")
    u, v = uv
    if not _is_finite_number(u) or not _is_finite_number(v):
        raise SplatError("uv must be a pair of finite floats")
    return float(u), float(v)


def _wrap01(value):
    """Wrap a finite value into [0, 1)."""
    wrapped = value - math.floor(value)
    # ``math.floor`` already guarantees [0, 1) for finite input; the clamps
    # only guard against any floating-point edge case.
    if wrapped < 0.0:
        wrapped += 1.0
    elif wrapped >= 1.0:
        wrapped -= 1.0
    return float(wrapped)


def _validate_layer_full(layer):
    """Validate a layer against the full module-level shape."""
    _layer_id(layer)
    _layer_weight(layer)
    _layer_uv_scale(layer)
    _layer_tile(layer)
    _layer_color(layer)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_weights(layers):
    """Normalize the layer weights so the returned values sum to 1.

    Contract:
      * Every layer's ``weight`` must be a finite number in [0, 1].  Missing,
        non-numeric, negative, greater-than-one, NaN, and infinite weights
        raise ``SplatError``.
      * The returned list has one float per layer, in input order, and sums to
        1 within an absolute tolerance of 1e-6.
      * If all weights are exactly zero, the total weight is distributed
        uniformly instead of dividing by zero: four zero-weight layers become
        four weights of 0.25.
      * An empty layer list returns an empty list; it does not raise.
      * The input sequence and its dictionaries are not mutated.
    """
    layers = _sequence(layers, "layers")
    weights = [_layer_weight(layer) for layer in layers]
    if not weights:
        return []

    total = math.fsum(weights)
    if total == 0.0:
        # No usable weight: distribute the total uniformly instead of
        # dividing by zero (which would produce NaNs).
        uniform = 1.0 / len(weights)
        return [uniform] * len(weights)

    return [weight / total for weight in weights]


def resolve_tiles(layers, tile_size):
    """Resolve each layer's integer tiled texture dimensions.

    Contract:
      * ``tile_size`` is a positive integer base size in texels.  Values less
        than or equal to zero, or non-integers, raise ``SplatError``.
      * Each layer's positive finite ``uv_scale`` multiplies ``tile_size``.
        The product is rounded to the nearest integer and clamped to a minimum
        of 1, so sampling can never divide by a zero-sized tile.
      * The return value is a list of ``(width, height)`` tuples in the same
        order as ``layers``.  Both dimensions of a layer are equal because the
        tile is square.
      * An empty layer list returns an empty list.
      * Invalid ``uv_scale`` values raise ``SplatError``.
    """
    if not _is_int(tile_size) or tile_size <= 0:
        raise SplatError("tile_size must be a positive integer")

    layers = _sequence(layers, "layers")
    sizes = []
    for layer in layers:
        scale = _layer_uv_scale(layer)
        try:
            product = float(tile_size) * scale
        except OverflowError:
            raise SplatError("tile size is too large to resolve")
        if not math.isfinite(product):
            raise SplatError("tile size is too large to resolve")
        size = int(math.floor(product + 0.5))
        if size < 1:
            size = 1
        sizes.append((size, size))
    return sizes


def blend(layers, weights, uv):
    """Blend layer colors at ``uv`` and return the resulting RGB triple.

    Contract:
      * ``uv`` is a pair of finite floats.  It is wrapped component-wise into
        [0, 1) before sampling, so ``(0.25, 0.5)``, ``(1.25, 0.5)``, and
        ``(-0.75, 0.5)`` address the same location and must blend identically.
        The result must not contain a seam or a black line at a wrap boundary.
      * ``weights`` must have the same length as ``layers``.  Every weight is
        in [0, 1] and the values sum to 1 within 1e-6; otherwise raise
        ``SplatError``.  Four layer weights of 0.5 each are therefore invalid
        until they are normalized; blending must not double the brightness.
      * Each layer's ``color`` is an RGB triple with components in [0, 1].
        The output is the weighted sum of those colors, as a list of three
        floats in [0, 1] and accurate to 1e-6.
      * Empty ``layers`` and empty ``weights`` return ``[0.0, 0.0, 0.0]``
        (black), which is the defined empty-terrain color.
    """
    # Wrapping the UVs first keeps sampling continuous across tile boundaries:
    # an out-of-range coordinate addresses the same wrapped sample rather than
    # falling off the texture and returning a black (unbound) texel.
    u, v = _uv_pair(uv)
    _wrap01(u)
    _wrap01(v)

    layers = _sequence(layers, "layers")
    weights = _sequence(weights, "weights")
    if len(layers) != len(weights):
        raise SplatError("weights must have the same length as layers")
    if not layers:
        return [0.0, 0.0, 0.0]

    for weight in weights:
        if not _is_finite_number(weight) or weight < 0.0 or weight > 1.0:
            raise SplatError("weights must be finite numbers in [0, 1]")
    if abs(math.fsum(float(weight) for weight in weights) - 1.0) > 1e-6:
        raise SplatError("weights must sum to 1")

    result = [0.0, 0.0, 0.0]
    for layer, weight in zip(layers, weights):
        red, green, blue = _layer_color(layer)
        weight = float(weight)
        result[0] += weight * red
        result[1] += weight * green
        result[2] += weight * blue

    # The weighted sum of values in [0, 1] is already in [0, 1]; clamp only to
    # absorb floating-point drift at the extremes.
    return [min(1.0, max(0.0, component)) for component in result]


def height_blend(layers, heights, slope):
    """Select normalized layer weights from height affinity and slope.

    Contract:
      * ``heights`` contains one finite, non-negative height affinity per
        layer.  Larger affinity means the material belongs higher on the
        terrain.  Height affinities are relative, not absolute: multiplying
        every affinity by the same positive constant must not change the
        returned weights, so no fixed metre-scale threshold may be baked in.
      * ``slope`` is a finite float in [0, 1], where 0 is flat ground and 1 is
        a vertical cliff.  Values outside that range raise ``SplatError``.
      * Grass (``id == "grass"``) fades out as slope increases and must have
        exactly zero weight on a steep slope (``slope >= 0.65``); steep terrain
        must not grow grass.  Other materials are weighted by their height
        affinities.
      * The returned list has one float per layer and sums to 1 within 1e-6.
        If all effective affinities are zero, raise ``SplatError`` rather than
        dividing by zero.
    """
    if not _is_finite_number(slope) or slope < 0.0 or slope > 1.0:
        raise SplatError("slope must be a finite number in [0, 1]")

    layers = _sequence(layers, "layers")
    heights = _sequence(heights, "heights")
    if len(layers) != len(heights):
        raise SplatError("heights must have the same length as layers")

    for height in heights:
        if not _is_finite_number(height) or height < 0.0:
            raise SplatError("height affinities must be finite and non-negative")

    # Slope is handled by a relative grass factor, and all affinities are
    # normalized by their own total, so no absolute metre-scale threshold is
    # baked in and rescaling the affinities leaves the weights unchanged.
    effective = []
    for layer, height in zip(layers, heights):
        layer_id = _layer_id(layer)
        if layer_id == "grass":
            factor = _grass_factor(slope)
        else:
            factor = 1.0
        effective.append(float(height) * factor)

    total = math.fsum(effective)
    if total == 0.0:
        raise SplatError("all effective height affinities are zero")

    return [value / total for value in effective]


def _grass_factor(slope):
    """Relative grass affinity for a slope: 1 on flat ground, 0 on a cliff."""
    if slope <= 0.0:
        return 1.0
    if slope >= 0.65:
        return 0.0
    return 1.0 - slope / 0.65


def mip_bias(layers, distance):
    """Return a per-layer mip bias for a camera-to-surface distance.

    Contract:
      * ``distance`` is a finite, non-negative float in world units.  A
        distance of zero is treated as the near plane (1.0) so the logarithm
        stays finite; negative, NaN, or infinite distances raise
        ``SplatError``.
      * The returned list has one finite float per layer.  Bias increases
        with distance and is never below 0.0, the minimum/base mip layer, so
        distant terrain never samples below its coarsest valid layer.
      * A layer with a larger positive ``uv_scale`` (higher texel frequency)
        receives at least as much bias as a lower-frequency layer at the same
        distance.  Values are deterministic and must not change when the
        layer list is re-read without modification.
    """
    if not _is_finite_number(distance) or distance < 0.0:
        raise SplatError("distance must be a finite, non-negative number")

    layers = _sequence(layers, "layers")
    near_plane = 1.0
    effective_distance = distance if distance > 0.0 else near_plane
    distance_bias = math.log2(effective_distance / near_plane)

    biases = []
    for layer in layers:
        scale = _layer_uv_scale(layer)
        # A higher texel frequency needs more bias at the same distance, so
        # the per-layer term is the base-2 logarithm of its uv_scale.
        bias = distance_bias + math.log2(scale)
        if bias < 0.0:
            bias = 0.0
        biases.append(float(bias))
    return biases


def stitch_uv(uv, tile):
    """Map a tile-local UV pair to continuous global UV coordinates.

    Contract:
      * ``uv`` is a pair of finite floats and ``tile`` is a non-negative
        integer tile index.  Invalid input raises ``SplatError``.
      * Both UV components are offset by the integer ``tile`` index.  Thus the
        right/top edge of tile ``t`` and the left/bottom edge of tile ``t+1``
        land on exactly the same global coordinate, with no half-tile seam:
        ``stitch_uv((1.0, 1.0), 0) == stitch_uv((0.0, 0.0), 1)``.
      * The returned value is a ``(u, v)`` tuple of floats accurate to 1e-6.
        It is not wrapped back into [0, 1); callers that need wrapping pass
        the result to ``blend``.
    """
    u, v = _uv_pair(uv)
    if not _is_int(tile) or tile < 0:
        raise SplatError("tile must be a non-negative integer")
    return (float(u + tile), float(v + tile))


def limit_layers(layers, max_layers):
    """Cap a layer list at ``max_layers`` and renormalize retained weights.

    Contract:
      * ``max_layers`` must be a positive integer; zero or a negative value
        raises ``SplatError``.
      * When ``len(layers) <= max_layers``, return shallow copies of the
        layers in their original order without changing their weights.
      * When the cap is exceeded, sort by weight in descending order (stable
        for ties), keep the first ``max_layers`` layers, and set their weights
        so the kept weights sum to 1 within 1e-6.  The discarded layers must
        not contribute weight and the retained list must not contain a
        truncated, unnormalized remainder.
      * The returned list contains dictionaries in the retained order.  The
        input list and dictionaries are not mutated.
    """
    if not _is_int(max_layers) or max_layers <= 0:
        raise SplatError("max_layers must be a positive integer")

    layers = _sequence(layers, "layers")
    # Validate every weight, including those of layers that may be discarded.
    weights = [_layer_weight(layer) for layer in layers]

    if len(layers) <= max_layers:
        return [dict(layer) for layer in layers]

    # Stable sort: equal weights keep their original relative order.
    ordered = sorted(zip(weights, layers), key=lambda item: -item[0])
    kept = ordered[:max_layers]
    kept_total = math.fsum(weight for weight, _layer in kept)

    result = []
    for weight, layer in kept:
        copy = dict(layer)
        if kept_total == 0.0:
            # All surviving weights are zero: redistribute uniformly rather
            # than leaving a division by zero and a NaN weight.
            copy["weight"] = 1.0 / len(kept)
        else:
            copy["weight"] = weight / kept_total
        result.append(copy)
    return result


def splat_batch(tiles, config):
    """Splat a batch of tiles atomically and return the generated results.

    Contract:
      * ``config`` is a dictionary with ``tile_size`` (positive int, default
        64), ``max_layers`` (positive int, default 4), and ``slope`` (float in
        [0, 1], default 0.0).  Missing optional keys use those defaults.
      * ``tiles`` is a sequence of dictionaries.  Each has a unique string
        ``id``, a non-empty ``layers`` list, and an optional ``uv`` pair
        (default ``(0.5, 0.5)``).  Layer dictionaries follow the module-level
        shape and carry an RGB ``color`` for blending.
      * Every tile is validated before any result is produced.  If any tile is
        invalid, raise ``SplatError`` and return no partial batch: the bad tile
        must not simply be skipped, because that would leave a missing block
        in the output.
      * On success return ``{"count": len(tiles), "tiles": [...]}``.  The
        inner list preserves input order; each entry is a dictionary with
        ``id``, ``weights``, ``tiles`` (the resolved per-layer sizes), and
        ``color``.  ``count`` equals the number of input tiles.
    """
    config = _validated_batch_config(config)
    tile_size = config["tile_size"]
    max_layers = config["max_layers"]

    tiles = _sequence(tiles, "tiles")

    # Validate every tile up front.  A bad tile must abort the whole batch
    # instead of being skipped, which would leave a missing block in output.
    validated = []
    seen_ids = set()
    for tile in tiles:
        if not isinstance(tile, dict):
            raise SplatError("each tile must be a dictionary")
        tile_id = tile.get("id")
        if not isinstance(tile_id, str) or not tile_id:
            raise SplatError("each tile must have a non-empty string id")
        if tile_id in seen_ids:
            raise SplatError("tile ids must be unique")
        seen_ids.add(tile_id)

        layers = tile.get("layers")
        if not isinstance(layers, (list, tuple)) or not layers:
            raise SplatError("each tile must have a non-empty layers list")
        for layer in layers:
            _validate_layer_full(layer)

        uv = _uv_pair(tile.get("uv", (0.5, 0.5)))
        validated.append((tile_id, list(layers), uv))

    results = []
    for tile_id, layers, uv in validated:
        limited = limit_layers(layers, max_layers)
        weights = normalize_weights(limited)
        sizes = resolve_tiles(limited, tile_size)
        color = blend(limited, weights, uv)
        results.append(
            {
                "id": tile_id,
                "weights": weights,
                "tiles": sizes,
                "color": color,
            }
        )

    return {"count": len(tiles), "tiles": results}


def _validated_batch_config(config):
    """Validate a batch config dict and fill in the documented defaults."""
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise SplatError("config must be a dictionary")

    tile_size = config.get("tile_size", 64)
    max_layers = config.get("max_layers", 4)
    slope = config.get("slope", 0.0)

    if not _is_int(tile_size) or tile_size <= 0:
        raise SplatError("config tile_size must be a positive integer")
    if not _is_int(max_layers) or max_layers <= 0:
        raise SplatError("config max_layers must be a positive integer")
    if not _is_finite_number(slope) or slope < 0.0 or slope > 1.0:
        raise SplatError("config slope must be a finite number in [0, 1]")

    return {"tile_size": tile_size, "max_layers": max_layers, "slope": float(slope)}
