"""Contract tests for :mod:`terrain.splat`.

In the initial repository every stubbed function raises
``NotImplementedError``, so all fifteen tests fail.  The tests are the
specification: do not weaken or edit them while implementing the stubs.
"""

import math

import pytest

from terrain.splat import (
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


TOL = 1e-6


def _layer(layer_id, weight, uv_scale=1.0, tile=0, color=None):
    """Build the inline layer shape used throughout these tests."""
    layer = {
        "id": layer_id,
        "weight": weight,
        "uv_scale": uv_scale,
        "tile": tile,
    }
    if color is not None:
        layer["color"] = color
    return layer


def test_weight_normalization_prevents_double_brightness():
    """Defect 1: four 0.5 weights must become 0.25 each, not sum to 2.0."""
    layers = [
        _layer("grass", 0.5, 1.0, 0),
        _layer("rock", 0.5, 1.0, 1),
        _layer("dirt", 0.5, 1.0, 2),
        _layer("sand", 0.5, 1.0, 3),
    ]

    weights = normalize_weights(layers)

    assert len(weights) == 4
    assert all(0.0 <= weight <= 1.0 for weight in weights)
    assert math.isclose(sum(weights), 1.0, abs_tol=TOL)
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=TOL)


def test_all_zero_weights_fall_back_to_uniform():
    """Defect 2: all-zero weights must not divide by zero or produce NaN."""
    layers = [
        _layer("grass", 0.0, 1.0, 0),
        _layer("rock", 0.0, 1.0, 1),
        _layer("dirt", 0.0, 1.0, 2),
        _layer("sand", 0.0, 1.0, 3),
    ]

    weights = normalize_weights(layers)

    assert all(math.isfinite(weight) for weight in weights)
    assert math.isclose(sum(weights), 1.0, abs_tol=TOL)
    assert weights == pytest.approx([0.25, 0.25, 0.25, 0.25], abs=TOL)


def test_tile_size_never_rounds_down_to_zero():
    """Defect 3: a tiny uv_scale must still yield a 1x1 tile, not 0x0."""
    layers = [
        _layer("detail", 0.5, 0.001, 0),
        _layer("base", 0.5, 0.25, 1),
    ]

    sizes = resolve_tiles(layers, 64)

    assert sizes[0] == (1, 1)
    assert sizes[1] == (16, 16)


def test_blend_wraps_uv_instead_of_leaving_a_black_seam():
    """Defect 4: out-of-range UVs must wrap to the same sample."""
    layers = [
        _layer("rock", 1.0, 1.0, 0, color=(0.2, 0.4, 0.6)),
    ]
    weights = [1.0]

    base = blend(layers, weights, (0.25, 0.75))
    wrapped = blend(layers, weights, (4.25, 3.75))
    negative = blend(layers, weights, (-1.75, -2.25))

    assert base == pytest.approx([0.2, 0.4, 0.6], abs=TOL)
    assert wrapped == pytest.approx(base, abs=TOL)
    assert negative == pytest.approx(base, abs=TOL)


def test_steep_slope_has_no_grass_weight():
    """Defect 5: slope matters, so a cliff must not grow grass."""
    layers = [
        _layer("grass", 0.5, 1.0, 0),
        _layer("rock", 0.5, 1.0, 1),
    ]

    weights = height_blend(layers, [0.6, 0.4], 0.9)

    assert math.isclose(weights[0], 0.0, abs_tol=TOL)
    assert math.isclose(sum(weights), 1.0, abs_tol=TOL)
    assert weights[1] > 0.99


def test_height_weights_are_scale_invariant():
    """Defect 6: metres and centimetres must give the same relative weights."""
    layers = [
        _layer("grass", 0.5, 1.0, 0),
        _layer("rock", 0.5, 1.0, 1),
    ]

    small = height_blend(layers, [0.2, 0.8], 0.2)
    large = height_blend(layers, [20.0, 80.0], 0.2)

    assert small == pytest.approx(large, abs=TOL)


def test_mip_bias_grows_with_distance():
    """Defect 7: far terrain needs more mip bias to stop shimmering."""
    layers = [
        _layer("detail", 0.5, 2.0, 0),
        _layer("base", 0.5, 1.0, 1),
    ]

    near = mip_bias(layers, 1.0)
    far = mip_bias(layers, 64.0)

    assert len(near) == len(far) == 2
    assert all(math.isfinite(bias) for bias in near + far)
    assert all(far_bias > near_bias + TOL for near_bias, far_bias in zip(near, far))
    assert all(bias >= 0.0 for bias in far)


def test_stitch_uv_offsets_by_the_tile_index():
    """Defect 8: cross-tile UVs must be shifted, not sampled at the seam."""
    assert stitch_uv((0.25, 0.5), 2) == pytest.approx((2.25, 2.5), abs=TOL)


def test_limit_layers_renormalizes_the_survivors():
    """Defect 9: dropping layers must redistribute, not truncate, weight."""
    layers = [
        _layer("low", 0.1, 1.0, 0),
        _layer("top", 0.4, 1.0, 1),
        _layer("mid", 0.3, 1.0, 2),
        _layer("bottom", 0.2, 1.0, 3),
    ]

    kept = limit_layers(layers, 2)

    assert [layer["id"] for layer in kept] == ["top", "mid"]
    assert [layer["weight"] for layer in kept] == pytest.approx(
        [0.4 / 0.7, 0.3 / 0.7], abs=TOL
    )
    assert math.isclose(sum(layer["weight"] for layer in kept), 1.0, abs_tol=TOL)


def test_batch_with_an_invalid_tile_rolls_back():
    """Defect 10: a bad tile must abort the batch, not leave a missing block."""
    good = [_layer("grass", 1.0, 1.0, 0, color=(0.1, 0.5, 0.1))]
    tiles = [
        {"id": "a", "layers": good, "uv": (0.5, 0.5)},
        {"id": "broken", "layers": [], "uv": (0.5, 0.5)},
        {"id": "c", "layers": good, "uv": (0.5, 0.5)},
    ]
    config = {"tile_size": 64, "max_layers": 4, "slope": 0.1}

    with pytest.raises(SplatError):
        splat_batch(tiles, config)


def test_single_full_weight_layer_matches_its_color():
    """Combination: one layer with weight 1 must reproduce its color at 1e-6."""
    color = (0.125, 0.25, 0.5)
    layers = [_layer("rock", 1.0, 1.0, 0, color=color)]

    result = blend(layers, [1.0], (0.3, 0.8))

    assert result == pytest.approx(list(color), abs=TOL)


def test_empty_layers_blend_to_black():
    """Combination: an empty terrain is defined as black, not an error."""
    result = blend([], [], (0.2, 0.8))

    assert result == pytest.approx([0.0, 0.0, 0.0], abs=TOL)


def test_limit_layers_rejects_non_positive_cap():
    """Combination: a cap of zero or less is a contract violation."""
    layers = [_layer("only", 1.0, 1.0, 0)]

    for bad_cap in (0, -3):
        with pytest.raises(SplatError):
            limit_layers(layers, bad_cap)


def test_stitch_uv_is_continuous_at_the_tile_boundary():
    """Combination: adjoining tile edges must map to one global coordinate."""
    right_edge = stitch_uv((1.0, 1.0), 0)
    next_left_edge = stitch_uv((0.0, 0.0), 1)

    assert right_edge == pytest.approx((1.0, 1.0), abs=TOL)
    assert next_left_edge == pytest.approx(right_edge, abs=TOL)


def test_batch_success_preserves_input_order():
    """Combination: a valid batch returns every tile in its input order."""
    layers = [_layer("grass", 1.0, 1.0, 0, color=(0.1, 0.5, 0.1))]
    tiles = [
        {"id": "a", "layers": layers, "uv": (0.5, 0.5)},
        {"id": "b", "layers": layers, "uv": (0.5, 0.5)},
        {"id": "c", "layers": layers, "uv": (0.5, 0.5)},
    ]
    config = {"tile_size": 64, "max_layers": 4, "slope": 0.1}

    result = splat_batch(tiles, config)

    assert result["count"] == 3
    assert [tile["id"] for tile in result["tiles"]] == ["a", "b", "c"]
    assert all(
        {"id", "weights", "tiles", "color"} <= set(tile)
        for tile in result["tiles"]
    )
