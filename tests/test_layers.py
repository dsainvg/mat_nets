from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax.core import freeze, unfreeze

from matnet.layers.decompression import DecompressionLayer
from matnet.layers.input_scaling import InputScaling
from matnet.layers.matrix_layer import MatrixLayer
from matnet.normalization import MatrixBatchNorm, MatrixLayerNorm


def test_input_scaling_vector_mode_outputs_matrix() -> None:
    layer = InputScaling(n=3, input_dim=4)
    x = jnp.ones((2, 4), dtype=jnp.float32)
    params = layer.init(jax.random.PRNGKey(0), x)
    y = layer.apply(params, x)
    assert y.shape == (2, 3, 3)


def test_input_scaling_scalar_mode_outputs_matrix() -> None:
    layer = InputScaling(n=2)
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    params = layer.init(jax.random.PRNGKey(1), x)
    y = layer.apply(params, x)
    assert y.shape == (3, 2, 2)


def test_matrix_layer_matches_documented_formula() -> None:
    layer = MatrixLayer(n=2, input_dim=2, output_dim=1, use_bias=True)
    x = jnp.array(
        [[[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]],
        dtype=jnp.float32,
    )
    params = layer.init(jax.random.PRNGKey(2), x)
    updated = unfreeze(params)
    updated["params"]["kernel"] = jnp.array(
        [[[[2.0, 3.0], [4.0, 5.0]], [[6.0, 7.0], [8.0, 9.0]]]],
        dtype=jnp.float32,
    )
    updated["params"]["bias"] = jnp.array([[[10.0, 11.0], [12.0, 13.0]]], dtype=jnp.float32)
    params = freeze(updated)

    y = layer.apply(params, x)
    expected = jnp.array([[[[42.0, 59.0], [80.0, 105.0]]]], dtype=jnp.float32)
    np.testing.assert_allclose(y, expected)


def test_matrix_layer_validates_input_neuron_axis() -> None:
    layer = MatrixLayer(n=2, input_dim=3, output_dim=1)
    valid_x = jnp.ones((4, 3, 2, 2), dtype=jnp.float32)
    invalid_x = jnp.ones((4, 2, 2, 2), dtype=jnp.float32)
    params = layer.init(jax.random.PRNGKey(3), valid_x)
    try:
        layer.apply(params, invalid_x)
    except ValueError as exc:
        assert "Expected input neuron axis" in str(exc)
    else:
        raise AssertionError("MatrixLayer should reject mismatched neuron axis.")


def test_decompression_layer_returns_vector_output() -> None:
    layer = DecompressionLayer(output_dim=5)
    x = jnp.ones((4, 3, 2, 2), dtype=jnp.float32)
    params = layer.init(jax.random.PRNGKey(4), x)
    y = layer.apply(params, x)
    assert y.shape == (4, 5)


def test_matrix_layer_norm_preserves_shape() -> None:
    layer = MatrixLayerNorm()
    x = jax.random.normal(jax.random.PRNGKey(5), (3, 4, 2, 2))
    params = layer.init(jax.random.PRNGKey(6), x)
    y = layer.apply(params, x)
    assert y.shape == x.shape


def test_matrix_batch_norm_updates_batch_stats() -> None:
    layer = MatrixBatchNorm()
    x = jax.random.normal(jax.random.PRNGKey(7), (3, 4, 2, 2))
    variables = layer.init(jax.random.PRNGKey(8), x, training=True)
    y, mutated = layer.apply(variables, x, training=True, mutable=["batch_stats"])
    assert y.shape == x.shape
    assert "batch_stats" in mutated
    assert "mean" in mutated["batch_stats"]
    assert "var" in mutated["batch_stats"]
