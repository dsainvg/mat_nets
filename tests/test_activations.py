from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from matnet import activations


def test_matrix_relu_applies_elementwise() -> None:
    x = jnp.array([[[1.0, -2.0], [-3.0, 4.0]]], dtype=jnp.float32)
    y = activations.matrix_relu(x)
    expected = jnp.array([[[1.0, 0.0], [0.0, 4.0]]], dtype=jnp.float32)
    np.testing.assert_allclose(y, expected)


def test_matrix_sigmoid_preserves_shape_and_range() -> None:
    x = jax.random.normal(jax.random.PRNGKey(0), (5, 3, 3))
    y = activations.matrix_sigmoid(x)
    assert y.shape == x.shape
    assert jnp.all(y > 0.0)
    assert jnp.all(y < 1.0)


def test_get_activation_accepts_string_and_callable() -> None:
    relu_from_name = activations.get_activation("relu")
    relu_from_callable = activations.get_activation(activations.matrix_relu)
    x = jnp.array([[-1.0, 2.0]], dtype=jnp.float32)
    np.testing.assert_allclose(relu_from_name(x), relu_from_callable(x))
