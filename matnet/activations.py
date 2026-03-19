"""Matrix activation functions applied element-wise."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

ActivationFn = Callable[[jax.Array], jax.Array]


def matrix_relu(x: jax.Array) -> jax.Array:
    return jax.nn.relu(x)


def matrix_leaky_relu(x: jax.Array, negative_slope: float = 0.01) -> jax.Array:
    return jax.nn.leaky_relu(x, negative_slope)


def matrix_sigmoid(x: jax.Array) -> jax.Array:
    return jax.nn.sigmoid(x)


def matrix_tanh(x: jax.Array) -> jax.Array:
    return jnp.tanh(x)


def matrix_swish(x: jax.Array) -> jax.Array:
    return x * jax.nn.sigmoid(x)


def matrix_gelu(x: jax.Array) -> jax.Array:
    return jax.nn.gelu(x)


def matrix_elu(x: jax.Array) -> jax.Array:
    return jax.nn.elu(x)


ACTIVATIONS: dict[str, ActivationFn] = {
    "relu": matrix_relu,
    "leaky_relu": matrix_leaky_relu,
    "sigmoid": matrix_sigmoid,
    "tanh": matrix_tanh,
    "swish": matrix_swish,
    "gelu": matrix_gelu,
    "elu": matrix_elu,
}


def get_activation(activation: str | ActivationFn) -> ActivationFn:
    if callable(activation):
        return activation
    key = activation.lower().strip()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{activation}'.")
    return ACTIVATIONS[key]
