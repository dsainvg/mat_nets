"""High-level model builders for MatNet."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from flax import linen as nn
import jax.numpy as jnp

from ..activations import get_activation
from ..layers.decompression import DecompressionLayer
from ..layers.input_scaling import InputScaling
from ..layers.matrix_layer import MatrixLayer
from ..normalization import MatrixLayerNorm


class MatrixNetwork(nn.Module):
    """Configurable matrix neural network."""

    matrix_size: int
    hidden_dims: Sequence[int]
    output_dim: int
    activation: str | Callable[[jnp.ndarray], jnp.ndarray] = "relu"
    use_bias: bool = True
    use_input_scaling: bool = True
    use_normalization: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        del training
        activation_fn = get_activation(self.activation)

        if self.use_input_scaling:
            input_dim = None if x.ndim == 1 else x.shape[-1]
            x = InputScaling(n=self.matrix_size, input_dim=input_dim, dtype=self.dtype)(x)

        if x.ndim == 2:
            x = x[None, :, :]
        if x.ndim == 3:
            x = x[:, None, :, :]

        current_dim = x.shape[-3]
        for hidden_dim in self.hidden_dims:
            x = MatrixLayer(
                n=self.matrix_size,
                input_dim=current_dim,
                output_dim=hidden_dim,
                use_bias=self.use_bias,
                dtype=self.dtype,
            )(x)
            x = activation_fn(x)
            if self.use_normalization:
                x = MatrixLayerNorm(dtype=self.dtype)(x)
            current_dim = hidden_dim

        return DecompressionLayer(output_dim=self.output_dim, dtype=self.dtype)(x)


class SimpleMatrixNet(nn.Module):
    """Pre-configured 2-layer matrix network for quick prototyping."""

    matrix_size: int = 8
    hidden_dim: int = 16
    output_dim: int = 1
    input_dim: int = 10
    activation: str | Callable[[jnp.ndarray], jnp.ndarray] = "relu"
    use_bias: bool = True
    use_normalization: bool = True
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        model = MatrixNetwork(
            matrix_size=self.matrix_size,
            hidden_dims=(self.hidden_dim,),
            output_dim=self.output_dim,
            activation=self.activation,
            use_bias=self.use_bias,
            use_input_scaling=True,
            use_normalization=self.use_normalization,
            dtype=self.dtype,
        )
        return model(x, training=training)


def build_matrix_network(
    matrix_size: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str | Callable[[jnp.ndarray], jnp.ndarray] = "relu",
    use_bias: bool = True,
    use_input_scaling: bool = True,
    use_normalization: bool = True,
    dtype: Any = jnp.float32,
) -> MatrixNetwork:
    return MatrixNetwork(
        matrix_size=matrix_size,
        hidden_dims=tuple(hidden_dims),
        output_dim=output_dim,
        activation=activation,
        use_bias=use_bias,
        use_input_scaling=use_input_scaling,
        use_normalization=use_normalization,
        dtype=dtype,
    )
