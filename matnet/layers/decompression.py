"""Decompression layer for matrix outputs."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class DecompressionLayer(nn.Module):
    """Projects matrix representations back to standard vector outputs."""

    output_dim: int
    use_bias: bool = True
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        flat = x.reshape(x.shape[0], -1)
        dense = nn.Dense(
            features=self.output_dim,
            use_bias=self.use_bias,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            dtype=self.dtype,
        )
        return dense(flat)
