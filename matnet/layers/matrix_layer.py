"""Core matrix layer defined in DOCS_SUMMARY.md."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class MatrixLayer(nn.Module):
    """Connects input matrix neurons to output matrix neurons."""

    n: int
    input_dim: int
    output_dim: int
    kernel_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.shape[-3] != self.input_dim:
            raise ValueError(
                f"Expected input neuron axis {self.input_dim}, got {x.shape[-3]}."
            )
        if x.shape[-2:] != (self.n, self.n):
            raise ValueError(
                f"Expected matrix shape ({self.n}, {self.n}), got {x.shape[-2:]}."
            )

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (self.output_dim, self.input_dim, self.n, self.n),
            self.dtype,
        )
        y = jnp.einsum("o i r c, ... i r c -> ... o r c", kernel, x)

        if self.use_bias:
            bias = self.param(
                "bias",
                self.bias_init,
                (self.output_dim, self.n, self.n),
                self.dtype,
            )
            y = y + bias

        return y.astype(self.dtype)
