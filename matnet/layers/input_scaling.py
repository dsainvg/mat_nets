"""Input scaling layer for MatNet."""

from __future__ import annotations

from typing import Optional

from flax import linen as nn
import jax.numpy as jnp


class InputScaling(nn.Module):
    """Maps scalars or vectors into n x n matrices."""

    n: int
    input_dim: Optional[int] = None
    projection_init: nn.initializers.Initializer = nn.initializers.lecun_normal()
    bias_init: nn.initializers.Initializer = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.input_dim is None:
            scale = self.param(
                "scale",
                self.projection_init,
                (self.n, self.n),
                self.dtype,
            )
            bias = self.param(
                "bias",
                self.bias_init,
                (self.n, self.n),
                self.dtype,
            )
            return scale * x[..., None, None] + bias

        projection = self.param(
            "projection",
            self.projection_init,
            (self.input_dim, self.n * self.n),
            self.dtype,
        )
        bias = self.param(
            "bias",
            self.bias_init,
            (self.n * self.n,),
            self.dtype,
        )
        flat = jnp.matmul(x, projection) + bias
        return flat.reshape(*x.shape[:-1], self.n, self.n).astype(self.dtype)
