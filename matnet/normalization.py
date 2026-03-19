"""Normalization modules for matrix inputs."""

from __future__ import annotations

from flax import linen as nn
import jax.numpy as jnp


class MatrixLayerNorm(nn.Module):
    """Layer normalization over the final matrix dimensions."""

    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        layer_norm = nn.LayerNorm(
            epsilon=self.epsilon,
            use_bias=self.use_bias,
            use_scale=self.use_scale,
            reduction_axes=(-2, -1),
            feature_axes=(-2, -1),
            dtype=self.dtype,
        )
        return layer_norm(x)


class MatrixBatchNorm(nn.Module):
    """Batch normalization adapted for matrix inputs."""

    epsilon: float = 1e-6
    momentum: float = 0.99
    use_running_average: bool | None = None
    dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray, training: bool = True) -> jnp.ndarray:
        use_running_average = self.use_running_average
        if use_running_average is None:
            use_running_average = not training

        matrix_shape = x.shape[-2:]
        mean = self.variable(
            "batch_stats",
            "mean",
            lambda: jnp.zeros(matrix_shape, dtype=self.dtype),
        )
        var = self.variable(
            "batch_stats",
            "var",
            lambda: jnp.ones(matrix_shape, dtype=self.dtype),
        )

        reduction_axes = tuple(range(x.ndim - 2))
        if use_running_average:
            batch_mean = mean.value
            batch_var = var.value
        else:
            batch_mean = jnp.mean(x, axis=reduction_axes, keepdims=False)
            batch_var = jnp.var(x, axis=reduction_axes, keepdims=False)
            mean.value = self.momentum * mean.value + (1.0 - self.momentum) * batch_mean
            var.value = self.momentum * var.value + (1.0 - self.momentum) * batch_var

        y = (x - batch_mean) / jnp.sqrt(batch_var + self.epsilon)

        if self.use_scale:
            scale = self.param("scale", nn.initializers.ones, matrix_shape, self.dtype)
            y = y * scale
        if self.use_bias:
            bias = self.param("bias", nn.initializers.zeros, matrix_shape, self.dtype)
            y = y + bias

        return y.astype(self.dtype)
