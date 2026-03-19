"""JAX transformation helpers used by MatNet."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp


def vmap_module(module_fn: Callable[..., Any], in_axes: Any = 0, out_axes: Any = 0) -> Callable[..., Any]:
    return jax.vmap(module_fn, in_axes=in_axes, out_axes=out_axes)


def jit_module_forward(module: Any, static_argnums: tuple[int, ...] = ()) -> Callable[..., Any]:
    def forward(params: Any, *args: Any, **kwargs: Any) -> Any:
        return module.apply(params, *args, **kwargs)

    return jax.jit(forward, static_argnums=static_argnums)


def parallel_batch_process(
    module_fn: Callable[..., jax.Array],
    batch_size: int = 32,
    device_count: int | None = None,
) -> Callable[..., jax.Array]:
    del device_count

    def batched(params: Any, inputs: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
        outputs = []
        for start in range(0, inputs.shape[0], batch_size):
            stop = start + batch_size
            outputs.append(module_fn(params, inputs[start:stop], *args, **kwargs))
        return jnp.concatenate(outputs, axis=0)

    return batched


def create_batched_forward(module: Any, batch_size: int = 32) -> Callable[..., Any]:
    jitted_forward = jit_module_forward(module)

    def batched_forward(params: Any, inputs: jax.Array, *args: Any, **kwargs: Any) -> Any:
        outputs = []
        for start in range(0, inputs.shape[0], batch_size):
            stop = start + batch_size
            outputs.append(jitted_forward(params, inputs[start:stop], *args, **kwargs))
        return jnp.concatenate(outputs, axis=0)

    return batched_forward
