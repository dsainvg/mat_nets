from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from matnet.models.builder import SimpleMatrixNet
from matnet.utils.parallel import create_batched_forward, jit_module_forward, parallel_batch_process, vmap_module


def test_jit_module_forward_matches_apply() -> None:
    model = SimpleMatrixNet(matrix_size=4, hidden_dim=6, output_dim=3, input_dim=5)
    x = jnp.ones((7, 5), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x)
    jitted = jit_module_forward(model)
    expected = model.apply(params, x)
    actual = jitted(params, x)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_create_batched_forward_handles_partial_final_batch() -> None:
    model = SimpleMatrixNet(matrix_size=4, hidden_dim=6, output_dim=2, input_dim=5)
    x = jnp.ones((9, 5), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(1), x)
    forward = create_batched_forward(model, batch_size=4)
    y = forward(params, x)
    assert y.shape == (9, 2)


def test_parallel_batch_process_chunks_inputs() -> None:
    def fn(params, x):
        del params
        return x + 2.0

    processor = parallel_batch_process(fn, batch_size=3)
    x = jnp.arange(10, dtype=jnp.float32).reshape(10, 1)
    y = processor(None, x)
    np.testing.assert_allclose(y, x + 2.0)


def test_vmap_module_vectorizes_single_sample_function() -> None:
    def single_sample(params, x):
        del params
        return x * 3.0

    vmapped = vmap_module(single_sample, in_axes=(None, 0), out_axes=0)
    x = jnp.arange(6, dtype=jnp.float32)
    y = vmapped(None, x)
    np.testing.assert_allclose(y, x * 3.0)
