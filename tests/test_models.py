from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
from flax.core import freeze, unfreeze

from matnet.layers.matrix_layer import MatrixLayer
from matnet.models.builder import MatrixNetwork, SimpleMatrixNet, build_matrix_network


def _count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def _expected_matrix_network_param_count(
    *,
    input_dim: int,
    matrix_size: int,
    hidden_dims: tuple[int, ...],
    output_dim: int,
    use_bias: bool = True,
    use_input_scaling: bool = True,
    use_normalization: bool = True,
) -> int:
    n2 = matrix_size * matrix_size
    total = 0

    if use_input_scaling:
        total += input_dim * n2
        total += n2
        current_dim = 1
    else:
        current_dim = 1

    for hidden_dim in hidden_dims:
        total += hidden_dim * current_dim * n2
        if use_bias:
            total += hidden_dim * n2
        if use_normalization:
            total += 2 * n2
        current_dim = hidden_dim

    total += current_dim * n2 * output_dim
    total += output_dim
    return total


def test_matrix_network_forward_shape() -> None:
    model = MatrixNetwork(matrix_size=3, hidden_dims=(4, 2), output_dim=6)
    x = jnp.ones((5, 7), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(0), x)
    y = model.apply(params, x)
    assert y.shape == (5, 6)


def test_matrix_network_without_input_scaling_accepts_matrix_input() -> None:
    model = MatrixNetwork(
        matrix_size=3,
        hidden_dims=(2,),
        output_dim=4,
        use_input_scaling=False,
    )
    x = jnp.ones((5, 3, 3), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(1), x)
    y = model.apply(params, x)
    assert y.shape == (5, 4)


def test_simple_matrix_net_forward_shape() -> None:
    model = SimpleMatrixNet(matrix_size=4, hidden_dim=3, output_dim=2, input_dim=5)
    x = jnp.ones((8, 5), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(2), x)
    y = model.apply(params, x)
    assert y.shape == (8, 2)


def test_build_matrix_network_returns_module() -> None:
    model = build_matrix_network(matrix_size=4, hidden_dims=(3, 2), output_dim=1)
    assert isinstance(model, MatrixNetwork)


def test_matrix_layer_n_one_matches_dense_layer() -> None:
    matrix_layer = MatrixLayer(n=1, input_dim=3, output_dim=2, use_bias=True)
    dense_layer = nn.Dense(features=2, use_bias=True)

    scalar_x = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=jnp.float32)
    matrix_x = scalar_x[..., None, None]

    matrix_params = matrix_layer.init(jax.random.PRNGKey(3), matrix_x)
    dense_params = dense_layer.init(jax.random.PRNGKey(4), scalar_x)

    matrix_params_mut = unfreeze(matrix_params)
    dense_params_mut = unfreeze(dense_params)

    kernel = jnp.array(
        [
            [[[2.0]], [[3.0]], [[4.0]]],
            [[[5.0]], [[6.0]], [[7.0]]],
        ],
        dtype=jnp.float32,
    )
    bias = jnp.array([[[8.0]], [[9.0]]], dtype=jnp.float32)

    matrix_params_mut["params"]["kernel"] = kernel
    matrix_params_mut["params"]["bias"] = bias
    dense_params_mut["params"]["kernel"] = jnp.array(
        [
            [2.0, 5.0],
            [3.0, 6.0],
            [4.0, 7.0],
        ],
        dtype=jnp.float32,
    )
    dense_params_mut["params"]["bias"] = jnp.array([8.0, 9.0], dtype=jnp.float32)

    matrix_params = freeze(matrix_params_mut)
    dense_params = freeze(dense_params_mut)

    matrix_y = matrix_layer.apply(matrix_params, matrix_x)[..., 0, 0]
    dense_y = dense_layer.apply(dense_params, scalar_x)

    np.testing.assert_allclose(matrix_y, dense_y)


def test_same_model_surface_runs_for_n_one_and_n_greater_than_one() -> None:
    batch = 4
    feature_dim = 5
    output_dim = 3
    x = jnp.ones((batch, feature_dim), dtype=jnp.float32)

    model_n1 = SimpleMatrixNet(matrix_size=1, hidden_dim=4, output_dim=output_dim, input_dim=feature_dim)
    params_n1 = model_n1.init(jax.random.PRNGKey(5), x)
    y_n1 = model_n1.apply(params_n1, x)

    model_n3 = SimpleMatrixNet(matrix_size=3, hidden_dim=4, output_dim=output_dim, input_dim=feature_dim)
    params_n3 = model_n3.init(jax.random.PRNGKey(6), x)
    y_n3 = model_n3.apply(params_n3, x)

    assert y_n1.shape == (batch, output_dim)
    assert y_n3.shape == (batch, output_dim)


def test_large_four_layer_model_param_count_for_n_one_is_exact() -> None:
    input_dim = 12
    hidden_dims = (20, 24, 28, 22)
    output_dim = 7
    matrix_size = 1

    model = MatrixNetwork(
        matrix_size=matrix_size,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_bias=True,
        use_input_scaling=True,
        use_normalization=False,
    )
    x = jnp.ones((3, input_dim), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(7), x)

    actual = _count_params(params)
    expected = _expected_matrix_network_param_count(
        input_dim=input_dim,
        matrix_size=matrix_size,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_bias=True,
        use_input_scaling=True,
        use_normalization=False,
    )

    assert actual == expected


def test_large_three_layer_model_param_count_for_n_three_is_exact() -> None:
    input_dim = 10
    hidden_dims = (20, 24, 30)
    output_dim = 6
    matrix_size = 3

    model = MatrixNetwork(
        matrix_size=matrix_size,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_bias=True,
        use_input_scaling=True,
        use_normalization=True,
    )
    x = jnp.ones((2, input_dim), dtype=jnp.float32)
    params = model.init(jax.random.PRNGKey(8), x)

    actual = _count_params(params)
    expected = _expected_matrix_network_param_count(
        input_dim=input_dim,
        matrix_size=matrix_size,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        use_bias=True,
        use_input_scaling=True,
        use_normalization=True,
    )

    assert actual == expected


def test_large_models_with_20_to_30_neurons_run_for_multiple_n() -> None:
    input_dim = 11
    hidden_dims = (20, 26, 30, 24)
    output_dim = 5
    x = jnp.ones((4, input_dim), dtype=jnp.float32)

    model_n1 = MatrixNetwork(matrix_size=1, hidden_dims=hidden_dims, output_dim=output_dim, use_normalization=False)
    params_n1 = model_n1.init(jax.random.PRNGKey(9), x)
    y_n1 = model_n1.apply(params_n1, x)

    model_n2 = MatrixNetwork(matrix_size=2, hidden_dims=hidden_dims, output_dim=output_dim, use_normalization=True)
    params_n2 = model_n2.init(jax.random.PRNGKey(10), x)
    y_n2 = model_n2.apply(params_n2, x)

    assert y_n1.shape == (4, output_dim)
    assert y_n2.shape == (4, output_dim)
    assert _count_params(params_n2) > _count_params(params_n1)
