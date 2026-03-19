# MatNet

MatNet is a JAX/Flax library for matrix neural networks where neurons, weights, and biases are all `n x n` matrices.

The implementation in this repository is rebuilt from `DOCS_SUMMARY.md` only.

## Included Modules

- `matnet/activations.py`: element-wise matrix activations
- `matnet/normalization.py`: matrix layer norm and batch norm
- `matnet/layers/`: `MatrixLayer`, `InputScaling`, `DecompressionLayer`
- `matnet/models/builder.py`: `MatrixNetwork`, `SimpleMatrixNet`, `build_matrix_network`
- `matnet/utils/parallel.py`: simple `vmap`, `jit`, and batched-forward helpers

## Core Layer

For each output matrix neuron `j`, MatNet computes:

`Y_j = sum_i (W_ji ? X_i) + B_j`

where `?` is element-wise multiplication and every `W_ji`, `X_i`, and `B_j` is an `n x n` matrix.

## Quick Start

```python
import jax
import jax.numpy as jnp
from matnet.models.builder import SimpleMatrixNet

net = SimpleMatrixNet(matrix_size=8, hidden_dim=16, output_dim=10, input_dim=20)
rng = jax.random.PRNGKey(0)
inputs = jnp.ones((32, 20))
params = net.init(rng, inputs)
outputs = net.apply(params, inputs)
```
