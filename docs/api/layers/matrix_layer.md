# MatrixLayer

The `matnet.layers.matrix_layer` module implements the core matrix-neuron layer used throughout MatNet.

## Overview

Each neuron is an `n x n` matrix, and each connection between an input neuron and an output neuron is also an `n x n` matrix.

For a layer with:
- `l` input neurons
- `i` output neurons
- matrix size `n`

the layer stores:
- `kernel` with shape `(i, l, n, n)`
- optional `bias` with shape `(i, n, n)`

This means:
- kernel parameters: `i * l * n * n`
- total parameters with bias: `(i * l + i) * n * n`

The forward pass is:

```python
output_j = sum_k(kernel[j, k] * input_k) + bias_j
```

where `*` is element-wise multiplication over the `n x n` matrices and the sum is over input neurons.

## MatrixLayer

```python
class MatrixLayer(nn.Module):
    n: int
    input_dim: int
    output_dim: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
```

**Attributes:**
- `n`: Matrix size for each neuron
- `input_dim`: Number of input neurons
- `output_dim`: Number of output neurons
- `kernel_init`: Initializer for the connection matrices
- `bias_init`: Initializer for the output-neuron bias matrices
- `use_bias`: Whether to add one `n x n` bias matrix per output neuron
- `dtype`: Parameter dtype

**Parameter shapes:**
- `kernel`: `(output_dim, input_dim, n, n)`
- `bias`: `(output_dim, n, n)` when `use_bias=True`

**Input and output shapes:**
- input: `(batch, input_dim, n, n)`
- output: `(batch, output_dim, n, n)`

**Example:**

```python
import jax
import jax.numpy as jnp
from matnet.layers.matrix_layer import MatrixLayer

layer = MatrixLayer(n=8, input_dim=3, output_dim=5)

rng = jax.random.PRNGKey(0)
inputs = jnp.ones((4, 3, 8, 8))
params = layer.init(rng, inputs)

outputs = layer.apply(params, inputs)
print(outputs.shape)  # (4, 5, 8, 8)
```

## BatchedMatrixLayer

```python
class BatchedMatrixLayer(nn.Module):
    n: int
    input_dim: int
    output_dim: int
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
```

`BatchedMatrixLayer` uses the same API and parameterization as `MatrixLayer`.

```python
from matnet.layers.matrix_layer import BatchedMatrixLayer

layer = BatchedMatrixLayer(n=8, input_dim=3, output_dim=5)
outputs = layer.apply(params, inputs)
```

## Mathematical Operation

The implementation uses:

```python
output = jnp.einsum('bijk,jikl->bjkl', x, kernel)
```

with:
- `x`: `(batch, input_dim, n, n)`
- `kernel`: `(output_dim, input_dim, n, n)`
- `output`: `(batch, output_dim, n, n)`

Expanded per output neuron:

```python
output[b, j, r, c] = sum_k(kernel[j, k, r, c] * x[b, k, r, c])
```

## Parameter Counting

For `l` inputs, `i` outputs, and matrix size `n`:

```python
kernel_params = i * l * n * n
bias_params = i * n * n
total_params = kernel_params + bias_params  # if use_bias=True
```

Example for `l=3`, `i=5`, `n=8`:

```python
kernel_params = 5 * 3 * 8 * 8   # 960
bias_params = 5 * 8 * 8         # 320
total_params = 1280
```

## Usage Patterns

### First hidden layer after input scaling

`InputScalingLayer` returns `(batch, n, n)`, so the builder adds a singleton neuron axis before the first matrix layer.

```python
x = InputScalingLayer(n=8, input_dim=20)(x)  # (batch, 8, 8)
x = x[:, None, :, :]                         # (batch, 1, 8, 8)
x = MatrixLayer(n=8, input_dim=1, output_dim=16)(x)
```

### Stacking matrix layers

```python
x = MatrixLayer(n=8, input_dim=1, output_dim=16)(x)
x = activations.matrix_relu(x)

x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = activations.matrix_relu(x)
```

### Custom initialization

```python
from flax import linen as nn

layer = MatrixLayer(
    n=8,
    input_dim=16,
    output_dim=32,
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=0.01),
)
```

## Comparison with Dense

```python
dense = nn.Dense(features=64)
# params: (in_features, 64) + (64,)

matrix = MatrixLayer(n=8, input_dim=16, output_dim=32)
# params: (32, 16, 8, 8) + (32, 8, 8)
```

The important distinction is that a MatrixLayer keeps an explicit neuron axis and each connection carries an `n x n` matrix, so parameter count scales with both neuron counts and `n^2`.

## Common Issues

### Missing neuron dimension

```python
# Wrong: missing input_dim axis
x = jnp.ones((4, 8, 8))
layer = MatrixLayer(n=8, input_dim=3, output_dim=5)

# Correct: include the neuron axis
x = jnp.ones((4, 3, 8, 8))
```

### Mismatched `input_dim`

```python
x = jnp.ones((4, 3, 8, 8))

# Wrong: declares 4 input neurons but data has 3
layer = MatrixLayer(n=8, input_dim=4, output_dim=5)

# Correct
layer = MatrixLayer(n=8, input_dim=3, output_dim=5)
```