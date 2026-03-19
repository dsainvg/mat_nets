# Activations Module

The `matnet.activations` module provides activation functions that work element-wise on matrix inputs. All activations apply the standard activation function to each element of the n×n matrix independently.

## Overview

All activation functions in this module:
- Accept input tensors of shape `(..., n, n)`
- Apply activation element-wise to each matrix element
- Return tensors with the same shape as input
- Support JAX automatic differentiation

## Available Functions

### matrix_relu

```python
def matrix_relu(x: jnp.ndarray) -> jnp.ndarray
```

ReLU activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`

**Returns:**
- Activated tensor with same shape

**Example:**
```python
import jax.numpy as jnp
from matnet import activations

x = jnp.array([[[1.0, -2.0], [3.0, -4.0]]])
y = activations.matrix_relu(x)
# Result: [[[1.0, 0.0], [3.0, 0.0]]]
```

### matrix_leaky_relu

```python
def matrix_leaky_relu(x: jnp.ndarray, negative_slope: float = 0.01) -> jnp.ndarray
```

Leaky ReLU activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`
- `negative_slope`: Slope for negative values (default: 0.01)

**Returns:**
- Activated tensor with same shape

**Example:**
```python
x = jnp.array([[[1.0, -2.0], [3.0, -4.0]]])
y = activations.matrix_leaky_relu(x, negative_slope=0.1)
# Result: [[[1.0, -0.2], [3.0, -0.4]]]
```

### matrix_sigmoid

```python
def matrix_sigmoid(x: jnp.ndarray) -> jnp.ndarray
```

Sigmoid activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`

**Returns:**
- Activated tensor with same shape, values in (0, 1)

**Example:**
```python
x = jnp.array([[[0.0, 2.0], [-2.0, 0.0]]])
y = activations.matrix_sigmoid(x)
# Result: [[[0.5, 0.88], [0.12, 0.5]]]
```

### matrix_tanh

```python
def matrix_tanh(x: jnp.ndarray) -> jnp.ndarray
```

Tanh activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`

**Returns:**
- Activated tensor with same shape, values in (-1, 1)

### matrix_swish

```python
def matrix_swish(x: jnp.ndarray, beta: float = 1.0) -> jnp.ndarray
```

Swish activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`
- `beta`: Scaling parameter (default: 1.0)

**Returns:**
- Activated tensor with same shape

### matrix_gelu

```python
def matrix_gelu(x: jnp.ndarray, approximate: bool = True) -> jnp.ndarray
```

GELU activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`
- `approximate`: Whether to use approximate GELU (default: True)

**Returns:**
- Activated tensor with same shape

### matrix_elu

```python
def matrix_elu(x: jnp.ndarray, alpha: float = 1.0) -> jnp.ndarray
```

ELU activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`
- `alpha`: Scaling parameter for negative values (default: 1.0)

**Returns:**
- Activated tensor with same shape

### matrix_softplus

```python
def matrix_softplus(x: jnp.ndarray) -> jnp.ndarray
```

Softplus activation applied element-wise to matrix.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`

**Returns:**
- Activated tensor with same shape

### matrix_identity

```python
def matrix_identity(x: jnp.ndarray) -> jnp.ndarray
```

Identity activation (no-op) for matrices.

**Args:**
- `x`: Input tensor of shape `(..., n, n)`

**Returns:**
- Same tensor (no transformation)

**Use case:** Useful for skip connections or when you want to disable activation.

## Flax Module Wrapper

### MatrixActivation

```python
class MatrixActivation(nn.Module):
    """Flax module wrapper for matrix activations."""
    activation_fn: str = 'relu'
```

A Flax module that provides a convenient way to use matrix activations within Flax models.

**Attributes:**
- `activation_fn`: Name of activation function to use (default: 'relu')

**Available activation names:**
- `'relu'`, `'leaky_relu'`, `'sigmoid'`, `'tanh'`, `'swish'`, `'gelu'`, `'elu'`, `'softplus'`, `'identity'`

**Example:**
```python
from matnet.activations import MatrixActivation
import flax.linen as nn

class MyNetwork(nn.Module):
    def setup(self):
        self.activation = MatrixActivation('relu')
    
    def __call__(self, x):
        return self.activation(x)
```

## Usage Patterns

### Direct Function Calls

```python
from matnet import activations
import jax.numpy as jnp

# Create matrix input
x = jnp.random.normal(jax.random.PRNGKey(0), (32, 8, 8))

# Apply activation
x_relu = activations.matrix_relu(x)
x_sigmoid = activations.matrix_sigmoid(x)
x_tanh = activations.matrix_tanh(x)
```

### In Model Definitions

```python
from flax import linen as nn
from matnet import activations

class MyMatrixNet(nn.Module):
    n: int = 8
    
    @nn.compact
    def __call__(self, x):
        # Matrix layer
        x = nn.Dense(self.n * self.n)(x)
        x = x.reshape(-1, self.n, self.n)
        
        # Apply activation
        x = activations.matrix_relu(x)
        
        return x
```

### With Custom Parameters

```python
# Leaky ReLU with custom slope
x = activations.matrix_leaky_relu(x, negative_slope=0.2)

# Swish with custom beta
x = activations.matrix_swish(x, beta=1.5)

# ELU with custom alpha
x = activations.matrix_elu(x, alpha=0.5)
```

## Implementation Details

All activation functions:
- Use JAX's element-wise operations for efficiency
- Support automatic differentiation through `jax.grad`
- Work with arbitrary batch dimensions
- Preserve input shape exactly
- Are numerically stable

## Performance Considerations

- All activations are implemented using JAX's native operations
- No additional memory allocation beyond the output tensor
- Compatible with JIT compilation via `jax.jit`
- Vectorize automatically with `jax.vmap`

## Comparison with Standard Activations

The matrix activations behave identically to their scalar counterparts when applied element-wise:

```python
# Standard activation on scalar
scalar = jnp.array(2.0)
scalar_relu = jnp.maximum(0, scalar)  # 2.0

# Matrix activation on matrix element
matrix = jnp.array([[2.0]])
matrix_relu = activations.matrix_relu(matrix)  # [[2.0]]

# Both produce the same result for the same input value
```

The key difference is that matrix activations operate on entire matrices while preserving their structure.