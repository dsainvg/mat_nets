# Normalization Module

The `matnet.normalization` module provides normalization layers that operate element-wise across matrix layers. These layers help stabilize training and improve convergence for matrix neural networks.

## Overview

Normalization layers in MatNet:
- Operate on tensors of shape `(..., n, n)`
- Normalize across specified axes while preserving matrix structure
- Support both training and inference modes
- Compatible with JAX's automatic differentiation

## Available Layers

### MatrixLayerNorm

```python
class MatrixLayerNorm(nn.Module):
    """Layer normalization for matrix inputs."""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    reduction_axes: Optional[Tuple[int, ...]] = None
```

Layer normalization for matrix inputs. Applies layer normalization element-wise across the n×n matrix. Normalizes over the last two dimensions (n, n) for each feature independently.

**Attributes:**
- `epsilon`: Small constant for numerical stability (default: 1e-6)
- `dtype`: Data type for computations (default: jnp.float32)
- `param_dtype`: Data type for parameters (default: jnp.float32)
- `use_bias`: Whether to use bias term (default: True)
- `use_scale`: Whether to use scale term (default: True)
- `reduction_axes`: Axes to reduce over for normalization (default: last two dimensions)

**Parameters:**
- `scale`: Learnable scale parameter of shape `(n, n)`
- `bias`: Learnable bias parameter of shape `(n, n)`

**Example:**
```python
from matnet.normalization import MatrixLayerNorm
import jax.numpy as jnp

# Create layer
norm = MatrixLayerNorm()

# Input: (batch, n, n)
x = jnp.random.normal(jax.random.PRNGKey(0), (32, 8, 8))

# Apply normalization
x_norm = norm(x)  # Shape: (32, 8, 8)
```

**Normalization Formula:**

```
mean = mean(x, axis=(-2, -1), keepdims=True)
var = var(x, axis=(-2, -1), keepdims=True)
x_norm = (x - mean) / sqrt(var + epsilon)
x_out = scale * x_norm + bias  # if use_scale and use_bias
```

### MatrixBatchNorm

```python
class MatrixBatchNorm(nn.Module):
    """Batch normalization for matrix inputs."""
    epsilon: float = 1e-6
    momentum: float = 0.99
    use_running_average: Optional[bool] = None
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    use_bias: bool = True
    use_scale: bool = True
    axis_name: Optional[str] = None
    axis_index_groups: Optional[Tuple[Tuple[int, ...], ...]] = None
```

Batch normalization for matrix inputs. Applies batch normalization over the batch dimension while normalizing element-wise across the n×n matrix.

**Attributes:**
- `epsilon`: Small constant for numerical stability (default: 1e-6)
- `momentum`: Momentum for running statistics (default: 0.99)
- `use_running_average`: Whether to use running averages (default: None)
- `dtype`: Data type for computations (default: jnp.float32)
- `param_dtype`: Data type for parameters (default: jnp.float32)
- `use_bias`: Whether to use bias term (default: True)
- `use_scale`: Whether to use scale term (default: True)
- `axis_name`: Name of axis for distributed computation (default: None)
- `axis_index_groups`: Groups for distributed computation (default: None)

**Parameters:**
- `scale`: Learnable scale parameter of shape `(n, n)`
- `bias`: Learnable bias parameter of shape `(n, n)`
- `mean`: Running mean (in `batch_stats` collection)
- `var`: Running variance (in `batch_stats` collection)

**Example:**
```python
from matnet.normalization import MatrixBatchNorm
import jax.numpy as jnp

# Create layer
batch_norm = MatrixBatchNorm()

# Input: (batch, n, n)
x = jnp.random.normal(jax.random.PRNGKey(0), (32, 8, 8))

# Training mode
x_norm_train = batch_norm(x, training=True)

# Inference mode (uses running statistics)
x_norm_eval = batch_norm(x, training=False)
```

**Normalization Formula (Training):**

```
mean = mean(x, axis=0)  # Over batch dimension
var = var(x, axis=0)    # Over batch dimension
x_norm = (x - mean) / sqrt(var + epsilon)
x_out = scale * x_norm + bias

# Update running statistics
running_mean = momentum * running_mean + (1 - momentum) * mean
running_var = momentum * running_var + (1 - momentum) * var
```

**Normalization Formula (Inference):**

```
x_norm = (x - running_mean) / sqrt(running_var + epsilon)
x_out = scale * x_norm + bias
```

## Usage Patterns

### In Model Definitions

```python
from flax import linen as nn
from matnet import normalization
from matnet import activations

class MyMatrixNet(nn.Module):
    n: int = 8
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        # Matrix layer
        x = nn.Dense(self.n * self.n)(x)
        x = x.reshape(-1, self.n, self.n)
        
        # Apply layer normalization
        x = normalization.MatrixLayerNorm()(x)
        
        # Apply activation
        x = activations.matrix_relu(x)
        
        # Apply batch normalization
        x = normalization.MatrixBatchNorm()(x, training=training)
        
        return x
```

### With Different Axes

```python
# Normalize over different axes
custom_norm = normalization.MatrixLayerNorm(
    reduction_axes=(-1,)  # Normalize over last dimension only
)

# Normalize over all dimensions
all_norm = normalization.MatrixLayerNorm(
    reduction_axes=(-3, -2, -1)  # For shape (batch, channels, n, n)
)
```

### Without Scale/Bias

```python
# Pure normalization without learned parameters
pure_norm = normalization.MatrixLayerNorm(
    use_scale=False,
    use_bias=False
)

# Only scale, no bias
scale_only = normalization.MatrixLayerNorm(
    use_bias=False
)
```

### Different Data Types

```python
# Use float16 for memory efficiency
fp16_norm = normalization.MatrixLayerNorm(
    dtype=jnp.float16,
    param_dtype=jnp.float32  # Keep parameters in float32 for stability
)
```

## Training vs Inference

### MatrixBatchNorm Behavior

```python
batch_norm = normalization.MatrixBatchNorm()

# Training mode: uses batch statistics, updates running averages
x_train = batch_norm(x, training=True)

# Inference mode: uses running statistics
x_eval = batch_norm(x, training=False)

# Or use the mutable collection flag
variables = {'params': params, 'batch_stats': batch_stats}
x_eval = batch_norm.apply(variables, x, mutable=False)
```

### MatrixLayerNorm Behavior

MatrixLayerNorm behaves the same in training and inference since it doesn't use running statistics:

```python
layer_norm = normalization.MatrixLayerNorm()

# Same behavior regardless of training flag
x_out = layer_norm(x)  # No training flag needed
```

## Implementation Details

### MatrixLayerNorm

- Normalizes over the matrix dimensions independently for each batch element
- Computes mean and variance over the specified reduction axes
- Applies element-wise scale and bias
- No running statistics (like standard LayerNorm)

### MatrixBatchNorm

- Normalizes over the batch dimension for each matrix element independently
- Maintains running mean and variance in `batch_stats` collection
- Updates running statistics during training
- Uses running statistics during inference
- Supports distributed training via `axis_name` and `axis_index_groups`

## Performance Considerations

- Both layers are implemented using JAX's efficient reduction operations
- Compatible with JIT compilation via `jax.jit`
- Vectorize automatically with `jax.vmap`
- MatrixBatchNorm has slightly higher overhead due to running statistics
- Use MatrixLayerNorm for simpler, faster normalization
- Use MatrixBatchNorm when batch statistics are beneficial

## Numerical Stability

- Both layers add `epsilon` to variance for numerical stability
- Default `epsilon=1e-6` works well for most cases
- Increase `epsilon` for very low precision (e.g., float16)
- Parameters are kept in higher precision by default (`param_dtype=jnp.float32`)

## Comparison with Standard Normalization

### MatrixLayerNorm vs LayerNorm

```python
# Standard LayerNorm (normalizes over feature dimension)
standard_ln = nn.LayerNorm()
x_out = standard_ln(x)  # x shape: (batch, features)

# Matrix LayerNorm (normalizes over matrix dimensions)
matrix_ln = normalization.MatrixLayerNorm()
x_out = matrix_ln(x)  # x shape: (batch, n, n)
```

### MatrixBatchNorm vs BatchNorm

```python
# Standard BatchNorm (normalizes over batch, per-channel)
standard_bn = nn.BatchNorm()
x_out = standard_bn(x, training=True)  # x shape: (batch, channels, height, width)

# Matrix BatchNorm (normalizes over batch, per-matrix-element)
matrix_bn = normalization.MatrixBatchNorm()
x_out = matrix_bn(x, training=True)  # x shape: (batch, n, n)
```

## Best Practices

1. **Use MatrixLayerNorm** for:
   - Smaller models where speed is important
   - When batch size varies significantly
   - For stable training without running statistics

2. **Use MatrixBatchNorm** for:
   - Larger models where batch statistics help
   - When batch size is consistent
   - For potentially better generalization

3. **General Tips**:
   - Place normalization after activation or before (experiment!)
   - Use `use_bias=True` and `use_scale=True` for full expressiveness
   - Keep `param_dtype=jnp.float32` even when using mixed precision
   - Set appropriate `epsilon` for your data type