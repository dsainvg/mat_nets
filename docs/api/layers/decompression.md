# Decompression Layer

The `matnet.layers.decompression` module provides layers that convert n×n matrix representations back to k-dimensional output vectors.

## Overview

Decompression layers are the counterpart to input scaling layers. They convert the internal matrix representations used by matrix neural networks back to vector outputs for tasks like classification or regression.

## DecompressionLayer

```python
class DecompressionLayer(nn.Module):
    """Converts n×n matrices to k-dimensional output vectors."""
    n: int
    k: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32
```

Converts n×n matrices to k-dimensional output vectors through a learned linear transformation.

**Attributes:**
- `n`: Size of input square matrices (n×n)
- `k`: Number of output dimensions
- `use_bias`: Whether to use bias term (default: True)
- `kernel_init`: Initialization function for weight matrix (default: lecun_normal)
- `bias_init`: Initialization function for bias vector (default: zeros)
- `dtype`: Data type for parameters (default: jnp.float32)

**Parameters:**
- `kernel`: Weight matrix of shape `(n*n, k)`
- `bias`: Bias vector of shape `(k,)` (if `use_bias=True`)

**Example:**
```python
from matnet.layers.decompression import DecompressionLayer
import jax.numpy as jnp

# Create layer
layer = DecompressionLayer(n=8, k=10)

# Input: (batch, n, n)
x = jnp.ones((32, 8, 8))

# Initialize
params = layer.init(rng, x)

# Forward pass
output = layer.apply(params, x)  # Shape: (32, 10)
```

## Mathematical Operation

### Forward Pass

```python
# Flatten input matrix
flat_x = x.reshape(*x.shape[:-2], n * n)  # Shape: (..., n*n)

# Linear projection
output = jnp.dot(flat_x, kernel)  # Shape: (..., k)

# Add bias (if enabled)
if use_bias:
    output = output + bias  # Shape: (..., k)
```

### Parameter Shapes

```python
# kernel: (n*n, k)
# bias: (k,)
# input: (..., n, n)
# output: (..., k)
```

## GlobalPoolingDecompressionLayer

```python
class GlobalPoolingDecompressionLayer(nn.Module):
    """Decompresses n×n matrices using global pooling operations."""
    n: int
    k: int
    pool_mode: str = 'mean'
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32
```

Decompresses n×n matrices using global pooling operations (mean, max, or sum) before projecting to k dimensions. This can be more parameter-efficient than full linear projection.

**Attributes:**
- `n`: Size of input square matrices (n×n)
- `k`: Number of output dimensions
- `pool_mode`: Pooling mode ('mean', 'max', or 'sum') (default: 'mean')
- `use_bias`: Whether to use bias term (default: True)
- `kernel_init`: Initialization function for weight matrix (default: lecun_normal)
- `bias_init`: Initialization function for bias vector (default: zeros)
- `dtype`: Data type for parameters (default: jnp.float32)

**Parameters:**
- `kernel`: Weight matrix of shape `(1, k)`
- `bias`: Bias vector of shape `(k,)` (if `use_bias=True`)

**Example:**
```python
from matnet.layers.decompression import GlobalPoolingDecompressionLayer

# Create layer with mean pooling
layer = GlobalPoolingDecompressionLayer(n=8, k=10, pool_mode='mean')

# Input: (batch, n, n)
x = jnp.ones((32, 8, 8))

# Forward pass
output = layer.apply(params, x)  # Shape: (32, 10)
```

### Pooling Modes

#### Mean Pooling

```python
layer = GlobalPoolingDecompressionLayer(n=8, k=10, pool_mode='mean')

# Operation:
pooled = jnp.mean(x, axis=(-2, -1))  # Shape: (batch,)
pooled = pooled[..., None]  # Shape: (batch, 1)
output = jnp.dot(pooled, kernel)  # Shape: (batch, k)
```

#### Max Pooling

```python
layer = GlobalPoolingDecompressionLayer(n=8, k=10, pool_mode='max')

# Operation:
pooled = jnp.max(x, axis=(-2, -1))  # Shape: (batch,)
```

#### Sum Pooling

```python
layer = GlobalPoolingDecompressionLayer(n=8, k=10, pool_mode='sum')

# Operation:
pooled = jnp.sum(x, axis=(-2, -1))  # Shape: (batch,)
```

## Usage Patterns

### In Model Definitions

```python
from flax import linen as nn
from matnet.layers.input_scaling import InputScalingLayer
from matnet.layers.matrix_layer import MatrixLayer
from matnet.layers.decompression import DecompressionLayer
from matnet import activations

class MatrixClassifier(nn.Module):
    n: int = 8
    input_dim: int = 20
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x):
        # Input: (batch, input_dim)
        
        # Scale to matrices
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        # x: (batch, n, n)
        x = x[:, None, :, :]
        
        # Process with matrix layers
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = x[:, 0, :, :]
        
        # Decompress to output
        x = DecompressionLayer(n=self.n, k=self.num_classes)(x)
        # x: (batch, num_classes)
        
        return x
```

### Classification Output

```python
# For classification with 10 classes
output_layer = DecompressionLayer(n=8, k=10)

# Output shape: (batch, 10)
# Apply softmax for probabilities
logits = output_layer.apply(params, x)
probabilities = jax.nn.softmax(logits, axis=-1)
```

### Regression Output

```python
# For regression with single output
output_layer = DecompressionLayer(n=8, k=1)

# Output shape: (batch, 1)
prediction = output_layer.apply(params, x)
```

### Multi-Task Output

```python
# For multiple regression targets
output_layer = DecompressionLayer(n=8, k=5)

# Output shape: (batch, 5)
predictions = output_layer.apply(params, x)
```

### Parameter-Efficient Output

```python
# Use pooling for fewer parameters
# Full linear: (n*n, k) parameters
full_layer = DecompressionLayer(n=8, k=10)
# Parameters: 64 * 10 = 640

# Pooling: (1, k) parameters
pool_layer = GlobalPoolingDecompressionLayer(n=8, k=10, pool_mode='mean')
# Parameters: 1 * 10 = 10
```

## Parameter Count Comparison

```python
# DecompressionLayer
params = n * n * k + k  # kernel + bias

# GlobalPoolingDecompressionLayer
params = 1 * k + k  # kernel + bias

# Example: n=8, k=10
# DecompressionLayer: 64 * 10 + 10 = 650 parameters
# GlobalPoolingDecompressionLayer: 1 * 10 + 10 = 20 parameters
```

## Custom Initialization

```python
from flax import linen as nn

# Custom initialization
layer = DecompressionLayer(
    n=8,
    k=10,
    kernel_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=0.01)
)
```

## Implementation Details

### DecompressionLayer

```python
# Flatten: (..., n, n) -> (..., n*n)
# Project: (..., n*n) @ (n*n, k) -> (..., k)
# Add bias: (..., k) + (k,) -> (..., k)
```

### GlobalPoolingDecompressionLayer

```python
# Pool: (..., n, n) -> (...,)
# Add feature dim: (...,) -> (..., 1)
# Project: (..., 1) @ (1, k) -> (..., k)
# Add bias: (..., k) + (k,) -> (..., k)
```

## Performance Considerations

### Computational Complexity

```python
# DecompressionLayer
# Time: O(... * n*n * k)
# Memory: O(... * k)

# GlobalPoolingDecompressionLayer
# Time: O(... * n*n) for pooling + O(... * k) for projection
# Memory: O(... * k)
```

### When to Use Each

**DecompressionLayer:**
- Full expressiveness needed
- Output dimension k is large
- Matrix structure is important for output

**GlobalPoolingDecompressionLayer:**
- Parameter efficiency is important
- Output dimension k is small
- Global aggregation is sufficient
- Memory constrained

## Best Practices

1. **Output Dimension**:
   - Match k to your task (e.g., num_classes for classification)
   - Consider parameter count trade-offs
   - Use pooling for efficiency when appropriate

2. **Architecture Placement**:
   - Place decompression layer at the end
   - After all matrix processing is complete
   - Before final activation (e.g., softmax)

3. **Initialization**:
   - Use `lecun_normal()` or `xavier_uniform()` for most cases
   - Initialize bias to zeros or small values
   - Consider output scale requirements

4. **Performance**:
   - Use `GlobalPoolingDecompressionLayer` for efficiency
   - Consider parameter count vs accuracy trade-off
   - JIT compile for production

5. **Numerical Stability**:
   - Monitor output magnitudes
   - Use appropriate activations before decompression
   - Consider layer normalization before decompression

## Common Patterns

### Complete Classification Network

```python
class CompleteClassifier(nn.Module):
    n: int = 8
    input_dim: int = 784
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input scaling
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        x = x[:, None, :, :]
        
        # Matrix layers with normalization
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=16)(x)
        x = activations.matrix_relu(x)
        x = normalization.MatrixLayerNorm()(x)
        
        x = MatrixLayer(n=self.n, input_dim=16, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = normalization.MatrixLayerNorm()(x)
        x = x[:, 0, :, :]
        
        # Output decompression
        x = DecompressionLayer(n=self.n, k=self.num_classes)(x)
        
        return x
```

### Efficient Network with Pooling

```python
class EfficientClassifier(nn.Module):
    n: int = 8
    input_dim: int = 20
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x):
        # Input scaling
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        x = x[:, None, :, :]
        
        # Matrix layers
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = x[:, 0, :, :]
        
        # Pooling decompression (parameter-efficient)
        x = GlobalPoolingDecompressionLayer(
            n=self.n, k=self.num_classes, pool_mode='mean'
        )(x)
        
        return x
```