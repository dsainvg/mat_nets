# Input Scaling Layer

The `matnet.layers.input_scaling` module provides layers that scale input vectors or scalars to n×n matrices using learnable parameters.

## Overview

Input scaling layers convert traditional vector inputs into matrix representations that can be processed by matrix neural networks. They support:
- Scalar inputs → n×n matrices
- Vector inputs → n×n matrices
- Learnable or fixed transformations

## InputScalingLayer

```python
class InputScalingLayer(nn.Module):
    """Scales input data to n×n matrices using learnable parameters."""
    n: int
    input_dim: Optional[int] = None
    scale_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32
```

Scales input data to n×n matrices using learnable parameters. This layer takes input data (vectors or scalars) and converts them to n×n matrices using a learned scaling transformation.

**Attributes:**
- `n`: Size of the output square matrices (n×n)
- `input_dim`: Dimension of input vector. If None, assumes scalar input
- `scale_init`: Initialization function for scaling parameters
- `bias_init`: Initialization function for bias parameters
- `dtype`: Data type for parameters

**Parameters:**
- `scale`: Scaling parameters
- `bias`: Bias parameters
- `projection`: Projection matrix (for vector inputs)

**Example:**
```python
from matnet.layers.input_scaling import InputScalingLayer
import jax.numpy as jnp

# For scalar inputs
scalar_layer = InputScalingLayer(n=8)

# For vector inputs
vector_layer = InputScalingLayer(n=8, input_dim=20)
```

## Scalar Input Mode

When `input_dim=None`, the layer treats inputs as scalars.

### Single Scalar

```python
# Create layer for scalar inputs
layer = InputScalingLayer(n=8)

# Input: single scalar
x = jnp.array(2.5)  # shape: ()

# Initialize
params = layer.init(rng, x)

# Forward pass
output = layer.apply(params, x)  # shape: (8, 8)

# Operation: output = scale * x + bias
# where scale and bias are (8, 8) matrices
```

### Batched Scalars

```python
# Input: batched scalars
x = jnp.array([1.0, 2.0, 3.0, 4.0])  # shape: (4,)

# Forward pass
output = layer.apply(params, x)  # shape: (4, 8, 8)

# Operation: output[i] = scale * x[i] + bias
```

### Parameters (Scalar Mode)

```python
# scale: (n, n) matrix
# bias: (n, n) matrix

params = {
    'params': {
        'scale': (8, 8),
        'bias': (8, 8)
    }
}
```

## Vector Input Mode

When `input_dim` is specified, the layer treats inputs as vectors.

### Single Vector

```python
# Create layer for vector inputs
layer = InputScalingLayer(n=8, input_dim=20)

# Input: single vector
x = jnp.ones(20)  # shape: (20,)

# Initialize
params = layer.init(rng, x)

# Forward pass
output = layer.apply(params, x)  # shape: (8, 8)

# Operation: output = reshape(projection @ x + bias)
```

### Batched Vectors

```python
# Input: batched vectors
x = jnp.ones((32, 20))  # shape: (batch, 20)

# Forward pass
output = layer.apply(params, x)  # shape: (32, 8, 8)
```

### Parameters (Vector Mode)

```python
# projection: (input_dim, n*n) matrix
# bias: (n*n,) vector

params = {
    'params': {
        'projection': (20, 64),  # 8*8 = 64
        'bias': (64,)
    }
}
```

### Mathematical Operation

```python
# For vector input x with shape (input_dim,)
# projection: (input_dim, n*n)
# bias: (n*n,)

flat_output = jnp.dot(x, projection) + bias  # shape: (n*n,)
output = flat_output.reshape(n, n)  # shape: (n, n)
```

## FixedInputScalingLayer

```python
class FixedInputScalingLayer:
    """Fixed (non-learnable) input scaling to n×n matrices."""
    def __init__(self, n: int, input_dim: Optional[int] = None):
        self.n = n
        self.input_dim = input_dim
```

Fixed (non-learnable) input scaling to n×n matrices. This version uses fixed transformations without learnable parameters, useful for baseline comparisons or when you want deterministic scaling.

**Attributes:**
- `n`: Size of the output square matrices (n×n)
- `input_dim`: Dimension of input vector. If None, assumes scalar input

**Example:**
```python
from matnet.layers.input_scaling import FixedInputScalingLayer

# Fixed scalar scaling
fixed_scalar = FixedInputScalingLayer(n=8)

# Fixed vector scaling
fixed_vector = FixedInputScalingLayer(n=8, input_dim=20)
```

## Fixed Scalar Mode

```python
# Create fixed scalar layer
layer = FixedInputScalingLayer(n=8)

# Single scalar
x = jnp.array(2.5)
output = layer(x)  # shape: (8, 8)
# Operation: tile scalar to all elements

# Batched scalars
x = jnp.array([1.0, 2.0, 3.0])
output = layer(x)  # shape: (3, 8, 8)
# Operation: x[i] * ones((8, 8))
```

## Fixed Vector Mode

```python
# Create fixed vector layer
layer = FixedInputScalingLayer(n=8, input_dim=20)

# Single vector
x = jnp.ones(20)
output = layer(x)  # shape: (8, 8)
# Operation: tile and reshape

# Batched vectors
x = jnp.ones((32, 20))
output = layer(x)  # shape: (32, 8, 8)
```

## Usage Patterns

### In Model Definitions

```python
from flax import linen as nn
from matnet.layers.input_scaling import InputScalingLayer
from matnet.layers.matrix_layer import MatrixLayer
from matnet import activations

class MatrixClassifier(nn.Module):
    n: int = 8
    input_dim: int = 20
    output_dim: int = 10
    
    @nn.compact
    def __call__(self, x):
        # Scale input to matrix
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        
        # Now x has shape (batch, n, n)
        x = x[:, None, :, :]
        
        # Apply matrix layers
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = x[:, 0, :, :]
        
        return x
```

### Learnable vs Fixed Scaling

```python
# Learnable scaling (recommended for most cases)
learnable = InputScalingLayer(n=8, input_dim=20)

# Fixed scaling (for baselines or deterministic behavior)
fixed = FixedInputScalingLayer(n=8, input_dim=20)

# Compare
x = jnp.ones((32, 20))

learnable_output = learnable.apply(learnable_params, x)
fixed_output = fixed(x)  # No parameters needed
```

### Custom Initialization

```python
from flax import linen as nn

# Custom initialization for scaling
layer = InputScalingLayer(
    n=8,
    input_dim=20,
    scale_init=nn.initializers.xavier_uniform(),
    bias_init=nn.initializers.normal(stddev=0.01)
)
```

## Comparison: Scalar vs Vector Mode

### Scalar Mode (input_dim=None)

**Pros:**
- Fewer parameters: 2 × n²
- Simple operation: scaling + bias
- Good for single-value features

**Cons:**
- Limited expressiveness
- Can't capture feature interactions

**Use when:**
- Input is a single scalar value
- You want simple scaling
- Memory is constrained

### Vector Mode (input_dim specified)

**Pros:**
- Can process vector inputs
- Learns input projections
- More expressive

**Cons:**
- More parameters: input_dim × n² + n²
- More computation

**Use when:**
- Input is a vector of features
- You need to preserve input information
- Model capacity is important

## Implementation Details

### Learnable Scalar Mode

```python
# Parameters:
scale: (n, n) matrix
bias: (n, n) matrix

# Forward pass:
output = scale * x + bias
# where x is broadcast to (n, n)
```

### Learnable Vector Mode

```python
# Parameters:
projection: (input_dim, n*n) matrix
bias: (n*n,) vector

# Forward pass:
flat = jnp.dot(x, projection) + bias  # shape: (n*n,)
output = flat.reshape(n, n)  # shape: (n, n)
```

### Fixed Scalar Mode

```python
# No parameters
# Forward pass:
output = jnp.full((n, n), x)  # Tile scalar to matrix
```

### Fixed Vector Mode

```python
# No parameters
# Forward pass:
# Tile vector elements to fill matrix
output = reshape_and_tile(x, (n, n))
```

## Performance Considerations

### Parameter Count

```python
# Scalar mode
params = 2 * n * n  # scale + bias

# Vector mode
params = input_dim * n * n + n * n  # projection + bias

# Example: n=8, input_dim=20
# Scalar: 2 * 8 * 8 = 128 parameters
# Vector: 20 * 8 * 8 + 8 * 8 = 1280 + 64 = 1344 parameters
```

### Computational Complexity

```python
# Scalar mode: O(n²)
# Vector mode: O(input_dim * n²)
```

### Memory Usage

```python
# Input: (batch, input_dim) or (batch,)
# Output: (batch, n, n)
# Parameters: depends on mode
```

## Best Practices

1. **Choose the Right Mode**:
   - Use scalar mode for single-value inputs
   - Use vector mode for feature vectors
   - Consider fixed scaling for baselines

2. **Matrix Size Selection**:
   - Start with small n (4-8) for initial experiments
   - Larger n (16-32) for more capacity
   - Balance parameters vs performance

3. **Initialization**:
   - Use `lecun_normal()` for most cases
   - Scale bias initialization based on input range
   - Consider zero bias for centered inputs

4. **Architecture Integration**:
   - Place InputScalingLayer at the beginning
   - Follow with MatrixLayer and activation
   - Consider normalization after activation

5. **Numerical Stability**:
   - Monitor output magnitudes
   - Use appropriate activation functions
   - Add normalization if outputs grow too large

## Common Patterns

### Classification Pipeline

```python
class MatrixClassifier(nn.Module):
    n: int = 8
    input_dim: int = 784  # e.g., 28x28 images flattened
    num_classes: int = 10
    
    @nn.compact
    def __call__(self, x, training=True):
        # Input: (batch, 784)
        
        # Scale to matrices
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        # x: (batch, n, n)
        x = x[:, None, :, :]
        
        # Matrix layers
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = normalization.MatrixLayerNorm()(x)
        x = x[:, 0, :, :]
        
        # Decompress to output
        x = DecompressionLayer(n=self.n, k=self.num_classes)(x)
        # x: (batch, num_classes)
        
        return x
```

### Regression Pipeline

```python
class MatrixRegressor(nn.Module):
    n: int = 8
    input_dim: int = 10
    output_dim: int = 1
    
    @nn.compact
    def __call__(self, x):
        # Scale input
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        x = x[:, None, :, :]
        
        # Process
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = x[:, 0, :, :]
        
        # Output
        x = DecompressionLayer(n=self.n, k=self.output_dim)(x)
        
        return x
```