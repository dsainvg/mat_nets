# Model Builder

The `matnet.models.builder` module provides a high-level API for easily constructing matrix neural networks with various architectures and configurations.

## Overview

The builder module offers:
- `MatrixNetwork`: Configurable multi-layer matrix network
- `SimpleMatrixNet`: Simple 2-layer network for quick prototyping
- `build_matrix_network()`: Factory function for building networks
- High-level configuration options

## MatrixNetwork

```python
class MatrixNetwork(nn.Module):
    """A configurable matrix neural network."""
    matrix_size: int
    hidden_dims: Sequence[int]
    output_dim: int
    activation: Callable = activations.matrix_relu
    use_bias: bool = True
    use_input_scaling: bool = True
    use_normalization: bool = True
    dtype: Any = jnp.float32
```

A configurable matrix neural network consisting of:
1. Input scaling layer (optional)
2. Multiple matrix layers with activations and normalization
3. Output decompression layer

Each matrix layer treats the hidden width as a neuron count. A layer with `l` input neurons and `i` output neurons has `i * l * matrix_size * matrix_size` kernel parameters, plus `i * matrix_size * matrix_size` bias parameters when bias is enabled.

**Attributes:**
- `matrix_size`: Size n for n×n matrices
- `hidden_dims`: List of hidden dimensions for each layer
- `output_dim`: Number of output features
- `activation`: Activation function to use (default: matrix_relu)
- `use_bias`: Whether to use bias in matrix layers (default: True)
- `use_input_scaling`: Whether to use input scaling (default: True)
- `use_normalization`: Whether to use layer normalization (default: True)
- `dtype`: Data type for computations (default: jnp.float32)

**Example:**
```python
from matnet.models.builder import MatrixNetwork

# Create network
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32, 16],
    output_dim=10,
    activation=activations.matrix_relu
)

# Initialize
rng = jax.random.PRNGKey(0)
inputs = jnp.ones((32, 20))  # (batch, input_features)
params = net.init(rng, inputs)

# Forward pass
outputs = net.apply(params, inputs)  # Shape: (32, 10)
```

## SimpleMatrixNet

```python
class SimpleMatrixNet(nn.Module):
    """A simple 2-layer matrix network for quick prototyping."""
    matrix_size: int = 8
    hidden_dim: int = 16
    output_dim: int = 1
    input_dim: int = 10
    activation: Callable = activations.matrix_relu
```

A simple 2-layer matrix network for quick prototyping. This network has a minimal architecture:
1. Input scaling
2. One matrix layer with activation
3. Output decompression

**Attributes:**
- `matrix_size`: Size n for n×n matrices (default: 8)
- `hidden_dim`: Hidden dimension for the middle layer (default: 16)
- `output_dim`: Number of output features (default: 1)
- `input_dim`: Input dimension (default: 10)
- `activation`: Activation function (default: matrix_relu)

**Example:**
```python
from matnet.models.builder import SimpleMatrixNet

# Create simple network
net = SimpleMatrixNet(
    matrix_size=8,
    hidden_dim=16,
    output_dim=1,
    input_dim=10
)

# Initialize
rng = jax.random.PRNGKey(0)
inputs = jnp.ones((32, 10))
params = net.init(rng, inputs)

# Forward pass
outputs = net.apply(params, inputs)  # Shape: (32, 1)
```

## build_matrix_network

```python
def build_matrix_network(
    matrix_size: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    activation: str = "relu",
    use_bias: bool = True,
    use_input_scaling: bool = True,
    use_normalization: bool = True,
    dtype: Any = jnp.float32
) -> MatrixNetwork
```

Build a matrix neural network with specified architecture.

**Args:**
- `matrix_size`: Size n for n×n matrices
- `hidden_dims`: List of hidden dimensions for each layer
- `output_dim`: Number of output features
- `activation`: Name of activation function ("relu", "sigmoid", "tanh", etc.) (default: "relu")
- `use_bias`: Whether to use bias in matrix layers (default: True)
- `use_input_scaling`: Whether to use input scaling (default: True)
- `use_normalization`: Whether to use layer normalization (default: True)
- `dtype`: Data type for computations (default: jnp.float32)

**Returns:**
- Configured MatrixNetwork instance

**Example:**
```python
from matnet.models.builder import build_matrix_network

# Build a 3-layer network with matrix size 8
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 16],
    output_dim=10,
    activation="relu"
)
```

## Architecture Details

### MatrixNetwork Architecture

```
Input: (batch, input_features)
    ↓
InputScalingLayer: (batch, matrix_size, matrix_size)
    ↓
Add neuron axis: (batch, 1, matrix_size, matrix_size)
    ↓
For each hidden_dim in hidden_dims:
    MatrixLayer: (batch, hidden_dim, matrix_size, matrix_size)
    Activation: (batch, hidden_dim, matrix_size, matrix_size)
    MatrixLayerNorm: (batch, hidden_dim, matrix_size, matrix_size) [if use_normalization]
    ↓
DecompressionLayer: (batch, output_dim)
```

### SimpleMatrixNet Architecture

```
Input: (batch, input_dim)
    ↓
InputScalingLayer: (batch, matrix_size, matrix_size)
    ↓
Add neuron axis: (batch, 1, matrix_size, matrix_size)
    ↓
MatrixLayer: (batch, hidden_dim, matrix_size, matrix_size)
    ↓
Activation: (batch, hidden_dim, matrix_size, matrix_size)
    ↓
DecompressionLayer: (batch, output_dim)
```

## Usage Patterns

### Basic Classification

```python
from matnet.models.builder import build_matrix_network
from matnet import activations

# Build classifier
classifier = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10,  # 10 classes
    activation="relu"
)

# Training
rng = jax.random.PRNGKey(0)
inputs = jnp.ones((32, 784))  # 32 images, 784 pixels each
labels = jnp.ones((32,), dtype=jnp.int32)  # Class labels

params = classifier.init(rng, inputs)

# Forward pass
logits = classifier.apply(params, inputs)
probabilities = jax.nn.softmax(logits, axis=-1)
```

### Regression

```python
# Build regressor
regressor = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 16],
    output_dim=1,  # Single output
    activation="relu"
)

# Training
inputs = jnp.ones((32, 10))  # 32 samples, 10 features
targets = jnp.ones((32, 1))  # Target values

params = regressor.init(rng, inputs)

# Forward pass
predictions = regressor.apply(params, inputs)
```

### Custom Activation

```python
from matnet import activations

# Use custom activation function
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10,
    activation=activations.matrix_swish,  # Swish activation
    use_normalization=True
)
```

### Without Input Scaling

```python
# If your inputs are already matrices
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10,
    use_input_scaling=False  # Skip input scaling
)

# Input must be matrices: (batch, matrix_size, matrix_size)
inputs = jnp.ones((32, 8, 8))
params = net.init(rng, inputs)
```

### Without Normalization

```python
# Simpler network without normalization
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10,
    use_normalization=False  # Skip normalization
)
```

### Deep Networks

```python
# Build deep network
deep_net = build_matrix_network(
    matrix_size=16,  # Larger matrices for deeper network
    hidden_dims=[32, 64, 128, 64, 32],  # 5 hidden layers
    output_dim=10,
    activation="relu",
    use_normalization=True  # Important for deep networks
)
```

### Quick Prototyping

```python
from matnet.models.builder import SimpleMatrixNet

# Quick prototype
net = SimpleMatrixNet(
    matrix_size=4,  # Small for speed
    hidden_dim=8,
    output_dim=1,
    input_dim=10
)

# Minimal parameters, fast training
```

## Configuration Options

### Activation Functions

```python
# Available activation names for build_matrix_network:
activation_map = {
    "relu": activations.matrix_relu,
    "sigmoid": activations.matrix_sigmoid,
    "tanh": activations.matrix_tanh,
    "leaky_relu": activations.matrix_leaky_relu,
    # Default: matrix_relu
}

# Or pass activation function directly to MatrixNetwork
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16],
    output_dim=10,
    activation=activations.matrix_swish  # Custom activation
)
```

### Architecture Variations

```python
# Minimal network
minimal = MatrixNetwork(
    matrix_size=4,
    hidden_dims=[8],
    output_dim=2,
    use_bias=False,
    use_input_scaling=False,
    use_normalization=False
)

# Full-featured network
full = MatrixNetwork(
    matrix_size=16,
    hidden_dims=[32, 64, 32],
    output_dim=10,
    use_bias=True,
    use_input_scaling=True,
    use_normalization=True
)
```

## Training Integration

### With Optax

```python
import optax

# Create network
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10
)

# Initialize
rng = jax.random.PRNGKey(0)
inputs = jnp.ones((32, 20))
params = net.init(rng, inputs)

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(params, opt_state, inputs, labels):
    def loss_fn(params):
        logits = net.apply(params, inputs)
        loss = -jnp.mean(jax.nn.log_softmax(logits) * jax.nn.one_hot(labels, 10))
        return loss
    
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss
```

### With Custom Loss

```python
# Regression with MSE loss
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=1
)

def mse_loss(params, inputs, targets):
    predictions = net.apply(params, inputs)
    return jnp.mean((predictions - targets) ** 2)

# Compute gradients
grads = jax.grad(mse_loss)(params, inputs, targets)
```

## Comparison: MatrixNetwork vs SimpleMatrixNet

### MatrixNetwork

**Pros:**
- Flexible architecture
- Multiple hidden layers
- Configurable normalization
- Suitable for complex tasks

**Cons:**
- More parameters
- More complex configuration
- May be overkill for simple tasks

**Use for:**
- Complex classification/regression
- Deep networks
- Production models

### SimpleMatrixNet

**Pros:**
- Simple and fast
- Minimal parameters
- Easy to use
- Good for prototyping

**Cons:**
- Only 2 layers
- Limited capacity
- No normalization

**Use for:**
- Quick experiments
- Simple datasets
- Baseline models
- Prototyping

## Best Practices

1. **Start Simple**:
   - Begin with `SimpleMatrixNet` for prototyping
   - Use small `matrix_size` (4-8) initially
   - Gradually increase complexity

2. **Architecture Design**:
   - Use `build_matrix_network()` for most cases
   - Enable normalization for deep networks
   - Match `output_dim` to your task

3. **Hyperparameters**:
   - Start with `matrix_size=8`
   - Use hidden_dims like `[16, 32, 16]`
   - Adjust based on performance

4. **Training**:
   - Use appropriate activation for your task
   - Add normalization for stability
   - Monitor gradient flow

5. **Performance**:
   - Use `SimpleMatrixNet` for speed
   - Consider matrix size trade-offs
   - JIT compile for production