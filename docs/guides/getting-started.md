# Getting Started Guide

This guide will walk you through installing MatNet and building your first matrix neural network.

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install Dependencies

```bash
# Clone or download the mat_nets repository
cd mat_nets

# Install dependencies
pip install -r requirements.txt
```

This will install:
- `jax` >= 0.4.0
- `jaxlib` >= 0.4.0
- `flax` >= 0.7.0
- `optax` >= 0.1.0
- `numpy` >= 1.21.0

### Verify Installation

```python
import jax
import jax.numpy as jnp
import flax
import optax

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"Flax version: {flax.__version__}")
print(f"Optax version: {optax.__version__}")

# Test JAX
x = jnp.array([1.0, 2.0, 3.0])
print(f"JAX test: {x * 2}")
```

## Your First Matrix Network

### Step 1: Import MatNet

```python
import jax
import jax.numpy as jnp
from matnet.models.builder import build_matrix_network
```

### Step 2: Create a Network

```python
# Build a simple matrix network
net = build_matrix_network(
    matrix_size=8,      # Use 8x8 matrices
    hidden_dims=[16],   # One hidden layer
    output_dim=10,      # 10 output classes
    activation="relu"   # ReLU activation
)

print(f"Network created with matrix size 8")
```

### Step 3: Initialize Parameters

```python
# Create random key for initialization
rng = jax.random.PRNGKey(0)

# Create dummy input for initialization
# Shape: (batch_size, input_features)
dummy_input = jnp.ones((1, 20))  # 1 sample, 20 features

# Initialize network parameters
params = net.init(rng, dummy_input)

print(f"Parameters initialized")
print(f"Total parameters: {sum(x.size for x in jax.tree_util.tree_leaves(params))}")
```

### Step 4: Forward Pass

```python
# Create batch of inputs
batch_inputs = jnp.ones((32, 20))  # 32 samples, 20 features each

# Forward pass
outputs = net.apply(params, batch_inputs)

print(f"Input shape: {batch_inputs.shape}")
print(f"Output shape: {outputs.shape}")
print(f"Output range: [{outputs.min():.3f}, {outputs.max():.3f}]")
```

### Complete Example

```python
import jax
import jax.numpy as jnp
from matnet.models.builder import build_matrix_network

# 1. Create network
print("Creating network...")
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10,
    activation="relu"
)

# 2. Initialize
print("\nInitializing parameters...")
rng = jax.random.PRNGKey(42)
dummy_input = jnp.ones((1, 20))
params = net.init(rng, dummy_input)

# 3. Forward pass
print("\nRunning forward pass...")
batch_inputs = jax.random.normal(rng, (64, 20))
outputs = net.apply(params, batch_inputs)

print(f"\nResults:")
print(f"  Input shape: {batch_inputs.shape}")
print(f"  Output shape: {outputs.shape}")
print(f"  Output mean: {outputs.mean():.3f}")
print(f"  Output std: {outputs.std():.3f}")
```

## Understanding the Architecture

### Data Flow

```
Input: (batch, 20)
    ↓
InputScalingLayer: (batch, 8, 8)
    ↓
MatrixLayer: (batch, 8, 8)
    ↓
Activation: (batch, 8, 8)
    ↓
MatrixLayer: (batch, 8, 8)
    ↓
Activation: (batch, 8, 8)
    ↓
DecompressionLayer: (batch, 10)
```

### Parameter Count

```python
# MatrixLayer parameter counting
def matrix_layer_params(input_dim, output_dim, n, use_bias=True):
    kernel = output_dim * input_dim * n * n
    bias = output_dim * n * n if use_bias else 0
    return kernel + bias

first_hidden = matrix_layer_params(input_dim=1, output_dim=16, n=8)
second_hidden = matrix_layer_params(input_dim=16, output_dim=32, n=8)

print(f"First hidden matrix layer: {first_hidden}")
print(f"Second hidden matrix layer: {second_hidden}")

# For matrix_size=8, hidden_dims=[16, 32]:
# First matrix layer: 16 * 1 * 64 + 16 * 64 = 2048
# Second matrix layer: 32 * 16 * 64 + 32 * 64 = 34816
```

## Training Your Network

### Step 1: Create Dummy Data

```python
# Create synthetic classification data
rng, data_rng = jax.random.split(rng)

# 100 samples, 20 features
X = jax.random.normal(data_rng, (100, 20))

# 100 labels, 10 classes
y = jax.random.randint(data_rng, (100,), 0, 10)
```

### Step 2: Define Loss Function

```python
def cross_entropy_loss(params, inputs, labels):
    # Forward pass
    logits = net.apply(params, inputs)
    
    # One-hot encode labels
    one_hot = jax.nn.one_hot(labels, 10)
    
    # Cross entropy loss
    loss = -jnp.mean(jax.nn.log_softmax(logits) * one_hot)
    
    return loss
```

### Step 3: Compute Gradients

```python
# Test gradient computation
sample_X = X[:32]  # 32 samples
sample_y = y[:32]

# Compute loss and gradients
loss, grads = jax.value_and_grad(cross_entropy_loss)(
    params, sample_X, sample_y
)

print(f"Loss: {loss:.3f}")
print(f"Gradient shapes: {jax.tree_map(lambda x: x.shape, grads)}")
```

### Step 4: Training Loop

```python
import optax

# Create optimizer
optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

# Training step
@jax.jit
def train_step(params, opt_state, inputs, labels):
    # Compute loss and gradients
    loss, grads = jax.value_and_grad(cross_entropy_loss)(
        params, inputs, labels
    )
    
    # Update parameters
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    return params, opt_state, loss

# Train for a few steps
print("\nTraining...")
for step in range(10):
    # Get batch
    batch_X = X[step*10:(step+1)*10]
    batch_y = y[step*10:(step+1)*10]
    
    # Training step
    params, opt_state, loss = train_step(
        params, opt_state, batch_X, batch_y
    )
    
    print(f"Step {step}: Loss = {loss:.3f}")
```

## Using Different Activations

```python
from matnet import activations

# Build with different activations
nets = {
    'relu': build_matrix_network(
        matrix_size=8, hidden_dims=[16], output_dim=10,
        activation="relu"
    ),
    'sigmoid': build_matrix_network(
        matrix_size=8, hidden_dims=[16], output_dim=10,
        activation="sigmoid"
    ),
    'tanh': build_matrix_network(
        matrix_size=8, hidden_dims=[16], output_dim=10,
        activation="tanh"
    )
}

# Test each
for name, net in nets.items():
    outputs = net.apply(params, batch_inputs)
    print(f"{name:8s}: mean={outputs.mean():6.3f}, std={outputs.std():6.3f}")
```

## Custom Network Architecture

### Using Layers Directly

```python
from flax import linen as nn
from matnet.layers.input_scaling import InputScalingLayer
from matnet.layers.matrix_layer import MatrixLayer
from matnet.layers.decompression import DecompressionLayer
from matnet import activations

class CustomMatrixNet(nn.Module):
    n: int = 8
    input_dim: int = 20
    output_dim: int = 10
    
    @nn.compact
    def __call__(self, x):
        # Input scaling
        x = InputScalingLayer(n=self.n, input_dim=self.input_dim)(x)
        x = x[:, None, :, :]
        
        # Matrix layer 1
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=8)(x)
        x = activations.matrix_relu(x)
        
        # Matrix layer 2
        x = MatrixLayer(n=self.n, input_dim=8, output_dim=1)(x)
        x = activations.matrix_relu(x)

        # Collapse the single output neuron before decompression
        x = x[:, 0, :, :]
        
        # Output decompression
        x = DecompressionLayer(n=self.n, k=self.output_dim)(x)
        
        return x

# Use custom network
net = CustomMatrixNet(n=8, input_dim=20, output_dim=10)
params = net.init(rng, dummy_input)
outputs = net.apply(params, batch_inputs)
```

## Performance Optimization

### JIT Compilation

```python
from matnet.utils.parallel import jit_module_forward

# JIT compile the network
jitted_net = jit_module_forward(net)

# Use compiled version
outputs = jitted_net(params, batch_inputs)

# Much faster after first compilation!
```

### Vectorization

```python
from matnet.utils.parallel import vmap_module

# Vectorize over batch
vmapped_net = vmap_module(net.apply)

# Process entire batch
outputs = vmapped_net(params, batch_inputs)
```

### Combined Optimization

```python
from matnet.utils.parallel import jit_module_forward, vmap_module

# Combine vmap and jit for maximum performance
optimized_net = jit_module_forward(vmap_module(net.apply))

# Fastest version
outputs = optimized_net(params, batch_inputs)
```

## Next Steps

- Read the [API Reference](../api/) for detailed documentation
- Check out [Examples](../examples/) for complete working code
- Learn about [Best Practices](./best-practices.md) for optimal results
- Explore [Advanced Topics](./advanced.md) for complex architectures

## Troubleshooting

### Import Errors

```python
# If you get import errors, check:
# 1. You're in the mat_nets directory
# 2. Requirements are installed
# 3. Python path includes the current directory

import sys
sys.path.append('.')
```

### Shape Errors

```python
# Common shape errors:

# Wrong input shape
# Error: Expected (batch, features), got (features,)
batch_inputs = jnp.ones(20)  # Wrong
batch_inputs = jnp.ones((32, 20))  # Correct

# Wrong matrix size
# Error: Expected n=8, got different size
net = build_matrix_network(matrix_size=8, ...)
inputs = jnp.ones((32, 16, 16))  # Wrong size
inputs = jnp.ones((32, 8, 8))  # Correct size
```

### Memory Errors

```python
# If you run out of memory:
# 1. Reduce batch size
batch_size = 32  # Instead of 128

# 2. Reduce matrix size
net = build_matrix_network(matrix_size=4, ...)  # Instead of 16

# 3. Use gradient accumulation
# Process smaller batches and accumulate gradients
```

## Getting Help

- Check the [API Reference](../api/) for detailed documentation
- Look at [Examples](../examples/) for working code
- Review [Tests](../../tests/) for usage patterns
- Open an issue on GitHub