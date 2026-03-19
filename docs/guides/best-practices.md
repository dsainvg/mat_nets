# Best Practices

This guide provides best practices for building and training matrix neural networks with MatNet.

## Architecture Design

### Choosing Matrix Size

The matrix size (`n`) is a crucial hyperparameter that affects both model capacity and computational cost.

**Recommendations:**
- **Start small**: Begin with `n=4` or `n=8` for initial experiments
- **Scale up**: Increase to `n=16` or `n=32` if needed
- **Balance trade-offs**: Larger `n` = more parameters, smaller `n` = faster training

```python
# Good starting points
small_net = build_matrix_network(matrix_size=4, ...)   # Fast, few parameters
medium_net = build_matrix_network(matrix_size=8, ...)  # Balanced
large_net = build_matrix_network(matrix_size=16, ...)  # More expressive
```

**MatrixLayer parameter formula:**
```
kernel parameters     = output_dim * input_dim * n * n
bias parameters       = output_dim * n * n
total with bias       = (output_dim * input_dim + output_dim) * n * n
```

For a `1 -> 1` matrix layer, this reduces to:
```
n=4:  16 kernel params, 32 total with bias
n=8:  64 kernel params, 128 total with bias
n=16: 256 kernel params, 512 total with bias
n=32: 1024 kernel params, 2048 total with bias
```

### Network Depth

The number of hidden layers affects model capacity and training dynamics.

**Recommendations:**
- **Shallow**: 1-2 layers for simple problems
- **Medium**: 3-5 layers for moderate complexity
- **Deep**: 6+ layers for complex problems (with normalization)

```python
# Shallow network
shallow = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16],  # 1 layer
    output_dim=10
)

# Medium network
medium = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 16],  # 3 layers
    output_dim=10
)

# Deep network (use normalization!)
deep = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 64, 32, 16],  # 5 layers
    output_dim=10,
    use_normalization=True  # Important for deep networks
)
```

### Hidden Dimensions

Hidden dimensions control the width of each layer.

**Recommendations:**
- **Increasing**: `[16, 32, 64]` - gradually increase capacity
- **Decreasing**: `[64, 32, 16]` - bottleneck architecture
- **Symmetric**: `[32, 64, 32]` - hourglass architecture
- **Uniform**: `[32, 32, 32]` - consistent capacity

```python
# Increasing (good for feature extraction)
increasing = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 64],
    output_dim=10
)

# Symmetric (good for autoencoders)
symmetric = build_matrix_network(
    matrix_size=8,
    hidden_dims=[32, 64, 32],
    output_dim=10
)
```

## Activation Functions

### Choosing Activations

Different activation functions work better for different tasks.

**Recommendations:**
- **ReLU**: Default choice, works well most of the time
- **Leaky ReLU**: Prevents dying ReLU problem
- **Tanh**: Zero-centered output, good for RNNs
- **Sigmoid**: Output in (0, 1), good for gating
- **Swish/GELU**: Smooth activations, may improve accuracy

```python
from matnet import activations

# ReLU (default)
net = build_matrix_network(..., activation="relu")

# Leaky ReLU
net = build_matrix_network(..., activation="leaky_relu")

# Tanh
net = build_matrix_network(..., activation="tanh")

# Custom activation
net = MatrixNetwork(
    ...,
    activation=activations.matrix_swish
)
```

### Activation Placement

Place activations after matrix layers:

```python
# Good: Activation after matrix layer
x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = activations.matrix_relu(x)

# Bad: No activation (linear network)
x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = MatrixLayer(n=8, input_dim=32, output_dim=32)(x)  # Still linear!
```

## Normalization

### When to Use Normalization

**Always use normalization for:**
- Deep networks (4+ layers)
- Small batch sizes
- Unstable training

**Optional for:**
- Shallow networks
- Large batch sizes
- Stable datasets

```python
# With normalization (recommended)
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 64],
    output_dim=10,
    use_normalization=True  # Enable normalization
)

# Without normalization (faster, but less stable)
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16],
    output_dim=10,
    use_normalization=False
)
```

### Normalization Placement

Place normalization after activation:

```python
# Good: Norm after activation
x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = activations.matrix_relu(x)
x = MatrixLayerNorm()(x)  # After activation

# Alternative: Norm before activation (also works)
x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = MatrixLayerNorm()(x)  # Before activation
x = activations.matrix_relu(x)
```

## Initialization

### Weight Initialization

Good initialization is crucial for training stability.

**Recommendations:**
- **Lecun Normal**: Default, good for most cases
- **Xavier Uniform**: Alternative, similar performance
- **Orthogonal**: For very deep networks
- **Small Normal**: For specific architectures

```python
from flax import linen as nn
from matnet.layers.matrix_layer import MatrixLayer

# Lecun Normal (default)
layer = MatrixLayer(n=8, input_dim=16, output_dim=32)

# Xavier Uniform
layer = MatrixLayer(n=8, input_dim=16, output_dim=32, kernel_init=nn.initializers.xavier_uniform())

# Orthogonal
layer = MatrixLayer(n=8, input_dim=16, output_dim=32, kernel_init=nn.initializers.orthogonal())

# Small Normal
layer = MatrixLayer(
    n=8,
    input_dim=16,
    output_dim=32,
    kernel_init=nn.initializers.normal(stddev=0.01)
)
```

### Bias Initialization

```python
# Zero bias (default)
layer = MatrixLayer(n=8, input_dim=16, output_dim=32, bias_init=nn.initializers.zeros)

# Small constant bias
layer = MatrixLayer(n=8, input_dim=16, output_dim=32, bias_init=nn.initializers.constant(0.1))

# Normal bias
layer = MatrixLayer(n=8, input_dim=16, output_dim=32, bias_init=nn.initializers.normal(stddev=0.01))
```

## Training

### Learning Rate

**Recommendations:**
- **Start**: 1e-3 for Adam
- **Tune**: Use learning rate schedule
- **Monitor**: Watch for divergence

```python
import optax

# Constant learning rate
optimizer = optax.adam(learning_rate=1e-3)

# Learning rate schedule
schedule = optax.exponential_decay(
    init_value=1e-3,
    transition_steps=1000,
    decay_rate=0.9
)
optimizer = optax.adam(learning_rate=schedule)
```

### Batch Size

**Recommendations:**
- **Small**: 16-32 for quick iterations
- **Medium**: 64-128 for stable training
- **Large**: 256+ for optimal GPU utilization

```python
# Small batches for quick experiments
batch_size = 32

# Medium batches for stable training
batch_size = 128

# Large batches for production
batch_size = 256
```

### Training Duration

**Recommendations:**
- **Monitor loss**: Train until validation loss plateaus
- **Early stopping**: Stop if no improvement for 10-20 epochs
- **Maximum**: Set max epochs to prevent overfitting

```python
# Training loop with early stopping
best_loss = float('inf')
patience = 10
no_improve = 0

for epoch in range(100):  # Max 100 epochs
    # Train...
    
    # Validate
    val_loss = compute_validation_loss()
    
    # Early stopping
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
        best_params = params
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

## Regularization

### Weight Decay

```python
import optax

# Adam with weight decay
optimizer = optax.adamw(
    learning_rate=1e-3,
    weight_decay=1e-4
)
```

### Dropout

```python
from flax import linen as nn

class MatrixNetWithDropout(nn.Module):
    n: int = 8
    dropout_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, training: bool = True):
        x = InputScalingLayer(n=self.n, input_dim=20)(x)
        x = x[:, None, :, :]
        
        x = MatrixLayer(n=self.n, input_dim=1, output_dim=16)(x)
        x = activations.matrix_relu(x)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        
        x = MatrixLayer(n=self.n, input_dim=16, output_dim=1)(x)
        x = activations.matrix_relu(x)
        x = x[:, 0, :, :]
        
        x = DecompressionLayer(n=self.n, k=10)(x)
        return x
```

## Performance Optimization

### JIT Compilation

Always JIT compile your training and inference functions:

```python
@jax.jit
def train_step(params, opt_state, inputs, labels):
    # Your training code
    ...

@jax.jit
def inference_fn(params, inputs):
    # Your inference code
    ...
```

### Vectorization

Use `vmap` for batch processing:

```python
from matnet.utils.parallel import vmap_module

# Vectorize network forward pass
vmapped_net = vmap_module(net.apply)

# Process entire batch efficiently
outputs = vmapped_net(params, batch_inputs)
```

### Batch Size Optimization

Choose batch size based on your hardware:

```python
# For CPU
batch_size = 32

# For single GPU
batch_size = 128

# For multiple GPUs
batch_size = 256  # Split across devices
```

## Debugging

### Gradient Checking

```python
# Check gradient flow
def check_gradients(params, inputs, labels):
    grads = jax.grad(loss_fn)(params, inputs, labels)
    
    # Check for NaN or Inf
    for name, grad in grads['params'].items():
        if jnp.any(jnp.isnan(grad)):
            print(f"NaN in {name} gradients")
        if jnp.any(jnp.isinf(grad)):
            print(f"Inf in {name} gradients")
    
    # Check gradient magnitudes
    grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(x), grads)
    print(f"Gradient norms: {grad_norms}")
```

### Activation Monitoring

```python
# Monitor activation statistics
def forward_with_stats(params, inputs):
    # Manually implement forward pass
    x = InputScalingLayer(n=8, input_dim=20)(inputs)
    
    print(f"After input scaling: mean={x.mean():.3f}, std={x.std():.3f}")
    
    x = x[:, None, :, :]
    x = MatrixLayer(n=8, input_dim=1, output_dim=1)(x)
    print(f"After matrix layer: mean={x.mean():.3f}, std={x.std():.3f}")
    
    x = activations.matrix_relu(x)
    print(f"After activation: mean={x.mean():.3f}, std={x.std():.3f}")
    
    return x
```

### Parameter Counting

```python
def count_parameters(params):
    """Count total and per-layer parameters."""
    total = 0
    for name, param in params['params'].items():
        count = param.size
        total += count
        print(f"{name}: {count:,} parameters")
    print(f"Total: {total:,} parameters")
    return total
```

## Common Pitfalls

### 1. Forgetting Activations

```python
# Wrong: Linear network (no non-linearity)
x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = MatrixLayer(n=8, input_dim=32, output_dim=32)(x)  # Still linear!

# Correct: Add activations
x = MatrixLayer(n=8, input_dim=16, output_dim=32)(x)
x = activations.matrix_relu(x)
x = MatrixLayer(n=8, input_dim=32, output_dim=32)(x)
x = activations.matrix_relu(x)
```

### 2. Wrong Input Shapes

```python
# Wrong: Input must be (batch, features)
inputs = jnp.ones(20)  # Shape: (20,)

# Correct: Add batch dimension
inputs = jnp.ones((1, 20))  # Shape: (1, 20)
```

### 3. Not Using Normalization in Deep Networks

```python
# Wrong: Deep network without normalization
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 64, 128],  # 4 layers
    output_dim=10,
    use_normalization=False  # Will be unstable!
)

# Correct: Add normalization
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 64, 128],
    output_dim=10,
    use_normalization=True  # Stable training
)
```

### 4. Too Large Learning Rate

```python
# Wrong: Learning rate too high
optimizer = optax.adam(learning_rate=1e-1)  # Too high!

# Correct: Start smaller
optimizer = optax.adam(learning_rate=1e-3)  # Better
```

### 5. Not JIT Compiling

```python
# Wrong: No JIT compilation
def train_step(params, inputs, labels):
    loss = compute_loss(params, inputs, labels)
    grads = jax.grad(loss_fn)(params, inputs, labels)
    return params - 1e-3 * grads

# Correct: JIT compile
@jax.jit
def train_step(params, inputs, labels):
    loss = compute_loss(params, inputs, labels)
    grads = jax.grad(loss_fn)(params, inputs, labels)
    return params - 1e-3 * grads
```

## Summary Checklist

- [ ] Choose appropriate matrix size (start with 8)
- [ ] Design network depth based on problem complexity
- [ ] Select activation function (ReLU is good default)
- [ ] Enable normalization for deep networks
- [ ] Initialize with appropriate scheme
- [ ] Set learning rate (start with 1e-3)
- [ ] Choose batch size based on hardware
- [ ] JIT compile training functions
- [ ] Monitor gradients and activations
- [ ] Use early stopping to prevent overfitting