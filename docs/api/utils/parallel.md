# Parallelization Utilities

The `matnet.utils.parallel` module provides utilities for efficiently processing batches of inputs through matrix neural networks using JAX's parallelization primitives.

## Overview

This module offers:
- `vmap_module()`: Vectorize module forward passes
- `jit_module_forward()`: JIT compile module forward passes
- `parallel_batch_process()`: Automatic batch parallelization
- `create_batched_forward()`: Efficient batched processing

## vmap_module

```python
def vmap_module(module_fn: Callable, in_axes: Any = 0, out_axes: Any = 0) -> Callable
```

Vectorize a module's forward pass over a batch dimension. This wrapper applies `jax.vmap` to a module's `__call__` method, enabling efficient batch processing. The module parameters are shared across all batch elements.

**Args:**
- `module_fn`: The module's `__call__` method or a function that calls the module
- `in_axes`: Which axes of inputs to map over (default: 0 for batch dimension)
- `out_axes`: Where to place the mapped axis in outputs (default: 0)

**Returns:**
- A vectorized version of the module function

**Example:**
```python
from matnet.utils.parallel import vmap_module
from matnet.layers.matrix_layer import MatrixLayer

# Create layer
layer = MatrixLayer(n=8, input_dim=3, output_dim=5)

def single_forward(params, x):
    return layer.apply(params, x[None, ...])[0]

# Vectorize
vmapped_layer = vmap_module(single_forward)

# Process batch
batch_inputs = jnp.ones((32, 3, 8, 8))  # (batch, input_dim, n, n)
params = layer.init(rng, batch_inputs[:1])

# Forward pass on entire batch
outputs = vmapped_layer(params, batch_inputs)  # Shape: (32, 5, 8, 8)
```

## jit_module_forward

```python
def jit_module_forward(module: nn.Module, static_argnums: tuple = ()) -> Callable
```

JIT compile a module's forward pass for better performance. This function creates a jitted version of a module's forward pass. The compilation is cached and reused for inputs with the same shape.

**Args:**
- `module`: The Flax module to compile
- `static_argnums`: Arguments to treat as static (e.g., training mode flag)

**Returns:**
- A JIT-compiled version of the module's `__call__` method

**Example:**
```python
from matnet.utils.parallel import jit_module_forward
from matnet.models.builder import MatrixNetwork

# Create network
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10
)

# JIT compile
jitted_net = jit_module_forward(net)

# Use in forward pass
params = net.init(rng, inputs)
outputs = jitted_net(params, inputs, training=False)
```

### With Static Arguments

```python
# Compile with static training flag
jitted_net = jit_module_forward(net, static_argnums=(2,))  # training is arg 2

# Different compilation for training vs inference
outputs_train = jitted_net(params, inputs, True)   # Compiled for training
outputs_eval = jitted_net(params, inputs, False)   # Compiled for inference
```

## parallel_batch_process

```python
def parallel_batch_process(
    module_fn: Callable,
    batch_size: int = 32,
    device_count: int = None
) -> Callable
```

Create a batched processing function with automatic parallelization. This utility combines `vmap` for batch processing with optional `pmap` for multi-device parallelization. It automatically handles splitting batches across available devices.

**Args:**
- `module_fn`: The module function to parallelize
- `batch_size`: Target batch size for processing
- `device_count`: Number of devices to use (None for all available)

**Returns:**
- A function that processes inputs in parallel batches

**Example:**
```python
from matnet.utils.parallel import parallel_batch_process
from matnet.models.builder import MatrixNetwork

# Create network
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10
)

# Create parallel processing function
parallel_fn = parallel_batch_process(
    net.apply,
    batch_size=64,
    device_count=jax.device_count()
)

# Process large batch across multiple devices
large_inputs = jnp.ones((256, 20))  # 256 samples
params = net.init(rng, large_inputs[:1])

outputs = parallel_fn(params, large_inputs)  # Shape: (256, 10)
```

### Single Device

```python
# Single device processing
single_fn = parallel_batch_process(
    net.apply,
    batch_size=32,
    device_count=1
)

# Processes in batches of 32 on one device
outputs = single_fn(params, large_inputs)
```

### Multi-Device

```python
# Multi-device processing
multi_fn = parallel_batch_process(
    net.apply,
    batch_size=64,  # Total batch size
    device_count=4  # Split across 4 devices
)

# Each device processes 16 samples (64/4)
outputs = multi_fn(params, large_inputs)
```

## create_batched_forward

```python
def create_batched_forward(module: nn.Module, batch_size: int = 32) -> Callable
```

Create a batched forward pass with JIT compilation. This combines JIT compilation with batch processing for optimal performance. The returned function automatically handles batching of arbitrary-sized inputs.

**Args:**
- `module`: The module to create batched forward pass for
- `batch_size`: Batch size for processing

**Returns:**
- A function that efficiently processes inputs in batches

**Example:**
```python
from matnet.utils.parallel import create_batched_forward
from matnet.models.builder import MatrixNetwork

# Create network
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10
)

# Create batched forward pass
batched_forward = create_batched_forward(net, batch_size=32)

# Process arbitrary-sized inputs
inputs = jnp.ones((137, 20))  # 137 samples (not multiple of 32)
params = net.init(rng, inputs[:1])

# Automatically handles batching
outputs = batched_forward(params, inputs, training=False)
# Shape: (137, 10)
```

## Usage Patterns

### Basic Vectorization

```python
from matnet.utils.parallel import vmap_module
from matnet.layers.matrix_layer import MatrixLayer

layer = MatrixLayer(n=8, input_dim=3, output_dim=5)
params = layer.init(rng, jnp.ones((1, 3, 8, 8)))

def single_forward(params, x):
    return layer.apply(params, x[None, ...])[0]

# Without vmap (manual loop)
def manual_batch(params, inputs):
    outputs = []
    for i in range(inputs.shape[0]):
        output = single_forward(params, inputs[i])
        outputs.append(output)
    return jnp.stack(outputs)

# With vmap (automatic)
vmapped_layer = vmap_module(single_forward)
def auto_batch(params, inputs):
    return vmapped_layer(params, inputs)

# vmapped version is much faster!
```

### JIT Compilation

```python
from matnet.utils.parallel import jit_module_forward

# Without JIT (slow)
def slow_forward(params, inputs):
    return net.apply(params, inputs, training=False)

# With JIT (fast)
jitted_forward = jit_module_forward(net)
def fast_forward(params, inputs):
    return jitted_forward(params, inputs, training=False)

# JIT version compiles once, runs fast
```

### Combined Optimization

```python
from matnet.utils.parallel import vmap_module, jit_module_forward

# Create network
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10
)

# Combine vmap and jit for maximum performance
optimized_forward = jit_module_forward(
    vmap_module(net.apply)
)

# Process batches efficiently
outputs = optimized_forward(params, batch_inputs)
```

### Multi-Device Training

```python
from matnet.utils.parallel import parallel_batch_process

# Create network
net = MatrixNetwork(
    matrix_size=8,
    hidden_dims=[16, 32],
    output_dim=10
)

# Create parallel training function
train_fn = parallel_batch_process(
    lambda p, x, y: compute_loss_and_grads(net, p, x, y),
    batch_size=128,
    device_count=jax.device_count()
)

# Train on multiple devices
loss, grads = train_fn(params, inputs, labels)
```

### Production Inference

```python
from matnet.utils.parallel import create_batched_forward

# Create optimized inference function
def create_inference_fn(net):
    return create_batched_forward(net, batch_size=64)

# Use in production
inference_fn = create_inference_fn(net)

# Process any number of inputs efficiently
outputs = inference_fn(params, inputs)
```

## Performance Comparison

### Without Utilities

```python
# Manual batching (slow)
def process_batch(params, inputs):
    batch_size = inputs.shape[0]
    outputs = []
    for i in range(batch_size):
        output = net.apply(params, inputs[i:i+1])
        outputs.append(output)
    return jnp.concatenate(outputs, axis=0)

# Manual JIT (cumbersome)
@jax.jit
def jitted_forward(params, inputs):
    return net.apply(params, inputs)
```

### With Utilities

```python
# Automatic batching (fast)
from matnet.utils.parallel import create_batched_forward
batched_fn = create_batched_forward(net, batch_size=32)

# Automatic vectorization (fast)
from matnet.utils.parallel import vmap_module
vmapped_fn = vmap_module(net.apply)

# Combined (fastest)
from matnet.utils.parallel import jit_module_forward, vmap_module
optimized_fn = jit_module_forward(vmap_module(net.apply))
```

## Performance Benefits

### Vectorization (vmap)

- **Automatic**: No manual loops needed
- **Efficient**: Uses optimized JAX operations
- **Clean**: Simpler code
- **Fast**: Typically 10-100x speedup

### Compilation (jit)

- **Optimized**: XLA compilation
- **Cached**: Reuse compiled code
- **Fast**: 2-10x speedup after compilation
- **Portable**: Works across devices

### Parallelization (pmap)

- **Multi-device**: Use all available devices
- **Scalable**: Linear speedup with devices
- **Automatic**: Handles device communication
- **Efficient**: Overlaps computation and communication

## Implementation Details

### vmap_module

```python
def vmap_module(module_fn, in_axes=0, out_axes=0):
    return jax.vmap(module_fn, in_axes=in_axes, out_axes=out_axes)
```

- Wraps `jax.vmap` for convenience
- Preserves parameter structure
- Handles multiple input arguments
- Supports custom axis specifications

### jit_module_forward

```python
def jit_module_forward(module, static_argnums=()):
    def forward(params, *args, **kwargs):
        return module.apply(params, *args, **kwargs)
    return jax.jit(forward, static_argnums=static_argnums)
```

- Creates jitted forward function
- Handles variable arguments
- Supports static arguments for conditional compilation
- Returns compiled function

### parallel_batch_process

```python
def parallel_batch_process(module_fn, batch_size=32, device_count=None):
    vmap_fn = vmap_module(module_fn)
    
    if device_count > 1:
        # Multi-device: use pmap
        def parallel_fn(params, inputs):
            # Split across devices
            inputs = reshape_for_devices(inputs, device_count)
            pmapped_fn = jax.pmap(vmap_fn)
            outputs = pmapped_fn(params, inputs)
            return reshape_back(outputs)
    else:
        # Single device: use vmap with batching
        def parallel_fn(params, inputs):
            # Process in batches
            outputs = []
            for batch in split_into_batches(inputs, batch_size):
                batch_outputs = vmap_fn(params, batch)
                outputs.append(batch_outputs)
            return concatenate(outputs)
    
    return parallel_fn
```

- Combines vmap and pmap
- Handles batch splitting
- Manages device placement
- Provides unified interface

### create_batched_forward

```python
def create_batched_forward(module, batch_size=32):
    jitted_forward = jit_module_forward(module)
    
    def batched_forward(params, inputs, **kwargs):
        n_samples = inputs.shape[0]
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        outputs = []
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            batch = inputs[start:end]
            batch_outputs = jitted_forward(params, batch, **kwargs)
            outputs.append(batch_outputs)
        
        return concatenate(outputs)
    
    return batched_forward
```

- Combines JIT and batching
- Handles arbitrary input sizes
- Manages batch boundaries
- Provides clean interface

## Best Practices

1. **Always Use vmap**: Replace manual loops with `vmap_module()`
2. **JIT for Production**: Use `jit_module_forward()` for deployment
3. **Batch Appropriately**: Choose batch size based on memory constraints
4. **Use Multi-Device**: Enable `pmap` when multiple devices available
5. **Combine Optimizations**: Use vmap + jit for maximum performance
6. **Profile First**: Measure performance before optimizing
7. **Static Arguments**: Use `static_argnums` for conditional compilation

## Common Patterns

### Training Loop

```python
def create_train_fn(net, optimizer):
    # Create optimized forward pass
    forward_fn = jit_module_forward(net)
    
    @jax.jit
    def train_step(params, opt_state, inputs, labels):
        def loss_fn(p):
            logits = forward_fn(p, inputs, training=True)
            loss = cross_entropy_loss(logits, labels)
            return loss
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state, loss
    
    return train_step
```

### Inference Pipeline

```python
def create_inference_pipeline(net):
    # Create optimized inference function
    inference_fn = create_batched_forward(net, batch_size=64)
    
    def predict(params, inputs):
        # Process in optimal batches
        outputs = inference_fn(params, inputs, training=False)
        return outputs
    
    return predict
```

### Distributed Training

```python
def create_distributed_train_fn(net, optimizer, device_count):
    # Create parallel processing function
    parallel_fn = parallel_batch_process(
        lambda p, x, y: compute_grads(net, p, x, y),
        batch_size=128,
        device_count=device_count
    )
    
    def distributed_train_step(params, opt_state, inputs, labels):
        # Compute gradients in parallel
        grads = parallel_fn(params, inputs, labels)
        
        # Average gradients across devices
        grads = jax.tree_map(lambda g: jnp.mean(g, axis=0), grads)
        
        # Update parameters
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        
        return params, opt_state
    
    return distributed_train_step
```