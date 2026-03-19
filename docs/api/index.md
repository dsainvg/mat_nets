# API Reference

This section provides detailed documentation for all MatNet modules and classes.

## Core Modules

- **[Activations](activations.md)**: Matrix-compatible activation functions
- **[Normalization](normalization.md)**: Normalization layers for matrices
- **[Layers](layers/)**: Core layer implementations
  - [MatrixLayer](layers/matrix_layer.md): Core matrix multiplication layer
  - [InputScalingLayer](layers/input_scaling.md): Input preprocessing
  - [DecompressionLayer](layers/decompression.md): Output conversion
- **[Models](models/)**: Model building API
  - [Builder](models/builder.md): High-level model construction
- **[Utils](utils/)**: Utility functions
  - [Parallel](utils/parallel.md): Parallelization utilities

## Module Structure

```
matnet/
├── activations.py      # Element-wise activation functions for matrices
├── normalization.py    # LayerNorm and BatchNorm for matrices
├── layers/
│   ├── matrix_layer.py      # Core matrix multiplication
│   ├── input_scaling.py     # Input scaling to matrices
│   └── decompression.py     # Output decompression
├── models/
│   └── builder.py          # Model construction API
└── utils/
    └── parallel.py         # JAX parallelization utilities
```

## Quick API Overview

### Building Networks

```python
from matnet.models.builder import build_matrix_network

# Build a network
net = build_matrix_network(
    matrix_size=8,
    hidden_dims=[16, 32, 16],
    output_dim=10,
    activation="relu"
)
```

### Core Layers

```python
from matnet.layers.matrix_layer import MatrixLayer
from matnet.layers.input_scaling import InputScalingLayer
from matnet.layers.decompression import DecompressionLayer

# Matrix-neuron layer
layer = MatrixLayer(n=8, input_dim=4, output_dim=6)

# Input scaling
input_layer = InputScalingLayer(n=8, input_dim=20)

# Output decompression
output_layer = DecompressionLayer(n=8, k=10)
```

### Activations

```python
from matnet import activations

# Apply activation to matrices
x_activated = activations.matrix_relu(x)
x_activated = activations.matrix_sigmoid(x)
x_activated = activations.matrix_tanh(x)
```

### Normalization

```python
from matnet import normalization

# Layer normalization for matrices
norm = normalization.MatrixLayerNorm()
x_normalized = norm(x)

# Batch normalization
batch_norm = normalization.MatrixBatchNorm()
x_normalized = batch_norm(x, training=True)
```

## Data Flow

The typical data flow through a MatNet network is:

1. **Input**: Vector of shape `(batch, input_dim)`
2. **InputScalingLayer**: Converts to matrices of shape `(batch, n, n)`
3. **Neuron Axis**: First matrix layer receives `(batch, 1, n, n)`
4. **MatrixLayer**: Maps `(batch, input_dim, n, n)` to `(batch, output_dim, n, n)`
5. **Activation**: Element-wise activation, preserves `(batch, output_dim, n, n)`
6. **Normalization**: Optional normalization over the trailing `(n, n)` dimensions
7. **DecompressionLayer**: Projects matrix outputs back to vector outputs

## Type Conventions

- **Matrix Size**: Denoted by `n`, creates `n×n` matrices
- **Batch Dimension**: First dimension, can be any size
- **Data Types**: All modules support `dtype` parameter (default: `jnp.float32`)
- **Initialization**: Custom initialization functions via `kernel_init`, `bias_init`

## Performance Considerations

- Use `BatchedMatrixLayer` for better batch performance
- JIT compile with `utils.parallel.jit_module_forward()`
- Use `utils.parallel.vmap_module()` for automatic vectorization
- Consider `matrix_size` trade-offs: larger matrices = more parameters, smaller matrices = less expressiveness