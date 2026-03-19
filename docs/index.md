# MatNet Documentation

Welcome to the MatNet (Matrix Neural Networks) documentation. MatNet is a JAX/Flax implementation of neural networks where all parameters (weights, biases, neurons) are n×n matrices instead of vectors.

## Overview

Traditional neural networks use scalar-valued connections. MatNet replaces those with matrix-valued connections between matrix neurons, where:

- **Weights**: n×n matrices instead of scalars
- **Biases**: n×n matrices instead of scalars  
- **Neurons**: Represented as n×n matrices
- **Inputs**: Scaled to n×n matrices using learnable parameters
- **Operations**: Element-wise weighted sums over matrix neurons instead of scalar-weighted sums

## Key Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules
- **Parallelization**: Built-in support for JAX's vmap, pmap, and jit
- **Standard Activations**: All standard activation functions (ReLU, sigmoid, tanh, etc.)
- **Normalization**: Element-wise layer normalization for matrix layers
- **Flexible Input/Output**: Input scaling and output decompression layers
- **Easy Model Building**: High-level API for constructing networks

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

Requirements:
- jax >= 0.4.0
- jaxlib >= 0.4.0
- flax >= 0.7.0
- optax >= 0.1.0
- numpy >= 1.21.0

### Basic Usage

```python
import jax
import jax.numpy as jnp
from matnet.models.builder import create_simple_network

# Create a simple 2-layer network
net = create_simple_network(
    matrix_size=8,  # Use 8×8 matrices
    hidden_dim=16,  # Hidden dimension
    output_dim=10,  # 10 output features
    activation="relu"
)

# Initialize parameters
rng = jax.random.PRNGKey(0)
dummy_input = jnp.ones((1, 20))  # Single sample with 20 input features
params = net.init(rng, dummy_input)

# Forward pass
batch_inputs = jnp.ones((32, 20))  # Batch of 32 samples
outputs = net.apply(params, batch_inputs)
print(outputs.shape)  # (32, 10)
```

## Documentation Structure

- **[API Reference](api/)**: Detailed documentation for all modules and classes
- **[User Guides](guides/)**: Tutorials and best practices
- **[Examples](examples/)**: Complete working examples
- **[Development](development.md)**: Contributing and development guide

## Project Structure

```
mat_nets/
├── matnet/                    # Main package
│   ├── __init__.py
│   ├── activations.py         # Matrix-compatible activation functions
│   ├── normalization.py       # Normalization layers
│   ├── layers/                  # Core layer implementations
│   │   ├── matrix_layer.py    # Core matrix multiplication layer
│   │   ├── input_scaling.py   # Input preprocessing layer
│   │   └── decompression.py   # Output conversion layer
│   ├── models/                  # Model building API
│   │   └── builder.py         # High-level model construction
│   └── utils/                   # Utility functions
│       └── parallel.py        # Parallelization utilities
├── tests/                     # Test suite
├── docs/                      # Documentation
└── requirements.txt          # Dependencies
```

## Next Steps

- Read the [Installation Guide](guides/installation.md) for detailed setup instructions
- Follow the [Getting Started Tutorial](guides/getting-started.md) for a hands-on introduction
- Explore the [API Reference](api/) for detailed documentation
- Check out the [Examples](examples/) for complete working code

## Contributing

We welcome contributions! Please see our [Development Guide](development.md) for information on how to contribute to MatNet.

## License

This project is licensed under the MIT License.