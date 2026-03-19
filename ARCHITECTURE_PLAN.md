# MatNet JAX Architecture Plan

## Goal

Rebuild the project as a modular JAX-first codebase that preserves the intended MatNet architecture rather than the oversimplified element-wise version described in parts of the current docs.

## Recovered Architecture

These points are consistent across the current repo materials:

1. Inputs begin as standard vectors, then pass through an input-scaling stage to become `n x n` matrices.
2. The network keeps an explicit neuron axis, so hidden activations have shape:
   `batch x neurons x n x n`
3. The notebook architecture uses a stack of matrix layers with varied neuron counts.
4. Hidden activations use element-wise nonlinearities such as ReLU.
5. Optional normalization is applied after each hidden block.
6. The final hidden bank is reduced across the neuron axis, then decompressed to the task output.

Notebook-defined reference architecture:

- Input dimension: `4`
- Matrix size example: `n = 3`
- Hidden widths: `(16, 24, 32, 24, 16)`
- Hidden activation: `ReLU`
- Output activation: `sigmoid`
- Final reduction: `mean` over the neuron axis

## Critical Mismatch In Current Docs

The current markdown docs describe the core layer as a Hadamard-style operator:

`Y_j = sum_i (W_ji ⊙ X_i) + B_j`

That construction does not really use matrix algebra. It behaves like `n^2` independent scalar subnetworks that happen to be packed into matrices.

Your requirement says the architecture is only the same as a traditional network when `n = 1`. That means the real core layer must couple matrix structure for `n > 1`. The notebook also hints at this by saying the blocks use "matrix multiplication".

Conclusion:

- The current docs are not sufficient as the implementation spec for `MatrixLayer`.
- The main unresolved item is the exact matrix interaction rule inside each connection.

## Recommended Formal Spec

To keep the implementation modular, the project should treat the connection rule as a pluggable matrix operator.

Core hidden state:

- `x`: shape `(batch, in_neurons, n, n)`

Core parameters:

- `weights`: shape depends on the chosen operator
- `bias`: shape `(out_neurons, n, n)` when enabled

Core layer contract:

- `MatrixLayer(x) -> y`
- `y` shape: `(batch, out_neurons, n, n)`

Recommended abstraction:

- `matnet/core/operators.py`
- Define operator functions such as:
  - `hadamard_connection`
  - `left_multiply_connection`
  - `right_multiply_connection`
  - `bilinear_connection`
- `MatrixLayer` delegates the per-connection transform to one selected operator.

This lets us implement the library cleanly once your intended operator is confirmed, without rewriting the whole stack later.

## JAX-First Modular Design

I recommend pure JAX for the core implementation, with Optax for optimization. Flax wrappers can be added later only if needed.

Proposed package layout:

```text
matnet/
  __init__.py
  config.py
  types.py
  activations.py
  core/
    __init__.py
    operators.py
    init.py
    shape_checks.py
  layers/
    __init__.py
    input_scaling.py
    matrix_layer.py
    normalization.py
    readout.py
    pooling.py
  models/
    __init__.py
    matnet.py
    builder.py
  train/
    __init__.py
    losses.py
    metrics.py
    state.py
    steps.py
  utils/
    __init__.py
    parallel.py
    tree.py
tests/
examples/
```

## Module Responsibilities

### `core/operators.py`

The architectural heart of the project.

- Implements the actual matrix-valued connection rule
- Owns the `n = 1` reduction-to-traditional property
- Keeps operator logic isolated from model wiring

### `layers/input_scaling.py`

Vector-to-matrix projection:

- Input: `(batch, features)`
- Output: `(batch, n, n)`

This should support:

- learned dense projection to `n * n`
- optional structured projection modes later

### `layers/matrix_layer.py`

Owns:

- parameter initialization
- neuron-axis mixing
- optional bias
- application of the selected matrix operator

Expected input/output:

- input: `(batch, in_neurons, n, n)`
- output: `(batch, out_neurons, n, n)`

### `layers/normalization.py`

Keep normalization separate from the core layer so experiments stay cheap:

- layer norm over matrix axes
- optional norm over neuron + matrix axes

### `layers/pooling.py`

Neuron-bank reduction operators:

- mean
- sum
- max
- learned attention pooling later if useful

### `layers/readout.py`

Matrix-to-vector output:

- flatten + linear projection
- optional pooled readout path

### `models/matnet.py`

High-level network assembly:

1. input scaling
2. initial neuron-bank expansion
3. stacked matrix blocks
4. neuron-bank pooling
5. readout
6. optional output activation

## Tensor Contracts

Recommended canonical shapes:

- raw input: `(batch, features)`
- scaled input matrix: `(batch, n, n)`
- hidden bank: `(batch, neurons, n, n)`
- pooled matrix: `(batch, n, n)`
- output: `(batch, output_dim)`

Keeping these contracts fixed will make testing and future model variants much easier.

## Implementation Phases

### Phase 1

Create the foundation:

- package structure
- config dataclasses
- type aliases
- activation functions
- shape-check helpers

### Phase 2

Implement the core algebra:

- operator interface
- `MatrixLayer`
- parameter initialization
- unit tests for shapes and gradients

### Phase 3

Implement model plumbing:

- input scaling
- normalization
- pooling
- readout
- `MatNet` model builder

### Phase 4

Training utilities:

- Optax optimizer setup
- loss functions
- metrics
- JIT-compiled train/eval steps

### Phase 5

Validation and examples:

- example script matching the notebook architecture
- tests for `n = 1`
- tests for batched forward pass
- tests for gradient flow

## Non-Negotiable Test Cases

We should lock these in from the start:

1. Every layer preserves the documented tensor contracts.
2. `jax.grad` works through the full model.
3. The model handles batch size `1` and larger batches.
4. The selected core operator reduces to the scalar/traditional case when `n = 1`.
5. Parameter counts stay predictable from the operator definition.

## Immediate Next Step

Before implementing the whole project, I need the exact `MatrixLayer` connection law confirmed.

Everything else is now structured enough to build cleanly in JAX.
