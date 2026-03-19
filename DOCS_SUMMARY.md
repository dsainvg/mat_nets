# MatNet Project Documentation Summary

## 1. Project Overview
**MatNet (Matrix Neural Networks)** is a deep learning library built on **JAX** and **Flax**. 

**Core Concept:**
Unlike traditional neural networks where neurons and weights are scalars (single numbers), MatNet implements neural networks where **every parameter is an n×n matrix**.
- **Traditional Neuron**: A scalar value $x$.
- **MatNet Neuron**: An $n \times n$ matrix $X$.
- **Traditional Weight**: A scalar $w$.
- **MatNet Weight**: An $n \times n$ matrix $W$.

It allows you to build, train, and run neural networks that process information using matrix algebra, offering a richer representation capacity than standard scalar networks. It fully integrates with JAX's powerful transformations like `vmap` (vectorization), `pmap` (parallelization), and `jit` (just-in-time compilation).

## 2. Core Layers (`matnet/layers/`)

### **MatrixLayer**
The fundamental building block of the network.
*   **Structure**: It connects a layer of $M$ input matrix neurons to a layer of $L$ output matrix neurons.
*   **Operation**: For each output matrix neuron $j$ (where $j = 1...L$), it computes a weighted sum of all input matrices $i$ (where $i = 1...M$).
*   **Mathematical Formula**:
    $$Y_j = \sum_{i=1}^{M} (W_{ji} \odot X_i) + B_j$$
    *   $Y_j$: The $j$-th output matrix ($n \times n$).
    *   $X_i$: The $i$-th input matrix ($n \times n$).
    *   $W_{ji}$: The learnable weight matrix ($n \times n$) connecting input $i$ to output $j$.
    *   $B_j$: The learnable bias matrix ($n \times n$) for output $j$.
    *   $\odot$: **Element-wise multiplication**.

### **InputScaling**
A preprocessing layer that transforms standard input vectors or scalars into the $n \times n$ matrix format required by the network. It learns optimal scaling parameters to map inputs to the matrix space.

### **DecompressionLayer**
An output layer that transforms the final $n \times n$ matrix representations back into standard vector outputs (e.g., class scores or regression targets).

## 3. Activations (`matnet/activations.py`)
Standard activation functions adapted for matrix operations. 
*   **Behavior**: They are applied **element-wise** to every single element of the $n \times n$ matrices.
*   **Available**: `matrix_relu`, `matrix_leaky_relu`, `matrix_sigmoid`, `matrix_tanh`, `matrix_swish`, `matrix_gelu`, `matrix_elu`.

## 4. Normalization (`matnet/normalization.py`)
*   **MatrixLayerNorm**: Applies layer normalization to the matrix neurons to stabilize training dynamics.
*   **MatrixBatchNorm**: Batch normalization adapted for matrix inputs.

## 5. Model Builders (`matnet/models/`)
High-level APIs to construct networks easily:
*   **`MatrixNetwork`**: A flexible class to build multi-layer matrix networks with configurable depth, hidden dimensions (number of matrix neurons per layer), and activations.
*   **`SimpleMatrixNet`**: A pre-configured 2-layer network for quick prototyping.
*   **`build_matrix_network`**: A factory function to create model instances from configuration.
