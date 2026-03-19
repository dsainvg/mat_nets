from __future__ import annotations

import jax
import jax.numpy as jnp

from matnet.models.builder import SimpleMatrixNet


def main() -> None:
    net = SimpleMatrixNet(
        matrix_size=8,
        hidden_dim=16,
        output_dim=10,
        input_dim=20,
        activation="relu",
    )

    rng = jax.random.PRNGKey(0)
    inputs = jnp.ones((32, 20), dtype=jnp.float32)
    params = net.init(rng, inputs)
    outputs = net.apply(params, inputs)

    print("MatNet simple example")
    print(f"Input shape: {inputs.shape}")
    print(f"Output shape: {outputs.shape}")
    print(f"Output dtype: {outputs.dtype}")


if __name__ == "__main__":
    main()
