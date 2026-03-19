from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from matnet.datasets import load_covertype
from matnet.models.builder import MatrixNetwork


def _batch_iterator(
    X: np.ndarray,
    y: np.ndarray,
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    indices = rng.permutation(len(X))
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        yield X[batch_indices], y[batch_indices]


def _count_params(params) -> int:
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MatNet on the UCI Covertype dataset.")
    parser.add_argument(
        "--source",
        default="https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz",
        help="Local path or URL to covtype.data.gz",
    )
    parser.add_argument("--sample-size", type=int, default=50_000, help="Rows to keep before train/test split.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction reserved for testing.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--matrix-size", type=int, default=1)
    parser.add_argument(
        "--hidden-dims",
        type=int,
        nargs="+",
        default=[24, 160, 80, 96],
        help="Hidden layer widths. Only the output layer is dataset-specific.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-normalization", action="store_true")
    args = parser.parse_args()

    dataset = load_covertype(
        args.source,
        sample_size=args.sample_size,
        test_size=args.test_size,
        random_state=args.seed,
    )

    model = MatrixNetwork(
        matrix_size=args.matrix_size,
        hidden_dims=tuple(args.hidden_dims),
        output_dim=dataset.output_dim,
        use_normalization=not args.disable_normalization,
    )

    X_test = jnp.asarray(dataset.X_test)
    y_test = jnp.asarray(dataset.y_test)

    variables = model.init(jax.random.PRNGKey(args.seed), jnp.asarray(dataset.X_train[:1]))
    params = variables["params"]
    optimizer = optax.adam(args.learning_rate)
    opt_state = optimizer.init(params)

    @jax.jit
    def train_step(params, opt_state, x_batch, y_batch):
        def loss_fn(current_params):
            logits = model.apply({"params": current_params}, x_batch, training=True)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, y_batch).mean()
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y_batch)
        return params, opt_state, loss, accuracy

    @jax.jit
    def evaluate(params, x, y):
        logits = model.apply({"params": params}, x, training=False)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return loss, accuracy

    rng = np.random.default_rng(args.seed)
    print("Covertype experiment")
    print(f"Train shape: {dataset.X_train.shape}")
    print(f"Test shape: {dataset.X_test.shape}")
    print(f"Classes: {dataset.class_names}")
    print(f"Hidden dims: {tuple(args.hidden_dims)}")
    print(f"Output dim: {dataset.output_dim}")
    print(f"Parameter count: {_count_params(params):,}")

    for epoch in range(1, args.epochs + 1):
        train_losses: list[float] = []
        train_accuracies: list[float] = []

        for x_batch_np, y_batch_np in _batch_iterator(
            dataset.X_train,
            dataset.y_train,
            batch_size=args.batch_size,
            rng=rng,
        ):
            x_batch = jnp.asarray(x_batch_np)
            y_batch = jnp.asarray(y_batch_np)
            params, opt_state, loss, accuracy = train_step(params, opt_state, x_batch, y_batch)
            train_losses.append(float(loss))
            train_accuracies.append(float(accuracy))

        test_loss, test_accuracy = evaluate(params, X_test, y_test)
        print(
            f"Epoch {epoch:02d} | "
            f"train_loss={np.mean(train_losses):.4f} "
            f"train_acc={np.mean(train_accuracies):.4f} "
            f"test_loss={float(test_loss):.4f} "
            f"test_acc={float(test_accuracy):.4f}"
        )


if __name__ == "__main__":
    main()
