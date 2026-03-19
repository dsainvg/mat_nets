"""Dataset helpers for MatNet experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_COVERTYPE_SOURCE = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
COVERTYPE_FEATURE_DIM = 54
COVERTYPE_NUMERIC_FEATURES = 10
COVERTYPE_CLASS_NAMES = (
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
)


@dataclass(frozen=True)
class ClassificationDataset:
    """Prepared train/test split for integer-label classification."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    input_dim: int
    output_dim: int
    class_names: tuple[str, ...]


def load_covertype(
    source: str | Path = DEFAULT_COVERTYPE_SOURCE,
    *,
    sample_size: int | None = 50_000,
    test_size: float = 0.2,
    random_state: int = 42,
) -> ClassificationDataset:
    """Load, subsample, and standardize the UCI Covertype dataset.

    The first 10 quantitative features are standardized. The remaining 44 binary
    indicators are left unchanged. Labels are remapped from `1..7` to `0..6`
    so they can be used directly with integer-label cross-entropy losses.
    """

    frame = pd.read_csv(source, header=None)
    if frame.shape[1] != COVERTYPE_FEATURE_DIM + 1:
        raise ValueError(
            f"Expected {COVERTYPE_FEATURE_DIM + 1} columns for Covertype, got {frame.shape[1]}."
        )

    data = frame.to_numpy()
    X = data[:, :COVERTYPE_FEATURE_DIM].astype(np.float32)
    y = data[:, COVERTYPE_FEATURE_DIM].astype(np.int32) - 1

    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be a positive integer when provided.")
        if sample_size < len(X):
            X, _, y, _ = train_test_split(
                X,
                y,
                train_size=sample_size,
                random_state=random_state,
                stratify=y,
            )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_numeric = scaler.fit_transform(X_train[:, :COVERTYPE_NUMERIC_FEATURES]).astype(np.float32)
    X_test_numeric = scaler.transform(X_test[:, :COVERTYPE_NUMERIC_FEATURES]).astype(np.float32)

    X_train_processed = np.concatenate(
        [X_train_numeric, X_train[:, COVERTYPE_NUMERIC_FEATURES:]],
        axis=1,
    ).astype(np.float32)
    X_test_processed = np.concatenate(
        [X_test_numeric, X_test[:, COVERTYPE_NUMERIC_FEATURES:]],
        axis=1,
    ).astype(np.float32)

    return ClassificationDataset(
        X_train=X_train_processed,
        X_test=X_test_processed,
        y_train=y_train.astype(np.int32),
        y_test=y_test.astype(np.int32),
        input_dim=COVERTYPE_FEATURE_DIM,
        output_dim=len(COVERTYPE_CLASS_NAMES),
        class_names=COVERTYPE_CLASS_NAMES,
    )
