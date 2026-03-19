from __future__ import annotations

from io import StringIO

import numpy as np
import pandas as pd

from matnet.datasets import COVERTYPE_FEATURE_DIM, COVERTYPE_NUMERIC_FEATURES, load_covertype


def _dataset_buffer(rows: list[list[int | float]]) -> StringIO:
    buffer = StringIO()
    pd.DataFrame(rows).to_csv(buffer, header=False, index=False)
    buffer.seek(0)
    return buffer


def test_load_covertype_scales_numeric_features_only() -> None:
    rows: list[list[int | float]] = []
    for label in range(1, 8):
        for repeat in range(3):
            numeric = [100 * label + 10 * repeat + offset for offset in range(COVERTYPE_NUMERIC_FEATURES)]
            wilderness = [1 if idx == (label - 1) % 4 else 0 for idx in range(4)]
            soil = [1 if idx == (label + repeat) % 40 else 0 for idx in range(40)]
            rows.append([*numeric, *wilderness, *soil, label])

    dataset = load_covertype(_dataset_buffer(rows), sample_size=21, test_size=1 / 3, random_state=0)

    assert dataset.X_train.shape == (14, COVERTYPE_FEATURE_DIM)
    assert dataset.X_test.shape == (7, COVERTYPE_FEATURE_DIM)
    assert dataset.input_dim == COVERTYPE_FEATURE_DIM
    assert dataset.output_dim == 7
    assert dataset.y_train.dtype == np.int32
    assert dataset.y_test.dtype == np.int32
    assert set(np.unique(dataset.y_train)).issubset(set(range(7)))
    assert set(np.unique(dataset.y_test)).issubset(set(range(7)))

    numeric_mean = dataset.X_train[:, :COVERTYPE_NUMERIC_FEATURES].mean(axis=0)
    np.testing.assert_allclose(numeric_mean, np.zeros(COVERTYPE_NUMERIC_FEATURES), atol=1e-6)

    categorical_values = np.unique(dataset.X_train[:, COVERTYPE_NUMERIC_FEATURES:])
    assert set(categorical_values.tolist()).issubset({0.0, 1.0})


def test_load_covertype_rejects_non_positive_sample_size() -> None:
    rows: list[list[int | float]] = []
    for label in range(1, 8):
        for repeat in range(2):
            numeric = [100 * label + 10 * repeat + offset for offset in range(COVERTYPE_NUMERIC_FEATURES)]
            wilderness = [1 if idx == (label - 1) % 4 else 0 for idx in range(4)]
            soil = [1 if idx == (label + repeat) % 40 else 0 for idx in range(40)]
            rows.append([*numeric, *wilderness, *soil, label])

    try:
        load_covertype(_dataset_buffer(rows), sample_size=0)
    except ValueError as exc:
        assert "sample_size" in str(exc)
    else:
        raise AssertionError("Expected load_covertype to reject non-positive sample sizes.")
