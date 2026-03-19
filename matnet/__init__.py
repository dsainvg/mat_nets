"""Public API for the DOCS_SUMMARY-backed MatNet package."""

from . import activations
from .datasets import ClassificationDataset, load_covertype
from .models.builder import MatrixNetwork, SimpleMatrixNet, build_matrix_network
from .normalization import MatrixBatchNorm, MatrixLayerNorm

__all__ = [
    "activations",
    "ClassificationDataset",
    "load_covertype",
    "MatrixBatchNorm",
    "MatrixLayerNorm",
    "MatrixNetwork",
    "SimpleMatrixNet",
    "build_matrix_network",
]
