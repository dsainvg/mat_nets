from setuptools import find_packages, setup

setup(
    name="matnet",
    version="0.1.0",
    description="Matrix neural networks built with JAX and Flax",
    packages=find_packages(),
    install_requires=[
        "jax>=0.4.0",
        "jaxlib>=0.4.0",
        "flax>=0.7.0",
        "numpy>=1.24.0",
    ],
    python_requires=">=3.10",
)
