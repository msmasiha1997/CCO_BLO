from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np


ExampleName = Literal["example1", "gaussian_linear"]


@dataclass(frozen=True)
class Example:
    name: ExampleName
    dim: int
    f: Callable[[np.ndarray], float]
    grad_f: Callable[[np.ndarray], np.ndarray]
    g: Callable[[np.ndarray, np.ndarray], np.ndarray]
    grad_g: Callable[[np.ndarray, np.ndarray], np.ndarray]
    sample_z: Callable[[np.random.Generator, int], np.ndarray]


def make_example(name: ExampleName, dim: int) -> Example:
    if name == "example1":
        if dim != 1:
            raise ValueError("example1 is 1D (dim must be 1).")

        def f(x: np.ndarray) -> float:
            x0 = float(x[0])
            return (x0**3) * np.exp(x0)

        def grad_f(x: np.ndarray) -> np.ndarray:
            x0 = float(x[0])
            return np.array([np.exp(x0) * (3.0 * x0**2 + x0**3)])

        def g(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            x0 = float(x[0])
            return 50.0 * z * np.exp(x0) - 5.0

        def grad_g(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            x0 = float(x[0])
            return (50.0 * z * np.exp(x0))[:, None]

        def sample_z(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.exponential(scale=3.0, size=(n,))

        return Example(
            name=name,
            dim=1,
            f=f,
            grad_f=grad_f,
            g=g,
            grad_g=grad_g,
            sample_z=sample_z,
        )

    if name == "gaussian_linear":
        def f(x: np.ndarray) -> float:
            return float(np.sum((x - 2.0) ** 2))

        def grad_f(x: np.ndarray) -> np.ndarray:
            return 2.0 * (x - 2.0)

        def g(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            return (z @ x) - 1.0

        def grad_g(x: np.ndarray, z: np.ndarray) -> np.ndarray:
            return z

        def sample_z(rng: np.random.Generator, n: int) -> np.ndarray:
            return rng.normal(size=(n, dim))

        return Example(
            name=name,
            dim=dim,
            f=f,
            grad_f=grad_f,
            g=g,
            grad_g=grad_g,
            sample_z=sample_z,
        )

    raise ValueError(f"Unknown example: {name}")
