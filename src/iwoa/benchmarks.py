"""The nine benchmark functions and domains reported in Table 6 of the paper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Benchmark:
    name: str
    function: Callable[[np.ndarray], np.ndarray]
    lower: float
    upper: float
    dimension: int
    optimum: float


def sphere(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2, axis=1)


def schwefel_222(x: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(x), axis=1) + np.prod(np.abs(x), axis=1)


def schwefel_221(x: np.ndarray) -> np.ndarray:
    return np.max(np.abs(x), axis=1)


def schwefel_12(x: np.ndarray) -> np.ndarray:
    return np.sum(np.cumsum(x, axis=1) ** 2, axis=1)


def schwefel(x: np.ndarray) -> np.ndarray:
    return np.sum(-x * np.sin(np.sqrt(np.abs(x))), axis=1)


def rastrigin(x: np.ndarray) -> np.ndarray:
    return np.sum(x**2 - 10.0 * np.cos(2.0 * np.pi * x) + 10.0, axis=1)


def ackley(x: np.ndarray) -> np.ndarray:
    d = x.shape[1]
    return -20.0 * np.exp(-0.2 * np.sqrt(np.sum(x**2, axis=1) / d)) - np.exp(
        np.sum(np.cos(2.0 * np.pi * x), axis=1) / d
    ) + 20.0 + np.e


def griewank(x: np.ndarray) -> np.ndarray:
    indices = np.sqrt(np.arange(1, x.shape[1] + 1, dtype=float))
    return np.sum(x**2, axis=1) / 4000.0 - np.prod(np.cos(x / indices), axis=1) + 1.0


_SHEKEL_A = np.array(
    [[4, 4, 4, 4], [1, 1, 1, 1], [8, 8, 8, 8], [6, 6, 6, 6], [3, 7, 3, 7]], dtype=float
)
_SHEKEL_C = np.array([0.1, 0.2, 0.2, 0.4, 0.4], dtype=float)


def shekel_5(x: np.ndarray) -> np.ndarray:
    """Fixed-dimension F9 in Table 6: the four-dimensional Shekel-5 function."""
    inner = np.sum((x[:, None, :] - _SHEKEL_A[None, :, :]) ** 2, axis=2) + _SHEKEL_C[None, :]
    return -np.sum(1.0 / inner, axis=1)


BENCHMARKS: tuple[Benchmark, ...] = (
    Benchmark("F1", sphere, -100, 100, 30, 0.0),
    Benchmark("F2", schwefel_222, -10, 10, 30, 0.0),
    Benchmark("F3", schwefel_221, -10, 10, 30, 0.0),
    Benchmark("F4", schwefel_12, -100, 100, 30, 0.0),
    Benchmark("F5", schwefel, -500, 500, 30, -418.9829 * 30),
    Benchmark("F6", rastrigin, -5.12, 5.12, 30, 0.0),
    Benchmark("F7", ackley, -32, 32, 30, 0.0),
    Benchmark("F8", griewank, -600, 600, 30, 0.0),
    Benchmark("F9", shekel_5, 0, 10, 4, -10.1532),
)
