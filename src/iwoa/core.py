"""Shared types and numerical helpers for population optimizers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

Objective = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class OptimizerResult:
    best_position: np.ndarray
    best_score: float
    convergence: np.ndarray


def as_bounds(lower: float | np.ndarray, upper: float | np.ndarray, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """Return validated, per-coordinate floating-point bounds."""
    lo = np.broadcast_to(np.asarray(lower, dtype=float), (dim,)).copy()
    hi = np.broadcast_to(np.asarray(upper, dtype=float), (dim,)).copy()
    if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(lo >= hi):
        raise ValueError("Bounds must be finite and satisfy lower < upper in every dimension.")
    return lo, hi


def evaluate(objective: Objective, population: np.ndarray) -> np.ndarray:
    """Evaluate a vectorized objective, with a clear error for scalar-only callables."""
    values = np.asarray(objective(population), dtype=float)
    if values.shape != (population.shape[0],):
        raise ValueError(
            "Objective must accept an array shaped (population, dimension) and return one score per row."
        )
    return np.where(np.isfinite(values), values, np.inf)
