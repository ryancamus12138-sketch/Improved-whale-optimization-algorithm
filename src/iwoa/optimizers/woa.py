"""Baseline Whale Optimization Algorithm (WOA)."""

from __future__ import annotations

import numpy as np

from iwoa.core import Objective, OptimizerResult, as_bounds, evaluate


def optimize(
    objective: Objective,
    dimension: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    population_size: int = 30,
    iterations: int = 1000,
    seed: int | None = None,
) -> OptimizerResult:
    """Minimize ``objective`` with the standard WOA update equations.

    Equations are the standard form restated in Supplementary Note 3 of Chu et al.
    (2024). One independent random vector is drawn per whale and coordinate.
    """
    if population_size < 2 or iterations < 1:
        raise ValueError("population_size must be >= 2 and iterations must be >= 1")
    lo, hi = as_bounds(lower, upper, dimension)
    rng = np.random.default_rng(seed)
    population = rng.uniform(lo, hi, size=(population_size, dimension))
    scores = evaluate(objective, population)
    best_index = int(np.argmin(scores))
    best_position, best_score = population[best_index].copy(), float(scores[best_index])
    convergence = np.empty(iterations)

    for t in range(iterations):
        a = 2.0 - 2.0 * (t / iterations)
        candidates = np.empty_like(population)
        for i, x in enumerate(population):
            r1, r2 = rng.random(dimension), rng.random(dimension)
            A, C = 2.0 * a * r1 - a, 2.0 * r2
            p = rng.random()
            if p < 0.5:
                if np.max(np.abs(A)) < 1.0:
                    distance = np.abs(C * best_position - x)
                    candidates[i] = best_position - A * distance
                else:
                    random_position = population[rng.integers(population_size)]
                    distance = np.abs(C * random_position - x)
                    candidates[i] = random_position - A * distance
            else:
                l = rng.uniform(-1.0, 1.0)
                distance = np.abs(best_position - x)
                candidates[i] = distance * np.exp(l) * np.cos(2.0 * np.pi * l) + best_position

        population = np.clip(candidates, lo, hi)
        scores = evaluate(objective, population)
        current = int(np.argmin(scores))
        if scores[current] < best_score:
            best_position, best_score = population[current].copy(), float(scores[current])
        convergence[t] = best_score
    return OptimizerResult(best_position, best_score, convergence)
