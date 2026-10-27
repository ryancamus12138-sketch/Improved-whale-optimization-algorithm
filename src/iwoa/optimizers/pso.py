"""Particle Swarm Optimization using the Table 7 parameters."""

from __future__ import annotations

import numpy as np

from iwoa.core import Objective, OptimizerResult, as_bounds, evaluate


def optimize(objective: Objective, dimension: int, lower: float | np.ndarray, upper: float | np.ndarray, population_size: int = 30, iterations: int = 1000, seed: int | None = None, w: float = 0.5, c1: float = 2.0, c2: float = 2.0) -> OptimizerResult:
    lo, hi = as_bounds(lower, upper, dimension)
    rng = np.random.default_rng(seed)
    position = rng.uniform(lo, hi, size=(population_size, dimension))
    velocity = rng.uniform(-(hi - lo), hi - lo, size=(population_size, dimension)) * 0.1
    score = evaluate(objective, position)
    personal_best, personal_score = position.copy(), score.copy()
    idx = int(np.argmin(score))
    global_best, global_score = position[idx].copy(), float(score[idx])
    convergence = np.empty(iterations)
    for t in range(iterations):
        velocity = w * velocity + c1 * rng.random((population_size, dimension)) * (personal_best - position) + c2 * rng.random((population_size, dimension)) * (global_best - position)
        position = np.clip(position + velocity, lo, hi)
        score = evaluate(objective, position)
        improved = score < personal_score
        personal_best[improved], personal_score[improved] = position[improved], score[improved]
        idx = int(np.argmin(personal_score))
        if personal_score[idx] < global_score:
            global_best, global_score = personal_best[idx].copy(), float(personal_score[idx])
        convergence[t] = global_score
    return OptimizerResult(global_best, global_score, convergence)
