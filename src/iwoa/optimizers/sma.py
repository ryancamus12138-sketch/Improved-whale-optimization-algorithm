"""Slime Mould Algorithm (SMA), included for the paper's Table 8 comparison."""

from __future__ import annotations

import math
import numpy as np

from iwoa.core import Objective, OptimizerResult, as_bounds, evaluate


def optimize(objective: Objective, dimension: int, lower: float | np.ndarray, upper: float | np.ndarray, population_size: int = 30, iterations: int = 1000, seed: int | None = None, z: float = 0.03) -> OptimizerResult:
    """Minimize with SMA; ``z`` follows Table 7 (often named z/VC in implementations)."""
    lo, hi = as_bounds(lower, upper, dimension)
    rng = np.random.default_rng(seed)
    position = rng.uniform(lo, hi, size=(population_size, dimension))
    scores = evaluate(objective, position)
    best_idx = int(np.argmin(scores))
    best, best_score = position[best_idx].copy(), float(scores[best_idx])
    convergence = np.empty(iterations)
    eps = np.finfo(float).eps

    for t in range(iterations):
        order = np.argsort(scores)
        position, scores = position[order], scores[order]
        if scores[0] < best_score:
            best, best_score = position[0].copy(), float(scores[0])
        worst = float(scores[-1])
        denom = max(worst - float(scores[0]), eps)
        weights = np.ones((population_size, dimension))
        for rank in range(population_size):
            ratio = (worst - float(scores[rank])) / denom + 1.0
            sign = 1.0 if rank < population_size / 2 else -1.0
            weights[rank] = 1.0 + sign * rng.random(dimension) * math.log(ratio)

        a = math.atanh(max(-0.999999, min(0.999999, 1.0 - (t + 1) / iterations)))
        b = 1.0 - (t + 1) / iterations
        candidates = position.copy()
        for i in range(population_size):
            if rng.random() < z:
                candidates[i] = rng.uniform(lo, hi, dimension)
                continue
            vb = rng.uniform(-a, a, dimension)
            vc = rng.uniform(-b, b, dimension)
            p = math.tanh(abs(float(scores[i]) - best_score))
            left, right = position[rng.integers(population_size)], position[rng.integers(population_size)]
            if rng.random() < p:
                candidates[i] = best + vb * (weights[i] * left - right)
            else:
                candidates[i] = vc * position[i]
        position = np.clip(candidates, lo, hi)
        scores = evaluate(objective, position)
        idx = int(np.argmin(scores))
        if scores[idx] < best_score:
            best, best_score = position[idx].copy(), float(scores[idx])
        convergence[t] = best_score
    return OptimizerResult(best, best_score, convergence)
