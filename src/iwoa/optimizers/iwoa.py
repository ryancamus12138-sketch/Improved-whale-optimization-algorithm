"""Improved WOA (IWOA) implemented from Eqs. (14)-(22) in Chu et al. (2024)."""

from __future__ import annotations

import math
import numpy as np

from iwoa.core import Objective, OptimizerResult, as_bounds, evaluate


def tent_logistic_cosine(x: np.ndarray, r: float = 0.7) -> np.ndarray:
    """One Tent-Logistic-Cosine update (Eq. 14).

    Eq. 14 emits values on [-1, 1].  The caller maps them to [0, 1] before
    scaling into a search interval; this keeps the published map intact while
    respecting arbitrary optimizer bounds.
    """
    first = 2.0 * r * x + 4.0 * (1.0 - r) * x * (1.0 - x) - 0.5
    second = 2.0 * r * (1.0 - x) + 4.0 * (1.0 - r) * x * (1.0 - x) - 0.5
    return np.cos(np.pi * np.where(x < 0.5, first, second))


def mantegna_levy(rng: np.random.Generator, shape: tuple[int, ...], beta: float = 1.5) -> np.ndarray:
    """Sample Levy steps with Mantegna's algorithm (Eq. 19)."""
    sigma_u = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.normal(0.0, sigma_u, size=shape)
    v = rng.normal(0.0, 1.0, size=shape)
    return u / (np.abs(v) ** (1.0 / beta) + np.finfo(float).eps)


def optimize(
    objective: Objective,
    dimension: int,
    lower: float | np.ndarray,
    upper: float | np.ndarray,
    population_size: int = 30,
    iterations: int = 1000,
    seed: int | None = None,
    chaotic_r: float = 0.7,
    a_initial: float = 2.0,
    a_final: float = 0.0,
    spiral_b: float = 1.0,
) -> OptimizerResult:
    """Minimize a vectorized objective using the paper's four IWOA changes.

    The paper does not publish source code.  Two operational choices are made
    explicitly: the Eq. 14 output is rescaled from [-1,1] to [0,1] for bounded
    initialization, and Eq. 21/Eq. 22 candidates are compared greedily using
    the objective, as the prose specifies.
    """
    if population_size < 2 or iterations < 1:
        raise ValueError("population_size must be >= 2 and iterations must be >= 1")
    lo, hi = as_bounds(lower, upper, dimension)
    rng = np.random.default_rng(seed)

    # Eq. 14: independent seeds make a full population rather than a single orbit.
    chaos = tent_logistic_cosine(rng.random((population_size, dimension)), chaotic_r)
    population = lo + ((chaos + 1.0) / 2.0) * (hi - lo)
    scores = evaluate(objective, population)
    best_index = int(np.argmin(scores))
    best_position, best_score = population[best_index].copy(), float(scores[best_index])
    convergence = np.empty(iterations)
    y = float(rng.uniform(-1.0, 1.0))

    for t in range(iterations):
        # Eqs. 15-16. t is 1-indexed in the paper.
        step = t + 1
        a = a_initial * abs(y) - (a_initial - a_final) * math.tan(0.875 * step / iterations)
        candidates = np.empty_like(population)

        for i, x in enumerate(population):
            r1, r2 = rng.random(dimension), rng.random(dimension)
            A, C = 2.0 * a * r1 - a, 2.0 * r2
            p = rng.random()
            if p < 0.5:
                if np.max(np.abs(A)) < 1.0:
                    # Standard WOA encircling remains the exploitation update.
                    distance = np.abs(C * best_position - x)
                    candidates[i] = best_position - A * distance
                else:
                    # Eq. 17 in the prey-searching phase, with Eq. 18/19.
                    distance = np.abs(C * best_position - x)
                    h = 0.01 * step * math.exp(-step / iterations)
                    levy = mantegna_levy(rng, (dimension,))
                    candidates[i] = best_position - h * levy * A * distance
            else:
                # Eqs. 20-22, then greedy selection of sine/cosine candidates.
                r1_scalar = a - a * step / iterations
                r2_scalar = rng.uniform(-1.0, 1.0)
                r3 = a * step / iterations * math.cos(math.pi * step)
                d_prime = np.abs(r3 * C * best_position - x)
                common = r1_scalar * d_prime * math.exp(spiral_b * r2_scalar)
                sine = best_position + common * math.sin(2.0 * math.pi * r2_scalar)
                cosine = best_position + common * math.cos(2.0 * math.pi * r2_scalar)
                pair = np.clip(np.vstack((sine, cosine)), lo, hi)
                pair_scores = evaluate(objective, pair)
                candidates[i] = pair[int(np.argmin(pair_scores))]

        population = np.clip(candidates, lo, hi)
        scores = evaluate(objective, population)
        current = int(np.argmin(scores))
        if scores[current] < best_score:
            best_position, best_score = population[current].copy(), float(scores[current])
        convergence[t] = best_score
        y = 1.0 - 2.0 * y**2

    return OptimizerResult(best_position, best_score, convergence)
