import sys

import numpy as np

sys.path.insert(0, "src")

from iwoa.benchmarks import sphere
from iwoa.optimizers import iwoa, pso, sma, woa


def test_all_optimizers_respect_bounds_and_return_convergence() -> None:
    for optimizer in (iwoa.optimize, woa.optimize, pso.optimize, sma.optimize):
        result = optimizer(sphere, 5, -3, 3, population_size=10, iterations=15, seed=7)
        assert result.convergence.shape == (15,)
        assert np.all(result.best_position >= -3) and np.all(result.best_position <= 3)
        assert np.all(np.diff(result.convergence) <= 1e-12)
