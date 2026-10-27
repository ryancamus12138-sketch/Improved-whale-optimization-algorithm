#!/usr/bin/env python3
"""Reproduce Table 6-8-style benchmark experiments and save auditable artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iwoa.benchmarks import BENCHMARKS
from iwoa.optimizers import iwoa, pso, sma, woa

ALGORITHMS = {"IWOA": iwoa.optimize, "WOA": woa.optimize, "SMA": sma.optimize, "PSO": pso.optimize}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=30, help="Independent runs per benchmark (paper: 30).")
    parser.add_argument("--population", type=int, default=30, help="Whale/swarm population size.")
    parser.add_argument("--iterations", type=int, default=1000, help="Iterations (paper: 1000).")
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float | int | str]] = []
    curves: dict[tuple[str, str], list[np.ndarray]] = {}
    master = np.random.default_rng(args.seed)

    for benchmark in BENCHMARKS:
        print(f"Running {benchmark.name} ...", flush=True)
        for name, optimizer in ALGORITHMS.items():
            run_curves = []
            for run in range(args.runs):
                result = optimizer(benchmark.function, benchmark.dimension, benchmark.lower, benchmark.upper, args.population, args.iterations, int(master.integers(0, 2**32 - 1)))
                records.append({"function": benchmark.name, "algorithm": name, "run": run + 1, "best_score": result.best_score})
                run_curves.append(result.convergence)
            curves[(benchmark.name, name)] = run_curves

    runs = pd.DataFrame(records)
    summary = runs.groupby(["function", "algorithm"])["best_score"].agg(["min", "max", "mean", "std"]).reset_index()
    summary.columns = ["function", "algorithm", "best", "worst", "mean", "std"]
    runs.to_csv(args.out / "benchmark_runs.csv", index=False)
    summary.to_csv(args.out / "benchmark_summary.csv", index=False)

    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, benchmark in zip(axes.flat, BENCHMARKS):
        for name in ALGORITHMS:
            mean_curve = np.mean(curves[(benchmark.name, name)], axis=0)
            ax.semilogy(np.maximum(np.abs(mean_curve), 1e-300), label=name)
        ax.set_title(benchmark.name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("|best fitness|")
        ax.grid(alpha=0.25)
    axes.flat[0].legend()
    fig.suptitle("Mean convergence over independent runs")
    fig.tight_layout()
    fig.savefig(args.out / "benchmark_convergence.png", dpi=180)
    print("Saved benchmark_runs.csv, benchmark_summary.csv, and benchmark_convergence.png")


if __name__ == "__main__":
    main()
