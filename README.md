# Improved Whale Optimization Algorithm: Benchmark Reproduction

Independent Python implementation of the Improved Whale Optimization Algorithm (IWOA) and the F1-F9 benchmark experiment from Chu et al. (2024).

This deliberately contains **only the algorithm and benchmark reproduction**:

- implementations of IWOA, baseline WOA, PSO and SMA;
- nine benchmark functions F1-F9 with the domains and dimensions reported in Table 6;
- a fixed-seed, 30-run experiment harness using the paper's 1,000-iteration configuration;
- committed formal outputs: per-run scores, summary statistics and a convergence plot.

No water-resource scheduling model, real-world case data, allocation results, or related assumptions are included.

## IWOA implementation

`src/iwoa/optimizers/iwoa.py` implements the four mechanisms in Eqs. (14)-(22):

1. Tent-Logistic-Cosine chaotic population initialization;
2. nonlinear chaotic convergence factor;
3. Mantegna adaptive Levy-flight search update;
4. greedy sine/cosine spiral position update.

The paper does not release original source code. This is a clean-room implementation: Eq. 14 is mapped from its cosine output interval `[-1, 1]` to optimizer bounds only when initializing the population, and the two Eq. 21/22 candidates are compared directly using the objective as described in the paper.

## Layout

```text
experiments/run_benchmarks.py  Run the F1-F9 comparison
src/iwoa/benchmarks.py         Benchmark definitions
src/iwoa/optimizers/           IWOA, WOA, PSO and SMA
tests/                         Optimizer invariants
results/                       Formal 30-run outputs and convergence figure
```

## Setup and reproduction

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
pytest -q

# Paper-scale benchmark protocol: 30 runs, 1,000 iterations
python experiments/run_benchmarks.py
```

For a quick validation run:

```bash
python experiments/run_benchmarks.py --runs 2 --population 15 --iterations 80 --out results/smoke
```

## Included formal outputs

- `results/benchmark_runs.csv`: final score from every independent run.
- `results/benchmark_summary.csv`: best, worst, mean and standard deviation per algorithm/function pair.
- `results/benchmark_convergence.png`: mean best-so-far convergence across runs.

The committed F9 result uses the four-dimensional **Shekel-5** function, as evidenced by the paper's stated optimum `-10.1532`; the IWOA best score is `-10.153198`.

## Citation

```bibtex
@article{chu2024iwoa,
  title   = {Addressing the contradiction between water supply and demand: a study on multi-objective regional water resources optimization allocation},
  author  = {Chu, Jingyi and Wang, Zhaocai and Bao, Xiaoguang and Yao, Zhiyuan and Cui, Xuefei},
  journal = {Environment, Development and Sustainability},
  year    = {2024},
  doi     = {10.1007/s10668-024-05214-z}
}
```

MIT License. The source-paper copyright remains with its rightsholders.
