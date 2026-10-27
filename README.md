# Improved Whale Optimization Algorithm (IWOA)

A clean-room Python reproduction of the **Improved Whale Optimization Algorithm (IWOA)** and its F1-F9 benchmark study.

This repository reproduces the optimisation algorithm proposed in:

> Chu, J., Wang, Z., Bao, X., Yao, Z., & Cui, X. (2024). *Addressing the contradiction between water supply and demand: a study on multi-objective regional water resources optimization allocation*. **Environment, Development and Sustainability**.  
> DOI: [10.1007/s10668-024-05214-z](https://doi.org/10.1007/s10668-024-05214-z)

## Overview

The Whale Optimization Algorithm (WOA) is a population-based metaheuristic inspired by humpback whales' bubble-net hunting strategy. It alternates between encircling the current best solution, spiral movement, and exploration around randomly selected individuals.

The paper improves standard WOA to reduce premature convergence and strengthen global search. This repository implements the following four mechanisms from Eqs. (14)-(22):

1. **Tent-Logistic-Cosine chaotic initialization** - generates a more diverse initial population than uniform random initialization.
2. **Chaotic convergence factor** - replaces WOA's linear decay of the convergence factor with a nonlinear chaotic schedule.
3. **Adaptive Levy flight** - introduces long-range search steps during exploration to help escape local optima.
4. **Greedy sine-cosine spiral update** - generates sine and cosine spiral candidates around the best whale, retaining the candidate with better fitness.

## Reproduction Scope

This is an **algorithm and benchmark reproduction only**. It contains:

- IWOA, standard WOA, Particle Swarm Optimization (PSO), and Slime Mould Algorithm (SMA);
- nine benchmark functions F1-F9, including the domains and dimensions reported in Table 6;
- a fixed-seed experiment harness for 30 independent runs and 1,000 iterations;
- committed result tables and convergence curves.

It intentionally excludes the paper's water-allocation case study and real-world scheduling data.

## Benchmark Experiment

The experiment compares IWOA, WOA, PSO, and SMA on nine benchmark functions.

| Group | Functions | Purpose |
| --- | --- | --- |
| Unimodal | F1-F4 | Evaluates convergence accuracy on simpler search landscapes |
| Multimodal | F5-F8 | Evaluates the ability to avoid local optima |
| Fixed-dimensional multimodal | F9 | Four-dimensional Shekel-5 function |

The included experiment follows the paper-scale protocol: **30 independent runs** and **1,000 iterations**.

The reproduced F9 implementation is the four-dimensional **Shekel-5** function. IWOA reaches a best value of **-10.153198**, consistent with the paper's reported optimum of **-10.1532**.
