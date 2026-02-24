# Bayes vs Dempster for 2D Occupancy Grids

Reproducibility package for:

> T. Berlenko and K. Krinkin, "Equivalence and Divergence of Bayesian
> Log-Odds and Dempster's Combination Rule for 2D Occupancy Grids,"
> arXiv:2602.18872, 2026.
> https://arxiv.org/abs/2602.18872

## Overview

This repository contains the experiment code and analysis scripts to reproduce
all simulation and real-data results reported in the paper.

## Requirements

- Python 3.10+
- ~500 MB disk space for results
- ~50 MB for datasets (Intel Lab, Freiburg 079)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Datasets

### Intel Research Lab

Public domain dataset (Dirk Haehnel, Intel Research Seattle, 2003):

```bash
bash scripts/download-intel-lab.sh
```

### Freiburg Building 079

Public dataset (University of Freiburg, 2003). Download `fr079-sm.log` from
the Robotics Data Set Repository (Radish) and place it at:

```
data/freiburg-079/fr079-sm.log
```

## Reproducing Results

### Step 1: Run all experiments

```bash
bash scripts/run-publication-experiments.sh
```

This runs six experiments and writes raw results to `results/`.
Expected runtime: 40-70 minutes on a modern CPU.

For a quick smoke test (small configs, ~2 min):

```bash
python src/experiments/single_agent_tbm/run.py --config configs/single-agent-tbm/small.yaml
python src/experiments/hybrid_pgo_ds/run.py --config configs/hybrid-pgo-ds/small.yaml
```

### Step 2: Compute bootstrap confidence intervals

```bash
python scripts/bootstrap-intel-lab.py
python scripts/bootstrap-freiburg.py
```

### Step 3: Generate figures and tables

```bash
python scripts/analyze-publication.py
python scripts/generate-paper-figures.py
python scripts/generate-ral-figures.py
```

Figures are written to `results/publication/figures/` and tables to
`results/publication/tables/`.

## Experiment Structure

| Experiment | Code | Config | Paper section |
|---|---|---|---|
| H-002: Single-agent Bayes vs DS | `src/experiments/single_agent_tbm/` | `configs/single-agent-tbm/publication.yaml` | Sec. V-A |
| H-003: Multi-robot dynamic | `src/experiments/hybrid_pgo_ds/` | `configs/hybrid-pgo-ds/pub-dynamic-baseline.yaml` | Sec. V-A |
| H-003: Multi-robot noisy sensor | `src/experiments/hybrid_pgo_ds/` | `configs/hybrid-pgo-ds/pub-noisy-sensor.yaml` | Sec. V-A |
| Mechanism isolation test | `src/experiments/mechanism_test/` | `configs/mechanism-test/h002-matched.yaml` | Sec. V-B |
| Intel Lab validation | `src/experiments/intel_lab/` | `configs/intel-lab/default.yaml` | Sec. V-C |
| Freiburg 079 validation | `src/experiments/intel_lab/` | `configs/freiburg-079/default.yaml` | Sec. V-C |

## Core Library

`src/experiments/hybrid_pgo_ds/` is the shared simulation and fusion library:

- `simulation/` -- 2D environment, LiDAR ray casting, robot trajectories
- `fusion/` -- Bayesian, Bayesian+count, Dempster-Shafer/TBM, Yager fusion methods
- `experiment/` -- experiment runner, metrics, config loader

All other experiments import from this library.

## Downstream Evaluation

Path planning clearance evaluation (Sec. V-D):

```bash
python src/experiments/downstream_eval/run_eval.py --from-experiment \
    --config configs/hybrid-pgo-ds/small.yaml \
    --output results/downstream-eval/
```

## License

Code: MIT License. See LICENSE.

Datasets: Intel Lab and Freiburg 079 datasets are in the public domain.
