# Strang Splitting Estimator for Nonlinear Multivariate SDEs with Pearson-Type Multiplicative Noise

**Supplementary code repository for the paper:**

> *Strang splitting estimator for nonlinear multivariate stochastic differential equations with Pearson-type multiplicative noise*

This repository contains all code required to reproduce every simulation study, real-data application result, and figure reported in the paper.

---

## Repository Structure

```
nonlinear_multivariate_pearson_diffusions/
├── student_kramers/          # Simulation study: Student–Kramers oscillator
├── wright_fisher/            # Simulation study: Wright–Fisher diffusion
└── greenland_application/    # Real-data application: Greenland Ca²⁺ record
```

Each folder has its own `requirements.txt` and writes all outputs to its own `results/` subfolder. The `student_kramers/` and `wright_fisher/` modules are self-contained. `greenland_application/` reuses the SDE simulator and L-BFGS optimiser from `student_kramers/` via runtime path insertion. Therefore, both folders must be present together in the same parent directory.

---

## Module Overview

### `student_kramers/`

Simulation study for the Student–Kramers oscillator, a two-dimensional SDE with Pearson-type multiplicative noise.

| File | Description |
|---|---|
| `config.py` | True parameters, grid settings, estimator list, palette |
| `sde_simulator.py` | Milstein simulator |
| `likelihoods.py` | EM, GA, SS and LL pseudo log-likehood functions |
| `estimation.py` | L-BFGS optimization wrapper |
| `inference_utils.py` | Cleaning outliers, Fisher information matrix `C`, asymtotic standard deviation |
| `figures.py` | Trajectory plot, estimator density plots, computation time bar chart |
| `run_single.py` | Single test simulation and estimation; generates **Figure 2** |
| `run_simulation.py` | **Long run.** 1000 Monte Carlo iterations. Saves CSVs to `results/` |
| `run_analysis.py` | Loads CSVs, prints summary tables, generates **Figures 4 and 5** |
| `requirements.txt` | Python dependencies |

### `wright_fisher/`

Simulation study for the one-locus four-allele Wright–Fisher diffusion, a four-dimensional SDE on a probability simplex.

| File | Description |
|---|---|
| `config.py` | True parameters, grid settings, estimator list, palette |
| `model.py` | Defines model, maps `(κ, K, λ)` to transition matrix `(P, q)` |
| `sde_simulator.py` | Euler-Mariyama simulator |
| `likelihoods.py` | EM, GA, SS and LL pseudo log-likehood functions |
| `estimation.py` | Adam warm-start + BFGS optimization implementations |
| `inference_utils.py` | Asymptotic covariance matrix `C`, asymtotic standard deviations |
| `figures.py` | Trajectory plot, estimator violin plots, computation time bar chart |
| `run_single.py` | Single test simulation and estimation; generates generates **Figure 1** |
| `run_simulation.py` | **Long run.** 1000 Monte Carlo iterations. Saves CSVs to `results/` |
| `run_analysis.py` | Loads CSVs, inverts parameters, cleans outliers, prints summary tables, generates **Figures 3, 7, and 8** |
| `requirements.txt` | Python dependencies |

### `greenland_application/`

Real-data application to the Greenland ice-core Ca²⁺ record. Files carry an `_app` suffix to avoid naming conflicts when imported alongside `student_kramers/`.

| File | Description |
|---|---|
| `config_app.py` | Model specifications, data path, estimator settings, palette |
| `Ca2data.csv` | Greenland ice-core Ca²⁺ dataset |
| `data_loading.py` | Data loading and preprocessing |
| `simulation_app.py` | Bootstrap SDE simulation, invariant density prediction bands, waiting time calculations |
| `likelihoods_app.py` | Strang-splitting corrected pseudo log-likelihood for the partially observed Student Kramers model |
| `estimation_app.py` | Estimation using function `student_kramers/estimation.py` |
| `bootstrap_app.py` | Parametric bootstrap LR test with outlier removal and resume-on-interrupt support |
| `figures_app.py` | Ca²⁺ series plot, trajectory fit, waiting time distributions, bootstrap distribution |
| `run_application.py` | Fits three nested models to the Ca²⁺ data, runs validation simulation, generates **Figure 6** |
| `run_bootstrap.py` | **Long run.**  1000 bootstrap iterations for the LR test and 1000 large-model bootstrap iterations for the estimator distribution |
| `requirements.txt` | Python dependencies |

---

## Paper Figures — Reproduction Guide

| Figure | Module | Script to run | Prerequisite |
|---|---|---|---|
| Figure 1 | `wright_fisher/` | `run_single.py` | None |
| Figure 2 | `student_kramers/` | `run_single.py` | None |
| Figure 3, 7 and 8 | `wright_fisher/` | `run_analysis.py` | `run_simulation.py` must finish first |
| Figures 4 and 5 | `student_kramers/` | `run_analysis.py` | `run_simulation.py` must finish first |
| Figure 6 | `greenland_application/` | `run_application.py` | `Ca2data.csv` must be load and `run_bootstrap.py` must finish first |

All figures are saved to the module's `results/` folder as 300 dpi PNG files.

---

## Installation

Each module has its own `requirements.txt`. To install dependencies for a given module:

```bash
cd student_kramers     # or wright_fisher / greenland_application
pip install -r requirements.txt
```

**Python and JAX requirements.** All modules require Python 3.10+ and JAX with 64-bit precision enabled. JAX defaults to 32-bit arithmetic unless `jax_enable_x64` is switched on at startup — without it, optimisation, likelihood, and covariance calculations may silently lose precision and produce results that do not match the paper. The flag is set at the top of every script; no manual action is needed. GPU is not required; all experiments were run on CPU.

---

## Reproducing the Results

### Step 1 — Verify the setup

Run the single-iteration test script to confirm the environment is working and produce the trajectory figures (Figures 1 and 2):

```bash
cd wright_fisher      && python run_single.py    # Figure 1
cd student_kramers    && python run_single.py    # Figure 2
```

### Step 2 — Run simulations

> **Warning: Long-running jobs.** `run_simulation.py` runs 1000 Monte Carlo iterations per estimator per step size. Expect runtimes of several hours on a standard laptop. Results are saved as CSV files in `results/` so runs do not need to be repeated if interrupted. `run_analysis.py` requires these CSVs to be present before it can produce any figures.

```bash
cd wright_fisher    && python run_simulation.py
cd student_kramers  && python run_simulation.py
```

### Step 3 — Produce simulation figures

Once the CSVs are in `results/`:

```bash
cd wright_fisher    && python run_analysis.py    # Figures 3, 7, 8
cd student_kramers  && python run_analysis.py    # Figures 4, 5
```

### Step 4 — Run the Greenland application

The application uses the provided `Ca2data.csv` directly and does not depend on the simulation CSVs:

```bash
cd greenland_application && python run_application.py    # Figure 6
```

> **Warning: Long-running job.** `run_bootstrap.py` runs a full parametric bootstrap for the likelihood ratio test and can take several hours. It supports resuming if interrupted.

```bash
cd greenland_application && python run_bootstrap.py
```

---

## Estimators

All three modules implement the following estimators:

| Key | Name |
|---|---|
| `EM` | Euler–Maruyama estimator |
| `GA` | Gaussian approximation estimator |
| `SS` | **Strang splitting estimator** (proposed method) |
| `LL` | Local Linearization estimator |

---

## Data

The Greenland Ca²⁺ dataset used in `greenland_application/` is sourced from:

> Seierstad, I. K., Abbott, P. M., Bigler, M., Blunier, T., Bourne, A. J., Brook, E.,
> Buchardt, S. L., Buizert, C., Clausen, H. B., Cook, E., Dahl-Jensen, D., Davies, S. M.,
> Guillevic, M., Johnsen, S. J., Pedersen, D. S., Popp, T. J., Rasmussen, S. O., Sepp,
> J., Steffensen, J. P., & Svensson, A. (2014). Consistently dated records from the
> Greenland GRIP, GISP2 and NGRIP ice cores for the past 104 ka reveal regional
> millennial-scale δ¹⁸O gradients with possible Heinrich event imprint.
> *Quaternary Science Reviews*, 106, 29–46.
> https://doi.org/10.1016/j.quascirev.2014.10.032

The raw Excel file is downloaded automatically on first run of `run_application.py` from
[iceandclimate.nbi.ku.dk](https://www.iceandclimate.nbi.ku.dk/data/) and cached locally
as `Ca2data.csv`. An internet connection is required the first time only.

---

## Citation

If you use this code, please cite:

```bibtex
@article{strang_splitting_pearson,
  title   = {Strang splitting estimator for nonlinear multivariate stochastic
             differential equations with Pearson-type multiplicative noise},
  author={Pilipović, Predrag and Samson, Adeline and Ditlevsen, Susanne},
  journal = {},
  year    = {2026},
  doi     = {}
}
```

---

## License

MIT License. See `LICENSE` for details.
