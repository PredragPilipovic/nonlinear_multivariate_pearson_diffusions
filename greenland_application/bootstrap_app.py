"""
bootstrap_app.py — Parametric bootstrap LR test 
                   (resuming if interrupted), with outlier removal
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "student_kramers"))

import gc
import os
import time
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import tqdm

from sde_simulator import make_milstein_step
from estimation import run_estimator_lbfgs
from estimation_app import make_loss_fn
from config_app import (MODELS, SMALL_MODEL_PARAMS, H_OBS, H_SIM,
                         SUBSAMPLE_RATE, N_BOOTSTRAP, BOOTSTRAP_SEED,
                         RESULTS_DIR, LR_OBS, embed_params)


def _row(params_full, nll, elapsed, idx, conv):
    return {"traj": idx, "nll": nll, "time_sec": elapsed, "convergence": conv,
            "eta":    float(params_full[0]), "a":      float(params_full[1]),
            "b":      float(params_full[2]), "c":      float(params_full[3]),
            "d":      float(params_full[4]), "alpha":  float(params_full[5]),
            "beta":   float(params_full[6]), "gamma_": float(params_full[7])}


def run_bootstrap(params_null=None, n_boot: int = None, results_dir: str = RESULTS_DIR,
                  data_shape: int = None, init_state: jnp.ndarray = None,
                  verbose: bool = True):
    if params_null is None: params_null = SMALL_MODEL_PARAMS
    if n_boot is None:      n_boot = N_BOOTSTRAP

    os.makedirs(results_dir, exist_ok=True)
    result_files = {m: os.path.join(results_dir, f"bootstrap_estimates_{m}.csv")
                    for m in ("middle", "small")}
    start = 0
    for fp in result_files.values():
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if len(df): start = max(start, int(df["traj"].max()) + 1)
    if verbose: print(f"Resuming from bootstrap trajectory {start}")

    steps_per_traj = (data_shape - 1) * SUBSAMPLE_RATE
    step_fn_null   = make_milstein_step(params_null, H_SIM)

    for idx in (pbar := tqdm(range(start, n_boot), desc="Bootstrap", disable=not verbose)):
        key   = jax.random.PRNGKey(BOOTSTRAP_SEED + idx)
        _, xs = jax.lax.scan(step_fn_null, init_state, jax.random.split(key, steps_per_traj))
        traj  = jnp.concatenate([init_state[None], xs], axis=0)[::SUBSAMPLE_RATE]
        chunk = traj[:, 0]
        data_sim = jnp.column_stack((chunk[:-1], jnp.diff(chunk) / H_OBS))

        rows = {}
        for model_name in ("middle", "small"):
            loss = make_loss_fn(model_name, data_sim, H_OBS)
            t0   = time.perf_counter()
            free_hat, nll_val, conv = run_estimator_lbfgs(
                MODELS[model_name]["init"], loss, maxiter=300, tol=1e-4, verbose=False)
            params_full = np.array(embed_params(jnp.array(free_hat), model_name))
            rows[model_name] = _row(params_full, nll_val, time.perf_counter()-t0, idx, conv)

        for m in ("middle", "small"):
            pd.DataFrame([rows[m]]).to_csv(
                result_files[m], mode="a",
                header=not os.path.exists(result_files[m]), index=False)

        lr_sim = 2.0 * (rows["small"]["nll"] - rows["middle"]["nll"])
        pbar.set_postfix(traj=idx, LR=f"{lr_sim:.2f}")
        gc.collect()


def compute_bootstrap_pvalue(results_dir: str = RESULTS_DIR,
                              obs_LR: float = None, iqr_factor: float = 3.0) -> dict:
    if obs_LR is None: obs_LR = LR_OBS
    df_m   = pd.read_csv(os.path.join(results_dir, "bootstrap_estimates_middle.csv"))
    df_s   = pd.read_csv(os.path.join(results_dir, "bootstrap_estimates_small.csv"))
    merged = df_s.merge(df_m, on="traj", suffixes=("_small", "_middle"))
    merged["LR"] = 2.0 * (merged["nll_small"] - merged["nll_middle"])
    Q1, Q3  = merged["LR"].quantile(0.25), merged["LR"].quantile(0.75)
    IQR     = Q3 - Q1
    clean   = merged[(merged["LR"] >= Q1 - iqr_factor*IQR) &
                     (merged["LR"] <= Q3 + iqr_factor*IQR)].copy()
    extreme = int(np.sum(clean["LR"] >= obs_LR))
    p_val   = (extreme + 1) / (len(clean) + 1)
    return {"obs_LR": obs_LR, "p_value": p_val, "n_boot": len(clean),
            "n_removed": len(merged) - len(clean), "extreme_count": extreme,
            "pct_95": float(np.percentile(clean["LR"], 95)),
            "LR_values": clean["LR"].values}
