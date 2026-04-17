"""
run_simulation.py — Monte Carlo simulation study: N_TRAJ trajectories, coarse and fine h. Resumable.
"""
import os, gc, time
import jax
import jax.numpy as jnp
from jax import random, jit
from functools import partial
import pandas as pd
from tqdm import tqdm

jax.config.update("jax_enable_x64", True)

from config import (KAPPA0, K0, LAM0, X0,
                    N_SIM, H_SIM,
                    H_COARSE, H_FINE,
                    OBS_STEP_COARSE, OBS_STEP_FINE,
                    N_TRAJ, RESULTS_DIR, ESTIMATORS, H_SETTINGS)
from sde_simulator import make_em_step
from likelihoods import EM_negloglik, GA_negloglik, SS_negloglik, LL_negloglik
from estimation import run_estimator, init_params

os.makedirs(RESULTS_DIR, exist_ok=True)

result_files = {
    (est, tag): f"{RESULTS_DIR}/estimates_{est}_h{tag}.csv"
    for est in ESTIMATORS for tag in H_SETTINGS
}

existing = {}
for key, fpath in result_files.items():
    try:    existing[key] = pd.read_csv(fpath)
    except: existing[key] = pd.DataFrame()

start_traj = 0
for key, df in existing.items():
    if len(df) > 0:
        start_traj = max(start_traj, int(df["traj"].max()) + 1)
print(f"Resuming from traj {start_traj}")

results = {key: [] for key in result_files}


def to_row(params, nll_val, elapsed, idx):
    return {
        "traj": idx, "nll": nll_val, "time_sec": elapsed,
        "k1": float(params[0]), "k2": float(params[1]), "k3": float(params[2]),
        **{f"K{i}{j}": float(params[3 + i*3 + j]) for i in range(3) for j in range(3)},
        "lam1": float(params[12]), "lam2": float(params[13]), "lam3": float(params[14]),
    }


def run_all_estimators(X_obs, h_obs):
    p0 = init_params()
    rows = {}
    for est in ESTIMATORS:
        if   est == "EM": nll_fn = jit(partial(EM_negloglik, Xobs=X_obs, h=h_obs))
        elif est == "GA": nll_fn = jit(partial(GA_negloglik, Xobs=X_obs, h=h_obs))
        elif est == "SS": nll_fn = jit(partial(SS_negloglik, Xobs=X_obs, h=h_obs))
        elif est == "LL": nll_fn = jit(partial(LL_negloglik, Xobs=X_obs, h=h_obs))
        _ = nll_fn(p0)
        t0 = time.perf_counter()
        params_hat, _ = run_estimator(p0, nll_fn, n_iter=1000, verbose=False)
        elapsed = time.perf_counter() - t0
        rows[est] = to_row(params_hat, float(nll_fn(params_hat)), elapsed, None)
    return rows


for traj_idx in (pbar := tqdm(range(start_traj, N_TRAJ), desc="Trajectories")):
    key  = random.PRNGKey(traj_idx)
    keys = random.split(key, N_SIM)
    _, xs = jax.lax.scan(make_em_step(KAPPA0, K0, LAM0, H_SIM), X0, keys)
    xs = jnp.concatenate([X0[None], xs], axis=0)

    X_coarse = xs[::OBS_STEP_COARSE]
    X_fine   = xs[::OBS_STEP_FINE]

    for tag, (h_obs, X_obs) in zip(
        H_SETTINGS.keys(),
        [(H_COARSE, X_coarse), (H_FINE, X_fine)]
    ):
        rows = run_all_estimators(X_obs, h_obs)
        for est in ESTIMATORS:
            row = rows[est]
            row["traj"] = traj_idx
            results[(est, tag)].append(row)

    for key_r, fpath in result_files.items():
        df_new = pd.DataFrame(results[key_r])
        if len(existing[key_r]) > 0:
            df_all = pd.concat([existing[key_r], df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_csv(fpath, index=False)
        existing[key_r] = df_all
        results[key_r] = []

    pbar.set_postfix(traj=traj_idx)
    gc.collect()

print("Done.")
for key_r, fpath in result_files.items():
    print(f"  {fpath}: {len(pd.read_csv(fpath))} rows")