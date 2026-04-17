"""
run_simulation.py — Full Monte Carlo simulation study (resumable)
Results saved to results/estimates_{EST}_h{TAG}.csv
EST in {EM, GA, LL, SS} and TAG in {fine, coarse}
"""
import os, time, gc
import jax, jax.numpy as jnp, pandas as pd
from tqdm import tqdm
jax.config.update("jax_enable_x64", True)

import config
from sde_simulator import make_milstein_step
from likelihoods import EM_neg_log_lik, GA_neg_log_lik, LL_neg_log_lik, SS_neg_log_lik
from estimation import run_estimator_lbfgs

_FNS = {"EM": EM_neg_log_lik, "GA": GA_neg_log_lik,
        "LL": LL_neg_log_lik, "SS": SS_neg_log_lik}

def _row(p, nll, t, idx, conv):
    return {"traj": idx, "nll": nll, "time_sec": t, "convergence": conv,
            "eta": float(p[0]), "a": float(p[1]), "b": float(p[2]),
            "c": float(p[3]), "d": float(p[4]), "alpha": float(p[5]),
            "beta": float(p[6]), "gamma_": float(p[7])}

def _run(data_obs, h_obs):
    rows = {}
    for est, fn in _FNS.items():
        nll = jax.jit(lambda p, _f=fn: _f(p, data_obs, h_obs))
        _ = nll(config.INIT_PARAMS)
        t0 = time.perf_counter()
        hat, loss, conv = run_estimator_lbfgs(config.INIT_PARAMS, nll, verbose=False)
        rows[est] = _row(hat, loss, time.perf_counter()-t0, None, conv)
    return rows

def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    files = {(e,t): f"{config.RESULTS_DIR}/estimates_{e}_h{t}.csv"
             for e in config.ESTIMATORS for t in config.H_TAGS}
    start = 0
    for fp in files.values():
        if os.path.exists(fp):
            df = pd.read_csv(fp)
            if len(df): start = max(start, int(df["traj"].max()) + 1)
    print(f"Resuming from {start} / {config.N_TRAJ}")

    n_sim   = int(config.T / config.H_SIM)
    step_fn = make_milstein_step(config.TRUE_PARAMS, config.H_SIM)

    for idx in (pbar := tqdm(range(start, config.N_TRAJ))):
        key  = jax.random.PRNGKey(idx)
        _, xs = jax.lax.scan(step_fn, config.X0, jax.random.split(key, n_sim))
        traj  = jnp.concatenate([config.X0[None], xs], axis=0)
        for tag in config.H_TAGS:
            h_obs = config.H_VALUES[tag]
            rows  = _run(traj[::int(h_obs/config.H_SIM)], h_obs)
            for est in config.ESTIMATORS:
                rows[est]["traj"] = idx
                fp = files[(est, tag)]
                pd.DataFrame([rows[est]]).to_csv(
                    fp, mode="a", header=not os.path.exists(fp), index=False)
        pbar.set_postfix(traj=idx)
        gc.collect()

if __name__ == "__main__":
    main()