"""
run_single.py — Simulate one trajectory and estimate with all four methods
"""
import jax, jax.numpy as jnp
from functools import partial
jax.config.update("jax_enable_x64", True)

import config
from sde_simulator import (simulate_trajectory, subsample,
                   make_stationary_x_density, make_stationary_v_density)
from likelihoods import EM_neg_log_lik, GA_neg_log_lik, LL_neg_log_lik, SS_neg_log_lik
from estimation import run_estimator_lbfgs
from inference_utils import compute_C
from figures import plot_trajectory

def main():
    traj  = simulate_trajectory(config.TRUE_PARAMS, config.T, config.H_SIM, config.X0)
    data  = subsample(traj, config.H_SIM, config.H_FINE)
    t_obs = jnp.linspace(0, config.T, data.shape[0])
    print(f"{traj.shape[0]} fine steps | {data.shape[0]} observed")

    C1, C2 = compute_C(traj, config.TRUE_PARAMS)
    pi_x   = make_stationary_x_density(config.TRUE_PARAMS)
    pi_v   = make_stationary_v_density(config.TRUE_PARAMS)
    plot_trajectory(data, t_obs, pi_x, pi_v,
                    save_path=f"{config.RESULTS_DIR}/trajectory.png")

    names = ["eta","a","b","c","d","alpha","beta","gamma"]
    for label, fn in [("EM", EM_neg_log_lik), ("GA", GA_neg_log_lik),
                      ("LL", LL_neg_log_lik), ("SS", SS_neg_log_lik)]:
        nll = jax.jit(partial(fn, data=data, h=config.H_FINE))
        hat, _, conv = run_estimator_lbfgs(config.INIT_PARAMS, nll)
        print(f"\n--- {label} ---")
        for n, e, t in zip(names, hat, config.TRUE_PARAMS):
            print(f"  {n:6s}: est={float(e):10.3f}  true={float(t):10.3f}")
        if conv: print("  Warning: did not converge.")

if __name__ == "__main__":
    main()