"""
run_single.py — Estimate parameters from a single simulated trajectory
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

jax.config.update("jax_enable_x64", True)
os.makedirs("results", exist_ok=True)

from config import (KAPPA0, K0, LAM0, TAU0, Q0, TRUE_PARAMS, X0,
                    N_SIM, H_SIM, OBS_STEP_COARSE, H_COARSE, ESTIMATORS)
from sde_simulator import simulate
from likelihoods import EM_negloglik, GA_negloglik, SS_negloglik, LL_negloglik
from estimation import run_estimator, init_params
from inference_utils import compute_C, assemble_C, compute_stddict
from figures import plot_trajectory

xs = simulate(KAPPA0, K0, LAM0, X0, N_SIM, H_SIM, seed=42)
print(f"Simulated {len(xs)} steps with h={H_SIM:.5f} for T={N_SIM*H_SIM:.2f}")
print(f"Simplex min x4:       {float(jnp.min(1.0 - xs.sum(axis=1))):.4f}")
print(f"Negative x4 steps:    {int(jnp.sum(1.0 - xs.sum(axis=1) < 0))}")


Xobs = xs[::OBS_STEP_COARSE]
h    = H_COARSE
t  = jnp.arange(Xobs.shape[0]) * h

plot_trajectory(np.array(Xobs), np.array(t), save_path="results/WF_trajectory.png")

p0   = init_params()
loss_fns = {
    "EM": jax.jit(partial(EM_negloglik,     Xobs=Xobs, h=h)),
    "GA": jax.jit(partial(GA_negloglik,     Xobs=Xobs, h=h)),
    "SS": jax.jit(partial(SS_negloglik,     Xobs=Xobs, h=h)),
    "LL": jax.jit(partial(LL_negloglik,     Xobs=Xobs, h=h)),
}

results = {}
for name in ESTIMATORS:
    print(f"\n--- {name} ---")
    params_hat, _ = run_estimator(p0, loss_fns[name], n_iter=1000, verbose=True)
    results[name] = params_hat
    print(f"  kappa est: {jnp.round(params_hat[:3], 2)}  true: {KAPPA0}")
    print(f"  lam   est: {jnp.round(params_hat[12:15], 2)}  true: {LAM0}")
    print(f"  K est:\n{jnp.round(params_hat[3:12].reshape(3,3), 3)}\n  K true:\n{K0}")

blocks  = compute_C(np.array(xs))
C_mat   = np.array(assemble_C(blocks))
stddict = compute_stddict(C_mat, np.array(TRUE_PARAMS),
                           N=len(Xobs) - 1, h=h, tau=TAU0, q4=float(Q0[3]))
print("\n--- Standard deviations for free parameters in (P, q) ---")
for k, v in stddict.items():
    print(f"  {k:6s}: {v:.5f}")