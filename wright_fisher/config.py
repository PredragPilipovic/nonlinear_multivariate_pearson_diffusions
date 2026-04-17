"""
config.py — Wright-Fisher diffusion study configuration

Parameter vector layout: [kappa (3), vec(K) (9), lam (3)] = 15 entries
"""
import jax.numpy as jnp

KAPPA0 = jnp.array([7.5,  5.0, 5.0])
K0     = jnp.array([
    -32.5,  2.5,  5.0,
     10.0, -22.5, 12.5,
      2.5,  25.0, -30.0
]).reshape(3, 3)
LAM0   = jnp.array([15.0, 30.0, 20.0])

TRUE_PARAMS = jnp.concatenate([KAPPA0, K0.ravel(), LAM0])

TAU0 = 100.0
Q0   = jnp.array([25.0, 40.0, 30.0, 10.0], dtype=float)
P0   = jnp.array([
    [0.20, 0.30, 0.15, 0.35],
    [0.20, 0.05, 0.35, 0.40],
    [0.25, 0.60, 0.10, 0.05],
    [0.15, 0.10, 0.10, 0.65],
], dtype=float)

X0 = jnp.array([0.25, 0.25, 0.25])

T               = 20.0
N_SIM           = 200000
H_SIM           = T / N_SIM

H_COARSE        = 0.2
H_FINE          = 0.02
OBS_STEP_COARSE = int(H_COARSE / H_SIM)
OBS_STEP_FINE   = int(H_FINE   / H_SIM)
N_OBS_COARSE    = int(T / H_COARSE) + 1
N_OBS_FINE      = int(T / H_FINE)   + 1

N_TRAJ      = 2
RESULTS_DIR = "results"

ESTIMATORS = ["EM", "GA", "SS", "LL"]
H_SETTINGS = {"coarse": H_COARSE, "fine": H_FINE}

PALETTE = {'EM': '#ff9999', 'GA': '#99ff99', 'SS': '#99ccff', 'LL': '#d9b3ff'}
