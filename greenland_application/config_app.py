"""
config_app.py

Parameter order: [eta, a, b, c, d, alpha, beta, gamma]
                  [0]  [1] [2] [3] [4]  [5]   [6]   [7]
"""
import jax.numpy as jnp
import numpy as np

DATA_URL = (
    "https://www.iceandclimate.nbi.ku.dk/data/"
    "GICC05modelext_GRIP_and_GISP2_and_resampled_data_series_"
    "Seierstad_et_al._2014_version_10Dec2014-2.xlsx"
)
DATA_CACHE     = "Ca2data.csv"
AGE_PREFILTER  = (17000, 90000)
AGE_WINDOW     = (30000, 80000)

H_OBS          = 0.02
H_SIM          = 0.001
SUBSAMPLE_RATE = int(round(H_OBS / H_SIM))

N_BOOTSTRAP      = 1000
N_BOOTSTRAP_LARGE = 1000
BOOTSTRAP_SEED   = 123

SIM_SEED_MIDDLE  = 42
SIM_SEED_SMALL   = 43
SIM_TRAJ_OVERLAY = 0
K                = 11
LVL              = 0.6  
RESULTS_DIR      = "results"

MODELS = {
    "large": {
        "free_indices": [0, 1, 2, 3, 4, 5, 6, 7],
        "fixed":        {},
        "init":         jnp.array([100.0, -100.0, 0.0, 100.0, 0.0, 100.0, 0.0, 1000.0]),
        "n_free":       8,
    },
    "middle": {
        "free_indices": [0, 1, 3, 5, 6, 7],
        "fixed":        {2: 0.0, 4: 0.0},
        "init":         jnp.array([100.0, -10.0, 100.0, 100.0, 0.0, 1000.0]),
        "n_free":       6,
    },
    "small": {
        "free_indices": [0, 1, 3, 7],
        "fixed":        {2: 0.0, 4: 0.0, 5: 0.0, 6: 0.0},
        "init":         jnp.array([60.0, -200.0, 300.0, 9000.0]),
        "n_free":       4,
    },
}

OLD_SMALL_MODEL_PARAMS = jnp.array([
    62.4887, -219.0646, 0.0, 296.6795, 0.0, 0.0, 0.0, 9125.4170,
])

SMALL_MODEL_PARAMS = jnp.array([
    57.6876, -130.0412, 0.0, 115.2705, 0.0, 0.0, 0.0, 8322.5068,
])

NLL_MIDDLE_OBS = 8524.35
NLL_SMALL_OBS  = 8673.11
LR_OBS         = 2.0 * (NLL_SMALL_OBS - NLL_MIDDLE_OBS)

PARAM_NAMES = ["eta", "a", "b", "c", "d", "alpha", "beta", "gamma_"]

PARAM_LABELS = {
    "eta":    r"$\eta$",
    "a":      r"$a$",
    "b":      r"$b$",
    "c":      r"$c$",
    "d":      r"$d$",
    "alpha":  r"$\alpha$",
    "beta":   r"$\beta$",
    "gamma_": r"$\gamma$",   
}


def embed_params(free_vals, model_name: str):
    """Reconstruct the full 8-parameter vector from the free parameters of a named model."""
    cfg  = MODELS[model_name]
    full = jnp.zeros(8)
    for idx, val in cfg["fixed"].items():
        full = full.at[idx].set(val)
    for pos, idx in enumerate(cfg["free_indices"]):
        full = full.at[idx].set(free_vals[pos])
    return full
