"""
config.py — True parameters and simulation settings
Parameters ordered as: [eta, a, b, c, d, alpha, beta, gamma]
"""
import jax.numpy as jnp

ETA, A, B, C, D      = 30.0, -125.0, 40.0, 150.0, -20.0
ALPHA, BETA, GAMMA   = 20.0, -8.0, 1280.8

TRUE_PARAMS      = jnp.array([ETA, A, B, C, D, ALPHA, BETA, GAMMA])
TRUE_PARAMS_DICT = {"eta": ETA, "a": A, "b": B, "c": C,
                    "d": D, "alpha": ALPHA, "beta": BETA, "gamma_": GAMMA}

T, H_SIM, H_FINE, H_COARSE = 50.0, 0.0001, 0.01, 0.02
N_TRAJ           = 1000
X0               = jnp.array([1.5, 0.0])
INIT_PARAMS      = jnp.array([50.0, -200.0, 10.0, 100.0, 10.0, 30.0, -5.0, 1000.0])
LBFGS_MAXITER, LBFGS_TOL = 1000, 1e-5

ESTIMATORS = ["EM", "GA", "SS", "LL"]
H_TAGS     = ["fine", "coarse"]
H_LABELS   = {"coarse": "h = 0.02", "fine": "h = 0.01"}
H_VALUES   = {"fine": H_FINE, "coarse": H_COARSE}
PALETTE    = {"EM": "#ff9999", "GA": "#99ff99", "SS": "#99ccff", "LL": "#d9b3ff"}
LATEX_MAP  = {"eta": r"$\eta$", "a": r"$a$", "b": r"$b$", "c": r"$c$",
              "d": r"$d$", "alpha": r"$\alpha$", "beta": r"$\beta$", "gamma_": r"$\gamma$"}
ALL_PARAMS  = list(TRUE_PARAMS_DICT.keys())
GROUP_DRIFT = ["a", "b", "c", "d"]
GROUP_VOL   = ["eta", "alpha", "beta", "gamma_"]
RESULTS_DIR = "results"