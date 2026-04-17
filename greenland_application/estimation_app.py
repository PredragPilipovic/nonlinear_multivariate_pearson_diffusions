"""
estimation_app.py — Estimation for the Ca2+ application
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "student_kramers"))

import numpy as np
import jax.numpy as jnp
from scipy.stats import chi2

from estimation import run_estimator_lbfgs
from likelihoods_app import SS_quasi_lik
from config_app import MODELS, PARAM_NAMES, embed_params


def make_loss_fn(model_name: str, data, h: float):
    def loss(free_params):
        return SS_quasi_lik(embed_params(free_params, model_name), data, h)
    return loss


def estimate_model(model_name: str, data, h: float,
                   maxiter: int = 1000, tol: float = 1e-5,
                   verbose: bool = True):
    cfg      = MODELS[model_name]
    free_hat, nll, conv = run_estimator_lbfgs(
        cfg["init"], make_loss_fn(model_name, data, h),
        maxiter=maxiter, tol=tol, verbose=verbose)
    return np.array(embed_params(jnp.array(free_hat), model_name)), nll, conv


def compute_derived_params(params: np.ndarray) -> dict:
    eta, alpha, beta, gamma = params[0], params[5], params[6], params[7]
    disc = 4*alpha*gamma - beta**2
    return {
        "nu":        2*eta/alpha + 1,
        "mu":        -beta / (2*alpha),
        "nu_sigma2": disc / (4*alpha**2),
        "omega":     2*beta*eta / (alpha * np.sqrt(disc)),
    }


def likelihood_ratio_test(params_large: np.ndarray, params_small: np.ndarray,
                          data, h: float, df: int = None) -> dict:
    nll_l = float(SS_quasi_lik(jnp.array(params_large), data, h))
    nll_s = float(SS_quasi_lik(jnp.array(params_small), data, h))
    LR    = 2.0 * (nll_s - nll_l)
    if df is None:
        df = MODELS["middle"]["n_free"] - MODELS["small"]["n_free"]
    return {"nll_large": nll_l, "nll_small": nll_s, "LR": LR,
            "df": df, "p_value": float(chi2.sf(LR, df))}
