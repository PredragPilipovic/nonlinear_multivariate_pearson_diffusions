"""
estimation.py — L-BFGS wrapper for parameter estimation
"""
import jax, jax.numpy as jnp, numpy as np
from jaxopt import LBFGS
from config import LBFGS_MAXITER, LBFGS_TOL


def run_estimator_lbfgs(params_init, loss_fn, maxiter=None, tol=None, verbose=True):
    """
    Minimise loss_fn with L-BFGS (Strong Wolfe line search).
    Returns (params_hat, final_loss, convergence)  where convergence=0 means OK.
    """
    maxiter = maxiter or LBFGS_MAXITER
    tol     = tol     or LBFGS_TOL
    opt = LBFGS(fun=loss_fn, maxiter=maxiter, tol=tol,
                linesearch="zoom", history_size=10, implicit_diff=False)

    @jax.jit
    def _step(p, s): return opt.update(p, s)

    params    = jnp.array(params_init)
    state     = opt.init_state(params)
    par_old   = params
    conv      = 0
    if verbose: print("--- L-BFGS ---")

    for i in range(1, maxiter + 1):
        params, state = _step(params, state)
        loss_val = float(state.value)
        par_diff = float(jnp.linalg.norm(params - par_old))
        if verbose:
            print(f"  iter {i:4d} | NLL {loss_val:.6f} | ||delta|| {par_diff:.2e}")
        if jnp.isnan(state.value) or jnp.any(jnp.isnan(params)):
            conv = 1; break
        if par_diff < tol:
            break
        par_old = params

    if i == maxiter: conv = 1
    return np.array(params), loss_val, conv