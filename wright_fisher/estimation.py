"""
estimation.py — Adam warm-start + BFGS optimization
"""
import jax
import jax.numpy as jnp
import optax
from jax.scipy.optimize import minimize as jax_minimize


def init_params():
    kappa_init = jnp.array([2.5, 2.5, 2.5])
    K_init     = jnp.diag(jnp.array([-25.0, -25.0, -25.0]))
    lam_init   = jnp.ones(3)
    return jnp.concatenate([kappa_init, K_init.ravel(), lam_init])


def run_estimator(params_init, loss_fn, learning_rate=0.01, n_iter=1000,
                  tol=1e-6, patience=200, verbose_every=100,
                  refine=True, verbose=True):
    optimizer     = optax.adam(learning_rate=learning_rate)
    opt_state     = optimizer.init(params_init)
    params        = params_init
    loss_and_grad = jax.value_and_grad(loss_fn)
    losses        = []
    best_loss     = jnp.inf
    no_improve    = 0

    for i in range(n_iter):
        loss, g    = loss_and_grad(params)
        loss       = float(loss)
        updates, opt_state = optimizer.update(g, opt_state)
        params     = optax.apply_updates(params, updates)
        losses.append(loss)
        if loss < best_loss - tol:
            best_loss  = loss
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            if verbose: print(f"Converged at iter {i}  NLL {loss:.6f}")
            break
        if verbose and i % verbose_every == 0:
            print(f"Iter {i:5d}  NLL {loss:.6f}")

    if refine:
        result = jax_minimize(loss_fn, params, method="BFGS")
        params = result.x
        if verbose: print(f"BFGS done  NLL {float(loss_fn(params)):.6f}")

    return params, losses