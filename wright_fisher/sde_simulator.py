"""
sde_simulator.py — Euler-Maruyama simulator for the Wright-Fisher diffusion
"""
import jax
import jax.numpy as jnp
from model import F_drift, Sigma_noise


def make_em_step(kappa, K, lam, h):
    def em_step(x, key):
        x_new = x + F_drift(x, kappa, K, lam) * h + jnp.sqrt(h) * Sigma_noise(x, key)
        return x_new, x_new
    return em_step


def simulate(kappa, K, lam, x0, n_steps, h, seed=42):
    keys   = jax.random.split(jax.random.PRNGKey(seed), n_steps)
    _, xs  = jax.lax.scan(make_em_step(kappa, K, lam, h), x0, keys)
    return jnp.concatenate([x0[None], xs], axis=0)