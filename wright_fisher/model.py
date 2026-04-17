"""
model.py — Wright-Fisher drift, diffusion, and parameter transforms

State x in R^3 represents the first three allele frequencies; x4 = 1 - sum(x)
"""
import jax.numpy as jnp
from jax import random


def F_drift(x, kappa, K, lam):
    x1, x2, x3 = x[0], x[1], x[2]
    f1 = kappa[0] + K[0,0]*x1 + K[0,1]*x2 + K[0,2]*x3 - lam[0]*x1*x1 - lam[1]*x1*x2 - lam[2]*x1*x3
    f2 = kappa[1] + K[1,0]*x1 + K[1,1]*x2 + K[1,2]*x3 - lam[0]*x1*x2 - lam[1]*x2*x2 - lam[2]*x2*x3
    f3 = kappa[2] + K[2,0]*x1 + K[2,1]*x2 + K[2,2]*x3 - lam[0]*x1*x3 - lam[1]*x2*x3 - lam[2]*x3*x3
    return jnp.array([f1, f2, f3])


def DF_drift(x, kappa, K, lam):
    return K - jnp.dot(x, lam) * jnp.eye(3) - jnp.outer(x, lam)


def SigmaSigmaT_reduced(x):
    x_full = jnp.array([x[0], x[1], x[2], 1.0 - x[0] - x[1] - x[2]])
    SST    = jnp.diag(x_full) - jnp.outer(x_full, x_full)
    return SST[:3, :3]


def Sigma_noise(x, key):
    x1, x2, x3 = x[0], x[1], x[2]
    x4 = 1.0 - x1 - x2 - x3
    keys = random.split(key, 6)
    w    = jnp.stack([random.normal(k) for k in keys])
    dX1  =  jnp.sqrt(x1*x2)*w[0] + jnp.sqrt(x1*x3)*w[1] + jnp.sqrt(x1*x4)*w[2]
    dX2  = -jnp.sqrt(x1*x2)*w[0] + jnp.sqrt(x2*x3)*w[3] + jnp.sqrt(x2*x4)*w[4]
    dX3  = -jnp.sqrt(x1*x3)*w[1] - jnp.sqrt(x2*x3)*w[3] + jnp.sqrt(x3*x4)*w[5]
    return jnp.array([dX1, dX2, dX3])


def invert_params(kappa, lam, K, tau=100.0, q4=10.0):
    t2   = 2.0 / tau
    q    = jnp.append(lam + q4, q4)
    p4   = t2 * kappa
    p4_row = jnp.append(p4, 1.0 - p4.sum())

    P33 = jnp.array([
        [t2*(K[0,0] - lam[0] + tau/2) + p4[0],  t2*K[1,0] + p4[1],               t2*K[2,0] + p4[2]              ],
        [t2*K[0,1] + p4[0],                      t2*(K[1,1] - lam[1] + tau/2) + p4[1],  t2*K[1,2] + p4[1]       ],
        [t2*K[0,2] + p4[0],                      t2*K[2,1] + p4[2],               t2*(K[2,2] - lam[2] + tau/2) + p4[2]],
    ])

    pcol4 = 1.0 - P33.sum(axis=1)
    P = jnp.vstack([jnp.hstack([P33, pcol4.reshape(-1, 1)]), p4_row])
    return P, q
