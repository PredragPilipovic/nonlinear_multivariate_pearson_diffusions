"""
inference_utils.py — Information matrix C, reparametrization and std dev.
"""
import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np

from config import TAU0, Q0


def _S_inv(x):
    xd = 1.0 - jnp.sum(x)
    return jnp.diag(1.0 / x) + (1.0 / xd) * jnp.ones((x.shape[0], x.shape[0]))


def _per_sample(x):
    r   = x.shape[0]
    Si  = _S_inv(x)                                       
    g   = Si @ x                                          
    q   = jnp.dot(x, g)                                   
    xxT = jnp.outer(x, x)                                

    C_kk = Si
    C_KK = jnp.kron(Si, xxT)                              
    C_ll = q * xxT                                       

    C_kK = (Si[:, :, None] * x[None, None, :]).reshape(r, r * r)

    C_kl = -jnp.outer(g, x)

    C_Kl = -(g[:, None, None]
              * x[None, :, None]
              * x[None, None, :]).reshape(r * r, r)

    return C_kk, C_KK, C_ll, C_kK, C_kl, C_Kl


def compute_C(Xobs):
    outs = vmap(_per_sample)(Xobs)
    keys = ["Ckk", "CKK", "Cll", "CkK", "Ckl", "CKl"]
    return {k: jnp.mean(v, axis=0) for k, v in zip(keys, outs)}


def assemble_C(blocks):
    Ckk, CKK, Cll = blocks["Ckk"], blocks["CKK"], blocks["Cll"]
    CkK, Ckl, CKl = blocks["CkK"], blocks["Ckl"], blocks["CKl"]
    row1 = jnp.concatenate([Ckk,   CkK,   Ckl], axis=1)
    row2 = jnp.concatenate([CkK.T, CKK,   CKl], axis=1)
    row3 = jnp.concatenate([Ckl.T, CKl.T, Cll], axis=1)
    return jnp.concatenate([row1, row2, row3], axis=0)


def phi(params_vec, tau=100.0, q4=10.0):
    r     = 3
    kappa = params_vec[:r]
    K     = params_vec[r:r + r*r].reshape(r, r)
    lam   = params_vec[r + r*r:]
    t2    = 2.0 / tau

    q    = jnp.append(lam + q4, q4)
    p4   = t2 * kappa
    p4_4 = 1.0 - p4.sum()

    P33 = jnp.array([
        [t2*(K[0,0] - lam[0] + tau/2) + p4[0],  t2*K[1,0] + p4[1],                          t2*K[2,0] + p4[2]                          ],
        [t2*K[0,1] + p4[0],                      t2*(K[1,1] - lam[1] + tau/2) + p4[1],        t2*K[1,2] + p4[1]                        ],
        [t2*K[0,2] + p4[0],                      t2*K[2,1] + p4[2],                          t2*(K[2,2] - lam[2] + tau/2) + p4[2]      ],
    ])
    p_col4 = 1.0 - P33.sum(axis=1)
    P_full = jnp.vstack([
        jnp.hstack([P33, p_col4.reshape(-1, 1)]),
        jnp.append(p4, p4_4),
    ])                                                     

    return jnp.concatenate([P_full.ravel(), q])            


def compute_stddict(C_matrix, true_params, N, h, tau=None, q4=None):
    if tau is None: tau = float(TAU0)
    if q4  is None: q4  = float(Q0[3])

    C = np.array(C_matrix, dtype=np.float64)
    J = np.array(jax.jacobian(lambda v: phi(v, tau=tau, q4=q4))(
                     jnp.array(true_params, dtype=jnp.float64)))  # (20, 15)

    cond    = np.linalg.cond(C)
    eigvals = np.linalg.eigvalsh(C)
    print(f"  C condition number : {cond:.3e}")
    print(f"  C eigenvalue range : [{eigvals.min():.3e}, {eigvals.max():.3e}]")

    free_idx = [0,1,2, 4,5,6, 8,9,10, 12,13,14, 16,17,18]
    labels   = [f"p{i+1}{j+1}" for i in range(4) for j in range(3)] + ["q1","q2","q3"]

    J_free = J[free_idx, :]                            

    Z    = np.linalg.solve(C, J_free.T)                
    diag = np.sum(J_free * Z.T, axis=1) / (N * h)     

    n_neg = int((diag < 0).sum())
    if n_neg:
        print(f"  WARNING: {n_neg} negative variance entries clipped to 0")

    std = np.sqrt(np.clip(diag, 0.0, None))
    return dict(zip(labels, std))