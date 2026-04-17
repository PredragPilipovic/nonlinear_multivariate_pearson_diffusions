"""
likelihoods.py — EM, GA, SS, LL negative log-likelihoods for the Wright-Fisher diffusion

params: flat vector [kappa(3), vec(K)(9), lam(3)].
Xobs:   shape (M+1, 3).
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from model import F_drift, DF_drift, SigmaSigmaT_reduced


def _unpack(params):
    return params[:3], params[3:12].reshape(3, 3), params[12:15]


# Euler-Maruyama
def EM_negloglik(params, Xobs, h):
    kappa, K, lam = _unpack(params)

    def step(carry, obs):
        xp, xc = obs
        F   = F_drift(xp, kappa, K, lam)
        cov = h * SigmaSigmaT_reduced(xp) + 1e-8 * jnp.eye(3)
        inc = xc - xp - h * F
        L   = jnp.linalg.cholesky(cov)
        v   = jnp.linalg.solve(L, inc)
        return None, 0.5 * (2.0 * jnp.sum(jnp.log(jnp.diag(L))) + jnp.dot(v, v))

    _, nlls = jax.lax.scan(step, None, (Xobs[:-1], Xobs[1:]))
    return jnp.sum(nlls)


# Gaussian approximation
def _LF(x, kappa, K, lam):
    return DF_drift(x, kappa, K, lam) @ F_drift(x, kappa, K, lam) \
           - SigmaSigmaT_reduced(x) @ lam


def _LSST(x, kappa, K, lam):
    F = F_drift(x, kappa, K, lam)
    S = SigmaSigmaT_reduced(x)
    term1 = jnp.diag(F) - jnp.outer(F, x) - jnp.outer(x, F)
    return term1 - S


def _mu_GA(x, h, kappa, K, lam):
    return x + h * F_drift(x, kappa, K, lam) \
             + (h**2 / 2) * _LF(x, kappa, K, lam)


def _Omega_GA(x, h, kappa, K, lam):
    S   = SigmaSigmaT_reduced(x)
    DFx = DF_drift(x, kappa, K, lam)
    LS  = _LSST(x, kappa, K, lam)
    return h * S + (h**2 / 2) * (DFx @ S + S @ DFx.T + LS)


def GA_negloglik(params, Xobs, h):
    kappa, K, lam = _unpack(params)

    def step(carry, obs):
        xp, xc          = obs
        mu               = _mu_GA(xp, h, kappa, K, lam)
        Omega            = _Omega_GA(xp, h, kappa, K, lam)
        z                = xc - mu
        eigvals, eigvecs = jnp.linalg.eigh(Omega)
        eigvals          = jnp.maximum(eigvals, 1e-8)
        Omega_inv        = eigvecs @ jnp.diag(1.0 / eigvals) @ eigvecs.T
        return None, 0.5 * (jnp.sum(jnp.log(eigvals)) + z @ Omega_inv @ z)

    _, nlls = jax.lax.scan(step, None, (Xobs[:-1], Xobs[1:]))
    return jnp.sum(nlls)


# Strang Splitting
def _A_mean(kappa, K, lam, x_bar):
    return K - jnp.dot(x_bar, lam) * jnp.eye(3) - jnp.outer(x_bar, lam)


def _omega_h_precompute(b, A, h):
    d  = 3
    I  = jnp.eye(d)
    At = A - 0.5 * I                               
    SST_b = SigmaSigmaT_reduced(b)

    eAh  = jlinalg.expm(A  * h)
    eAth = jlinalg.expm(At * h)

    c1 = (1.0 - jnp.exp(h)) / jnp.exp(h)          

    AmI     = A - I
    intAmIT = jnp.linalg.solve(AmI.T, jlinalg.expm(AmI.T * h) - I)

    check_beta = jnp.zeros((d*d, d)).at[jnp.array([0, 4, 8]), jnp.arange(d)].set(1.0)
    AtAt     = jnp.kron(At, I) + jnp.kron(I, At)   
    block_J4 = jnp.block([
        [AtAt,                 check_beta        ],  
        [jnp.zeros((d, d*d)),  A                 ], 
    ])
    J4_mat = jlinalg.expm(block_J4 * h)[:d*d, d*d:]  

    block_J5 = jnp.block([
        [-At,                SST_b             ],
        [jnp.zeros((d, d)),  At.T              ],
    ])
    J5 = eAth @ jlinalg.expm(block_J5 * h)[:d, d:]

    return eAh, eAth, c1, intAmIT, J4_mat, J5       


def _omega_h_from_cache(x, b, eAh, c1, intAmIT, J4_mat, J5):
    d  = 3
    dm = x - b
    J1 = c1 * eAh @ jnp.outer(dm, dm) @ eAh.T
    J2 = -eAh @ jnp.outer(dm, b) @ intAmIT
    J3 = J2.T
    J4 = (J4_mat @ dm).reshape(d, d)
    return J1 + J2 + J3 + J4 + J5


def _Nfun(x, kappa, K, lam, A, b):
    return F_drift(x, kappa, K, lam) - A @ (x - b)


def _DNfun(x, lam, A, b):
    dx = x - b
    return -jnp.dot(dx, lam) * jnp.eye(3) - jnp.outer(dx, lam)


def _fh_rk4(x, h, kappa, K, lam, A, b):
    k1 = _Nfun(x,             kappa, K, lam, A, b)
    k2 = _Nfun(x + h/2 * k1,  kappa, K, lam, A, b)
    k3 = _Nfun(x + h/2 * k2,  kappa, K, lam, A, b)
    k4 = _Nfun(x + h   * k3,  kappa, K, lam, A, b)
    return x + h/6 * (k1 + 2*k2 + 2*k3 + k4)


def _Dfh_rk4(x, h, kappa, K, lam, A, b):
    I   = jnp.eye(3)
    k1v = _Nfun(x,              kappa, K, lam, A, b)
    k2v = _Nfun(x + h/2 * k1v,  kappa, K, lam, A, b)
    k3v = _Nfun(x + h/2 * k2v,  kappa, K, lam, A, b)
    Dk1 = _DNfun(x,              lam, A, b)
    Dk2 = _DNfun(x + h/2 * k1v, lam, A, b) @ (I + h/2 * Dk1)
    Dk3 = _DNfun(x + h/2 * k2v, lam, A, b) @ (I + h/2 * Dk2)
    Dk4 = _DNfun(x + h   * k3v, lam, A, b) @ (I + h   * Dk3)
    return I + h/6 * (Dk1 + 2*Dk2 + 2*Dk3 + Dk4)


def _log_det_Dfh_rk4(x, h, kappa, K, lam, A, b):
    return jnp.log(jnp.abs(jnp.linalg.det(_Dfh_rk4(x, h, kappa, K, lam, A, b))))


def _mu_h_linear(x, eAh, b):
    return eAh @ (x - b) + b


def SS_negloglik(params, Xobs, h):
    kappa, K, lam = _unpack(params)
    b    = jnp.mean(Xobs, axis=0)
    A    = _A_mean(kappa, K, lam, b)
    eAh, eAth, c1, intAmIT, J4_mat, J5 = _omega_h_precompute(b, A, h)

    def step(carry, obs):
        xp, xc = obs
        f_fwd = _fh_rk4(xp,  h/2, kappa, K, lam, A, b)  
        f_inv = _fh_rk4(xc, -h/2, kappa, K, lam, A, b)  

        z           = f_inv - _mu_h_linear(f_fwd, eAh, b)
        Omega       = _omega_h_from_cache(f_fwd, b, eAh, c1, intAmIT, J4_mat, J5)
        log_det_jac = _log_det_Dfh_rk4(xc, -h/2, kappa, K, lam, A, b)

        eigvals, eigvecs = jnp.linalg.eigh(Omega)
        eigvals          = jnp.maximum(eigvals, 1e-8)
        Omega_inv        = eigvecs @ jnp.diag(1.0 / eigvals) @ eigvecs.T
        quad             = z @ Omega_inv @ z

        return None, 0.5 * (jnp.sum(jnp.log(eigvals)) + quad) - log_det_jac

    _, nlls = jax.lax.scan(step, None, (Xobs[:-1], Xobs[1:]))
    return jnp.sum(nlls)


# Local Linearization

def _ll_precompute(x, kappa, K, lam, h):
    d = 3
    I = jnp.eye(d)
    A = DF_drift(x, kappa, K, lam)       
    S = SigmaSigmaT_reduced(x)

    C0 = jnp.block([[A, I],
                    [jnp.zeros((d, d)), jnp.zeros((d, d))]])
    R0 = jlinalg.expm(C0 * h)[:d, d:]

    C1 = jnp.block([[A,                I,               jnp.zeros((d, d))],
                    [jnp.zeros((d,d)), A,               I               ],
                    [jnp.zeros((d,d)), jnp.zeros((d,d)), jnp.zeros((d,d))]])
    R1 = jlinalg.expm(C1 * h)[:d, 2*d:]

    Com   = jnp.block([[-A, S], [jnp.zeros((d, d)), A.T]])
    Omega = jlinalg.expm(A * h) @ jlinalg.expm(Com * h)[:d, d:]

    return A, R0, R1, Omega


def _mu_h_LL(x, kappa, K, lam, h, R0, R1):
    F = F_drift(x, kappa, K, lam)
    M = -SigmaSigmaT_reduced(x) @ lam   
    return x + R0 @ F + (h * R0 - R1) @ M


def LL_negloglik(params, Xobs, h):
    kappa, K, lam = _unpack(params)

    def step(carry, obs):
        xp, xc          = obs
        A, R0, R1, Omega = _ll_precompute(xp, kappa, K, lam, h)
        mu               = _mu_h_LL(xp, kappa, K, lam, h, R0, R1)
        z                = xc - mu
        eigvals, eigvecs = jnp.linalg.eigh(Omega)
        eigvals          = jnp.maximum(eigvals, 1e-8)
        Omega_inv        = eigvecs @ jnp.diag(1.0 / eigvals) @ eigvecs.T
        return None, 0.5 * (jnp.sum(jnp.log(eigvals)) + z @ Omega_inv @ z)

    _, nlls = jax.lax.scan(step, None, (Xobs[:-1], Xobs[1:]))
    return jnp.sum(nlls)