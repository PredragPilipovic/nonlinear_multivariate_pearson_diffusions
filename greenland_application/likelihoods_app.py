"""
likelihoods_app.py — SS corrected pseudo-likelihood for the Greenland Ca2+ application

Uses the approximated V-component pseudo-likelihood with a 3/2*h covariance correction
data has shape (N, 2) with columns [X_t, (X_{t+1} - X_t)/h].
"""
import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg


def _A_mat(params, drift_const):
    return jnp.array([[0.0, 1.0], [drift_const, -params[0]]])


def _A_kron_sum(params, drift_const):
    A, I = _A_mat(params, drift_const), jnp.eye(2)
    return jnp.kron(A, I) + jnp.kron(I, A)


def _alpha_mat(params):
    return jnp.zeros((4, 4)).at[3, 3].set(params[5])


def _precompute_matrices(params, drift_const, h):
    A  = _A_mat(params, drift_const)
    Ak = _A_kron_sum(params, drift_const)
    al = _alpha_mat(params)
    eAh = jlinalg.expm(h * A)

    def _block_exp(TL, TR, BR):
        n = TL.shape[0]
        P = jnp.zeros((2*n, 2*n)).at[:n,:n].set(TL).at[:n,n:].set(TR).at[n:,n:].set(BR)
        return jlinalg.expm(h * P)[:n, n:]

    I1    = _block_exp(Ak + al, al, Ak)
    I3    = _block_exp(Ak + al, al, jnp.kron(A, jnp.eye(2)) + jnp.eye(4))
    P4    = (jnp.zeros((6, 6))
             .at[0:4, 0:4].set(Ak + al)
             .at[3, 5].set(params[6])
             .at[4:6, 4:6].set(A))
    I4    = jlinalg.expm(h * P4)[0:4, 4:6]
    I5    = _block_exp(Ak + al, jnp.eye(4), jnp.zeros((4, 4)))
    I5_G5 = I5 @ jnp.array([0.0, 0.0, 0.0, params[7]])
    return eAh, I1, I3, I4, I5_G5


def _step_omega(Y_mid, kappa, I1, I3, I4, I5_G5):
    x0, x1 = Y_mid[0] - kappa, Y_mid[1]
    x_out   = jnp.array([x0*x0, x0*x1, x1*x0, x1*x1])
    xxs     = jnp.array([x0*kappa, 0.0, x1*kappa, 0.0])
    return (I1 @ x_out + 2.0*(I3 @ xxs) + I4 @ jnp.array([x0, x1]) + I5_G5).reshape(2, 2)


def _f_step(Y, h_step, params, drift_const, kappa):
    a, b, c, d = params[1], params[2], params[3], params[4]
    V_new = Y[1] + h_step*(a*Y[0]**3 + b*Y[0]**2 + c*Y[0] + d - drift_const*(Y[0] - kappa))
    return jnp.array([Y[0], V_new])


def _single_step(Y_old, Y_new, cond, params, drift_const, kappa_pos, kappa_neg,
                 mats_h, mats_15h, h):
    kappa               = jnp.where(cond, kappa_pos, kappa_neg)
    eAh, I1, I3, I4, I5_G5 = mats_h
    f_new  = _f_step(Y_new, -h/2., params, drift_const, kappa)
    Y_mid  = _f_step(Y_old,  h/2., params, drift_const, kappa)
    mu     = eAh @ (Y_mid - jnp.array([kappa, 0.])) + jnp.array([kappa, 0.])
    x_val  = (f_new - mu)[1]
    omega  = _step_omega(Y_mid, kappa, I1, I3, I4, I5_G5)[1, 1]
    _, I1_15, I3_15, I4_15, I5_G5_15 = mats_15h
    Y_mid_15 = _f_step(Y_old, 3./4.*h, params, drift_const, kappa)
    omega1   = _step_omega(Y_mid_15, kappa, I1_15, I3_15, I4_15, I5_G5_15)[1, 1]
    return x_val**2 / omega + 2./3. * jnp.log(omega1)


@jax.jit
def SS_quasi_lik(params, data, h):
    a, b, c = params[1], params[2], params[3]
    m     = jnp.mean(data[:-1, 0])
    m2    = jnp.mean(data[:-1, 0]**2)
    var_x = jnp.var(data[:-1, 0], ddof=1)
    dc    = 3*a*m2 + 2*b*m + c
    root  = jnp.sqrt(jnp.maximum((3*a*m + b)**2 + 9*a**2*var_x, 0.))
    k_pos = (-b - root) / (3*a)
    k_neg = (-b + root) / (3*a)
    cond  = data[:-1, 0] > 0
    mh    = _precompute_matrices(params, dc, h)
    m15h  = _precompute_matrices(params, dc, 1.5*h)
    nlls  = jax.vmap(_single_step, in_axes=(0, 0, 0, None, None, None, None, None, None, None))(
        data[:-1], data[1:], cond, params, dc, k_pos, k_neg, mh, m15h, h)
    return jnp.sum(nlls)
