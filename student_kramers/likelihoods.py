"""
likelihoods.py — Four pseudo-log-likelihoods for the Student Kramers oscillator

  EM  Euler-Maruyama
  GA  Gaussian Approximation 
  LL  Local Linearization 
  SS  Strang Splitting 

All have scalar signature: neg_log_lik(params, data, h)
data has shape (N+1, 2) with columns [X, V].
"""
import jax, jax.numpy as jnp
import jax.scipy.linalg as jlinalg
from sde_simulator import F_drift, DF_drift


# Euler-Maruyama
@jax.jit
def EM_neg_log_lik(params, data, h):
    """V_{t+h}|(X_t,V_t) ~ N(v + h*f_v, h*sigma^2(v))."""
    eta, a, b, c, d, alpha, beta, gamma = params
    Y_old, Y_new = data[:-1], data[1:]
    x_old, v_old = Y_old[:, 0], Y_old[:, 1]
    v_new        = Y_new[:, 1]
    mu_v  = v_old + h*(a*x_old**3 + b*x_old**2 + c*x_old + d - eta*v_old)
    Omega = h*(alpha*v_old**2 + beta*v_old + gamma)
    nlls  = (v_new - mu_v)**2 / Omega + jnp.log(Omega)
    return 0.5*jnp.sum(nlls)


# Gaussian Approximation 
def _LF(Y, params):
    return DF_drift(Y, params) @ F_drift(Y, params)

def _mu_GA(Y, h, params):
    return Y + h*F_drift(Y, params) + (h**2/2.0)*_LF(Y, params)

def _Omega_GA(Y, h, params):
    """Second-order Ito-Taylor conditional covariance (O(h^3))."""
    eta, a, b, c, d, alpha, beta, gamma = params
    x, v = Y[0], Y[1]
    sigma2 = v*(v*alpha + beta) + gamma
    C11 = (1.0/3.0)*sigma2*h**3
    shared = (2*b*v*x**2*alpha + 2*a*v*x**3*alpha + v**2*alpha**2
              + b*x**2*beta + a*x**3*beta + v*alpha*beta
              + d*(2*v*alpha + beta) + c*x*(2*v*alpha + beta)
              + alpha*gamma - (5*v**2*alpha + 4*v*beta + 3*gamma)*eta)
    C12 = 0.5*sigma2*h**2 + (1.0/6.0)*shared*h**3
    C22 = (sigma2*h
           + 0.5*(2*b*v*x**2*alpha + 2*a*v*x**3*alpha + v**2*alpha**2
                  + b*x**2*beta + a*x**3*beta + v*alpha*beta
                  + d*(2*v*alpha + beta) + c*x*(2*v*alpha + beta)
                  + alpha*gamma - (4*v**2*alpha + 3*v*beta + 2*gamma)*eta)*h**2
           + (1.0/6.0)*(
               2*d**2*alpha + 8*b*v**2*x*alpha + 2*c**2*x**2*alpha
               + 12*a*v**2*x**2*alpha + 2*b**2*x**4*alpha
               + 4*a*b*x**5*alpha + 2*a**2*x**6*alpha
               + 2*b*v*x**2*alpha**2 + 2*a*v*x**3*alpha**2 + v**2*alpha**3
               + 6*b*v*x*beta + 9*a*v*x**2*beta + b*x**2*alpha*beta
               + a*x**3*alpha*beta + v*alpha**2*beta
               + d*alpha*(4*x*(c + x*(b + a*x)) + 2*v*alpha + beta)
               + 4*b*x*gamma + 6*a*x**2*gamma + alpha**2*gamma
               - d*(10*v*alpha + 3*beta)*eta
               - (b*x**2*(10*v*alpha + 3*beta) + a*x**3*(10*v*alpha + 3*beta)
                  + alpha*(6*v**2*alpha + 5*v*beta + 4*gamma))*eta
               + (12*v**2*alpha + 7*v*beta + 4*gamma)*eta**2
               + c*(4*v**2*alpha + 4*x**3*(b + a*x)*alpha + 3*v*beta
                    + x*alpha*beta + 2*gamma
                    + 2*v*x*alpha*(alpha - 5*eta) - 3*x*beta*eta)
           )*h**3)
    return jnp.array([[C11, C12], [C12, C22]])

def _step_GA(Y_prev, Y_curr, h, params):
    mu    = _mu_GA(Y_prev, h, params)
    Omega = _Omega_GA(Y_prev, h, params)
    z     = Y_curr - mu
    a11, a12, a21, a22 = Omega[0,0], Omega[0,1], Omega[1,0], Omega[1,1]
    det       = jnp.maximum(a11*a22 - a12*a21, 1e-8)
    quad_form = z @ (jnp.array([[a22, -a12], [-a21, a11]]) / det) @ z
    return 0.5*(jnp.log(det) + quad_form)

@jax.jit
def GA_neg_log_lik(params, data, h):
    """Gaussian Approximation (second-order Ito-Taylor) negative log-likelihood."""
    nlls = jax.vmap(_step_GA, in_axes=(0, 0, None, None))(data[:-1], data[1:], h, params)
    return jnp.sum(nlls)


# Local Linearization 
def _lamperti_transform(Y, params):
    """V -> U: unit diffusion via arcsinh transform."""
    alpha, beta, gamma = params[5], params[6], params[7]
    X, V = Y[0], Y[1]
    sq_alpha = jnp.sqrt(alpha)
    denom    = jnp.sqrt(4*alpha*gamma - beta**2)
    U = (1.0/sq_alpha)*jnp.arcsinh((2*alpha*V + beta)/denom)
    return jnp.array([X, U])

def _FdriftLL(Y_trans, params):
    eta, a, b, c, d, alpha, beta, gamma = params
    x, u = Y_trans[0], Y_trans[1]
    sq_alpha = jnp.sqrt(alpha)
    c_term   = jnp.sqrt(gamma - beta**2/(4*alpha))
    X_val = c_term*jnp.sinh(sq_alpha*u)/sq_alpha - beta/(2*alpha)
    U_val = (-(eta + alpha/2.0)*jnp.tanh(sq_alpha*u)/sq_alpha
             + (eta*beta/(2*alpha) + a*x**3 + b*x**2 + c*x + d)
               /(c_term*jnp.cosh(sq_alpha*u)))
    return jnp.array([X_val, U_val])

def _DFdriftLL(Y_trans, params):
    eta, a, b, c, d, alpha, beta, gamma = params
    x, u = Y_trans[0], Y_trans[1]
    sq_alpha = jnp.sqrt(alpha)
    c_term   = jnp.sqrt(gamma - beta**2/(4*alpha))
    cosh_u, tanh_u = jnp.cosh(sq_alpha*u), jnp.tanh(sq_alpha*u)
    df12 = c_term*cosh_u
    df21 = (3*a*x**2 + 2*b*x + c)/(c_term*cosh_u)
    df22 = (-(eta + alpha/2.0)/cosh_u**2
            - sq_alpha*(a*x**3 + b*x**2 + c*x + d + beta*eta/(2*alpha))
              *tanh_u/(c_term*cosh_u))
    return jnp.array([[0.0, df12], [df21, df22]])

def _step_LL(Y_old, Y_new, V_new_raw, h, params):
    """LL contribution: locally linearised drift + Lamperti Jacobian correction."""
    eta, a, b, c, d, alpha, beta, gamma = params
    x, u = Y_old[0], Y_old[1]
    sq_alpha = jnp.sqrt(alpha)
    c_term   = jnp.sqrt(gamma - beta**2/(4*alpha))
    DF    = _DFdriftLL(Y_old, params)
    F_val = _FdriftLL(Y_old, params)
    M_vec = jnp.array([
        sq_alpha*c_term*jnp.sinh(u*sq_alpha),
        (2*(eta + alpha/2)*sq_alpha*jnp.tanh(u*sq_alpha)/jnp.cosh(u*sq_alpha)**2
         + (a*x**3 + b*x**2 + c*x + d + beta*eta/(2*alpha))
         * (-alpha/jnp.cosh(sq_alpha*u)**3
            + alpha/jnp.cosh(sq_alpha*u)*jnp.tanh(sq_alpha*u)**2)/c_term),
    ])
    M1 = jnp.block([[jnp.zeros((2,2)), jnp.eye(2)],
                    [jnp.zeros((2,2)), DF]])
    r0  = jlinalg.expm(h*M1)[0:2, 2:4]
    M2  = jnp.block([[-DF, jnp.eye(2), jnp.zeros((2,2))],
                     [jnp.zeros((2,2)), jnp.zeros((2,2)), jnp.eye(2)],
                     [jnp.zeros((2,2)), jnp.zeros((2,2)), jnp.zeros((2,2))]])
    H1  = jlinalg.expm(h*M2)[0:2, 4:6]
    r1  = jlinalg.expm(h*DF) @ H1
    mu  = Y_old + r0@F_val + (h*r0 - r1)@M_vec
    M3  = jnp.block([[DF, jnp.array([[0.,0.],[0.,1.]])],
                     [jnp.zeros((2,2)), -DF.T]])
    exphM3 = jlinalg.expm(h*M3)
    Omega  = exphM3[0:2, 2:4] @ exphM3[0:2, 0:2].T
    z = Y_new - mu
    a11, a12, a21, a22 = Omega[0,0], Omega[0,1], Omega[1,0], Omega[1,1]
    det       = jnp.maximum(a11*a22 - a12*a21, 1e-8)
    quad_form = (a22*z[0]**2 - (a12+a21)*z[0]*z[1] + a11*z[1]**2)/det
    log_jac   = jnp.log(alpha*V_new_raw**2 + beta*V_new_raw + gamma)
    return quad_form + jnp.log(det) + log_jac

@jax.jit
def LL_neg_log_lik(params, data, h):
    """Local Linearisation (Lamperti) negative log-likelihood."""
    data_trans = jax.vmap(_lamperti_transform, in_axes=(0, None))(data, params)
    nlls = jax.vmap(_step_LL, in_axes=(0, 0, 0, None, None))(
        data_trans[:-1], data_trans[1:], data[1:, 1], h, params)
    return 0.5*jnp.sum(nlls)


# Strang Splitting
def _precompute_SS_matrices(params, drift_const, h):
    """Pre-compute matrix-exponential integrals for the SS likelihood."""
    A  = jnp.array([[0.0, 1.0], [drift_const, -params[0]]])
    I2 = jnp.eye(2)
    Ak = jnp.kron(A, I2) + jnp.kron(I2, A)
    al = jnp.zeros((4,4)).at[3,3].set(params[5])
    eAh = jlinalg.expm(h*A)

    def _block(TL, TR, BR, n=8):
        P = jnp.zeros((n, n))
        s = TL.shape[0]
        P = P.at[0:s, 0:s].set(TL)
        P = P.at[0:s, s:].set(TR)
        P = P.at[s:, s:].set(BR)
        return jlinalg.expm(h*P)[0:s, s:]

    I1    = _block(Ak + al, al, Ak)
    I3    = _block(Ak + al, al, jnp.kron(A, I2) + jnp.eye(4))
    I5    = _block(Ak + al, jnp.eye(4), jnp.zeros((4,4)))
    I5_G5 = I5 @ jnp.array([0., 0., 0., params[7]])
    P4 = jnp.zeros((6,6))
    P4 = P4.at[0:4,0:4].set(Ak + al)
    P4 = P4.at[3,5].set(params[6])
    P4 = P4.at[4:6,4:6].set(A)
    I4 = jlinalg.expm(h*P4)[0:4, 4:6]
    return eAh, I1, I3, I4, I5_G5

def _step_Omega_SS(Y_mid, kappa, I1, I3, I4, I5_G5):
    x0, x1 = Y_mid[0] - kappa, Y_mid[1]
    x_out  = jnp.array([x0*x0, x0*x1, x1*x0, x1*x1])
    xxs    = jnp.array([x0*kappa, 0., x1*kappa, 0.])
    return (I1@x_out + 2.*(I3@xxs) + I4@jnp.array([x0, x1]) + I5_G5).reshape((2,2))

def _f_step_SS(Y, h_step, params, drift_const, kappa):
    a, b, c, d = params[1], params[2], params[3], params[4]
    X, V = Y[0], Y[1]
    return jnp.array([X, V + h_step*(a*X**3 + b*X**2 + c*X + d - drift_const*(X - kappa))])

def _step_SS(Y_old, Y_new, cond_val, h, params, drift_const, kappa_pos, kappa_neg, I_mats):
    eAh, I1, I3, I4, I5_G5 = I_mats
    kappa  = jnp.where(cond_val, kappa_pos, kappa_neg)
    f_new  = _f_step_SS(Y_new, -h/2., params, drift_const, kappa)
    Y_mid  = _f_step_SS(Y_old,  h/2., params, drift_const, kappa)
    xs     = jnp.array([kappa, 0.])
    mu     = eAh@(Y_mid - xs) + xs
    z      = f_new - mu
    Omega  = _step_Omega_SS(Y_mid, kappa, I1, I3, I4, I5_G5)
    a11, a12, a21, a22 = Omega[0,0], Omega[0,1], Omega[1,0], Omega[1,1]
    det       = jnp.maximum(a11*a22 - a12*a21, 1e-8)
    quad_form = (a22*z[0]**2 - (a12+a21)*z[0]*z[1] + a11*z[1]**2)/det
    return 0.5*(jnp.log(det) + quad_form)

@jax.jit
def SS_neg_log_lik(params, data, h):
    """Strang Splitting negative log-likelihood."""
    a, b, c = params[1], params[2], params[3]
    data_old, data_new = data[:-1], data[1:]
    cond  = data_old[:, 0] > 0
    m     = jnp.mean(data_old[:, 0])
    m2    = jnp.mean(data_old[:, 0]**2)
    var_x = jnp.var(data_old[:, 0], ddof=1)
    drift_const = 3*a*m2 + 2*b*m + c
    inner       = jnp.maximum((3*a*m + b)**2 + 9*a**2*var_x, 0.)
    root        = jnp.sqrt(inner)
    kappa_pos, kappa_neg = (-b - root)/(3*a), (-b + root)/(3*a)
    I_mats = _precompute_SS_matrices(params, drift_const, h)
    nlls = jax.vmap(_step_SS, in_axes=(0,0,0,None,None,None,None,None,None))(
        data_old, data_new, cond, h, params,
        drift_const, kappa_pos, kappa_neg, I_mats)
    return jnp.sum(nlls)