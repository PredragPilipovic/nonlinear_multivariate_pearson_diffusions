"""
sde_simulator.py — SDE components for the Student Kramers oscillator

    dX = V dt
    dV = [-eta*V + a*X^3 + b*X^2 + c*X + d] dt + sqrt(alpha*V^2 + beta*V + gamma) dW
"""
import jax, jax.numpy as jnp
from jax import random
from scipy.integrate import quad


def F_drift(Y, params):
    """Drift vector [v, -eta*v + a*x^3 + b*x^2 + c*x + d]."""
    x, v = Y[0], Y[1]
    eta, a, b, c, d = params[0], params[1], params[2], params[3], params[4]
    return jnp.array([v, -eta*v + a*x**3 + b*x**2 + c*x + d])


def DF_drift(Y, params):
    """Jacobian of drift (2x2)."""
    x = Y[0]
    eta, a, b, c = params[0], params[1], params[2], params[3]
    return jnp.array([[0.0, 1.0], [3*a*x**2 + 2*b*x + c, -eta]])


def sigma_scalar(v, params):
    """sigma(v) = sqrt(alpha*v^2 + beta*v + gamma)."""
    alpha, beta, gamma = params[5], params[6], params[7]
    return jnp.sqrt(alpha*v**2 + beta*v + gamma)


def make_milstein_step(params, h):
    """
    Returns a JAX-scannable Milstein step for jax.lax.scan.
    Milstein correction term: 0.25*(2*alpha*v + beta)*(dW^2 - h).
    """
    @jax.jit
    def step(xv, key):
        x, v = xv[0], xv[1]
        dW = jnp.sqrt(h) * random.normal(key)
        eta, a, b, c, d, alpha, beta, gamma = (
            params[0], params[1], params[2], params[3], params[4],
            params[5], params[6], params[7])
        sig   = jnp.sqrt(alpha*v**2 + beta*v + gamma)
        x_new = x + h*v
        v_new = (v + h*(-eta*v + a*x**3 + b*x**2 + c*x + d)
                 + sig*dW + 0.25*(2.0*alpha*v + beta)*(dW**2 - h))
        xv_new = jnp.array([x_new, v_new])
        return xv_new, xv_new
    return step


def simulate_trajectory(params, T, h_sim, x0, seed=42):
    """Simulate one trajectory on the fine grid. Returns shape (N_sim+1, 2)."""
    N_sim = int(T / h_sim)
    key   = random.PRNGKey(seed)
    keys  = random.split(key, N_sim)
    _, xs = jax.lax.scan(make_milstein_step(params, h_sim), x0, keys)
    return jnp.concatenate([x0[None], xs], axis=0)


def subsample(traj, h_sim, h_obs):
    """Subsample fine-grid trajectory to the observation grid."""
    return traj[::int(h_obs / h_sim)]


def _potential(x, params):
    a, b, c, d = params[1], params[2], params[3], params[4]
    return -(a*x**4/4.0 + b*x**3/3.0 + c*x**2/2.0 + d*x)


def make_stationary_x_density(params, x_lo=-3.0, x_hi=3.0):
    """Normalised X-marginal: pi_X propto exp(-U(x)*(2*eta - alpha)/gamma)."""
    eta, alpha, gamma = float(params[0]), float(params[5]), float(params[7])
    def unnorm(x):
        return float(jnp.exp(-_potential(x, params) * (2*eta - alpha) / gamma))
    Z, _ = quad(unnorm, x_lo, x_hi)
    return lambda x: unnorm(x) / Z


def make_stationary_v_density(params):
    """Normalised V-marginal (skew-t family)."""
    eta, alpha, beta, gamma = (float(params[0]), float(params[5]),
                                float(params[6]), float(params[7]))
    sqrt_disc = float(jnp.sqrt(4*alpha*gamma - beta**2))
    def unnorm(v):
        poly = (alpha*v**2 + beta*v + gamma) ** -(eta/alpha + 1)
        exp  = jnp.exp((2*beta*eta)/(alpha*sqrt_disc)
                       * jnp.arctan((2*alpha*v + beta)/sqrt_disc))
        return float(poly * exp)
    Z, _ = quad(unnorm, -float("inf"), float("inf"))
    return lambda v: unnorm(v) / Z