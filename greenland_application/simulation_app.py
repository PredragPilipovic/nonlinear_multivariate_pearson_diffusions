"""
simulation_app.py — Simulation from the estimated model, KDE prediction bands, waiting times
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "student_kramers"))

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import medfilt

from sde_simulator import make_milstein_step
from config_app import H_OBS, H_SIM, SUBSAMPLE_RATE


def simulate_long(params, n_obs: int, n_rep: int, init_state: jnp.ndarray,
                  h_sim: float = H_SIM, seed: int = 42) -> np.ndarray:
    """
    Simulate one trajectory equivalent to n_rep independent realisations.
    Returns subsampled array of shape (n_obs * n_rep, 2).
    """
    total_steps = (n_obs * n_rep - 1) * SUBSAMPLE_RATE
    step_keys   = jax.random.split(jax.random.PRNGKey(seed), total_steps)
    _, traj     = jax.lax.scan(make_milstein_step(params, h_sim), init_state, step_keys)
    full        = jnp.vstack([init_state, traj])
    return np.array(full[::SUBSAMPLE_RATE, :])


def compute_kde_bands(sim_long: np.ndarray, n_obs: int, n_rep: int,
                      grid: np.ndarray, axis: int = 0):
    """
    Pointwise mean ± 1.96*std of KDE densities across n_rep simulation chunks.
    Returns mean, lower, upper arrays of shape (len(grid),).
    """
    Y = np.zeros((n_rep, len(grid)))
    for i in range(n_rep):
        Y[i] = gaussian_kde(sim_long[i*n_obs:(i+1)*n_obs, axis])(grid)
    mu = Y.mean(axis=0)
    sd = Y.std(axis=0, ddof=1)
    return mu, mu - 1.96*sd, mu + 1.96*sd


def calculate_waiting_times(X: np.ndarray, k: int = 11, level: float = 0.6,
                             h: float = H_OBS):
    """
    Stadial and interstadial waiting times via median-filter threshold crossings.
    Returns stadial_wt, interstadial_wt (in ka), and the smoothed trajectory.
    """
    smoothX    = medfilt(X, kernel_size=k)
    shift_u    = np.diff(np.sign(smoothX - level)) / 2.0
    shift_l    = np.diff(np.sign(smoothX + level)) / 2.0
    upper_up   = np.where(shift_u ==  1)[0]
    upper_down = np.where(shift_u == -1)[0]
    lower_up   = np.where(shift_l ==  1)[0]
    lower_down = np.where(shift_l == -1)[0]

    def _intervals(starts, ends):
        if not len(starts) or not len(ends): return np.array([])
        if starts[0] > ends[0]:             ends   = ends[1:]
        if len(starts) > len(ends):         starts = starts[:len(ends)]
        elif len(ends) > len(starts):       ends   = ends[:len(starts)]
        return (ends - starts) * h

    return (_intervals(lower_down, lower_up),
            _intervals(upper_up, upper_down),
            smoothX)
