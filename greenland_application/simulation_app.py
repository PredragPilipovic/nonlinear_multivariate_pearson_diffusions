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
    Stadial and interstadial waiting times via median-filter threshold crossings,

    Returns
    -------
    stadial_wt       : np.ndarray  – stadial waiting times in ka
    interstadial_wt  : np.ndarray  – interstadial waiting times in ka
    smoothX          : np.ndarray  – median-filtered trajectory
    upper_up         : np.ndarray  – indices of upward crossings of +level
    upper_down       : np.ndarray  – indices of downward crossings of +level
    lower_up         : np.ndarray  – indices of upward crossings of -level
    lower_down       : np.ndarray  – indices of downward crossings of -level
    """
    smoothX = medfilt(X, kernel_size=k)

    shift_upp = np.diff(np.sign(smoothX - level)) / 2.0
    shift_low = np.diff(np.sign(smoothX + level)) / 2.0

    # Remove downward crossings of -level that have no intervening ±level crossing
    tmp_low = shift_low.copy()
    prev_point = 0
    for i in np.where(shift_low == -1)[0]:
        if np.sum(np.abs(shift_upp[prev_point:i])) == 0:
            tmp_low[i] = 0
        prev_point = i  # unconditional, mirrors R's prev.point <- i

    # Backwards alternation enforcer: remove consecutive same-sign crossings
    one_indicator, one_flag = 0, False
    for i in np.where(tmp_low != 0)[0][::-1]:
        if one_indicator + tmp_low[i] != 2:
            one_flag = False
        if one_indicator + tmp_low[i] == 2:
            tmp_low[i] = 0
            one_flag = True
        if not one_flag:
            one_indicator = tmp_low[i]

    lower_up   = np.where(tmp_low ==  1)[0]
    lower_down = np.where(tmp_low == -1)[0]

    # Pair down→up crossings of -level to form stadial intervals
    if len(lower_up) and len(lower_down):
        if lower_up[0] > lower_down[0]:
            n = min(len(lower_up), len(lower_down))
            stadial_wt = (lower_up[:n] - lower_down[:n]) * h
        else:
            n = min(len(lower_up) - 1, len(lower_down))
            stadial_wt = (lower_up[1:n + 1] - lower_down[:n]) * h
    else:
        stadial_wt = np.array([])

    # Remove upward crossings of +level that have no intervening ±level crossing
    tmp_upp = shift_upp.copy()
    prev_point = 0
    for i in np.where(shift_upp == 1)[0]:
        if np.sum(np.abs(shift_low[prev_point:i])) == 0:
            tmp_upp[i] = 0
        prev_point = i  # unconditional

    # Backwards alternation enforcer
    one_indicator, one_flag = 0, False
    for i in np.where(tmp_upp != 0)[0][::-1]:
        if one_indicator + tmp_upp[i] != -2:
            one_flag = False
        if one_indicator + tmp_upp[i] == -2:
            tmp_upp[i] = 0
            one_flag = True
        if not one_flag:
            one_indicator = tmp_upp[i]

    upper_up   = np.where(tmp_upp ==  1)[0]
    upper_down = np.where(tmp_upp == -1)[0]

    # Pair up→down crossings of +level to form interstadial intervals
    if len(upper_up) and len(upper_down):
        if upper_up[0] < upper_down[0]:
            n = min(len(upper_up), len(upper_down))
            interstadial_wt = (upper_down[:n] - upper_up[:n]) * h
        else:
            n = min(len(upper_up), len(upper_down) - 1)
            interstadial_wt = (upper_down[1:n + 1] - upper_up[:n]) * h
    else:
        interstadial_wt = np.array([])

    return (stadial_wt, interstadial_wt, smoothX,
            upper_up, upper_down, lower_up, lower_down)