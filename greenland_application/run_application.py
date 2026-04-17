"""
run_application.py — Main execution script for the Greenland Ca2+ application

Execution order:
  1. Load and preprocess Ca2+ data
  2. Estimate middle, small, and large models; print params + NLL
  3. Long validation simulation with KDE prediction bands
  4. Waiting time distributions
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "student_kramers"))

import os
import numpy as np
import jax.numpy as jnp

import config_app as cfg
from data_loading import load_ca2, build_data_matrix
from estimation_app import estimate_model, compute_derived_params, make_loss_fn
from simulation_app import simulate_long, compute_kde_bands, calculate_waiting_times
from figures_app import plot_ca2_series, plot_trajectory_fit, plot_waiting_times
from sde_simulator import make_stationary_x_density, make_stationary_v_density

os.makedirs(cfg.RESULTS_DIR, exist_ok=True)


def _print_model(label, params, nll=None):
    print(f"\n=== {label} ===")
    for name, val in zip(cfg.PARAM_NAMES, params):
        print(f"  {name:7s}: {val:10.4f}")
    if nll is not None:
        print(f"  {'NLL':7s}: {float(nll):10.4f}")


def main():
    df   = load_ca2()
    Ca2  = df["X_Ca2"].values[::-1]
    X_np = Ca2[:-1]
    V_np = np.diff(Ca2) / cfg.H_OBS
    t_obs = np.arange(len(X_np)) * cfg.H_OBS
    data  = build_data_matrix(Ca2, cfg.H_OBS)

    plot_ca2_series(df, save_path=f"{cfg.RESULTS_DIR}/ca2_series.png")

    params_middle, nll_middle, _ = estimate_model("middle", data, cfg.H_OBS, verbose=True)
    params_small,  nll_small,  _ = estimate_model("small",  data, cfg.H_OBS, verbose=True)
    params_large,  nll_large,  _ = estimate_model("large",  data, cfg.H_OBS, verbose=True)

    _print_model("Middle Model", params_middle, nll_middle)
    derived = compute_derived_params(params_middle)
    print(f"  nu={derived['nu']:.4f}, mu={derived['mu']:.4f}, "
          f"nu*sigma2={derived['nu_sigma2']:.4f}, omega={derived['omega']:.4f}")

    _print_model("Small Model",  params_small,  nll_small)
    _print_model("Large Model",  params_large,  nll_large)

    print(f"\n  NLL comparison — Middle: {float(nll_middle):.4f}  "
          f"Small: {float(nll_small):.4f}  Large: {float(nll_large):.4f}")

    n_obs, n_rep = len(X_np) + 1, 1000
    init_state   = jnp.array([float(X_np[0]), float(V_np[0])])
    sim_middle = simulate_long(jnp.array(params_middle), n_obs, n_rep, init_state,
                               seed=cfg.SIM_SEED_MIDDLE)
    sim_small  = simulate_long(cfg.SMALL_MODEL_PARAMS, n_obs, n_rep,
                               init_state, seed=cfg.SIM_SEED_SMALL)

    x_grid_emp = np.linspace(-2.4,  2.4, 512)
    v_grid_emp = np.linspace(-50., 50., 512)
    _, pi_x_low, pi_x_upp = compute_kde_bands(sim_middle, n_obs, n_rep, x_grid_emp, 0)
    _, pi_v_low, pi_v_upp = compute_kde_bands(sim_middle, n_obs, n_rep, v_grid_emp, 1)

    pi_x_fn   = make_stationary_x_density(jnp.array(params_middle))
    pi_v_fn   = make_stationary_v_density(jnp.array(params_middle))
    x_grid    = np.linspace(X_np.min(), X_np.max(), 300)
    v_grid    = np.linspace(V_np.min(), V_np.max(), 300)
    pi_x_vals = np.array([pi_x_fn(xi) for xi in x_grid])
    pi_v_vals = np.array([pi_v_fn(vi) for vi in v_grid])

    l = cfg.SIM_TRAJ_OVERLAY
    plot_trajectory_fit(
        X_np, V_np, t_obs,
        x_grid, pi_x_vals, pi_x_low, pi_x_upp,
        v_grid, pi_v_vals, pi_v_low, pi_v_upp,
        x_grid_emp=x_grid_emp, v_grid_emp=v_grid_emp,
        sim_X=sim_middle[(l*n_obs):(l*n_obs + len(X_np)), 0],
        sim_V=sim_middle[(l*n_obs):(l*n_obs + len(X_np)), 1],
        save_path=f"{cfg.RESULTS_DIR}/ca2_fit.png")

    k, lvl = cfg.K, cfg.LVL
    stad_d, inter_d, _ = calculate_waiting_times(X_np,             k=k, level=lvl)
    stad_m, inter_m, _ = calculate_waiting_times(sim_middle[:, 0], k=k, level=lvl)
    stad_s, inter_s, _ = calculate_waiting_times(sim_small[:, 0],  k=k, level=lvl)

    print("\n=== Mean Occupancy Times (years) ===")
    print(f"  Data   — Stadial: {np.mean(stad_d)*1000:.0f}  "
          f"Interstadial: {np.mean(inter_d)*1000:.0f}")
    print(f"  Middle — Stadial: {np.mean(stad_m)*1000:.0f}  "
          f"Interstadial: {np.mean(inter_m)*1000:.0f}")
    print(f"  Small  — Stadial: {np.mean(stad_s)*1000:.0f}  "
          f"Interstadial: {np.mean(inter_s)*1000:.0f}")

    plot_waiting_times(stad_m, inter_m, stad_s, inter_s, stad_d, inter_d,
                       save_path=f"{cfg.RESULTS_DIR}/waiting_times.png")


if __name__ == "__main__":
    main()