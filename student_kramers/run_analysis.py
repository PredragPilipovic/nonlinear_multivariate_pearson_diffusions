"""
run_analysis.py — Load simulation results and reproduce all figures
"""
import os
import jax, jax.numpy as jnp, pandas as pd
jax.config.update("jax_enable_x64", True)

import config
from sde_simulator import simulate_trajectory
from inference_utils import compute_C, get_std_dict, load_and_filter_data
from figures import plot_density_grid, plot_timing

def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    traj = simulate_trajectory(config.TRUE_PARAMS, config.T, config.H_SIM, config.X0)
    C1, C2 = compute_C(traj, config.TRUE_PARAMS)
    std_dicts = {tag: get_std_dict(C1, C2, int(config.T/config.H_VALUES[tag]),
                                   config.H_VALUES[tag])
                 for tag in config.H_TAGS}

    plot_data, timing_rows, summary = [], [], []
    for tag in config.H_TAGS:
        for est in config.ESTIMATORS:
            try:
                df_f, _, n_tot, n_na, n_out, n_fin = load_and_filter_data(est, tag)
                summary.append({"h": tag, "estimator": est, "N_total": n_tot,
                                 "N_NAs": n_na, "N_outliers": n_out, "N_final": n_fin})
                if df_f is None: continue
                for p in config.ALL_PARAMS:
                    if p in df_f.columns:
                        plot_data.append(pd.DataFrame({
                            "parameter": p,
                            "error": (df_f[p]-config.TRUE_PARAMS_DICT[p])/config.TRUE_PARAMS_DICT[p],
                            "estimator": est, "h": tag}))
                timing_rows.append({"estimator": est, "h": tag,
                                    "median_time_sec": df_f["time_sec"].median()})
            except Exception as e:
                print(f"Skipping {est} h={tag}: {e}")

    df_plot = pd.concat(plot_data, ignore_index=True)
    print(pd.DataFrame(summary).to_string(index=False))

    plot_density_grid(df_plot, config.GROUP_DRIFT,
                      r"Densities of estimation errors for potential $U(x)$ of the Student Kramers oscillator",
                      std_dicts=std_dicts,
                      save_path=f"{config.RESULTS_DIR}/density_drift.png")
    plot_density_grid(df_plot, config.GROUP_VOL,
                      r"Densities of estimation errors for damping and diffusion parameters of the Student Kramers oscillator",
                      std_dicts=std_dicts,
                      save_path=f"{config.RESULTS_DIR}/density_diffusion.png")
    plot_timing(pd.DataFrame(timing_rows),
                save_path=f"{config.RESULTS_DIR}/estimation_times.png")

if __name__ == "__main__":
    main()