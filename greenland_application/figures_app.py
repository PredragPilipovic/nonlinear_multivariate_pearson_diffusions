"""
figures_app.py — Paper figures for the Greenland Ca2+ application
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config_app import PARAM_LABELS


def _plabel(name: str) -> str:
    return PARAM_LABELS.get(name, name)


def _style(ax):
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.grid(True, color="#E0E0E0", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


def plot_ca2_series(df, save_path: str = None):
    fig = plt.figure(figsize=(14, 5))
    gs      = fig.add_gridspec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax_ts   = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1], sharey=ax_ts)

    ax_ts.plot(df["age_ka"], df["X_Ca2"], color="black", lw=1)
    ax_ts.set_ylabel(r"Centered $-\log(\mathrm{Ca}^{2+})$", fontweight="bold")
    ax_ts.set_xlabel("Age (ka before 2000 AD)", fontweight="bold")
    ax_ts.invert_xaxis()

    ax_hist.hist(df["X_Ca2"], bins=40, density=True, orientation="horizontal",
                 color="#8A0F13", alpha=0.7)
    ax_hist.set_xlabel("Density", fontweight="bold")
    ax_hist.tick_params(axis="y", left=False, labelleft=False)

    for ax in (ax_ts, ax_hist): _style(ax)
    fig.suptitle(r"Greenland Ice Core $-\log(\mathrm{Ca}^{2+})$ with Marginal Density",
                 fontweight="bold", fontsize=13)
    _save(fig, save_path)


def plot_trajectory_fit(X_np, V_np, t_obs,
                        x_grid, pi_x_vals, pi_x_low, pi_x_upp,
                        v_grid, pi_v_vals, pi_v_low, pi_v_upp,
                        x_grid_emp=None, v_grid_emp=None,
                        sim_X=None, sim_V=None,
                        save_path: str = None):
    from scipy.stats import gaussian_kde

    fig = plt.figure(figsize=(13, 12))
    fig.suptitle(r"Student Kramers oscillator fitted on Ice core data from Greenland",
                 fontweight="bold", fontsize=14, y=0.94)

    gs    = fig.add_gridspec(5, 2, width_ratios=[3, 1],
                              height_ratios=[1, 0.05, 1, 0.35, 1.5],
                              wspace=0.05, hspace=0.0, top=0.9)
    ax_x  = fig.add_subplot(gs[0, 0])
    ax_xd = fig.add_subplot(gs[0, 1], sharey=ax_x)
    ax_v  = fig.add_subplot(gs[2, 0])
    ax_vd = fig.add_subplot(gs[2, 1], sharey=ax_v)

    tick_pos    = np.array([0, 10, 20, 30, 40, 50])
    tick_labels = [str(80 - int(t)) for t in tick_pos]

    ax_x.plot(t_obs, X_np, color="black", label="Data")
    if sim_X is not None:
        ax_x.plot(t_obs, sim_X, color="#8A0F13", alpha=0.8, lw=1.5)
    ax_x.set_ylabel(r"$X_{t_k} := -\log(\mathrm{Ca}^{2+})$", fontweight="bold", fontsize=12)
    ax_x.tick_params(axis="x", bottom=False, labelbottom=False)
    ax_x.tick_params(axis="y", left=False, labelleft=True)

    ax_v.plot(t_obs, V_np, color="black")
    if sim_V is not None:
        ax_v.plot(t_obs, sim_V, color="#8A0F13", alpha=0.8, lw=1)
    ax_v.set_ylabel(r"$(X_{t_{k+1}} - X_{t_k})/h$", fontweight="bold", fontsize=12)
    ax_v.set_xlabel("Time before 2000 AD [ka]", fontweight="bold", fontsize=12)
    ax_v.set_xticks(tick_pos)
    ax_v.set_xticklabels(tick_labels)
    ax_v.tick_params(axis="y", left=False, labelleft=True)
    ax_v.tick_params(axis="x", bottom=False, labelbottom=True)

    ax_xd.hist(X_np, bins=30, density=True, orientation="horizontal",
               color="black", alpha=0.6)
    if pi_x_low is not None and x_grid_emp is not None:
        ax_xd.fill_betweenx(x_grid_emp, pi_x_low, pi_x_upp, color="#8A0F13", alpha=0.4)
    ax_xd.plot(pi_x_vals, x_grid, color="black", lw=1.5)
    if sim_X is not None and x_grid_emp is not None:
        ax_xd.plot(gaussian_kde(sim_X)(x_grid_emp), x_grid_emp, color="#8A0F13", lw=1.5)
    ax_xd.tick_params(axis="y", left=False, labelleft=False)
    ax_xd.tick_params(axis="x", top=False, bottom=False, labeltop=True, labelbottom=False)
    ax_xd.xaxis.set_label_position("top")

    ax_vd.hist(V_np, bins=30, density=True, orientation="horizontal",
               color="black", alpha=0.6)
    if pi_v_low is not None and v_grid_emp is not None:
        ax_vd.fill_betweenx(v_grid_emp, pi_v_low, pi_v_upp, color="#8A0F13", alpha=0.4)
    ax_vd.plot(pi_v_vals, v_grid, color="black", lw=1.5)
    if sim_V is not None and v_grid_emp is not None:
        ax_vd.plot(gaussian_kde(sim_V)(v_grid_emp), v_grid_emp, color="#8A0F13", lw=1.5)
    ax_vd.tick_params(axis="y", left=False, labelleft=False)
    ax_vd.set_xlabel("Density", fontweight="bold", fontsize=12)
    ax_vd.tick_params(axis="x", bottom=False, labelbottom=True)

    for ax in (ax_x, ax_xd, ax_v, ax_vd): _style(ax)
    _save(fig, save_path)


def plot_waiting_times(stad_middle, inter_middle, stad_small, inter_small,
                       stad_data, inter_data, xlim: int = 8000,
                       save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    pairs = [
        (axes[0], stad_middle, stad_small, stad_data, "Stadial (Cold) Occupancy Times"),
        (axes[1], inter_middle, inter_small, inter_data, "Interstadial (Warm) Occupancy Times"),
    ]
    for ax, mid, sml, dat, title in pairs:
        sns.kdeplot(mid*1000, color="#8A0F13", fill=True, alpha=0.4, label="Middle model", ax=ax)
        sns.kdeplot(sml*1000, color="#1f77b4", fill=True, alpha=0.4, label="Small model",  ax=ax)
        sns.histplot(dat*1000, stat="density", color="black", alpha=0.6, label="Data", ax=ax)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Time (years)", fontweight="bold")
        ax.set_xlim(0, xlim)
        ax.legend(frameon=False)
        _style(ax)
    fig.suptitle("Waiting Time Distributions", fontweight="bold", fontsize=13)
    plt.tight_layout()
    _save(fig, save_path)


def plot_bootstrap_distribution(LR_values: np.ndarray, obs_LR: float,
                                 save_path: str = None):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(LR_values, density=True, color="#8A0F13", alpha=0.7,
            label="Bootstrap LR")
    ax.axvline(obs_LR, color="black", lw=2, linestyle="--",
               label=f"Observed LR = {obs_LR:.1f}")
    ax.set_xlabel("Likelihood Ratio Statistic", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.legend(frameon=False)
    _style(ax)
    fig.suptitle("Parametric Bootstrap Distribution of LR Statistic",
                 fontweight="bold", fontsize=13)
    _save(fig, save_path)


def plot_large_model(large_boot: np.ndarray, params_large: np.ndarray,
                     param_names: list, save_path: str = None):
    n_params = len(param_names)
    ncols    = 4
    nrows    = int(np.ceil(n_params / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    for col, (ax, name, est) in enumerate(zip(axes, param_names, params_large)):
        label = _plabel(name)                   # safe display label, no raw underscore in math
        ax.hist(large_boot[:, col], bins=30, density=True, color="#8A0F13", alpha=0.7)
        ax.axvline(0.0, color="black",   lw=2, linestyle="--", label="0")
        ax.axvline(est, color="#1f77b4", lw=2, label=f"Est = {est:.3f}")
        ax.set_title(label, fontweight="bold")
        ax.set_xlabel(label, fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.legend(frameon=False, fontsize=8)
        _style(ax)

    for ax in axes[n_params:]:
        ax.set_visible(False)

    fig.suptitle("Large model bootstrap parameter distributions",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    _save(fig, save_path)