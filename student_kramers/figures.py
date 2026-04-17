"""
figures.py — Trajectory plot, density grid, timing bar chart
"""
import os, numpy as np
import matplotlib.pyplot as plt, matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import norm as sp_norm
from config import ESTIMATORS, H_TAGS, H_LABELS, PALETTE, LATEX_MAP


def plot_trajectory(data, t_obs, pi_x, pi_v, save_path=None):
    X_np, V_np, t_np = np.array(data[:,0]), np.array(data[:,1]), np.array(t_obs)
    xg = np.linspace(X_np.min(), X_np.max(), 300)
    vg = np.linspace(V_np.min(), V_np.max(), 300)
    fig = plt.figure(figsize=(13, 12))
    fig.suptitle("A trajectory of Student Kramers oscillator",
                 fontweight="bold", fontsize=14, y=0.94)
    gs   = fig.add_gridspec(5, 2, width_ratios=[3,1],
                            height_ratios=[1,0.05,1,0.35,1.5],
                            wspace=0.05, hspace=0.0, top=0.91)
    ax_x  = fig.add_subplot(gs[0,0])
    ax_xd = fig.add_subplot(gs[0,1], sharey=ax_x)
    ax_v  = fig.add_subplot(gs[2,0])
    ax_vd = fig.add_subplot(gs[2,1], sharey=ax_v)
    ax_x.plot(t_np, X_np, color="black")
    ax_x.set_ylabel(r"$X_t$", fontweight="bold", fontsize=12)
    ax_x.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_v.plot(t_np, V_np, color="black")
    ax_v.set_ylabel(r"$V_t$", fontweight="bold", fontsize=12)
    ax_v.set_xlabel("Time", fontweight="bold", fontsize=12)
    ax_xd.hist(X_np, bins=60, density=True, orientation="horizontal",
               color="#8A0F13", alpha=0.7)
    ax_xd.plot([pi_x(xi) for xi in xg], xg, color="black", lw=1.5)
    ax_xd.tick_params(axis="y", left=False, labelleft=False)
    ax_xd.tick_params(axis="x", top=True, bottom=False, labeltop=True, labelbottom=False)
    ax_vd.hist(V_np, bins=60, density=True, orientation="horizontal",
               color="#8A0F13", alpha=0.7)
    ax_vd.plot([pi_v(vi) for vi in vg], vg, color="black", lw=1.5)
    ax_vd.tick_params(axis="y", left=False, labelleft=False)
    ax_vd.set_xlabel("Density", fontweight="bold", fontsize=12)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.show()


def plot_density_grid(df_plot, params_to_plot, title, std_dicts=None, save_path=None):
    fig, axes = plt.subplots(2, len(params_to_plot),
                             figsize=(4*len(params_to_plot), 8), sharex="col")
    order = ["SS","LL","GA","EM"]
    for ri, tag in enumerate(H_TAGS):
        df_h = df_plot[df_plot["h"] == tag]
        for ci, param in enumerate(params_to_plot):
            ax = axes[ri, ci]
            df_p = df_h[df_h["parameter"] == param]
            if std_dicts and tag in std_dicts and param in std_dicts[tag]:
                std_i = std_dicts[tag][param]
                if std_i > 0 and np.isfinite(std_i):
                    vals = df_p["error"].dropna()
                    x_lo = max(vals.min() if len(vals) else -4*std_i, -4*std_i)
                    x_hi = min(vals.max() if len(vals) else  4*std_i,  4*std_i)
                    xg   = np.linspace(x_lo, x_hi, 500)
                    ax.plot(xg, sp_norm.pdf(xg,0,std_i), color="black", lw=2.5,
                            alpha=0.85, zorder=2,
                            label="CLT" if (ci==0 and ri==0) else "_nolegend_")
                    ax.fill_between(xg, sp_norm.pdf(xg,0,std_i), color="black",
                                    alpha=0.08, zorder=2)
            if not df_p.empty:
                sns.kdeplot(data=df_p, x="error", hue="estimator", hue_order=order,
                            palette=PALETTE, fill=True, alpha=0.7, linewidth=1.5,
                            ax=ax, legend=False, common_norm=False, zorder=1)
            ax.axvline(0, color="black", lw=2, linestyle="--", zorder=10)
            if ri == 0: ax.set_title(LATEX_MAP[param], fontsize=20, pad=10, fontweight="bold")
            ax.set_xlabel("")
            ax.set_ylabel(H_LABELS[tag] if ci==0 else "", fontweight="bold", fontsize=16)
            sns.despine(ax=ax)
    handles = [mpatches.Patch(color=PALETTE[e], label=e) for e in ESTIMATORS]
    handles.append(plt.Line2D([0],[0], color="black", lw=1.5, label="CLT"))
    fig.legend(handles=handles, title="Estimator", fontsize=16, title_fontsize=18,
               loc="upper center", bbox_to_anchor=(0.5,1.04), ncol=5)
    fig.suptitle(title, fontweight="bold", y=1.1, fontsize=20)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_timing(timing_df, save_path=None):
    tags  = list(H_TAGS)
    x     = np.arange(len(tags))
    width = 0.8 / len(ESTIMATORS)
    fig, ax = plt.subplots(figsize=(7, 3))
    for i, est in enumerate(ESTIMATORS):
        vals = [float(timing_df.loc[(timing_df.estimator == est) & (timing_df.h == t),
                                    "median_time_sec"].values[0])
                if len(timing_df.loc[(timing_df.estimator == est) & (timing_df.h == t)]) > 0
                else float("nan") for t in tags]
        bars = ax.bar(x + (i - len(ESTIMATORS) / 2 + 0.5) * width, vals,
                      width=width * 0.9, label=est, color=PALETTE[est])
        ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([H_LABELS[t] for t in tags])
    ax.set_ylabel("Median time (s)", fontsize=12, fontweight="bold")
    ax.set_title("Median estimation time of Student Kramers oscillator per estimator and step size",
                 fontsize=12, fontweight="bold")
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(title="Estimator", fontsize=10, title_fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    y_max = ax.get_ylim()[1]
    ax.set_yticks(np.arange(0, y_max, 5))
    plt.tight_layout()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="#E0E0E0", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()