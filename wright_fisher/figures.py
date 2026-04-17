"""
figures.py — Trajectory, centered split violins, and timing bar chart
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.stats import norm as sp_norm


def _style(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, color="#E0E0E0", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig, path):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.show()
    plt.close(fig)


def plot_trajectory(xs, t, save_path=None):
    x4 = 1.0 - xs[:, 0] - xs[:, 1] - xs[:, 2]
    fig, ax = plt.subplots(figsize=(13, 6))
    for col, label in zip([xs[:, 0], xs[:, 1], xs[:, 2], x4], ["X1", "X2", "X3", "X4"]):
        ax.plot(t, col, label=label)
    ax.set_xlabel("Time", fontweight="bold")
    ax.set_ylabel("Allele frequency", fontweight="bold")
    ax.set_title("A trajectory of one locus four-allele Wright-Fisher diffusion", pad=20)
    ax.legend(loc="upper center", bbox_to_anchor=(0.25, 0.99), title="Allele",
              ncol=4, frameon=True, fancybox=True, columnspacing=1, handletextpad=0.5)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    plt.xlim(0, t[-1])
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.3)
    _style(ax)
    _save(fig, save_path)


def _get_max_density(series):
    s = series.dropna()
    if len(s) < 2:
        return 0
    try:
        if s.std() == 0:
            s = s + np.random.normal(0, 1e-6, len(s))
        kde = gaussian_kde(s)
        xgrid = np.linspace(s.min(), s.max(), 200)
        return kde(xgrid).max()
    except Exception:
        return 0


def plot_centered_split_violins(ax, dfplot, params, palette, stddict=None):
    v_parts = sns.__version__.split(".")
    if int(v_parts[0]) > 0 or int(v_parts[1]) >= 13:
        norm_kws = {"density_norm": "area", "common_norm": True}
    else:
        norm_kws = {"scale": "area", "scale_hue": True}

    dfsub = dfplot[dfplot["parameter"].isin(params)].copy()
    if dfsub.empty:
        return

    dfsub["slot"] = dfsub["estimator"].map({"EM": "Left", "GA": "Left",
                                             "SS": "Right", "LL": "Right"})
    dfsub["x_cat"] = dfsub["parameter"] + "_" + dfsub["slot"]

    order = []
    for p in params:
        order.extend([f"{p}_Left", f"{p}_Right"])

    dfemga = dfsub[dfsub["slot"] == "Left"].copy()
    dfssll = dfsub[dfsub["slot"] == "Right"].copy()

    valid = dfsub["error"].dropna()
    if not valid.empty:
        data_half = max(abs(valid.min()), abs(valid.max()))
        clt_half  = max((4 * stddict.get(p, 0) for p in params), default=0) if stddict else 0
        half      = max(data_half, clt_half) * 1.1
    else:
        half = 1.0
    y_lower, y_upper = -half, half

    base_width = 0.95
    w_rights = {}
    ss_peaks = {}

    for p in params:
        pemga  = dfemga[dfemga["parameter"] == p]
        pssll  = dfssll[dfssll["parameter"] == p]
        l_max  = max(_get_max_density(pemga[pemga["estimator"] == e]["error"]) for e in ["EM", "GA"])
        ss_pk  = _get_max_density(pssll[pssll["estimator"] == "SS"]["error"])
        ll_pk  = _get_max_density(pssll[pssll["estimator"] == "LL"]["error"])
        r_max  = max(ss_pk, ll_pk)
        g_max  = max(l_max, r_max)
        w_rights[p] = base_width * r_max / g_max if g_max > 0 else base_width
        ss_peaks[p] = ss_pk

    if stddict is not None:
        for p_idx, p in enumerate(params):
            if p not in stddict:
                continue
            std_i = stddict[p]
            if std_i <= 0 or not np.isfinite(std_i):
                continue
            trunc_lo = max(-4.0 * std_i, y_lower)
            trunc_hi = min(+4.0 * std_i, y_upper)
            ygrid    = np.linspace(trunc_lo, trunc_hi, 500)
            density  = sp_norm.pdf(ygrid, loc=0, scale=std_i)
            sp       = ss_peaks.get(p, 0)
            hw       = w_rights.get(p, base_width) * 0.5
            scale    = (hw / sp) if sp > 0 else (hw / density.max() if density.max() > 0 else 1.0)
            x_center = p_idx * 2 + 1
            ax.plot(x_center + density * scale, ygrid,
                    color="black", lw=1.5, ls="-", zorder=1,
                    label="CLT" if p_idx == 0 else "_nolegend_")
            ax.plot(x_center - density * scale, ygrid,
                    color="black", lw=1.5, ls="-", zorder=1)

    for p in params:
        pemga  = dfemga[dfemga["parameter"] == p]
        pssll  = dfssll[dfssll["parameter"] == p].copy()
        l_max  = max(_get_max_density(pemga[pemga["estimator"] == e]["error"]) for e in ["EM", "GA"])
        r_max  = max(ss_peaks[p], _get_max_density(pssll[pssll["estimator"] == "LL"]["error"]))
        g_max  = max(l_max, r_max)
        wleft  = base_width * l_max / g_max if g_max > 0 else base_width
        wright = w_rights[p]

        if not pemga.empty:
            sns.violinplot(data=pemga, x="x_cat", y="error", hue="estimator",
                           hue_order=["EM", "GA"], split=True, order=order,
                           palette=palette, inner="quartile", linewidth=0, ax=ax,
                           legend=False, width=wleft, **norm_kws)

        if not pssll.empty:
            ss_d = pssll[pssll["estimator"] == "SS"]["error"].dropna()
            ll_d = pssll[pssll["estimator"] == "LL"]["error"].dropna()
            if len(ss_d) >= 1 and len(ll_d) < 2:
                dummy = pssll[pssll["estimator"] == "SS"].copy()
                dummy["estimator"] = "LL"
                dummy["error"] *= 1e6
                pssll = pd.concat([pssll, dummy], ignore_index=True)
            sns.violinplot(data=pssll, x="x_cat", y="error", hue="estimator",
                           hue_order=["SS", "LL"], split=True, order=order,
                           palette=palette, inner="quartile", linewidth=0, ax=ax,
                           legend=False, width=wright, **norm_kws)

    for coll in ax.collections:
        if hasattr(coll, "get_facecolor"):
            fc = coll.get_facecolor()
            if fc is not None and len(fc) > 0:
                fc[:, 3] = 0.9
                coll.set_facecolor(fc)

    for idx, line in enumerate(ax.lines):
        if len(line.get_ydata()) != 2:
            continue
        line.set_color("black")
        line.set_alpha(0.9)
        line.set_linewidth(1.5 if idx % 3 == 1 else 1.0)

    ax.set_ylim(y_lower, y_upper)

    padding = 0.5
    ax.set_xlim(-padding, (len(params) * 2) - 1 + padding)

    tick_positions = np.arange(0.5, len(params) * 2, 2)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(params, fontsize=14)

    for i in range(1, len(params)):
        ax.axvline(i * 2 - 0.5, color="gray", linestyle=":", alpha=0.6)

    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.grid(True, alpha=0.5, axis="y")
    sns.despine(ax=ax)


def plot_violin_figure(df_plot, prob_params, q_params, palette, h_labels,
                       stddict=None, save_dir="results_final"):
    p_row1 = prob_params[:6]
    p_row2 = prob_params[6:]

    for tag, h_label in h_labels.items():
        df_tag = df_plot[df_plot["h"] == tag]

        fig = plt.figure(figsize=(16, 16))
        gs  = fig.add_gridspec(3, 1, hspace=0.35)

        ax_p1 = fig.add_subplot(gs[0, 0])
        plot_centered_split_violins(ax_p1, df_tag, p_row1, palette, stddict=stddict)
        ax_p1.set_xlabel(r"Mutation probabilities $p_{ij}$",
                         fontweight="bold", labelpad=10, fontsize=16)

        ax_p2 = fig.add_subplot(gs[1, 0])
        plot_centered_split_violins(ax_p2, df_tag, p_row2, palette, stddict=stddict)
        ax_p2.set_xlabel(r"Mutation probabilities $p_{ij}$",
                         fontweight="bold", labelpad=10, fontsize=16)

        ax_q = fig.add_subplot(gs[2, 0])
        plot_centered_split_violins(ax_q, df_tag, q_params, palette, stddict=stddict)
        ax_q.set_xlabel(r"Selection coefficients $q_i$",
                        fontweight="bold", labelpad=25, fontsize=16)

        handles = [mpatches.Patch(color=palette[est], label=est)
                   for est in ["EM", "GA", "SS", "LL"]]
        handles.append(plt.Line2D([0], [0], color="black", lw=2.0, label="CLT"))
        fig.legend(handles=handles, title="Estimator", fontsize=14, title_fontsize=16,
                   loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=5)

        for ax in [ax_p1, ax_p2, ax_q]:
            ax.set_ylabel(r"Estimation error $\hat{\theta}_N - \theta_0$",
                          fontweight="bold", fontsize=16)

        fig.suptitle(
            f"One locus four alleles Wright–Fisher diffusion parameter estimators for {h_label}",
            fontweight="bold", y=0.95, fontsize=18
        )
        
        fig.subplots_adjust(top=0.92, bottom=0.05, hspace=0.35)
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(f"{save_dir}/violins_{tag}.png", dpi=300, bbox_inches="tight")
        plt.show()
        plt.close(fig)


def plot_timing(timing_df, h_labels, palette, save_path=None):
    estimator_list = list(palette.keys())
    h_list         = list(h_labels.keys())
    x              = np.arange(len(h_list))
    width          = 0.8 / len(estimator_list)

    fig, ax = plt.subplots(figsize=(7, 3))

    for i, est in enumerate(estimator_list):
        vals = []
        for h in reversed(h_list):
            mask = (timing_df["estimator"] == est) & (timing_df["h"] == h)
            rows = timing_df.loc[mask, "median_time_sec"]
            vals.append(float(rows.values[0]) if len(rows) > 0 and not np.isnan(rows.values[0]) else np.nan)
        offset = (i - len(estimator_list) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, vals, width=width * 0.9, label=est, color=palette[est])
        ax.bar_label(bars, fmt="%.2f", fontsize=10, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels([h_labels[h] for h in reversed(h_list)])
    ax.set_ylabel("Median time (s)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Median estimation time of Wright-Fisher diffusion per estimator and step size",
        fontsize=12, fontweight="bold"
    )
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(title="Estimator", fontsize=10, title_fontsize=12)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    plt.tight_layout()
    _style(ax)
    _save(fig, save_path)