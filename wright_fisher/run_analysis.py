"""
run_analysis.py — Load CSVs, compute errors, print summary, produce violin and timing figures
"""
import os
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config import (TRUE_PARAMS, TAU0, Q0, RESULTS_DIR, ESTIMATORS,
                     H_SETTINGS, PALETTE, H_COARSE, KAPPA0, K0, LAM0,
                     N_SIM, H_SIM, OBS_STEP_COARSE)
from model import invert_params
from inference_utils import compute_C, assemble_C, compute_stddict
from sde_simulator import simulate
from figures import plot_violin_figure, plot_timing

os.makedirs(RESULTS_DIR, exist_ok=True)

dfs   = {(est, tag): pd.read_csv(f"{RESULTS_DIR}/estimates_{est}_h{tag}.csv")
         for est in ESTIMATORS for tag in H_SETTINGS}
q4    = float(Q0[3])


def _invert_row(row, tau, q4):
    kappa = jnp.array([row["k1"], row["k2"], row["k3"]])
    K     = jnp.array([[row[f"K{i}{j}"] for j in range(3)] for i in range(3)])  
    lam   = jnp.array([row["lam1"], row["lam2"], row["lam3"]])
    P, q  = invert_params(kappa, lam, K, tau=tau, q4=q4)
    out   = {f"p{i+1}{j+1}": float(P[i, j]) for i in range(4) for j in range(4)}
    out.update({f"q{i+1}": float(q[i]) for i in range(3)})  
    return out


dfspq   = {key: pd.DataFrame(df.apply(lambda r: _invert_row(r, TAU0, q4), axis=1).tolist())
           for key, df in dfs.items()}
P_true, q_true = invert_params(KAPPA0, LAM0, K0, tau=TAU0, q4=q4)
true_pq = {f"p{i+1}{j+1}": float(P_true[i, j]) for i in range(4) for j in range(4)}
true_pq.update({f"q{i+1}": float(q_true[i]) for i in range(3)})

prob_params = [f"p{i+1}{j+1}" for i in range(4) for j in range(3)]  
q_params    = ["q1", "q2", "q3"]                                      
all_params  = prob_params + q_params     

print("true_pq sample:", {k: true_pq[k] for k in list(true_pq)[:5]})

summary_rows, plot_data = [], []

for tag in H_SETTINGS:
    for est in ESTIMATORS:
        dfpq = dfspq[(est, tag)]
        n_total  = len(dfpq)
        error_df = pd.DataFrame({p: dfpq[p] - true_pq[p] for p in all_params if p in dfpq.columns})
        is_bad   = error_df.isna().any(axis=1) | np.isinf(error_df).any(axis=1)
        valid_df = error_df[~is_bad]; n_na = int(is_bad.sum())

        if len(valid_df) == 0:
            for p in all_params:
                summary_rows.append(dict(h=tag, estimator=est, parameter=p,
                                        Ntotal=n_total, NNAs=n_na, Noutliers=0, N=0,
                                        Bias=np.nan, Std=np.nan, Median=np.nan, Status="All NAs"))
            continue

        omask = pd.Series(False, index=valid_df.index)
        for p in all_params:
            if p not in valid_df.columns: continue
            v = valid_df[p]; Q1, Q3 = v.quantile(0.25), v.quantile(0.75); IQR = Q3 - Q1
            omask = omask | ((v < Q1 - 3*IQR) | (v > Q3 + 3*IQR))

        final_df  = valid_df[~omask]; n_out = int(omask.sum())

        if len(final_df) == 0:
            for p in all_params:
                summary_rows.append(dict(h=tag, estimator=est, parameter=p,
                                        Ntotal=n_total, NNAs=n_na, Noutliers=n_out,
                                        N=0, Bias=np.nan, Std=np.nan, Median=np.nan,
                                        Status="All Outliers"))
            continue

        for p in all_params:
            if p not in final_df.columns: continue
            vals = final_df[p]
            summary_rows.append(dict(h=tag, estimator=est, parameter=p,
                                    Ntotal=n_total, NNAs=n_na, Noutliers=n_out,
                                    N=len(final_df), Bias=float(vals.mean()),
                                    Std=float(vals.std()), Median=float(vals.median()),
                                    Status="OK"))
            plot_data.append(pd.DataFrame({"parameter": p, "error": vals.values,
                                            "estimator": est, "h": tag}))

df_summary = pd.DataFrame(summary_rows)
df_plot    = pd.concat(plot_data, ignore_index=True) if plot_data else pd.DataFrame()

print("\nSYSTEMATIC DATA SUMMARY")
print(df_summary.groupby(["h", "estimator"])
      .agg(AvgBias=("Bias","mean"), Total=("Ntotal","first"),
           NAs=("NNAs","first"), Outliers=("Noutliers","first"), FinalN=("N","first"))
      .round(4).to_string())

xs      = simulate(KAPPA0, K0, LAM0, jnp.array([0.25,0.25,0.25]), N_SIM, H_SIM, seed=0)
blocks  = compute_C(np.array(xs))
C_mat   = np.array(assemble_C(blocks))
stddict = compute_stddict(C_mat, np.array(TRUE_PARAMS),
                           N=len(xs[::OBS_STEP_COARSE]) - 1,
                           h=H_COARSE, tau=TAU0, q4=q4)

if not df_plot.empty:
    plot_violin_figure(df_plot, prob_params, q_params, PALETTE,
                       {"coarse": "h = 0.2", "fine": "h = 0.02"},
                       stddict=stddict, save_dir=RESULTS_DIR)

    timing_rows = []
    for (est, tag), df in dfs.items():
        if "time_sec" not in df.columns:
            continue
        dfpq = dfspq[(est, tag)]
        error_df = pd.DataFrame({p: dfpq[p] - true_pq[p]
                                  for p in all_params if p in dfpq.columns})
        is_bad   = error_df.isna().any(axis=1) | np.isinf(error_df).any(axis=1)
        valid_df = error_df[~is_bad]
        if len(valid_df) == 0:
            continue
        omask = pd.Series(False, index=valid_df.index)
        for p in all_params:
            if p not in valid_df.columns:
                continue
            v = valid_df[p]; Q1, Q3 = v.quantile(0.25), v.quantile(0.75); IQR = Q3 - Q1
            omask = omask | ((v < Q1 - 3 * IQR) | (v > Q3 + 3 * IQR))
        final_idx = valid_df[~omask].index
        if len(final_idx) == 0:
            continue
        timing_rows.append({"estimator": est, "h": tag,
                             "median_time_sec": df.loc[final_idx, "time_sec"].median()})

    if timing_rows:
        plot_timing(pd.DataFrame(timing_rows),
                    {"coarse": "h = 0.2", "fine": "h = 0.02"},
                    PALETTE, save_path=f"{RESULTS_DIR}/estimation_times.png")