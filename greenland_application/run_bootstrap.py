"""
run_bootstrap.py — Asymptotic and parametric bootstrap LR test,
                   and large model validation of b=0, d=0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "student_kramers"))

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

import config_app as cfg
from data_loading import load_ca2, build_data_matrix
from estimation_app import estimate_model, likelihood_ratio_test, make_loss_fn
from bootstrap_app import run_bootstrap, compute_bootstrap_pvalue
from figures_app import plot_bootstrap_distribution, plot_large_model
from sde_simulator import make_milstein_step
from estimation import run_estimator_lbfgs
from config_app import embed_params


def _print_model(label, params, nll=None):
    print(f"\n=== {label} ===")
    for name, val in zip(cfg.PARAM_NAMES, params):
        print(f"  {name:7s}: {val:10.4f}")
    if nll is not None:
        print(f"  {'NLL':7s}: {float(nll):10.4f}")


def _estimate_one_bootstrap(step_fn, init_state, steps_per_traj, key, idx):
    try:
        _, xs    = jax.lax.scan(step_fn, init_state,
                                 jax.random.split(key, steps_per_traj))
        traj     = jnp.concatenate([init_state[None], xs], axis=0)[::cfg.SUBSAMPLE_RATE]
        chunk    = traj[:, 0]
        data_sim = jnp.column_stack((chunk[:-1], jnp.diff(chunk) / cfg.H_OBS))

        free_hat, _, _ = run_estimator_lbfgs(
            cfg.MODELS["large"]["init"],
            make_loss_fn("large", data_sim, cfg.H_OBS),
            maxiter=300, tol=1e-4, verbose=False)

        nll_val = float(make_loss_fn("large", data_sim, cfg.H_OBS)(free_hat))
        full    = np.array(embed_params(jnp.array(free_hat), "large"))

        if np.any(np.isnan(full)) or np.any(np.isinf(full)) or not np.isfinite(nll_val):
            print(f"  [boot {idx}] NaN/Inf in params or NLL — skipping")
            return None, None

        return full, nll_val

    except Exception as exc:
        print(f"  [boot {idx}] {type(exc).__name__}: {exc} — skipping")
        return None, None

def _filter_boot_samples(large_boot: np.ndarray, nll_values: np.ndarray,
                          param_names: list,
                          iqr_multiplier: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Row-level filtering of bootstrap samples.

    Applies IQR filter over all parameter columns AND the NLL column.
    A row is dropped if ANY column is an outlier.
    Returns (filtered_params, filtered_nlls).
    """
    n_before = len(large_boot)

    # stack NLL as an extra column so the IQR loop covers it uniformly
    all_cols   = np.column_stack([large_boot, nll_values])
    col_labels = list(param_names) + ["nll"]

    finite_mask = np.isfinite(all_cols).all(axis=1)
    all_cols    = all_cols[finite_mask]
    n_after_nan = len(all_cols)

    if len(all_cols) == 0:
        print("  WARNING: all bootstrap rows contain NaN/Inf after filtering")
        return all_cols[:, :-1], all_cols[:, -1]

    outlier_mask = np.zeros(len(all_cols), dtype=bool)
    for col_idx, name in enumerate(col_labels):
        vals = all_cols[:, col_idx]
        Q1, Q3 = np.percentile(vals, 25), np.percentile(vals, 75)
        IQR    = Q3 - Q1
        col_mask = (vals < Q1 - iqr_multiplier * IQR) | (vals > Q3 + iqr_multiplier * IQR)
        outlier_mask |= col_mask

    all_cols = all_cols[~outlier_mask]
    n_final  = len(all_cols)

    print(f"\n  Bootstrap filtering summary:")
    print(f"    Raw valid samples:     {n_before}")
    print(f"    After NaN/Inf removal: {n_after_nan}  ({n_before - n_after_nan} dropped)")
    print(f"    After IQR filter:      {n_final}  ({n_after_nan - n_final} dropped, "
          f"multiplier={iqr_multiplier})")

    return all_cols[:, :-1], all_cols[:, -1]   

def main():
    df    = load_ca2()
    Ca2   = df["X_Ca2"].values[::-1]
    X_np  = Ca2[:-1]
    V_np  = np.diff(Ca2) / cfg.H_OBS
    data  = build_data_matrix(Ca2, cfg.H_OBS)

    params_middle, nll_middle, _ = estimate_model("middle", data, cfg.H_OBS, verbose=False)
    params_small,  nll_small,  _ = estimate_model("small",  data, cfg.H_OBS, verbose=False)
    params_large,  nll_large,  _ = estimate_model("large",  data, cfg.H_OBS, verbose=False)

    _print_model("Small Model",  params_small,  nll_small)
    _print_model("Middle Model", params_middle, nll_middle)
    _print_model("Large Model",  params_large,  nll_large)
    print(f"\n  NLL comparison — Small: {float(nll_small):.4f}  "
          f"Middle: {float(nll_middle):.4f}  Large: {float(nll_large):.4f}")

    if np.any(np.isnan(np.array(params_large))):
        raise RuntimeError("Large model estimation returned NaN — cannot run bootstrap.")

    print("\n=== Asymptotic LR Test ===")
    lrt = likelihood_ratio_test(params_middle, params_small, data, cfg.H_OBS)
    print(f"  LR = {lrt['LR']:.2f},  df = {lrt['df']},  p = {lrt['p_value']:.2e}")

    init_state = jnp.array([float(X_np[0]), float(V_np[0])])

    run_bootstrap(
        params_null=jnp.array(params_small),
        n_boot=cfg.N_BOOTSTRAP,
        data_shape=len(X_np) + 1,
        init_state=init_state,
    )

    boot = compute_bootstrap_pvalue()
    print(f"\n=== Bootstrap LR Test ===")
    print(f"  Observed LR  = {boot['obs_LR']:.2f}")
    print(f"  95th pct     = {boot['pct_95']:.2f}")
    print(f"  Extreme runs = {boot['extreme_count']} / {boot['n_boot']}")
    print(f"  p-value      = {boot['p_value']:.5f}")

    plot_bootstrap_distribution(
        boot["LR_values"], boot["obs_LR"],
        save_path=f"{cfg.RESULTS_DIR}/bootstrap_lr.png")

    steps_per_traj = len(X_np) * cfg.SUBSAMPLE_RATE
    step_fn_large  = make_milstein_step(jnp.array(params_large), cfg.H_SIM)
    large_boot_rows = []
    nll_rows        = []
    n_failed        = 0

    for idx in tqdm(range(cfg.N_BOOTSTRAP_LARGE), desc="Large model bootstrap"):
        key           = jax.random.PRNGKey(cfg.BOOTSTRAP_SEED + idx)
        result, nll   = _estimate_one_bootstrap(step_fn_large, init_state,
                                                 steps_per_traj, key, idx)
        if result is not None:
            large_boot_rows.append(result)
            nll_rows.append(nll)
        else:
            n_failed += 1

    n_valid = len(large_boot_rows)
    print(f"\nLarge bootstrap: {n_valid}/{cfg.N_BOOTSTRAP_LARGE} valid  "
          f"({n_failed} failed/skipped)")

    if n_valid < 10:
        raise RuntimeError(f"Only {n_valid} valid bootstrap samples before filtering.")

    large_boot, boot_nlls = _filter_boot_samples(
        np.array(large_boot_rows),
        np.array(nll_rows),
        cfg.PARAM_NAMES,
        iqr_multiplier=3.0,
    )
    n_valid = len(large_boot)

    if n_valid < 10:
        raise RuntimeError(f"Only {n_valid} samples after filtering — cannot compute CIs.")

    ci_low  = np.nanpercentile(large_boot, 2.5,  axis=0)
    ci_high = np.nanpercentile(large_boot, 97.5, axis=0)

    print(f"\n=== Large Model — 95% Bootstrap CIs  (n={n_valid}) ===")
    print(f"  {'param':>7}  {'estimate':>10}  {'2.5%':>10}  {'97.5%':>10}  {'zero in CI':>10}")
    for name, est, lo, hi in zip(cfg.PARAM_NAMES, params_large, ci_low, ci_high):
        in_ci = "yes" if lo <= 0.0 <= hi else "no"
        print(f"  {name:>7}  {est:10.4f}  {lo:10.4f}  {hi:10.4f}  {in_ci:>10}")

    plot_large_model(
        large_boot, params_large,
        param_names=cfg.PARAM_NAMES,
        save_path=f"{cfg.RESULTS_DIR}/large_model_bd_bootstrap.png")


if __name__ == "__main__":
    main()