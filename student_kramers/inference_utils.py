"""
inference_utils.py — Information matrices, asymptotic std devs, CSV filtering
"""
import numpy as np, pandas as pd, jax.numpy as jnp
from config import TRUE_PARAMS_DICT, ALL_PARAMS


def compute_C(traj, params):
    """C1 (5x5, drift) and C2 (3x3, diffusion) from trajectory."""
    alpha, beta, gamma = params[5], params[6], params[7]
    X, V   = traj[:, 0], traj[:, 1]
    sigma2 = alpha*V**2 + beta*V + gamma
    g1  = jnp.stack([-V, X**3, X**2, X, jnp.ones_like(X)], axis=1)
    C1  = (g1/sigma2[:,None])[:,:,None] * g1[:,None,:]
    C1  = C1.mean(axis=0)
    g2  = jnp.stack([V**2, V, jnp.ones_like(V)], axis=1)
    C2  = 0.5 * (g2/sigma2[:,None]**2)[:,:,None] * g2[:,None,:]
    C2  = C2.mean(axis=0)
    return C1, C2


def get_std_dict(C1, C2, N_obs, h, true_params_dict=None):
    """
    Asymptotic relative std devs.
    theta_1=(eta,a,b,c,d): rate sqrt(N*h).  theta_2=(alpha,beta,gamma): rate sqrt(N).
    """
    if true_params_dict is None: true_params_dict = TRUE_PARAMS_DICT
    C1_inv, C2_inv = np.linalg.inv(np.array(C1)), np.linalg.inv(np.array(C2))
    keys1 = ["eta","a","b","c","d"]
    keys2 = ["alpha","beta","gamma_"]
    std = {}
    for i, p in enumerate(keys1): std[p] = float(np.sqrt(C1_inv[i,i]/(N_obs*h)))
    for i, p in enumerate(keys2): std[p] = float(np.sqrt(C2_inv[i,i]/N_obs))
    return {p: std[p]/abs(true_params_dict[p]) for p in std}


def load_and_filter_data(est, tag, results_dir="results",
                         true_params_dict=None, all_params=None):
    """Load CSV, compute relative errors, remove NAs and 3*IQR outliers (2 passes)."""
    if true_params_dict is None: true_params_dict = TRUE_PARAMS_DICT
    if all_params is None:       all_params = ALL_PARAMS
    df      = pd.read_csv(f"{results_dir}/estimates_{est}_h{tag}.csv")
    n_total = len(df)
    err = {p: (df[p]-true_params_dict[p])/true_params_dict[p]
           for p in all_params if p in df.columns}
    if "nll" in df.columns: err["nll"] = df["nll"]
    err_df  = pd.DataFrame(err)
    bad     = err_df.isna().any(axis=1) | np.isinf(err_df).any(axis=1)
    n_na    = int(bad.sum())
    cur     = err_df[~bad].copy()
    if len(cur) == 0: return None, None, n_total, n_na, 0, 0
    n_out = 0
    for _ in range(2):
        mask = pd.Series(False, index=cur.index)
        for p in cur.columns:
            q1, q3 = cur[p].quantile(0.25), cur[p].quantile(0.75)
            iqr = q3 - q1
            mask |= (cur[p] < q1 - 3*iqr) | (cur[p] > q3 + 3*iqr)
        n = int(mask.sum())
        if n == 0: break
        n_out += n; cur = cur[~mask]
    return df.loc[cur.index], cur.index, n_total, n_na, n_out, len(cur)