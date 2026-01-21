"""
Polychoric correlation + correlogram (heatmap) utilities + unit tests.

Key implementation notes:
- Uses L-BFGS-B bounds for rho (and finite bounds for thresholds) rather than clipping rho
  inside the objective, to keep the objective smooth for the optimizer.
- Does NOT clip the probability matrix P to EPS (which can break sum-to-1). Instead, EPS
  is applied only inside log() and tiny numeric negatives in P are clipped to 0.
- ML standard error is curvature-based in rho holding thresholds fixed at their MLEs.
  (Not a full profile-likelihood SE.)

Run this file directly to execute tests and show an example correlogram.

Requires: numpy, scipy, matplotlib
Optional (for clustered correlogram ordering): scipy.cluster.hierarchy
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import warnings

# -----------------------------------
# Logging configuration
# -----------------------------------
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# -----------------------------------
# Constants
# -----------------------------------
EPS = 1e-8
DEFAULT_MAXCOR = 0.9999
DEFAULT_BINS = 4
LARGE_PENALTY = 1e50
THRESH_BOUND = 10.0

# -----------------------------------
# Data structures
# -----------------------------------
@dataclass
class PolychoricResult:
    """Container for polychoric correlation results."""
    rho: float
    row_thresholds: np.ndarray
    col_thresholds: np.ndarray
    n: int
    chisq: float
    df: int
    ML: bool
    var_rho: Optional[float] = None
    optimization_success: bool = True
    optimization_message: str = ""
    type: str = "polychoric"

    def __repr__(self) -> str:
        result = f"Polychoric correlation: {self.rho:.4f}"
        if self.var_rho is not None:
            result += f" (SE: {np.sqrt(self.var_rho):.4f})"
        return result

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

# -----------------------------------
# Exceptions
# -----------------------------------
class ParameterError(ValueError):
    pass

# -----------------------------------
# Helpers for preprocessing
# -----------------------------------
def _is_small_integer_ordinal(a: np.ndarray, max_levels: int = 12) -> bool:
    """
    Heuristic: treat arrays with a small number of integer-coded levels as ordinal.
    Accepts ints or floats that are basically integers (e.g., 1.0, 2.0).
    """
    if a.dtype.kind in "iu":
        uniq = np.unique(a)
        return 1 < uniq.size <= max_levels
    if a.dtype.kind == "f":
        ai = np.rint(a)
        mask = np.isfinite(a)
        if not np.any(mask):
            return False
        if np.allclose(a[mask], ai[mask]):
            uniq = np.unique(ai[mask])
            return 1 < uniq.size <= max_levels
    return False


def _crosstab_codes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fast crosstab for integer-coded ordinals."""
    x_vals, x_idx = np.unique(x, return_inverse=True)
    y_vals, y_idx = np.unique(y, return_inverse=True)
    tab = np.zeros((x_vals.size, y_vals.size), dtype=float)
    np.add.at(tab, (x_idx, y_idx), 1.0)
    return tab

# -----------------------------------
# Core probability computation
# -----------------------------------
def compute_bvn_probabilities(rho: float,
                              row_thresholds: np.ndarray,
                              col_thresholds: np.ndarray) -> np.ndarray:
    """
    Vectorized bivariate normal cell probabilities for an ordinal×ordinal table.
    Returns probabilities (not EPS-clipped). Clips only tiny numerical noise to [0,1].
    """
    rB = np.concatenate(([-np.inf], row_thresholds, [np.inf]))
    cB = np.concatenate(([-np.inf], col_thresholds, [np.inf]))
    R = len(rB) - 1
    C = len(cB) - 1

    rb_low  = np.repeat(rB[:-1], C)
    rb_high = np.repeat(rB[1:],  C)
    cb_low  = np.tile(cB[:-1],   R)
    cb_high = np.tile(cB[1:],    R)

    mean = np.zeros(2)
    cov  = np.array([[1.0, rho], [rho, 1.0]], dtype=float)

    pts_hh = np.column_stack([rb_high, cb_high])
    pts_lh = np.column_stack([rb_low,  cb_high])
    pts_hl = np.column_stack([rb_high, cb_low ])
    pts_ll = np.column_stack([rb_low,  cb_low ])

    def _cdf_batch(points: np.ndarray) -> np.ndarray:
        out = np.empty(points.shape[0], dtype=float)
        x1, x2 = points[:, 0], points[:, 1]

        mask_neginf = (x1 == -np.inf) | (x2 == -np.inf)
        mask_pospos = (x1 ==  np.inf) & (x2 ==  np.inf)
        mask_x1inf  = (x1 ==  np.inf) & ~mask_pospos
        mask_x2inf  = (x2 ==  np.inf) & ~mask_pospos

        out[mask_neginf] = 0.0
        out[mask_pospos] = 1.0

        if np.any(mask_x1inf):
            out[mask_x1inf] = norm.cdf(x2[mask_x1inf])
        if np.any(mask_x2inf):
            out[mask_x2inf] = norm.cdf(x1[mask_x2inf])

        mask_rest = ~(mask_neginf | mask_pospos | mask_x1inf | mask_x2inf)
        if np.any(mask_rest):
            pts = points[mask_rest]
            try:
                out[mask_rest] = multivariate_normal.cdf(pts, mean=mean, cov=cov)
            except Exception:
                out[mask_rest] = np.array(
                    [multivariate_normal.cdf(p, mean=mean, cov=cov) for p in pts],
                    dtype=float
                )
        return out

    F_hh = _cdf_batch(pts_hh)
    F_lh = _cdf_batch(pts_lh)
    F_hl = _cdf_batch(pts_hl)
    F_ll = _cdf_batch(pts_ll)

    P = (F_hh - F_lh - F_hl + F_ll).reshape(R, C)
    P = np.maximum(P, 0.0)
    P = np.minimum(P, 1.0)
    return P

# -----------------------------------
# Likelihood pieces
# -----------------------------------
def negative_log_likelihood(params: np.ndarray,
                            tab: np.ndarray,
                            n_row: int,
                            n_col: int,
                            default_row_thresh: np.ndarray,
                            default_col_thresh: np.ndarray,
                            maxcor: float,
                            full_ml: bool) -> float:
    rho = float(params[0])
    if not (-maxcor <= rho <= maxcor):
        return LARGE_PENALTY

    if full_ml and params.size > 1:
        row_thresh = params[1:n_row]
        col_thresh = params[n_row:n_row + n_col - 1]
        if np.any(np.diff(row_thresh) <= 0) or np.any(np.diff(col_thresh) <= 0):
            return LARGE_PENALTY
    else:
        row_thresh = default_row_thresh
        col_thresh = default_col_thresh

    P = compute_bvn_probabilities(rho, row_thresh, col_thresh)
    return -np.sum(tab * np.log(np.maximum(P, EPS)))


def compute_degrees_of_freedom(n_row: int, n_col: int) -> int:
    return (n_row * n_col) - (n_row + n_col)


def compute_chi_square(nll: float, tab: np.ndarray, n_total: float) -> float:
    return 2.0 * (nll + np.sum(tab * np.log((tab + EPS) / n_total)))


def compute_standard_error_ml(opt_result: Any,
                              tab: np.ndarray,
                              n_row: int,
                              n_col: int,
                              default_row_thresh: np.ndarray,
                              default_col_thresh: np.ndarray,
                              maxcor: float) -> Optional[float]:
    """
    Curvature-based variance estimate for rho at the ML solution, holding thresholds fixed.
    Not full profile-likelihood.
    """
    try:
        p = np.asarray(opt_result.x, dtype=float)
        rho_hat = float(p[0])
        row_hat = p[1:n_row]
        col_hat = p[n_row:n_row + n_col - 1]

        h = max(1e-4, 1e-2 * (1.0 - abs(rho_hat)))

        def nll_at(r: float) -> float:
            r = float(np.clip(r, -maxcor, maxcor))
            return negative_log_likelihood(
                np.concatenate(([r], row_hat, col_hat)),
                tab, n_row, n_col, default_row_thresh, default_col_thresh,
                maxcor, full_ml=True
            )

        f0 = nll_at(rho_hat)
        f1 = nll_at(rho_hat + h)
        f2 = nll_at(rho_hat - h)
        d2f = (f1 - 2.0 * f0 + f2) / (h * h)
        if d2f <= 0 or not np.isfinite(d2f):
            return None
        return 1.0 / d2f
    except Exception as e:
        logger.warning(f"Failed to compute curvature SE for rho: {e}")
        try:
            hess_inv = opt_result.hess_inv.todense() if hasattr(opt_result.hess_inv, "todense") else opt_result.hess_inv
            var_rho = float(hess_inv[0, 0])
            return var_rho if np.isfinite(var_rho) and var_rho > 0 else None
        except Exception:
            return None

# -----------------------------------
# Preprocessing (table construction)
# -----------------------------------
def preprocess_data(x: Union[np.ndarray, list],
                    y: Optional[Union[np.ndarray, list]] = None,
                    bins: int = DEFAULT_BINS) -> Tuple[np.ndarray, int, int]:
    if y is None:
        tab = np.asarray(x, dtype=float)
    else:
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float)

        valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
        if not np.all(valid_mask):
            n_drop = int(np.sum(~valid_mask))
            warnings.warn(f"Removed {n_drop} observations with NaN values.")
        x_array = x_array[valid_mask]
        y_array = y_array[valid_mask]
        if x_array.size == 0 or y_array.size == 0:
            raise ParameterError("No valid observations after removing NaNs.")

        if _is_small_integer_ordinal(x_array) and _is_small_integer_ordinal(y_array):
            tab = _crosstab_codes(np.rint(x_array).astype(int), np.rint(y_array).astype(int))
        else:
            tab, _, _ = np.histogram2d(x_array, y_array, bins=[bins, bins])

    valid_rows = ~np.all(tab == 0, axis=1)
    valid_cols = ~np.all(tab == 0, axis=0)
    cleaned_tab = tab[valid_rows, :][:, valid_cols]

    n_row, n_col = cleaned_tab.shape
    if n_row < 2 or n_col < 2:
        raise ParameterError("Contingency table must have at least 2 rows and 2 columns after cleaning.")
    return cleaned_tab, n_row, n_col


def compute_default_thresholds(tab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n_total = np.sum(tab)
    if n_total == 0:
        raise ParameterError("Contingency table has no counts.")
    row_sums = np.sum(tab, axis=1)
    col_sums = np.sum(tab, axis=0)
    row_cum_props = np.clip(np.cumsum(row_sums) / n_total, 0.001, 0.999)[:-1]
    col_cum_props = np.clip(np.cumsum(col_sums) / n_total, 0.001, 0.999)[:-1]
    return norm.ppf(row_cum_props), norm.ppf(col_cum_props)


def validate_start_parameters(start: Union[float, Dict[str, Any]],
                              n_row: int,
                              n_col: int) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    if isinstance(start, dict):
        rho = start.get('rho', 0.0)
        row_thresh = start.get('row_thresholds')
        col_thresh = start.get('col_thresholds')
    else:
        rho = float(start)
        row_thresh = None
        col_thresh = None

    if not isinstance(rho, (int, float, np.floating)):
        raise ParameterError("Start value for rho must be a number.")

    if row_thresh is not None:
        row_thresh = np.asarray(row_thresh, dtype=float)
        if row_thresh.ndim != 1 or len(row_thresh) != n_row - 1:
            raise ParameterError(f"Row thresholds must be a 1D array of length {n_row - 1}.")
        if np.any(np.diff(row_thresh) <= 0):
            raise ParameterError("Row thresholds must be strictly increasing.")
    if col_thresh is not None:
        col_thresh = np.asarray(col_thresh, dtype=float)
        if col_thresh.ndim != 1 or len(col_thresh) != n_col - 1:
            raise ParameterError(f"Column thresholds must be a 1D array of length {n_col - 1}.")
        if np.any(np.diff(col_thresh) <= 0):
            raise ParameterError("Column thresholds must be strictly increasing.")
    return float(rho), row_thresh, col_thresh

# -----------------------------------
# Public API
# -----------------------------------
def polychoric_correlation(x: Union[np.ndarray, list],
                           y: Optional[Union[np.ndarray, list]] = None,
                           ML: bool = True,
                           compute_std_err: bool = False,
                           maxcor: float = DEFAULT_MAXCOR,
                           start: Optional[Union[float, Dict[str, Any]]] = None,
                           return_thresholds: bool = False,
                           bins: int = DEFAULT_BINS,
                           return_dict: bool = False,
                           maxiter: int = 200,
                           tol: float = 1e-6) -> Union[float, Dict[str, Any], PolychoricResult]:
    tab, n_row, n_col = preprocess_data(x, y, bins)
    n_total = float(np.sum(tab))
    if n_total < (n_row + n_col - 1):
        raise ParameterError("Not enough observations to reliably estimate thresholds.")

    default_row_thresh, default_col_thresh = compute_default_thresholds(tab)

    if start is not None:
        init_rho, init_row_thresh, init_col_thresh = validate_start_parameters(start, n_row, n_col)
        if init_row_thresh is None:
            init_row_thresh = default_row_thresh
        if init_col_thresh is None:
            init_col_thresh = default_col_thresh
    else:
        init_rho = 0.0
        init_row_thresh = default_row_thresh
        init_col_thresh = default_col_thresh

    if ML:
        if init_rho is None or abs(init_rho) < 1e-6:
            res_prelim = minimize_scalar(
                lambda r: negative_log_likelihood(
                    np.array([r], dtype=float), tab, n_row, n_col,
                    default_row_thresh, default_col_thresh, maxcor, False
                ),
                bounds=(-maxcor, maxcor),
                method='bounded',
                options={'xatol': tol, 'maxiter': maxiter}
            )
            init_rho = float(res_prelim.x)
            logger.debug(f"Preliminary optimization for rho yielded: {init_rho:.6f}")

        initial_params = np.concatenate(([init_rho], init_row_thresh, init_col_thresh)).astype(float)
        bounds = [(-maxcor, maxcor)] + [(-THRESH_BOUND, THRESH_BOUND)] * ((n_row - 1) + (n_col - 1))

        opt_result = minimize(
            negative_log_likelihood,
            initial_params,
            args=(tab, n_row, n_col, default_row_thresh, default_col_thresh, maxcor, True),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': tol}
        )
        if not opt_result.success:
            warnings.warn(f"Optimization warning: {opt_result.message}")

        est_params = np.asarray(opt_result.x, dtype=float)
        est_rho = float(est_params[0])
        est_row_thresh = est_params[1:n_row]
        est_col_thresh = est_params[n_row:n_row + n_col - 1]

        nll = float(opt_result.fun)
        chisq = compute_chi_square(nll, tab, n_total)
        df = compute_degrees_of_freedom(n_row, n_col)
        var_rho = compute_standard_error_ml(
            opt_result, tab, n_row, n_col, default_row_thresh, default_col_thresh, maxcor
        ) if compute_std_err else None

        result = PolychoricResult(
            rho=est_rho,
            row_thresholds=est_row_thresh,
            col_thresholds=est_col_thresh,
            n=int(n_total),
            chisq=float(chisq),
            df=int(df),
            ML=True,
            var_rho=var_rho,
            optimization_success=bool(opt_result.success),
            optimization_message=str(opt_result.message)
        )
    else:
        res = minimize_scalar(
            lambda r: negative_log_likelihood(
                np.array([r], dtype=float), tab, n_row, n_col,
                default_row_thresh, default_col_thresh, maxcor, False
            ),
            bounds=(-maxcor, maxcor),
            method='bounded',
            options={'xatol': tol, 'maxiter': maxiter}
        )
        est_rho = float(res.x)
        nll = float(res.fun)
        chisq = compute_chi_square(nll, tab, n_total)
        df = compute_degrees_of_freedom(n_row, n_col)

        var_rho = None
        if compute_std_err:
            h = max(1e-4, 1e-2 * (1.0 - abs(est_rho)))
            f0 = negative_log_likelihood(np.array([est_rho], dtype=float), tab, n_row, n_col,
                                         default_row_thresh, default_col_thresh, maxcor, False)
            f1 = negative_log_likelihood(np.array([np.clip(est_rho + h, -maxcor, maxcor)], dtype=float),
                                         tab, n_row, n_col,
                                         default_row_thresh, default_col_thresh, maxcor, False)
            f2 = negative_log_likelihood(np.array([np.clip(est_rho - h, -maxcor, maxcor)], dtype=float),
                                         tab, n_row, n_col,
                                         default_row_thresh, default_col_thresh, maxcor, False)
            d2f = (f1 - 2.0 * f0 + f2) / (h * h)
            var_rho = 1.0 / d2f if (d2f > 0 and np.isfinite(d2f)) else None

        result = PolychoricResult(
            rho=est_rho,
            row_thresholds=default_row_thresh,
            col_thresholds=default_col_thresh,
            n=int(n_total),
            chisq=float(chisq),
            df=int(df),
            ML=False,
            var_rho=var_rho,
            optimization_success=True,
            optimization_message=str(getattr(res, 'message', ""))
        )

    if not compute_std_err and not return_thresholds and not return_dict:
        return result.rho
    elif return_dict:
        return result.as_dict()
    else:
        return result

# Alias
polychor = polychoric_correlation

# ===================================
# Correlogram utilities
# ===================================
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

try:
    import pandas as pd  # optional
except Exception:
    pd = None


def polychoric_corr_matrix(
    data,
    *,
    labels=None,
    ML=True,
    bins=4,
    maxcor=0.9999,
    compute_std_err=False,
    **polychor_kwargs
):
    """
    Pairwise polychoric correlation matrix.

    Returns:
      - corr, labels
      - or corr, se, labels if compute_std_err=True (se = std error matrix; NaN if unavailable)
    """
    if pd is not None and hasattr(data, "values") and hasattr(data, "columns"):
        X = np.asarray(data.values)
        if labels is None:
            labels = [str(c) for c in data.columns]
    else:
        X = np.asarray(data)
        if X.ndim != 2:
            raise ValueError("data must be 2D: (n_samples, n_vars)")
        if labels is None:
            labels = [f"V{i+1}" for i in range(X.shape[1])]

    p = X.shape[1]
    corr = np.full((p, p), np.nan, dtype=float)
    se = np.full((p, p), np.nan, dtype=float) if compute_std_err else None
    np.fill_diagonal(corr, 1.0)
    if compute_std_err:
        np.fill_diagonal(se, 0.0)

    for i in range(p):
        for j in range(i + 1, p):
            try:
                if compute_std_err:
                    res = polychoric_correlation(
                        X[:, i], X[:, j],
                        ML=ML,
                        bins=bins,
                        maxcor=maxcor,
                        compute_std_err=True,
                        return_thresholds=True,
                        **polychor_kwargs
                    )
                    r = float(res.rho)
                    s = float(np.sqrt(res.var_rho)) if res.var_rho is not None else np.nan
                    corr[i, j] = corr[j, i] = r
                    se[i, j] = se[j, i] = s
                else:
                    r = float(polychoric_correlation(
                        X[:, i], X[:, j],
                        ML=ML,
                        bins=bins,
                        maxcor=maxcor,
                        **polychor_kwargs
                    ))
                    corr[i, j] = corr[j, i] = r
            except Exception:
                continue

    if compute_std_err:
        return corr, se, labels
    return corr, labels


def reorder_by_clustering(corr, labels):
    """
    Reorder variables using hierarchical clustering on distance = 1 - corr.
    Requires SciPy.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    C = np.asarray(corr, dtype=float).copy()
    C = np.where(np.isfinite(C), C, 0.0)
    np.fill_diagonal(C, 1.0)

    D = 1.0 - C
    np.fill_diagonal(D, 0.0)

    Z = linkage(squareform(D, checks=False), method="average")
    order = leaves_list(Z)

    return corr[np.ix_(order, order)], [labels[i] for i in order]


def plot_correlogram_pro(
    corr,
    labels,
    *,
    title="Polychoric correlogram",
    annotate=True,
    decimals=2,
    show_upper=False,
    cluster=False,
    figsize=(9, 7),
    vmin=-1.0,
    vmax=1.0,
    cmap="coolwarm",
    fontsize=9,
):
    """
    Professional correlogram heatmap with:
      - numbers rounded to `decimals`
      - upper triangle fully blank (transparent) by default
      - cleaner styling (no spines; x labels on top; tight layout)
    """
    C = np.array(corr, dtype=float)
    labs = list(labels)

    if cluster:
        C, labs = reorder_by_clustering(C, labs)

    mask = np.triu(np.ones_like(C, dtype=bool), k=1) if not show_upper else np.zeros_like(C, dtype=bool)
    mask |= ~np.isfinite(C)
    C_plot = np.ma.array(C, mask=mask)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor(fig.get_facecolor())

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color=to_rgba(ax.get_facecolor(), alpha=0.0))

    im = ax.imshow(
        C_plot,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap_obj,
        interpolation="nearest",
        aspect="equal",
    )

    ax.set_title(title, pad=14)

    ax.set_xticks(np.arange(len(labs)))
    ax.set_yticks(np.arange(len(labs)))
    ax.set_xticklabels(labs, rotation=45, ha="left")
    ax.set_yticklabels(labs)

    ax.xaxis.tick_top()
    ax.tick_params(axis="both", which="major", length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation")

    if annotate:
        fmt = f"{{:.{decimals}f}}"
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C_plot.mask[i, j]:
                    continue
                ax.text(
                    j, i, fmt.format(C[i, j]),
                    ha="center", va="center",
                    fontsize=fontsize
                )

    fig.tight_layout()
    return fig, ax

"""
# ===================================
# Unit Tests
# ===================================
import unittest


class TestPolychoricCorrelation(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.x = np.random.randint(1, 5, 100)
        self.y = np.random.randint(1, 5, 100)

        # Latent correlation construction that enforces Corr(z1,z2)=rho
        rho = 0.7
        z1 = np.random.normal(0, 1, 2000)
        e  = np.random.normal(0, 1, 2000)
        z2 = rho * z1 + np.sqrt(1.0 - rho**2) * e

        self.correlated_x = np.digitize(z1, [-1, 0, 1])
        self.correlated_y = np.digitize(z2, [-1, 0, 1])

        self.small_x = np.array([1, 2])
        self.small_y = np.array([1, 2])

    def test_basic_usage(self):
        rho = polychor(self.x, self.y)
        self.assertTrue(-1 <= rho <= 1)

    def test_full_results(self):
        result = polychor(self.x, self.y, return_thresholds=True)
        self.assertIsInstance(result, PolychoricResult)
        self.assertTrue(-1 <= result.rho <= 1)

    def test_dictionary_output(self):
        result = polychor(self.x, self.y, return_dict=True)
        self.assertIsInstance(result, dict)
        self.assertIn('rho', result)

    def test_standard_error(self):
        result = polychor(self.x, self.y, compute_std_err=True)
        self.assertIsNotNone(result.var_rho)
        self.assertGreater(result.var_rho, 0)

    def test_known_correlation(self):
        rho_hat = polychor(self.correlated_x, self.correlated_y)
        self.assertAlmostEqual(rho_hat, 0.7, delta=0.15)

    def test_contingency_table_input(self):
        tab, _, _ = np.histogram2d(self.x, self.y, bins=4)
        rho = polychor(tab)
        self.assertTrue(-1 <= rho <= 1)

    def test_small_sample_error(self):
        with self.assertRaises(ParameterError):
            polychor(self.small_x, self.small_y)

    def test_custom_bins(self):
        rho1 = polychor(self.x, self.y, bins=4)
        rho2 = polychor(self.x, self.y, bins=5)
        self.assertLess(abs(rho1 - rho2), 0.3)

    def test_two_step_vs_ml(self):
        rho1 = polychor(self.x, self.y, ML=True)
        rho2 = polychor(self.x, self.y, ML=False)
        self.assertLess(abs(rho1 - rho2), 0.3)

    def test_start_parameter(self):
        rho = polychor(self.x, self.y, start=0.5)
        self.assertTrue(-1 <= rho <= 1)

    def test_nan_handling(self):
        x_with_nan = self.x.astype(float)
        x_with_nan[0] = np.nan
        rho = polychor(x_with_nan, self.y)
        self.assertTrue(-1 <= rho <= 1)

    def test_discrete_ordinal_crosstab(self):
        x = np.random.randint(1, 6, 500)
        y = np.random.randint(1, 6, 500)
        r1 = polychor(x, y, bins=20)
        r2 = polychor(x.astype(float), y.astype(float), bins=5)
        self.assertTrue(-1 <= r1 <= 1)
        self.assertTrue(-1 <= r2 <= 1)

    def test_corr_matrix_and_plot(self):
        # Smoke test: compute matrix and build a correlogram figure
        X = np.column_stack([
            np.random.randint(1, 6, 300),
            np.random.randint(1, 6, 300),
            np.random.randint(1, 6, 300),
            np.random.randint(1, 6, 300),
        ])
        corr, labels = polychoric_corr_matrix(X, labels=["A", "B", "C", "D"], ML=True)
        self.assertEqual(corr.shape, (4, 4))
        fig, ax = plot_correlogram_pro(corr, labels, annotate=True, decimals=2, show_upper=False, cluster=False)
        self.assertIsNotNone(fig)
        self.assertIsNotNone(ax)
        plt.close(fig)


if __name__ == "__main__":
    # 1) Run tests first (so the plot appears at the end)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestPolychoricCorrelation)
    runner = unittest.TextTestRunner(verbosity=2)
    test_result = runner.run(suite)

    # 2) Example: compute + display a correlogram AFTER tests complete
    np.random.seed(123)
    X = np.column_stack([
        np.random.randint(1, 6, 400),
        np.random.randint(1, 6, 400),
        np.random.randint(1, 6, 400),
        np.random.randint(1, 6, 400),
        np.random.randint(1, 6, 400),
    ])
    labels = ["Item 1", "Item 2", "Item 3", "Item 4", "Item 5"]

    corr, labels = polychoric_corr_matrix(X, labels=labels, ML=True)
    fig, ax = plot_correlogram_pro(
        corr, labels,
        title="Polychoric correlogram",
        annotate=True,
        decimals=2,
        show_upper=False,
        cluster=True,     # requires SciPy; set False if you prefer natural order
        cmap="coolwarm",
        figsize=(8.5, 6.5),
        fontsize=9,
    )

    plt.show()

    # Optional: fail the process if tests failed (useful in CI)
    # import sys
    # sys.exit(0 if test_result.wasSuccessful() else 1)
"""