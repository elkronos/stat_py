"""
Polychoric correlation estimator with vectorized BVN probabilities, robust input
handling for ordinal data, and profile-likelihood standard errors.

Updated to avoid:
- SciPy DeprecationWarning for L-BFGS-B `disp`/`iprint` by removing `disp` option.
- RuntimeWarning from SciPy numdiff (inf - inf) by using a large finite penalty
  instead of returning np.inf for invalid threshold configurations.

Includes a UAT (unit tests) suite at the bottom. Run this file directly to see
an example and to execute the tests.

Requires: numpy, scipy
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
EPS = 1e-8            # Epsilon to avoid log(0) and tiny negatives from numeric noise
DEFAULT_MAXCOR = 0.9999
DEFAULT_BINS = 4
LARGE_PENALTY = 1e50  # Large finite penalty for infeasible parameter proposals


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
        """String representation of results."""
        result = f"Polychoric correlation: {self.rho:.4f}"
        if self.var_rho is not None:
            std_err = np.sqrt(self.var_rho)
            result += f" (SE: {std_err:.4f})"
        return result

    def as_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return {k: v for k, v in self.__dict__.items()}


# -----------------------------------
# Exceptions
# -----------------------------------
class ParameterError(ValueError):
    """Exception raised for errors in the input parameters."""
    pass


# -----------------------------------
# Helpers for preprocessing
# -----------------------------------
def _is_small_integer_ordinal(a: np.ndarray, max_levels: int = 12) -> bool:
    """
    Heuristic: treat arrays with a small number of integer-coded levels as ordinal.
    Accepts int arrays or float arrays that are basically integers (e.g., 1.0, 2.0).
    """
    if a.dtype.kind in "iu":
        uniq = np.unique(a)
        return 1 < uniq.size <= max_levels
    if a.dtype.kind == "f":
        ai = np.rint(a)
        # allow NaNs; compare only finite
        mask = np.isfinite(a)
        if not np.any(mask):
            return False
        if np.allclose(a[mask], ai[mask]):
            uniq = np.unique(ai[mask])
            return 1 < uniq.size <= max_levels
    return False


def _crosstab_codes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Fast crosstab for integer-coded ordinals.
    """
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
    Builds all rectangle corners and evaluates multivariate_normal.cdf in batches.

    Parameters
    ----------
    rho : float
        Latent correlation.
    row_thresholds, col_thresholds : np.ndarray
        Strictly increasing thresholds.

    Returns
    -------
    P : (R, C) ndarray
        Cell probabilities, clipped to [EPS, 1].
    """
    # Build finite bounds
    rB = np.concatenate(([-np.inf], row_thresholds, [np.inf]))
    cB = np.concatenate(([-np.inf], col_thresholds, [np.inf]))
    R = len(rB) - 1
    C = len(cB) - 1

    # All corner combinations as flat arrays (upper/upper, lower/upper, upper/lower, lower/lower)
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
        """
        Compute BVN CDF for an (N,2) array of points with fast paths for infinities.
        Returns a 1D array of length N.
        """
        out = np.empty(points.shape[0], dtype=float)
        x1, x2 = points[:, 0], points[:, 1]

        # Masks for special infinite cases
        mask_neginf = (x1 == -np.inf) | (x2 == -np.inf)
        mask_pospos = (x1 ==  np.inf) & (x2 ==  np.inf)
        mask_x1inf  = (x1 ==  np.inf) & ~mask_pospos
        mask_x2inf  = (x2 ==  np.inf) & ~mask_pospos

        out[mask_neginf] = 0.0
        out[mask_pospos] = 1.0

        # Reductions: if x1 = +inf, CDF reduces to Φ(x2); if x2 = +inf, reduces to Φ(x1)
        if np.any(mask_x1inf):
            out[mask_x1inf] = norm.cdf(x2[mask_x1inf])
        if np.any(mask_x2inf):
            out[mask_x2inf] = norm.cdf(x1[mask_x2inf])

        # The rest need true BVN cdf
        mask_rest = ~(mask_neginf | mask_pospos | mask_x1inf | mask_x2inf)
        if np.any(mask_rest):
            pts = points[mask_rest]
            try:
                out[mask_rest] = multivariate_normal.cdf(pts, mean=mean, cov=cov)
            except Exception:
                # Fallback to pointwise if vector call unsupported
                out[mask_rest] = np.array(
                    [multivariate_normal.cdf(p, mean=mean, cov=cov) for p in pts],
                    dtype=float
                )
        return out

    F_hh = _cdf_batch(pts_hh)
    F_lh = _cdf_batch(pts_lh)
    F_hl = _cdf_batch(pts_hl)
    F_ll = _cdf_batch(pts_ll)

    # Inclusion–exclusion for rectangle probabilities, then reshape
    P = (F_hh - F_lh - F_hl + F_ll).reshape(R, C)
    return np.clip(P, EPS, 1.0)


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
    """
    Compute negative log-likelihood for optimization.

    Parameters
    ----------
    params : np.ndarray
        If full_ml:
           [rho, row_thresh_1..R-1, col_thresh_1..C-1]
        Else:
           [rho]
    """
    rho = np.clip(params[0], -maxcor, maxcor)

    if full_ml and params.size > 1:
        row_thresh = params[1:n_row]
        col_thresh = params[n_row:n_row + n_col - 1]
        # Enforce monotonic thresholds; infeasible → large finite penalty
        if np.any(np.diff(row_thresh) <= 0) or np.any(np.diff(col_thresh) <= 0):
            return LARGE_PENALTY
    else:
        row_thresh = default_row_thresh
        col_thresh = default_col_thresh

    P = compute_bvn_probabilities(rho, row_thresh, col_thresh)
    return -np.sum(tab * np.log(P))


def compute_degrees_of_freedom(n_row: int, n_col: int) -> int:
    """Degrees of freedom for goodness-of-fit: RC - (R + C)"""
    return (n_row * n_col) - (n_row + n_col)


def compute_chi_square(nll: float, tab: np.ndarray, n_total: float) -> float:
    """
    G^2 deviance comparing fitted model vs saturated model:
    2 * ( -logL_fitted + logL_saturated )
    Here nll = -logL_fitted; logL_saturated = sum n_ij log(n_ij / n_total)
    """
    return 2.0 * (nll + np.sum(tab * np.log((tab + EPS) / n_total)))


def compute_standard_error_ml(opt_result: Any,
                              tab: np.ndarray,
                              n_row: int,
                              n_col: int,
                              default_row_thresh: np.ndarray,
                              default_col_thresh: np.ndarray,
                              maxcor: float) -> Optional[float]:
    """
    Profile-based SE for rho at the ML solution.

    We hold the estimated thresholds fixed and approximate the second derivative
    d2/d rho^2 of the (profile) NLL using a central difference. This avoids relying
    on the limited-precision L-BFGS inverse-Hessian product.
    """
    try:
        p = opt_result.x
        rho_hat = float(np.clip(p[0], -maxcor, maxcor))
        row_hat = p[1:n_row]
        col_hat = p[n_row:n_row + n_col - 1]

        # Adaptive step size: smaller near the boundaries where curvature increases
        h = max(1e-4, 1e-2 * (1.0 - abs(rho_hat)))

        def nll_at(r):
            return negative_log_likelihood(
                np.concatenate(([r], row_hat, col_hat)),
                tab, n_row, n_col, default_row_thresh, default_col_thresh,
                maxcor, full_ml=True
            )

        f0 = nll_at(rho_hat)
        f1 = nll_at(np.clip(rho_hat + h, -maxcor, maxcor))
        f2 = nll_at(np.clip(rho_hat - h, -maxcor, maxcor))
        d2f = (f1 - 2.0 * f0 + f2) / (h * h)
        if d2f <= 0 or not np.isfinite(d2f):
            return None
        return 1.0 / d2f
    except Exception as e:
        logger.warning(f"Failed to compute profile SE for rho: {e}")
        # Fall back to optimizer's inverse-Hessian product if available
        try:
            hess_inv = opt_result.hess_inv.todense() if hasattr(opt_result.hess_inv, "todense") else opt_result.hess_inv
            var_rho = hess_inv[0, 0]
            return var_rho if np.isfinite(var_rho) and var_rho > 0 else None
        except Exception:
            return None


# -----------------------------------
# Preprocessing (table construction)
# -----------------------------------
def preprocess_data(x: Union[np.ndarray, list],
                    y: Optional[Union[np.ndarray, list]] = None,
                    bins: int = DEFAULT_BINS) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess input data into a contingency table.

    Behavior:
    - If `y` is None: treat `x` as a precomputed table.
    - If `x` and `y` look like small integer-coded ordinals: build a crosstab.
    - Otherwise: fall back to histogram2d with the given `bins`.
    """
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
            # Discrete ordinal → crosstab (preserves ordinal categories)
            tab = _crosstab_codes(np.rint(x_array).astype(int), np.rint(y_array).astype(int))
        else:
            # Continuous-ish → equal-width histogram
            tab, _, _ = np.histogram2d(x_array, y_array, bins=[bins, bins])

    # Remove all-zero rows/cols (no information)
    valid_rows = ~np.all(tab == 0, axis=1)
    valid_cols = ~np.all(tab == 0, axis=0)
    if np.sum(~valid_rows) > 0:
        logger.info(f"Removed {np.sum(~valid_rows)} rows with zero marginals.")
    if np.sum(~valid_cols) > 0:
        logger.info(f"Removed {np.sum(~valid_cols)} columns with zero marginals.")
    cleaned_tab = tab[valid_rows, :][:, valid_cols]

    n_row, n_col = cleaned_tab.shape
    if n_row < 2 or n_col < 2:
        raise ParameterError("Contingency table must have at least 2 rows and 2 columns after cleaning.")
    return cleaned_tab, n_row, n_col


def compute_default_thresholds(tab: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute default thresholds from marginal distributions using inverse-normal
    of cumulative proportions. Values clipped to (0.001, 0.999) to avoid infinities.
    """
    n_total = np.sum(tab)
    if n_total == 0:
        raise ParameterError("Contingency table has no counts.")
    row_sums = np.sum(tab, axis=1)
    col_sums = np.sum(tab, axis=0)
    row_cum_props = np.clip(np.cumsum(row_sums) / n_total, 0.001, 0.999)[:-1]
    col_cum_props = np.clip(np.cumsum(col_sums) / n_total, 0.001, 0.999)[:-1]
    row_thresh = norm.ppf(row_cum_props)
    col_thresh = norm.ppf(col_cum_props)
    return row_thresh, col_thresh


def validate_start_parameters(start: Union[float, Dict[str, Any]],
                              n_row: int,
                              n_col: int) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate and extract starting parameter values.
    Accepts either a numeric rho or a dict with keys: 'rho', 'row_thresholds', 'col_thresholds'
    """
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
    """
    Compute polychoric correlation between ordinal variables.

    Parameters
    ----------
    x, y : array-like or 2D contingency table if y is None
    ML : bool
        If True, jointly estimate rho and thresholds (full ML).
        If False, do two-step: fixed thresholds from marginals, optimize rho only.
    compute_std_err : bool
        If True, compute variance of rho (SE^2).
        For ML=True, uses a profile-curvature approach holding thresholds at MLEs.
        For ML=False, uses numeric second derivative of NLL wrt rho.
    maxcor : float
        Maximum absolute correlation allowed (to avoid numerical singularities).
    start : float or dict
        Starting values. If dict, keys: 'rho', 'row_thresholds', 'col_thresholds'.
    return_thresholds : bool
        If True, return a PolychoricResult (instead of just rho).
    bins : int
        Number of bins for histogram if data aren't recognized as ordinal.
    return_dict : bool
        If True, return a dict instead of PolychoricResult (back-compat).
    maxiter : int
        Maximum iterations for optimizers.
    tol : float
        Tolerance for optimizers (ftol/xatol).

    Returns
    -------
    float or dict or PolychoricResult
    """
    tab, n_row, n_col = preprocess_data(x, y, bins)
    n_total = float(np.sum(tab))
    if n_total < (n_row + n_col - 1):
        raise ParameterError("Not enough observations to reliably estimate thresholds.")

    default_row_thresh, default_col_thresh = compute_default_thresholds(tab)

    # Starting values
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
        # Preliminary scalar optimization for rho if near zero or None
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
            logger.info(f"Preliminary optimization for rho yielded: {init_rho:.6f}")

        initial_params = np.concatenate(([init_rho], init_row_thresh, init_col_thresh)).astype(float)
        opt_result = minimize(
            negative_log_likelihood,
            initial_params,
            args=(tab, n_row, n_col, default_row_thresh, default_col_thresh, maxcor, True),
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': tol}  # removed deprecated 'disp'
        )
        if not opt_result.success:
            warnings.warn(f"Optimization warning: {opt_result.message}")

        est_params = np.asarray(opt_result.x, dtype=float)
        est_rho = float(np.clip(est_params[0], -maxcor, maxcor))
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
        # Two-step: thresholds from marginals, optimize rho only
        res = minimize_scalar(
            lambda r: negative_log_likelihood(
                np.array([r], dtype=float), tab, n_row, n_col,
                default_row_thresh, default_col_thresh, maxcor, False
            ),
            bounds=(-maxcor, maxcor),
            method='bounded',
            options={'xatol': tol, 'maxiter': maxiter}
        )
        est_rho = float(np.clip(res.x, -maxcor, maxcor))
        nll = float(res.fun)
        chisq = compute_chi_square(nll, tab, n_total)
        df = compute_degrees_of_freedom(n_row, n_col)

        var_rho = None
        if compute_std_err:
            # Profile curvature in 1D (thresholds fixed)
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


# Alias for backward compatibility.
polychor = polychoric_correlation


# ===================================
# UAT: Unit Tests using unittest
# ===================================
import unittest

class TestPolychoricCorrelation(unittest.TestCase):
    """Tests for polychoric correlation function."""

    def setUp(self):
        np.random.seed(0)
        self.x = np.random.randint(1, 5, 100)
        self.y = np.random.randint(1, 5, 100)
        # Data with known latent correlation ~0.7
        z1 = np.random.normal(0, 1, 1000)
        z2 = 0.7 * z1 + 0.7 * np.random.normal(0, 1, 1000)
        self.correlated_x = np.digitize(z1, [-1, 0, 1])
        self.correlated_y = np.digitize(z2, [-1, 0, 1])
        # A small sample that should trigger a ParameterError.
        self.small_x = np.array([1, 2])
        self.small_y = np.array([1, 2])

    def test_basic_usage(self):
        """Test basic estimation works."""
        rho = polychor(self.x, self.y)
        self.assertTrue(-1 <= rho <= 1)

    def test_full_results(self):
        """Test full results object."""
        result = polychor(self.x, self.y, return_thresholds=True)
        self.assertIsInstance(result, PolychoricResult)
        self.assertTrue(-1 <= result.rho <= 1)

    def test_dictionary_output(self):
        """Test dictionary output format."""
        result = polychor(self.x, self.y, return_dict=True)
        self.assertIsInstance(result, dict)
        self.assertIn('rho', result)

    def test_standard_error(self):
        """Test standard error computation (ML profile SE)."""
        result = polychor(self.x, self.y, compute_std_err=True)
        self.assertIsNotNone(result.var_rho)
        self.assertGreater(result.var_rho, 0)

    def test_known_correlation(self):
        """Test with data having known correlation."""
        rho = polychor(self.correlated_x, self.correlated_y)
        self.assertAlmostEqual(rho, 0.7, delta=0.15)

    def test_contingency_table_input(self):
        """Test with contingency table input."""
        tab, _, _ = np.histogram2d(self.x, self.y, bins=4)
        rho = polychor(tab)
        self.assertTrue(-1 <= rho <= 1)

    def test_small_sample_error(self):
        """Test error handling with very small samples."""
        with self.assertRaises(ParameterError):
            polychor(self.small_x, self.small_y)

    def test_custom_bins(self):
        """Test with custom bin count."""
        rho1 = polychor(self.x, self.y, bins=4)
        rho2 = polychor(self.x, self.y, bins=5)
        self.assertLess(abs(rho1 - rho2), 0.3)

    def test_two_step_vs_ml(self):
        """Test two-step vs full ML estimation."""
        rho1 = polychor(self.x, self.y, ML=True)
        rho2 = polychor(self.x, self.y, ML=False)
        self.assertLess(abs(rho1 - rho2), 0.3)

    def test_start_parameter(self):
        """Test with custom starting values."""
        rho = polychor(self.x, self.y, start=0.5)
        self.assertTrue(-1 <= rho <= 1)

    def test_nan_handling(self):
        """Test handling of NaN values."""
        x_with_nan = self.x.astype(float)
        x_with_nan[0] = np.nan
        rho = polychor(x_with_nan, self.y)
        self.assertTrue(-1 <= rho <= 1)

    def test_discrete_ordinal_crosstab(self):
        """Integer-coded 1..5 should be treated as ordinal, not binned."""
        x = np.random.randint(1, 6, 500)
        y = np.random.randint(1, 6, 500)
        r1 = polychor(x, y, bins=20)  # bins ignored due to ordinal auto-detect
        r2 = polychor(x.astype(float), y.astype(float), bins=5)
        self.assertTrue(-1 <= r1 <= 1)
        self.assertTrue(-1 <= r2 <= 1)


if __name__ == "__main__":
    # Example usage.
    np.random.seed(123)
    x = np.random.randint(1, 5, 200)
    y = np.random.randint(1, 5, 200)
    result = polychoric_correlation(x, y, compute_std_err=True)
    print(result)

    # Run tests.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
