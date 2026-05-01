"""
Polychoric correlation + correlogram (heatmap) utilities + unit tests.

Key implementation notes:
- Uses L-BFGS-B bounds for rho (and finite bounds for thresholds) rather than clipping rho
  inside the objective, to keep the objective smooth for the optimizer.
- Does NOT clip the probability matrix P to EPS (which can break sum-to-1). Instead, EPS
  is applied only inside log() and tiny numeric negatives in P are clipped to 0.
- ML standard error is the observed-information SE: the full Hessian of the
  negative log-likelihood is formed on the joint parameter vector at the MLE
  (by central finite differences with mixed partials), inverted, and the
  (rho, rho) entry of the inverse is reported as Var(rho).

Run this file directly to execute tests and show an example correlogram.

Requires: numpy, scipy, matplotlib
Optional (for clustered correlogram ordering): scipy.cluster.hierarchy
"""

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal
from scipy.special import expit
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
    flat = np.bincount(x_idx * y_vals.size + y_idx, minlength=x_vals.size * y_vals.size)
    tab = flat.reshape(x_vals.size, y_vals.size).astype(float)
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

    `multivariate_normal.cdf` handles infinite limits correctly (reducing to the
    appropriate univariate CDF, or to 0/1 at the boundary corners), so no manual
    masking of ±∞ rows/columns is required.
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

    F_hh = multivariate_normal.cdf(pts_hh, mean=mean, cov=cov)
    F_lh = multivariate_normal.cdf(pts_lh, mean=mean, cov=cov)
    F_hl = multivariate_normal.cdf(pts_hl, mean=mean, cov=cov)
    F_ll = multivariate_normal.cdf(pts_ll, mean=mean, cov=cov)

    P = (F_hh - F_lh - F_hl + F_ll).reshape(R, C)
    P = np.maximum(P, 0.0)
    P = np.minimum(P, 1.0)
    return P

# -----------------------------------
# Threshold reparameterization
# -----------------------------------
# To keep the objective C-infinity (so L-BFGS-B's finite-difference gradients
# behave), thresholds are optimized over an unconstrained parameterization:
#   raw = [tau_1, delta_1, delta_2, ..., delta_{n-2}]
# and reconstructed as
#   thresh_k = tau_1 + sum_{j<k} softplus(delta_j)
# Since softplus(.) > 0 for all real inputs, monotonicity is structural and
# no penalty / bound is required.
def _softplus(x: np.ndarray) -> np.ndarray:
    """Numerically stable softplus: log(1 + exp(x))."""
    return np.logaddexp(0.0, x)


def _inv_softplus(y: np.ndarray) -> np.ndarray:
    """
    Inverse of softplus for y > 0:  x = log(exp(y) - 1) = log(expm1(y)).
    Used to seed the optimizer from strictly increasing threshold guesses.
    """
    y = np.asarray(y, dtype=float)
    return np.log(np.expm1(np.maximum(y, 1e-12)))


def _reconstruct_thresholds(raw: np.ndarray, n_levels: int) -> np.ndarray:
    """
    Map unconstrained raw parameters [tau_1, delta_1, ..., delta_{n_levels-2}]
    to a strictly increasing threshold vector of length (n_levels - 1).
    """
    if n_levels < 2:
        raise ValueError("n_levels must be >= 2")
    raw = np.asarray(raw, dtype=float)
    tau1 = raw[0]
    if n_levels == 2:
        return np.array([tau1], dtype=float)
    increments = _softplus(raw[1:])
    return tau1 + np.concatenate(([0.0], np.cumsum(increments)))


def _deconstruct_thresholds(thresh: np.ndarray) -> np.ndarray:
    """
    Inverse of _reconstruct_thresholds: convert a strictly increasing threshold
    vector into the unconstrained (tau_1, delta_1, ..., delta_{n-2}) form so it
    can be used as an L-BFGS-B starting point.
    """
    thresh = np.asarray(thresh, dtype=float)
    if thresh.size == 1:
        return thresh.copy()
    diffs = np.diff(thresh)
    deltas = _inv_softplus(diffs)
    return np.concatenate(([thresh[0]], deltas))


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
    # rho is kept in [-maxcor, maxcor] structurally by the optimizer's bounds
    # (L-BFGS-B for ML, bounded minimize_scalar for the two-step path), so no
    # penalty branch is needed here. Threshold monotonicity is also structural
    # via the softplus reparameterization, so the objective is C-infinity.
    rho = float(params[0])

    if full_ml and params.size > 1:
        row_thresh = _reconstruct_thresholds(params[1:n_row], n_row)
        col_thresh = _reconstruct_thresholds(params[n_row:n_row + n_col - 1], n_col)
    else:
        row_thresh = default_row_thresh
        col_thresh = default_col_thresh

    P = compute_bvn_probabilities(rho, row_thresh, col_thresh)
    # Use the convention 0 · log 0 = 0 explicitly so empty cells contribute
    # nothing (and so this matches the saturated term in compute_chi_square).
    return -np.sum(np.where(tab > 0, tab * np.log(np.maximum(P, EPS)), 0.0))


# -----------------------------------
# Analytic gradient of the NLL (used by L-BFGS-B via jac=True)
# -----------------------------------
def _bvn_pdf(a: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    """
    Standard bivariate-normal density φ₂(a, b; ρ) evaluated elementwise.
    Returns 0 wherever either argument is infinite (i.e. on the cell boundary
    extensions ±∞ used by `compute_bvn_probabilities`).
    """
    out = np.zeros(a.shape, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(finite):
        a_f = a[finite]
        b_f = b[finite]
        one_minus_rho2 = 1.0 - rho * rho
        z = (a_f * a_f - 2.0 * rho * a_f * b_f + b_f * b_f) / (2.0 * one_minus_rho2)
        out[finite] = np.exp(-z) / (2.0 * np.pi * np.sqrt(one_minus_rho2))
    return out


def _dPhi2_da(a: np.ndarray, b: np.ndarray, rho: float) -> np.ndarray:
    """
    Closed form for ∂Φ₂(a, b; ρ)/∂a = φ(a) · Φ((b − ρa)/√(1 − ρ²)).
    Returns 0 wherever a is infinite; b may be ±∞ (then Φ(...) → 1 or 0).
    The symmetric partial w.r.t. b is obtained by swapping the arguments.
    """
    out = np.zeros(a.shape, dtype=float)
    finite = np.isfinite(a)
    if np.any(finite):
        a_f = a[finite]
        b_f = b[finite]
        sqrt_1mr2 = np.sqrt(1.0 - rho * rho)
        # b_f − ρ·a_f propagates ±∞ from b_f correctly through norm.cdf.
        arg = (b_f - rho * a_f) / sqrt_1mr2
        out[finite] = norm.pdf(a_f) * norm.cdf(arg)
    return out


def _grad_through_softplus(grad_thresh: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """
    Chain-rule a gradient w.r.t. reconstructed thresholds back to a gradient
    w.r.t. the unconstrained raw parameters used by the optimizer:
        thresh[0] = raw[0]
        thresh[k] = raw[0] + Σ_{j=1..k} softplus(raw[j])     (k ≥ 1)
    so ∂thresh[k]/∂raw[0] = 1 and ∂thresh[k]/∂raw[j] = sigmoid(raw[j]) for 1 ≤ j ≤ k.
    """
    grad_thresh = np.asarray(grad_thresh, dtype=float)
    raw = np.asarray(raw, dtype=float)
    T = raw.size
    if T == 0:
        return np.zeros(0, dtype=float)
    # rev_cumsum[j] = Σ_{k=j..T-1} grad_thresh[k]
    rev_cumsum = np.cumsum(grad_thresh[::-1])[::-1]
    out = np.empty(T, dtype=float)
    out[0] = rev_cumsum[0]
    if T > 1:
        out[1:] = expit(raw[1:]) * rev_cumsum[1:]
    return out


def _negative_log_likelihood_and_grad(params: np.ndarray,
                                      tab: np.ndarray,
                                      n_row: int,
                                      n_col: int) -> Tuple[float, np.ndarray]:
    """
    Joint NLL and analytic gradient over (rho, raw row params, raw col params)
    for the full-ML path. Used by L-BFGS-B via `jac=True`, which avoids the
    finite-difference gradient that would otherwise cost
        O(1 + (R − 1) + (C − 1))
    extra NLL evaluations per optimizer step (each NLL evaluation itself being
    four batched BVN-CDF calls).

    Closed forms used:
        ∂Φ₂(a,b;ρ)/∂ρ = φ₂(a,b;ρ)
        ∂Φ₂(a,b;ρ)/∂a = φ(a) · Φ((b − ρa)/√(1 − ρ²))
        ∂Φ₂(a,b;ρ)/∂b = φ(b) · Φ((a − ρb)/√(1 − ρ²))
    These are combined with the four-corner expression
        P_{j,k} = Φ₂(a_high_j, b_high_k) − Φ₂(a_low_j, b_high_k)
                − Φ₂(a_high_j, b_low_k) + Φ₂(a_low_j, b_low_k)
    and chain-ruled through the softplus threshold reparameterization. Each
    interior threshold τ^r_i (resp. τ^c_k) is the upper boundary of one row
    (resp. column) and the lower boundary of the next, which is what produces
    the W[i, :] − W[i+1, :] (resp. W[:, k] − W[:, k+1]) differences below.
    """
    rho = float(params[0])
    raw_row = np.asarray(params[1:n_row], dtype=float)
    raw_col = np.asarray(params[n_row:n_row + n_col - 1], dtype=float)
    row_thresh = _reconstruct_thresholds(raw_row, n_row)
    col_thresh = _reconstruct_thresholds(raw_col, n_col)

    rB = np.concatenate(([-np.inf], row_thresh, [np.inf]))
    cB = np.concatenate(([-np.inf], col_thresh, [np.inf]))
    R, C = n_row, n_col

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

    F_hh = multivariate_normal.cdf(pts_hh, mean=mean, cov=cov)
    F_lh = multivariate_normal.cdf(pts_lh, mean=mean, cov=cov)
    F_hl = multivariate_normal.cdf(pts_hl, mean=mean, cov=cov)
    F_ll = multivariate_normal.cdf(pts_ll, mean=mean, cov=cov)

    P_flat = F_hh - F_lh - F_hl + F_ll
    P_flat = np.clip(P_flat, 0.0, 1.0)
    P = P_flat.reshape(R, C)

    nll = -np.sum(tab * np.log(np.maximum(P, EPS)))

    # W[j, k] = ∂NLL/∂P[j, k] = -tab[j, k] / max(P[j, k], EPS).
    W = -tab / np.maximum(P, EPS)

    # ----- ∂NLL/∂ρ via Plackett's identity: ∂Φ₂/∂ρ = φ₂. -----
    dP_drho = (
        _bvn_pdf(rb_high, cb_high, rho)
        - _bvn_pdf(rb_low,  cb_high, rho)
        - _bvn_pdf(rb_high, cb_low,  rho)
        + _bvn_pdf(rb_low,  cb_low,  rho)
    )
    grad_rho = float(np.sum(W.reshape(-1) * dP_drho))

    # ----- ∂NLL/∂row_thresh[i]: row threshold i is a_high in row i and
    # a_low in row i+1, with opposite signs in the four-corner sum. -----
    grad_row_thresh = np.zeros(n_row - 1)
    cb_high_arr = cB[1:]
    cb_low_arr  = cB[:-1]
    for i in range(n_row - 1):
        tau_arr = np.full(C, row_thresh[i])
        g = _dPhi2_da(tau_arr, cb_high_arr, rho) - _dPhi2_da(tau_arr, cb_low_arr, rho)
        grad_row_thresh[i] = float(np.sum(g * (W[i, :] - W[i + 1, :])))

    # ----- ∂NLL/∂col_thresh[k]: same pattern with axes swapped.
    # Using ∂Φ₂(a, b; ρ)/∂b = ∂Φ₂(b, a; ρ)/∂a, so we reuse `_dPhi2_da`. -----
    grad_col_thresh = np.zeros(n_col - 1)
    rb_high_arr = rB[1:]
    rb_low_arr  = rB[:-1]
    for k in range(n_col - 1):
        tau_arr = np.full(R, col_thresh[k])
        g = _dPhi2_da(tau_arr, rb_high_arr, rho) - _dPhi2_da(tau_arr, rb_low_arr, rho)
        grad_col_thresh[k] = float(np.sum(g * (W[:, k] - W[:, k + 1])))

    # ----- Chain rule through the softplus reparameterization. -----
    grad_raw_row = _grad_through_softplus(grad_row_thresh, raw_row)
    grad_raw_col = _grad_through_softplus(grad_col_thresh, raw_col)

    grad = np.concatenate(([grad_rho], grad_raw_row, grad_raw_col))
    return float(nll), grad


def compute_degrees_of_freedom(n_row: int, n_col: int) -> int:
    return (n_row * n_col) - (n_row + n_col)


def compute_chi_square(nll: float, tab: np.ndarray, n_total: float) -> float:
    # Use the convention 0 · log 0 = 0 explicitly in the saturated term:
    # adding EPS inside log(tab / n_total) inflates the LR statistic whenever
    # any cell is zero. The model term equals -nll, where the NLL is computed
    # under the same convention in negative_log_likelihood, keeping the two
    # halves of the LR statistic consistent.
    sat = np.where(tab > 0, tab * np.log(tab / n_total), 0.0)
    return 2.0 * (np.sum(sat) + nll)


def compute_standard_error_ml(opt_result: Any,
                              tab: np.ndarray,
                              n_row: int,
                              n_col: int,
                              default_row_thresh: np.ndarray,
                              default_col_thresh: np.ndarray,
                              maxcor: float) -> Optional[float]:
    """
    Observed-information SE for rho at the joint ML solution.

    Forms the full observed-information matrix (Hessian of the negative
    log-likelihood) on the joint parameter vector — rho together with the
    unconstrained threshold parameters — by central finite differences
    (diagonal second differences plus 4-point mixed partials), inverts it,
    and returns the (rho, rho) entry of the inverse. This is the marginal
    variance of rho that accounts for joint uncertainty in the thresholds,
    so 1 / H_{rho,rho} is NOT used (it would systematically underestimate
    variance whenever rho and the thresholds are correlated under H, which
    is the generic case).
    """
    try:
        p_hat = np.asarray(opt_result.x, dtype=float).copy()
        rho_hat = float(p_hat[0])
        n_params = p_hat.size
        # Optimizer parameter layout (same as negative_log_likelihood expects):
        #   p[0]                            = rho
        #   p[1 : n_row]                    = raw row threshold params
        #   p[n_row : n_row + n_col - 1]    = raw col threshold params
        # Thresholds are reconstructed from the raw params via softplus inside
        # negative_log_likelihood, so the Hessian is taken with respect to the
        # unconstrained parameterization the optimizer actually used.

        def nll_at(p: np.ndarray) -> float:
            q = p.copy()
            # Keep rho strictly inside the admissible region for the BVN CDF.
            q[0] = float(np.clip(q[0], -maxcor, maxcor))
            return negative_log_likelihood(
                q, tab, n_row, n_col,
                default_row_thresh, default_col_thresh,
                maxcor, full_ml=True
            )

        # Per-parameter step sizes. For rho the step shrinks near the bounds
        # so rho ± h stays inside (-maxcor, maxcor); for the unconstrained
        # threshold raw params the step is scaled to their magnitude.
        h = np.empty(n_params)
        h[0] = max(1e-4, 1e-2 * (1.0 - abs(rho_hat)))
        for i in range(1, n_params):
            h[i] = 1e-4 * max(1.0, abs(p_hat[i]))

        f0 = nll_at(p_hat)
        H = np.empty((n_params, n_params))

        # Diagonal entries: central second differences.
        for i in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = h[i]
            f_p = nll_at(p_hat + ei)
            f_m = nll_at(p_hat - ei)
            H[i, i] = (f_p - 2.0 * f0 + f_m) / (h[i] * h[i])

        # Off-diagonal entries: 4-point mixed central differences.
        for i in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = h[i]
            for j in range(i + 1, n_params):
                ej = np.zeros(n_params)
                ej[j] = h[j]
                f_pp = nll_at(p_hat + ei + ej)
                f_pm = nll_at(p_hat + ei - ej)
                f_mp = nll_at(p_hat - ei + ej)
                f_mm = nll_at(p_hat - ei - ej)
                H[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
                H[j, i] = H[i, j]

        if not np.all(np.isfinite(H)):
            return None

        # Invert the observed-information matrix; (0, 0) entry is Var(rho).
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None

        var_rho = float(H_inv[0, 0])
        if not np.isfinite(var_rho) or var_rho <= 0:
            return None
        return var_rho
    except Exception as e:
        logger.warning(f"Failed to compute observed-information SE for rho: {e}")
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

        x_is_ordinal = _is_small_integer_ordinal(x_array)
        y_is_ordinal = _is_small_integer_ordinal(y_array)
        if x_is_ordinal and y_is_ordinal:
            tab = _crosstab_codes(np.rint(x_array).astype(int), np.rint(y_array).astype(int))
        else:
            non_ordinal = []
            if not x_is_ordinal:
                non_ordinal.append("x")
            if not y_is_ordinal:
                non_ordinal.append("y")
            warnings.warn(
                f"Input variable(s) {', '.join(non_ordinal)} do not appear to be "
                f"integer-coded ordinals; binning into {bins} bins per axis via "
                f"np.histogram2d. For more reliable results, either pass "
                f"integer-coded ordinal data or set the `bins` argument explicitly.",
                UserWarning,
                stacklevel=2,
            )
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

        initial_params = np.concatenate((
            [init_rho],
            _deconstruct_thresholds(init_row_thresh),
            _deconstruct_thresholds(init_col_thresh),
        )).astype(float)
        # rho stays bounded; threshold raw params (tau_1 and the deltas) are
        # unconstrained because monotonicity is enforced by the softplus
        # reconstruction inside negative_log_likelihood.
        bounds = [(-maxcor, maxcor)] + [(None, None)] * ((n_row - 1) + (n_col - 1))

        opt_result = minimize(
            _negative_log_likelihood_and_grad,
            initial_params,
            args=(tab, n_row, n_col),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': maxiter, 'ftol': tol}
        )
        if not opt_result.success:
            warnings.warn(f"Optimization warning: {opt_result.message}")

        est_params = np.asarray(opt_result.x, dtype=float)
        est_rho = float(est_params[0])
        est_row_thresh = _reconstruct_thresholds(est_params[1:n_row], n_row)
        est_col_thresh = _reconstruct_thresholds(est_params[n_row:n_row + n_col - 1], n_col)

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
            except Exception as exc:
                logger.warning(f"polychoric pair ({i}, {j}) failed: {exc!r}")
                continue

    if compute_std_err:
        return corr, se, labels
    return corr, labels


def reorder_by_clustering(corr, labels, signed_distance: bool = False):
    """
    Reorder variables using hierarchical clustering.

    By default, the distance is ``1 - |corr|``, which groups variables by
    association strength regardless of sign (strongly anti-correlated
    variables are treated as similar). Set ``signed_distance=True`` to use
    the legacy ``1 - corr`` distance, where strongly anti-correlated
    variables are treated as maximally distant.

    Requires SciPy.
    """
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform

    C = np.asarray(corr, dtype=float).copy()
    C = np.where(np.isfinite(C), C, 0.0)
    np.fill_diagonal(C, 1.0)

    if signed_distance:
        D = 1.0 - C
    else:
        D = 1.0 - np.abs(C)
    np.fill_diagonal(D, 0.0)

    # Symmetrize to absorb floating-point roundoff so squareform's
    # symmetry check (and downstream linkage) sees an exactly symmetric
    # distance matrix.
    D = 0.5 * (D + D.T)

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