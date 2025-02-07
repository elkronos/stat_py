import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal
import logging

# Configure a basic logger (you can reconfigure or disable as needed)
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# A small constant to avoid log(0)
EPS = 1e-6

def validate_start_parameters(start, n_row, n_col):
    """
    Validate starting parameters for the ML optimization.

    Parameters
    ----------
    start : float or dict
        A starting value for the correlation (or a dictionary with keys
        'rho', 'row_thresholds', and 'col_thresholds').
    n_row : int
        Number of rows in the contingency table.
    n_col : int
        Number of columns in the contingency table.

    Returns
    -------
    rho : float
        Starting value for the correlation.
    row_thresholds : np.ndarray or None
        Starting values for the row thresholds.
    col_thresholds : np.ndarray or None
        Starting values for the column thresholds.
    """
    if isinstance(start, dict):
        rho = start.get('rho', 0.0)
        row_thresholds = start.get('row_thresholds')
        col_thresholds = start.get('col_thresholds')
    else:
        rho = start
        row_thresholds = None
        col_thresholds = None

    if not isinstance(rho, (int, float)):
        raise ValueError("Start value for rho must be a single number.")

    if row_thresholds is not None:
        row_thresholds = np.asarray(row_thresholds)
        if row_thresholds.ndim != 1 or len(row_thresholds) != n_row - 1:
            raise ValueError("Row thresholds must be a one-dimensional array of length n_row - 1.")
    if col_thresholds is not None:
        col_thresholds = np.asarray(col_thresholds)
        if col_thresholds.ndim != 1 or len(col_thresholds) != n_col - 1:
            raise ValueError("Column thresholds must be a one-dimensional array of length n_col - 1.")

    return rho, row_thresholds, col_thresholds

def compute_bvn_probabilities(rho, row_thresholds, col_thresholds, n_row, n_col):
    """
    Compute the bivariate normal cell probabilities given thresholds and correlation.

    Parameters
    ----------
    rho : float
        The correlation coefficient.
    row_thresholds : array-like or None
        Thresholds for the rows (if None, defaults will be used).
    col_thresholds : array-like or None
        Thresholds for the columns (if None, defaults will be used).
    n_row : int
        Number of ordinal categories (rows).
    n_col : int
        Number of ordinal categories (columns).

    Returns
    -------
    P : np.ndarray
        A 2D array of bivariate normal probabilities for each cell.
    """
    # If thresholds are not provided, use equally spaced quantiles (a fallback)
    if row_thresholds is None:
        row_thresholds = np.linspace(-2, 2, n_row - 1)
    else:
        row_thresholds = np.asarray(row_thresholds)
    if col_thresholds is None:
        col_thresholds = np.linspace(-2, 2, n_col - 1)
    else:
        col_thresholds = np.asarray(col_thresholds)

    # Extend thresholds with -infty and +infty
    row_bounds = np.concatenate(([-np.inf], row_thresholds, [np.inf]))
    col_bounds = np.concatenate(([-np.inf], col_thresholds, [np.inf]))

    P = np.zeros((n_row, n_col))
    cov = [[1, rho], [rho, 1]]
    for i in range(n_row):
        for j in range(n_col):
            lower = [row_bounds[i], col_bounds[j]]
            upper = [row_bounds[i+1], col_bounds[j+1]]
            P[i, j] = (multivariate_normal.cdf(upper, mean=[0, 0], cov=cov)
                       - multivariate_normal.cdf([lower[0], upper[1]], mean=[0, 0], cov=cov)
                       - multivariate_normal.cdf([upper[0], lower[1]], mean=[0, 0], cov=cov)
                       + multivariate_normal.cdf(lower, mean=[0, 0], cov=cov))
            if not np.isfinite(P[i, j]):
                P[i, j] = EPS
    # Clip probabilities to avoid exact 0 or 1 values
    return np.clip(P, EPS, 1)

def compute_log_likelihood(params, tab, n_row, n_col, row_thresholds, col_thresholds, maxcor):
    """
    Compute the negative log-likelihood for the polychoric correlation.

    Parameters
    ----------
    params : array-like
        Optimization parameters. The first parameter is rho. If additional
        parameters are provided they represent thresholds.
    tab : np.ndarray
        The contingency table.
    n_row : int
        Number of rows.
    n_col : int
        Number of columns.
    row_thresholds : array-like
        Default (or starting) row thresholds.
    col_thresholds : array-like
        Default (or starting) column thresholds.
    maxcor : float
        Maximum allowable absolute correlation.

    Returns
    -------
    nll : float
        The negative log-likelihood.
    """
    params = np.atleast_1d(params)
    # First parameter is the correlation; constrain it within [-maxcor, maxcor]
    rho = np.clip(params[0], -maxcor, maxcor)
    if params.size == 1:
        # Use externally provided thresholds
        current_row_thresh = row_thresholds
        current_col_thresh = col_thresholds
    else:
        current_row_thresh = params[1:n_row]
        current_col_thresh = params[n_row:n_row+n_col-1]
        # Enforce that thresholds are in strictly increasing order
        if np.any(np.diff(current_row_thresh) <= 0) or np.any(np.diff(current_col_thresh) <= 0):
            return np.inf

    P = compute_bvn_probabilities(rho, current_row_thresh, current_col_thresh, n_row, n_col)
    nll = -np.sum(tab * np.log(P))
    return nll

def preprocess_data(x, y, bins=4):
    """
    Preprocess input data by creating a contingency table.

    Parameters
    ----------
    x : array-like
        If y is provided, x is raw ordinal data; if y is None, x is assumed to be a table.
    y : array-like or None
        The second ordinal variable. If None, x is taken as the contingency table.
    bins : int, optional
        Number of bins to use when constructing the table.

    Returns
    -------
    tab : np.ndarray
        A 2D contingency table with any all-zero rows or columns removed.
    """
    if y is None:
        tab = np.asarray(x)
    else:
        tab, _, _ = np.histogram2d(x, y, bins=[bins, bins])
    # Remove any rows or columns that are entirely zero
    valid_rows = ~np.all(tab == 0, axis=1)
    valid_cols = ~np.all(tab == 0, axis=0)
    if np.sum(~valid_rows) > 0:
        logger.info(f"Removed {np.sum(~valid_rows)} rows with zero marginals.")
    if np.sum(~valid_cols) > 0:
        logger.info(f"Removed {np.sum(~valid_cols)} columns with zero marginals.")
    tab = tab[valid_rows, :][:, valid_cols]
    return tab

def compute_default_thresholds(tab):
    """
    Compute default threshold values using the marginal cumulative proportions.

    Parameters
    ----------
    tab : np.ndarray
        The contingency table.

    Returns
    -------
    row_cuts : np.ndarray
        Thresholds for rows.
    col_cuts : np.ndarray
        Thresholds for columns.
    """
    n = np.sum(tab)
    if n == 0:
        raise ValueError("The contingency table has no counts.")
    row_sums = np.sum(tab, axis=1)
    col_sums = np.sum(tab, axis=0)
    row_cuts = norm.ppf(np.cumsum(row_sums) / n)[:-1]
    col_cuts = norm.ppf(np.cumsum(col_sums) / n)[:-1]
    return row_cuts, col_cuts

def polychoric_correlation(x, y=None, ML=True, compute_std_err=False,
                            maxcor=0.9999, start=None, return_thresholds=False, bins=4):
    """
    Compute the polychoric correlation coefficient between two ordinal variables.

    Parameters
    ----------
    x : array-like
        If y is None, x is assumed to be a contingency table; otherwise,
        x is raw ordinal data.
    y : array-like, optional
        Second ordinal variable. If provided, a contingency table is constructed.
    ML : bool, optional
        If True, the full maximum likelihood (ML) estimation is used. If False,
        only the correlation is optimized (with thresholds fixed at their defaults).
    compute_std_err : bool, optional
        If True, the standard error for the estimated correlation is computed.
    maxcor : float, optional
        Maximum absolute value allowed for the correlation.
    start : float or dict, optional
        Starting values for the optimization. If a dict, it should have keys:
        'rho', 'row_thresholds', and 'col_thresholds'.
    return_thresholds : bool, optional
        If True, the returned output includes the estimated thresholds.
    bins : int, optional
        Number of bins (categories) to use when constructing the contingency table.

    Returns
    -------
    result : float or dict
        If neither compute_std_err nor return_thresholds is True, returns the estimated correlation.
        Otherwise, returns a dictionary with the estimated correlation, thresholds (if requested),
        variance (if computed), total sample size, a chi-square statistic, degrees of freedom,
        and optimization diagnostics.
    """
    # Build the contingency table
    tab = preprocess_data(x, y, bins=bins)
    if tab.ndim != 2:
        raise ValueError("The contingency table must be two-dimensional.")
    n_row, n_col = tab.shape
    if n_row < 2 or n_col < 2:
        raise ValueError("Contingency table must have at least 2 rows and 2 columns.")
    n_total = np.sum(tab)
    if n_total == 0:
        raise ValueError("Total count in the contingency table is zero.")

    # Obtain default threshold estimates based on marginal proportions.
    default_row_thresh, default_col_thresh = compute_default_thresholds(tab)

    # If a starting value was provided, validate it; otherwise use defaults.
    if start is not None and (ML or compute_std_err):
        init_rho, init_row_thresh, init_col_thresh = validate_start_parameters(start, n_row, n_col)
    else:
        init_rho = None
        init_row_thresh = default_row_thresh
        init_col_thresh = default_col_thresh

    if ML:
        # Use a preliminary optimization for rho if no starting value was provided.
        if init_rho is None:
            res_scalar = minimize_scalar(
                lambda r: compute_log_likelihood(np.array([r]), tab, n_row, n_col,
                                                 init_row_thresh, init_col_thresh, maxcor),
                bounds=(-maxcor, maxcor), method='bounded'
            )
            init_rho = res_scalar.x

        # Form the full parameter vector: [rho, row_thresholds, col_thresholds]
        initial_params = np.concatenate(([init_rho], init_row_thresh, init_col_thresh))
        opt_result = minimize(
            compute_log_likelihood,
            initial_params,
            args=(tab, n_row, n_col, init_row_thresh, init_col_thresh, maxcor),
            method='L-BFGS-B'
        )
        if not opt_result.success:
            raise RuntimeError("Optimization failed: " + opt_result.message)
        est_params = opt_result.x
        est_rho = np.clip(est_params[0], -maxcor, maxcor)
        est_row_thresh = est_params[1:n_row]
        est_col_thresh = est_params[n_row:n_row+n_col-1]
        chisq = 2 * (opt_result.fun + np.sum(tab * np.log((tab + EPS) / n_total)))
        # A rough degrees-of-freedom estimate: cells minus number of estimated parameters
        df = (n_row * n_col) - ((n_row + n_col - 1) + 1)
        result_dict = {
            'type': 'polychoric',
            'rho': est_rho,
            'row_thresholds': est_row_thresh,
            'col_thresholds': est_col_thresh,
            'n': n_total,
            'chisq': chisq,
            'df': df,
            'ML': True,
            'optimization_success': opt_result.success,
            'optimization_message': opt_result.message
        }
        if compute_std_err:
            try:
                # L-BFGS-B returns a "LinearOperator" for hess_inv; convert it to a dense matrix.
                hess_inv = opt_result.hess_inv.todense() if hasattr(opt_result.hess_inv, "todense") else opt_result.hess_inv
                var_rho = hess_inv[0, 0] if hess_inv[0, 0] > 0 else np.nan
            except Exception:
                var_rho = np.nan
            result_dict['var_rho'] = var_rho
        # If only a simple correlation is requested, return that.
        if not compute_std_err and not return_thresholds:
            return est_rho
        return result_dict
    else:
        # Non-ML: Optimize only over rho, with thresholds fixed at their defaults.
        res = minimize(
            lambda params: compute_log_likelihood(params, tab, n_row, n_col,
                                                  init_row_thresh, init_col_thresh, maxcor),
            x0=[0],
            method='BFGS'
        )
        if not res.success:
            raise RuntimeError("Optimization failed: " + res.message)
        est_rho = np.clip(res.x[0], -maxcor, maxcor)
        chisq = 2 * (res.fun + np.sum(tab * np.log((tab + EPS) / n_total)))
        df = (n_row * n_col) - ((n_row + n_col - 1) + 1)
        result_dict = {
            'type': 'polychoric',
            'rho': est_rho,
            'row_thresholds': init_row_thresh,
            'col_thresholds': init_col_thresh,
            'n': n_total,
            'chisq': chisq,
            'df': df,
            'ML': False,
            'optimization_success': res.success,
            'optimization_message': res.message
        }
        if compute_std_err:
            try:
                hess_inv = res.hess_inv if isinstance(res.hess_inv, np.ndarray) else res.hess_inv
                var_rho = hess_inv[0, 0] if hess_inv[0, 0] > 0 else np.nan
            except Exception:
                var_rho = np.nan
            result_dict['var_rho'] = var_rho
        if not compute_std_err and not return_thresholds:
            return est_rho
        return result_dict

# For backward compatibility, alias polychoric_correlation as polychor.
polychor = polychoric_correlation

# =========================
# Unit Tests using unittest
# =========================

import unittest

class TestPolychoricCorrelation(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.x = np.random.randint(1, 5, 100)
        self.y = np.random.randint(1, 5, 100)
        # Very small samples (should trigger an error because table is too small)
        self.small_x = np.array([1, 2])
        self.small_y = np.array([1, 2])

    def test_polychor_basic(self):
        result = polychor(self.x, self.y, ML=True, compute_std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_without_y(self):
        tab, _, _ = np.histogram2d(self.x, self.y, bins=[4, 4])
        result = polychor(tab, ML=True, compute_std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_edge_case_few_rows(self):
        # Expect an error because the contingency table is too small for estimation.
        with self.assertRaises(ValueError):
            polychor(self.small_x, self.small_y, ML=True, compute_std_err=True)

    def test_polychor_edge_case_few_columns(self):
        tab = np.array([[0, 1], [2, 3]])
        result = polychor(tab, ML=True, compute_std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_invalid_start(self):
        with self.assertRaises(ValueError):
            polychor(self.x, self.y, start={'rho': 'invalid',
                                              'row_thresholds': np.array([0.5]),
                                              'col_thresholds': np.array([0.5])}, ML=True)

    def test_polychor_thresholds(self):
        result = polychor(self.x, self.y, return_thresholds=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)
        self.assertIsNotNone(result['row_thresholds'])
        self.assertIsNotNone(result['col_thresholds'])

    def test_bvn_probabilities(self):
        rho = 0.5
        row_thresholds = np.array([0])
        col_thresholds = np.array([0])
        P = compute_bvn_probabilities(rho, row_thresholds, col_thresholds, 2, 2)
        self.assertEqual(P.shape, (2, 2))
        self.assertTrue(np.all(P >= 0))
        self.assertTrue(np.all(P <= 1))

    def test_large_input_performance(self):
        large_x = np.random.randint(1, 5, 10000)
        large_y = np.random.randint(1, 5, 10000)
        result = polychor(large_x, large_y, ML=True, compute_std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
