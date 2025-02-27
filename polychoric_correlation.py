import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal
import logging
from functools import lru_cache
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union, Any
import warnings

# Configure logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# Constants
EPS = 1e-8  # Slightly larger epsilon to avoid numerical issues
DEFAULT_MAXCOR = 0.9999
DEFAULT_BINS = 4


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


class ParameterError(ValueError):
    """Exception raised for errors in the input parameters."""
    pass


@lru_cache(maxsize=128)
def bvn_cdf(x1: float, x2: float, rho: float) -> float:
    """
    Cached bivariate normal CDF computation with handling for infinite limits.
    
    Parameters
    ----------
    x1, x2 : float
        The upper integration limits.
    rho : float
        Correlation coefficient.
        
    Returns
    -------
    float
        Probability from (-∞, -∞) to (x1, x2)
    """
    if np.isinf(x1) or np.isinf(x2):
        if x1 == -np.inf or x2 == -np.inf:
            return 0.0
        if x1 == np.inf and x2 == np.inf:
            return 1.0
        if x1 == np.inf:
            return norm.cdf(x2)
        if x2 == np.inf:
            return norm.cdf(x1)
    return multivariate_normal.cdf([x1, x2], mean=[0, 0], cov=[[1, rho], [rho, 1]])


def compute_bvn_probabilities(rho: float, row_thresholds: np.ndarray, 
                              col_thresholds: np.ndarray) -> np.ndarray:
    """
    Compute bivariate normal cell probabilities.
    """
    row_bounds = np.concatenate(([-np.inf], row_thresholds, [np.inf]))
    col_bounds = np.concatenate(([-np.inf], col_thresholds, [np.inf]))
    n_row = len(row_bounds) - 1
    n_col = len(col_bounds) - 1
    P = np.empty((n_row, n_col))
    
    for i in range(n_row):
        for j in range(n_col):
            P[i, j] = (bvn_cdf(row_bounds[i+1], col_bounds[j+1], rho) - 
                       bvn_cdf(row_bounds[i], col_bounds[j+1], rho) - 
                       bvn_cdf(row_bounds[i+1], col_bounds[j], rho) + 
                       bvn_cdf(row_bounds[i], col_bounds[j], rho))
    
    return np.clip(P, EPS, 1.0)


def negative_log_likelihood(params: np.ndarray, tab: np.ndarray, 
                            n_row: int, n_col: int, 
                            default_row_thresh: np.ndarray, 
                            default_col_thresh: np.ndarray,
                            maxcor: float,
                            full_ml: bool) -> float:
    """
    Compute negative log-likelihood for optimization.
    """
    rho = np.clip(params[0], -maxcor, maxcor)
    if full_ml and params.size > 1:
        row_thresh = params[1:n_row]
        col_thresh = params[n_row:n_row+n_col-1]
        if np.any(np.diff(row_thresh) <= 0) or np.any(np.diff(col_thresh) <= 0):
            return np.inf
    else:
        row_thresh = default_row_thresh
        col_thresh = default_col_thresh

    P = compute_bvn_probabilities(rho, row_thresh, col_thresh)
    return -np.sum(tab * np.log(P))


def preprocess_data(x: Union[np.ndarray, list], 
                    y: Optional[Union[np.ndarray, list]] = None, 
                    bins: int = DEFAULT_BINS) -> Tuple[np.ndarray, int, int]:
    """
    Preprocess input data into a contingency table.
    """
    if y is None:
        tab = np.asarray(x)
    else:
        x_array = np.asarray(x)
        y_array = np.asarray(y)
        valid_mask = ~(np.isnan(x_array) | np.isnan(y_array))
        if not np.all(valid_mask):
            warnings.warn(f"Removed {np.sum(~valid_mask)} observations with NaN values.")
            x_array = x_array[valid_mask]
            y_array = y_array[valid_mask]
        if len(x_array) == 0 or len(y_array) == 0:
            raise ParameterError("No valid observations after removing NaNs.")
        tab, _, _ = np.histogram2d(x_array, y_array, bins=[bins, bins])
        
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
    Compute default thresholds from marginal distributions.
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
                              n_row: int, n_col: int) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Validate and extract starting parameter values.
    """
    if isinstance(start, dict):
        rho = start.get('rho', 0.0)
        row_thresh = start.get('row_thresholds')
        col_thresh = start.get('col_thresholds')
    else:
        rho = float(start)
        row_thresh = None
        col_thresh = None

    if not isinstance(rho, (int, float)):
        raise ParameterError("Start value for rho must be a number.")

    if row_thresh is not None:
        row_thresh = np.asarray(row_thresh)
        if row_thresh.ndim != 1 or len(row_thresh) != n_row - 1:
            raise ParameterError(f"Row thresholds must be a 1D array of length {n_row - 1}.")
    if col_thresh is not None:
        col_thresh = np.asarray(col_thresh)
        if col_thresh.ndim != 1 or len(col_thresh) != n_col - 1:
            raise ParameterError(f"Column thresholds must be a 1D array of length {n_col - 1}.")
    return rho, row_thresh, col_thresh


def compute_degrees_of_freedom(n_row: int, n_col: int) -> int:
    """Compute degrees of freedom."""
    return (n_row * n_col) - (n_row + n_col)


def compute_chi_square(nll: float, tab: np.ndarray, n_total: float) -> float:
    """Compute chi-square statistic."""
    return 2 * (nll + np.sum(tab * np.log((tab + EPS) / n_total)))


def compute_standard_error_ml(opt_result: Any) -> Optional[float]:
    """
    Compute the variance of rho from the Hessian inverse.
    """
    try:
        hess_inv = opt_result.hess_inv.todense() if hasattr(opt_result.hess_inv, "todense") else opt_result.hess_inv
        var_rho = hess_inv[0, 0]
        return var_rho if var_rho > 0 else None
    except Exception as e:
        logger.warning(f"Failed to compute standard error: {e}")
        return None


def polychoric_correlation(x: Union[np.ndarray, list], 
                           y: Optional[Union[np.ndarray, list]] = None, 
                           ML: bool = True, 
                           compute_std_err: bool = False,
                           maxcor: float = DEFAULT_MAXCOR, 
                           start: Optional[Union[float, Dict[str, Any]]] = None, 
                           return_thresholds: bool = False, 
                           bins: int = DEFAULT_BINS,
                           return_dict: bool = False) -> Union[float, Dict[str, Any], PolychoricResult]:
    """
    Compute polychoric correlation between ordinal variables.
    """
    tab, n_row, n_col = preprocess_data(x, y, bins)
    n_total = np.sum(tab)
    if n_total < (n_row + n_col - 1):
        raise ParameterError("Not enough observations to reliably estimate thresholds.")

    default_row_thresh, default_col_thresh = compute_default_thresholds(tab)

    if start is not None:
        init_rho, init_row_thresh, init_col_thresh = validate_start_parameters(start, n_row, n_col)
        # Use default thresholds if not provided.
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
                lambda r: negative_log_likelihood(np.array([r]), tab, n_row, n_col,
                                                  default_row_thresh, default_col_thresh, 
                                                  maxcor, False),
                bounds=(-maxcor, maxcor), 
                method='bounded'
            )
            init_rho = res_prelim.x
            logger.info(f"Preliminary optimization for rho yielded: {init_rho}")

        initial_params = np.concatenate(([init_rho], init_row_thresh, init_col_thresh))
        opt_result = minimize(
            negative_log_likelihood,
            initial_params,
            args=(tab, n_row, n_col, default_row_thresh, default_col_thresh, maxcor, True),
            method='L-BFGS-B',
            options={'disp': False}
        )
        if not opt_result.success:
            warnings.warn(f"Optimization warning: {opt_result.message}")
        est_params = opt_result.x
        est_rho = np.clip(est_params[0], -maxcor, maxcor)
        est_row_thresh = est_params[1:n_row]
        est_col_thresh = est_params[n_row:n_row+n_col-1]
        nll = opt_result.fun
        chisq = compute_chi_square(nll, tab, n_total)
        df = compute_degrees_of_freedom(n_row, n_col)
        var_rho = compute_standard_error_ml(opt_result) if compute_std_err else None

        result = PolychoricResult(
            rho=est_rho,
            row_thresholds=est_row_thresh,
            col_thresholds=est_col_thresh,
            n=int(n_total),
            chisq=chisq,
            df=df,
            ML=True,
            var_rho=var_rho,
            optimization_success=opt_result.success,
            optimization_message=opt_result.message
        )
    else:
        res = minimize_scalar(
            lambda r: negative_log_likelihood(np.array([r]), tab, n_row, n_col,
                                              default_row_thresh, default_col_thresh, 
                                              maxcor, False),
            bounds=(-maxcor, maxcor),
            method='bounded'
        )
        est_rho = np.clip(res.x, -maxcor, maxcor)
        nll = res.fun
        chisq = compute_chi_square(nll, tab, n_total)
        df = compute_degrees_of_freedom(n_row, n_col)
        var_rho = None
        if compute_std_err:
            h = 1e-4
            f0 = negative_log_likelihood(np.array([est_rho]), tab, n_row, n_col,
                                         default_row_thresh, default_col_thresh, maxcor, False)
            f1 = negative_log_likelihood(np.array([est_rho + h]), tab, n_row, n_col,
                                         default_row_thresh, default_col_thresh, maxcor, False)
            f2 = negative_log_likelihood(np.array([est_rho - h]), tab, n_row, n_col,
                                         default_row_thresh, default_col_thresh, maxcor, False)
            d2f = (f1 - 2*f0 + f2) / (h*h)
            var_rho = 1/d2f if d2f > 0 else None

        result = PolychoricResult(
            rho=est_rho,
            row_thresholds=default_row_thresh,
            col_thresholds=default_col_thresh,
            n=int(n_total),
            chisq=chisq,
            df=df,
            ML=False,
            var_rho=var_rho,
            optimization_success=True,
            optimization_message=res.message if hasattr(res, 'message') else ""
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
# Unit Tests using unittest
# ===================================

import unittest

class TestPolychoricCorrelation(unittest.TestCase):
    """Tests for polychoric correlation function."""
    
    def setUp(self):
        np.random.seed(0)
        self.x = np.random.randint(1, 5, 100)
        self.y = np.random.randint(1, 5, 100)
        # Data with known correlation.
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
        """Test standard error computation."""
        result = polychor(self.x, self.y, compute_std_err=True)
        self.assertIsNotNone(result.var_rho)
        
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


if __name__ == "__main__":
    # Example usage.
    np.random.seed(123)
    x = np.random.randint(1, 5, 200)
    y = np.random.randint(1, 5, 200)
    result = polychoric_correlation(x, y, compute_std_err=True)
    print(result)
    
    # Run tests.
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
