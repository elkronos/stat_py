import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal

def polychor(x, y=None, ML=False, std_err=False, maxcor=0.9999, start=None, thresholds=False):
    """
    Computes the polychoric correlation coefficient between two ordinal variables.

    Parameters:
    x : array-like
        If y is None, x is assumed to be a contingency table. Otherwise, x is an array of ordinal data.
    y : array-like, optional
        An array of ordinal data. If provided, x and y are used to compute the contingency table.
    ML : bool, optional
        If True, the maximum likelihood estimate of the correlation is computed.
    std_err : bool, optional
        If True, the standard error of the estimate is computed.
    maxcor : float, optional
        The maximum allowable value for the correlation coefficient.
    start : float or dict, optional
        Starting values for the optimization. Can be a single number or a dictionary with keys 'rho', 'row_thresholds', and 'col_thresholds'.
    thresholds : bool, optional
        If True, the function also returns the thresholds for the ordinal variables.

    Returns:
    float or dict
        If ML is False, returns the polychoric correlation coefficient. If ML is True and std_err is False, returns the coefficient and thresholds.
        If ML and std_err are True, returns a dictionary with additional information including the standard error.
    """
    
    def validate_start_parameters(start, r, c):
        """Validate the start parameter values."""
        if isinstance(start, dict):
            rho = start.get('rho')
            rc = start.get('row_thresholds')
            cc = start.get('col_thresholds')
        else:
            rho = start
            rc = None
            cc = None
        
        if not isinstance(rho, (int, float)) or len(np.atleast_1d(rho)) != 1:
            raise ValueError("Start value for rho must be a single number")
        if rc is not None and (not isinstance(rc, np.ndarray) or len(rc) != r - 1):
            raise ValueError("Start values for row thresholds must be an array of length r - 1")
        if cc is not None and (not isinstance(cc, np.ndarray) or len(cc) != c - 1):
            raise ValueError("Start values for column thresholds must be an array of length c - 1")
        
        return rho, rc, cc
    
    def log_likelihood(pars):
        """Calculate the negative log-likelihood for the polychoric correlation."""
        pars = np.atleast_1d(pars)
        rho = np.clip(pars[0], -maxcor, maxcor)
        if len(pars) == 1:
            row_cuts, col_cuts = rc, cc
        else:
            row_cuts, col_cuts = pars[1:r], pars[r:r+c-1]
            if any(np.diff(row_cuts) < 0) or any(np.diff(col_cuts) < 0):
                return np.inf
        P = binBvn(rho, row_cuts, col_cuts)
        return -np.sum(tab * np.log(P + 1e-6))
    
    def preprocess_data(x, y):
        """Preprocess the input data to create a contingency table."""
        if y is None:
            return x
        return np.histogram2d(x, y, bins=[4, 4])[0]
    
    tab = preprocess_data(x, y)
    zerorows, zerocols = np.all(tab == 0, axis=1), np.all(tab == 0, axis=0)
    zr, zc = np.sum(zerorows), np.sum(zerocols)
    
    if zr > 0:
        print(f"{zr} rows with zero marginal removed")
    if zc > 0:
        print(f"{zc} columns with zero marginal removed")
    
    tab = tab[~zerorows, :]
    tab = tab[:, ~zerocols]
    r, c = tab.shape
    
    if r < 2 or c < 2:
        print("The table has fewer than 2 rows or columns")
        return None
    
    n = np.sum(tab)
    rc = norm.ppf(np.cumsum(np.sum(tab, axis=1)) / n)[:-1]
    cc = norm.ppf(np.cumsum(np.sum(tab, axis=0)) / n)[:-1]
    
    if start is not None and (ML or std_err):
        rho, rc, cc = validate_start_parameters(start, r, c)
    
    if ML:
        initial_guess = np.concatenate(([rho], rc, cc)) if start is not None else np.concatenate(([minimize_scalar(log_likelihood).x], rc, cc))
        result = minimize(log_likelihood, initial_guess, method='L-BFGS-B')
        rho = np.clip(result.x[0], -maxcor, maxcor)
        
        if std_err:
            chisq = 2 * (result.fun + np.sum(tab * np.log((tab + 1e-6) / n)))
            df = len(tab) - r - c
            return {
                'type': 'polychoric',
                'rho': rho,
                'row_cuts': result.x[1:r],
                'col_cuts': result.x[r:r+c-1],
                'var': np.linalg.inv(result.hess_inv.todense()),
                'n': n,
                'chisq': chisq,
                'df': df,
                'ML': True
            }
        elif thresholds:
            return {
                'type': 'polychoric',
                'rho': rho,
                'row_cuts': result.x[1:r],
                'col_cuts': result.x[r:r+c-1],
                'var': None,
                'n': n,
                'chisq': None,
                'df': None,
                'ML': True
            }
        else:
            return rho
    
    elif std_err:
        result = minimize(log_likelihood, [0], method='BFGS')
        rho = np.clip(result.x[0], -maxcor, maxcor)
        chisq = 2 * (result.fun + np.sum(tab * np.log((tab + 1e-6) / n)))
        df = len(tab) - r - c
        return {
            'type': 'polychoric',
            'rho': rho,
            'row_cuts': rc,
            'col_cuts': cc,
            'var': 1 / result.hess_inv[0, 0],
            'n': n,
            'chisq': chisq,
            'df': df,
            'ML': False
        }
    else:
        rho = minimize_scalar(log_likelihood).x
        if thresholds:
            return {
                'type': 'polychoric',
                'rho': rho,
                'row_cuts': rc,
                'col_cuts': cc,
                'var': None,
                'n': n,
                'chisq': None,
                'df': None,
                'ML': False
            }
        else:
            return rho

def binBvn(rho, row_cuts, col_cuts, bins=4):
    """
    Computes the bivariate normal probabilities for given thresholds and correlation.

    Parameters:
    rho : float
        The correlation coefficient.
    row_cuts : array-like
        Thresholds for the rows.
    col_cuts : array-like
        Thresholds for the columns.
    bins : int, optional
        The number of bins.

    Returns:
    array
        A 2D array of bivariate normal probabilities.
    """
    row_cuts = np.concatenate(([-np.inf], row_cuts, [np.inf])) if row_cuts is not None else np.concatenate(([-np.inf], np.linspace(0, 1, bins)[1:], [np.inf]))
    col_cuts = np.concatenate(([-np.inf], col_cuts, [np.inf])) if col_cuts is not None else np.concatenate(([-np.inf], np.linspace(0, 1, bins)[1:], [np.inf]))
    r = len(row_cuts) - 1
    c = len(col_cuts) - 1
    P = np.zeros((r, c))

    for i in range(r):
        for j in range(c):
            lower = np.array([row_cuts[i], col_cuts[j]])
            upper = np.array([row_cuts[i+1], col_cuts[j+1]])

            # Check for invalid input values
            if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
                continue

            # Calculate the multivariate normal CDF
            P[i, j] = multivariate_normal.cdf(upper, mean=[0, 0], cov=[[1, rho], [rho, 1]]) - \
                      multivariate_normal.cdf([lower[0], upper[1]], mean=[0, 0], cov=[[1, rho], [rho, 1]]) - \
                      multivariate_normal.cdf([upper[0], lower[1]], mean=[0, 0], cov=[[1, rho], [rho, 1]]) + \
                      multivariate_normal.cdf(lower, mean=[0, 0], cov=[[1, rho], [rho, 1]])

            # Check for invalid output values
            if not np.isfinite(P[i, j]):
                P[i, j] = 0

    return P

import unittest
import numpy as np

class TestPolychor(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.x = np.random.randint(1, 5, 100)
        self.y = np.random.randint(1, 5, 100)
        self.small_x = np.array([1, 2])
        self.small_y = np.array([1, 2])

    def test_polychor_basic(self):
        result = polychor(self.x, self.y, ML=True, std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_without_y(self):
        tab = np.histogram2d(self.x, self.y, bins=[4, 4])[0]
        result = polychor(tab, ML=True, std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_edge_case_few_rows(self):
        result = polychor(self.small_x, self.small_y, ML=True, std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_edge_case_few_columns(self):
        tab = np.array([[0, 1], [2, 3]])
        result = polychor(tab, ML=True, std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

    def test_polychor_invalid_start(self):
        with self.assertRaises(ValueError):
            polychor(self.x, self.y, start={'rho': 'invalid', 'row_thresholds': np.array([0.5]), 'col_thresholds': np.array([0.5])}, ML=True)

    def test_polychor_thresholds(self):
        result = polychor(self.x, self.y, thresholds=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)
        self.assertIsNotNone(result['row_cuts'])
        self.assertIsNotNone(result['col_cuts'])

    def test_binBvn(self):
        rho = 0.5
        row_cuts = np.array([0])
        col_cuts = np.array([0])
        P = binBvn(rho, row_cuts, col_cuts)
        self.assertEqual(P.shape, (2, 2))
        self.assertTrue(np.all(P >= 0))
        self.assertTrue(np.all(P <= 1))

    def test_large_input_performance(self):
        large_x = np.random.randint(1, 5, 10000)
        large_y = np.random.randint(1, 5, 10000)
        result = polychor(large_x, large_y, ML=True, std_err=True)
        self.assertIn('type', result)
        self.assertEqual(result['type'], 'polychoric')
        self.assertTrue(-1 <= result['rho'] <= 1)

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
