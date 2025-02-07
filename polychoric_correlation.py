import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal

def polychor(x, y=None, ML=False, std_err=False, maxcor=0.9999, start=None, thresholds=False):
    """
    Computes the polychoric correlation coefficient between two ordinal variables.

    Parameters:
    x : array-like or 2D array
        If y is None, x is assumed to be a contingency table. Otherwise, x is an array of ordinal data.
    y : array-like, optional
        An array of ordinal data. If provided, x and y are used to compute the contingency table.
    ML : bool, optional
        If True, the maximum likelihood estimate of the correlation is computed.
    std_err : bool, optional
        If True, the standard error of the estimate is computed.
    maxcor : float, optional
        The maximum allowable absolute value for the correlation coefficient.
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
        """Validate and process start parameter values."""
        rho, rc, cc = None, None, None
        if isinstance(start, dict):
            rho = start.get('rho')
            rc = start.get('row_thresholds')
            cc = start.get('col_thresholds')
        else:
            rho = start
        
        if not isinstance(rho, (int, float)) or not (-1 <= rho <= 1):
            raise ValueError("Start value for rho must be a number between -1 and 1.")
        if rc is not None and (not isinstance(rc, np.ndarray) or len(rc) != r - 1):
            raise ValueError("Start values for row thresholds must be an array of length r - 1.")
        if cc is not None and (not isinstance(cc, np.ndarray) or len(cc) != c - 1):
            raise ValueError("Start values for column thresholds must be an array of length c - 1.")
        
        return rho, rc, cc
    
    def log_likelihood(pars):
        """Calculate the negative log-likelihood for the polychoric correlation."""
        rho = np.clip(pars[0], -maxcor, maxcor)
        row_cuts = pars[1:r] if len(pars) > 1 else rc
        col_cuts = pars[r:r+c-1] if len(pars) > 1 else cc
        if any(np.diff(row_cuts) < 0) or any(np.diff(col_cuts) < 0):
            return np.inf
        P = binBvn(rho, row_cuts, col_cuts)
        return -np.sum(tab * np.log(P + 1e-10))  # Prevent log(0)
    
    def preprocess_data(x, y):
        """Preprocess the input data to create a contingency table."""
        return np.histogram2d(x, y, bins=[np.unique(x).size, np.unique(y).size])[0] if y is not None else x
    
    tab = preprocess_data(x, y)
    if np.any(tab.sum(axis=1) == 0) or np.any(tab.sum(axis=0) == 0):
        raise ValueError("Contingency table contains zero-marginal rows or columns.")
    
    r, c = tab.shape
    if r < 2 or c < 2:
        raise ValueError("The table must have at least two rows and two columns.")
    
    n = np.sum(tab)
    rc = norm.ppf(np.cumsum(np.sum(tab, axis=1)) / n)[:-1]
    cc = norm.ppf(np.cumsum(np.sum(tab, axis=0)) / n)[:-1]
    
    if start is not None:
        rho, rc, cc = validate_start_parameters(start, r, c)
    
    if ML:
        initial_guess = np.concatenate(([rho], rc, cc)) if start else np.concatenate(([0], rc, cc))
        result = minimize(log_likelihood, initial_guess, method='L-BFGS-B')
        rho = np.clip(result.x[0], -maxcor, maxcor)
        
        return {
            'type': 'polychoric',
            'rho': rho,
            'row_cuts': result.x[1:r],
            'col_cuts': result.x[r:r+c-1],
            'var': np.linalg.inv(result.hess_inv.todense()) if std_err else None,
            'n': n,
            'ML': True
        }
    else:
        rho = minimize_scalar(log_likelihood).x
        return rho if not thresholds else {'type': 'polychoric', 'rho': rho, 'row_cuts': rc, 'col_cuts': cc}

def binBvn(rho, row_cuts, col_cuts):
    """Computes bivariate normal probabilities for given thresholds and correlation."""
    row_cuts = np.concatenate(([-np.inf], row_cuts, [np.inf]))
    col_cuts = np.concatenate(([-np.inf], col_cuts, [np.inf]))
    r, c = len(row_cuts) - 1, len(col_cuts) - 1
    P = np.zeros((r, c))
    
    for i in range(r):
        for j in range(c):
            lower, upper = [row_cuts[i], col_cuts[j]], [row_cuts[i+1], col_cuts[j+1]]
            P[i, j] = (multivariate_normal.cdf(upper, mean=[0, 0], cov=[[1, rho], [rho, 1]])
                       - multivariate_normal.cdf([lower[0], upper[1]], mean=[0, 0], cov=[[1, rho], [rho, 1]])
                       - multivariate_normal.cdf([upper[0], lower[1]], mean=[0, 0], cov=[[1, rho], [rho, 1]])
                       + multivariate_normal.cdf(lower, mean=[0, 0], cov=[[1, rho], [rho, 1]]))
    return P
