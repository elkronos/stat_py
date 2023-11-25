import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm, multivariate_normal

def polychor(x, y=None, ML=False, std_err=False, maxcor=0.9999, start=None, thresholds=False):
    def f(pars):
        pars = np.atleast_1d(pars)
        rho = pars[0]
        rho = np.clip(rho, -maxcor, maxcor)
        if len(pars) == 1:
            row_cuts = rc
            col_cuts = cc
        else:
            row_cuts = pars[1:r]
            col_cuts = pars[r:r+c-1]
            if any(np.diff(row_cuts) < 0) or any(np.diff(col_cuts) < 0):
                return np.inf
        P = binBvn(rho, row_cuts, col_cuts)
        return -np.sum(tab * np.log(P + 1e-6))

    if y is None:
        tab = x
    else:
        tab = np.histogram2d(x, y, bins=[4, 4])[0] 
        
    zerorows = np.all(tab == 0, axis=1)
    zerocols = np.all(tab == 0, axis=0)
    zr = np.sum(zerorows)
    zc = np.sum(zerocols)
    
    if zr > 0:
        print(f"{zr} rows with zero marginal removed")
    if zc > 0:
        print(f"{zc} columns with zero marginal removed")
        
    tab = tab[~zerorows, :]
    tab = tab[:, ~zerocols]
    r, c = tab.shape
    
    if r < 2:
        print("The table has fewer than 2 rows")
        return None
    if c < 2:
        print("The table has fewer than 2 columns")
        return None
    
    n = np.sum(tab)
    rc = norm.ppf(np.cumsum(np.sum(tab, axis=1)) / n)[:-1]
    cc = norm.ppf(np.cumsum(np.sum(tab, axis=0)) / n)[:-1]
    
    if start is not None and (ML or std_err):
        if isinstance(start, dict):
            rho = start['rho']
            rc = start['row_thresholds']
            cc = start['col_thresholds']
        else:
            rho = start
        if not isinstance(rho, (int, float)) or len(np.atleast_1d(rho)) != 1:
            raise ValueError("Start value for rho must be a number")
        if not isinstance(rc, np.ndarray) or len(rc) != r - 1:
            raise ValueError("Start values for row thresholds must be r - 1 numbers")
        if not isinstance(cc, np.ndarray) or len(cc) != c - 1:
            raise ValueError("Start values for column thresholds must be c - 1 numbers")
    
    if ML:
        if start is None:
            rho = minimize_scalar(f).x
            initial_guess = np.concatenate(([rho], rc, cc))
        else:
            initial_guess = np.concatenate(([rho], rc, cc))
        result = minimize(f, initial_guess, method='L-BFGS-B')
        rho = result.x[0]
        rho = np.clip(rho, -maxcor, maxcor)
        
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
        result = minimize(f, [0], method='BFGS')
        rho = result.x[0]
        rho = np.clip(rho, -maxcor, maxcor)
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
        rho = minimize_scalar(f).x
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


# Generating random data for x and y
np.random.seed(0)
x = np.random.randint(1, 5, 100)  # Random integers between 1 and 4
y = np.random.randint(1, 5, 100)  # Random integers between 1 and 4

# Using the polychor function
result = polychor(x, y, ML=True, std_err=True)
print(result)