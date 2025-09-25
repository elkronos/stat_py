import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import norm

def initial_smoothing_values(beta0, beta1, alpha):
    S1_0 = beta0 - ((1 - alpha) / alpha) * beta1
    S2_0 = beta0 - ((2 * (1 - alpha)) / alpha) * beta1
    return S1_0, S2_0

def _compute_states(alpha, y, S1_0, S2_0):
    n = len(y)
    S1 = np.zeros(n, dtype=float)
    S2 = np.zeros(n, dtype=float)
    a  = np.zeros(n, dtype=float)
    b  = np.zeros(n, dtype=float)

    S1[0], S2[0] = S1_0, S2_0
    a[0] = 2*S1[0] - S2[0]
    b[0] = (alpha/(1 - alpha)) * (S1[0] - S2[0])

    for t in range(1, n):
        S1[t] = alpha * y[t] + (1 - alpha) * S1[t-1]
        S2[t] = alpha * S1[t] + (1 - alpha) * S2[t-1]
        a[t]  = 2*S1[t] - S2[t]
        b[t]  = (alpha/(1 - alpha)) * (S1[t] - S2[t])
    return a, b

def compute_forecast(alpha, Pt, S1_0, S2_0, l):
    a, b = _compute_states(alpha, Pt, S1_0, S2_0)
    n = len(Pt)
    forecasts = np.full(n, np.nan, dtype=float)
    errors = np.full(n, np.nan, dtype=float)

    # l-step-ahead in-sample forecasts aligned to actuals
    for t in range(n - l):
        f = a[t] + b[t] * l
        forecasts[t + l] = f
        errors[t + l] = (Pt[t + l] - f)**2

    # out-of-sample (1..l) from last state
    future = np.array([a[-1] + b[-1] * h for h in range(1, l+1)], dtype=float)
    return forecasts, errors, future

def plot_results(Pt, optimal_forecasts, future_forecasts=None, intervals=None):
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.arange(len(Pt))
    fig, ax = plt.subplots(figsize=(10, 6))

    # Main series
    ax.plot(x, Pt, marker='o', linewidth=1.6, label='Actual', zorder=3)
    ax.plot(x, optimal_forecasts, linestyle='--', linewidth=1.8, label='In-sample forecast', zorder=2)

    # In-sample confidence band (if provided)
    if intervals and 'insample' in intervals:
        lo_ins, hi_ins = intervals['insample']
        mask = ~np.isnan(optimal_forecasts)
        ax.fill_between(x[mask], lo_ins[mask], hi_ins[mask], alpha=0.18, label='CI (in-sample)', zorder=1)

    # Out-of-sample line + band
    if future_forecasts is not None and len(future_forecasts) > 0:
        xf = np.arange(len(Pt) + 1, len(Pt) + 1 + len(future_forecasts))
        ax.plot(xf, future_forecasts, linestyle=':', linewidth=2.0, label='Out-of-sample', zorder=3)
        if intervals and 'future' in intervals:
            lo_fut, hi_fut = intervals['future']
            ax.fill_between(xf, lo_fut, hi_fut, alpha=0.18, label='CI (out-of-sample)', zorder=1)

    # Styling
    ax.grid(True, which='major', linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    ax.set_title('Exponential Smoothing and Forecasting', pad=12)
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')

    # Legend outside to the right
    leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    plt.tight_layout(rect=[0, 0, 0.82, 1])  # leave room for legend
    plt.show()

def exponential_smoother(Pt, k, alpha_range, l, plot=False, ci_level=0.90):
    x = np.arange(1, k + 1).reshape(-1, 1)
    y0 = Pt[:k]
    model = LinearRegression().fit(x, y0)
    beta0 = model.intercept_
    beta1 = model.coef_[0]

    lo = max(1e-2, alpha_range[0] + 1e-2)
    hi = min(0.99, alpha_range[1])
    alphas = np.linspace(lo, hi, num=100)

    best_err = np.inf
    best = None

    forecasts_list = []
    for alpha in alphas:
        S1_0, S2_0 = initial_smoothing_values(beta0, beta1, alpha)
        f_ins, errs, f_out = compute_forecast(alpha, Pt, S1_0, S2_0, l)
        sse = np.nansum(errs)
        forecasts_list.append((alpha, f_ins, f_out, errs))
        if sse < best_err:
            best_err = sse
            best = (alpha, f_ins, f_out, errs)

    alpha_opt, optimal_forecasts, future_forecasts, errs = best

    # intervals (normal approximation using in-sample residuals where defined)
    mask = ~np.isnan(optimal_forecasts)
    resid = Pt[mask] - optimal_forecasts[mask]
    if resid.size >= 2:
        sigma = float(np.std(resid, ddof=1))
        z = float(norm.ppf(0.5 + ci_level/2))
        lo_ins = optimal_forecasts.copy()
        hi_ins = optimal_forecasts.copy()
        lo_ins[mask] = optimal_forecasts[mask] - z * sigma
        hi_ins[mask] = optimal_forecasts[mask] + z * sigma

        lo_fut = future_forecasts - z * sigma
        hi_fut = future_forecasts + z * sigma
        intervals = {'insample': (lo_ins, hi_ins), 'future': (lo_fut, hi_fut)}
    else:
        intervals = None

    if plot:
        plot_results(Pt, optimal_forecasts, future_forecasts, intervals)

    return {'alpha_opt': float(alpha_opt), 'Forecasts': future_forecasts, 'intervals': intervals}

# Example
if __name__ == "__main__":
    Pt = np.array([12, 15, 14, 16, 19, 20, 22, 25, 24, 23], dtype=float)
    k = 3
    alpha_range = [0, 0.99]
    l = 2
    result = exponential_smoother(Pt, k, alpha_range, l, plot=True, ci_level=0.9)
    print(result['alpha_opt'], result['Forecasts'])
