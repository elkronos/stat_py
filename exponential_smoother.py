import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def initial_smoothing_values(beta0, beta1, alphas):
    """Compute initial smoothing values S1_0 and S2_0."""
    S1_0 = beta0 - ((1 - alphas) / alphas) * beta1
    S2_0 = beta0 - ((2 * (1 - alphas)) / alphas) * beta1
    return S1_0, S2_0

def compute_forecast(alpha, Pt, S1_0, S2_0, l):
    """Compute forecast for a given alpha."""
    S1_t = np.zeros(len(Pt))
    S2_t = np.zeros(len(Pt))
    S1_t[0] = S1_0
    S2_t[0] = S2_0
    forecasts = np.zeros(len(Pt))
    errors = np.zeros(len(Pt))
    for i in range(1, len(Pt)):
        S1_t[i] = alpha * Pt[i] + (1 - alpha) * S1_t[i - 1]
        S2_t[i] = alpha * S1_t[i] + (1 - alpha) * S2_t[i - 1]
        
        # Calculate forecast for l-steps ahead
        a_val = 2 * S1_t[i] - S2_t[i]
        b_val = (alpha / (1 - alpha)) * (S1_t[i] - S2_t[i])
        if i + l < len(Pt):
            forecasts[i + l] = a_val + b_val * l
            errors[i + l] = (Pt[i + l] - forecasts[i + l])**2
    return forecasts, errors

def plot_results(Pt, optimal_forecasts):
    """Plot the original data points alongside the forecasts."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(Pt)), Pt, label='Actual', color='blue', linewidth=1.2)
    plt.plot(range(len(Pt)), optimal_forecasts, label='Forecast', color='red', linestyle='--', linewidth=1.2)
    plt.legend()
    plt.title('Exponential Smoothing and Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.show()

def exponential_smoother(Pt, k, alpha_range, l, plot=False):
    # Fit a linear model to the first k points to get initial estimates
    x = np.arange(1, k + 1).reshape(-1, 1)
    y = Pt[:k]
    model = LinearRegression().fit(x, y)
    beta0 = model.intercept_
    beta1 = model.coef_[0]
    
    # Generate alphas, avoiding zero and one
    alphas = np.linspace(alpha_range[0] + 0.01, min(alpha_range[1], 0.99), num=100)
    
    # Compute initial values for smoothing
    S1_0, S2_0 = initial_smoothing_values(beta0, beta1, alphas)
    
    # Prepare to store errors for each alpha
    errors_list = []
    forecasts_list = []
    
    # Loop through each alpha to find the one that minimizes the error
    for a, alpha in enumerate(alphas):
        forecasts, errors = compute_forecast(alpha, Pt, S1_0[a], S2_0[a], l)
        errors_list.append(np.sum(errors))
        forecasts_list.append(forecasts)
    
    # Determine the optimal alpha
    alpha_opt_index = np.argmin(errors_list)
    alpha_opt = alphas[alpha_opt_index]
    optimal_forecasts = forecasts_list[alpha_opt_index]
    
    # Optionally plot the results
    if plot:
        plot_results(Pt, optimal_forecasts)
    
    # Return only future forecasts
    future_forecasts = optimal_forecasts[len(Pt):]
    
    return {'alpha_opt': alpha_opt, 'Forecasts': future_forecasts}

# Example usage
Pt = np.array([12, 15, 14, 16, 19, 20, 22, 25, 24, 23])
k = 3
alpha_range = [0, 0.99]  # Avoiding alpha = 1 to prevent division by zero
l = 2  # Forecasting 2 steps ahead
result = exponential_smoother(Pt, k, alpha_range, l, plot=True)
print(result)