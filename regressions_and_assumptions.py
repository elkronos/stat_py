import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

class AssumptionChecks:
    @staticmethod
    def check_linearity(X, y, predictions):
        sns.pairplot(pd.concat([X, y], axis=1))
        plt.suptitle('Linearity Check: Pairplot', y=1.02)
        plt.show()

    @staticmethod
    def check_normality(residuals):
        sns.histplot(residuals, bins=30, kde=True)
        plt.title('Normality Check: Residuals Histogram')
        plt.show()
        
        _, p_value = stats.shapiro(residuals)
        print(f"Shapiro-Wilk normality test p-value: {p_value:.5f}")

    @staticmethod
    def check_homoscedasticity(X, predictions, residuals):
        sns.scatterplot(x=predictions, y=residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted values')
        plt.ylabel('Residuals')
        plt.title('Homoscedasticity Check')
        
        if X.shape[1] > 1:
            bp_test = het_breuschpagan(residuals, X)
            bp_test_result = f'BP Test Statistic: {bp_test[0]:.5f}, p-value: {bp_test[1]:.5f}'
        else:
            bp_test_result = 'Breusch-Pagan test not applicable for univariate regression'
        
        plt.annotate(bp_test_result, xy=(0.05, 0.95), xycoords='axes fraction')
        plt.show()

    @staticmethod
    def check_independence(residuals):
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.title('Independence Check: Residuals vs. Index')
        plt.show()
        
        lb_value, p_value = acorr_ljungbox(residuals, lags=[10])
        if isinstance(p_value[0], str):
            print(f"Ljung-Box test message: {p_value[0]}")
        else:
            print(f"Ljung-Box test p-value: {p_value[0]:.5f}")

    @staticmethod
    def check_multicollinearity(X):
        vif_data = pd.DataFrame()
        vif_data["feature"] = X.columns
        vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
        print("Variance Inflation Factor (VIF)")
        print(vif_data)
        
        correlation_matrix = X.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap')
        plt.show()

def plot_regression_model(X, y, predictions):
    if X.shape[1] == 1:
        sns.scatterplot(x=X.iloc[:, 0], y=y, color='blue', label='Observed')
        sns.scatterplot(x=X.iloc[:, 0], y=predictions, color='red', label='Predicted')
        sns.regplot(x=X.iloc[:, 0], y=predictions, color='green', label='Regression Line', scatter=False)
        plt.title('Regression Model')
        plt.legend()
    else:
        print("Multivariate plot not available for more than one predictor variable.")
    plt.show()

def make_predictions(model, new_data):
    new_data = sm.add_constant(new_data)
    predictions = model.predict(new_data)
    return predictions

def linear_regression_and_check_assumptions(X, y):
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("Input data (X, y) cannot be None or empty")
    
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)
    residuals = y - predictions
    
    results = {
        "Model Summary": model.summary(),
        "Predictions": predictions,
        "Residuals": residuals
    }
    
    try:
        AssumptionChecks.check_linearity(X, y, predictions)
    except Exception as e:
        print(f"Error in checking linearity: {e}")
    
    try:
        AssumptionChecks.check_normality(residuals)
    except Exception as e:
        print(f"Error in checking normality: {e}")
    
    try:
        AssumptionChecks.check_homoscedasticity(X, predictions, residuals)
    except Exception as e:
        print(f"Error in checking homoscedasticity: {e}")
    
    try:
        AssumptionChecks.check_independence(residuals)
    except Exception as e:
        print(f"Error in checking independence: {e}")
    
    try:
        AssumptionChecks.check_multicollinearity(X)
    except Exception as e:
        print(f"Error in checking multicollinearity: {e}")
    
    try:
        dw_test = durbin_watson(residuals)
        print(f"Durbin-Watson statistic: {dw_test:.5f} (Values close to 2 suggest no autocorrelation)")
    except Exception as e:
        print(f"Error in Durbin-Watson test: {e}")
    
    try:
        plot_regression_model(X, y, predictions)
    except Exception as e:
        print(f"Error in plotting regression model: {e}")

    return results

if __name__ == "__main__":
    np.random.seed(0)
    X = pd.DataFrame({'X1': np.linspace(0, 100, 100), 'X2': np.linspace(0, 200, 100)})
    y = 3*X['X1'] + 2*X['X2'] + np.random.normal(0, 10, size=100)
    
    results = linear_regression_and_check_assumptions(X, y)
    print(results["Model Summary"])
