import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor

def check_linearity(X, y, predictions):
    """
    This function checks the linearity assumption using seaborn's pairplot to visualize 
    the relationships between the variables.
    
    :param X: pd.DataFrame, predictor variables
    :param y: pd.Series, response variable
    :param predictions: pd.Series, predictions made by the model
    """
    sns.pairplot(pd.concat([X, y], axis=1))
    plt.title('Linearity Check: Pairplot')
    plt.show()

def check_normality(residuals):
    """
    This function checks the normality of the residuals by plotting a histogram and 
    conducting the Shapiro-Wilk test.
    
    :param residuals: pd.Series, residuals of the model
    """
    sns.histplot(residuals, bins=30, kde=True)
    plt.title('Normality Check: Residuals Histogram')
    plt.show()
    
    _, p_value = stats.shapiro(residuals)
    print(f"Shapiro-Wilk normality test p-value: {p_value:.5f}")

def check_homoscedasticity(X, predictions, residuals):
    """
    This function checks for homoscedasticity using a scatter plot of residuals versus predicted values and a Breusch-Pagan test (if applicable).
    
    Parameters:
    X (pd.DataFrame): Predictor variables
    predictions (pd.Series): Predicted values from the regression model
    residuals (pd.Series): Residuals from the regression model
    """
    sns.scatterplot(x=predictions, y=residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted values')
    plt.ylabel('Residuals')
    plt.title('Homoscedasticity Check')
    
    # Conduct Breusch-Pagan test if there are two or more predictor variables
    if X.shape[1] > 1:
        from statsmodels.stats.diagnostic import het_breuschpagan
        bp_test = het_breuschpagan(residuals, X)
        bp_test_result = f'BP Test Statistic: {bp_test[0]:.5f}, p-value: {bp_test[1]:.5f}'
    else:
        bp_test_result = 'Breusch-Pagan test not applicable for univariate regression'
    
    plt.annotate(bp_test_result, xy=(0.05, 0.95), xycoords='axes fraction')
    plt.show()

def check_independence(residuals):
    """
    This function checks the independence of the residuals by plotting them against the index 
    and conducting the Ljung-Box test.
    
    :param residuals: pd.Series, residuals of the model
    """
    plt.plot(residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Independence Check: Residuals vs. Index')
    plt.show()
    
    lb_value, p_value = acorr_ljungbox(residuals, lags=[10])
    if isinstance(p_value[0], str):
        print(f"Ljung-Box test message: {p_value[0]}")
    else:
        print(f"Ljung-Box test p-value: {p_value[0]:.5f}")


def check_multicollinearity(X):
    """
    This function checks for multicollinearity among predictor variables using the variance inflation factor (VIF)
    and a correlation matrix heatmap.
    
    :param X: pd.DataFrame, predictor variables
    """
    # Check multicollinearity using Variance Inflation Factor (VIF)
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    print("Variance Inflation Factor (VIF)")
    print(vif_data)
    
    # Plotting correlation matrix heatmap
    correlation_matrix = X.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix Heatmap')
    plt.show()

# Plot the model
def plot_regression_model(X, y, predictions):
    """
    This function plots the regression model, including observed values, predicted values, and the regression line/plane.

    Parameters:
    X (pd.DataFrame): Predictor variables
    y (pd.Series): Response variable
    predictions (pd.Series): Predicted values from the regression model
    """
    if X.shape[1] == 1:
        sns.scatterplot(x=X.iloc[:, 0], y=y, color='blue', label='Observed')
        sns.scatterplot(x=X.iloc[:, 0], y=predictions, color='red', label='Predicted')
        sns.regplot(x=X.iloc[:, 0], y=predictions, color='green', label='Regression Line', scatter=False)
        plt.title('Regression Model')
    else:
        print("Multivariate plot not available for more than one predictor variable.")
    plt.show()

# Make predictions using the model
def make_predictions(model, new_data):
    """
    This function uses the fitted model to make predictions on new data.

    Parameters:
    model (statsmodels.regression.linear_model.RegressionResultsWrapper): Fitted regression model
    new_data (pd.DataFrame): New data for which predictions are to be made

    Returns:
    pd.Series: Predictions
    """
    new_data = sm.add_constant(new_data)
    predictions = model.predict(new_data)
    return predictions

def linear_regression_and_check_assumptions(X, y):
    """
    This function performs linear regression and checks the common assumptions: 
    linearity, normality of residuals, homoscedasticity, independence of residuals, and multicollinearity.
    
    :param X: pd.DataFrame, predictor variables
    :param y: pd.Series, response variable
    
    :return: dict, a dictionary containing the regression model, predictions, and residuals
    """
    # Error handling for invalid input data
    if X is None or y is None or X.empty or y.empty:
        raise ValueError("Input data (X, y) cannot be None or empty")
    
    # Adding a constant term to the predictor
    X = sm.add_constant(X)
    
    # Create a model and fit it
    model = sm.OLS(y, X).fit()
    
    # Get the predictions
    predictions = model.predict(X)
    
    # Calculate residuals
    residuals = y - predictions
    
    # Creating a dictionary to store results
    results = {
        "Model Summary": model.summary(),
        "Predictions": predictions,
        "Residuals": residuals
    }
    
    # Calling individual check functions
    check_linearity(X, y, predictions)
    check_normality(residuals)
    check_homoscedasticity(X, predictions, residuals)
    check_independence(residuals)
    check_multicollinearity(X)
    
    # Print Durbin-Watson statistic
    dw_test = durbin_watson(residuals)
    print(f"Durbin-Watson statistic: {dw_test:.5f} (Values close to 2 suggest no autocorrelation)")
    
    plot_regression_model(X, y, predictions)

    return results

# Example usage
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = pd.DataFrame({'X1': np.linspace(0, 100, 100), 'X2': np.linspace(0, 200, 100)})
    y = 3*X['X1'] + 2*X['X2'] + np.random.normal(0, 10, size=100)
    
    # Calling the function
    results = linear_regression_and_check_assumptions(X, y)
    
    # Print the model summary
    print(results["Model Summary"])