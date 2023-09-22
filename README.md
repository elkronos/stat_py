# stat_py
This repo contains statistics functions made in python.


## Linear Regression and Assumptions Checker

This Python script offers utility functions for performing linear regression and checking various assumptions related to the linear regression model. The script uses libraries such as `numpy`, `pandas`, `statsmodels`, `scipy`, `matplotlib`, and `seaborn`.

### Key Features

1. **Linear Regression**: The script performs linear regression using the Ordinary Least Squares (OLS) method.
2. **Assumption Checks**: The code includes functions to check five primary assumptions:
    - Linearity
    - Normality of residuals
    - Homoscedasticity (constant variance of residuals)
    - Independence of residuals
    - Absence of multicollinearity

### Utility Functions

- `check_linearity(X, y, predictions)`: Visualizes the relationships between variables using a pairplot.
- `check_normality(residuals)`: Checks the normality of residuals using a histogram and the Shapiro-Wilk test.
- `check_homoscedasticity(X, predictions, residuals)`: Checks for homoscedasticity using a scatter plot and the Breusch-Pagan test.
- `check_independence(residuals)`: Checks the independence of residuals using a plot and the Ljung-Box test.
- `check_multicollinearity(X)`: Checks for multicollinearity using the Variance Inflation Factor (VIF) and a correlation matrix heatmap.
- `plot_regression_model(X, y, predictions)`: Plots the regression model with observed values, predicted values, and the regression line.
- `make_predictions(model, new_data)`: Predicts the outcome using a fitted model.
- `linear_regression_and_check_assumptions(X, y)`: Performs the linear regression, plots the results, and checks all assumptions.

### Dependencies

To run this script, ensure you have the following libraries installed:
- numpy
- pandas
- statsmodels
- scipy
- matplotlib
- seaborn

## T-Test And Assumptions Checker

This Python script provides utility functions to perform both independent and dependent t-tests. The script utilizes libraries such as `numpy`, `scipy`, `matplotlib`, and `logging`.

### Key Features

1. **T-Test**: Offers functions to perform independent and dependent t-tests with options for one-tailed or two-tailed tests.
2. **Normality and Variance Check**: Includes functions to check the normality of the samples and the equality of variances.
3. **Input Validation**: Validates the input parameters for tail and direction.
4. **Plotting Means**: Plots the means of the two samples with error bars.

### Utility Functions

- `check_normality(sample1, sample2)`: Checks the normality of two input samples using the Shapiro-Wilk test.
- `check_variance(sample1, sample2)`: Checks the equality of variances of two input samples using Levene's test.
- `validate_input(tail, direction)`: Validates the input parameters for tail and direction.
- `perform_ttest(sample1, sample2, sample="independent", tail="two", direction=None)`: Wrapper function to perform an independent or dependent t-test.
- `independent_ttest(sample1, sample2, tail="two", direction=None)`: Perform an independent t-test.
- `dependent_ttest(sample1, sample2, tail="two", direction=None)`: Perform a dependent t-test.
- `plot_means(sample1, sample2, filename=None)`: Plots the means of the two samples with error bars.

### Dependencies

To run this script, ensure you have the following libraries installed:
- numpy
- scipy
- matplotlib
- logging
