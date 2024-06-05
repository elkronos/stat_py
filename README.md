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


## Exponential Smoothing and Forecasting for Time Series Data

This Python script provides a utility function for applying second-order exponential smoothing to time series data and computing forecasts l-steps ahead from each point in the series. It fits a linear model to the initial set of data points to derive starting values for smoothing. The function tests a range of alpha values, computes smoothing and forecasts for each, and selects the optimal alpha that minimizes the sum of squared errors. Optionally, it can plot the original data points alongside the forecasts for visual comparison.

### Key Features

1. **Exponential Smoothing**: Applies second-order exponential smoothing to time series data.
2. **Forecasting**: Computes forecasts l-steps ahead from each point in the series.
3. **Alpha Optimization**: Tests a range of alpha values and selects the optimal one that minimizes the sum of squared errors.
4. **Visualization**: Optionally plots the original data points alongside the forecasts for visual comparison.

### Utility Functions

- `initial_smoothing_values(beta0, beta1, alphas)`: Computes initial smoothing values S1_0 and S2_0.
- `compute_forecast(alpha, Pt, S1_0, S2_0, l)`: Computes forecast for a given alpha.
- `plot_results(Pt, optimal_forecasts)`: Plots the original data points alongside the forecasts.
- `exponential_smoother(Pt, k, alpha_range, l, plot=False)`: Applies second-order exponential smoothing to time series data, tests a range of alpha values, and computes forecasts l-steps ahead.

### Dependencies

To run this script, ensure you have the following libraries installed:
- numpy
- pandas
- matplotlib
- scikit-learn


## MNIST Classifier with Early Stopping

This Python script builds and trains a neural network to classify handwritten digits using the MNIST dataset. It leverages PyTorch for model construction and training, with an implementation of early stopping for enhanced training efficiency.

### Key Features

1. **MNIST Dataset Loading and Preprocessing**: Loads the MNIST dataset, applies transformations like random rotations and flips, and normalizes the data.
2. **Neural Network Construction**: Defines a multi-layer neural network with dropout layers to prevent overfitting.
3. **Training with Early Stopping**: Implements a training loop with early stopping based on validation loss to prevent overtraining.
4. **Performance Metrics**: Calculates accuracy and displays a confusion matrix for performance evaluation.

### Neural Network Architecture

The script defines a `NeuralNetwork` class with the following layers:
- Linear layer (784 to 256 neurons) with ReLU activation and dropout
- Linear layer (256 to 128 neurons) with ReLU activation and dropout
- Linear layer (128 to 64 neurons) with ReLU activation and dropout
- Output linear layer (64 to 10 neurons)

### Training Process

The script trains the model using the Adam optimizer and a step learning rate scheduler. The training loop includes:
- Loss computation using Cross-Entropy Loss
- Backpropagation and optimizer step
- Validation loss calculation for early stopping
- Accuracy and confusion matrix computation for each epoch

### Dependencies

To run this script, ensure you have the following libraries installed:
- torch
- torchvision
- sklearn
- numpy


## Polychoric Correlation

This Python script calculates the polychoric correlation coefficient between two ordinal variables. It's specifically designed to handle datasets where the underlying variables are assumed to follow a bivariate normal distribution. The script uses the `numpy` and `scipy` libraries to perform the calculations.

### Key Features

1. **Polychoric Correlation Calculation**: Computes the polychoric correlation coefficient, which is useful for understanding relationships between ordinal variables.
2. **Data Preprocessing**: Includes functionality for preprocessing data, such as removing rows and columns with zero marginal.
3. **Thresholds Calculation**: Determines the optimal threshold values for the ordinal variables.
4. **Maximum Likelihood Estimation**: Uses Maximum Likelihood Estimation (MLE) for computing the polychoric correlation.
5. **Standard Error and Confidence Interval Calculation**: Optionally calculates standard errors and confidence intervals for the correlation coefficient.

### Utility Functions

- `polychor(x, y=None, ML=False, std_err=False, maxcor=0.9999, start=None, thresholds=False)`: Main function to compute the polychoric correlation. Parameters:
  - `x`, `y`: Input arrays or ordinal data.
  - `ML`: If set to True, uses MLE for estimation.
  - `std_err`: If set to True, calculates the standard error.
  - `maxcor`: Maximum correlation threshold.
  - `start`: Starting values for optimization.
  - `thresholds`: If set to True, returns the calculated thresholds.

### Dependencies

To run this script, ensure you have the following libraries installed:
- numpy
- scipy


## Robust ANOVA and Posthoc Tests

This Python script offers comprehensive functions to perform various types of ANOVA (Analysis of Variance) tests, including General Linear Model (GLM) ANOVA, Two-Way ANOVA, Repeated Measures ANOVA, and the Mann-Whitney U test for non-parametric data. It utilizes libraries such as `pandas`, `numpy`, `scipy.stats`, `statsmodels.api`, and `logging`.

### Key Features

1. **Various ANOVA Tests**: Supports GLM ANOVA, Two-Way ANOVA, Repeated Measures ANOVA, and Mann-Whitney U tests.
2. **Normality and Homogeneity Checks**: Functions to check the normality of data (Shapiro-Wilk test) and homogeneity of variances (Levene's test).
3. **Tukey HSD Posthoc Test**: If ANOVA results are significant, perform Tukey's Honestly Significant Difference posthoc test.
4. **Robust Analysis**: Capable of handling both parametric and non-parametric data.

### Utility Functions

- `check_normality(*groups)`: Checks the normality of given groups using the Shapiro-Wilk test.
- `check_homogeneity(*groups)`: Checks the homogeneity of variances across given groups using Levene's test.
- `glm_anova(groups)`: Performs GLM ANOVA on given groups.
- `tukey_hsd_posthoc(data, response_column, group_column)`: Performs Tukey's HSD posthoc test.
- `mann_whitney_test(groups)`: Performs the Mann-Whitney U test for non-parametric data comparison.
- `two_way_anova(data, formula, typ=2)`: Performs a Two-Way ANOVA test.
- `repeated_measures_anova(data, formula, subject_column)`: Performs Repeated Measures ANOVA.
- `robust_anova(groups=None, data=None, test_type='GLM', formula=None, subject_column=None, typ=2)`: Wrapper function to perform the specified type of ANOVA test.

### Dependencies

To run this script, ensure you have the following libraries installed:
- pandas
- numpy
- scipy.stats
- statsmodels.api
- logging


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
