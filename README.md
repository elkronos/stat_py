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
