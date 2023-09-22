import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)

# Define constants
SIGNIFICANCE_LEVEL = 0.05

def check_normality(sample1, sample2):
    """
    Check the normality of two input samples using the Shapiro-Wilk test.

    Parameters
    ----------
    sample1, sample2 : array-like
        Input samples.

    Returns
    -------
    bool
        True if both samples are normally distributed, False otherwise.
    """
    _, p1 = stats.shapiro(sample1)
    _, p2 = stats.shapiro(sample2)
    
    if p1 < SIGNIFICANCE_LEVEL or p2 < SIGNIFICANCE_LEVEL:
        logging.warning("Data may not be normally distributed")
        return False
    return True

def check_variance(sample1, sample2):
    """
    Checks the equality of variances of two input samples using Levene's test.

    Parameters
    ----------
    sample1, sample2 : array-like
        Input samples.

    Returns
    -------
    bool
        True if samples have equal variances, False otherwise.
    """
    _, p = stats.levene(sample1, sample2)
    
    if p < SIGNIFICANCE_LEVEL:
        logging.warning("Variances are not equal")
        return False
    return True

def validate_input(tail, direction):
    """
    Validates the input parameters for tail and direction.

    Parameters
    ----------
    tail : str
        Specifies the type of t-test, should be "one" or "two".
    direction : str or None
        Specifies the direction of a one-tailed test, should be "greater", "less", or None.

    Raises
    ------
    ValueError
        If the input values are not valid.
    """
    valid_tails = ["one", "two"]
    valid_directions = ["greater", "less", None]
    
    if tail not in valid_tails:
        raise ValueError(f"Invalid value for tail: {tail}. It should be one of {valid_tails}.")
    
    if direction not in valid_directions:
        raise ValueError(f"Invalid value for direction: {direction}. It should be one of {valid_directions}.")
    
    if tail == "two" and direction is not None:
        raise ValueError("Direction should be None for a two-tailed test.")
    
    if tail == "one" and direction is None:
        raise ValueError("Direction should be specified for a one-tailed test.")

def perform_ttest(sample1, sample2, sample="independent", tail="two", direction=None):
    """
    Wrapper function to perform an independent or dependent t-test.

    Parameters
    ----------
    sample1, sample2 : array-like
        Input samples.
    sample : str, optional
        Type of t-test, should be "dependent" or "independent". Default is "independent".
    tail : str, optional
        Specifies the type of t-test, should be "one" or "two". Default is "two".
    direction : str or None, optional
        Specifies the direction of a one-tailed test, should be "greater", "less", or None. Default is None.

    Returns
    -------
    tuple
        t-statistic and the p-value.
    """
    validate_input(tail, direction)
    
    if not check_normality(sample1, sample2):
        logging.warning("At least one of the samples is not normally distributed, results may be unreliable")
    
    if sample == "dependent":
        return dependent_ttest(sample1, sample2, tail=tail, direction=direction)
    elif sample == "independent":
        return independent_ttest(sample1, sample2, tail=tail, direction=direction)
    else:
        raise ValueError("Invalid value for sample: {}. It should be 'dependent' or 'independent'.".format(sample))

def independent_ttest(sample1, sample2, tail="two", direction=None):
    """
    Perform an independent t-test.

    Parameters
    ----------
    sample1, sample2 : array-like
        Input samples.
    tail : str, optional
        Specifies the type of t-test, should be "one" or "two". Default is "two".
    direction : str or None, optional
        Specifies the direction of a one-tailed test, should be "greater", "less", or None. Default is None.

    Returns
    -------
    tuple
        t-statistic and the p-value.
    """
    equal_var = check_variance(sample1, sample2)
    
    t_stat, p_value = stats.ttest_ind(sample1, sample2, equal_var=equal_var)
    
    if tail == "one":
        p_value /= 2
        if (direction == "less" and t_stat > 0) or (direction == "greater" and t_stat < 0):
            p_value = 1 - p_value

    return t_stat, p_value

def dependent_ttest(sample1, sample2, tail="two", direction=None):
    """
    Perform a dependent t-test.

    Parameters
    ----------
    sample1, sample2 : array-like
        Input samples.
    tail : str, optional
        Specifies the type of t-test, should be "one" or "two". Default is "two".
    direction : str or None, optional
        Specifies the direction of a one-tailed test, should be "greater", "less", or None. Default is None.

    Returns
    -------
    tuple
        t-statistic and the p-value.
    """
    if len(sample1) != len(sample2):
        raise ValueError("Samples must be the same length for a dependent t-test")

    t_stat, p_value = stats.ttest_rel(sample1, sample2)

    if tail == "one":
        p_value /= 2
        if (direction == "less" and t_stat > 0) or (direction == "greater" and t_stat < 0):
            p_value = 1 - p_value

    return t_stat, p_value

def plot_means(sample1, sample2, filename=None):
    """
    Plots the means of the two samples with error bars.

    Parameters
    ----------
    sample1, sample2 : array-like
        Input samples.
    filename : str or None, optional
        The filename for saving the plot. If None, the plot will be displayed. Default is None.
    """
    means = [np.mean(sample1), np.mean(sample2)]
    errors = [stats.sem(sample1), stats.sem(sample2)]

    plt.bar(['Sample 1', 'Sample 2'], means, yerr=errors, color=['blue', 'green'])
    plt.ylabel('Mean')
    plt.title('Means of the two samples with error bars')
    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()

# Example usage
sample1 = np.random.randn(30) * 10 + 50
sample2 = np.random.randn(30) * 10 + 51

t_stat, p_value = perform_ttest(sample1, sample2, sample="independent", tail="two")
print("Independent two-tailed t-test:")
print("T-statistic:", t_stat)
print("P-value:", p_value)

t_stat, p_value = perform_ttest(sample1, sample2, sample="dependent", tail="one", direction="less")
print("\nDependent one-tailed t-test (less):")
print("T-statistic:", t_stat)
print("P-value:", p_value)

# Plotting the means
plot_means(sample1, sample2)