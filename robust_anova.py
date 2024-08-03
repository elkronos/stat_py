import pandas as pd
import numpy as np
from scipy.stats import f_oneway, mannwhitneyu, levene, shapiro
import statsmodels.api as sm
from statsmodels.formula.api import ols, mixedlm
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import logging
from itertools import combinations

logging.basicConfig(level=logging.INFO)
SIGNIFICANCE_LEVEL = 0.05


def check_normality(*groups):
    """
    Check for the normality of the given groups using the Shapiro-Wilk test.

    Parameters:
    *groups (array-like): Variable number of array-like group data.

    Logs a warning if any group is not normally distributed.
    """
    for i, group in enumerate(groups):
        stat, p = shapiro(group)
        if p < SIGNIFICANCE_LEVEL:
            logging.warning(f"Group {i+1} may not be normally distributed (Shapiro-Wilk p-value: {p:.3f}).")


def check_homogeneity(*groups):
    """
    Check for the homogeneity of variances across given groups using Levene's test.

    Parameters:
    *groups (array-like): Variable number of array-like group data.

    Logs a warning if groups do not have equal variances.
    """
    stat, p = levene(*groups)
    if p < SIGNIFICANCE_LEVEL:
        logging.warning(f"Groups may not have equal variances (Levene's test p-value: {p:.3f}).")


def glm_anova(groups):
    """
    Perform General Linear Model (GLM) ANOVA on given groups.

    Parameters:
    groups (list of array-like): List of groups to be compared.

    Returns:
    dict: ANOVA results including test type, statistic, and p-value.
    """
    check_normality(*groups)
    check_homogeneity(*groups)

    statistic, p_value = f_oneway(*groups)
    return {'test_type': 'GLM ANOVA', 'statistic': statistic, 'p_value': p_value}


def tukey_hsd_posthoc(data, response_column, group_column):
    """
    Perform Tukey's Honestly Significant Difference (HSD) posthoc test.

    Parameters:
    data (pd.DataFrame): The dataset containing the response and group columns.
    response_column (str): The name of the response column in the dataset.
    group_column (str): The name of the group column in the dataset.

    Returns:
    DataFrame: Summary of the Tukey HSD test results.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a Pandas DataFrame.")
    if response_column not in data.columns or group_column not in data.columns:
        raise ValueError("Specified columns not found in the DataFrame.")

    tukey_results = pairwise_tukeyhsd(endog=data[response_column], 
                                      groups=data[group_column], 
                                      alpha=SIGNIFICANCE_LEVEL)
    return tukey_results.summary()


def mann_whitney_test(groups):
    """
    Perform the Mann-Whitney U test for non-parametric data comparison.

    Parameters:
    groups (list of array-like): List of groups to be compared.

    Returns:
    dict: Test results including comparison details, statistics, and p-values.
    """
    results = []
    for i, j in combinations(range(len(groups)), 2):
        statistic, p_value = mannwhitneyu(groups[i], groups[j])
        results.append({'comparison': f'Group {i+1} vs Group {j+1}', 'statistic': statistic, 'p_value': p_value})
    
    return {'test_type': 'Mann-Whitney U', 'results': results}


def two_way_anova(data, formula, typ=2):
    """
    Perform a Two-Way ANOVA test.

    Parameters:
    data (pd.DataFrame): The dataset for ANOVA analysis.
    formula (str): The formula representing the model to be fitted.
    typ (int): The type of ANOVA test to perform.

    Returns:
    DataFrame: ANOVA test results.
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Data must be a Pandas DataFrame.")
    if not isinstance(formula, str):
        raise ValueError("Formula must be a string.")

    model = ols(formula, data=data).fit()

    formula_components = formula.split('~')[1].strip().split('+')
    factor_names = [comp.split('(')[1].split(')')[0] for comp in formula_components if 'C(' in comp]
    
    if factor_names:
        combined_groups = data[factor_names].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
        groups = [data[formula.split('~')[0].strip()][combined_groups == level] for level in combined_groups.unique()]

        check_homogeneity(*groups)
    else:
        logging.warning("No categorical factors for homogeneity check in two-way ANOVA.")

    anova_table = sm.stats.anova_lm(model, typ=typ)
    return anova_table


def repeated_measures_anova(data, formula, subject_column):
    """
    Perform Repeated Measures ANOVA.

    Parameters:
    data (pd.DataFrame): The dataset for ANOVA analysis.
    formula (str): The formula representing the model to be fitted.
    subject_column (str): The name of the subject column in the dataset.

    Returns:
    tuple: Summary of the mixed linear model fit and ANOVA test results.
    """
    model = mixedlm(formula, data, groups=data[subject_column])
    result = model.fit()

    residuals = result.resid
    check_normality(residuals)

    dependent_var = formula.split('~')[0].strip()
    within_factors = [factor.strip() for factor in formula.split('~')[1].strip().split('+')]
    rm_anova = AnovaRM(data, dependent_var, subject_column, within=within_factors)
    rm_result = rm_anova.fit()

    return result.summary(), rm_result


def robust_anova(groups=None, data=None, test_type='GLM', formula=None, subject_column=None, typ=2):
    """
    Perform a robust ANOVA analysis based on specified test type.

    Parameters:
    groups (list of array-like): List of groups for GLM and Mann-Whitney tests.
    data (pd.DataFrame): The dataset for Two-way and Repeated Measures ANOVA.
    test_type (str): The type of ANOVA test to perform.
    formula (str): The formula for the model in Two-way and Repeated Measures ANOVA.
    subject_column (str): The subject column for Repeated Measures ANOVA.
    typ (int): The type of Two-way ANOVA test to perform.

    Returns:
    Varies: The result of the chosen ANOVA test.
    """
    if test_type == 'GLM':
        return glm_anova(groups)
    elif test_type == 'Mann-Whitney':
        return mann_whitney_test(groups)
    elif test_type == 'Two-way':
        return two_way_anova(data, formula, typ)
    elif test_type == 'Repeated Measures':
        return repeated_measures_anova(data, formula, subject_column)
    else:
        raise ValueError("Invalid test type. Choose 'GLM', 'Mann-Whitney', 'Two-way', or 'Repeated Measures'.")


# Example usage:

# Mann-Whitney U Test
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(1, 1, 30)
print(robust_anova(groups=[group1, group2], test_type='Mann-Whitney'))

# Two-Way ANOVA
df = pd.DataFrame({
    'response': np.random.randn(60),
    'factor1': np.repeat(['A', 'B'], 30),
    'factor2': np.tile(['C', 'D', 'E'], 20)
})
print(robust_anova(data=df, formula='response ~ C(factor1) + C(factor2) + C(factor1):C(factor2)', test_type='Two-way'))

# Repeated Measures ANOVA
df = pd.DataFrame({
    'response': np.random.randn(90),
    'time': np.tile(['T1', 'T2', 'T3'], 30),
    'subject': np.repeat(np.arange(30), 3)
})
print(robust_anova(data=df, formula='response ~ time', subject_column='subject', test_type='Repeated Measures'))

# One-way ANOVA (GLM) with Tukey HSD post-hoc test
group1 = np.random.normal(0, 1, 30)
group2 = np.random.normal(1, 1, 30)
group3 = np.random.normal(2, 1, 30)

anova_result = robust_anova(groups=[group1, group2, group3], test_type='GLM')
print(anova_result)

if anova_result['p_value'] < 0.05:
    df_tukey = pd.DataFrame({
        'response': np.concatenate([group1, group2, group3]),
        'group': np.repeat(['Group1', 'Group2', 'Group3'], 30)
    })
    tukey_result = tukey_hsd_posthoc(df_tukey, 'response', 'group')
    print(tukey_result)
else:
    print("No significant difference found in ANOVA. Tukey test not performed.")