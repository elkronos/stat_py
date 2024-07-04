# An attempt to implement and enhance this R script in python: https://github.com/elkronos/public_examples/blob/main/helpers/outliers.R

import numpy as np
import pandas as pd
import logging
from scipy.stats import zscore, chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration parameters
DEFAULT_CONFIG = {
    "z_thresh": 1.96,
    "tukey_mult": 1.5,
    "mahalanobis_thresh": 0.95,
    "grubbs_thresh": 0.05,
    "mad_mult": 3,
    "iglewicz_hoaglin_thresh": 3.5,
    "isolation_forest_contamination": 0.1,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "one_class_svm_nu": 0.05,
    "elliptic_envelope_contamination": 0.1,
    "lof_n_neighbors": 20,
    "lof_thresh": 1
}

def get_config_param(param):
    return DEFAULT_CONFIG.get(param)

class OutlierDetection:

    @staticmethod
    def validate_params(data, column):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")
        logging.info(f"Validated parameters for column: {column}")

    @staticmethod
    def z_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        z_thresh = float(get_config_param('z_thresh'))
        z_scores = zscore(data[column])
        return (np.abs(z_scores) > z_thresh).astype(int)

    @staticmethod
    def tukey_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        tukey_mult = float(get_config_param('tukey_mult'))
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - tukey_mult * IQR
        upper_bound = Q3 + tukey_mult * IQR
        return ((data[column] < lower_bound) | (data[column] > upper_bound)).astype(int)

    @staticmethod
    def mahalanobis_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        mahalanobis_thresh = float(get_config_param('mahalanobis_thresh'))
        X = data.select_dtypes(include=[np.number]).drop(columns=[column])
        cov = np.cov(X, rowvar=False)
        inv_cov = np.linalg.inv(cov)
        mean = X.mean(axis=0)
        diff = X - mean
        md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        threshold = chi2.ppf(mahalanobis_thresh, df=X.shape[1])
        return (md > threshold).astype(int)

    @staticmethod
    def grubbs_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        grubbs_thresh = float(get_config_param('grubbs_thresh'))
        n = len(data)
        mean_val = data[column].mean()
        std_dev = data[column].std()
        max_dev = np.max(np.abs(data[column] - mean_val))
        G = max_dev / std_dev
        p_value = 1 - (1 - grubbs_thresh / (2 * n))
        critical_value = (n - 1) / np.sqrt(n) * np.sqrt(chi2.ppf(p_value, df=n - 2) / (n - 2 + chi2.ppf(p_value, df=n - 2)))
        return (G > critical_value).astype(int)

    @staticmethod
    def mad_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        mad_mult = float(get_config_param('mad_mult'))
        med = data[column].median()
        mad = np.median(np.abs(data[column] - med))
        mad_thresh = mad_mult * mad
        return (np.abs(data[column] - med) > mad_thresh).astype(int)

    @staticmethod
    def iglewicz_hoaglin_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        threshold = float(get_config_param('iglewicz_hoaglin_thresh'))
        med = data[column].median()
        mad = np.median(np.abs(data[column] - med))
        modified_z = 0.6745 * (data[column] - med) / mad
        return (np.abs(modified_z) > threshold).astype(int)

    @staticmethod
    def isolation_forest_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        contamination = float(get_config_param('isolation_forest_contamination'))
        X = data[[column]]
        clf = IsolationForest(contamination=contamination)
        clf.fit(X)
        return (clf.predict(X) == -1).astype(int)

    @staticmethod
    def dbscan_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        eps = float(get_config_param('dbscan_eps'))
        min_samples = int(get_config_param('dbscan_min_samples'))
        X = data[[column]].values
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        return (db.labels_ == -1).astype(int)

    @staticmethod
    def one_class_svm_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        nu = float(get_config_param('one_class_svm_nu'))
        X = data[[column]]
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma='auto')
        clf.fit(X)
        return (clf.predict(X) == -1).astype(int)

    @staticmethod
    def elliptic_envelope_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        contamination = float(get_config_param('elliptic_envelope_contamination'))
        X = data.select_dtypes(include=[np.number])
        clf = MinCovDet(support_fraction=1 - contamination).fit(X)
        md = clf.mahalanobis(X)
        threshold = chi2.ppf((1 - contamination), df=X.shape[1])
        return (md > threshold).astype(int)

    @staticmethod
    def lof_outliers(data, column):
        OutlierDetection.validate_params(data, column)
        n_neighbors = int(get_config_param('lof_n_neighbors'))
        X = data[[column]]
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        y_pred = lof.fit_predict(X)
        return (y_pred == -1).astype(int)

    @staticmethod
    def plot_outliers(data, outliers_dict):
        method_names = list(outliers_dict.keys())
        variable_names = list(data.select_dtypes(include=[np.number]).columns)

        for method in method_names:
            outlier_counts = []
            for column in variable_names:
                outlier_counts.append(np.sum(outliers_dict[method][column]))

            total_counts = len(data)
            outlier_percents = [count / total_counts * 100 for count in outlier_counts]
            non_outlier_percents = [100 - percent for percent in outlier_percents]

            fig, ax = plt.subplots(figsize=(12, 6))
            bars_out = ax.bar(variable_names, outlier_percents, color='#FFA07A', edgecolor='white', label='Outliers')
            bars_non = ax.bar(variable_names, non_outlier_percents, bottom=outlier_percents, color='#1E90FF', edgecolor='white', label='Non-Outliers')

            for bar_out, bar_non, out_percent, non_out_percent in zip(bars_out, bars_non, outlier_percents, non_outlier_percents):
                height_out = bar_out.get_height()
                height_non = bar_non.get_height()
                if out_percent > 0:
                    ax.annotate(f'{out_percent:.1f}%', xy=(bar_out.get_x() + bar_out.get_width() / 2, height_out / 2),
                                xytext=(0, 0), textcoords="offset points", ha='center', color='black', weight='bold')
                if non_out_percent > 0:
                    ax.annotate(f'{non_out_percent:.1f}%', xy=(bar_non.get_x() + bar_non.get_width() / 2, height_out + height_non / 2),
                                xytext=(0, 0), textcoords="offset points", ha='center', color='black', weight='bold')

            ax.set_xlabel('Variable')
            ax.set_ylabel('Percentage')
            ax.set_title(f'Percentage of Outliers Detected by {method} Method')
            ax.legend()
            plt.xticks(rotation=45)
            sns.despine()
            plt.show()

        # Individual scatter plots for each method
        for method, columns in outliers_dict.items():
            for column, outliers in columns.items():
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=data.index, y=data[column], hue=outliers, palette={0: '#1E90FF', 1: '#FFA07A'}, legend=None)
                plt.title(f'Outliers in {column} Detected by {method}')
                plt.xlabel('Index')
                plt.ylabel(column)
                plt.legend(title='Outlier', loc='upper right', labels=['Non-Outlier', 'Outlier'])
                sns.despine()
                plt.show()

def detect_outliers(data, methods=None, columns=None):
    if methods is None:
        methods = ["z", "tukey", "mahalanobis", "grubbs", "mad", "iglewicz_hoaglin", "isolation_forest", "dbscan", "one_class_svm", "elliptic_envelope", "lof"]

    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()

    results = pd.DataFrame(index=data.index)
    od = OutlierDetection()
    outliers_dict = {method: {column: None for column in columns} for method in methods}

    method_map = {
        "z": od.z_outliers,
        "tukey": od.tukey_outliers,
        "mahalanobis": od.mahalanobis_outliers,
        "grubbs": od.grubbs_outliers,
        "mad": od.mad_outliers,
        "iglewicz_hoaglin": od.iglewicz_hoaglin_outliers,
        "isolation_forest": od.isolation_forest_outliers,
        "dbscan": od.dbscan_outliers,
        "one_class_svm": od.one_class_svm_outliers,
        "elliptic_envelope": od.elliptic_envelope_outliers,
        "lof": od.lof_outliers
    }

    for column in columns:
        for method in methods:
            try:
                if method in method_map:
                    logging.info(f"Applying {method} method on column: {column}")
                    outliers = method_map[method](data, column)
                    results[f"{column}_{method}_outlier"] = outliers
                    outliers_dict[method][column] = outliers
            except Exception as e:
                logging.error(f"Error applying {method} method on column {column}: {e}")
    
    od.plot_outliers(data, outliers_dict)
    return results

# Example Usage
np.random.seed(123)
example_data = pd.DataFrame({
    'A': np.random.normal(10, 2, 100),
    'B': np.random.normal(20, 5, 100),
    'C': np.random.normal(30, 10, 100)
})
# Add some artificial outliers
example_data.loc[[25, 50, 75], 'B'] = [40, 10, 35]
example_data.loc[[10, 90], 'C'] = [5, 50]

# Detect outliers using specified methods
methods = ["z", "tukey", "mahalanobis", "grubbs", "mad", "iglewicz_hoaglin", "isolation_forest", "dbscan", "one_class_svm", "elliptic_envelope", "lof"]
outlier_results = detect_outliers(example_data, methods=methods)

# Review results for each method
print(outlier_results.head())
