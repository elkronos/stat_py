import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore, chi2
from sklearn.covariance import MinCovDet
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration parameters
CONFIG = {
    "z_thresh": 2.0,
    "tukey_mult": 1.5,
    "mahalanobis_thresh": 0.975,
    "mad_mult": 3.5,
    "isolation_forest_contamination": 0.1,
    "dbscan_eps": 0.5,
    "dbscan_min_samples": 5,
    "one_class_svm_nu": 0.05,
    "elliptic_envelope_contamination": 0.1,
    "lof_n_neighbors": 20,
}

def validate_params(data, column):
    if column not in data.columns:
        raise ValueError(f"Column '{column}' not found in data.")
    if not pd.api.types.is_numeric_dtype(data[column]):
        raise ValueError(f"Column '{column}' must be numeric.")
    logging.info(f"Validated column: {column}")

def detect_outliers(data, methods=None, columns=None):
    if methods is None:
        methods = ["z", "tukey", "mahalanobis", "mad", "isolation_forest", "dbscan", "one_class_svm", "elliptic_envelope", "lof"]
    
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    
    results = pd.DataFrame(index=data.index)
    outliers_dict = {method: {column: None for column in columns} for method in methods}
    
    method_map = {
        "z": z_outliers,
        "tukey": tukey_outliers,
        "mahalanobis": mahalanobis_outliers,
        "mad": mad_outliers,
        "isolation_forest": isolation_forest_outliers,
        "dbscan": dbscan_outliers,
        "one_class_svm": one_class_svm_outliers,
        "elliptic_envelope": elliptic_envelope_outliers,
        "lof": lof_outliers
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
    
    plot_outliers(data, outliers_dict)
    return results

def z_outliers(data, column):
    validate_params(data, column)
    return (np.abs(zscore(data[column])) > CONFIG['z_thresh']).astype(int)

def tukey_outliers(data, column):
    validate_params(data, column)
    Q1, Q3 = data[column].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    return ((data[column] < (Q1 - CONFIG['tukey_mult'] * IQR)) | (data[column] > (Q3 + CONFIG['tukey_mult'] * IQR))).astype(int)

def mahalanobis_outliers(data, column):
    validate_params(data, column)
    X = data.select_dtypes(include=[np.number]).drop(columns=[column])
    cov = np.cov(X, rowvar=False)
    inv_cov = np.linalg.inv(cov)
    mean = X.mean(axis=0)
    diff = X - mean
    md = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
    return (md > chi2.ppf(CONFIG['mahalanobis_thresh'], df=X.shape[1])).astype(int)

def mad_outliers(data, column):
    validate_params(data, column)
    med = data[column].median()
    mad = np.median(np.abs(data[column] - med))
    return (np.abs(data[column] - med) > CONFIG['mad_mult'] * mad).astype(int)

def isolation_forest_outliers(data, column):
    validate_params(data, column)
    clf = IsolationForest(contamination=CONFIG['isolation_forest_contamination'])
    return (clf.fit_predict(data[[column]]) == -1).astype(int)

def dbscan_outliers(data, column):
    validate_params(data, column)
    db = DBSCAN(eps=CONFIG['dbscan_eps'], min_samples=CONFIG['dbscan_min_samples']).fit(data[[column]])
    return (db.labels_ == -1).astype(int)

def one_class_svm_outliers(data, column):
    validate_params(data, column)
    clf = OneClassSVM(nu=CONFIG['one_class_svm_nu'], kernel="rbf", gamma='auto')
    return (clf.fit_predict(data[[column]]) == -1).astype(int)

def elliptic_envelope_outliers(data, column):
    validate_params(data, column)
    clf = MinCovDet(support_fraction=1 - CONFIG['elliptic_envelope_contamination']).fit(data[[column]])
    md = clf.mahalanobis(data[[column]])
    return (md > chi2.ppf(1 - CONFIG['elliptic_envelope_contamination'], df=1)).astype(int)

def lof_outliers(data, column):
    validate_params(data, column)
    lof = LocalOutlierFactor(n_neighbors=CONFIG['lof_n_neighbors'])
    return (lof.fit_predict(data[[column]]) == -1).astype(int)

def plot_outliers(data, outliers_dict):
    for method, columns in outliers_dict.items():
        for column, outliers in columns.items():
            if outliers is not None:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=data.index, y=data[column], hue=outliers, palette={0: '#1E90FF', 1: '#FFA07A'})
                plt.title(f'Outliers in {column} Detected by {method}')
                plt.xlabel('Index')
                plt.ylabel(column)
                plt.show()
                plt.close()

# Example Usage
np.random.seed(123)
data = pd.DataFrame({
    'A': np.random.normal(10, 2, 100),
    'B': np.random.normal(20, 5, 100),
    'C': np.random.normal(30, 10, 100)
})
data.loc[[25, 50, 75], 'B'] = [40, 10, 35]
data.loc[[10, 90], 'C'] = [5, 50]
outlier_results = detect_outliers(data)
print(outlier_results.head())
