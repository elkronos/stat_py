import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import zscore, chi2, t
from sklearn.covariance import MinCovDet, EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration parameters
DEFAULT_CONFIG = {
    "z_thresh": 1.96,
    "tukey_mult": 1.5,
    "mahalanobis_thresh": 0.95,  # quantile threshold for chi-square (e.g., 95% quantile)
    "grubbs_alpha": 0.05,        # significance level for Grubbs' test
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


def get_config_param(param: str, config: dict = DEFAULT_CONFIG):
    """Helper to fetch a configuration parameter."""
    return config.get(param)


class OutlierDetection:
    """
    A collection of static methods to detect outliers in a pandas DataFrame
    using different statistical and machine learning techniques.
    """

    @staticmethod
    def validate_params(data: pd.DataFrame, column: str) -> None:
        """Ensure that the column exists and contains numeric data."""
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data.")
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")
        if data[column].dropna().empty:
            raise ValueError(f"Column '{column}' has no valid numeric data.")
        logging.info(f"Validated parameters for column: {column}")

    @staticmethod
    def z_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers based on the Z-score method.
        Returns a binary numpy array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        z_thresh = float(get_config_param('z_thresh', config))
        col_data = data[column]
        if col_data.std() == 0:
            logging.warning(f"Standard deviation of column '{column}' is zero; no variation to detect outliers.")
            return np.zeros(len(col_data), dtype=int)
        zs = zscore(col_data, nan_policy='omit')
        outliers = (np.abs(zs) > z_thresh).astype(int)
        return outliers

    @staticmethod
    def tukey_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using Tukey's method (IQR rule).
        Returns a binary numpy array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        tukey_mult = float(get_config_param('tukey_mult', config))
        col_data = data[column]
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - tukey_mult * IQR
        upper_bound = Q3 + tukey_mult * IQR
        outliers = ((col_data < lower_bound) | (col_data > upper_bound)).astype(int)
        return outliers

    @staticmethod
    def mahalanobis_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers based on the Mahalanobis distance.
        This method uses all numeric columns (except the target column) to build the covariance matrix.
        If no additional columns exist, returns zeros.
        """
        OutlierDetection.validate_params(data, column)
        threshold_quantile = float(get_config_param('mahalanobis_thresh', config))
        # Use all numeric columns except the one of interest
        other_cols = data.select_dtypes(include=[np.number]).columns.drop(column, errors='ignore')
        if other_cols.empty:
            logging.warning(f"No additional numeric columns for Mahalanobis detection on '{column}'.")
            return np.zeros(len(data), dtype=int)
        X = data[other_cols].dropna()
        if X.shape[0] < X.shape[1] + 1:
            logging.warning(f"Insufficient data to compute covariance for Mahalanobis detection on '{column}'.")
            return np.zeros(len(data), dtype=int)
        try:
            cov_matrix = np.cov(X, rowvar=False)
            inv_cov = np.linalg.pinv(cov_matrix)
        except Exception as e:
            logging.error(f"Error computing covariance for Mahalanobis method: {e}")
            return np.zeros(len(data), dtype=int)
        mean_vec = X.mean(axis=0)
        diff = X - mean_vec
        md = np.sqrt(np.sum(diff.dot(inv_cov) * diff, axis=1))
        # Chi-square threshold (take square root because md is a distance)
        df = X.shape[1]
        threshold = np.sqrt(chi2.ppf(threshold_quantile, df))
        # Map computed distances back to the full dataset (set non-calculated rows as 0)
        outlier_flags = pd.Series(0, index=data.index)
        outlier_flags.loc[X.index] = (md > threshold).astype(int)
        return outlier_flags.values

    @staticmethod
    def grubbs_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect a single outlier using Grubbs' test.
        Returns a binary array marking the most extreme point as an outlier if significant.
        """
        OutlierDetection.validate_params(data, column)
        alpha = float(get_config_param('grubbs_alpha', config))
        col_data = data[column].dropna()
        n = len(col_data)
        if n < 3:
            logging.warning(f"Not enough data points in '{column}' for Grubbs' test (n={n}).")
            return np.zeros(len(data), dtype=int)
        mean_val = col_data.mean()
        std_val = col_data.std()
        if std_val == 0:
            logging.warning(f"Standard deviation of '{column}' is zero; no outlier detection possible via Grubbs' test.")
            return np.zeros(len(data), dtype=int)
        abs_deviation = np.abs(col_data - mean_val)
        G = abs_deviation.max() / std_val
        # Compute the critical value using the t-distribution
        t_crit = t.ppf(1 - alpha / (2 * n), n - 2)
        G_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))
        outliers = np.zeros(len(data), dtype=int)
        if G > G_crit:
            # Mark the observation(s) with maximum deviation as outlier(s)
            max_mask = (np.abs(data[column] - mean_val) == abs_deviation.max())
            outliers[max_mask] = 1
        return outliers

    @staticmethod
    def mad_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using Median Absolute Deviation (MAD).
        Returns a binary array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        mad_mult = float(get_config_param('mad_mult', config))
        col_data = data[column]
        median_val = col_data.median()
        mad = np.median(np.abs(col_data - median_val))
        if mad == 0:
            logging.warning(f"MAD is zero for '{column}'; cannot compute modified z-scores.")
            return np.zeros(len(col_data), dtype=int)
        deviation = np.abs(col_data - median_val)
        outliers = (deviation > mad_mult * mad).astype(int)
        return outliers

    @staticmethod
    def iglewicz_hoaglin_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using the Iglewicz and Hoaglin modified z-score method.
        Returns a binary array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        threshold = float(get_config_param('iglewicz_hoaglin_thresh', config))
        col_data = data[column]
        median_val = col_data.median()
        mad = np.median(np.abs(col_data - median_val))
        if mad == 0:
            logging.warning(f"MAD is zero for '{column}' in Iglewicz-Hoaglin method; returning no outliers.")
            return np.zeros(len(col_data), dtype=int)
        modified_z = 0.6745 * (col_data - median_val) / mad
        outliers = (np.abs(modified_z) > threshold).astype(int)
        return outliers

    @staticmethod
    def isolation_forest_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using the Isolation Forest algorithm.
        Returns a binary array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        contamination = float(get_config_param('isolation_forest_contamination', config))
        X = data[[column]].values
        clf = IsolationForest(contamination=contamination, random_state=42)
        clf.fit(X)
        outliers = (clf.predict(X) == -1).astype(int)
        return outliers

    @staticmethod
    def dbscan_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using the DBSCAN clustering algorithm.
        Returns a binary array where 1 indicates an outlier (noise points).
        """
        OutlierDetection.validate_params(data, column)
        eps = float(get_config_param('dbscan_eps', config))
        min_samples = int(get_config_param('dbscan_min_samples', config))
        X = data[[column]].values
        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(X)
        outliers = (labels == -1).astype(int)
        return outliers

    @staticmethod
    def one_class_svm_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using One-Class SVM.
        Returns a binary array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        nu = float(get_config_param('one_class_svm_nu', config))
        X = data[[column]].values
        clf = OneClassSVM(nu=nu, kernel="rbf", gamma='auto')
        clf.fit(X)
        outliers = (clf.predict(X) == -1).astype(int)
        return outliers

    @staticmethod
    def elliptic_envelope_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using Elliptic Envelope.
        This method fits a robust Gaussian estimate to the entire numeric dataset.
        Returns a binary array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        contamination = float(get_config_param('elliptic_envelope_contamination', config))
        # Use all numeric columns for a multivariate approach
        X = data.select_dtypes(include=[np.number]).values
        if X.shape[0] < X.shape[1] + 1:
            logging.warning("Insufficient observations for Elliptic Envelope; returning no outliers.")
            return np.zeros(len(data), dtype=int)
        try:
            ee = EllipticEnvelope(contamination=contamination, random_state=42)
            ee.fit(X)
            preds = ee.predict(X)  # -1 for outliers, 1 for inliers
            outliers = (preds == -1).astype(int)
        except Exception as e:
            logging.error(f"Error in Elliptic Envelope detection: {e}")
            outliers = np.zeros(len(data), dtype=int)
        return outliers

    @staticmethod
    def lof_outliers(data: pd.DataFrame, column: str, config: dict = DEFAULT_CONFIG) -> np.ndarray:
        """
        Detect outliers using Local Outlier Factor (LOF).
        Returns a binary array where 1 indicates an outlier.
        """
        OutlierDetection.validate_params(data, column)
        n_neighbors = int(get_config_param('lof_n_neighbors', config))
        X = data[[column]].values
        try:
            lof = LocalOutlierFactor(n_neighbors=n_neighbors)
            preds = lof.fit_predict(X)  # -1 for outliers, 1 for inliers
            outliers = (preds == -1).astype(int)
        except Exception as e:
            logging.error(f"Error in LOF detection for column '{column}': {e}")
            outliers = np.zeros(len(data), dtype=int)
        return outliers

    @staticmethod
    def plot_outliers(data: pd.DataFrame, outliers_dict: dict, show_plots: bool = True) -> None:
        """
        Plot the results of the outlier detection.
        
        Parameters:
        - data: The original DataFrame.
        - outliers_dict: A dict mapping method names to dicts mapping column names to binary outlier arrays.
        - show_plots: If False, plotting is skipped.
        """
        if not show_plots:
            return

        method_names = list(outliers_dict.keys())
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()

        # Create bar plots showing the percentage of outliers per column for each method.
        for method in method_names:
            outlier_counts = []
            for col in numeric_columns:
                count = np.sum(outliers_dict[method].get(col, np.zeros(len(data))))
                outlier_counts.append(count)
            total = len(data)
            outlier_percents = [count / total * 100 for count in outlier_counts]
            non_outlier_percents = [100 - p for p in outlier_percents]

            fig, ax = plt.subplots(figsize=(10, 6))
            bars_out = ax.bar(numeric_columns, outlier_percents, color='#FFA07A', label='Outliers')
            ax.bar(numeric_columns, non_outlier_percents, bottom=outlier_percents, color='#1E90FF', label='Inliers')
            
            for bar, pct in zip(bars_out, outlier_percents):
                if pct > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, f'{pct:.1f}%',
                            ha='center', va='center', color='black', weight='bold')
            ax.set_xlabel('Variable')
            ax.set_ylabel('Percentage')
            ax.set_title(f'Outlier Percentage by {method.capitalize()} Method')
            ax.legend()
            plt.xticks(rotation=45)
            sns.despine()
            plt.tight_layout()
            plt.show()

        # Scatter plots for each method and each column
        for method, cols in outliers_dict.items():
            for col, outliers in cols.items():
                if outliers is None:
                    continue
                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=data.index, y=data[col],
                                hue=outliers,
                                palette={0: '#1E90FF', 1: '#FFA07A'},
                                legend='full')
                plt.title(f'{method.capitalize()} Outlier Detection in {col}')
                plt.xlabel('Index')
                plt.ylabel(col)
                plt.legend(title='Outlier', labels=['Inlier', 'Outlier'])
                sns.despine()
                plt.tight_layout()
                plt.show()


def detect_outliers(data: pd.DataFrame,
                    methods: list = None,
                    columns: list = None,
                    config: dict = DEFAULT_CONFIG,
                    show_plots: bool = True) -> pd.DataFrame:
    """
    Detect outliers on specified columns using one or more methods.
    
    Parameters:
      - data: A pandas DataFrame containing the data.
      - methods: List of method names to apply. (Defaults to all implemented methods.)
      - columns: List of columns to process. (Defaults to all numeric columns.)
      - config: Configuration dictionary with parameters.
      - show_plots: If True, displays plots of the detection results.
    
    Returns:
      A DataFrame containing binary flags for outliers for each column and method.
    """
    if methods is None:
        methods = ["z", "tukey", "mahalanobis", "grubbs", "mad",
                   "iglewicz_hoaglin", "isolation_forest", "dbscan",
                   "one_class_svm", "elliptic_envelope", "lof"]
    if columns is None:
        columns = data.select_dtypes(include=[np.number]).columns.tolist()
    if data.empty:
        raise ValueError("Input data is empty.")

    results = pd.DataFrame(index=data.index)
    # Dictionary to store outlier arrays by method and column (for plotting)
    outliers_dict = {method: {col: None for col in columns} for method in methods}

    # Map method names to the corresponding static method
    method_map = {
        "z": OutlierDetection.z_outliers,
        "tukey": OutlierDetection.tukey_outliers,
        "mahalanobis": OutlierDetection.mahalanobis_outliers,
        "grubbs": OutlierDetection.grubbs_outliers,
        "mad": OutlierDetection.mad_outliers,
        "iglewicz_hoaglin": OutlierDetection.iglewicz_hoaglin_outliers,
        "isolation_forest": OutlierDetection.isolation_forest_outliers,
        "dbscan": OutlierDetection.dbscan_outliers,
        "one_class_svm": OutlierDetection.one_class_svm_outliers,
        "elliptic_envelope": OutlierDetection.elliptic_envelope_outliers,
        "lof": OutlierDetection.lof_outliers
    }

    for col in columns:
        for method in methods:
            func = method_map.get(method)
            if func is None:
                logging.warning(f"Method '{method}' is not recognized. Skipping.")
                continue
            try:
                logging.info(f"Applying method '{method}' on column '{col}'.")
                outlier_flags = func(data, col, config=config)
                results[f"{col}_{method}_outlier"] = outlier_flags
                outliers_dict[method][col] = outlier_flags
            except Exception as e:
                logging.error(f"Error applying method '{method}' on column '{col}': {e}")
                results[f"{col}_{method}_outlier"] = np.zeros(len(data), dtype=int)
                outliers_dict[method][col] = np.zeros(len(data), dtype=int)

    # Optionally, display plots summarizing the results
    OutlierDetection.plot_outliers(data, outliers_dict, show_plots=show_plots)
    return results


if __name__ == "__main__":
    # --- Example Usage ---
    np.random.seed(123)
    example_data = pd.DataFrame({
        'A': np.random.normal(10, 2, 100),
        'B': np.random.normal(20, 5, 100),
        'C': np.random.normal(30, 10, 100)
    })
    # Add some artificial outliers
    example_data.loc[[25, 50, 75], 'B'] = [40, 10, 35]
    example_data.loc[[10, 90], 'C'] = [5, 50]

    methods = ["z", "tukey", "mahalanobis", "grubbs", "mad",
               "iglewicz_hoaglin", "isolation_forest", "dbscan",
               "one_class_svm", "elliptic_envelope", "lof"]

    outlier_results = detect_outliers(example_data, methods=methods, show_plots=True)
    print(outlier_results.head())
