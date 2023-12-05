import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import shap
import joblib
import random

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Load and preprocess dataset
def load_and_preprocess_data():
    try:
        data = fetch_california_housing()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, data.feature_names
    except Exception as e:
        print(f"Error in data loading and preprocessing: {e}")
        raise

# Split dataset
def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Perform Randomized Search for Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_dist = {
        'max_depth': range(3, 10),
        'min_child_weight': range(1, 6),
        'eta': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1],
        'colsample_bytree': [0.5, 0.7, 1],
        'objective': ['reg:squarederror'],
        'eval_metric': ['rmse']
    }
    clf = xgb.XGBRegressor()
    rs_clf = RandomizedSearchCV(clf, param_dist, n_iter=25, scoring='neg_mean_squared_error', n_jobs=-1, cv=3, random_state=SEED)
    rs_clf.fit(X_train, y_train)
    return rs_clf.best_params_

# Model training and evaluation
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_parameters, feature_names):
    try:
        best_model = xgb.XGBRegressor(**best_parameters)
        best_model.fit(X_train, y_train)

        # Evaluation
        val_predictions = best_model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_predictions)
        test_predictions = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)

        # SHAP for interpretability
        explainer = shap.Explainer(best_model, X_train)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=feature_names)

        return best_model, val_mse, test_mse
    except Exception as e:
        print(f"Error in model training and evaluation: {e}")
        raise

# Main execution
if __name__ == "__main__":
    X, y, feature_names = load_and_preprocess_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    best_parameters = hyperparameter_tuning(X_train, y_train)
    best_model, val_mse, test_mse = train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_parameters, feature_names)

    print(f"Validation MSE: {val_mse}")
    print(f"Test MSE: {test_mse}")
    
    # Model Serialization for Deployment
    joblib.dump(best_model, 'xgb_model.pkl')
