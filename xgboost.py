import xgboost as xgb
print(xgb.__version__)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import shap
import joblib
import random
import logging
import time
import yaml
from pathlib import Path
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration
DEFAULT_CONFIG = {
    'seed': 42,
    'normalize_target': False,
    'hyperparameters': {
        'max_depth': [3, 4, 5, 6, 7, 8, 9],
        'min_child_weight': [1, 2, 3, 4, 5],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.5, 0.7, 1.0],
        'gamma': [0, 0.1, 0.2],
        'reg_alpha': [0, 0.1, 1.0, 10.0],
        'reg_lambda': [0.1, 1.0, 10.0],
        'n_estimators': [100, 200, 300]
    },
    'n_iter': 25
}

# Load configuration
def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.warning(f"Configuration file '{config_path}' not found. Using default configuration.")
        return DEFAULT_CONFIG
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return DEFAULT_CONFIG

CONFIG = load_config()

# Set seed for reproducibility
SEED = CONFIG['seed']
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
        
        # Save scaler
        joblib.dump(scaler, 'scaler.pkl')
        
        # Normalize target variable if needed
        if CONFIG['normalize_target']:
            y = (y - y.mean()) / y.std()
        
        return X_scaled, y, data.feature_names
    except Exception as e:
        logging.error(f"Error in data loading and preprocessing: {e}")
        raise

# Split dataset
def split_dataset(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=SEED + 1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED + 2)
    return X_train, X_val, X_test, y_train, y_val, y_test

# Perform Randomized Search for Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train):
    param_dist = CONFIG['hyperparameters']
    clf = xgb.XGBRegressor(random_state=SEED)
    
    # Use KFold for cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    
    rs_clf = RandomizedSearchCV(
        clf, 
        param_dist, 
        n_iter=CONFIG['n_iter'], 
        scoring='neg_mean_squared_error', 
        n_jobs=-1, 
        cv=cv, 
        random_state=SEED,
        verbose=1
    )
    
    rs_clf.fit(X_train, y_train)
    return rs_clf.best_params_

# Model training and evaluation
def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, best_parameters, feature_names):
    try:
        start_time = time.time()
        best_model = xgb.XGBRegressor(**best_parameters, random_state=SEED)
        best_model.fit(
            X_train, y_train, 
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        training_time = time.time() - start_time
        
        # Cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        
        # Evaluation
        val_predictions = best_model.predict(X_val)
        val_mse = mean_squared_error(y_val, val_predictions)
        test_predictions = best_model.predict(X_test)
        test_mse = mean_squared_error(y_test, test_predictions)
        
        # SHAP for interpretability
        try:
            explainer = shap.Explainer(best_model, X_train)
            shap_values = explainer.shap_values(X_test)
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            Path("plots").mkdir(exist_ok=True)
            plt.savefig('plots/shap_summary.png')
            plt.close()
        except Exception as e:
            logging.warning(f"Error generating SHAP plot: {e}")
        
        return best_model, val_mse, test_mse, cv_scores, training_time
    except Exception as e:
        logging.error(f"Error in model training and evaluation: {e}")
        raise

# Function to load model and scaler for inference
def load_model_for_inference():
    model = joblib.load('xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Main execution
def main():
    try:
        print("Starting the XGBoost model training process...")
        
        X, y, feature_names = load_and_preprocess_data()
        print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
        print(f"Data split. Training set size: {X_train.shape[0]}, Validation set size: {X_val.shape[0]}, Test set size: {X_test.shape[0]}")
        
        print("Starting hyperparameter tuning...")
        best_parameters = hyperparameter_tuning(X_train, y_train)
        print(f"Best parameters found: {best_parameters}")
        
        print("Training final model with best parameters...")
        best_model, val_mse, test_mse, cv_scores, training_time = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test, best_parameters, feature_names
        )
        
        print("\n--- Results ---")
        print(f"Validation MSE: {val_mse:.4f}")
        print(f"Test MSE: {test_mse:.4f}")
        print(f"Cross-validation scores (MSE): {[-score for score in cv_scores]}")
        print(f"Mean CV MSE: {-np.mean(cv_scores):.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Model Serialization for Deployment
        joblib.dump(best_model, 'xgb_model.pkl')
        print("Model saved successfully as 'xgb_model.pkl'")
        
        print("\nSHAP summary plot saved as 'plots/shap_summary.png'")
        
    except Exception as e:
        print(f"An error occurred during execution: {e}")
        logging.error(f"An error occurred during execution: {e}")

if __name__ == "__main__":
    main()