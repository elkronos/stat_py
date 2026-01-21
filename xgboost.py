import argparse
import json
import logging
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer, StandardScaler

import matplotlib.pyplot as plt


# ----------------------------
# Logging
# ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("xgb_train")


# ----------------------------
# Default configuration
# ----------------------------
DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "normalize_target": False,

    # Feature scaling is enabled by default; a scaler artifact is always saved for inference symmetry.
    "scale_features": True,

    # Ratios for train/val/test. If values don't sum to 1, they are auto-normalized.
    "splits": [0.6, 0.2, 0.2],

    # Randomized search configuration
    "n_iter": 25,
    "cv_folds": 5,

    # Optional early stopping configuration (disabled by default)
    "early_stopping_rounds": None,   # e.g., 50 to enable
    "eval_metric": "rmse",

    # Artifacts
    "output_dir": ".",               # defaults to current directory
    "make_shap_plot": True,

    # Base model parameters (extendable via YAML or CLI conventions)
    "model_params": {
        "objective": "reg:squarederror",
        "n_jobs": -1,
    },

    # Search space
    "hyperparameters": {
        "max_depth": [3, 4, 5, 6, 7, 8, 9],
        "min_child_weight": [1, 2, 3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "subsample": [0.5, 0.7, 1.0],
        "colsample_bytree": [0.5, 0.7, 1.0],
        "gamma": [0, 0.1, 0.2],
        "reg_alpha": [0, 0.1, 1.0, 10.0],
        "reg_lambda": [0.1, 1.0, 10.0],
        "n_estimators": [100, 200, 300],
    },
}


# ----------------------------
# Config helpers
# ----------------------------
def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base without mutating base."""
    out = deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def normalize_config_keys(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience aliases for shorter YAML:
      - hp / hparams -> hyperparameters
      - out -> output_dir
      - shap -> make_shap_plot
      - norm_y / normalize_y -> normalize_target
    """
    if not cfg:
        return {}

    cfg = dict(cfg)

    if "hyperparameters" not in cfg:
        if "hp" in cfg:
            cfg["hyperparameters"] = cfg.pop("hp")
        elif "hparams" in cfg:
            cfg["hyperparameters"] = cfg.pop("hparams")

    if "output_dir" not in cfg and "out" in cfg:
        cfg["output_dir"] = cfg.pop("out")

    if "make_shap_plot" not in cfg and "shap" in cfg:
        cfg["make_shap_plot"] = cfg.pop("shap")

    if "normalize_target" not in cfg:
        if "norm_y" in cfg:
            cfg["normalize_target"] = cfg.pop("norm_y")
        elif "normalize_y" in cfg:
            cfg["normalize_target"] = cfg.pop("normalize_y")

    return cfg


def load_config(path: str) -> Dict[str, Any]:
    """Load YAML config and merge it onto defaults (partial overrides supported)."""
    p = Path(path)
    if not p.exists():
        logger.warning("Config file '%s' not found. Using defaults.", path)
        return deepcopy(DEFAULT_CONFIG)

    try:
        raw = yaml.safe_load(p.read_text()) or {}
        raw = normalize_config_keys(raw)
        return deep_merge(DEFAULT_CONFIG, raw)
    except Exception as e:
        logger.error("Error loading config '%s': %s. Using defaults.", path, e)
        return deepcopy(DEFAULT_CONFIG)


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Apply simple, intuitive CLI overrides onto the merged configuration."""
    cfg = deepcopy(cfg)

    if args.seed is not None:
        cfg["seed"] = args.seed

    if args.n_iter is not None:
        cfg["n_iter"] = args.n_iter

    if args.cv_folds is not None:
        cfg["cv_folds"] = args.cv_folds

    if args.norm_y is True:
        cfg["normalize_target"] = True
    if args.norm_y is False:
        cfg["normalize_target"] = False

    if args.scale is True:
        cfg["scale_features"] = True
    if args.scale is False:
        cfg["scale_features"] = False

    if args.splits is not None:
        cfg["splits"] = list(args.splits)

    if args.out is not None:
        cfg["output_dir"] = args.out

    if args.shap is True:
        cfg["make_shap_plot"] = True
    if args.shap is False:
        cfg["make_shap_plot"] = False

    if args.early_stop is not None:
        cfg["early_stopping_rounds"] = args.early_stop

    if args.eval_metric is not None:
        cfg["eval_metric"] = args.eval_metric

    # Simple speed presets (affects only n_iter; keeps the same search space)
    if args.preset == "quick":
        cfg["n_iter"] = min(int(cfg.get("n_iter", 25)), 10)
    elif args.preset == "thorough":
        cfg["n_iter"] = max(int(cfg.get("n_iter", 25)), 75)

    # Device convenience sets tree_method if not already provided in model_params
    if args.device is not None:
        mp = cfg.setdefault("model_params", {})
        if args.device == "gpu":
            mp.setdefault("tree_method", "gpu_hist")
        else:
            mp.setdefault("tree_method", "hist")

    return cfg


def validate_splits(splits: Sequence[float]) -> Tuple[float, float, float]:
    """
    Accepts either normalized ratios (0.6 0.2 0.2) or integer-like ratios (60 20 20).
    Returns normalized (train, val, test).
    """
    if len(splits) != 3:
        raise ValueError("splits must have exactly 3 values: [train, val, test]")

    train, val, test = map(float, splits)
    if any(x <= 0 for x in (train, val, test)):
        raise ValueError("splits must all be > 0")

    total = train + val + test
    if not np.isclose(total, 1.0):
        train, val, test = train / total, val / total, test / total

    return train, val, test


# ----------------------------
# Data + preprocessing
# ----------------------------
def load_and_preprocess_data(cfg: Dict[str, Any]) -> Tuple[np.ndarray, pd.Series, List[str]]:
    """Load the dataset, optionally normalize target, scale features, and save the scaler artifact."""
    data = fetch_california_housing()
    X_df = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    if cfg.get("normalize_target", False):
        std = y.std()
        y = (y - y.mean()) / (std if std != 0 else 1.0)

    if cfg.get("scale_features", True):
        scaler = StandardScaler()
    else:
        # Identity transformer (still persisted as scaler.pkl to keep inference workflow consistent)
        scaler = FunctionTransformer(lambda x: x, validate=False)

    X_scaled = scaler.fit_transform(X_df)

    out_dir = Path(cfg.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, out_dir / "scaler.pkl")

    return X_scaled, y, list(data.feature_names)


def split_dataset(cfg: Dict[str, Any], X: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series, pd.Series, pd.Series]:
    """Split into train/val/test using the configured ratios."""
    seed = int(cfg["seed"])
    train_ratio, val_ratio, test_ratio = validate_splits(cfg.get("splits", [0.6, 0.2, 0.2]))

    temp_ratio = val_ratio + test_ratio
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_ratio, random_state=seed + 1
    )

    test_within_temp = test_ratio / temp_ratio
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_within_temp, random_state=seed + 2
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------
# Modeling
# ----------------------------
def build_base_model(cfg: Dict[str, Any]) -> xgb.XGBRegressor:
    """Build a baseline XGBRegressor from config model_params with sensible defaults."""
    seed = int(cfg["seed"])
    model_params = dict(cfg.get("model_params", {}))
    model_params.setdefault("random_state", seed)
    model_params.setdefault("n_jobs", -1)
    model_params.setdefault("objective", "reg:squarederror")
    return xgb.XGBRegressor(**model_params)


def hyperparameter_tuning(cfg: Dict[str, Any], X_train: np.ndarray, y_train: pd.Series) -> Dict[str, Any]:
    """RandomizedSearchCV over the configured hyperparameter space."""
    base_model = build_base_model(cfg)
    param_dist = cfg["hyperparameters"]

    cv = KFold(
        n_splits=int(cfg.get("cv_folds", 5)),
        shuffle=True,
        random_state=int(cfg["seed"]),
    )

    rs = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=int(cfg["n_iter"]),
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        cv=cv,
        random_state=int(cfg["seed"]),
        verbose=1,
    )
    rs.fit(X_train, y_train)
    return rs.best_params_


def maybe_make_shap_plot(
    cfg: Dict[str, Any],
    model: xgb.XGBRegressor,
    X_test: np.ndarray,
    feature_names: List[str],
) -> Optional[Path]:
    """Generate and save a SHAP summary plot when enabled and available."""
    if not cfg.get("make_shap_plot", True):
        return None

    try:
        import shap  # lazy import
    except Exception as e:
        logger.warning("SHAP not available (%s). Skipping SHAP plot.", e)
        return None

    out_dir = Path(cfg.get("output_dir", "."))
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    X_test_df = pd.DataFrame(X_test, columns=feature_names)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)
        shap.summary_plot(shap_values, X_test_df, show=False)
        out_path = plots_dir / "shap_summary.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    except Exception as e:
        logger.warning("Error generating SHAP plot: %s", e)
        try:
            plt.close()
        except Exception:
            pass
        return None


def train_and_evaluate(
    cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    X_test: np.ndarray,
    y_test: pd.Series,
    best_params: Dict[str, Any],
    feature_names: List[str],
) -> Dict[str, Any]:
    """Train a final model with tuned params, evaluate on val/test, run CV, and optionally produce SHAP plot."""
    start = time.time()

    base = build_base_model(cfg)
    final_params = base.get_params()
    final_params.update(best_params)

    model = xgb.XGBRegressor(**final_params)

    fit_kwargs: Dict[str, Any] = {
        "eval_set": [(X_val, y_val)],
        "verbose": False,
    }

    esr = cfg.get("early_stopping_rounds", None)
    if esr is not None:
        fit_kwargs["early_stopping_rounds"] = int(esr)
        fit_kwargs["eval_metric"] = cfg.get("eval_metric", "rmse")

    model.fit(X_train, y_train, **fit_kwargs)
    training_time = time.time() - start

    cv_folds = int(cfg.get("cv_folds", 5))
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv_folds,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )

    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    val_mse = mean_squared_error(y_val, val_pred)
    test_mse = mean_squared_error(y_test, test_pred)

    shap_path = maybe_make_shap_plot(cfg, model, X_test, feature_names)

    return {
        "model": model,
        "val_mse": float(val_mse),
        "test_mse": float(test_mse),
        "cv_mse": [float(-s) for s in cv_scores],
        "cv_mse_mean": float(-np.mean(cv_scores)),
        "training_time_sec": float(training_time),
        "shap_plot": str(shap_path) if shap_path else None,
    }


# ----------------------------
# Inference loader
# ----------------------------
def load_model_for_inference(
    model_path: str = "xgb_model.pkl",
    scaler_path: str = "scaler.pkl",
):
    """Load persisted model and scaler artifacts for inference."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


# ----------------------------
# CLI + main
# ----------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Argument parsing that supports both terminal execution and notebook kernels.
    When running under an IPython kernel and argv is not explicitly provided,
    kernel-provided argv is ignored to keep user-facing flags clean.
    """
    p = argparse.ArgumentParser(description="Train an XGBoost regressor with tuning + SHAP.")

    # Common kernel argument injected by notebook environments
    p.add_argument("-f", "--f", dest="_ipykernel_f", default=None, help=argparse.SUPPRESS)

    p.add_argument("-c", "--config", default="config.yaml", help="Path to YAML config.")
    p.add_argument("-s", "--seed", type=int, default=None, help="Random seed override.")
    p.add_argument("--n-iter", type=int, default=None, help="RandomizedSearchCV n_iter override.")
    p.add_argument("--cv-folds", type=int, default=None, help="Number of CV folds override.")

    p.add_argument("--norm-y", dest="norm_y", action="store_true", default=None, help="Normalize target y.")
    p.add_argument("--no-norm-y", dest="norm_y", action="store_false", help="Do not normalize target y.")

    p.add_argument("--scale", dest="scale", action="store_true", default=None, help="Scale X features.")
    p.add_argument("--no-scale", dest="scale", action="store_false", help="Do not scale X features.")

    p.add_argument(
        "--splits",
        nargs=3,
        type=float,
        default=None,
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split ratios. Example: --splits 0.6 0.2 0.2 (auto-normalizes if needed).",
    )

    p.add_argument("--out", default=None, help="Output directory for artifacts (model/scaler/plots/metrics).")

    p.add_argument("--shap", dest="shap", action="store_true", default=None, help="Enable SHAP plot.")
    p.add_argument("--no-shap", dest="shap", action="store_false", help="Disable SHAP plot.")

    p.add_argument("--early-stop", type=int, default=None, help="Enable early stopping with given rounds.")
    p.add_argument("--eval-metric", type=str, default=None, help="Eval metric for early stopping (e.g., rmse, mae).")

    p.add_argument("--preset", choices=["quick", "standard", "thorough"], default="standard",
                   help="Simple speed preset (adjusts n_iter only).")

    p.add_argument("--device", choices=["cpu", "gpu"], default=None,
                   help="Convenience: sets tree_method to hist/gpu_hist if not already in model_params.")

    # Notebook-friendly behavior: ignore sys.argv kernel flags unless argv is explicitly provided.
    if argv is None and "ipykernel" in sys.modules:
        argv = []

    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
    print(xgb.__version__)  # preserves the original behavior

    args = parse_args(argv)
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    # Seed everything for reproducibility
    seed = int(cfg["seed"])
    random.seed(seed)
    np.random.seed(seed)

    out_dir = Path(cfg.get("output_dir", "."))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Persist effective config for repeatability
    try:
        (out_dir / "effective_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    except Exception as e:
        logger.warning("Could not write effective_config.yaml: %s", e)

    print("Starting the XGBoost model training process...")

    X, y, feature_names = load_and_preprocess_data(cfg)
    print(f"Data loaded. Shape of X: {X.shape}, Shape of y: {y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(cfg, X, y)
    print(
        "Data split. "
        f"Training set size: {X_train.shape[0]}, "
        f"Validation set size: {X_val.shape[0]}, "
        f"Test set size: {X_test.shape[0]}"
    )

    print("Starting hyperparameter tuning...")
    best_parameters = hyperparameter_tuning(cfg, X_train, y_train)
    print(f"Best parameters found: {best_parameters}")

    print("Training final model with best parameters...")
    results = train_and_evaluate(
        cfg,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        best_parameters,
        feature_names,
    )

    model: xgb.XGBRegressor = results["model"]

    print("\n--- Results ---")
    print(f"Validation MSE: {results['val_mse']:.4f}")
    print(f"Test MSE: {results['test_mse']:.4f}")
    print(f"Cross-validation scores (MSE): {results['cv_mse']}")
    print(f"Mean CV MSE: {results['cv_mse_mean']:.4f}")
    print(f"Training time: {results['training_time_sec']:.2f} seconds")

    # Persist artifacts
    model_path = out_dir / "xgb_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved successfully as '{model_path}'")

    # Extra metrics artifact (does not change existing printed output)
    metrics_path = out_dir / "metrics.json"
    try:
        metrics_payload = {k: v for k, v in results.items() if k != "model"}
        metrics_payload["best_parameters"] = best_parameters
        metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    except Exception as e:
        logger.warning("Could not write metrics.json: %s", e)

    if results.get("shap_plot"):
        print(f"SHAP summary plot saved as '{results['shap_plot']}'")


'''
if __name__ == "__main__":
    main()
'''