from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import chi2, t
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import MinCovDet, EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN


# -------------------------
# Logging (library-friendly)
# -------------------------
logger = logging.getLogger(__name__)


# -------------------------
# Defaults (short + nested)
# -------------------------
DEFAULTS: Dict[str, Dict[str, Any]] = {
    "z": {"thresh": 1.96},
    "tukey": {"k": 1.5},
    "mahalanobis": {"quantile": 0.95, "robust": True},
    "grubbs": {"alpha": 0.05, "max_outliers": 1},
    "mad": {"k": 3.0},
    "modified_z": {"thresh": 3.5},
    "iforest": {"contamination": 0.1, "n_estimators": 200},
    "dbscan": {"eps": 0.5, "min_samples": 5},
    "ocsvm": {"nu": 0.05, "kernel": "rbf", "gamma": "scale"},
    "elliptic": {"contamination": 0.1},
    "lof": {"n_neighbors": 20, "contamination": "auto"},
}

# Backward-compat mapping for your old flat keys (optional)
FLAT_KEY_MAP = {
    "z_thresh": ("z", "thresh"),
    "tukey_mult": ("tukey", "k"),
    "mahalanobis_thresh": ("mahalanobis", "quantile"),
    "grubbs_alpha": ("grubbs", "alpha"),
    "mad_mult": ("mad", "k"),
    "iglewicz_hoaglin_thresh": ("modified_z", "thresh"),
    "isolation_forest_contamination": ("iforest", "contamination"),
    "dbscan_eps": ("dbscan", "eps"),
    "dbscan_min_samples": ("dbscan", "min_samples"),
    "one_class_svm_nu": ("ocsvm", "nu"),
    "elliptic_envelope_contamination": ("elliptic", "contamination"),
    "lof_n_neighbors": ("lof", "n_neighbors"),
}


# -------------------------
# Small parsing helpers
# -------------------------
MethodsArg = Union[None, str, Sequence[str]]
ColumnsArg = Union[None, str, Sequence[str]]
MLFeaturesArg = Union[Literal["column", "all"], Sequence[str]]


def _tokenize(x: str) -> List[str]:
    # Accept commas, semicolons, pipes, and whitespace as separators
    if not isinstance(x, str):
        return []
    for ch in [",", ";", "|"]:
        x = x.replace(ch, " ")
    return [t for t in x.split() if t]


def _as_list(x: Union[str, Sequence[str]]) -> List[str]:
    if isinstance(x, str):
        return _tokenize(x)
    return list(x)


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _validate_df(df: pd.DataFrame) -> None:
    if df is None or not isinstance(df, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    if df.empty:
        raise ValueError("Input data is empty.")


def _validate_numeric_column(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found.")
    if not pd.api.types.is_numeric_dtype(df[col]):
        raise ValueError(f"Column '{col}' must be numeric.")


def _merge_config(user_cfg: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Supports:
      - None (use defaults)
      - nested: {"z": {"thresh": 2.5}, "iforest": {"contamination": 0.05}}
      - shorthand numeric: {"z": 2.5, "iforest": 0.05}
      - old flat keys: {"z_thresh": 2.5, "isolation_forest_contamination": 0.05}
    """
    cfg: Dict[str, Dict[str, Any]] = {k: v.copy() for k, v in DEFAULTS.items()}
    if not user_cfg:
        return cfg

    # First: apply old flat keys if present
    for flat_key, (m, p) in FLAT_KEY_MAP.items():
        if flat_key in user_cfg:
            cfg.setdefault(m, {}).update({p: user_cfg[flat_key]})

    # Second: apply nested or shorthand keys
    for k, v in user_cfg.items():
        if k in FLAT_KEY_MAP:
            continue
        if k in cfg and isinstance(v, dict):
            cfg[k].update(v)
        elif k in cfg and not isinstance(v, dict):
            # shorthand interpretation: numeric/str value sets the primary parameter
            # (kept intentionally simple and intuitive)
            if k == "z":
                cfg[k]["thresh"] = v
            elif k == "tukey":
                cfg[k]["k"] = v
            elif k == "mahalanobis":
                cfg[k]["quantile"] = v
            elif k == "grubbs":
                cfg[k]["alpha"] = v
            elif k == "mad":
                cfg[k]["k"] = v
            elif k == "modified_z":
                cfg[k]["thresh"] = v
            elif k in ("iforest", "elliptic"):
                cfg[k]["contamination"] = v
            elif k == "ocsvm":
                cfg[k]["nu"] = v
            elif k == "lof":
                cfg[k]["contamination"] = v
            elif k == "dbscan":
                # allow dbscan: 0.7 -> eps
                cfg[k]["eps"] = v
        # ignore unknown keys quietly (keeps UX forgiving)
    return cfg


# -------------------------
# Method registry + presets
# -------------------------
ALIASES: Dict[str, str] = {
    # canonical: itself
    "z": "z",
    "zscore": "z",
    "tukey": "tukey",
    "iqr": "tukey",
    "mahalanobis": "mahalanobis",
    "md": "mahalanobis",
    "grubbs": "grubbs",
    "mad": "mad",
    "modified_z": "modified_z",
    "iglewicz": "modified_z",
    "hoaglin": "modified_z",
    "iforest": "iforest",
    "isolation_forest": "iforest",
    "dbscan": "dbscan",
    "ocsvm": "ocsvm",
    "one_class_svm": "ocsvm",
    "elliptic": "elliptic",
    "elliptic_envelope": "elliptic",
    "lof": "lof",
}

PRESETS: Dict[str, List[str]] = {
    "basic": ["z", "tukey"],
    "robust": ["tukey", "mad", "modified_z"],
    "ml": ["iforest", "lof", "ocsvm"],
    "all": ["z", "tukey", "mahalanobis", "grubbs", "mad", "modified_z",
            "iforest", "dbscan", "ocsvm", "elliptic", "lof"],
}


def _parse_methods(methods: MethodsArg) -> List[str]:
    if methods is None:
        return PRESETS["all"].copy()
    if isinstance(methods, str):
        key = methods.strip().lower()
        if key in PRESETS:
            return PRESETS[key].copy()
        tokens = _tokenize(key)
        out: List[str] = []
        for tok in tokens:
            out.append(ALIASES.get(tok, tok))
        # keep order, drop unknowns later
        return out

    out = [ALIASES.get(m.strip().lower(), m.strip().lower()) for m in methods]
    return out


def _parse_columns(df: pd.DataFrame, columns: ColumnsArg) -> List[str]:
    if columns is None:
        return _numeric_columns(df)
    if isinstance(columns, str):
        cols = _tokenize(columns)
        return cols
    return list(columns)


# -------------------------
# Core detection primitives
# -------------------------
def _flags_like(df: pd.DataFrame) -> pd.Series:
    return pd.Series(0, index=df.index, dtype=int)


def _series_flags_like(s: pd.Series) -> pd.Series:
    return pd.Series(0, index=s.index, dtype=int)


def _dropna_rows(X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Index]:
    mask = ~X.isna().any(axis=1)
    return X.loc[mask], X.index[mask]


def _standardize(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


# -------------------------
# Univariate methods (Series -> flags)
# -------------------------
def detect_z(s: pd.Series, thresh: float) -> pd.Series:
    flags = _series_flags_like(s)
    x = s.astype(float)
    mu = np.nanmean(x.values)
    sd = np.nanstd(x.values)
    if not np.isfinite(sd) or sd == 0:
        return flags
    z = (x - mu) / sd
    flags.loc[z.index] = (np.abs(z.values) > float(thresh)).astype(int)
    flags.loc[s.isna()] = 0
    return flags


def detect_tukey(s: pd.Series, k: float) -> pd.Series:
    flags = _series_flags_like(s)
    x = s.astype(float)
    q1 = x.quantile(0.25)
    q3 = x.quantile(0.75)
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr == 0:
        return flags
    lo = q1 - float(k) * iqr
    hi = q3 + float(k) * iqr
    flags.loc[x.index] = ((x.values < lo) | (x.values > hi)).astype(int)
    flags.loc[s.isna()] = 0
    return flags


def detect_mad(s: pd.Series, k: float) -> pd.Series:
    flags = _series_flags_like(s)
    x = s.astype(float)
    med = np.nanmedian(x.values)
    mad = np.nanmedian(np.abs(x.values - med))
    if not np.isfinite(mad) or mad == 0:
        return flags
    flags.loc[x.index] = (np.abs(x.values - med) > float(k) * mad).astype(int)
    flags.loc[s.isna()] = 0
    return flags


def detect_modified_z(s: pd.Series, thresh: float) -> pd.Series:
    flags = _series_flags_like(s)
    x = s.astype(float)
    med = np.nanmedian(x.values)
    mad = np.nanmedian(np.abs(x.values - med))
    if not np.isfinite(mad) or mad == 0:
        return flags
    mz = 0.6745 * (x.values - med) / mad
    flags.loc[x.index] = (np.abs(mz) > float(thresh)).astype(int)
    flags.loc[s.isna()] = 0
    return flags


def detect_grubbs(s: pd.Series, alpha: float, max_outliers: int = 1) -> pd.Series:
    """
    Iterative Grubbs: repeatedly flags the most extreme point if significant,
    up to max_outliers. Keeps UX simple while adding flexibility.
    """
    flags = _series_flags_like(s)
    x_full = s.dropna().astype(float)
    if len(x_full) < 3:
        return flags

    x = x_full.copy()
    flagged_idx: List[Any] = []

    for _ in range(int(max_outliers)):
        n = len(x)
        if n < 3:
            break
        mu = x.mean()
        sd = x.std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            break

        deviations = (x - mu).abs()
        idx = deviations.idxmax()
        G = deviations.loc[idx] / sd

        t_crit = t.ppf(1 - float(alpha) / (2 * n), n - 2)
        G_crit = (n - 1) / np.sqrt(n) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

        if not (G > G_crit):
            break

        flagged_idx.append(idx)
        x = x.drop(index=idx)

    if flagged_idx:
        flags.loc[flagged_idx] = 1
    return flags


# -------------------------
# Multivariate (DataFrame -> flags)
# -------------------------
def detect_mahalanobis(
    df: pd.DataFrame,
    features: Sequence[str],
    quantile: float,
    robust: bool = True,
) -> pd.Series:
    flags = _flags_like(df)
    X = df.loc[:, list(features)].astype(float)

    Xc, idx = _dropna_rows(X)
    if Xc.shape[0] < Xc.shape[1] + 1:
        return flags

    try:
        if robust:
            cov = MinCovDet().fit(Xc.values)
            md2 = cov.mahalanobis(Xc.values)  # squared distances
        else:
            mu = Xc.values.mean(axis=0)
            covm = np.cov(Xc.values, rowvar=False)
            inv = np.linalg.pinv(covm)
            diff = Xc.values - mu
            md2 = np.einsum("ij,jk,ik->i", diff, inv, diff)
    except Exception:
        logger.exception("Mahalanobis computation failed; returning no outliers.")
        return flags

    dfree = Xc.shape[1]
    thresh = chi2.ppf(float(quantile), dfree)
    out = (md2 > thresh).astype(int)
    flags.loc[idx] = out
    return flags


def _fit_predict_ml(
    df: pd.DataFrame,
    features: Sequence[str],
    *,
    estimator_factory: Callable[[], Any],
    scale: bool = True,
) -> pd.Series:
    flags = _flags_like(df)
    X = df.loc[:, list(features)].astype(float)
    Xc, idx = _dropna_rows(X)
    if Xc.shape[0] < 3:
        return flags

    Xv = Xc.values
    if scale and Xv.shape[1] >= 1:
        Xv = _standardize(Xv)

    try:
        est = estimator_factory()
        preds = est.fit_predict(Xv)  # -1 outlier, 1 inlier (for most)
        flags.loc[idx] = (preds == -1).astype(int)
    except Exception:
        logger.exception("ML outlier detection failed; returning no outliers.")
    return flags


def detect_iforest(df: pd.DataFrame, features: Sequence[str], contamination: float, n_estimators: int, random_state: int) -> pd.Series:
    def factory():
        return IsolationForest(
            contamination=float(contamination),
            n_estimators=int(n_estimators),
            random_state=int(random_state),
        )
    return _fit_predict_ml(df, features, estimator_factory=factory, scale=True)


def detect_dbscan(df: pd.DataFrame, features: Sequence[str], eps: float, min_samples: int) -> pd.Series:
    flags = _flags_like(df)
    X = df.loc[:, list(features)].astype(float)
    Xc, idx = _dropna_rows(X)
    if Xc.shape[0] < max(int(min_samples), 3):
        return flags

    Xv = _standardize(Xc.values)

    try:
        model = DBSCAN(eps=float(eps), min_samples=int(min_samples))
        labels = model.fit_predict(Xv)
        flags.loc[idx] = (labels == -1).astype(int)
    except Exception:
        logger.exception("DBSCAN failed; returning no outliers.")
    return flags


def detect_ocsvm(df: pd.DataFrame, features: Sequence[str], nu: float, kernel: str, gamma: str) -> pd.Series:
    def factory():
        return OneClassSVM(nu=float(nu), kernel=str(kernel), gamma=str(gamma))
    return _fit_predict_ml(df, features, estimator_factory=factory, scale=True)


def detect_elliptic(df: pd.DataFrame, features: Sequence[str], contamination: float, random_state: int) -> pd.Series:
    def factory():
        return EllipticEnvelope(contamination=float(contamination), random_state=int(random_state))
    # EllipticEnvelope assumes roughly Gaussian; scaling usually helps
    return _fit_predict_ml(df, features, estimator_factory=factory, scale=True)


def detect_lof(df: pd.DataFrame, features: Sequence[str], n_neighbors: int, contamination: Union[str, float]) -> pd.Series:
    flags = _flags_like(df)
    X = df.loc[:, list(features)].astype(float)
    Xc, idx = _dropna_rows(X)
    if Xc.shape[0] < 5:
        return flags

    Xv = _standardize(Xc.values)

    nn = int(n_neighbors)
    nn = max(2, min(nn, Xc.shape[0] - 1))

    try:
        lof = LocalOutlierFactor(n_neighbors=nn, contamination=contamination)
        preds = lof.fit_predict(Xv)
        flags.loc[idx] = (preds == -1).astype(int)
    except Exception:
        logger.exception("LOF failed; returning no outliers.")
    return flags


# -------------------------
# Plotting (optional, light)
# -------------------------
def plot_outlier_summary(results: pd.DataFrame, title: str = "Outlier summary") -> None:
    if results.empty:
        return
    perc = results.mean(axis=0).sort_values(ascending=False) * 100
    plt.figure(figsize=(max(10, 0.4 * len(perc)), 5))
    plt.bar(perc.index.astype(str), perc.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Outliers (%)")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# -------------------------
# Public API
# -------------------------
def detect_outliers(
    data: pd.DataFrame,
    methods: MethodsArg = None,
    columns: ColumnsArg = None,
    config: Optional[Dict[str, Any]] = None,
    *,
    ml_features: MLFeaturesArg = "column",          # "column" (per-column) or "all" (run once on all numeric cols) or list[str]
    broadcast_multivariate: bool = False,           # if True, row-level methods also get copied per selected column
    plot: Union[bool, Literal["summary"]] = False,  # False or "summary"
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect outliers with an intuitive interface.

    Examples:
      detect_outliers(df)                               # all methods, all numeric columns
      detect_outliers(df, methods="robust", columns="A B")
      detect_outliers(df, methods="z tukey", config={"z": 2.8, "tukey": 2.0})
      detect_outliers(df, methods="ml", ml_features="all")   # row-level ML outliers across all numeric features
      detect_outliers(df, methods="mahalanobis", config={"mahalanobis": {"quantile": 0.99}})
    """
    _validate_df(data)
    cfg = _merge_config(config)

    cols = _parse_columns(data, columns)
    if not cols:
        raise ValueError("No columns selected.")
    for c in cols:
        _validate_numeric_column(data, c)

    meths = _parse_methods(methods)

    # Determine which methods are known
    known = set(PRESETS["all"])
    meths = [m for m in meths if m in known]
    if not meths:
        raise ValueError("No valid methods selected.")

    # Decide ML/multivariate feature set
    numeric_cols = _numeric_columns(data)
    if ml_features == "all":
        feat_all = numeric_cols
    elif ml_features == "column":
        feat_all = []  # per-column
    else:
        feat_all = list(ml_features)

    results = pd.DataFrame(index=data.index)

    def add_flags(name: str, flags: pd.Series) -> None:
        results[name] = flags.astype(int).reindex(data.index, fill_value=0)

    # Univariate per-column methods
    for col in cols:
        s = data[col]
        if "z" in meths:
            add_flags(f"{col}_z_outlier", detect_z(s, cfg["z"]["thresh"]))
        if "tukey" in meths:
            add_flags(f"{col}_tukey_outlier", detect_tukey(s, cfg["tukey"]["k"]))
        if "mad" in meths:
            add_flags(f"{col}_mad_outlier", detect_mad(s, cfg["mad"]["k"]))
        if "modified_z" in meths:
            add_flags(f"{col}_modified_z_outlier", detect_modified_z(s, cfg["modified_z"]["thresh"]))
        if "grubbs" in meths:
            add_flags(
                f"{col}_grubbs_outlier",
                detect_grubbs(s, cfg["grubbs"]["alpha"], cfg["grubbs"]["max_outliers"]),
            )

        # ML methods per-column (when ml_features="column")
        if ml_features == "column":
            feats = [col]
            if "iforest" in meths:
                add_flags(f"{col}_iforest_outlier", detect_iforest(data, feats, cfg["iforest"]["contamination"], cfg["iforest"]["n_estimators"], random_state))
            if "dbscan" in meths:
                add_flags(f"{col}_dbscan_outlier", detect_dbscan(data, feats, cfg["dbscan"]["eps"], cfg["dbscan"]["min_samples"]))
            if "ocsvm" in meths:
                add_flags(f"{col}_ocsvm_outlier", detect_ocsvm(data, feats, cfg["ocsvm"]["nu"], cfg["ocsvm"]["kernel"], cfg["ocsvm"]["gamma"]))
            if "lof" in meths:
                add_flags(f"{col}_lof_outlier", detect_lof(data, feats, cfg["lof"]["n_neighbors"], cfg["lof"]["contamination"]))

    # Multivariate/row-level methods (when ml_features != "column")
    # These run once on feat_all (or numeric columns), produce row flags.
    if ml_features != "column":
        feats = feat_all if feat_all else numeric_cols
        feats = [f for f in feats if f in numeric_cols]
        if len(feats) >= 1:
            if "mahalanobis" in meths:
                row_flags = detect_mahalanobis(data, feats, cfg["mahalanobis"]["quantile"], cfg["mahalanobis"]["robust"])
                add_flags("row_mahalanobis_outlier", row_flags)
                if broadcast_multivariate:
                    for col in cols:
                        add_flags(f"{col}_mahalanobis_outlier", row_flags)

            if "elliptic" in meths:
                row_flags = detect_elliptic(data, feats, cfg["elliptic"]["contamination"], random_state)
                add_flags("row_elliptic_outlier", row_flags)
                if broadcast_multivariate:
                    for col in cols:
                        add_flags(f"{col}_elliptic_outlier", row_flags)

            if "iforest" in meths:
                row_flags = detect_iforest(data, feats, cfg["iforest"]["contamination"], cfg["iforest"]["n_estimators"], random_state)
                add_flags("row_iforest_outlier", row_flags)
                if broadcast_multivariate:
                    for col in cols:
                        add_flags(f"{col}_iforest_outlier", row_flags)

            if "lof" in meths:
                row_flags = detect_lof(data, feats, cfg["lof"]["n_neighbors"], cfg["lof"]["contamination"])
                add_flags("row_lof_outlier", row_flags)
                if broadcast_multivariate:
                    for col in cols:
                        add_flags(f"{col}_lof_outlier", row_flags)

            if "ocsvm" in meths:
                row_flags = detect_ocsvm(data, feats, cfg["ocsvm"]["nu"], cfg["ocsvm"]["kernel"], cfg["ocsvm"]["gamma"])
                add_flags("row_ocsvm_outlier", row_flags)
                if broadcast_multivariate:
                    for col in cols:
                        add_flags(f"{col}_ocsvm_outlier", row_flags)

            if "dbscan" in meths:
                row_flags = detect_dbscan(data, feats, cfg["dbscan"]["eps"], cfg["dbscan"]["min_samples"])
                add_flags("row_dbscan_outlier", row_flags)
                if broadcast_multivariate:
                    for col in cols:
                        add_flags(f"{col}_dbscan_outlier", row_flags)

    if plot is True or plot == "summary":
        plot_outlier_summary(results, title="Outliers by output flag")

    return results


'''
# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    np.random.seed(123)
    df = pd.DataFrame({
        "A": np.random.normal(10, 2, 100),
        "B": np.random.normal(20, 5, 100),
        "C": np.random.normal(30, 10, 100),
    })

    df.loc[[25, 50, 75], "B"] = [40, 10, 35]
    df.loc[[10, 90], "C"] = [5, 50]

    # Intuitive calls:
    r1 = detect_outliers(df, methods="robust", columns="A B C", plot="summary")
    print(r1.head())

    r2 = detect_outliers(df, methods="ml", ml_features="all", plot="summary")
    print(r2.head())

    r3 = detect_outliers(df, methods="z tukey iforest", config={"z": 2.8, "iforest": 0.05}, columns="B C")
    print(r3.head())





# This script prints PASS/FAIL for each UAT case and exits with a non-zero code on failure.
import sys
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Import detect_outliers
# -----------------------------
MODULE_NAME = "outliers_refactor"  # change if your file/module name differs

try:
    # If running in a separate file, import from your module
    mod = __import__(MODULE_NAME, fromlist=["detect_outliers"])
    detect_outliers = getattr(mod, "detect_outliers")
except Exception:
    # If pasted below your detect_outliers implementation in the same file, it should already exist
    if "detect_outliers" not in globals():
        raise RuntimeError(
            "Could not import detect_outliers and it is not defined in this file. "
            "Either set MODULE_NAME to your module name or paste this script below your implementation."
        )

# -----------------------------
# Dataset builders
# -----------------------------
def make_d1(seed=123, n=100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "A": rng.normal(0, 1, n),
        "B": rng.normal(5, 2, n),
        "C": rng.normal(10, 5, n),
    })
    # Inject very strong outliers (deterministic flags for robust methods)
    df.loc[[25, 50, 75], "B"] = [200.0, -200.0, 150.0]
    df.loc[[10, 90], "C"] = [-250.0, 300.0]
    return df


def make_d2_constant(seed=123, n=100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "A": np.ones(n),                  # constant
        "B": rng.normal(0, 1, n),
    })
    return df


def make_d3_nans(seed=123, n=120) -> pd.DataFrame:
    df = make_d1(seed=seed, n=n)
    rng = np.random.default_rng(seed + 1)

    # Random NaNs in C
    nan_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[nan_idx, "C"] = np.nan

    # Entire rows NaN across all numeric features
    row_nan_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[row_nan_idx, ["A", "B", "C"]] = np.nan

    return df


def make_d4_small(n: int) -> pd.DataFrame:
    # Small sample sizes for edge-case checks
    df = pd.DataFrame({"A": np.arange(n, dtype=float), "B": np.arange(n, dtype=float) + 0.1})
    return df


def make_d5_non_numeric(seed=123, n=80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "A": rng.normal(0, 1, n),
        "B": rng.normal(0, 1, n),
        "Category": ["x" if i % 2 == 0 else "y" for i in range(n)],
    })
    return df


def make_d6_multivariate(seed=123, n=400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.95 * x + rng.normal(0, 0.2, n)  # strong correlation
    z = rng.normal(0, 1, n)

    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    # Off-line point: X and Y within ~2 stdev, but violates the relationship strongly
    outlier_idx = 10
    df.loc[outlier_idx, "X"] = 2.0
    df.loc[outlier_idx, "Y"] = -2.0
    df.loc[outlier_idx, "Z"] = 0.0

    return df, outlier_idx


# -----------------------------
# Helper assertions
# -----------------------------
def is_binary_df(df: pd.DataFrame) -> bool:
    for c in df.columns:
        vals = pd.Series(df[c]).dropna().unique()
        if not set(vals).issubset({0, 1}):
            return False
    return True


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"Expected {exc_type.__name__} to be raised, but no exception occurred.")


def count_figs() -> int:
    return len(plt.get_fignums())


# UAT runner for the refactored `detect_outliers(...)` implementation.

# How to use (pick ONE):
# 1) Same file: paste this at the bottom of your refactor file and run the file.
# 2) Separate file: save as uat_outliers.py, set MODULE_NAME to your refactor module (no .py), then run.

# Notes for notebooks:
# - If you run this in Jupyter/IPython, `sys.exit(...)` raises SystemExit. You can call `run_all_uat()` directly instead.

import sys
import re
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Import detect_outliers
# -----------------------------
MODULE_NAME = "outliers_refactor"  # change if your module name differs

try:
    mod = __import__(MODULE_NAME, fromlist=["detect_outliers"])
    detect_outliers = getattr(mod, "detect_outliers")
except Exception:
    if "detect_outliers" not in globals():
        raise RuntimeError(
            "Could not import detect_outliers and it is not defined in this file. "
            "Set MODULE_NAME correctly or paste this script below your implementation."
        )

# -----------------------------
# Dataset builders
# -----------------------------
def make_d1(seed=123, n=100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "A": rng.normal(0, 1, n),
        "B": rng.normal(5, 2, n),
        "C": rng.normal(10, 5, n),
    })
    df.loc[[25, 50, 75], "B"] = [200.0, -200.0, 150.0]
    df.loc[[10, 90], "C"] = [-250.0, 300.0]
    return df


def make_d2_constant(seed=123, n=100) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "A": np.ones(n),
        "B": rng.normal(0, 1, n),
    })


def make_d3_nans(seed=123, n=120) -> pd.DataFrame:
    df = make_d1(seed=seed, n=n)
    rng = np.random.default_rng(seed + 1)

    nan_idx = rng.choice(df.index, size=10, replace=False)
    df.loc[nan_idx, "C"] = np.nan

    row_nan_idx = rng.choice(df.index, size=5, replace=False)
    df.loc[row_nan_idx, ["A", "B", "C"]] = np.nan
    return df


def make_d4_small(n: int) -> pd.DataFrame:
    return pd.DataFrame({"A": np.arange(n, dtype=float), "B": np.arange(n, dtype=float) + 0.1})


def make_d5_non_numeric(seed=123, n=80) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "A": rng.normal(0, 1, n),
        "B": rng.normal(0, 1, n),
        "Category": ["x" if i % 2 == 0 else "y" for i in range(n)],
    })


def make_d6_multivariate(seed=123, n=400) -> tuple[pd.DataFrame, int]:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n)
    y = 0.95 * x + rng.normal(0, 0.2, n)
    z = rng.normal(0, 1, n)
    df = pd.DataFrame({"X": x, "Y": y, "Z": z})

    outlier_idx = 10
    df.loc[outlier_idx, "X"] = 2.0
    df.loc[outlier_idx, "Y"] = -2.0
    df.loc[outlier_idx, "Z"] = 0.0
    return df, outlier_idx


# -----------------------------
# Helper assertions
# -----------------------------
def is_binary_df(df: pd.DataFrame) -> bool:
    for c in df.columns:
        vals = pd.Series(df[c]).dropna().unique()
        if not set(vals).issubset({0, 1}):
            return False
    return True


def assert_true(cond: bool, msg: str):
    if not cond:
        raise AssertionError(msg)


def assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"Expected {exc_type.__name__} to be raised, but no exception occurred.")


# -----------------------------
# UAT runner
# -----------------------------
class UATRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.failures = []

    def run_case(self, case_id: str, fn):
        try:
            fn()
            print(f"[PASS] {case_id}")
            self.passed += 1
        except Exception as e:
            print(f"[FAIL] {case_id} -> {e}")
            tb = traceback.format_exc()
            self.failures.append((case_id, str(e), tb))
            self.failed += 1

    def summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 60)
        print(f"UAT Summary: {self.passed}/{total} passed, {self.failed} failed")
        if self.failed:
            print("\nFailures detail:")
            for cid, msg, tb in self.failures:
                print("-" * 60)
                print(f"{cid}: {msg}")
                print(tb)
        print("=" * 60)
        return 0 if self.failed == 0 else 1


# -----------------------------
# UAT cases
# -----------------------------
def uat_a01_default_run():
    df = make_d1()
    res = detect_outliers(df, plot=False)
    assert_true(isinstance(res, pd.DataFrame), "Result is not a DataFrame.")
    assert_true(res.index.equals(df.index), "Result index does not match input index.")
    assert_true(res.shape[0] == df.shape[0], "Row count mismatch.")
    assert_true(res.shape[1] > 0, "No output columns generated.")


def uat_a02_preset_robust_only_expected_methods():
    df = make_d1()
    res = detect_outliers(df, methods="robust", columns="A B C", plot=False)

    pat = re.compile(r"^(?P<col>.+?)_(?P<method>.+?)_outlier$")
    allowed = {"tukey", "mad", "modified_z"}

    bad = []
    for c in res.columns:
        m = pat.match(c)
        if not m:
            bad.append(c)
            continue
        method = m.group("method")
        if method not in allowed:
            bad.append(c)

    assert_true(len(bad) == 0, f"Unexpected columns for robust preset: {bad}")


def uat_a03_method_string_tokenization():
    df = make_d1()
    res = detect_outliers(df, methods="z, tukey  iforest|lof", columns="A", plot=False)
    expected_cols = {"A_z_outlier", "A_tukey_outlier", "A_iforest_outlier", "A_lof_outlier"}
    assert_true(expected_cols.issubset(set(res.columns)), f"Missing expected columns. Have: {sorted(res.columns)}")


def uat_a04_alias_mapping():
    df = make_d1()
    res = detect_outliers(df, methods="zscore iqr isolation_forest one_class_svm", columns="A", plot=False)
    expected_cols = {"A_z_outlier", "A_tukey_outlier", "A_iforest_outlier", "A_ocsvm_outlier"}
    assert_true(expected_cols.issubset(set(res.columns)), f"Alias mapping failed. Have: {sorted(res.columns)}")


def uat_a05_columns_selection_string():
    df = make_d1()
    res = detect_outliers(df, methods="tukey", columns="A,B C", plot=False)
    assert_true(set(res.columns) == {"A_tukey_outlier", "B_tukey_outlier", "C_tukey_outlier"},
                f"Columns parsing unexpected: {sorted(res.columns)}")


def uat_a06_default_numeric_only():
    df = make_d5_non_numeric()
    res = detect_outliers(df, methods="tukey", columns=None, plot=False)
    assert_true("Category_tukey_outlier" not in res.columns, "Non-numeric column was included unexpectedly.")


def uat_a07_invalid_column_raises():
    df = make_d1()
    assert_raises(ValueError, detect_outliers, df, methods="tukey", columns="DoesNotExist", plot=False)


def uat_a08_empty_df_raises():
    df = pd.DataFrame()
    assert_raises(ValueError, detect_outliers, df, plot=False)


def uat_b01_nested_config_changes_behavior():
    df = make_d1()
    r_default = detect_outliers(df, methods="z", columns="B", plot=False)
    r_loose = detect_outliers(df, methods="z", columns="B", config={"z": {"thresh": 10.0}}, plot=False)
    assert_true(r_loose["B_z_outlier"].sum() <= r_default["B_z_outlier"].sum(),
                "Higher z threshold should not increase outliers.")


def uat_b02_shorthand_config():
    df = make_d1()
    r1 = detect_outliers(df, methods="z", columns="B", config={"z": 1.0}, plot=False)
    r2 = detect_outliers(df, methods="z", columns="B", config={"z": 5.0}, plot=False)
    assert_true(r1["B_z_outlier"].sum() >= r2["B_z_outlier"].sum(),
                "Lower z thresh should flag >= outliers vs higher thresh.")


def uat_b03_legacy_flat_keys():
    df = make_d1()
    r_legacy = detect_outliers(
        df,
        methods="z iforest",
        columns="B",
        config={"z_thresh": 2.0, "isolation_forest_contamination": 0.02},
        plot=False,
    )
    assert_true("B_z_outlier" in r_legacy.columns and "B_iforest_outlier" in r_legacy.columns,
                "Legacy config mapping did not produce expected columns.")


def uat_c01_tukey_flags_injected_extremes():
    df = make_d1()
    res = detect_outliers(df, methods="tukey", columns="B C", plot=False)

    b_flags = res["B_tukey_outlier"]
    c_flags = res["C_tukey_outlier"]

    assert_true(int(b_flags.loc[25]) == 1 and int(b_flags.loc[50]) == 1 and int(b_flags.loc[75]) == 1,
                "Tukey should flag injected B outliers at [25, 50, 75].")
    assert_true(int(c_flags.loc[10]) == 1 and int(c_flags.loc[90]) == 1,
                "Tukey should flag injected C outliers at [10, 90].")


def uat_c02_mad_and_modified_z_flag_injected_extremes():
    df = make_d1()
    res = detect_outliers(df, methods="mad modified_z", columns="B C", plot=False)

    assert_true(int(res["B_mad_outlier"].loc[25]) == 1 and int(res["B_mad_outlier"].loc[50]) == 1 and int(res["B_mad_outlier"].loc[75]) == 1,
                "MAD should flag injected B outliers.")
    assert_true(int(res["C_mad_outlier"].loc[10]) == 1 and int(res["C_mad_outlier"].loc[90]) == 1,
                "MAD should flag injected C outliers.")

    assert_true(int(res["B_modified_z_outlier"].loc[25]) == 1 and int(res["B_modified_z_outlier"].loc[50]) == 1 and int(res["B_modified_z_outlier"].loc[75]) == 1,
                "Modified z should flag injected B outliers.")
    assert_true(int(res["C_modified_z_outlier"].loc[10]) == 1 and int(res["C_modified_z_outlier"].loc[90]) == 1,
                "Modified z should flag injected C outliers.")


def uat_c03_z_sensitivity():
    df = make_d1()
    r_low = detect_outliers(df, methods="z", columns="B", config={"z": 1.5}, plot=False)
    r_high = detect_outliers(df, methods="z", columns="B", config={"z": 5.0}, plot=False)
    assert_true(r_low["B_z_outlier"].sum() >= r_high["B_z_outlier"].sum(),
                "Lower z threshold should flag more or equal outliers.")


def uat_c04_constant_column_safe():
    df = make_d2_constant()
    res = detect_outliers(df, methods="z mad modified_z", columns="A", plot=False)
    assert_true(res.sum().sum() == 0, "Constant column should not produce any outliers for z/mad/modified_z.")


def uat_c05_grubbs_min_n():
    df = make_d4_small(2)
    res = detect_outliers(df, methods="grubbs", columns="A", plot=False)
    assert_true(res["A_grubbs_outlier"].sum() == 0, "Grubbs with n=2 should yield no outliers (all zeros).")


def uat_d01_ml_column_outputs_per_column_only():
    df = make_d1()
    res = detect_outliers(df, methods="ml", ml_features="column", columns="B C", plot=False)
    assert_true(any(c.startswith("B_") for c in res.columns), "Expected per-column ML outputs for B.")
    assert_true(all(not c.startswith("row_") for c in res.columns), "Did not expect row_* outputs when ml_features='column'.")


def uat_d02_ml_all_outputs_row_only():
    df = make_d1()
    res = detect_outliers(df, methods="ml", ml_features="all", plot=False)
    assert_true(set(res.columns) == {"row_iforest_outlier", "row_lof_outlier", "row_ocsvm_outlier"},
                f"Expected only row_* ML outputs. Got: {sorted(res.columns)}")


def uat_d03_broadcast_multivariate_copies_to_columns():
    df = make_d1()
    res = detect_outliers(df, methods="ml", ml_features="all", broadcast_multivariate=True, columns="A B", plot=False)

    expected_row = {"row_iforest_outlier", "row_lof_outlier", "row_ocsvm_outlier"}
    assert_true(expected_row.issubset(set(res.columns)), "Missing row_* columns with broadcast enabled.")
    assert_true("A_iforest_outlier" in res.columns and "B_iforest_outlier" in res.columns, "Missing broadcast columns for iforest.")
    assert_true((res["A_iforest_outlier"].values == res["row_iforest_outlier"].values).all(), "Broadcast A_iforest does not match row_iforest.")
    assert_true((res["B_iforest_outlier"].values == res["row_iforest_outlier"].values).all(), "Broadcast B_iforest does not match row_iforest.")


def uat_d04_mahalanobis_flags_multivariate_outlier():
    df, outlier_idx = make_d6_multivariate()
    res = detect_outliers(
        df,
        methods="mahalanobis",
        ml_features="all",
        config={"mahalanobis": {"quantile": 0.99, "robust": True}},
        plot=False,
    )
    assert_true("row_mahalanobis_outlier" in res.columns, "Expected row_mahalanobis_outlier output.")
    assert_true(int(res.loc[outlier_idx, "row_mahalanobis_outlier"]) == 1,
                "Expected Mahalanobis to flag the constructed multivariate outlier row.")


def uat_d05_nans_multivariate_safe_and_nan_rows_zero():
    df = make_d3_nans()
    res = detect_outliers(df, methods="ml mahalanobis elliptic", ml_features="all", plot=False)
    assert_true(is_binary_df(res), "Outputs must be binary 0/1 even with NaNs.")

    nan_rows = df.index[df[["A", "B", "C"]].isna().all(axis=1)]
    for c in [col for col in res.columns if col.startswith("row_")]:
        if len(nan_rows) > 0:
            assert_true((res.loc[nan_rows, c] == 0).all(), f"Rows with all-NaN features should be 0 for {c}.")


def uat_d06_iforest_determinism_same_seed():
    df = make_d1()
    r1 = detect_outliers(df, methods="iforest", columns="B", config={"iforest": {"contamination": 0.05}}, random_state=7, plot=False)
    r2 = detect_outliers(df, methods="iforest", columns="B", config={"iforest": {"contamination": 0.05}}, random_state=7, plot=False)
    assert_true((r1["B_iforest_outlier"].values == r2["B_iforest_outlier"].values).all(),
                "IsolationForest outputs should match with same random_state.")


def uat_e01_binary_output_and_index_alignment():
    df = make_d1()

    # Explicit method keys instead of preset names inside a list
    res = detect_outliers(
        df,
        methods=["tukey", "mad", "modified_z", "iforest", "lof", "ocsvm"],
        ml_features="all",
        plot=False,
    )

    assert_true(res.index.equals(df.index), "Index alignment failed.")
    assert_true(is_binary_df(res), "Non-binary outputs detected.")


def uat_f01_plot_false_no_figures():
    plt.close("all")

    calls = {"n": 0}
    orig = plt.figure

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    plt.figure = wrapped
    try:
        df = make_d1()
        _ = detect_outliers(df, methods="tukey", columns="A", plot=False)
    finally:
        plt.figure = orig
        plt.close("all")

    assert_true(calls["n"] == 0, "plot=False should not create figures.")


def uat_f02_plot_summary_one_figure_per_call():
    plt.close("all")
    df = make_d1()

    def count_figure_calls(fn):
        calls = {"n": 0}
        orig = plt.figure

        def wrapped(*args, **kwargs):
            calls["n"] += 1
            return orig(*args, **kwargs)

        plt.figure = wrapped
        try:
            fn()
        finally:
            plt.figure = orig
        return calls["n"]

    n1 = count_figure_calls(lambda: detect_outliers(df, methods="tukey", columns="A B", plot="summary"))
    n2 = count_figure_calls(lambda: detect_outliers(df, methods="mad", columns="A B", plot="summary"))

    assert_true(n1 == 1, f"Expected exactly 1 plt.figure() call for first plot, got {n1}.")
    assert_true(n2 == 1, f"Expected exactly 1 plt.figure() call for second plot, got {n2}.")

    plt.close("all")


# -----------------------------
# Run all tests
# -----------------------------
def run_all_uat():
    runner = UATRunner()

    runner.run_case("UAT-A01 Default run", uat_a01_default_run)
    runner.run_case("UAT-A02 Preset robust only expected methods", uat_a02_preset_robust_only_expected_methods)
    runner.run_case("UAT-A03 Method string tokenization", uat_a03_method_string_tokenization)
    runner.run_case("UAT-A04 Alias mapping", uat_a04_alias_mapping)
    runner.run_case("UAT-A05 Columns selection string", uat_a05_columns_selection_string)
    runner.run_case("UAT-A06 Default numeric-only", uat_a06_default_numeric_only)
    runner.run_case("UAT-A07 Invalid column raises", uat_a07_invalid_column_raises)
    runner.run_case("UAT-A08 Empty df raises", uat_a08_empty_df_raises)

    runner.run_case("UAT-B01 Nested config changes behavior", uat_b01_nested_config_changes_behavior)
    runner.run_case("UAT-B02 Shorthand config", uat_b02_shorthand_config)
    runner.run_case("UAT-B03 Legacy flat keys", uat_b03_legacy_flat_keys)

    runner.run_case("UAT-C01 Tukey flags injected extremes", uat_c01_tukey_flags_injected_extremes)
    runner.run_case("UAT-C02 MAD and modified z flag injected extremes", uat_c02_mad_and_modified_z_flag_injected_extremes)
    runner.run_case("UAT-C03 Z sensitivity", uat_c03_z_sensitivity)
    runner.run_case("UAT-C04 Constant column safe", uat_c04_constant_column_safe)
    runner.run_case("UAT-C05 Grubbs min n", uat_c05_grubbs_min_n)

    runner.run_case("UAT-D01 ML column outputs per-column only", uat_d01_ml_column_outputs_per_column_only)
    runner.run_case("UAT-D02 ML all outputs row-only", uat_d02_ml_all_outputs_row_only)
    runner.run_case("UAT-D03 Broadcast multivariate copies to columns", uat_d03_broadcast_multivariate_copies_to_columns)
    runner.run_case("UAT-D04 Mahalanobis flags multivariate outlier", uat_d04_mahalanobis_flags_multivariate_outlier)
    runner.run_case("UAT-D05 NaNs multivariate safe and NaN rows zero", uat_d05_nans_multivariate_safe_and_nan_rows_zero)
    runner.run_case("UAT-D06 IsolationForest determinism", uat_d06_iforest_determinism_same_seed)

    runner.run_case("UAT-E01 Binary output and index alignment", uat_e01_binary_output_and_index_alignment)

    runner.run_case("UAT-F01 plot=False no figures", uat_f01_plot_false_no_figures)
    runner.run_case("UAT-F02 plot='summary' one figure per call", uat_f02_plot_summary_one_figure_per_call)

    return runner.summary()


if __name__ == "__main__":
    run_all_uat()

'''