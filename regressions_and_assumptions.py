from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Union, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from scipy.linalg import qr as scipy_qr

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.stattools import jarque_bera as sm_jarque_bera


CheckSpec = Union[str, Sequence[str]]
PlotSpec = Union[bool, Literal["none", "basic", "all"]]
CollinearSpec = Literal["warn", "drop", "raise"]


def re_split(pattern: str, text: str) -> list[str]:
    import re
    return re.split(pattern, text)


def _as_dataframe(X: Any, *, name_prefix: str = "x") -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy()
    if isinstance(X, pd.Series):
        return X.to_frame()
    arr = np.asarray(X)
    if arr.ndim == 1:
        return pd.DataFrame({f"{name_prefix}1": arr})
    if arr.ndim == 2:
        cols = [f"{name_prefix}{i+1}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)
    raise ValueError("X must be a 1D/2D array-like, Series, or DataFrame.")


def _as_series(y: Any, *, name: str = "y") -> pd.Series:
    if isinstance(y, pd.Series):
        return y.copy()
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        return y.iloc[:, 0].rename(y.columns[0])
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError("y must be 1D array-like or a pandas Series.")
    return pd.Series(arr, name=name)


def _align_dropna(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series, int]:
    X, y = X.align(y, join="inner", axis=0)
    before = len(y)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X2, y2 = X.loc[mask].copy(), y.loc[mask].copy()
    return X2, y2, before - len(y2)


def _maybe_get_dummies(X: pd.DataFrame, *, encode_cats: bool) -> pd.DataFrame:
    if not encode_cats:
        return X
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    if not cat_cols:
        return X
    return pd.get_dummies(X, columns=cat_cols, drop_first=True)


def _parse_set(
    spec: CheckSpec,
    *,
    all_items: Sequence[str],
    aliases: dict[str, str],
    default: str,
) -> set[str]:
    canonical = set(all_items)

    if spec is None:
        spec = default

    if isinstance(spec, str):
        s = spec.strip().lower()
        if s in ("all", "*"):
            return set(all_items)
        if s in ("none", "no", "off", "false"):
            return set()
        if s in ("basic", "default"):
            s = "basic"
        parts = [p for p in re_split(r"[,\s\|]+", s) if p]
    else:
        parts = [str(p).strip().lower() for p in spec if str(p).strip()]

    out: set[str] = set()
    for p in parts:
        p = aliases.get(p, p)

        matches = [k for k in canonical if k.startswith(p)]
        if len(matches) == 1:
            out.add(matches[0])
            continue

        if p in canonical:
            out.add(p)
            continue

        if p == "basic":
            basic_expanded = aliases.get("basic_expanded", "norm homo auto").split()
            out.update(basic_expanded)
            continue

        raise ValueError(
            f"Unknown option '{p}'. Valid: {sorted(all_items)} or 'basic'/'all'/'none'."
        )
    return out


def _reduce_collinearity(
    X: pd.DataFrame,
    *,
    mode: CollinearSpec,
    notes: list[str],
    tol: Optional[float] = None,
) -> pd.DataFrame:
    """
    Detect rank deficiency among predictors and optionally drop redundant columns.

    mode:
      - "warn": keep columns, just note if rank deficient
      - "drop": drop redundant columns (QR pivoting)
      - "raise": raise ValueError if rank deficient
    """
    if X.shape[1] <= 1:
        return X

    # Identify intercept-like column by name first
    intercept_candidates = [c for c in X.columns if c.lower() in ("const", "intercept")]
    intercept = intercept_candidates[0] if intercept_candidates else None

    # Drop additional zero-variance columns (perfectly collinear with intercept or useless)
    stds = X.std(axis=0, ddof=0)
    zero_var_cols = stds[stds == 0].index.tolist()

    if intercept is not None:
        extras = [c for c in zero_var_cols if c != intercept]
        if extras:
            if mode == "raise":
                raise ValueError(f"Zero-variance column(s) found (besides intercept): {extras}")
            if mode == "drop":
                X = X.drop(columns=extras)
                notes.append(f"Dropped zero-variance column(s): {extras}")
            else:
                notes.append(f"[warn] Zero-variance column(s) present: {extras}")
    else:
        # If no intercept column exists, keep the first zero-variance column (if any), drop the rest
        if len(zero_var_cols) > 1:
            keep0 = zero_var_cols[0]
            extras = zero_var_cols[1:]
            if mode == "raise":
                raise ValueError(f"Multiple zero-variance columns found: {zero_var_cols}")
            if mode == "drop":
                X = X.drop(columns=extras)
                notes.append(f"Dropped extra zero-variance column(s): {extras} (kept {keep0})")
            else:
                notes.append(f"[warn] Multiple zero-variance columns present: {zero_var_cols}")

    # Rank check on full matrix
    A_full = X.to_numpy()
    rank_full = int(np.linalg.matrix_rank(A_full))
    if rank_full == X.shape[1]:
        return X

    msg = f"Design matrix is rank deficient (rank={rank_full}, cols={X.shape[1]})."
    if mode == "warn":
        notes.append(f"[warn] {msg} Consider collinear='drop' or remove redundant predictors.")
        return X
    if mode == "raise":
        raise ValueError(msg + " Remove redundant predictors or use collinear='drop'.")

    # mode == "drop": keep intercept (if present), pivot the rest
    keep_cols: list[str] = []
    other_cols = list(X.columns)

    if intercept is not None and intercept in other_cols:
        keep_cols.append(intercept)
        other_cols.remove(intercept)

    if not other_cols:
        # only intercept left
        return X[keep_cols] if keep_cols else X

    A = X[other_cols].to_numpy()
    # QR with column pivoting
    Q, R, piv = scipy_qr(A, mode="economic", pivoting=True)

    diag = np.abs(np.diag(R)) if R.size else np.array([])
    if tol is None:
        maxdiag = float(diag.max()) if diag.size else 0.0
        tol = np.finfo(float).eps * max(A.shape) * maxdiag

    rank = int(np.sum(diag > tol))
    keep_other = [other_cols[i] for i in piv[:rank]]
    drop_other = [other_cols[i] for i in piv[rank:]]

    X_reduced = X[keep_cols + keep_other]
    notes.append(f"Dropped collinear column(s): {drop_other}")
    return X_reduced


@dataclass
class OLSRun:
    model: Any
    X: pd.DataFrame
    X_raw: pd.DataFrame
    y: pd.Series
    fitted: pd.Series
    residuals: pd.Series
    diagnostics: dict[str, Any]
    notes: list[str]

    @property
    def summary(self):
        return self.model.summary()


class RegressionDiagnostics:
    CHECKS_ALL = ("lin", "norm", "homo", "auto", "vif", "infl")
    PLOTS_ALL = ("fit", "resid", "qq", "scale", "influence", "corr")

    def __init__(
        self,
        *,
        checks: CheckSpec = "basic",
        plots: PlotSpec = "basic",
        lags: Union[int, Sequence[int]] = 10,
        alpha: float = 0.05,
        sample: Optional[int] = 5000,
        encode_cats: bool = False,
        robust: Optional[str] = None,   # e.g. "HC3"
        add_const: bool = True,
        collinear: CollinearSpec = "warn",
        verbose: bool = True,
    ):
        self.alpha = float(alpha)
        self.lags = lags
        self.sample = sample
        self.encode_cats = encode_cats
        self.robust = robust
        self.add_const = add_const
        self.collinear = collinear
        self.verbose = verbose

        check_aliases = {
            "linearity": "lin",
            "linear": "lin",
            "normal": "norm",
            "normality": "norm",
            "homoscedasticity": "homo",
            "hetero": "homo",
            "heteroskedasticity": "homo",
            "autocorr": "auto",
            "independence": "auto",
            "multicollinearity": "vif",
            "collinearity": "vif",
            "influence": "infl",
            "outliers": "infl",
            "basic_expanded": "norm homo auto",
        }
        self.checks = _parse_set(
            checks,
            all_items=self.CHECKS_ALL,
            aliases=check_aliases,
            default="basic",
        )

        if isinstance(plots, bool):
            plots = "basic" if plots else "none"
        self._plot_mode = str(plots).strip().lower()
        if self._plot_mode not in ("none", "basic", "all"):
            raise ValueError("plots must be True/False or one of: 'none', 'basic', 'all'.")

    def fit(
        self,
        X: Any = None,
        y: Any = None,
        *,
        formula: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
    ) -> OLSRun:
        notes: list[str] = []

        if formula is not None:
            if data is None:
                raise ValueError("If you pass formula=..., you must also pass data=DataFrame.")

            # Build design via formula (patsy), then (optionally) reduce collinearity and refit
            tmp = sm.OLS.from_formula(formula, data=data).fit()
            y_s = pd.Series(np.asarray(tmp.model.endog).ravel(), name=str(getattr(tmp.model, "endog_names", "y")))
            X_df = pd.DataFrame(tmp.model.exog, columns=list(getattr(tmp.model, "exog_names", [])))

            X_df = _reduce_collinearity(X_df, mode=self.collinear, notes=notes)

            model = sm.OLS(y_s, X_df).fit(cov_type=self.robust if self.robust else "nonrobust")
            X_raw = X_df.drop(columns=[c for c in X_df.columns if c.lower() in ("intercept", "const")], errors="ignore")

        else:
            if X is None or y is None:
                raise ValueError("Provide either (X, y) or (formula, data).")

            X_raw = _as_dataframe(X)
            y_s = _as_series(y)

            X_raw = _maybe_get_dummies(X_raw, encode_cats=self.encode_cats)
            X_raw, y_s, dropped = _align_dropna(X_raw, y_s)
            if dropped:
                notes.append(f"Dropped {dropped} row(s) with missing values after aligning X and y.")
            if X_raw.empty or y_s.empty:
                raise ValueError("After cleaning/alignment, X and/or y is empty.")

            non_numeric = X_raw.columns[~X_raw.apply(pd.api.types.is_numeric_dtype)].tolist()
            if non_numeric:
                raise TypeError(
                    f"Non-numeric columns found: {non_numeric}. "
                    f"Use encode_cats=True or use the formula interface with C(...)."
                )

            X_df = sm.add_constant(X_raw, has_constant="add") if self.add_const else X_raw.copy()
            X_df = _reduce_collinearity(X_df, mode=self.collinear, notes=notes)

            model = sm.OLS(y_s, X_df).fit(cov_type=self.robust if self.robust else "nonrobust")

        fitted = pd.Series(model.fittedvalues, index=y_s.index, name="fitted")
        resid = pd.Series(model.resid, index=y_s.index, name="residuals")

        diagnostics: dict[str, Any] = {}
        self._run_checks(X_raw=X_raw, X_design=X_df, y=y_s, fitted=fitted, resid=resid, model=model, out=diagnostics)
        self._run_plots(X_raw=X_raw, y=y_s, fitted=fitted, resid=resid, model=model)

        if self.verbose:
            for n in notes:
                print(f"[note] {n}")

        return OLSRun(
            model=model,
            X=X_df,
            X_raw=X_raw,
            y=y_s,
            fitted=fitted,
            residuals=resid,
            diagnostics=diagnostics,
            notes=notes,
        )

    def _run_checks(self, *, X_raw, X_design, y, fitted, resid, model, out: dict[str, Any]) -> None:
        if not self.checks:
            return

        idx = y.index
        if self.sample and len(idx) > self.sample:
            rs = np.random.RandomState(0)
            take = rs.choice(len(idx), size=self.sample, replace=False)
            idx_s = idx.take(take)
            y_s, fitted_s, resid_s = y.loc[idx_s], fitted.loc[idx_s], resid.loc[idx_s]
            X_raw_s = X_raw.loc[idx_s]
            out["sampled"] = {"used": True, "n": self.sample, "original_n": len(idx)}
        else:
            y_s, fitted_s, resid_s = y, fitted, resid
            X_raw_s = X_raw
            out["sampled"] = {"used": False, "n": len(idx)}

        if "lin" in self.checks:
            lin = {}
            for col in X_raw_s.columns:
                x = X_raw_s[col].to_numpy()
                r = resid_s.to_numpy()
                lin[col] = {"corr_x_resid": float(np.corrcoef(x, r)[0, 1])} if (np.std(x) > 0 and np.std(r) > 0) else {"corr_x_resid": np.nan}
            out["linearity"] = lin

        if "norm" in self.checks:
            norm = {}
            r = resid_s.to_numpy()
            n = len(r)

            if n <= 5000:
                _, p_shapiro = stats.shapiro(r)
                norm["shapiro_p"] = float(p_shapiro)

            try:
                jb_stat, jb_p, skew, kurt = sm_jarque_bera(r)
                norm.update({
                    "jarque_bera_stat": float(jb_stat),
                    "jarque_bera_p": float(jb_p),
                    "skew": float(skew),
                    "kurtosis": float(kurt),
                })
            except Exception:
                res = stats.jarque_bera(r)
                jb_stat = float(getattr(res, "statistic", res[0]))
                jb_p = float(getattr(res, "pvalue", res[1]))
                norm.update({
                    "jarque_bera_stat": jb_stat,
                    "jarque_bera_p": jb_p,
                    "skew": float(stats.skew(r, bias=False)),
                    "kurtosis": float(stats.kurtosis(r, fisher=False, bias=False)),
                })

            out["normality"] = norm

        if "homo" in self.checks:
            if isinstance(X_design, pd.DataFrame):
                X_bp = X_design.loc[y_s.index].to_numpy()
            else:
                X_bp = np.asarray(X_design)
            bp = het_breuschpagan(resid_s.to_numpy(), X_bp)
            out["homoscedasticity"] = {
                "breusch_pagan": {
                    "lm_stat": float(bp[0]),
                    "lm_p": float(bp[1]),
                    "f_stat": float(bp[2]),
                    "f_p": float(bp[3]),
                }
            }

        if "auto" in self.checks:
            auto = {"durbin_watson": float(durbin_watson(resid_s.to_numpy()))}
            lb = acorr_ljungbox(resid_s.to_numpy(), lags=self.lags, return_df=True)
            auto["ljung_box"] = lb.reset_index().rename(columns={"index": "lag"}).to_dict(orient="records")
            out["autocorrelation"] = auto

        if "vif" in self.checks:
            vif = {}
            if X_raw_s.shape[1] >= 2:
                vals = X_raw_s.to_numpy()
                rows = [{"feature": col, "vif": float(variance_inflation_factor(vals, i))}
                        for i, col in enumerate(X_raw_s.columns)]
                vif["vif"] = sorted(rows, key=lambda d: d["vif"], reverse=True)
                corr = X_raw_s.corr().to_numpy()
                vif["corr_max_abs"] = float(np.nanmax(np.abs(corr - np.eye(corr.shape[0]))))
            else:
                vif["vif"] = []
                vif["note"] = "VIF requires at least 2 predictors."
            out["multicollinearity"] = vif

        if "infl" in self.checks:
            influence = model.get_influence()
            summ = influence.summary_frame()
            out["influence"] = {
                "top_cooks_d": (
                    summ["cooks_d"]
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                    .rename(columns={"index": "row"})
                    .to_dict(orient="records")
                ),
                "top_student_resid": (
                    summ["student_resid"]
                    .abs()
                    .sort_values(ascending=False)
                    .head(10)
                    .reset_index()
                    .rename(columns={"index": "row"})
                    .to_dict(orient="records")
                ),
            }

    def _run_plots(self, *, X_raw, y, fitted, resid, model) -> None:
        mode = self._plot_mode
        if mode == "none":
            return

        plots = {"basic": ("fit", "resid", "qq"), "all": self.PLOTS_ALL}[mode]

        if self.sample and len(y) > self.sample:
            rs = np.random.RandomState(0)
            idx = y.index
            take = rs.choice(len(idx), size=self.sample, replace=False)
            idx = idx.take(take)
            X_raw_p, y_p, fitted_p, resid_p = X_raw.loc[idx], y.loc[idx], fitted.loc[idx], resid.loc[idx]
        else:
            X_raw_p, y_p, fitted_p, resid_p = X_raw, y, fitted, resid

        if "fit" in plots:
            self._plot_fit(X_raw_p, y_p, fitted_p)

        if "resid" in plots:
            self._plot_residuals(fitted_p, resid_p, X_raw_p)

        if "qq" in plots:
            self._plot_qq(resid_p)

        if mode == "all":
            if "scale" in plots:
                self._plot_scale_location(fitted_p, model)
            if "influence" in plots:
                self._plot_influence(model)
            if "corr" in plots and X_raw_p.shape[1] >= 2:
                self._plot_corr(X_raw_p)

    def _plot_fit(self, X_raw: pd.DataFrame, y: pd.Series, fitted: pd.Series) -> None:
        if X_raw.shape[1] == 1:
            x = X_raw.iloc[:, 0]
            order = np.argsort(x.to_numpy())
            plt.figure()
            plt.scatter(x, y, alpha=0.7)
            plt.plot(x.to_numpy()[order], fitted.to_numpy()[order])
            plt.title(f"Fit: y vs {x.name}")
            plt.xlabel(x.name)
            plt.ylabel(y.name or "y")
            plt.show()
        else:
            plt.figure()
            plt.scatter(fitted, y, alpha=0.7)
            lo = min(float(fitted.min()), float(y.min()))
            hi = max(float(fitted.max()), float(y.max()))
            plt.plot([lo, hi], [lo, hi])
            plt.title("Fit: observed vs fitted")
            plt.xlabel("Fitted")
            plt.ylabel("Observed")
            plt.show()

    def _plot_residuals(self, fitted: pd.Series, resid: pd.Series, X_raw: pd.DataFrame) -> None:
        plt.figure()
        plt.scatter(fitted, resid, alpha=0.7)
        plt.axhline(0, linestyle="--")
        smth = lowess(resid.to_numpy(), fitted.to_numpy(), frac=0.3, return_sorted=True)
        plt.plot(smth[:, 0], smth[:, 1])
        plt.title("Residuals vs Fitted")
        plt.xlabel("Fitted")
        plt.ylabel("Residuals")
        plt.show()

        cols = list(X_raw.columns[:6])
        if cols:
            n = len(cols)
            fig, axes = plt.subplots(n, 1, figsize=(7, 2.2 * n), sharey=True)
            if n == 1:
                axes = [axes]
            for ax, col in zip(axes, cols):
                x = X_raw[col].to_numpy()
                r = resid.to_numpy()
                ax.scatter(x, r, alpha=0.7)
                ax.axhline(0, linestyle="--")
                smth = lowess(r, x, frac=0.3, return_sorted=True)
                ax.plot(smth[:, 0], smth[:, 1])
                ax.set_xlabel(col)
                ax.set_ylabel("Residuals")
            fig.suptitle("Residuals vs Predictors (first up to 6)")
            fig.tight_layout()
            plt.show()

    def _plot_qq(self, resid: pd.Series) -> None:
        fig, ax = plt.subplots()
        sm.qqplot(resid.to_numpy(), line="45", fit=True, ax=ax)
        ax.set_title("Normal Q-Q Plot (residuals)")
        plt.show()

    def _plot_scale_location(self, fitted: pd.Series, model) -> None:
        infl = model.get_influence()
        std_resid = infl.resid_studentized_internal
        plt.figure()
        plt.scatter(fitted, np.sqrt(np.abs(std_resid)), alpha=0.7)
        smth = lowess(np.sqrt(np.abs(std_resid)), fitted.to_numpy(), frac=0.3, return_sorted=True)
        plt.plot(smth[:, 0], smth[:, 1])
        plt.title("Scale-Location")
        plt.xlabel("Fitted")
        plt.ylabel("Sqrt(|standardized residuals|)")
        plt.show()

    def _plot_influence(self, model) -> None:
        fig, ax = plt.subplots()
        sm.graphics.influence_plot(model, criterion="cooks", ax=ax)
        ax.set_title("Influence Plot (Cook's distance)")
        plt.show()

    def _plot_corr(self, X_raw: pd.DataFrame) -> None:
        corr = X_raw.corr()
        plt.figure(figsize=(7, 5))
        plt.imshow(corr.to_numpy(), aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
        plt.yticks(range(len(corr.index)), corr.index)
        plt.title("Correlation Matrix")
        plt.tight_layout()
        plt.show()


def ols(
    X: Any = None,
    y: Any = None,
    *,
    formula: Optional[str] = None,
    data: Optional[pd.DataFrame] = None,
    checks: CheckSpec = "basic",
    plots: PlotSpec = "basic",
    lags: Union[int, Sequence[int]] = 10,
    alpha: float = 0.05,
    sample: Optional[int] = 5000,
    encode_cats: bool = False,
    robust: Optional[str] = None,
    add_const: bool = True,
    collinear: CollinearSpec = "warn",
    verbose: bool = True,
):
    diag = RegressionDiagnostics(
        checks=checks,
        plots=plots,
        lags=lags,
        alpha=alpha,
        sample=sample,
        encode_cats=encode_cats,
        robust=robust,
        add_const=add_const,
        collinear=collinear,
        verbose=verbose,
    )
    return diag.fit(X=X, y=y, formula=formula, data=data)


def predict(model, X_new: Any, *, add_const: bool = True) -> pd.Series:
    Xn = _as_dataframe(X_new)
    if add_const:
        Xn = sm.add_constant(Xn, has_constant="add")
    pred = model.predict(Xn)
    return pd.Series(pred, index=Xn.index, name="prediction")

'''

if __name__ == "__main__":
    np.random.seed(0)
    n = 200

    X = pd.DataFrame({
        "X1": np.linspace(0, 100, n),
        "X2": np.linspace(0, 200, n) + np.random.normal(0, 5, n),  # break perfect collinearity
    })
    y = 3 * X["X1"] + 2 * X["X2"] + np.random.normal(0, 10, size=n)

    # Add 'vif' if you want multicollinearity metrics in diagnostics:
    # checks="basic vif" or checks="all"
    run = ols(X, y, checks="basic vif", plots="basic", robust="HC3", collinear="warn")
    print(run.summary)
    print(run.diagnostics.keys())
'''