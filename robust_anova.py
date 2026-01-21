from __future__ import annotations

import logging
from itertools import combinations
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import mixedlm, ols
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.oneway import anova_oneway

import matplotlib.pyplot as plt
from statsmodels.graphics.factorplots import interaction_plot


ArrayLike = Union[Sequence[float], np.ndarray, pd.Series]
GroupsLike = Union[Sequence[ArrayLike], Mapping[str, ArrayLike]]
WithinLike = Union[str, Sequence[str]]
BetweenLike = Union[str, Sequence[str]]

_LOG = logging.getLogger(__name__)


# ---------------------------
# Small helpers
# ---------------------------

def _norm_key(s: str) -> str:
    """Normalize user-entered strings into a compact key (lowercase alphanumerics)."""
    return "".join(ch for ch in s.strip().lower() if ch.isalnum())


_TEST_ALIASES = {
    # one-way parametric
    "oneway": "oneway",
    "1w": "oneway",
    "glm": "oneway",
    "anova": "oneway",
    "welch": "oneway",
    "bf": "oneway",
    "trimmed": "oneway",
    # one-way nonparametric
    "mw": "mwu",
    "mwu": "mwu",
    "mannwhitney": "mwu",
    "mannwhitneyu": "mwu",
    "kruskal": "kruskal",
    "kw": "kruskal",
    # factorial
    "twoway": "factorial",
    "2w": "factorial",
    "factorial": "factorial",
    # repeated measures
    "rm": "rm",
    "repeated": "rm",
    "repeatedmeasures": "rm",
    "repeatedmeasuresanova": "rm",
    # automatic dispatch
    "auto": "auto",
}


def _as_list(x: Optional[Union[str, Sequence[str]]]) -> Optional[list[str]]:
    """Convert a single string or list-like into a list of strings."""
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _to_1d_float_array(x: ArrayLike, *, dropna: bool = True) -> np.ndarray:
    """Convert input to a flat float array, optionally dropping NaN values."""
    arr = np.asarray(x, dtype=float).reshape(-1)
    if dropna:
        arr = arr[~np.isnan(arr)]
    return arr


def _coerce_groups(groups: GroupsLike, *, dropna: bool = True) -> Dict[str, np.ndarray]:
    """Normalize groups into a dict[str, np.ndarray]."""
    if isinstance(groups, Mapping):
        out = {str(k): _to_1d_float_array(v, dropna=dropna) for k, v in groups.items()}
    else:
        out = {f"g{i+1}": _to_1d_float_array(g, dropna=dropna) for i, g in enumerate(groups)}

    if len(out) < 2:
        raise ValueError("Need at least 2 groups.")

    for name, arr in out.items():
        if arr.size == 0:
            raise ValueError(f"Group '{name}' has no usable (non-NaN) observations.")

    return out


def _summarize_groups(groups: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compact summary table for group sizes and basic descriptive stats."""
    rows = []
    for name, x in groups.items():
        rows.append({
            "group": name,
            "n": int(x.size),
            "mean": float(np.mean(x)) if x.size else np.nan,
            "std": float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
            "median": float(np.median(x)) if x.size else np.nan,
        })
    return pd.DataFrame(rows)


def _assumptions_oneway(
    groups: Dict[str, np.ndarray],
    *,
    alpha: float,
    normality: str = "shapiro",
    shapiro_max_n: int = 5000,
    levene_center: str = "median",
) -> Dict[str, Any]:
    """One-way assumption checks (normality per group, homogeneity across groups)."""
    norm_rows = []
    for name, x in groups.items():
        if x.size < 3:
            norm_rows.append({"group": name, "test": normality, "p": np.nan, "ok": None, "note": "n<3"})
            continue

        xs = x[:shapiro_max_n] if x.size > shapiro_max_n else x

        if normality == "shapiro":
            _, p = stats.shapiro(xs)
        else:
            raise ValueError("normality must be 'shapiro'.")

        norm_rows.append({"group": name, "test": normality, "p": float(p), "ok": bool(p >= alpha), "note": None})

    normality_df = pd.DataFrame(norm_rows)

    arrays = list(groups.values())
    if len(arrays) >= 2:
        _, p = stats.levene(*arrays, center=levene_center)
        hom = {"test": "levene", "center": levene_center, "p": float(p), "ok": bool(p >= alpha)}
    else:
        hom = {"test": "levene", "center": levene_center, "p": np.nan, "ok": None}

    return {"normality": normality_df, "homogeneity": hom}


def _maybe_log_assumptions(assumptions: Dict[str, Any], *, verbose: bool) -> None:
    """Optional warnings if assumption checks indicate possible issues."""
    if not verbose:
        return

    normality_df: pd.DataFrame = assumptions["normality"]
    hom: dict = assumptions["homogeneity"]

    bad_norm = normality_df[normality_df["ok"] == False]  # noqa: E712
    if len(bad_norm):
        _LOG.warning("Normality may be violated in: %s", ", ".join(bad_norm["group"].tolist()))

    if hom.get("ok") is False:
        _LOG.warning("Homogeneity may be violated (Levene p=%.4g).", hom["p"])


def _adjust_pvalues(pvals: Sequence[float], p_adjust: str) -> np.ndarray:
    """Multiple-comparison adjustment for a list of p-values (handles NaN p-values safely)."""
    method = _norm_key(p_adjust or "holm")
    p = np.asarray(pvals, dtype=float)

    if method in ("none", "no", "na"):
        return p

    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if mask.any():
        out[mask] = multipletests(p[mask], method=method)[1]
    return out


# ---------------------------
# Posthoc
# ---------------------------

def tukey_hsd(groups: GroupsLike, *, alpha: float = 0.05, dropna: bool = True) -> pd.DataFrame:
    """Tukey HSD posthoc across all pairs of groups."""
    g = _coerce_groups(groups, dropna=dropna)
    y = np.concatenate(list(g.values()))
    labels = np.concatenate([[name] * len(arr) for name, arr in g.items()])
    res = pairwise_tukeyhsd(endog=y, groups=labels, alpha=alpha)
    tbl = res.summary()
    return pd.DataFrame(tbl.data[1:], columns=tbl.data[0])


def pairwise_mwu(
    groups: GroupsLike,
    *,
    alpha: float = 0.05,
    p_adjust: str = "holm",
    alternative: str = "two-sided",
    dropna: bool = True,
) -> pd.DataFrame:
    """Pairwise Mann–Whitney U across all group pairs (with p-value adjustment)."""
    g = _coerce_groups(groups, dropna=dropna)
    names = list(g.keys())

    rows: list[dict[str, Any]] = []
    raw_p: list[float] = []

    for a, b in combinations(names, 2):
        xa, xb = g[a], g[b]
        stat, p = stats.mannwhitneyu(xa, xb, alternative=alternative)
        rows.append({"a": a, "b": b, "u": float(stat), "p": float(p)})
        raw_p.append(float(p))

    p_adj = _adjust_pvalues(raw_p, p_adjust=p_adjust)
    for r, pa in zip(rows, p_adj):
        r["p_adj"] = float(pa) if np.isfinite(pa) else np.nan
        r["sig"] = bool(pa < alpha) if np.isfinite(pa) else False

    return pd.DataFrame(rows)


# ---------------------------
# Core analyses
# ---------------------------

def one_way_test(
    groups: GroupsLike,
    *,
    alpha: float = 0.05,
    method: str = "auto",
    normality: str = "shapiro",
    shapiro_max_n: int = 5000,
    levene_center: str = "median",
    posthoc: Optional[str] = None,
    posthoc_on: str = "significant",
    p_adjust: str = "holm",
    dropna: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """One-way ANOVA family (standard/Welch/Brown–Forsythe/trimmed) with optional posthoc."""
    g = _coerce_groups(groups, dropna=dropna)
    group_summary = _summarize_groups(g)

    assumptions = _assumptions_oneway(
        g,
        alpha=alpha,
        normality=normality,
        shapiro_max_n=shapiro_max_n,
        levene_center=levene_center,
    )
    _maybe_log_assumptions(assumptions, verbose=verbose)

    m = _norm_key(method)
    if m == "auto":
        m = "welch" if assumptions["homogeneity"]["ok"] is False else "equal"

    if m in ("equal", "standard"):
        res = anova_oneway(list(g.values()), use_var="equal")
        main = {
            "test": "one-way ANOVA",
            "variant": "standard (equal variances)",
            "stat": float(res.statistic),
            "p": float(res.pvalue),
            "df_num": float(res.df_num),
            "df_denom": float(res.df_denom),
        }
    elif m in ("welch", "unequal"):
        res = anova_oneway(list(g.values()), use_var="unequal", welch_correction=True)
        main = {
            "test": "one-way ANOVA",
            "variant": "Welch (unequal variances)",
            "stat": float(res.statistic),
            "p": float(res.pvalue),
            "df_num": float(res.df_num),
            "df_denom": float(res.df_denom),
        }
    elif m in ("bf", "brownforsythe", "brown-forsythe"):
        res = anova_oneway(list(g.values()), use_var="bf")
        main = {
            "test": "one-way ANOVA",
            "variant": "Brown–Forsythe",
            "stat": float(res.statistic),
            "p": float(res.pvalue),
            "df_num": float(res.df_num),
            "df_denom": float(res.df_denom),
        }
    elif m in ("trimmed", "yuen"):
        res = anova_oneway(list(g.values()), use_var="unequal", trim_frac=0.2)
        main = {
            "test": "one-way ANOVA",
            "variant": "trimmed mean (20%)",
            "stat": float(res.statistic),
            "p": float(res.pvalue),
            "df_num": float(res.df_num),
            "df_denom": float(res.df_denom),
        }
    else:
        raise ValueError("method must be auto|equal|welch|bf|trimmed")

    posthoc_key = None if posthoc is None else _norm_key(posthoc)
    gate = _norm_key(posthoc_on or "significant")
    run_posthoc = (
        posthoc_key is not None
        and (gate == "always" or (gate == "significant" and main["p"] < alpha))
        and gate != "never"
    )

    posthoc_out: Optional[Any] = None
    if run_posthoc:
        if posthoc_key in ("tukey", "hsd"):
            posthoc_out = tukey_hsd(g, alpha=alpha, dropna=False)
        elif posthoc_key in ("mwu", "mw", "mannwhitney"):
            posthoc_out = pairwise_mwu(g, alpha=alpha, p_adjust=p_adjust, dropna=False)
        else:
            raise ValueError("posthoc must be 'tukey' or 'mwu' (or None)")

    return {
        "alpha": alpha,
        "groups": group_summary,
        "assumptions": assumptions,
        "main": main,
        "posthoc": posthoc_out,
    }


def kruskal_test(
    groups: GroupsLike,
    *,
    alpha: float = 0.05,
    posthoc: Optional[str] = None,
    posthoc_on: str = "significant",
    p_adjust: str = "holm",
    dropna: bool = True,
) -> Dict[str, Any]:
    """Kruskal–Wallis omnibus test for 3+ groups, with optional pairwise MWU posthoc."""
    g = _coerce_groups(groups, dropna=dropna)
    group_summary = _summarize_groups(g)

    stat, p = stats.kruskal(*list(g.values()))
    main = {"test": "Kruskal–Wallis", "stat": float(stat), "p": float(p)}

    posthoc_key = None if posthoc is None else _norm_key(posthoc)
    gate = _norm_key(posthoc_on or "significant")
    run_posthoc = (
        posthoc_key is not None
        and (gate == "always" or (gate == "significant" and p < alpha))
        and gate != "never"
    )

    posthoc_out = None
    if run_posthoc:
        if posthoc_key in ("mwu", "mw", "mannwhitney"):
            posthoc_out = pairwise_mwu(g, alpha=alpha, p_adjust=p_adjust, dropna=False)
        else:
            raise ValueError("posthoc for Kruskal should be 'mwu' (or None)")

    return {"alpha": alpha, "groups": group_summary, "main": main, "posthoc": posthoc_out}


def factorial_anova(
    data: pd.DataFrame,
    *,
    dv: str,
    factors: Sequence[str],
    alpha: float = 0.05,
    interaction: bool = True,
    typ: int = 2,
    dropna: bool = True,
    levene_center: str = "median",
    verbose: bool = True,
) -> Dict[str, Any]:
    """Factorial ANOVA using OLS with C(factor) terms; supports interaction on/off."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")
    if dv not in data.columns:
        raise ValueError(f"dv '{dv}' not found in data.")
    if not factors or any(f not in data.columns for f in factors):
        missing = [f for f in factors if f not in data.columns]
        raise ValueError(f"Missing factor columns: {missing}")

    cols = [dv, *factors]
    df = data[cols].copy()
    if dropna:
        df = df.dropna(subset=cols)

    rhs = " * ".join([f"C({f})" for f in factors]) if interaction else " + ".join([f"C({f})" for f in factors])
    formula = f"{dv} ~ {rhs}"

    model = ols(formula, data=df).fit()
    table = sm.stats.anova_lm(model, typ=typ)

    cell = df[factors].astype(str).agg("|".join, axis=1)
    cell_groups = {lvl: df.loc[cell == lvl, dv].to_numpy(dtype=float) for lvl in cell.unique()}

    hom = None
    if len(cell_groups) >= 2:
        _, p = stats.levene(*cell_groups.values(), center=levene_center)
        hom = {"test": "levene (cells)", "center": levene_center, "p": float(p), "ok": bool(p >= alpha)}
        if verbose and hom["ok"] is False:
            _LOG.warning("Homogeneity across cells may be violated (Levene p=%.4g).", hom["p"])

    return {
        "alpha": alpha,
        "formula": formula,
        "main": table,
        "assumptions": {"homogeneity_cells": hom},
        "model": model,
    }


def repeated_measures_anova(
    data: pd.DataFrame,
    *,
    dv: str,
    subject: str,
    within: WithinLike,
    between: Optional[BetweenLike] = None,
    alpha: float = 0.05,
    dropna: bool = True,
    fit_mixedlm: bool = False,
) -> Dict[str, Any]:
    """Repeated measures ANOVA (AnovaRM) with optional random-intercept mixed model."""
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a pandas DataFrame.")

    within_list = _as_list(within) or []
    between_list = _as_list(between) or []

    needed = [dv, subject, *within_list, *between_list]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = data[needed].copy()
    if dropna:
        df = df.dropna(subset=needed)

    rm = AnovaRM(df, depvar=dv, subject=subject, within=within_list, between=between_list or None)
    rm_res = rm.fit()

    out: Dict[str, Any] = {"alpha": alpha, "main": rm_res.anova_table}

    if fit_mixedlm:
        fixed_terms = [f"C({c})" for c in (within_list + between_list)]
        formula = f"{dv} ~ " + (" + ".join(fixed_terms) if fixed_terms else "1")

        m = mixedlm(formula, df, groups=df[subject])
        m_res = m.fit()

        resid = np.asarray(m_res.resid, dtype=float)
        resid_sample = resid[:5000] if resid.size > 5000 else resid
        resid_normality_p = float(stats.shapiro(resid_sample)[1]) if resid_sample.size >= 3 else np.nan

        out["mixedlm"] = {
            "formula": formula,
            "summary": m_res.summary(),
            "resid_normality_p": resid_normality_p,
        }

    return out


def analyze(
    *,
    test: str = "auto",
    groups: Optional[GroupsLike] = None,
    data: Optional[pd.DataFrame] = None,
    dv: Optional[str] = None,
    group: Optional[str] = None,
    factors: Optional[Sequence[str]] = None,
    subject: Optional[str] = None,
    within: Optional[WithinLike] = None,
    between: Optional[BetweenLike] = None,
    alpha: float = 0.05,
    method: str = "auto",
    posthoc: Optional[str] = None,
    posthoc_on: str = "significant",
    p_adjust: str = "holm",
    typ: int = 2,
    interaction: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Single entry point that routes to the appropriate analysis based on inputs."""
    test_key = _norm_key(test)
    t = _TEST_ALIASES.get(test_key, None)
    if t is None:
        raise ValueError("Unknown test. Try: auto, 1w/anova, welch, bf, trimmed, mw, kw, 2w, rm.")

    if test_key in ("welch", "bf", "trimmed"):
        method = test_key

    if groups is None and data is not None and dv and group and t in ("auto", "oneway"):
        if dv not in data.columns or group not in data.columns:
            raise ValueError("dv or group column not found in data.")
        groups = {
            str(k): data.loc[data[group] == k, dv].to_numpy(dtype=float)
            for k in data[group].dropna().unique()
        }

    if t == "auto":
        if groups is not None:
            return one_way_test(
                groups,
                alpha=alpha,
                method=method,
                posthoc=posthoc,
                posthoc_on=posthoc_on,
                p_adjust=p_adjust,
                verbose=verbose,
            )
        if data is not None and dv and subject and within:
            return repeated_measures_anova(data, dv=dv, subject=subject, within=within, between=between, alpha=alpha)
        if data is not None and dv and factors:
            return factorial_anova(
                data, dv=dv, factors=factors, alpha=alpha, interaction=interaction, typ=typ, verbose=verbose
            )
        raise ValueError(
            "auto needs either groups=... OR (data+dv+group) OR (data+dv+factors) OR (data+dv+subject+within)."
        )

    if t == "oneway":
        if groups is None:
            raise ValueError("oneway requires groups=... or (data+dv+group).")
        return one_way_test(
            groups,
            alpha=alpha,
            method=method,
            posthoc=posthoc,
            posthoc_on=posthoc_on,
            p_adjust=p_adjust,
            verbose=verbose,
        )

    if t == "mwu":
        if groups is None:
            raise ValueError("mw requires groups=... (two groups).")
        g = _coerce_groups(groups, dropna=True)
        if len(g) != 2:
            raise ValueError("Mann–Whitney U is a two-sample test. Use kw (Kruskal) for 3+ groups.")
        names = list(g.keys())
        stat, p = stats.mannwhitneyu(g[names[0]], g[names[1]], alternative="two-sided")
        return {
            "alpha": alpha,
            "groups": _summarize_groups(g),
            "main": {"test": "Mann–Whitney U", "a": names[0], "b": names[1], "u": float(stat), "p": float(p)},
        }

    if t == "kruskal":
        if groups is None:
            raise ValueError("kw requires groups=...")
        return kruskal_test(groups, alpha=alpha, posthoc=posthoc, posthoc_on=posthoc_on, p_adjust=p_adjust)

    if t == "factorial":
        if data is None or not dv:
            raise ValueError("2w requires data=... and dv='...'.")
        if not factors:
            raise ValueError("2w requires factors=[...].")
        return factorial_anova(
            data, dv=dv, factors=factors, alpha=alpha, interaction=interaction, typ=typ, verbose=verbose
        )

    if t == "rm":
        if data is None or not dv or not subject or not within:
            raise ValueError("rm requires data=..., dv='...', subject='...', within='...'.")
        return repeated_measures_anova(data, dv=dv, subject=subject, within=within, between=between, alpha=alpha)

    raise RuntimeError("Unhandled test type.")


# ---------------------------
# Plotting helpers
# ---------------------------

def _long_df_from_groups(groups, dv="value", group_col="group", dropna=True) -> pd.DataFrame:
    """Convert groups (dict or list) into a long-form DataFrame."""
    if isinstance(groups, dict):
        items = list(groups.items())
    else:
        items = [(f"g{i+1}", g) for i, g in enumerate(groups)]

    frames = []
    for name, arr in items:
        x = np.asarray(arr, dtype=float).reshape(-1)
        if dropna:
            x = x[~np.isnan(x)]
        frames.append(pd.DataFrame({group_col: name, dv: x}))

    return pd.concat(frames, ignore_index=True)


def plot_oneway(groups, dv="value", group_col="group", alpha=0.05, title="One-way: distribution + Tukey") -> None:
    """Boxplot + jitter for groups, and Tukey CI plot when 3+ groups are present."""
    df = _long_df_from_groups(groups, dv=dv, group_col=group_col, dropna=True)
    order = list(pd.unique(df[group_col]))

    _, ax = plt.subplots()
    data = [df.loc[df[group_col] == g, dv].to_numpy() for g in order]
    ax.boxplot(data, tick_labels=order)

    rng = np.random.default_rng(0)
    for i, g in enumerate(order, start=1):
        y = df.loc[df[group_col] == g, dv].to_numpy()
        x = i + rng.normal(0, 0.04, size=len(y))
        ax.plot(x, y, linestyle="none", marker="o", markersize=3, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(dv)
    plt.tight_layout()

    if len(order) >= 3:
        tukey = pairwise_tukeyhsd(endog=df[dv], groups=df[group_col], alpha=alpha)
        tukey.plot_simultaneous(figsize=(8, 4))
        plt.title("Tukey HSD: simultaneous confidence intervals")
        plt.tight_layout()

    plt.show()


def plot_factorial_interaction(df: pd.DataFrame, dv: str, factor_a: str, factor_b: str,
                               title: str = "Factorial: interaction plot") -> None:
    """Interaction plot for exactly two factors."""
    _, ax = plt.subplots()
    interaction_plot(x=df[factor_a], trace=df[factor_b], response=df[dv], ax=ax)
    ax.set_title(title)
    ax.set_xlabel(factor_a)
    ax.set_ylabel(dv)
    plt.tight_layout()
    plt.show()


def plot_repeated_measures(df: pd.DataFrame, dv: str, subject: str, within: str,
                           title: str = "Repeated measures: per-subject trends + mean") -> None:
    """Spaghetti plot per subject with an overlaid mean trend across within levels."""
    d = df[[dv, subject, within]].dropna().copy()
    within_levels = list(pd.unique(d[within]))
    d[within] = pd.Categorical(d[within], categories=within_levels, ordered=True)
    d = d.sort_values([subject, within])

    x_map = {lvl: i for i, lvl in enumerate(within_levels)}

    _, ax = plt.subplots()

    for _, subdf in d.groupby(subject, observed=False):
        xs = subdf[within].map(x_map).to_numpy()
        ys = subdf[dv].to_numpy(dtype=float)
        ax.plot(xs, ys, marker="o", linewidth=1, alpha=0.35)

    mean_by = d.groupby(within, observed=False)[dv].mean().reindex(within_levels)
    ax.plot(range(len(within_levels)), mean_by.to_numpy(), marker="o", linewidth=2)

    ax.set_title(title)
    ax.set_xlabel(within)
    ax.set_ylabel(dv)
    ax.set_xticks(range(len(within_levels)))
    ax.set_xticklabels(within_levels)
    plt.tight_layout()
    plt.show()


def plot_nonparametric(groups, dv="value", group_col="group", title="Nonparametric: distribution view") -> None:
    """Violin plot + jittered points (useful for Kruskal/MWU contexts)."""
    df = _long_df_from_groups(groups, dv=dv, group_col=group_col, dropna=True)
    order = list(pd.unique(df[group_col]))

    _, ax = plt.subplots()
    data = [df.loc[df[group_col] == g, dv].to_numpy() for g in order]
    ax.violinplot(data, showmeans=False, showmedians=True)

    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order)

    rng = np.random.default_rng(0)
    for i, g in enumerate(order, start=1):
        y = df.loc[df[group_col] == g, dv].to_numpy()
        x = i + rng.normal(0, 0.04, size=len(y))
        ax.plot(x, y, linestyle="none", marker="o", markersize=3, alpha=0.6)

    ax.set_title(title)
    ax.set_xlabel(group_col)
    ax.set_ylabel(dv)
    plt.tight_layout()
    plt.show()



'''
# UAT (User Acceptance Test) script for the full analysis + plotting toolkit.

# This script:
#  - Executes each analysis pathway (one-way variants, MWU, Kruskal+posthoc, factorial, RM-ANOVA, RM+MixedLM)
#  - Generates and saves plot images
#  - Writes a JSON report containing compact, JSON-safe summaries
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------
# Optional import section (if you put the toolkit in a module)
# ---------------------------------------
# from your_module_name import (
#     analyze,
#     one_way_test,
#     kruskal_test,
#     factorial_anova,
#     repeated_measures_anova,
#     plot_oneway,
#     plot_factorial_interaction,
#     plot_repeated_measures,
#     plot_nonparametric,
# )


# ---------------------------------------
# UAT helpers
# ---------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def _assert_true(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _is_finite_number(x: Any) -> bool:
    try:
        return np.isfinite(float(x))
    except Exception:
        return False


def _assert_main_has_stat_and_p(result: Dict[str, Any]) -> None:
    _assert_true("main" in result, "Result missing 'main'.")
    main = result["main"]
    if isinstance(main, dict):
        _assert_true("p" in main, "Main result missing p-value.")
        _assert_true(_is_finite_number(main["p"]), "Main p-value is not finite.")
        if "stat" in main:
            _assert_true(_is_finite_number(main["stat"]), "Main stat is not finite.")
    elif isinstance(main, pd.DataFrame):
        _assert_true(main.shape[0] > 0, "Main DataFrame is empty.")
    else:
        raise AssertionError("Main output is neither dict nor DataFrame.")


def _save_all_open_figs(out_dir: str, prefix: str) -> list[str]:
    """
    Saves all currently open Matplotlib figures to files.
    Returns a list of saved paths.
    """
    paths: list[str] = []
    fig_nums = list(plt.get_fignums())
    for i, num in enumerate(fig_nums, start=1):
        plt.figure(num)
        path = os.path.join(out_dir, f"{prefix}_{i}.png")
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        paths.append(path)
    plt.close("all")
    return paths


def _json_safe(obj: Any, *, max_rows: int = 25) -> Any:
    """
    Convert common scientific Python objects into JSON-safe representations.
    Keeps outputs compact (e.g., DataFrame head only).
    """
    if obj is None:
        return None

    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()

    if isinstance(obj, (list, tuple)):
        return [_json_safe(x, max_rows=max_rows) for x in obj]

    if isinstance(obj, dict):
        return {str(k): _json_safe(v, max_rows=max_rows) for k, v in obj.items()}

    if isinstance(obj, set):
        return [_json_safe(x, max_rows=max_rows) for x in sorted(list(obj), key=lambda x: str(x))]

    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist(), max_rows=max_rows)

    if isinstance(obj, pd.Series):
        return _json_safe(obj.head(max_rows).to_dict(), max_rows=max_rows)

    if isinstance(obj, pd.DataFrame):
        return {
            "type": "DataFrame",
            "shape": [int(obj.shape[0]), int(obj.shape[1])],
            "columns": [str(c) for c in obj.columns],
            "head": obj.head(max_rows).to_dict(orient="records"),
        }

    # statsmodels summary objects and other rich objects
    try:
        s = str(obj)
        if len(s) > 5000:
            return s[:5000] + "...(truncated)"
        return s
    except Exception:
        return repr(obj)


def _capture_result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compact snapshot used for reporting (JSON-safe).
    """
    return _json_safe(result, max_rows=25)


# ---------------------------------------
# UAT data generation
# ---------------------------------------

def make_oneway_data(seed: int = 0):
    rng = np.random.default_rng(seed)
    g1 = rng.normal(0.0, 1.0, 30)
    g2 = rng.normal(0.3, 1.0, 30)
    g3 = rng.normal(1.4, 1.0, 30)
    return g1, g2, g3


def make_oneway_hetero_data(seed: int = 1):
    rng = np.random.default_rng(seed)
    g1 = rng.normal(0.0, 0.5, 35)
    g2 = rng.normal(0.3, 1.5, 35)
    g3 = rng.normal(1.0, 3.0, 35)
    return g1, g2, g3


def make_factorial_data(seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "y": rng.normal(size=120),
        "a": np.repeat(["A", "B"], 60),
        "b": np.tile(np.repeat(["C", "D", "E"], 20), 2),
    })


def make_repeated_measures_data(seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_subjects = 25
    levels = ["T1", "T2", "T3"]
    df = pd.DataFrame({
        "id": np.repeat(np.arange(n_subjects), len(levels)),
        "time": np.tile(levels, n_subjects),
    })
    subj_effect = rng.normal(0, 0.6, n_subjects)
    time_effect = {"T1": 0.0, "T2": 0.2, "T3": 0.35}
    df["y"] = (
        df["id"].map({i: subj_effect[i] for i in range(n_subjects)}).to_numpy()
        + df["time"].map(time_effect).to_numpy()
        + rng.normal(0, 0.8, len(df))
    )
    return df


# ---------------------------------------
# UAT scenarios
# ---------------------------------------

def uat_oneway_standard_and_posthoc(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 1: One-way ANOVA (auto) + Tukey posthoc + plot_oneway")

    g1, g2, g3 = make_oneway_data(seed=0)

    result = analyze(groups=[g1, g2, g3], test="auto", posthoc="tukey", posthoc_on="always")
    _assert_main_has_stat_and_p(result)

    print("Main:", result["main"])
    if isinstance(result.get("posthoc"), pd.DataFrame):
        print("Posthoc head:\n", result["posthoc"].head(10))

    plt.close("all")
    plot_oneway([g1, g2, g3], alpha=0.05)
    saved = _save_all_open_figs(out_dir, "uat1_oneway")

    return {"result": _capture_result_summary(result), "plots": saved}


def uat_oneway_auto_on_hetero(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 2: One-way ANOVA (auto method on heteroscedastic data)")

    g1, g2, g3 = make_oneway_hetero_data(seed=1)

    result = one_way_test(groups=[g1, g2, g3], method="auto", posthoc="tukey", posthoc_on="significant")
    _assert_main_has_stat_and_p(result)

    print("Main:", result["main"])
    print("Homogeneity check:", result["assumptions"]["homogeneity"])

    plt.close("all")
    plot_oneway([g1, g2, g3], alpha=0.05, title="One-way (heteroscedastic): distribution + Tukey")
    saved = _save_all_open_figs(out_dir, "uat2_oneway_hetero")

    return {"result": _capture_result_summary(result), "plots": saved}


def uat_oneway_variants(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 3: One-way ANOVA variants (equal / welch / bf / trimmed)")

    g1, g2, g3 = make_oneway_hetero_data(seed=4)

    results: Dict[str, Any] = {}
    for m in ["equal", "welch", "bf", "trimmed"]:
        r = one_way_test(groups=[g1, g2, g3], method=m, posthoc=None)
        _assert_main_has_stat_and_p(r)
        print(f"{m}:", r["main"])
        results[m] = _capture_result_summary(r)

    plt.close("all")
    plot_nonparametric([g1, g2, g3], title="Distribution view (context)")
    saved = _save_all_open_figs(out_dir, "uat3_distribution")

    return {"results": results, "plots": saved}


def uat_kw_and_mwu_posthoc(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 4: Kruskal–Wallis + pairwise MWU posthoc + plot_nonparametric")

    g1, g2, g3 = make_oneway_data(seed=6)

    result = analyze(groups=[g1, g2, g3], test="kw", posthoc="mwu", posthoc_on="always", p_adjust="holm")
    _assert_main_has_stat_and_p(result)

    print("Main:", result["main"])
    if isinstance(result.get("posthoc"), pd.DataFrame):
        print("Posthoc head:\n", result["posthoc"].head(10))
        needed = {"a", "b", "u", "p", "p_adj", "sig"}
        _assert_true(needed.issubset(result["posthoc"].columns), "MWU posthoc columns missing.")

    plt.close("all")
    plot_nonparametric([g1, g2, g3], title="Kruskal context: violin + jitter")
    saved = _save_all_open_figs(out_dir, "uat4_kruskal")

    return {"result": _capture_result_summary(result), "plots": saved}


def uat_factorial(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 5: Factorial ANOVA + interaction plot")

    df = make_factorial_data(seed=7)

    result = analyze(data=df, dv="y", factors=["a", "b"], test="2w", typ=2, interaction=True)
    _assert_main_has_stat_and_p(result)

    print(result["main"])

    plt.close("all")
    plot_factorial_interaction(df, dv="y", factor_a="a", factor_b="b", title="Factorial interaction: a x b")
    saved = _save_all_open_figs(out_dir, "uat5_factorial")

    return {"result": _capture_result_summary(result), "plots": saved}


def uat_repeated_measures(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 6: Repeated measures ANOVA + plot_repeated_measures")

    df = make_repeated_measures_data(seed=8)

    result = analyze(data=df, dv="y", subject="id", within="time", test="rm")
    _assert_main_has_stat_and_p(result)

    print(result["main"])

    plt.close("all")
    plot_repeated_measures(df, dv="y", subject="id", within="time", title="RM: spaghetti + mean")
    saved = _save_all_open_figs(out_dir, "uat6_rm")

    return {"result": _capture_result_summary(result), "plots": saved}


def uat_repeated_measures_with_mixedlm(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 7: Repeated measures + MixedLM (optional)")

    df = make_repeated_measures_data(seed=9)

    result = repeated_measures_anova(
        data=df,
        dv="y",
        subject="id",
        within="time",
        alpha=0.05,
        fit_mixedlm=True,
    )
    _assert_main_has_stat_and_p(result)

    print("RM main:\n", result["main"])
    if isinstance(result.get("mixedlm"), dict):
        print("MixedLM formula:", result["mixedlm"].get("formula"))
        print("MixedLM resid normality p:", result["mixedlm"].get("resid_normality_p"))

    plt.close("all")
    plot_repeated_measures(df, dv="y", subject="id", within="time", title="RM + MixedLM context plot")
    saved = _save_all_open_figs(out_dir, "uat7_rm_mixedlm")

    return {"result": _capture_result_summary(result), "plots": saved}


def uat_dataframe_oneway_convenience(out_dir: str) -> Dict[str, Any]:
    _print_header("UAT 8: DataFrame convenience for one-way (dv + group columns)")

    rng = np.random.default_rng(10)
    df = pd.DataFrame({
        "y": np.concatenate([rng.normal(0, 1, 30), rng.normal(0.5, 1, 30), rng.normal(1.5, 1, 30)]),
        "g": np.repeat(["A", "B", "C"], 30),
    })

    result = analyze(data=df, dv="y", group="g", test="auto", posthoc="tukey", posthoc_on="always")
    _assert_main_has_stat_and_p(result)

    print("Main:", result["main"])
    if isinstance(result.get("groups"), pd.DataFrame):
        print("Groups summary:\n", result["groups"])
    if isinstance(result.get("posthoc"), pd.DataFrame):
        print("Posthoc head:\n", result["posthoc"].head(10))

    plt.close("all")
    plot_oneway({"A": df.loc[df["g"] == "A", "y"], "B": df.loc[df["g"] == "B", "y"], "C": df.loc[df["g"] == "C", "y"]})
    saved = _save_all_open_figs(out_dir, "uat8_df_oneway")

    return {"result": _capture_result_summary(result), "plots": saved}


# ---------------------------------------
# UAT runner
# ---------------------------------------

def run_uat(out_dir: Optional[str] = None) -> None:
    out_dir = out_dir or os.path.join(os.getcwd(), "uat_outputs")
    _ensure_dir(out_dir)

    start = time.time()

    report: Dict[str, Any] = {}
    report["uat1_oneway_standard_and_posthoc"] = uat_oneway_standard_and_posthoc(out_dir)
    report["uat2_oneway_auto_on_hetero"] = uat_oneway_auto_on_hetero(out_dir)
    report["uat3_oneway_variants"] = uat_oneway_variants(out_dir)
    report["uat4_kw_and_mwu_posthoc"] = uat_kw_and_mwu_posthoc(out_dir)
    report["uat5_factorial"] = uat_factorial(out_dir)
    report["uat6_repeated_measures"] = uat_repeated_measures(out_dir)
    report["uat7_repeated_measures_with_mixedlm"] = uat_repeated_measures_with_mixedlm(out_dir)
    report["uat8_dataframe_oneway_convenience"] = uat_dataframe_oneway_convenience(out_dir)

    report_path = os.path.join(out_dir, "uat_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - start

    _print_header("UAT COMPLETE")
    print(f"Outputs directory: {out_dir}")
    print(f"Report saved to:   {report_path}")
    print(f"Elapsed seconds:   {elapsed:.2f}")


if __name__ == "__main__":
    run_uat()
'''