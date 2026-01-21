"""
Lightweight, user-friendly t-test utility with:

- One main entry point: analyze_ttest(a, b, ...)
- Flexible short inputs:
    alt: "two"/"2"/"two-sided", "<"/"less", ">"/"greater"
    paired: True/False or "paired"/"dep", "ind", "auto"
    equal_var (independent only): True/False or "auto", "welch"
- Optional assumption checks:
    paired => normality on (a-b)
    independent => normality (min p) + Levene variance test (median-centered)
- Readable printouts:
    style="list" | "table" | "blocks"
- Nicer plots:
    kind="bar" | "box", optional raw points overlay, optional subtitle with t/p
- Unit tests with deterministic sample data, including “plot generated” verification
  (plot is saved to a temporary PNG and file existence/size is asserted)

Important plotting fixes:
- show_plot=True works reliably in notebooks by using IPython.display when available
  (even if a headless backend like Agg was activated earlier).
- Matplotlib boxplot deprecation: uses tick_labels when supported, falls back otherwise.
- Tests temporarily switch backend to Agg and then restore it to avoid breaking plots
  in the same kernel/session.

Run tests in a terminal:
    python -m unittest -v this_file.py

Run tests in a notebook:
    %run this_file.py
"""

from __future__ import annotations

import logging
import math
import os
import tempfile
import unittest
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ----------------------------
# Data classes (rich output)
# ----------------------------

@dataclass(frozen=True)
class Assumptions:
    normality_p: Optional[float]
    normality_test: Optional[str]
    variance_p: Optional[float]
    variance_test: Optional[str]
    notes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class TTestResult:
    t: float
    p: float
    df: Optional[float]
    paired: bool
    alternative: str               # "two-sided" | "less" | "greater"
    equal_var: Optional[bool]      # None for paired, else True/False
    n1: int
    n2: int
    mean1: float
    mean2: float
    assumptions: Optional[Assumptions] = None


# ----------------------------
# Parsing / normalization helpers
# ----------------------------

def _as_1d(x: Any, nan: str = "omit") -> np.ndarray:
    """Convert input to 1D float array; handle NaNs with nan policy."""
    a = np.asarray(x, dtype=float).ravel()

    if nan not in {"omit", "propagate", "raise"}:
        raise ValueError("nan must be one of {'omit','propagate','raise'}")

    if nan == "omit":
        a = a[~np.isnan(a)]
    elif nan == "raise":
        if np.isnan(a).any():
            raise ValueError("NaNs found in input and nan='raise'")

    return a


def _norm_alt(alt: Optional[str]) -> str:
    """Normalize alternative hypothesis to: 'two-sided' | 'less' | 'greater'."""
    if alt is None:
        return "two-sided"
    s = str(alt).strip().lower()

    if s in {"two", "2", "two-sided", "twosided", "both", "two_sided"}:
        return "two-sided"
    if s in {"less", "<", "lower", "smaller", "left"}:
        return "less"
    if s in {"greater", ">", "higher", "larger", "right"}:
        return "greater"

    raise ValueError("alt must be one of: 'two'/'2', 'less'/'<', 'greater'/'>'")


def _norm_paired(paired: Union[bool, str, None]) -> Union[bool, str]:
    """Normalize paired to True/False or 'auto'."""
    if isinstance(paired, bool) or paired is None:
        return bool(paired) if paired is not None else False

    s = str(paired).strip().lower()
    if s in {"paired", "dep", "dependent", "within", "rel", "matched"}:
        return True
    if s in {"ind", "independent", "between", "unpaired"}:
        return False
    if s == "auto":
        return "auto"

    raise ValueError("paired must be bool or one of: 'paired'/'dep', 'ind', 'auto'")


def _norm_equal_var(equal_var: Union[bool, str]) -> Union[bool, str]:
    """Normalize equal_var to True/False or 'auto'/'welch'."""
    if isinstance(equal_var, bool):
        return equal_var
    s = str(equal_var).strip().lower()
    if s in {"auto", "a"}:
        return "auto"
    if s in {"welch", "w"}:
        return "welch"
    if s in {"equal", "pooled", "true", "t"}:
        return True
    if s in {"unequal", "false", "f"}:
        return False
    raise ValueError("equal_var must be bool or one of: 'auto', 'welch'")


# ----------------------------
# Assumption checks
# ----------------------------

def _normality_pvalue(x: np.ndarray, alpha: float) -> Tuple[Optional[float], Optional[str], Tuple[str, ...]]:
    """
    Choose a reasonable normality test:
      - Shapiro-Wilk for n <= 5000
      - D'Agostino-Pearson normaltest for n > 5000 (requires n >= 8)
    """
    n = len(x)
    if n < 3:
        return None, None, ("Normality check skipped (n < 3).",)

    notes = []
    try:
        if n <= 5000:
            _, p = stats.shapiro(x)
            test = "shapiro"
            notes.append("Shapiro-Wilk used (n <= 5000).")
        else:
            if n >= 8:
                _, p = stats.normaltest(x)
                test = "normaltest"
                notes.append("D'Agostino-Pearson normaltest used (n > 5000).")
            else:
                _, p = stats.shapiro(x)
                test = "shapiro"
                notes.append("Shapiro-Wilk used as fallback; results may be sensitive for large n.")
    except Exception as e:
        return None, None, (f"Normality check failed: {e}",)

    if p < alpha:
        notes.append("Normality test rejected at alpha.")
    else:
        notes.append("Normality test not rejected at alpha.")
    if n >= 200:
        notes.append("Large samples can reject normality for minor deviations.")

    return float(p), test, tuple(notes)


def _variance_pvalue(a: np.ndarray, b: np.ndarray, alpha: float) -> Tuple[Optional[float], Optional[str], Tuple[str, ...]]:
    """Levene test (median-centered) for equality of variances."""
    if len(a) < 2 or len(b) < 2:
        return None, None, ("Variance check skipped (need >=2 per group).",)

    try:
        _, p = stats.levene(a, b, center="median")
    except Exception as e:
        return None, None, (f"Variance check failed: {e}",)

    notes = [
        "Levene did not reject equal variances at alpha."
        if p >= alpha
        else "Levene rejected equal variances at alpha."
    ]
    return float(p), "levene", tuple(notes)


# ----------------------------
# Core t-test
# ----------------------------

def ttest(
    a: Any,
    b: Any,
    *,
    paired: Union[bool, str, None] = False,
    alt: Optional[str] = "two",
    alpha: float = 0.05,
    equal_var: Union[bool, str] = "auto",
    check: bool = True,
    nan: str = "omit",
    return_full: bool = False,
) -> Union[Tuple[float, float], TTestResult]:
    """
    Perform independent or paired t-test with intuitive inputs.

    Returns (t, p) by default; set return_full=True for a rich TTestResult.
    """
    x1 = _as_1d(a, nan=nan)
    x2 = _as_1d(b, nan=nan)

    paired_norm = _norm_paired(paired)
    paired_bool = (len(x1) == len(x2)) if paired_norm == "auto" else bool(paired_norm)
    alternative = _norm_alt(alt)

    if paired_bool and len(x1) != len(x2):
        raise ValueError("Paired t-test requires same length after NaN handling.")

    assumptions: Optional[Assumptions] = None
    if check:
        if paired_bool:
            diff = x1 - x2
            norm_p, norm_test, notes = _normality_pvalue(diff, alpha=alpha)
            assumptions = Assumptions(norm_p, norm_test, None, None, notes=notes)
            if norm_p is not None and norm_p < alpha:
                logger.warning("Paired t-test: differences may be non-normal (p=%.4g).", norm_p)
        else:
            n1p, _, n1notes = _normality_pvalue(x1, alpha=alpha)
            n2p, _, n2notes = _normality_pvalue(x2, alpha=alpha)
            norm_p = None if (n1p is None or n2p is None) else float(min(n1p, n2p))
            var_p, var_test, vnotes = _variance_pvalue(x1, x2, alpha=alpha)
            notes = tuple(n1notes) + tuple(n2notes) + tuple(vnotes)
            assumptions = Assumptions(norm_p, "min(normality)", var_p, var_test, notes=notes)
            if norm_p is not None and norm_p < alpha:
                logger.warning("Independent t-test: at least one sample may be non-normal (min p=%.4g).", norm_p)

    eq_var_bool: Optional[bool] = None
    if not paired_bool:
        eq = _norm_equal_var(equal_var)
        if eq == "welch":
            eq_var_bool = False
        elif eq == "auto":
            if assumptions is not None and assumptions.variance_p is not None:
                eq_var_bool = assumptions.variance_p >= alpha
            else:
                pv, _, _ = _variance_pvalue(x1, x2, alpha=alpha)
                eq_var_bool = True if pv is None else (pv >= alpha)
        else:
            eq_var_bool = bool(eq)

    # Run scipy tests (two-sided p from SciPy)
    if paired_bool:
        res = stats.ttest_rel(x1, x2)
        t_stat = float(res.statistic)
        p_two = float(res.pvalue)
        df = float(len(x1) - 1) if len(x1) >= 2 else None
    else:
        res = stats.ttest_ind(x1, x2, equal_var=bool(eq_var_bool))
        t_stat = float(res.statistic)
        p_two = float(res.pvalue)
        if len(x1) >= 2 and len(x2) >= 2:
            if eq_var_bool:
                df = float(len(x1) + len(x2) - 2)
            else:
                # Welch-Satterthwaite df approximation
                s1 = np.var(x1, ddof=1)
                s2 = np.var(x2, ddof=1)
                n1 = len(x1)
                n2 = len(x2)
                num = (s1 / n1 + s2 / n2) ** 2
                den = (s1 * s1) / (n1 * n1 * (n1 - 1)) + (s2 * s2) / (n2 * n2 * (n2 - 1))
                df = float(num / den) if den > 0 else None
        else:
            df = None

    # Convert to one-sided if requested
    if alternative == "two-sided":
        p = p_two
    else:
        p = p_two / 2.0
        if alternative == "greater" and t_stat < 0:
            p = 1.0 - p
        elif alternative == "less" and t_stat > 0:
            p = 1.0 - p

    if not return_full:
        return t_stat, float(p)

    return TTestResult(
        t=t_stat,
        p=float(p),
        df=df,
        paired=paired_bool,
        alternative=alternative,
        equal_var=eq_var_bool if not paired_bool else None,
        n1=int(len(x1)),
        n2=int(len(x2)),
        mean1=float(np.mean(x1)) if len(x1) else float("nan"),
        mean2=float(np.mean(x2)) if len(x2) else float("nan"),
        assumptions=assumptions,
    )


# ----------------------------
# Readable formatting (list/table/blocks)
# ----------------------------

def _fmt(x: Optional[float], digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "—"
    if abs(x) < 1e-4 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{digits}f}"


def _kv_table(rows, *, col1="Metric", col2="Value") -> str:
    left_w = max(len(col1), max((len(r[0]) for r in rows), default=0))
    right_w = max(len(col2), max((len(r[1]) for r in rows), default=0))
    line = f"+-{'-'*left_w}-+-{'-'*right_w}-+"
    out = [
        line,
        f"| {col1.ljust(left_w)} | {col2.ljust(right_w)} |",
        line,
    ]
    for k, v in rows:
        out.append(f"| {k.ljust(left_w)} | {v.ljust(right_w)} |")
    out.append(line)
    return "\n".join(out)


def format_ttest(
    result: Union[Tuple[float, float], TTestResult],
    *,
    style: str = "blocks",  # "list" | "table" | "blocks"
    digits: int = 4,
    title: Optional[str] = None,
) -> str:
    if isinstance(result, tuple):
        t, p = result
        rows = [("t", _fmt(float(t), digits)), ("p", _fmt(float(p), digits))]
        if style == "list":
            return "\n".join([f"- t: {_fmt(float(t), digits)}", f"- p: {_fmt(float(p), digits)}"])
        return _kv_table(rows)

    r = result
    header = title or "t-test result"
    test_name = "Paired t-test" if r.paired else "Independent t-test"

    core_rows = [
        ("Test", test_name),
        ("Alternative", r.alternative),
        ("t", _fmt(r.t, digits)),
        ("p", _fmt(r.p, digits)),
        ("df", _fmt(r.df, digits) if r.df is not None else "—"),
    ]
    if not r.paired:
        core_rows.append(("Equal variances", "Yes" if r.equal_var else "No (Welch)"))

    sample_rows = [
        ("n1", str(r.n1)),
        ("n2", str(r.n2)),
        ("mean1", _fmt(r.mean1, digits)),
        ("mean2", _fmt(r.mean2, digits)),
        ("mean diff (1-2)", _fmt(r.mean1 - r.mean2, digits)),
    ]

    assum_rows = []
    notes = ()
    if r.assumptions is not None:
        a = r.assumptions
        assum_rows = [
            ("Normality test", a.normality_test or "—"),
            ("Normality p", _fmt(a.normality_p, digits)),
            ("Variance test", a.variance_test or "—"),
            ("Variance p", _fmt(a.variance_p, digits)),
        ]
        notes = a.notes or ()

    if style == "table":
        rows = core_rows + [("—", "—")] + sample_rows
        if assum_rows:
            rows += [("—", "—")] + assum_rows
        return f"{header}\n{_kv_table(rows, col1='Field', col2='Value')}"

    if style == "list":
        lines = [
            f"{header}",
            f"- {test_name}, alternative={r.alternative}",
            f"- t={_fmt(r.t, digits)}, p={_fmt(r.p, digits)}, df={_fmt(r.df, digits) if r.df is not None else '—'}",
        ]
        if not r.paired:
            lines.append(f"- equal variances: {'Yes' if r.equal_var else 'No (Welch)'}")
        lines.append(f"- n1={r.n1}, mean1={_fmt(r.mean1, digits)}")
        lines.append(f"- n2={r.n2}, mean2={_fmt(r.mean2, digits)}")
        if assum_rows:
            lines.append(f"- normality p={_fmt(r.assumptions.normality_p, digits)}")
            if r.assumptions.variance_p is not None:
                lines.append(f"- variance p={_fmt(r.assumptions.variance_p, digits)}")
        return "\n".join(lines)

    # "blocks"
    parts = [
        f"{header}\n{test_name} ({r.alternative}) completed.\n{_kv_table(core_rows)}",
        "Samples\n" + _kv_table(sample_rows),
    ]
    if assum_rows:
        parts.append("Assumptions\n" + _kv_table(assum_rows))
    if notes:
        parts.append("Notes\n" + "\n".join([f"- {n}" for n in notes]))
    return "\n\n".join(parts)


# ----------------------------
# Plot helpers
# ----------------------------

def _show_figure(fig: plt.Figure) -> None:
    """
    Robust show:
    - In notebooks, IPython.display.display(fig) reliably renders even on Agg.
    - Otherwise, fallback to plt.show().
    """
    try:
        from IPython.display import display  # type: ignore
        display(fig)
    except Exception:
        plt.show()


def plot_groups(
    a: Any,
    b: Any,
    *,
    labels: Tuple[str, str] = ("Sample 1", "Sample 2"),
    kind: str = "bar",          # "bar" | "box"
    error: str = "sem",         # "sem" | "sd" | "ci"
    ci: float = 0.95,
    show_points: bool = True,
    jitter: float = 0.08,
    seed: int = 0,
    result: Optional[Union[Tuple[float, float], TTestResult]] = None,
    title: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    filename: Optional[str] = None,
    nan: str = "omit",
) -> plt.Axes:
    x1 = _as_1d(a, nan=nan)
    x2 = _as_1d(b, nan=nan)

    def _err(x: np.ndarray) -> float:
        if error == "sem":
            return float(stats.sem(x)) if len(x) > 1 else float("nan")
        if error == "sd":
            return float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
        if error == "ci":
            z = stats.norm.ppf(0.5 + ci / 2.0)
            sem = float(stats.sem(x)) if len(x) > 1 else float("nan")
            return float(z * sem)
        raise ValueError("error must be one of: 'sem', 'sd', 'ci'")

    means = np.array([np.mean(x1), np.mean(x2)], dtype=float)
    errs = np.array([_err(x1), _err(x2)], dtype=float)

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 5.2))

    # Clean look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.8, alpha=0.6)

    xs = np.array([0, 1])

    if kind == "bar":
        bars = ax.bar(xs, means, yerr=errs, capsize=6)
        ax.set_xticks(xs, labels)
        ax.set_ylabel("Mean")
        for b_ in bars:
            y = b_.get_height()
            ax.annotate(
                f"{y:.3g}",
                (b_.get_x() + b_.get_width() / 2, y),
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )
    elif kind == "box":
        # Matplotlib 3.9+: 'labels' renamed to 'tick_labels' (labels is deprecated).
        try:
            ax.boxplot([x1, x2], tick_labels=list(labels), showmeans=True)
        except TypeError:
            ax.boxplot([x1, x2], labels=list(labels), showmeans=True)

        ax.set_ylabel("Value")
    else:
        raise ValueError("kind must be 'bar' or 'box'")

    if show_points and kind == "bar":
        rng = np.random.default_rng(seed)
        x1j = xs[0] + rng.uniform(-jitter, jitter, size=len(x1))
        x2j = xs[1] + rng.uniform(-jitter, jitter, size=len(x2))
        ax.scatter(x1j, x1, s=18, alpha=0.7)
        ax.scatter(x2j, x2, s=18, alpha=0.7)

    ax.set_title(title or "Group comparison")

    if result is not None:
        if isinstance(result, tuple):
            t_val, p_val = float(result[0]), float(result[1])
        else:
            t_val, p_val = float(result.t), float(result.p)
        ax.text(
            0.5,
            1.02,
            f"t={t_val:.3g}, p={p_val:.3g}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
        )

    ax.margins(y=0.15)
    ax.figure.tight_layout()

    if filename:
        ax.figure.savefig(filename, bbox_inches="tight")

    return ax


# ----------------------------
# One main function that takes data
# ----------------------------

def analyze_ttest(
    a: Any,
    b: Any,
    *,
    paired: Union[bool, str, None] = False,
    alt: Optional[str] = "two",
    alpha: float = 0.05,
    equal_var: Union[bool, str] = "auto",
    check: bool = True,
    nan: str = "omit",
    style: str = "blocks",                  # "list" | "table" | "blocks"
    digits: int = 4,
    title: str = "t-test result",
    plot: bool = True,
    plot_kind: str = "bar",                 # "bar" | "box"
    plot_labels: Tuple[str, str] = ("Sample 1", "Sample 2"),
    plot_file: Optional[str] = None,        # if provided, saves plot to path
    show_points: bool = True,
    show_plot: bool = False,                # if True and plot_file is None, display the plot
) -> Tuple[TTestResult, str]:
    """
    Main UX function:
      - takes data (a, b)
      - runs t-test (returns full result)
      - returns formatted text
      - optionally saves and/or shows a plot

    Returns:
      (TTestResult, formatted_text)
    """
    res = ttest(
        a, b,
        paired=paired,
        alt=alt,
        alpha=alpha,
        equal_var=equal_var,
        check=check,
        nan=nan,
        return_full=True,
    )
    text = format_ttest(res, style=style, digits=digits, title=title)

    if plot:
        ax = plot_groups(
            a, b,
            labels=plot_labels,
            kind=plot_kind,
            show_points=show_points,
            result=res,
            title=f"{plot_labels[0]} vs {plot_labels[1]}",
            filename=plot_file,
            nan=nan,
        )

        if plot_file is not None:
            # saving implies we should close to avoid figure accumulation
            plt.close(ax.figure)
        elif show_plot:
            _show_figure(ax.figure)
            plt.close(ax.figure)
        else:
            # neither saved nor shown: close by default
            plt.close(ax.figure)

    return res, text


# ----------------------------
# Unit tests with sample data
# ----------------------------

class TestAnalyzeTTest(unittest.TestCase):
    _orig_backend: Optional[str] = None

    @classmethod
    def setUpClass(cls) -> None:
        # Temporarily switch to a headless backend for tests, but restore afterwards
        cls._orig_backend = matplotlib.get_backend()
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

    @classmethod
    def tearDownClass(cls) -> None:
        if cls._orig_backend:
            try:
                plt.switch_backend(cls._orig_backend)
            except Exception:
                pass

    def test_independent_two_sided_matches_scipy(self) -> None:
        a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 11], dtype=float)

        t_expected, p_expected = stats.ttest_ind(a, b, equal_var=True)
        res = ttest(a, b, paired=False, alt="two", equal_var=True, check=False, return_full=True)

        self.assertAlmostEqual(res.t, float(t_expected), places=12)
        self.assertAlmostEqual(res.p, float(p_expected), places=12)

    def test_one_sided_greater_conversion(self) -> None:
        a = np.array([1, 2, 3, 4, 5], dtype=float)
        b = np.array([10, 11, 12, 13, 14], dtype=float)

        t_two, p_two = stats.ttest_ind(a, b, equal_var=True)
        t_two = float(t_two)
        p_two = float(p_two)

        res = ttest(a, b, paired=False, alt=">", equal_var=True, check=False, return_full=True)

        p_expected = p_two / 2.0
        if t_two < 0:
            p_expected = 1.0 - p_expected

        self.assertAlmostEqual(res.t, t_two, places=12)
        self.assertAlmostEqual(res.p, p_expected, places=12)
        self.assertEqual(res.alternative, "greater")

    def test_paired_length_mismatch_raises(self) -> None:
        a = np.array([1, 2, 3], dtype=float)
        b = np.array([1, 2, 3, 4], dtype=float)
        with self.assertRaises(ValueError):
            ttest(a, b, paired=True, check=False)

    def test_analyze_ttest_formats_and_saves_plot(self) -> None:
        rng = np.random.default_rng(123)
        a = rng.normal(loc=50, scale=5, size=60)
        b = rng.normal(loc=51, scale=5, size=60)

        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "plot.png")
            res, text = analyze_ttest(
                a, b,
                paired=False,
                alt="two",
                equal_var="auto",
                check=True,
                style="blocks",
                plot=True,
                plot_kind="bar",
                plot_labels=("Control", "Treatment"),
                plot_file=out,
                show_plot=False,  # tests validate file output, not display
            )

            self.assertIsInstance(res, TTestResult)
            self.assertTrue(isinstance(text, str) and len(text) > 0)
            self.assertTrue(os.path.exists(out) and os.path.getsize(out) > 0)
            self.assertIn("t-test result", text)
            self.assertIn("t", text)
            self.assertIn("p", text)


'''
if __name__ == "__main__":
    import sys

    # Notebook/IPython-safe runner: ignore injected argv, and avoid sys.exit() in kernels
    if "ipykernel" in sys.modules:
        unittest.main(argv=["unittest"], verbosity=2, exit=False)
    else:
        unittest.main(verbosity=2)


# Viusal example
rng = np.random.default_rng(123)
a = rng.normal(50, 5, 60)
b = rng.normal(51, 5, 60)

# Bar
res, text = analyze_ttest(a, b, plot=True, plot_kind="bar", show_plot=True, plot_file=None, plot_labels=("Control", "Treatment"))
print(text)

# Box
res, text = analyze_ttest(a, b, plot=True, plot_kind="box", show_plot=True, plot_file=None, plot_labels=("Control", "Treatment"))
print(text)
'''