from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, Dict, Any

import warnings
import numpy as np

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover
    norm = None


AlphaSpec = Union[
    float,                    # e.g. 0.3
    str,                      # "auto"
    Tuple[float, float],      # (low, high) bounds, grid inferred
    Iterable[float],          # explicit candidates
    np.ndarray
]


def _as_1d_float_array(x) -> np.ndarray:
    a = np.asarray(x, dtype=float).reshape(-1)
    if a.size == 0:
        raise ValueError("Input series is empty.")
    return a


def _validate_series(y: np.ndarray, nan_policy: str) -> None:
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'.")
    if np.isnan(y).any():
        if nan_policy == "raise":
            raise ValueError("Series contains NaNs. Use nan_policy='omit' or clean the data.")
        # 'omit' handled by filtering in caller


def _clamp_alpha(alpha: float, eps: float = 1e-6) -> float:
    # Keep alpha away from 0 and 1 to avoid division blow-ups in trend term.
    return float(np.clip(alpha, eps, 1.0 - eps))


def _initial_states_regression(y: np.ndarray, k: int, alpha: float) -> Tuple[float, float]:
    """
    Regression-based init using first k points.
    Replicates your beta0/beta1 intent without sklearn.
    """
    n = y.size
    k = int(k)
    if k < 2:
        k = 2
    if k > n:
        k = n

    x = np.arange(1, k + 1, dtype=float)
    # polyfit returns [slope, intercept]
    beta1, beta0 = np.polyfit(x, y[:k], deg=1)

    S1_0 = beta0 - ((1 - alpha) / alpha) * beta1
    S2_0 = beta0 - ((2 * (1 - alpha)) / alpha) * beta1
    return float(S1_0), float(S2_0)


def _initial_states_first(y: np.ndarray, alpha: float) -> Tuple[float, float]:
    """
    Simple init: uses first value and first difference as trend proxy.
    Kept intentionally simple for "I just want it to work" usage.
    """
    if y.size == 1:
        level = y[0]
        trend = 0.0
    else:
        level = y[0]
        trend = y[1] - y[0]

    beta0 = level
    beta1 = trend

    S1_0 = beta0 - ((1 - alpha) / alpha) * beta1
    S2_0 = beta0 - ((2 * (1 - alpha)) / alpha) * beta1
    return float(S1_0), float(S2_0)


def _compute_states(y: np.ndarray, alpha: float, S1_0: float, S2_0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Brown double exponential smoothing states:
      a_t = 2*S1_t - S2_t
      b_t = (alpha/(1-alpha))*(S1_t - S2_t)
    """
    n = y.size
    S1 = np.empty(n, dtype=float)
    S2 = np.empty(n, dtype=float)
    a = np.empty(n, dtype=float)
    b = np.empty(n, dtype=float)

    S1[0], S2[0] = S1_0, S2_0
    a[0] = 2 * S1[0] - S2[0]
    b[0] = (alpha / (1 - alpha)) * (S1[0] - S2[0])

    for t in range(1, n):
        S1[t] = alpha * y[t] + (1 - alpha) * S1[t - 1]
        S2[t] = alpha * S1[t] + (1 - alpha) * S2[t - 1]
        a[t] = 2 * S1[t] - S2[t]
        b[t] = (alpha / (1 - alpha)) * (S1[t] - S2[t])

    return a, b


def _aligned_insample_forecast(
    y: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    lead: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    lead-step-ahead in-sample forecasts aligned to y.
    Vectorized version of your loop.
    """
    n = y.size
    lead = int(lead)
    if lead < 1:
        raise ValueError("lead must be >= 1.")

    forecasts = np.full(n, np.nan, dtype=float)
    se = np.full(n, np.nan, dtype=float)

    if n <= lead:
        return forecasts, se

    t = np.arange(0, n - lead, dtype=int)
    f = a[t] + b[t] * lead
    idx = t + lead
    forecasts[idx] = f
    se[idx] = (y[idx] - f) ** 2
    return forecasts, se


def _future_forecast(a_last: float, b_last: float, horizon: int) -> np.ndarray:
    horizon = int(horizon)
    if horizon < 1:
        return np.array([], dtype=float)
    h = np.arange(1, horizon + 1, dtype=float)
    return a_last + b_last * h


def _z_for_ci(ci_level: float) -> float:
    if norm is None:
        raise RuntimeError("scipy is required for confidence intervals (scipy.stats.norm).")
    return float(norm.ppf(0.5 + ci_level / 2.0))


@dataclass(frozen=True)
class SmoothingResult:
    alpha: float
    insample: np.ndarray
    future: np.ndarray
    sse: float
    sigma: Optional[float] = None
    intervals: Optional[Dict[str, Any]] = None  # {'insample': (lo,hi), 'future': (lo,hi)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha_opt": float(self.alpha),
            "Forecasts": self.future,
            "insample": self.insample,
            "sse": float(self.sse),
            "sigma": None if self.sigma is None else float(self.sigma),
            "intervals": self.intervals,
        }


def brown_exponential_smoothing(
    series,
    *,
    alpha: AlphaSpec = "auto",
    horizon: int = 1,
    lead: int = 1,
    init: str = "regression",
    k: Optional[int] = None,
    alpha_bounds: Tuple[float, float] = (0.02, 0.98),
    grid: int = 200,
    objective: str = "sse",
    ci_level: Optional[float] = 0.90,
    nan_policy: str = "raise",
    plot: bool = False,
):
    """
    Brown (double) exponential smoothing with trend (single alpha).

    Alpha usage:
      - alpha=0.3
      - alpha="auto"
      - alpha=(0.05, 0.95)
      - alpha=[0.2, 0.4, 0.6]
    """
    y0 = _as_1d_float_array(series)
    _validate_series(y0, nan_policy)
    y = y0[~np.isnan(y0)] if nan_policy == "omit" else y0

    n = y.size
    if n == 0:
        raise ValueError("No valid data points after NaN handling.")

    lead = int(lead)
    horizon = int(horizon)
    if lead < 1:
        raise ValueError("lead must be >= 1.")

    if k is None:
        k = min(10, n)

    objective = objective.lower()
    if objective not in {"sse", "mse"}:
        raise ValueError("objective must be 'sse' or 'mse'.")

    init = init.lower()
    if init not in {"regression", "first"}:
        raise ValueError("init must be 'regression' or 'first'.")

    def candidates_from_spec(spec: AlphaSpec) -> np.ndarray:
        if isinstance(spec, (float, int, np.floating, np.integer)):
            return np.array([_clamp_alpha(float(spec))], dtype=float)

        if isinstance(spec, str):
            if spec.lower() != "auto":
                raise ValueError("alpha as string must be 'auto'.")
            lo, hi = alpha_bounds
            lo, hi = _clamp_alpha(lo), _clamp_alpha(hi)
            if lo >= hi:
                raise ValueError("alpha_bounds must satisfy low < high.")
            return np.linspace(lo, hi, int(grid), dtype=float)

        if (
            isinstance(spec, tuple)
            and len(spec) == 2
            and all(isinstance(v, (float, int, np.floating, np.integer)) for v in spec)
        ):
            lo, hi = spec
            lo, hi = _clamp_alpha(float(lo)), _clamp_alpha(float(hi))
            if lo >= hi:
                raise ValueError("alpha bounds must satisfy low < high.")
            return np.linspace(lo, hi, int(grid), dtype=float)

        arr = np.asarray(list(spec), dtype=float).reshape(-1)
        if arr.size == 0:
            raise ValueError("alpha candidates are empty.")
        return np.array([_clamp_alpha(float(a)) for a in arr], dtype=float)

    alphas = candidates_from_spec(alpha)

    # n == 1: cannot score any in-sample objective. Pick a deterministic alpha and proceed.
    if n == 1:
        chosen = float(alphas[alphas.size // 2]) if alphas.size > 1 else float(alphas[0])
        if init == "regression":
            warnings.warn("init='regression' needs >=2 points; using init='first' for n=1.", RuntimeWarning)
        S1_0, S2_0 = _initial_states_first(y, alpha=chosen)
        a_state, b_state = _compute_states(y, chosen, S1_0, S2_0)
        insample, _ = _aligned_insample_forecast(y, a_state, b_state, lead=lead)
        future = _future_forecast(a_state[-1], b_state[-1], horizon=horizon)
        result = SmoothingResult(alpha=chosen, insample=insample, future=future, sse=float("nan"))
        if plot:
            plot_smoothing(y, result)
        return result

    # If lead is too large to score, use lead_eff for scoring only.
    lead_eff = lead if lead < n else (n - 1)
    if lead_eff != lead:
        warnings.warn(
            f"lead={lead} >= n={n}: no in-sample points to score. Using lead_eff={lead_eff} for alpha selection only.",
            RuntimeWarning,
        )

    best_score = np.inf
    best_alpha = None
    best_state = None  # (a_state, b_state)

    for a_alpha in alphas:
        if init == "regression":
            S1_0, S2_0 = _initial_states_regression(y, k=k, alpha=a_alpha)
        else:
            S1_0, S2_0 = _initial_states_first(y, alpha=a_alpha)

        a_state, b_state = _compute_states(y, a_alpha, S1_0, S2_0)

        # Score using lead_eff (guaranteed to yield at least one defined point because n>=2 and lead_eff<=n-1)
        _, se_eff = _aligned_insample_forecast(y, a_state, b_state, lead=lead_eff)
        valid = ~np.isnan(se_eff)
        if not np.any(valid):
            continue

        score = float(np.nansum(se_eff))
        if objective == "mse":
            score = score / float(np.sum(valid))

        if score < best_score:
            best_score = score
            best_alpha = float(a_alpha)
            best_state = (a_state, b_state)

    if best_alpha is None or best_state is None:
        raise RuntimeError("Unable to fit model (no valid in-sample points to score).")

    a_state, b_state = best_state

    # Produce outputs for the requested lead (may be all-NaN when lead >= n)
    insample, _ = _aligned_insample_forecast(y, a_state, b_state, lead=lead)
    future = _future_forecast(a_state[-1], b_state[-1], horizon=horizon)

    # Confidence intervals: estimate sigma from lead_eff residuals so CI can exist even if lead >= n.
    sigma = None
    intervals = None
    if ci_level is not None:
        ins_eff, _ = _aligned_insample_forecast(y, a_state, b_state, lead=lead_eff)
        mask_eff = ~np.isnan(ins_eff)
        resid_eff = y[mask_eff] - ins_eff[mask_eff]

        if resid_eff.size >= 2:
            sigma = float(np.std(resid_eff, ddof=1))
            z = _z_for_ci(float(ci_level))

            intervals = {}
            # Insample interval only where requested insample exists
            mask_req = ~np.isnan(insample)
            if np.any(mask_req):
                lo_ins = insample.copy()
                hi_ins = insample.copy()
                lo_ins[mask_req] = insample[mask_req] - z * sigma
                hi_ins[mask_req] = insample[mask_req] + z * sigma
                intervals["insample"] = (lo_ins, hi_ins)

            intervals["future"] = (future - z * sigma, future + z * sigma)

    result = SmoothingResult(
        alpha=float(best_alpha),
        insample=insample,
        future=future,
        sse=float(best_score),
        sigma=sigma,
        intervals=intervals,
    )

    if plot:
        plot_smoothing(y, result)

    return result


def plot_smoothing(y: np.ndarray, result: SmoothingResult, *, title: str = "Exponential Smoothing and Forecasting"):
    import matplotlib.pyplot as plt

    y = _as_1d_float_array(y)
    x = np.arange(y.size)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o", linewidth=1.6, label="Actual", zorder=3)
    ax.plot(x, result.insample, linestyle="--", linewidth=1.8, label="In-sample forecast", zorder=2)

    if result.intervals and "insample" in result.intervals:
        lo_ins, hi_ins = result.intervals["insample"]
        mask = ~np.isnan(result.insample)
        ax.fill_between(x[mask], lo_ins[mask], hi_ins[mask], alpha=0.18, label="CI (in-sample)", zorder=1)

    if result.future.size > 0:
        xf = np.arange(y.size, y.size + result.future.size)
        ax.plot(xf, result.future, linestyle=":", linewidth=2.0, label="Out-of-sample", zorder=3)
        if result.intervals and "future" in result.intervals:
            lo_fut, hi_fut = result.intervals["future"]
            ax.fill_between(xf, lo_fut, hi_fut, alpha=0.18, label="CI (out-of-sample)", zorder=1)

    ax.grid(True, which="major", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.show()


# Backward-compatible wrapper (keeps your original call pattern mostly intact)
def exponential_smoother(Pt, k, alpha_range, l, plot=False, ci_level=0.90):
    res = brown_exponential_smoothing(
        Pt,
        alpha=tuple(alpha_range) if isinstance(alpha_range, (list, tuple)) and len(alpha_range) == 2 else "auto",
        horizon=int(l),
        lead=int(l),
        init="regression",
        k=int(k),
        ci_level=ci_level,
        plot=plot,
    )
    return res.to_dict()



'''
if __name__ == "__main__":
    Pt = np.array([12, 15, 14, 16, 19, 20, 22, 25, 24, 23], dtype=float)

    r1 = brown_exponential_smoothing(Pt, alpha="auto", horizon=2, lead=2, plot=True)
    print(r1.alpha, r1.future)

    r2 = brown_exponential_smoothing(Pt, alpha=0.35, horizon=3, lead=1, init="first")
    print(r2.to_dict())

    old = exponential_smoother(Pt, k=3, alpha_range=[0, 0.99], l=2, plot=True, ci_level=0.9)
    print(old["alpha_opt"], old["Forecasts"])





# Cell 2: Full UAT / smoke + behavior tests for the smoothing module
# Assumes you already executed the implementation cell (your big definition block).

import numpy as np

def _assert(cond, msg="Assertion failed"):
    if not cond:
        raise AssertionError(msg)

def _assert_close(a, b, tol=1e-9, msg="Not close"):
    if not (abs(a - b) <= tol):
        raise AssertionError(f"{msg}: {a} vs {b} (tol={tol})")

def _assert_array_finite(a, msg="Array has non-finite"):
    a = np.asarray(a, dtype=float)
    if not np.all(np.isfinite(a)):
        raise AssertionError(msg)

def _assert_array_shape(a, shape, msg="Bad shape"):
    a = np.asarray(a)
    if a.shape != shape:
        raise AssertionError(f"{msg}: expected {shape}, got {a.shape}")

def _assert_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return
    except Exception as e:
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}") from e
    raise AssertionError(f"Expected {exc_type.__name__} to be raised, but no exception occurred.")

def _run_test(name, fn):
    try:
        fn()
        print(f"PASS: {name}")
        return True
    except Exception as e:
        print(f"FAIL: {name} -> {type(e).__name__}: {e}")
        return False


# -----------------------------
# Test data helpers
# -----------------------------
def _toy_series():
    return np.array([12, 15, 14, 16, 19, 20, 22, 25, 24, 23], dtype=float)

def _trend_series(n=30, slope=2.0, intercept=10.0):
    x = np.arange(n, dtype=float)
    return intercept + slope * x

def _noisy_trend(n=60, slope=0.5, intercept=20.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=float)
    return intercept + slope * x + rng.normal(0.0, noise, size=n)

def _series_with_nans():
    return np.array([1.0, 2.0, np.nan, 4.0, 5.0], dtype=float)


# -----------------------------
# UAT tests
# -----------------------------
def test_basic_auto_runs_and_shapes():
    y = _toy_series()
    r = brown_exponential_smoothing(y, alpha="auto", horizon=3, lead=2, plot=False)
    _assert(isinstance(r.alpha, float), "alpha should be float")
    _assert_array_shape(r.insample, (y.size,), "insample should match series length")
    _assert_array_shape(r.future, (3,), "future should match horizon")
    _assert(np.isfinite(r.sse), "sse should be finite")
    _assert_array_finite(r.future, "future should be finite")

def test_alpha_fixed_runs():
    y = _toy_series()
    r = brown_exponential_smoothing(y, alpha=0.35, horizon=4, lead=1, init="first", plot=False, ci_level=None)
    _assert_close(r.alpha, 0.35, tol=1e-12, msg="alpha should match fixed value")
    _assert_array_shape(r.future, (4,), "future horizon mismatch")
    _assert(r.intervals is None, "intervals should be None when ci_level=None")

def test_alpha_bounds_tuple_runs():
    y = _toy_series()
    r = brown_exponential_smoothing(y, alpha=(0.10, 0.90), grid=50, horizon=2, lead=1)
    _assert(0.0 < r.alpha < 1.0, "alpha should be between 0 and 1")
    _assert_array_shape(r.future, (2,), "future horizon mismatch")

def test_alpha_candidates_iterable_runs():
    y = _toy_series()
    candidates = [0.2, 0.4, 0.6]
    r = brown_exponential_smoothing(y, alpha=candidates, horizon=2, lead=1)
    _assert(r.alpha in candidates, "alpha should be one of the candidates")

def test_objective_mse_runs():
    y = _toy_series()
    r = brown_exponential_smoothing(y, alpha="auto", objective="mse", horizon=2, lead=1)
    _assert(np.isfinite(r.sse), "mse objective result should be finite")

def test_lead_too_large_returns_nan_insample_but_future_ok():
    y = _toy_series()
    lead = y.size + 5
    r = brown_exponential_smoothing(y, alpha=0.4, horizon=3, lead=lead, ci_level=None)
    # With lead > n, insample should be all NaN; but future should still be produced.
    _assert(np.all(np.isnan(r.insample)), "insample should be all NaN when lead > n")
    _assert_array_shape(r.future, (3,), "future horizon mismatch")
    _assert_array_finite(r.future, "future should be finite")

def test_intervals_present_when_scipy_available():
    y = _noisy_trend(n=80, noise=2.0, seed=42)
    if norm is None:
        # If scipy isn't installed, CI should raise if requested.
        _assert_raises(RuntimeError, brown_exponential_smoothing, y, alpha="auto", horizon=3, lead=1, ci_level=0.9)
        return

    r = brown_exponential_smoothing(y, alpha="auto", horizon=3, lead=1, ci_level=0.9)
    _assert(r.intervals is not None, "intervals should be present when ci_level set and residuals exist")
    _assert("insample" in r.intervals and "future" in r.intervals, "intervals should include insample and future")
    lo_ins, hi_ins = r.intervals["insample"]
    _assert_array_shape(lo_ins, (y.size,), "lo_ins shape mismatch")
    _assert_array_shape(hi_ins, (y.size,), "hi_ins shape mismatch")
    _assert(np.all((hi_ins[~np.isnan(r.insample)] - lo_ins[~np.isnan(r.insample)]) >= 0), "CI band should be non-negative width")
    lo_f, hi_f = r.intervals["future"]
    _assert_array_shape(lo_f, (3,), "future lo shape mismatch")
    _assert_array_shape(hi_f, (3,), "future hi shape mismatch")

def test_nan_policy_raise_and_omit():
    y = _series_with_nans()
    _assert_raises(ValueError, brown_exponential_smoothing, y, alpha=0.5, horizon=2, lead=1, nan_policy="raise", ci_level=None)

    r = brown_exponential_smoothing(y, alpha=0.5, horizon=2, lead=1, nan_policy="omit", ci_level=None)
    _assert_array_shape(r.insample, (4,), "omit should drop NaNs and shorten series")
    _assert_array_shape(r.future, (2,), "future horizon mismatch")
    _assert_array_finite(r.future, "future should be finite")

def test_invalid_params_raise():
    y = _toy_series()

    _assert_raises(ValueError, brown_exponential_smoothing, y, alpha="bogus", horizon=2, lead=1)
    _assert_raises(ValueError, brown_exponential_smoothing, y, alpha="auto", horizon=2, lead=0)
    _assert_raises(ValueError, brown_exponential_smoothing, y, alpha="auto", horizon=2, lead=1, init="bad_init")
    _assert_raises(ValueError, brown_exponential_smoothing, y, alpha="auto", horizon=2, lead=1, objective="bad_obj")
    _assert_raises(ValueError, brown_exponential_smoothing, y, alpha="auto", horizon=2, lead=1, nan_policy="bad_nan_policy")
    _assert_raises(ValueError, brown_exponential_smoothing, [], alpha="auto")  # empty series

def test_backward_compatible_wrapper():
    y = _toy_series()
    out = exponential_smoother(y, k=3, alpha_range=[0, 0.99], l=2, plot=False, ci_level=None)
    _assert("alpha_opt" in out and "Forecasts" in out, "wrapper output keys missing")
    _assert_array_shape(out["Forecasts"], (2,), "wrapper Forecasts horizon mismatch")

def test_reasonable_on_perfect_trend():
    # On perfect trend, should forecast close to continuation (not necessarily exact, but should be sane)
    y = _trend_series(n=40, slope=3.0, intercept=5.0)
    r = brown_exponential_smoothing(y, alpha="auto", horizon=5, lead=1, ci_level=None)
    # Expected continuation: last value + slope*h
    last = y[-1]
    slope = 3.0
    expected = np.array([last + slope * h for h in range(1, 6)], dtype=float)
    # Allow some tolerance because smoothing/initialization may not be perfect
    max_abs_err = float(np.max(np.abs(r.future - expected)))
    _assert(max_abs_err < 2.0, f"forecast too far from expected continuation (max_abs_err={max_abs_err})")


# -----------------------------
# Run all tests
# -----------------------------
tests = [
    ("basic auto runs + shapes", test_basic_auto_runs_and_shapes),
    ("alpha fixed runs", test_alpha_fixed_runs),
    ("alpha bounds tuple runs", test_alpha_bounds_tuple_runs),
    ("alpha candidates iterable runs", test_alpha_candidates_iterable_runs),
    ("objective mse runs", test_objective_mse_runs),
    ("lead too large: insample NaN but future ok", test_lead_too_large_returns_nan_insample_but_future_ok),
    ("intervals present when scipy available", test_intervals_present_when_scipy_available),
    ("nan_policy raise and omit", test_nan_policy_raise_and_omit),
    ("invalid params raise", test_invalid_params_raise),
    ("backward compatible wrapper", test_backward_compatible_wrapper),
    ("reasonable on perfect trend", test_reasonable_on_perfect_trend),
]

passed = 0
for name, fn in tests:
    if _run_test(name, fn):
        passed += 1

print(f"\nUAT Summary: {passed}/{len(tests)} tests passed.")


'''