"""Diagnostics that make backtest assumptions explicit.

Every check returns a :class:`CheckResult` with two labels:

``status``
    ``PASS``, ``WARN``, ``FAIL`` or ``INFO``.

``scope``
    ``"verified"`` -- the software tested a concrete, mechanical property of
    *your inputs* (e.g. "this signal function's past values do not change when
    future rows are deleted"). A PASS means only that this specific test found
    nothing; it is not a certificate that the backtest is free of bias.

    ``"illustrative"`` -- the software measured a sensitivity (to costs, to
    delay, to the number of strategies tried) that *illustrates* a risk. These
    cannot prove or disprove bias; they tell you how fragile a result is.

Things this toolkit **cannot** detect include: look-ahead hidden in external
data (restated fundamentals, revised macro series, point-in-time index
membership), survivorship bias in a universe you supply, parameters tuned on
the full sample before the signal function was written, and any leakage that
does not change past signal values when future price rows are removed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from . import metrics
from .backtest import breakeven_cost_bps, run_backtest

Status = Literal["PASS", "WARN", "FAIL", "INFO"]
Scope = Literal["verified", "illustrative"]
SignalFn = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class CheckResult:
    name: str
    status: Status
    scope: Scope
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.status:<4}] ({self.scope}) {self.name}: {self.message}"


# --------------------------------------------------------------------------
# Verified checks: mechanical properties of the inputs
# --------------------------------------------------------------------------


def check_price_data(prices: pd.DataFrame) -> CheckResult:
    """Prices are non-empty, strictly positive, finite, with a sorted unique index."""
    problems = []
    if prices.empty:
        problems.append("price frame is empty")
    else:
        if not prices.index.is_monotonic_increasing:
            problems.append("index is not sorted ascending")
        if prices.index.has_duplicates:
            problems.append(f"{int(prices.index.duplicated().sum())} duplicate timestamps")
        n_nan = int(prices.isna().sum().sum())
        if n_nan:
            problems.append(f"{n_nan} missing values")
        values = prices.to_numpy(dtype=float)
        if np.isinf(values).any():
            problems.append("infinite values present")
        if (values[np.isfinite(values)] <= 0).any():
            problems.append("non-positive prices present")
    if problems:
        return CheckResult("price_data", "FAIL", "verified", "; ".join(problems), {"problems": problems})
    return CheckResult(
        "price_data",
        "PASS",
        "verified",
        f"{len(prices)} rows x {prices.shape[1]} assets, sorted, unique, positive, no gaps",
    )


def check_signal_alignment(prices: pd.DataFrame, signal: pd.DataFrame, max_gross: float = 1.0) -> CheckResult:
    """Signal has the same index/columns as prices, finite values, bounded leverage."""
    problems, warnings = [], []
    if not signal.index.equals(prices.index):
        missing = len(prices.index.difference(signal.index))
        extra = len(signal.index.difference(prices.index))
        problems.append(f"index mismatch ({missing} price dates missing, {extra} extra dates)")
    if list(signal.columns) != list(prices.columns):
        problems.append("columns differ from price columns")
    values = signal.to_numpy(dtype=float)
    if np.isinf(values).any():
        problems.append("infinite weights present")
    gross = signal.abs().sum(axis=1)
    max_seen = float(gross.max()) if len(gross) else 0.0
    if max_seen > max_gross + 1e-9:
        warnings.append(f"max gross exposure {max_seen:.2f} exceeds {max_gross:.2f}")
    nan_frac = float(signal.isna().to_numpy().mean()) if signal.size else 0.0
    details = {"max_gross_exposure": max_seen, "nan_fraction": nan_frac}
    if problems:
        return CheckResult("signal_alignment", "FAIL", "verified", "; ".join(problems + warnings), details)
    if warnings:
        return CheckResult("signal_alignment", "WARN", "verified", "; ".join(warnings), details)
    return CheckResult(
        "signal_alignment",
        "PASS",
        "verified",
        f"aligned with prices; max gross exposure {max_seen:.2f}; {nan_frac:.1%} NaN (treated as flat)",
        details,
    )


def check_execution_lag(execution_lag: int) -> CheckResult:
    """Make the execution timing assumption explicit and reject same-bar look-ahead."""
    if execution_lag <= 0:
        return CheckResult(
            "execution_lag",
            "FAIL",
            "verified",
            "lag 0: positions earn the same bar's return that produced the signal (look-ahead)",
            {"execution_lag": execution_lag},
        )
    if execution_lag == 1:
        return CheckResult(
            "execution_lag",
            "WARN",
            "verified",
            "lag 1: assumes you observe the close and trade at that same close; "
            "compare with lag 2 to see how much the result depends on this",
            {"execution_lag": execution_lag},
        )
    return CheckResult(
        "execution_lag",
        "PASS",
        "verified",
        f"lag {execution_lag}: trades occur {execution_lag - 1} bar(s) after the signal close",
        {"execution_lag": execution_lag},
    )


def _frames_equal(a: pd.DataFrame, b: pd.DataFrame, atol: float) -> tuple[bool, float]:
    a_vals = a.to_numpy(dtype=float)
    b_vals = b.to_numpy(dtype=float)
    nan_mismatch = np.isnan(a_vals) != np.isnan(b_vals)
    both = ~np.isnan(a_vals) & ~np.isnan(b_vals)
    max_diff = float(np.max(np.abs(a_vals[both] - b_vals[both]))) if both.any() else 0.0
    return (not nan_mismatch.any()) and max_diff <= atol, max_diff


def check_determinism(signal_fn: SignalFn, prices: pd.DataFrame, atol: float = 1e-12) -> CheckResult:
    """Calling the signal function twice on the same data gives the same output."""
    first, second = signal_fn(prices.copy()), signal_fn(prices.copy())
    same, diff = _frames_equal(first, second, atol)
    if same:
        return CheckResult("determinism", "PASS", "verified", "two calls on identical data agree")
    return CheckResult(
        "determinism",
        "FAIL",
        "verified",
        f"two calls on identical data differ (max abs diff {diff:.3g}); seed any randomness",
        {"max_abs_diff": diff},
    )


def check_truncation_invariance(
    signal_fn: SignalFn,
    prices: pd.DataFrame,
    n_cutoffs: int = 5,
    min_history: int = 100,
    atol: float = 1e-10,
    warmup_cutoffs: Sequence[int] = (5, 10, 25, 50),
) -> CheckResult:
    """Detect signals whose past values depend on future price rows.

    The signal is computed on the full history and again on histories truncated
    at several cutoff dates. For a causal signal, every value at or before a
    cutoff must be identical in both runs. Any difference proves the function
    reads rows after the cutoff.

    Both late cutoffs (from ``min_history`` onward) and a few early ones are
    tested. The early cutoffs catch leaks confined to the warm-up period, such
    as ``.bfill()`` after a rolling window, which are invisible if every cutoff
    falls after the warm-up has been populated.

    This catches common leaks such as ``shift(-1)``, centered rolling windows,
    full-sample normalization, and warm-up back-fill. It cannot catch leaks that
    come from outside ``prices`` (other data sources, globals, parameters chosen
    with hindsight).
    """
    n = len(prices)
    if n < min_history + 2:
        return CheckResult("truncation_invariance", "INFO", "verified", f"not enough rows (need > {min_history + 1})")
    full = signal_fn(prices)
    late = np.unique(np.linspace(min_history, n - 2, n_cutoffs).astype(int))
    early = np.array([c for c in warmup_cutoffs if 0 <= c <= n - 2 and c < min_history], dtype=int)
    cut_positions = np.unique(np.concatenate([early, late]))
    failures = []
    worst = 0.0
    for pos in cut_positions:
        truncated_prices = prices.iloc[: pos + 1]
        partial = signal_fn(truncated_prices)
        common_idx = truncated_prices.index
        ok, diff = _frames_equal(
            full.reindex(index=common_idx, columns=prices.columns),
            partial.reindex(index=common_idx, columns=prices.columns),
            atol,
        )
        worst = max(worst, diff)
        if not ok:
            failures.append(str(prices.index[pos].date()) if hasattr(prices.index[pos], "date") else str(pos))
    details = {"cutoffs_tested": len(cut_positions), "failed_cutoffs": failures, "max_abs_diff": worst}
    if failures:
        return CheckResult(
            "truncation_invariance",
            "FAIL",
            "verified",
            f"past signal values changed when future rows were removed at {len(failures)}/"
            f"{len(cut_positions)} cutoffs (max diff {worst:.3g}): the signal uses future data",
            details,
        )
    return CheckResult(
        "truncation_invariance",
        "PASS",
        "verified",
        f"past values unchanged at {len(cut_positions)} truncation cutoffs "
        "(rules out leaks from future price rows only)",
        details,
    )


# --------------------------------------------------------------------------
# Illustrative checks: sensitivities that reveal fragility
# --------------------------------------------------------------------------


def check_cost_sensitivity(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    execution_lag: int = 1,
    assumed_cost_bps: float = 10.0,
) -> CheckResult:
    """Break-even linear cost versus a cost you consider realistic."""
    res = run_backtest(prices, signal, execution_lag=execution_lag, cost_bps=0.0)
    be = breakeven_cost_bps(res)
    turnover = float(res.turnover.mean())
    details = {
        "breakeven_cost_bps": be,
        "avg_daily_turnover": turnover,
        "annualized_turnover": turnover * 252,
        "assumed_cost_bps": assumed_cost_bps,
    }
    if np.isinf(be):
        return CheckResult("cost_sensitivity", "INFO", "illustrative", "strategy never trades", details)
    if np.isnan(be):
        return CheckResult(
            "cost_sensitivity", "WARN", "illustrative", "no positive gross edge even before costs", details
        )
    msg = (
        f"break-even cost {be:.1f} bps per unit turnover "
        f"(avg turnover {turnover:.2f}x gross per day, ~{turnover * 252:.0f}x per year)"
    )
    if be < assumed_cost_bps:
        return CheckResult(
            "cost_sensitivity",
            "WARN",
            "illustrative",
            msg + f"; below your {assumed_cost_bps:g} bps assumption",
            details,
        )
    return CheckResult("cost_sensitivity", "PASS", "illustrative", msg, details)


def check_lag_sensitivity(
    prices: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float = 0.0,
    max_relative_drop: float = 0.5,
) -> CheckResult:
    """How much of the Sharpe ratio survives one extra bar of execution delay."""
    s1 = metrics.sharpe_ratio(run_backtest(prices, signal, 1, cost_bps).net_returns)
    s2 = metrics.sharpe_ratio(run_backtest(prices, signal, 2, cost_bps).net_returns)
    details = {"sharpe_lag1": s1, "sharpe_lag2": s2}
    msg = f"net Sharpe {s1:.2f} at lag 1 vs {s2:.2f} at lag 2"
    if s1 > 0 and (s2 < s1 * (1 - max_relative_drop)):
        return CheckResult(
            "lag_sensitivity",
            "WARN",
            "illustrative",
            msg + f"; more than {max_relative_drop:.0%} of the edge disappears with one bar of delay",
            details,
        )
    return CheckResult("lag_sensitivity", "PASS" if s1 > 0 else "INFO", "illustrative", msg, details)


def selection_bias_on_noise(
    n_trials: int = 50,
    n_days: int = 2520,
    n_assets: int = 8,
    seed: int = 0,
    lookbacks: Sequence[int] | None = None,
) -> CheckResult:
    """Best-of-N Sharpe when every candidate strategy is run on pure noise.

    Synthetic prices are generated with zero drift and no predictability, so
    every true Sharpe is ~0. Trying many lookbacks and keeping the best one
    still produces an impressive-looking number. Use it as a reference point
    for how much "edge" you should expect from search alone.
    """
    from .data import generate_prices
    from .signals import cross_sectional_momentum

    prices = generate_prices(n_days=n_days, n_assets=n_assets, seed=seed, annual_drift=0.0, autocorr=0.0)
    if lookbacks is None:
        lookbacks = list(range(1, n_trials + 1))
    lookbacks = list(lookbacks)[:n_trials]
    sharpes = [
        metrics.sharpe_ratio(run_backtest(prices, cross_sectional_momentum(prices, lb), 1).net_returns)
        for lb in lookbacks
    ]
    best = float(np.max(sharpes))
    details = {"n_trials": len(sharpes), "best_sharpe": best, "median_sharpe": float(np.median(sharpes))}
    return CheckResult(
        "selection_bias_on_noise",
        "INFO",
        "illustrative",
        f"best of {len(sharpes)} momentum lookbacks on pure noise has Sharpe {best:.2f} "
        f"(median {np.median(sharpes):.2f}); a searched-for result must clear this bar",
        details,
    )
