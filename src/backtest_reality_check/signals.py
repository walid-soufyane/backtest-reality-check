"""Example signal functions.

A *signal function* takes a price DataFrame and returns target weights with the
same index and columns. The value on row ``t`` must depend only on prices up to
and including row ``t``; :func:`backtest_reality_check.checks.check_truncation_invariance`
tests exactly that property.

The honest signals below are standard textbook strategies, included as
baselines, not as recommendations. The ``leaky_*`` signals are **deliberately
broken** so the checks have something to catch. Never use them for research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _normalize_gross(weights: pd.DataFrame, gross: float = 1.0) -> pd.DataFrame:
    """Scale each row so the sum of absolute weights equals ``gross``."""
    denom = weights.abs().sum(axis=1).replace(0.0, np.nan)
    return weights.div(denom, axis=0).mul(gross).fillna(0.0)


def cross_sectional_momentum(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Long recent winners, short recent losers; dollar neutral, gross exposure 1."""
    past = prices.pct_change(lookback, fill_method=None)
    demeaned = past.sub(past.mean(axis=1), axis=0)
    return _normalize_gross(demeaned)


def short_term_reversal(prices: pd.DataFrame, lookback: int = 1) -> pd.DataFrame:
    """Long recent losers, short recent winners (the mirror image of momentum)."""
    return -cross_sectional_momentum(prices, lookback=lookback)


def time_series_momentum(prices: pd.DataFrame, lookback: int = 60) -> pd.DataFrame:
    """Each asset long if its trailing return is positive, short otherwise, equal weight."""
    past = prices.pct_change(lookback, fill_method=None)
    return _normalize_gross(np.sign(past))


# --------------------------------------------------------------------------
# Deliberately leaky signals (educational counterexamples)
# --------------------------------------------------------------------------


def leaky_next_day_return(prices: pd.DataFrame) -> pd.DataFrame:
    """BROKEN ON PURPOSE: uses tomorrow's return. Obvious look-ahead."""
    future = prices.pct_change(fill_method=None).shift(-1)
    demeaned = future.sub(future.mean(axis=1), axis=0)
    return _normalize_gross(demeaned)


def leaky_centered_moving_average(prices: pd.DataFrame, window: int = 21) -> pd.DataFrame:
    """BROKEN ON PURPOSE: a centered rolling window peeks ``window // 2`` bars ahead.

    Centered smoothing is common in charting and signal processing, which is
    why it sneaks into research code.
    """
    smooth = prices.rolling(window, center=True).mean()
    trend = smooth.pct_change(fill_method=None)
    demeaned = trend.sub(trend.mean(axis=1), axis=0)
    return _normalize_gross(demeaned)


def leaky_full_sample_zscore(prices: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """BROKEN ON PURPOSE: z-scores each asset with full-sample mean and std.

    The trailing return itself is honest, but the normalization uses statistics
    from the entire history, including the future. This is a subtle and very
    common leak (e.g. ``StandardScaler().fit(all_data)``).
    """
    past = prices.pct_change(lookback, fill_method=None)
    z = (past - past.mean()) / past.std()
    return _normalize_gross(z.clip(-3, 3))


HONEST_SIGNALS = {
    "cross_sectional_momentum": cross_sectional_momentum,
    "short_term_reversal": short_term_reversal,
    "time_series_momentum": time_series_momentum,
}

LEAKY_SIGNALS = {
    "leaky_next_day_return": leaky_next_day_return,
    "leaky_centered_moving_average": leaky_centered_moving_average,
    "leaky_full_sample_zscore": leaky_full_sample_zscore,
}
