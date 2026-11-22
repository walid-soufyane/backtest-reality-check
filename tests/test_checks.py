import numpy as np
import pandas as pd
import pytest

from backtest_reality_check import (
    check_cost_sensitivity,
    check_determinism,
    check_execution_lag,
    check_lag_sensitivity,
    check_price_data,
    check_signal_alignment,
    check_truncation_invariance,
    generate_prices,
    reality_check,
    selection_bias_on_noise,
)
from backtest_reality_check.signals import HONEST_SIGNALS, LEAKY_SIGNALS


@pytest.fixture(scope="module")
def prices():
    return generate_prices(n_days=600, seed=11, autocorr=-0.05)


@pytest.mark.parametrize("name", sorted(HONEST_SIGNALS))
def test_honest_signals_pass_truncation(prices, name):
    assert check_truncation_invariance(HONEST_SIGNALS[name], prices).status == "PASS"


@pytest.mark.parametrize("name", sorted(LEAKY_SIGNALS))
def test_leaky_signals_fail_truncation(prices, name):
    res = check_truncation_invariance(LEAKY_SIGNALS[name], prices)
    assert res.status == "FAIL"
    assert res.scope == "verified"
    assert res.details["failed_cutoffs"]


def test_truncation_catches_warmup_bfill_leak(prices):
    """A bfill on the warm-up rows borrows a later value; early cutoffs must expose it."""

    def bfill_leak(p: pd.DataFrame) -> pd.DataFrame:
        return p.pct_change(20, fill_method=None).bfill()

    res = check_truncation_invariance(bfill_leak, prices)
    assert res.status == "FAIL"
    assert res.scope == "verified"
    assert res.details["failed_cutoffs"]


def test_truncation_needs_enough_history():
    short = generate_prices(n_days=50)
    assert check_truncation_invariance(HONEST_SIGNALS["cross_sectional_momentum"], short).status == "INFO"


def test_execution_lag_statuses():
    assert check_execution_lag(0).status == "FAIL"
    assert check_execution_lag(1).status == "WARN"
    assert check_execution_lag(2).status == "PASS"


def test_price_data_problems_detected(prices):
    assert check_price_data(prices).status == "PASS"
    bad = prices.copy()
    bad.iloc[3, 0] = np.nan
    bad.iloc[4, 1] = -1.0
    res = check_price_data(bad.iloc[::-1])
    assert res.status == "FAIL"
    assert any("missing" in p for p in res.details["problems"])
    assert any("non-positive" in p for p in res.details["problems"])
    assert any("sorted" in p for p in res.details["problems"])


def test_signal_alignment(prices):
    good = HONEST_SIGNALS["cross_sectional_momentum"](prices)
    assert check_signal_alignment(prices, good).status == "PASS"
    assert check_signal_alignment(prices, good * 3).status == "WARN"
    assert check_signal_alignment(prices, good.iloc[:-10]).status == "FAIL"


def test_determinism_catches_unseeded_randomness(prices):
    def noisy(p: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(np.random.default_rng().normal(size=p.shape), index=p.index, columns=p.columns)

    assert check_determinism(noisy, prices).status == "FAIL"
    assert check_determinism(HONEST_SIGNALS["time_series_momentum"], prices).status == "PASS"


def test_cost_sensitivity_flags_high_turnover(prices):
    signal = HONEST_SIGNALS["short_term_reversal"](prices)
    lenient = check_cost_sensitivity(prices, signal, assumed_cost_bps=0.01)
    strict = check_cost_sensitivity(prices, signal, assumed_cost_bps=1000)
    assert lenient.status == "PASS" and strict.status == "WARN"
    assert lenient.scope == "illustrative"
    assert lenient.details["breakeven_cost_bps"] > 0


def test_lag_sensitivity_flags_leaky_signal(prices):
    signal = LEAKY_SIGNALS["leaky_next_day_return"](prices)
    res = check_lag_sensitivity(prices, signal)
    assert res.status == "WARN"
    assert res.details["sharpe_lag1"] > res.details["sharpe_lag2"]


def test_selection_bias_best_exceeds_median():
    res = selection_bias_on_noise(n_trials=10, n_days=500, seed=1)
    assert res.status == "INFO" and res.scope == "illustrative"
    assert res.details["best_sharpe"] >= res.details["median_sharpe"]


def test_full_report(prices):
    honest = reality_check(prices, HONEST_SIGNALS["short_term_reversal"])
    leaky = reality_check(prices, LEAKY_SIGNALS["leaky_next_day_return"])
    assert not honest.has_failures
    assert leaky.has_failures
    text = leaky.to_text()
    assert "truncation_invariance" in text and "Not investment advice" in text
    assert len(honest.checks) == 7
