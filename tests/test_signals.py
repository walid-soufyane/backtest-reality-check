from backtest_reality_check import check_truncation_invariance, generate_prices, run_backtest
from backtest_reality_check.metrics import sharpe_ratio
from backtest_reality_check.signals import leaky_calendar_group_mean


def test_calendar_group_mean_is_caught_and_looks_too_good():
    prices = generate_prices(n_days=600, seed=3)
    res = check_truncation_invariance(leaky_calendar_group_mean, prices)
    assert res.status == "FAIL"
    # Knowing the rest of the month's average return makes a pure-noise market look tradeable.
    sharpe = sharpe_ratio(run_backtest(prices, leaky_calendar_group_mean(prices)).net_returns)
    assert sharpe > 2
