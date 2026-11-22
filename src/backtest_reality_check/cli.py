"""Command-line demo: ``brc demo`` or ``python -m backtest_reality_check demo``."""

from __future__ import annotations

import argparse
import sys

from .checks import selection_bias_on_noise
from .data import generate_prices
from .report import reality_check
from .signals import HONEST_SIGNALS, LEAKY_SIGNALS


def run_demo(seed: int = 42, n_days: int = 2520, cost_bps: float = 5.0, autocorr: float = -0.05) -> int:
    """Run honest and leaky strategies on the same synthetic market and print reports."""
    prices = generate_prices(seed=seed, n_days=n_days, autocorr=autocorr)
    print(
        f"Synthetic market: {prices.shape[1]} assets, {len(prices)} business days, seed={seed}, "
        f"planted idiosyncratic autocorrelation={autocorr:+.2f}\n"
    )
    strategies = {
        "short_term_reversal": HONEST_SIGNALS["short_term_reversal"],
        "leaky_next_day_return": LEAKY_SIGNALS["leaky_next_day_return"],
        "leaky_full_sample_zscore": LEAKY_SIGNALS["leaky_full_sample_zscore"],
    }
    for name, fn in strategies.items():
        print(reality_check(prices, fn, strategy_name=name, cost_bps=cost_bps).to_text())
        print()
    print("Selection-bias reference point:")
    print(f"  {selection_bias_on_noise(n_trials=30, n_days=n_days, seed=seed)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="brc", description="Backtest Reality Check")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the synthetic-data demo")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--days", type=int, default=2520)
    demo.add_argument("--cost-bps", type=float, default=5.0)
    demo.add_argument("--autocorr", type=float, default=-0.05, help="planted AR(1) coefficient")
    args = parser.parse_args(argv)
    if args.command == "demo":
        return run_demo(seed=args.seed, n_days=args.days, cost_bps=args.cost_bps, autocorr=args.autocorr)
    return 1  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
