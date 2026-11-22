# Launch notes

Ready-to-share text for introducing the project. Everything below describes what v0.1 actually does. Please keep it that way when you edit it.

## Short version (≈280 characters)

> I built Backtest Reality Check, a small open-source Python toolkit that shows how backtests mislead: look-ahead leaks, same-close execution, and costs that wipe out an edge. Runs offline on deterministic synthetic data. Beginner-friendly issues are open: https://github.com/Walid-Soufyane/backtest-reality-check

## Longer version

> **Backtest Reality Check** is an educational, MIT-licensed Python toolkit for spotting common ways a quantitative backtest can look better (or just different) than it should.
>
> It ships with a deterministic synthetic market, so anyone can run it with `pip install -e .` and `brc demo`. No API keys or paid data are needed. In the demo, a short-term reversal strategy has a gross Sharpe of 1.83 that turns negative at 5 bps of costs, and a deliberately leaky signal shows a Sharpe above 50 that the truncation check flags as using future data.
>
> Checks are labelled honestly. **Verified** checks test concrete properties of your inputs, for example whether a signal's past values change when future rows are deleted. **Illustrative** checks measure fragility: break-even costs, sensitivity to an extra bar of delay, and the best-of-N Sharpe on pure noise. It does not claim to catch every form of look-ahead bias, and it is not investment advice or a trading system.
>
> It's v0.1 and deliberately small. I've opened a set of `good first issue` tickets with context and acceptance criteria (new metrics, a warm-up leak the current check misses, a CSV loader, report export, plotting, and more). If you're learning quant development, or you've been bitten by a backtest pitfall I haven't covered, contributions and issue reports are very welcome.

## Contribution invitation (for a README, forum post, or class channel)

> Looking for a first open-source contribution in Python or quant finance? Backtest Reality Check has small, well-scoped issues labelled `good first issue`. Each has acceptance criteria, and the test suite runs offline in about a second. Comment on an issue to claim it, and feel free to ask questions there.

## Notes

- No claims about adoption, stars, or performance on real markets should be added unless they are true at the time of posting.
- Demo figures above come from `brc demo` with default arguments (seed 42). Re-run it before posting if the code has changed.
