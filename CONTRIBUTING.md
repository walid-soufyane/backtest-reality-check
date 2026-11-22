# Contributing to Backtest Reality Check

Thanks for your interest! This project is meant to be approachable. If you've ever been fooled by a backtest, you already have something to contribute.

## Ways to help

- **Good first issues.** Look for the [`good first issue`](https://github.com/Walid-Soufyane/backtest-reality-check/labels/good%20first%20issue) label. Each one has context and acceptance criteria.
- **New counterexamples.** Add a deliberately leaky signal to `signals.py` along with a test showing which check catches it. If no check catches it, add a test with an explanation; that's useful too.
- **New checks.** Say whether the check is *verified* (it tests a mechanical property of the inputs) or *illustrative* (it measures a sensitivity), and document what it cannot detect.
- **Docs.** Clearer explanations, worked examples, and fixes to the README.

## Development setup

```bash
git clone https://github.com/<your-username>/backtest-reality-check.git
cd backtest-reality-check
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Before opening a pull request

1. `pytest` passes.
2. `ruff check .` and `ruff format --check .` pass. Run `ruff format .` to fix formatting.
3. New behaviour has tests. Tests must be deterministic: seed every random generator.
4. Public functions have docstrings that state their assumptions.
5. Claims stay honest. Don't describe a check as detecting something it only illustrates.

Keep pull requests focused. One idea per PR is easiest to review. If you're planning a larger change, open an issue first so we can agree on the approach.

## Design principles

- **No external data or services in the core package.** Everything must run offline, deterministically.
- **Explicit over clever.** Timing and cost assumptions should be visible in code and output.
- **Small dependency footprint.** NumPy and pandas only for the core. Optional extras need a good reason.
- **Educational scope.** This is not a trading system and must not be presented as one.

## Reporting bugs

Use the bug report template. Include a minimal snippet, the output you expected, and the output you got, plus your Python, NumPy and pandas versions.

## Code of Conduct

By participating you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).
