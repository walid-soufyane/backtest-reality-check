from backtest_reality_check.cli import main


def test_demo_runs_and_reports_leak(capsys):
    assert main(["demo", "--days", "400", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "leaky_next_day_return" in out
    assert "[FAIL] (verified) truncation_invariance" in out
    assert "selection_bias_on_noise" in out


def test_quickstart_example_runs():
    import runpy
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
    runpy.run_path(str(path), run_name="__main__")


def test_compare_strategies_example():
    import importlib.util
    from pathlib import Path

    from backtest_reality_check import generate_prices
    from backtest_reality_check.signals import HONEST_SIGNALS, LEAKY_SIGNALS

    path = Path(__file__).resolve().parents[1] / "examples" / "compare_strategies.py"
    spec = importlib.util.spec_from_file_location("compare_strategies", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    table = module.compare(generate_prices(n_days=400, seed=1))
    assert len(table) == len(HONEST_SIGNALS) + len(LEAKY_SIGNALS)
    assert set(table.loc[table["kind"] == "leaky", "truncation_check"]) == {"FAIL"}
    assert set(table.loc[table["kind"] == "honest", "truncation_check"]) == {"PASS"}
