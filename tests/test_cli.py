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
