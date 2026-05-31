"""Story layer: the report assembles a complete, well-formed research memo."""

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import report, config


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "SWEEP_HOLD_DAYS", [1, 2])
    monkeypatch.setattr(config, "SWEEP_COST_BPS", [0.0, 5.0])


def _prices(n=150, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    out = []
    for t in ("A", "B", "C"):
        close = 100 * np.cumprod(1 + rng.normal(0.0008, 0.02, n))
        atr = pd.Series(close).rolling(14).std().bfill().to_numpy() + 1e-6
        move = np.concatenate([[0], np.diff(close)]) / atr
        out.append(pd.DataFrame({"ticker": t, "date": dates, "close": close,
                                 "move_in_atr": move}))
    return pd.concat(out, ignore_index=True)


def test_report_has_all_sections_without_charts():
    md = report.generate(_prices(), with_charts=False)
    for header in ("# ATR Breach Signal", "## 1. Executive summary",
                   "## 2. Today's actions", "## 3. Signal behaviour",
                   "## 4. Performance", "## 5. Robustness",
                   "## 6. Execution quality", "## 7. Data provenance"):
        assert header in md


def test_write_report_creates_markdown_and_artifacts():
    path = report.write_report(_prices(), with_charts=False)
    assert path.endswith("report.md")
    assert (config.RESULTS_DIR / "report.md").exists()
    # Snapshots written as a side effect (audit trail).
    assert (config.RESULTS_DIR / "run_manifest.json").exists()
    assert (config.RESULTS_DIR / "event_study.csv").exists()


def test_report_with_charts_embeds_images():
    md = report.write_report(_prices(), with_charts=True)
    text = (config.RESULTS_DIR / "report.md").read_text()
    assert "![equity curve](equity_curve.png)" in text
    assert (config.RESULTS_DIR / "equity_curve.png").exists()
    assert (config.RESULTS_DIR / "event_study_curve.png").exists()


def test_as_of_report_stamps_the_cutoff_date():
    # An historical memo should date-stamp "as of" the cutoff, not "now".
    md = report.generate(_prices(), with_charts=False, on=pd.Timestamp("2024-03-31").date())
    assert "as of 2024-03-31" in md


def test_empty_prices_report():
    assert "No price history" in report.generate(pd.DataFrame())
