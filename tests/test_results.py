"""Reproducibility layer: manifest + result snapshots.

Offline. RESULTS_DIR is redirected to a tmp dir so tests never write into the
real repo. Verifies the manifest captures provenance/params/metrics and that the
equity/trades/positions snapshots are written and well-formed.
"""

import json

import pandas as pd
import pytest

from atr_news_alert import results as R, portfolio as pf, config


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")


def _metrics(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    prices = pd.DataFrame({
        "ticker": "X", "date": dates,
        "close": [100.0, 100.0, 110.0, 110.0, 99.0, 99.0],
        "move_in_atr": [0.0, 2.0, 0.0, -2.0, 0.0, 0.0],
    })
    return pf.simulate(prices, hold=1, cost_bps=5), prices


def test_manifest_captures_provenance_and_params(monkeypatch):
    m, prices = _metrics(monkeypatch)
    man = R.build_manifest(m, prices, command="portfolio")
    assert man["command"] == "portfolio"
    assert man["data"]["tickers"] == 1
    assert man["data"]["start"] == "2024-01-01"
    assert man["params"]["hold_days"] == 1
    assert man["params"]["cost_bps"] == 5.0
    assert "total_return_pct" in man["metrics"]
    assert "timestamp_utc" in man
    assert "git_hash" in man          # may be None outside a git checkout


def test_snapshot_writes_all_artifacts(monkeypatch):
    m, prices = _metrics(monkeypatch)
    written = R.snapshot_portfolio(m, prices)
    for key in ("manifest", "equity_curve", "trades", "positions"):
        assert written[key].exists()

    # Manifest is valid JSON with the expected top-level shape.
    man = json.loads(written["manifest"].read_text())
    assert set(man) >= {"command", "timestamp_utc", "data", "params", "metrics"}

    # Equity curve has aligned strategy + benchmark + drawdown columns.
    curve = pd.read_csv(written["equity_curve"])
    assert set(curve.columns) >= {"date", "strategy_equity",
                                  "benchmark_equity", "strategy_drawdown"}
    assert len(curve) == m["days"]
    assert (curve["strategy_drawdown"] <= 1e-9).all()   # drawdown is <= 0

    trades = pd.read_csv(written["trades"])
    assert len(trades) == 2                              # one long, one short


def test_snapshot_sweep_writes_grid_and_manifest(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame({
        "ticker": "X", "date": dates,
        "close": [100, 100, 110, 108, 112, 110, 115, 113.0],
        "move_in_atr": [0, 2.0, 0, -2.0, 0, 2.0, 0, 0],
    })
    grid, summary = pf.sweep(prices, holds=[1, 2], costs=[0.0, 5.0])
    written = R.snapshot_sweep(grid, summary, prices)
    assert written["sweep"].exists() and written["manifest"].exists()
    man = json.loads(written["manifest"].read_text())
    assert man["command"] == "sweep"
    assert "robustness" in man
    assert man["params"]["hold_grid"] == [1, 2]
