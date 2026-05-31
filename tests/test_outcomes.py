"""Feedback loop: record predictions, resolve realized returns, calibrate."""

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import outcomes, config


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "OUTCOMES_CSV", tmp_path / "results" / "outcomes.csv")
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)


def _decisions():
    return [
        {"date": "2024-01-02", "ticker": "X", "signal": "BREACH_UP", "regime": "BULL",
         "action": "LONG", "confidence": 0.6, "has_news": True},
        {"date": "2024-01-02", "ticker": "Y", "signal": "BREACH_DOWN", "regime": "BULL",
         "action": "IGNORE", "confidence": 0.6, "has_news": False},
    ]


def _prices():
    # X up-breach on 1/2 then rises; Y down-breach on 1/2 then also rises (so the
    # bearish breach does NOT continue -> realized breach-direction return < 0).
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    x = pd.DataFrame({"ticker": "X", "date": dates,
                      "close": [100, 100, 102, 104, 106, 108, 110, 112.0],
                      "move_in_atr": [0, 2.0, 0, 0, 0, 0, 0, 0]})
    y = pd.DataFrame({"ticker": "Y", "date": dates,
                      "close": [100, 100, 101, 103, 105, 107, 109, 111.0],
                      "move_in_atr": [0, -2.0, 0, 0, 0, 0, 0, 0]})
    return pd.concat([x, y], ignore_index=True)


def test_record_dedups_by_date_ticker():
    assert outcomes.record(_decisions()) == 2
    assert outcomes.record(_decisions()) == 0     # same keys -> no new rows
    assert len(outcomes._load()) == 2


def test_update_fills_realized_and_resolves():
    outcomes.record(_decisions())
    log = outcomes.update(_prices())
    x = log[log["ticker"] == "X"].iloc[0]
    assert x["resolved"]
    # X: +2% by day1, breach-direction positive -> correct.
    assert x["realized_1d"] > 0 and x["correct"]
    y = log[log["ticker"] == "Y"].iloc[0]
    # Y: bearish breach but price rose -> breach-direction return negative.
    assert y["realized_1d"] < 0 and not y["correct"]


def test_calibration_reports_hit_rate_and_long_stats():
    outcomes.record(_decisions())
    outcomes.update(_prices())
    cal = outcomes.calibration()
    assert cal["resolved"] == 2
    assert cal["overall_hit_3d_pct"] == 50.0        # X correct, Y not
    assert cal["long_n"] == 1 and cal["long_hit_3d_pct"] == 100.0


def test_backfill_populates_from_history():
    n = outcomes.backfill(_prices())
    assert n == 2                                    # two historical breaches
    cal = outcomes.calibration()
    assert cal["resolved"] == 2


def test_format_calibration_empty():
    assert "No resolved decisions" in outcomes.format_calibration({"resolved": 0})
