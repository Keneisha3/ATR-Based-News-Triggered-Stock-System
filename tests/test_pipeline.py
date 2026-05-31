"""Canonical pipeline: one loop, injected deps, light/full execution boundaries."""

import json

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import pipeline, notify, config


@pytest.fixture(autouse=True)
def _tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    # Never fire a real desktop notification from the pipeline during tests.
    monkeypatch.setattr(notify, "send_macos_notification", lambda *a, **k: True)
    for k in ("ATR_SMTP_HOST", "ATR_NOTIFY_FROM", "ATR_NOTIFY_TO"):
        monkeypatch.delenv(k, raising=False)


def _prices(breach_today=True):
    n = 80
    rng = np.random.default_rng(0)
    close = list(100 * np.cumprod(1 + rng.normal(0.001, 0.015, n)))
    signs = np.zeros(n)
    signs[::9] = 2.0
    if breach_today:
        signs[-1] = 2.0
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({"ticker": "X", "date": dates, "close": close,
                         "move_in_atr": signs})


def test_light_run_persists_contract_and_manifest():
    prices = _prices()
    res = pipeline.run_daily(prices_loader=lambda: prices, do_scan=False,
                             mode=pipeline.LIGHT)
    assert res["mode"] == "light"
    assert (config.RESULTS_DIR / "decisions.json").exists()
    assert (config.RESULTS_DIR / "pipeline_manifest.json").exists()
    man = json.loads((config.RESULTS_DIR / "pipeline_manifest.json").read_text())
    assert man["mode"] == "light" and man["command"] == "pipeline"


def test_scan_fn_called_when_do_scan_true():
    calls = []
    pipeline.run_daily(prices_loader=_prices, scan_fn=lambda: calls.append(1),
                       do_scan=True, mode=pipeline.LIGHT)
    assert calls == [1]


def test_scan_skipped_when_do_scan_false():
    calls = []
    pipeline.run_daily(prices_loader=_prices, scan_fn=lambda: calls.append(1),
                       do_scan=False)
    assert calls == []


def test_scan_failure_is_swallowed():
    def boom():
        raise RuntimeError("network down")
    # Must not raise — falls back to cached prices.
    res = pipeline.run_daily(prices_loader=_prices, scan_fn=boom, do_scan=True)
    assert "decisions" in res


def test_full_mode_runs_report_overlay_light_does_not():
    seen = []
    # Light mode must NOT call the heavy report overlay (execution boundary).
    pipeline.run_daily(prices_loader=_prices, do_scan=False,
                       report_fn=lambda p: seen.append("report"), mode=pipeline.LIGHT)
    assert seen == []
    # Full mode runs it.
    pipeline.run_daily(prices_loader=_prices, do_scan=False,
                       report_fn=lambda p: seen.append("report"), mode=pipeline.FULL)
    assert seen == ["report"]


def test_delivery_can_be_disabled():
    res = pipeline.run_daily(prices_loader=_prices, do_scan=False, deliver=False)
    assert res["delivery"] == {}
