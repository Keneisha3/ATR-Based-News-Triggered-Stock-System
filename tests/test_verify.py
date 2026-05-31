"""Trust layer: the verification checks pass on clean data and catch corruption."""

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import verify, config


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):
    # Point RESULTS_DIR at a tmp dir so the manifest-cross-check is skipped cleanly.
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)


def _good_prices(n=120, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    out = []
    for t in ("A", "B"):
        ret = rng.normal(0.0005, 0.02, n)
        close = 100 * np.cumprod(1 + ret)
        atr = pd.Series(close).rolling(14).std().bfill().to_numpy() + 1e-6
        move_in_atr = np.concatenate([[0], np.diff(close)]) / atr
        out.append(pd.DataFrame({"ticker": t, "date": dates, "close": close,
                                 "move_in_atr": move_in_atr}))
    return pd.concat(out, ignore_index=True)


def test_all_checks_pass_on_clean_data():
    checks = verify.run_checks(_good_prices())
    names = {c["name"] for c in checks}
    assert {"data integrity", "return reconstruction", "no missing trading days",
            "no NaN propagation", "trade-position consistency",
            "determinism (idempotency)"} <= names
    assert verify.all_passed(checks), verify.format_report(checks)


def test_empty_prices_fails_cleanly():
    checks = verify.run_checks(pd.DataFrame())
    assert not verify.all_passed(checks)


def test_data_integrity_catches_duplicate_dates():
    p = _good_prices(60)
    # Corrupt: duplicate a (ticker, date) row to break monotonic-unique dates.
    p = pd.concat([p, p.iloc[[0]]], ignore_index=True)
    checks = verify.run_checks(p)
    integrity = next(c for c in checks if c["name"] == "data integrity")
    assert integrity["passed"] is False


def test_format_report_marks_pass_and_fail():
    out = verify.format_report([
        {"name": "x", "passed": True, "detail": ""},
        {"name": "y", "passed": False, "detail": "boom"},
    ])
    assert "✔ x" in out and "✖ y" in out and "boom" in out
