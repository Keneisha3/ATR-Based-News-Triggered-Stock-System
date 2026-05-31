"""ML layer: causal feature construction, no-leakage walk-forward, honest metrics.

Synthetic / offline. We test the *machinery* (features built without look-ahead,
walk-forward produces pooled OOS metrics and a rule baseline, importance ranks
features) rather than asserting a particular AUC on random data.
"""

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import ml, config


@pytest.fixture(autouse=True)
def _cfg(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "REGIME_MA_WINDOW", 15)


def _prices(years=(2021, 2022, 2023, 2024), per_year=120, seed=7):
    rng = np.random.default_rng(seed)
    frames = []
    for t in ("A", "B", "C"):
        dates, close_all, move_all, vol_all = [], [], [], []
        price = 100.0
        for y in years:
            d = pd.bdate_range(f"{y}-01-01", periods=per_year)
            rets = rng.normal(0.0007, 0.02, per_year)
            close = price * np.cumprod(1 + rets); price = close[-1]
            atr = pd.Series(close).rolling(14).std().bfill().to_numpy() + 1e-6
            move = np.concatenate([[0], np.diff(close)]) / atr
            dates += list(d); close_all += list(close); move_all += list(move)
            vol_all += list(rng.integers(1_000_000, 5_000_000, per_year))
        frames.append(pd.DataFrame({"ticker": t, "date": dates, "close": close_all,
                                    "volume": vol_all, "atr_pct": 0.02,
                                    "move_in_atr": move_all}))
    return pd.concat(frames, ignore_index=True)


def test_build_dataset_has_features_and_label():
    data = ml.build_dataset(_prices(), horizon=5)
    assert not data.empty
    for f in ml.FEATURES:
        assert f in data.columns
    assert set(data["continued"].unique()) <= {0, 1}
    assert data[ml.FEATURES].isna().sum().sum() == 0      # no NaNs leak through


def test_walk_forward_produces_oos_metrics_and_baseline():
    res = ml.run_ml(_prices(), horizon=5)
    assert res["ok"]
    assert "gradient_boosting" in res["models"] and "logistic" in res["models"]
    for m in res["models"].values():
        assert 0.0 <= m["auc"] <= 1.0
        assert 0.0 <= m["accuracy"] <= 1.0
    assert "rule_baseline" in res          # honest benchmark present
    assert res["n_oos"] > 0


def test_feature_importance_ranks_all_features():
    imp = ml.feature_importance(ml.build_dataset(_prices(), horizon=5))
    assert set(imp["feature"]) == set(ml.FEATURES)
    assert imp.iloc[0]["importance"] >= imp.iloc[-1]["importance"]   # sorted desc


def test_summary_mentions_verdict():
    out = ml.format_summary(ml.run_ml(_prices(), horizon=5))
    assert "Verdict" in out and "regime rule" in out


def test_insufficient_history():
    one_year = _prices(years=(2021,))
    res = ml.run_ml(one_year, horizon=5)
    assert res["ok"] is False
