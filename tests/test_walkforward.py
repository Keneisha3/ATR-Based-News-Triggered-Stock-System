"""Walk-forward: expanding folds, train-only rule learning, non-leaky OOS metrics."""

import numpy as np
import pandas as pd
import pytest

from atr_news_alert import walkforward as wf, config


@pytest.fixture(autouse=True)
def _atr(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "REGIME_MA_WINDOW", 10)   # short MA so regimes form fast


def _multiyear_prices(years=(2021, 2022, 2023, 2024), per_year=70, seed=1):
    rng = np.random.default_rng(seed)
    frames = []
    for t in ("A", "B"):
        all_dates, all_close, all_move = [], [], []
        price = 100.0
        for y in years:
            dates = pd.bdate_range(f"{y}-01-01", periods=per_year)
            rets = rng.normal(0.0006, 0.02, per_year)
            close = price * np.cumprod(1 + rets)
            price = close[-1]
            atr = pd.Series(close).rolling(14).std().bfill().to_numpy() + 1e-6
            move = np.concatenate([[0], np.diff(close)]) / atr
            all_dates += list(dates); all_close += list(close); all_move += list(move)
        frames.append(pd.DataFrame({"ticker": t, "date": all_dates,
                                    "close": all_close, "move_in_atr": all_move}))
    return pd.concat(frames, ignore_index=True)


def test_folds_are_expanding_and_oos():
    res = wf.walk_forward(_multiyear_prices(), horizon=3)
    folds = res["folds"]
    assert len(folds) == 3                       # 4 years -> 3 expanding test folds
    assert [f["test"] for f in folds] == ["2022", "2023", "2024"]
    # Training window expands; first fold trains only on 2021.
    assert folds[0]["train"] == "2021-2021"
    assert folds[-1]["train"] == "2021-2023"


def test_pooled_metrics_present():
    res = wf.walk_forward(_multiyear_prices(), horizon=3)
    p = res["pooled"]
    assert p["n_events"] > 0
    assert 0.0 <= p["oos_hit_pct"] <= 100.0
    assert "oos_avg_ret_pct" in p and "info_ratio" in p


def test_rule_learned_per_regime():
    res = wf.walk_forward(_multiyear_prices(), horizon=3)
    # Each fold records a learned rule mapping regimes to cont/rev.
    for f in res["folds"]:
        assert all(v in ("cont", "rev") for v in f["rule"].values())


def test_insufficient_history_is_handled():
    one_year = _multiyear_prices(years=(2021,))
    res = wf.walk_forward(one_year, horizon=3, min_train_years=1)
    assert res.get("folds") == [] or "note" in res


def test_empty_prices():
    assert wf.walk_forward(pd.DataFrame()) == {}
    assert "Not enough" in wf.format_summary({})
