"""Decision schema: stable contract, long-biased action rules, persistence."""

import json

import pytest

from atr_news_alert import decision as dec, config


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")


def _res(continuation=True, regime="bull", hit=62.0, signals=None):
    return {
        "regime": regime, "horizon": 3, "drift_pct": 0.33, "continuation": continuation,
        "regime_hit_rate": hit,
        "signals": signals if signals is not None else [
            {"ticker": "AAPL", "breach": "bullish", "move_in_atr": 2.5,
             "expected_move_pct": 0.33, "has_news": True},
            {"ticker": "XOM", "breach": "bearish", "move_in_atr": -2.0,
             "expected_move_pct": -0.33, "has_news": False},
        ],
    }


def test_long_only_in_continuation_up():
    d = dec.build_decisions(_res(continuation=True))
    by_ticker = {x["ticker"]: x for x in d}
    assert by_ticker["AAPL"]["action"] == "LONG"      # up-breach + momentum
    assert by_ticker["XOM"]["action"] == "IGNORE"     # down-breach, no shorting


def test_reversion_regime_holds_up_breaches():
    d = dec.build_decisions(_res(continuation=False))
    by_ticker = {x["ticker"]: x for x in d}
    assert by_ticker["AAPL"]["action"] == "HOLD"      # reversion risk on the pop
    assert by_ticker["XOM"]["action"] == "IGNORE"


def test_schema_fields_and_news_confidence_bump():
    d = dec.build_decisions(_res(hit=60.0))
    rec = next(x for x in d if x["ticker"] == "AAPL")
    assert rec["schema_version"] == dec.SCHEMA_VERSION
    assert rec["signal"] == "BREACH_UP" and rec["regime"] == "BULL"
    assert "expected_3d_return_pct" in rec
    assert rec["confidence"] == 0.65                  # 0.60 + 0.05 news bump
    assert "news-confirmed" in rec["rationale"]
    no_news = next(x for x in d if x["ticker"] == "XOM")
    assert no_news["confidence"] == 0.60              # no bump


def test_actions_are_in_vocabulary_and_sorted():
    d = dec.build_decisions(_res())
    assert all(x["action"] in dec.ACTIONS for x in d)
    # LONG sorts before IGNORE.
    assert d[0]["action"] == "LONG"


def test_summary_counts():
    s = dec.summary(dec.build_decisions(_res()))
    assert s["LONG"] == 1 and s["IGNORE"] == 1 and s["HOLD"] == 0


def test_write_decisions_is_valid_json():
    path = dec.write_decisions(dec.build_decisions(_res()))
    payload = json.loads(open(path).read())
    assert payload["schema_version"] == dec.SCHEMA_VERSION
    assert "summary" in payload and len(payload["decisions"]) == 2


def test_no_signals_yields_empty():
    assert dec.build_decisions(_res(signals=[])) == []
    assert "No decisions" in dec.format_decisions([])
