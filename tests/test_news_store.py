"""Historical news store: append/dedup, coverage, and the news-vs-breach A/B test.

Offline. The archive path is redirected to a tmp file so tests never touch real
data. The key behavioral guarantee: the A/B test refuses to answer for dates the
archive does not cover, instead of inventing a result.
"""

import pandas as pd
import pytest

from atr_news_alert import news_store, config


@pytest.fixture(autouse=True)
def _tmp_archive(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "NEWS_ARCHIVE_CSV", tmp_path / "news_archive.csv")


def _news(ids, ticker="X", published="2024-01-02 09:00"):
    return pd.DataFrame({
        "id": ids,
        "ticker": ticker,
        "title": [f"headline {i}" for i in ids],
        "published": pd.to_datetime(published),
        "sentiment": "bullish",
        "signal": 1.0,
        "sentiment_source": "lexicon",
    })


def test_append_dedups_by_id():
    assert news_store.append_news(_news(["a", "b"]), today="2024-01-02") == 2
    # "b" already stored; only "c" is new.
    assert news_store.append_news(_news(["b", "c"]), today="2024-01-03") == 1
    assert len(news_store.load_archive()) == 3


def test_append_empty_is_noop():
    assert news_store.append_news(pd.DataFrame()) == 0
    assert news_store.load_archive().empty


def test_coverage_reports_span():
    news_store.append_news(_news(["a"]), today="2024-01-02")
    news_store.append_news(_news(["b"]), today="2024-01-05")
    cov = news_store.coverage()
    assert cov["rows"] == 2
    assert cov["days"] == 2
    assert cov["start"] == "2024-01-02"
    assert cov["end"] == "2024-01-05"


def _prices_with_breach(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "FORWARD_HORIZONS", [1])
    # Breach on 2024-01-02 (index1), price continues up next day.
    return pd.DataFrame({
        "ticker": ["X"] * 3,
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        "close": [100.0, 100.0, 110.0],
        "move_in_atr": [0.0, 2.0, 0.0],
    })


def test_ab_test_not_ready_with_empty_archive(monkeypatch):
    prices = _prices_with_breach(monkeypatch)
    res = news_store.compare_news_vs_breach(prices)
    assert res["ready"] is False
    assert "too thin" in res["note"]


def test_ab_test_runs_once_archive_covers_breach(monkeypatch):
    prices = _prices_with_breach(monkeypatch)
    # Archive a headline for X dated at the breach day, recorded that day.
    news_store.append_news(
        _news(["n1"], ticker="X", published="2024-01-02 09:00"),
        today="2024-01-02")
    res = news_store.compare_news_vs_breach(prices)
    assert res["ready"] is True
    assert res["buckets"]["breach+news"]["events"] == 1
    assert res["buckets"]["breach-only"]["events"] == 0


def test_breach_outside_archive_window_is_excluded(monkeypatch):
    prices = _prices_with_breach(monkeypatch)
    # Archive only covers a much later date -> the breach is not covered.
    news_store.append_news(_news(["n1"]), today="2025-06-01")
    tagged = news_store.tag_breaches_with_news(prices)
    assert tagged.empty
