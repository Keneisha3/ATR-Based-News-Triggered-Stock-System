"""News lexicon, aggregation, and the LLM wiring (reproduces check #4)."""

import pandas as pd

from atr_news_alert import news as news_mod, llm


def test_score_headline():
    assert news_mod.score_headline("Apple beats and surges to record")[0] == "bullish"
    assert news_mod.score_headline("Company plunges on lawsuit and probe")[0] == "bearish"
    assert news_mod.score_headline("Apple holds annual meeting")[0] == "neutral"


def _news_df():
    now = pd.Timestamp.now()
    return pd.DataFrame([
        {"id": "1", "ticker": "AAA", "title": "AAA beats earnings", "link": "L1",
         "source": "x", "published": now, "is_recent": True, "sentiment": "bullish",
         "category": "Earnings", "confidence": 90, "word_score": 1, "signal": 0.9,
         "sentiment_source": "llm"},
        {"id": "2", "ticker": "AAA", "title": "AAA minor update", "link": "L2",
         "source": "x", "published": now, "is_recent": True, "sentiment": "neutral",
         "category": "Other", "confidence": 40, "word_score": 0, "signal": 0.0,
         "sentiment_source": "llm"},
        {"id": "3", "ticker": "BBB", "title": "old stale news", "link": "L3",
         "source": "x", "published": now, "is_recent": False, "sentiment": "bearish",
         "category": "Legal", "confidence": 70, "word_score": -1, "signal": -0.7,
         "sentiment_source": "llm"},
    ])


def test_summarize_only_recent_and_ranks_by_confidence():
    summ = news_mod.summarize_by_ticker(_news_df())
    # BBB's only article is stale -> excluded entirely.
    assert set(summ["ticker"]) == {"AAA"}
    row = summ.iloc[0]
    assert row["recent_count"] == 2
    assert row["headline_sentiment"] == "bullish"        # net signal 0.9 > 0
    assert row["top_category"] == "Earnings"             # highest-confidence pick
    assert row["top_headline"] == "AAA beats earnings"


def test_apply_llm_overrides_with_stub(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-dummy")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def fake_call(prompt):
        return ('[{"i":0,"sentiment":"bullish","category":"Earnings","confidence":88},'
                '{"i":1,"sentiment":"bearish","category":"Legal","confidence":73}]')
    monkeypatch.setattr(llm, "_call_anthropic", fake_call)

    now = pd.Timestamp.now()
    base = {"link": "", "source": "x", "published": now, "is_recent": True,
            "sentiment": "neutral", "category": "", "confidence": 0,
            "word_score": 0, "signal": 0.0, "sentiment_source": "lexicon"}
    df = pd.DataFrame([
        {**base, "id": "a", "ticker": "AAPL", "title": "Apple beats earnings"},
        {**base, "id": "b", "ticker": "AAPL", "title": "Apple faces lawsuit"},
    ])
    out = news_mod._apply_llm(df)
    assert out.loc[0, "sentiment"] == "bullish"
    assert out.loc[0, "category"] == "Earnings"
    assert out.loc[0, "signal"] == pytest_approx(0.88)
    assert out.loc[1, "sentiment"] == "bearish"
    assert out.loc[1, "signal"] == pytest_approx(-0.73)
    assert (out["sentiment_source"] == "llm").all()


def test_apply_llm_noop_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert not llm.available()
    df = pd.DataFrame([{
        "id": "a", "ticker": "AAPL", "title": "Apple beats earnings", "link": "",
        "source": "x", "published": pd.Timestamp.now(), "is_recent": True,
        "sentiment": "neutral", "category": "", "confidence": 0, "word_score": 0,
        "signal": 0.0, "sentiment_source": "lexicon"}])
    out = news_mod._apply_llm(df)
    # Unchanged: no key means lexicon stays in place.
    assert out.loc[0, "sentiment_source"] == "lexicon"


def pytest_approx(v, tol=1e-9):
    import pytest
    return pytest.approx(v, abs=tol)
