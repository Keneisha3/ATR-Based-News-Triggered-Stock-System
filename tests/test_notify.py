"""Notification layer: digest formatting, channel-arg building, email skip logic.

Offline — nothing is actually sent. We test the pure builders and that delivery
degrades gracefully when channels aren't configured.
"""

import pandas as pd
import pytest

from atr_news_alert import notify, config


@pytest.fixture(autouse=True)
def _tmp_results(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RESULTS_DIR", tmp_path / "results")
    # Never fire a real desktop notification during tests (no on-screen side effects).
    monkeypatch.setattr(notify, "send_macos_notification", lambda *a, **k: True)
    # Ensure email looks unconfigured regardless of the host environment.
    for k in ("ATR_SMTP_HOST", "ATR_NOTIFY_FROM", "ATR_NOTIFY_TO",
              "ATR_SMTP_USER", "ATR_SMTP_PASS", "ATR_SMTP_PORT"):
        monkeypatch.delenv(k, raising=False)


def _signals(n=2, regime="bull"):
    return {
        "regime": regime, "horizon": 3, "drift_pct": 0.33,
        "drift_band": (0.19, 0.48), "n_regime_events": 2089, "continuation": True,
        "signals": [
            {"ticker": "IBM", "breach": "bullish", "move_in_atr": 3.9,
             "expected_move_pct": 0.33, "has_news": True,
             "action": "watch for upside follow-through [news-confirmed]"},
            {"ticker": "COST", "breach": "bearish", "move_in_atr": -1.8,
             "expected_move_pct": -0.33, "has_news": False,
             "action": "watch for downside follow-through [no fresh news]"},
        ][:n],
    }


def test_digest_title_counts_breaches():
    title, body = notify.build_digest(_signals(2))
    assert "2 breaches" in title and "BULL" in title
    assert "IBM" in body and "COST" in body


def test_digest_handles_no_breaches():
    title, body = notify.build_digest(_signals(0))
    assert "no breaches" in title.lower()
    assert "No tickers" in body


def test_macos_notify_args_are_safe():
    args = notify._macos_notify_args('he said "hi"', "line1\nline2")
    assert args[0] == "osascript" and args[1] == "-e"
    script = args[2]
    assert "\n" not in script                 # newlines flattened to ' · '
    assert '"hi"' not in script                # double-quotes neutralized to single
    assert "line1 · line2" in script


def test_email_message_build():
    msg = notify.build_email_message("subj", "body text",
                                     from_addr="a@x.com", to_addr="b@y.com")
    assert msg["Subject"] == "subj"
    assert msg["From"] == "a@x.com" and msg["To"] == "b@y.com"
    assert "body text" in msg.get_content()


def test_email_skipped_when_unconfigured():
    assert notify.email_configured() is False
    assert notify.send_email("s", "b") is False


def test_deliver_writes_file_and_reports_channels():
    status = notify.deliver(_signals(2))
    assert "file" in status
    assert (config.RESULTS_DIR / "daily_digest.md").exists()
    assert status["email"] == "not configured"
    assert status["desktop"] in {"sent", "skipped"}   # platform-dependent
    text = (config.RESULTS_DIR / "daily_digest.md").read_text()
    assert "ATR daily digest" in text and "IBM" in text
