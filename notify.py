"""Deliver the daily decision aid to the user.

Turns the `forward_signals` output (today's breaches and regime-conditioned
expectations) into a digest and sends it through whatever channels are available:
a macOS desktop notification (via `osascript`, no dependency), email through
smtplib when the SMTP environment variables are set, and a markdown file at
`results/daily_digest.md`, which is always written.

The channel send-functions are separate from the message-building functions so
the formatting can be unit-tested without sending anything.
"""

from __future__ import annotations

import os
import smtplib
import subprocess
import sys
from datetime import date
from email.message import EmailMessage

from . import config

# Env vars that enable email (all optional; absent => email channel skipped).
_SMTP_HOST = "ATR_SMTP_HOST"
_SMTP_PORT = "ATR_SMTP_PORT"
_SMTP_USER = "ATR_SMTP_USER"
_SMTP_PASS = "ATR_SMTP_PASS"
_FROM = "ATR_NOTIFY_FROM"
_TO = "ATR_NOTIFY_TO"


def build_digest(signals: dict) -> tuple[str, str]:
    """Short (title, body) for a desktop notification. Body is a few lines max."""
    sigs = signals.get("signals", []) if signals else []
    regime = (signals.get("regime", "unknown") if signals else "unknown").upper()
    if not sigs:
        return (f"ATR: no breaches today ({regime})",
                "No tickers breached the ATR threshold on the latest bar.")
    n = len(sigs)
    title = f"ATR: {n} breach{'es' if n != 1 else ''} today ({regime})"
    lines = []
    for s in sigs[:6]:
        news = "📰" if s.get("has_news") else "  "
        lines.append(f"{news} {s['ticker']} {s['breach']} "
                     f"({s['move_in_atr']:+.1f} ATR)")
    if n > 6:
        lines.append(f"…and {n - 6} more")
    return title, "\n".join(lines)


def build_digest_markdown(signals: dict, *, today: date | None = None) -> str:
    """Full markdown digest written to results/daily_digest.md."""
    from . import signal_analyzer as sa  # reuse the formatted table
    today = today or date.today()
    md = [f"# ATR daily digest, {today.isoformat()}", ""]
    md.append("```")
    md.append(sa.format_signals(signals))
    md.append("```")
    md.append("")
    md.append("*Expected drift is the regime-conditioned post-breach average from "
              "the event study, and is guidance rather than a guarantee.*")
    return "\n".join(md)


# --- channel: macOS desktop notification --------------------------------------

def _macos_notify_args(title: str, message: str) -> list[str]:
    """Build the osascript command (separated for testing). Single-line message."""
    msg = message.replace("\n", " · ").replace('"', "'")
    title = title.replace('"', "'")
    script = f'display notification "{msg}" with title "{title}"'
    return ["osascript", "-e", script]


def send_macos_notification(title: str, message: str) -> bool:
    """Fire a desktop notification on macOS. No-op (returns False) elsewhere."""
    if sys.platform != "darwin":
        return False
    try:
        subprocess.run(_macos_notify_args(title, message), timeout=10, check=False)
        return True
    except Exception:
        return False


# --- channel: email (optional) ------------------------------------------------

def email_configured() -> bool:
    return all(os.environ.get(k) for k in (_SMTP_HOST, _FROM, _TO))


def build_email_message(subject: str, body: str, *, from_addr: str,
                        to_addr: str) -> EmailMessage:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = from_addr
    msg["To"] = to_addr
    msg.set_content(body)
    return msg


def send_email(subject: str, body: str) -> bool:
    """Send the digest by email if SMTP env vars are configured, else skip."""
    if not email_configured():
        return False
    msg = build_email_message(subject, body,
                              from_addr=os.environ[_FROM], to_addr=os.environ[_TO])
    try:
        port = int(os.environ.get(_SMTP_PORT, "587"))
        with smtplib.SMTP(os.environ[_SMTP_HOST], port, timeout=20) as s:
            s.starttls()
            if os.environ.get(_SMTP_USER) and os.environ.get(_SMTP_PASS):
                s.login(os.environ[_SMTP_USER], os.environ[_SMTP_PASS])
            s.send_message(msg)
        return True
    except Exception as exc:
        print(f"  ! email send failed: {exc}")
        return False


# --- orchestration ------------------------------------------------------------

def deliver(signals: dict, *, write_file: bool = True) -> dict:
    """Push the digest through every available channel. Returns {channel: status}."""
    title, body = build_digest(signals)
    status = {}

    if write_file:
        config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.RESULTS_DIR / "daily_digest.md"
        path.write_text(build_digest_markdown(signals))
        status["file"] = str(path)

    status["desktop"] = "sent" if send_macos_notification(title, body) else "skipped"
    status["email"] = ("sent" if send_email(title, body)
                       else ("failed" if email_configured() else "not configured"))
    return status
