"""Decision schema: the standard output record used across the system.

notify, report and the outcome log all read the same object, so the day's
decisions have one shape everywhere. `signal_analyzer.forward_signals` produces
the underlying numbers and this module turns them into a serializable record with
an assigned action.

Decision record:
    {
      "date": "2026-05-29",
      "ticker": "AAPL",
      "signal": "BREACH_UP" | "BREACH_DOWN",
      "regime": "BULL" | "BEAR" | "UNKNOWN",
      "move_in_atr": 3.9,
      "expected_3d_return_pct": 0.33,   # in the breach direction, regime-conditioned
      "confidence": 0.62,               # P(continuation | regime), nudged by news
      "has_news": true,
      "action": "LONG" | "HOLD" | "IGNORE",
      "rationale": "bull-regime momentum, news-confirmed"
    }

Actions are long-biased ({LONG, HOLD, IGNORE}, no SHORT). The research found the
usable edge is in timing long exposure, and that shorting breaches underperforms,
so the schema reflects that.
"""

from __future__ import annotations

import json
from datetime import date

from . import config, signal_analyzer as sa

SCHEMA_VERSION = 1
ACTIONS = ("LONG", "HOLD", "IGNORE")


def _decide_action(signal: str, continuation: bool) -> tuple[str, str]:
    """Map (breach direction, regime behaviour) to (action, rationale).

    The only actionable buy is an up-breach in a continuation (bull) regime.
    Everything else is HOLD or IGNORE, since the system does not short.
    """
    if continuation:
        if signal == "BREACH_UP":
            return "LONG", "continuation regime, momentum favours follow-through"
        return "IGNORE", "down-breach in a continuation regime, no long edge (we don't short)"
    # reversion regime
    if signal == "BREACH_UP":
        return "HOLD", "reversion regime, up-breach likely to fade, don't add"
    return "IGNORE", "reversion regime, down-breach may bounce, too noisy to act"


def _confidence(regime_hit_rate, has_news: bool) -> float:
    """P(continuation | regime) from the event study, nudged up by fresh news."""
    if regime_hit_rate is None:
        return 0.5
    conf = regime_hit_rate / 100.0
    if has_news:
        conf = min(1.0, conf + 0.05)   # small, bounded confirmation bump
    return round(conf, 3)


def build_decisions(res: dict, *, on: date | None = None) -> list[dict]:
    """Turn a `forward_signals` result into a list of decision records."""
    on = on or date.today()
    regime = res.get("regime", "unknown")
    continuation = res.get("continuation", True)
    horizon = res.get("horizon", config.PORTFOLIO_HOLD_DAYS)
    drift = res.get("drift_pct")

    out = []
    for s in res.get("signals", []):
        signal = "BREACH_UP" if s["breach"] == "bullish" else "BREACH_DOWN"
        action, why = _decide_action(signal, continuation)
        conf = _confidence(res.get("regime_hit_rate"), s.get("has_news", False))
        rationale = why + (", news-confirmed" if s.get("has_news") else "")
        out.append({
            "schema_version": SCHEMA_VERSION,
            "date": on.isoformat(),
            "ticker": s["ticker"],
            "signal": signal,
            "regime": regime.upper(),
            "move_in_atr": s["move_in_atr"],
            f"expected_{horizon}d_return_pct": s.get("expected_move_pct"),
            "confidence": conf,
            "has_news": bool(s.get("has_news", False)),
            "action": action,
            "rationale": rationale,
        })
    # Most actionable first: LONG before HOLD before IGNORE, then by ATR size.
    rank = {a: i for i, a in enumerate(ACTIONS)}
    out.sort(key=lambda d: (rank.get(d["action"], 9), -abs(d["move_in_atr"])))
    return out


def summary(decisions: list[dict]) -> dict:
    """Counts by action, used as a header for digests and reports."""
    s = {a: 0 for a in ACTIONS}
    for d in decisions:
        s[d["action"]] = s.get(d["action"], 0) + 1
    return s


def write_decisions(decisions: list[dict], *, on: date | None = None) -> str:
    """Persist the day's decisions as the canonical JSON contract output."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.RESULTS_DIR / "decisions.json"
    payload = {"date": (on or date.today()).isoformat(),
               "schema_version": SCHEMA_VERSION,
               "summary": summary(decisions),
               "decisions": decisions}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return str(path)


def format_decisions(decisions: list[dict]) -> str:
    """Human-readable decision table."""
    if not decisions:
        return "No decisions today (no breaches on the latest bar)."
    counts = summary(decisions)
    head = "  ".join(f"{a}:{counts[a]}" for a in ACTIONS)
    lines = [f"Decisions  {head}", "",
             f"  {'ticker':8}{'action':8}{'signal':12}{'conf':>6}  rationale"]
    for d in decisions:
        lines.append(f"  {d['ticker']:8}{d['action']:8}{d['signal']:12}"
                     f"{d['confidence']:>6.2f}  {d['rationale']}")
    return "\n".join(lines)
