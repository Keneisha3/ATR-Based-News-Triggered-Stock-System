"""The daily loop that scan, notify and the scheduler all run through.

`scan`, `notify`, and the scheduler call `run_daily`, so the daily flow is defined
in one place rather than spread across CLI commands.

There are two execution modes. The light mode is the scheduled default and does
signal generation only: scan, forward signals, decision objects, persist, notify.
It does not run the sweep, chart rendering or portfolio simulation, so it is fast
and safe to run unattended. The full mode is for manual use and adds the heavier
research overlays (portfolio, sweep, event study, charts) through an injected
report step.

The heavy work is an overlay on the light core and is never on the scheduled path.
Dependencies (scan, price/news loaders, report) are injected, so this module does
not import the CLI and has no import cycle.
"""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from typing import Callable

import pandas as pd

from . import config, signal_analyzer as sa, decision as decision_mod
from . import notify as notify_mod, results as results_mod, outcomes as outcomes_mod

LIGHT, FULL = "light", "full"


def run_daily(*, prices_loader: Callable[[], pd.DataFrame],
              news_loader: Callable[[], pd.DataFrame | None] | None = None,
              scan_fn: Callable[[], None] | None = None,
              report_fn: Callable[[pd.DataFrame], None] | None = None,
              mode: str = LIGHT, do_scan: bool = True, deliver: bool = True,
              on: date | None = None) -> dict:
    """Run the daily loop. Returns a result dict, which is also persisted."""
    on = on or date.today()

    # 1. Signal generation (scan): optional fresh data pull.
    if do_scan and scan_fn is not None:
        try:
            scan_fn()
        except Exception as exc:
            print(f"  ! scan failed, falling back to cached data: {exc}")

    prices = prices_loader()
    news_summary = news_loader() if news_loader else None

    # 2. Forward signals -> 3. canonical decision objects.
    signals = sa.forward_signals(prices, news_summary=news_summary)
    decisions = decision_mod.build_decisions(signals, on=on)

    # 4. Persist the decision record and a run manifest.
    decisions_path = decision_mod.write_decisions(decisions, on=on)
    manifest_path = _write_manifest(prices, decisions, mode=mode, on=on)

    # 4b. Feedback loop: log today's decisions and resolve any now mature.
    outcomes_mod.record(decisions)
    outcomes_mod.update(prices)

    # 5. Deliver the digest. The light path stops here.
    delivery = notify_mod.deliver(signals) if deliver else {}

    # Heavy overlays only in full mode, never on the scheduled path.
    if mode == FULL and report_fn is not None:
        report_fn(prices)

    return {"mode": mode, "regime": signals.get("regime"),
            "decisions": decisions, "summary": decision_mod.summary(decisions),
            "decisions_path": decisions_path, "manifest_path": manifest_path,
            "delivery": delivery, "signals": signals}


def _write_manifest(prices: pd.DataFrame, decisions: list[dict], *,
                    mode: str, on: date) -> str:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "command": "pipeline",
        "mode": mode,
        "run_date": on.isoformat(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": results_mod.git_hash(),
        "data": results_mod._data_provenance(prices),
        "decision_summary": decision_mod.summary(decisions),
        "n_decisions": len(decisions),
    }
    path = config.RESULTS_DIR / "pipeline_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return str(path)
