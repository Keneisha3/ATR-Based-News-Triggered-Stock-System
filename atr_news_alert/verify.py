"""Self-consistency checks on the cached data and generated artifacts.

`python main.py verify` runs a set of mechanical checks that can be re-derived
from the data: integrity of the price history, P&L reconstruction, no gaps or
NaNs, reconciliation of trades against position changes, and determinism (the
same inputs produce the same outputs). It exits non-zero if any check fails.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config, portfolio as pf, backtest as bt


def _check(name: str, passed: bool, detail: str = "") -> dict:
    return {"name": name, "passed": bool(passed), "detail": detail}


def run_checks(prices: pd.DataFrame) -> list[dict]:
    """Return a list of {name, passed, detail} self-consistency checks."""
    checks: list[dict] = []
    if prices is None or prices.empty:
        return [_check("data present", False, "no price history found")]

    dates = sorted(prices["date"].unique())

    # 1. Data integrity: dates strictly increasing within each ticker, no dupes.
    mono = all(
        g["date"].is_monotonic_increasing and g["date"].is_unique
        for _, g in prices.groupby("ticker"))
    checks.append(_check("data integrity", mono,
                         f"{len(prices):,} rows, {prices['ticker'].nunique()} tickers, "
                         f"{dates[0].date()} to {dates[-1].date()}"))

    m = pf.simulate(prices)

    # 2. Return reconstruction: final equity must reproduce the reported total.
    recon = (float(m["equity"].iloc[-1]) - 1.0) * 100
    checks.append(_check("return reconstruction",
                         abs(recon - m["total_return_pct"]) < 0.01,
                         f"equity {recon:.2f}% vs reported {m['total_return_pct']:.2f}%"))

    # 3. No missing trading days: equity curve spans the price calendar.
    aligned = list(m["equity"].index) == dates
    checks.append(_check("no missing trading days", aligned,
                         f"{len(m['equity'])} equity points vs {len(dates)} price days"))

    # 4. No NaN propagation in returns or equity.
    clean = not m["strat_returns"].isna().any() and not m["equity"].isna().any()
    checks.append(_check("no NaN propagation", clean))

    # 5. Trade vs position reconciliation: each trade is one non-zero position run.
    trades = pf.trade_log(m["frame"])
    starts = 0
    for _, g in m["frame"].groupby("ticker"):
        pos = g.sort_values("date")["pos"].to_numpy()
        prev = np.concatenate([[0.0], pos[:-1]])
        starts += int(np.sum((pos != 0) & (pos != prev)))
    checks.append(_check("trade-position consistency", len(trades) == starts,
                         f"{len(trades):,} trades vs {starts:,} position-run starts"))

    # 6. Determinism: same inputs reproduce identical headline metrics.
    m2 = pf.simulate(prices)
    det_pf = (m2["total_return_pct"] == m["total_return_pct"]
              and m2["sharpe"] == m["sharpe"])
    _, agg = bt.backtest(prices)
    b1 = bt.random_baseline(prices, agg.get("events", 0))
    b2 = bt.random_baseline(prices, agg.get("events", 0))
    det_base = b1 == b2
    checks.append(_check("determinism (idempotency)", det_pf and det_base,
                         "portfolio + seeded baseline reproduce bit-for-bit"))

    # 7. Manifest cross-check (only if a prior run was snapshotted).
    man_path = config.RESULTS_DIR / "run_manifest.json"
    if man_path.exists():
        man = json.loads(man_path.read_text())
        d = man.get("data", {})
        match = (d.get("rows") == len(prices)
                 and d.get("start") == dates[0].date().isoformat()
                 and d.get("end") == dates[-1].date().isoformat())
        checks.append(_check("manifest matches data", match,
                             f"manifest {d.get('start')} to {d.get('end')}, "
                             f"{d.get('rows')} rows"))

    return checks


def format_report(checks: list[dict]) -> str:
    lines = []
    for c in checks:
        mark = "✔" if c["passed"] else "✖"
        detail = f"   ({c['detail']})" if c["detail"] else ""
        lines.append(f"  {mark} {c['name']:28} {'PASS' if c['passed'] else 'FAIL'}{detail}")
    n_pass = sum(c["passed"] for c in checks)
    lines.append("")
    lines.append(f"  {n_pass}/{len(checks)} checks passed"
                 + ("" if n_pass == len(checks) else "  (review failures above)"))
    return "\n".join(lines)


def all_passed(checks: list[dict]) -> bool:
    return bool(checks) and all(c["passed"] for c in checks)
