"""Reproducibility layer: every run leaves an auditable trail.

A result you can't reproduce or inspect isn't research, it's an anecdote. This
module writes, for each portfolio run:

  * results/run_manifest.json        what was run, on what data, with what params,
                                     at what code version, and the headline metrics
  * results/equity_curve.csv         strategy + benchmark equity and drawdown by day
  * results/trades.csv               one row per discrete trade (entry/exit/return)
  * results/positions_over_time.csv  daily long/short exposure

Anyone can then replot the curve, validate the P&L, or inspect individual trades.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import pandas as pd

from . import config, portfolio as pf

# Metric keys that are JSON-serializable scalars (Series/frames are excluded).
_SCALAR_METRICS = [
    "hold_days", "long_only", "cost_bps", "trades", "days", "pct_days_invested",
    "total_return_pct", "benchmark_return_pct", "cagr_pct", "benchmark_cagr_pct",
    "sharpe", "benchmark_sharpe", "max_drawdown_pct", "benchmark_max_drawdown_pct",
]


def git_hash() -> str | None:
    """Short commit hash if this is a git checkout, else None (handled gracefully)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _data_provenance(prices: pd.DataFrame) -> dict:
    if prices is None or prices.empty:
        return {"rows": 0, "tickers": 0, "start": None, "end": None}
    d = pd.to_datetime(prices["date"])
    return {
        "rows": int(len(prices)),
        "tickers": int(prices["ticker"].nunique()),
        "start": d.min().date().isoformat(),
        "end": d.max().date().isoformat(),
    }


def build_manifest(metrics: dict, prices: pd.DataFrame, *,
                   command: str, extra_params: dict | None = None) -> dict:
    """Assemble the run manifest (timestamp, data, params, code version, metrics)."""
    return {
        "command": command,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "data": _data_provenance(prices),
        "params": {
            "atr_period": config.ATR_PERIOD,
            "atr_breach_mult": config.ATR_BREACH_MULT,
            "price_lookback": config.PRICE_LOOKBACK,
            "forward_horizons": list(config.FORWARD_HORIZONS),
            "hold_days": metrics.get("hold_days"),
            "cost_bps": metrics.get("cost_bps"),
            "long_only": metrics.get("long_only"),
            "baseline_seed": config.BASELINE_SEED,
            **(extra_params or {}),
        },
        "metrics": {k: metrics[k] for k in _SCALAR_METRICS if k in metrics},
    }


def snapshot_portfolio(metrics: dict, prices: pd.DataFrame, *,
                       command: str = "portfolio") -> dict:
    """Write manifest + equity/trades/positions CSVs. Returns paths written."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    manifest = build_manifest(metrics, prices, command=command)
    man_path = config.RESULTS_DIR / "run_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    written["manifest"] = man_path

    # Equity curve + drawdown.
    eq = metrics["equity"]
    curve = pd.DataFrame({
        "date": eq.index,
        "strategy_equity": eq.to_numpy(),
        "benchmark_equity": metrics["benchmark_equity"].to_numpy(),
    })
    curve["strategy_drawdown"] = (eq / eq.cummax() - 1.0).to_numpy()
    eq_path = config.RESULTS_DIR / "equity_curve.csv"
    curve.to_csv(eq_path, index=False)
    written["equity_curve"] = eq_path

    # Trades.
    trades = pf.trade_log(metrics["frame"])
    tr_path = config.RESULTS_DIR / "trades.csv"
    trades.to_csv(tr_path, index=False)
    written["trades"] = tr_path

    # Positions over time.
    timeline = pf.positions_timeline(metrics["frame"])
    pos_path = config.RESULTS_DIR / "positions_over_time.csv"
    timeline.to_csv(pos_path, index=False)
    written["positions"] = pos_path

    return written


def snapshot_event_study(study: dict, prices: pd.DataFrame) -> dict:
    """Write the event-study curves (overall + per regime) and a manifest."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    frames = []
    for label in ("overall", "bull", "bear"):
        df = study.get(label)
        if df is not None and not df.empty:
            tagged = df.copy()
            tagged.insert(0, "regime", label)
            frames.append(tagged)
    if frames:
        path = config.RESULTS_DIR / "event_study.csv"
        pd.concat(frames, ignore_index=True).to_csv(path, index=False)
        written["event_study"] = path

    manifest = {
        "command": "event-study",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "data": _data_provenance(prices),
        "params": {
            "atr_breach_mult": config.ATR_BREACH_MULT,
            "max_days": study.get("max_days"),
            "regime_ma_window": config.REGIME_MA_WINDOW,
        },
        "n_events": study.get("n_events"),
        "regime_counts": study.get("regime_counts"),
    }
    man_path = config.RESULTS_DIR / "event_study_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, default=str))
    written["manifest"] = man_path
    return written


def snapshot_sweep(grid: pd.DataFrame, summary: dict, prices: pd.DataFrame) -> dict:
    """Write the parameter-sweep grid + a manifest describing the robustness run."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    grid_path = config.RESULTS_DIR / "sweep.csv"
    grid.to_csv(grid_path, index=False)
    written["sweep"] = grid_path

    manifest = {
        "command": "sweep",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "data": _data_provenance(prices),
        "params": {
            "atr_breach_mult": config.ATR_BREACH_MULT,
            "long_only": summary.get("long_only"),
            "hold_grid": sorted(grid["hold_days"].unique().tolist()),
            "cost_grid": sorted(grid["cost_bps"].unique().tolist()),
        },
        "robustness": {k: v for k, v in summary.items() if k != "best"},
        "best_cell": summary.get("best"),
    }
    man_path = config.RESULTS_DIR / "sweep_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, default=str))
    written["manifest"] = man_path

    return written
