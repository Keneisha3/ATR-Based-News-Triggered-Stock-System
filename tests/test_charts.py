"""Visual layer: chart rendering produces valid, non-empty PNGs (headless).

Offline. Charts are saved to a tmp dir; we assert each file exists, is a real PNG
(magic bytes), and is non-trivial in size. The goal is to catch rendering crashes
and wiring regressions, not to pixel-compare images.
"""

import pandas as pd
import pytest

from atr_news_alert import portfolio as pf, config

charts = pytest.importorskip("atr_news_alert.charts")  # skip if matplotlib absent

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _is_png(path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(8) == PNG_MAGIC and path.stat().st_size > 1000


def _metrics(monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    prices = pd.DataFrame({
        "ticker": "X", "date": dates,
        "close": [100, 100, 110, 108, 112, 110, 99, 99.0],
        "move_in_atr": [0, 2.0, 0, -2.0, 0, 0, -2.0, 0],
    })
    return pf.simulate(prices, hold=2, cost_bps=5)


def test_render_portfolio_writes_pngs(tmp_path, monkeypatch):
    m = _metrics(monkeypatch)
    imgs = charts.render_portfolio(m, outdir=tmp_path)
    assert set(imgs) == {"equity", "trades", "exposure"}
    for name in ("equity_curve.png", "trade_distribution.png", "exposure_timeline.png"):
        assert _is_png(tmp_path / name)


def test_sweep_heatmap_writes_png(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    prices = pd.DataFrame({
        "ticker": "X", "date": dates,
        "close": [100, 100, 110, 108, 112, 110, 115, 113, 118, 116.0],
        "move_in_atr": [0, 2.0, 0, -2.0, 0, 2.0, 0, -2.0, 0, 0],
    })
    grid, _ = pf.sweep(prices, holds=[1, 2, 3], costs=[0.0, 5.0])
    imgs = charts.render_sweep(grid, outdir=tmp_path)
    assert _is_png(tmp_path / "sweep_heatmap.png")


def test_event_study_curve_writes_png(tmp_path, monkeypatch):
    from atr_news_alert import signal_analyzer as sa
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=80, freq="D")
    signs = np.zeros(80); signs[60] = 2.0; signs[65] = -2.0
    prices = pd.DataFrame({"ticker": "X", "date": dates,
                           "close": np.linspace(100, 180, 80), "move_in_atr": signs})
    study = sa.event_study(prices, max_days=5)
    imgs = charts.render_event_study(study, outdir=tmp_path)
    assert _is_png(tmp_path / "event_study_curve.png")


def test_ml_diagnostics_writes_png(tmp_path, monkeypatch):
    from atr_news_alert import ml
    monkeypatch.setattr(config, "ATR_BREACH_MULT", 1.5)
    monkeypatch.setattr(config, "REGIME_MA_WINDOW", 15)
    import numpy as np
    rng = np.random.default_rng(5)
    frames = []
    for t in ("A", "B"):
        dates = pd.bdate_range("2021-01-01", periods=500)
        close = 100 * np.cumprod(1 + rng.normal(0.0007, 0.02, 500))
        atr = pd.Series(close).rolling(14).std().bfill().to_numpy() + 1e-6
        move = np.concatenate([[0], np.diff(close)]) / atr
        frames.append(pd.DataFrame({"ticker": t, "date": dates, "close": close,
                                    "volume": 1_000_000, "atr_pct": 0.02,
                                    "move_in_atr": move}))
    res = ml.run_ml(pd.concat(frames, ignore_index=True), horizon=5)
    imgs = charts.render_ml(res, outdir=tmp_path)
    assert _is_png(tmp_path / "ml_diagnostics.png")


def test_charts_handle_no_trades(tmp_path):
    # An empty trade log must still render (a blank but valid histogram).
    empty = pd.DataFrame(columns=["ticker", "entry_date", "exit_date",
                                  "direction", "days_held", "gross_return_pct"])
    out = charts.trade_distribution(empty, tmp_path / "empty.png")
    assert _is_png(tmp_path / "empty.png")
