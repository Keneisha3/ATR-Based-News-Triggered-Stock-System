"""Scheduler loop runs repeatedly and stops cleanly (reproduces check #5)."""

import pandas as pd

import main


def test_watch_runs_cycles_then_stops(monkeypatch):
    calls = {"n": 0}

    def fake_scan(wl):
        calls["n"] += 1
        if calls["n"] >= 3:
            raise KeyboardInterrupt

    monkeypatch.setattr(main, "cmd_scan", fake_scan)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)  # don't actually wait

    try:
        main.cmd_watch(pd.DataFrame({"ticker": ["AAA"]}), interval_min=15)
    except KeyboardInterrupt:
        pass

    assert calls["n"] == 3


def test_watch_survives_a_failing_cycle(monkeypatch):
    calls = {"n": 0}

    def flaky_scan(wl):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("boom")   # first cycle errors
        if calls["n"] >= 2:
            raise KeyboardInterrupt       # second cycle stops the loop

    monkeypatch.setattr(main, "cmd_scan", flaky_scan)
    monkeypatch.setattr(main.time, "sleep", lambda s: None)

    try:
        main.cmd_watch(pd.DataFrame({"ticker": ["AAA"]}), interval_min=15)
    except KeyboardInterrupt:
        pass

    # Loop kept going after the RuntimeError instead of dying on cycle 1.
    assert calls["n"] == 2
