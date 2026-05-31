"""Scheduling layer: the launchd plist is well-formed and uses absolute argv."""

import plistlib
from pathlib import Path

from atr_news_alert import scheduler


def test_plist_is_valid_and_runs_notify_daily():
    xml = scheduler.build_plist(16, 45, python="/venv/bin/python",
                                main_py=Path("/proj/main.py"),
                                workdir=Path("/proj"), log_dir=Path("/proj/data"))
    parsed = plistlib.loads(xml.encode())
    assert parsed["Label"] == scheduler.LABEL
    assert parsed["ProgramArguments"] == ["/venv/bin/python", "/proj/main.py", "notify"]
    assert parsed["StartCalendarInterval"] == {"Hour": 16, "Minute": 45}
    assert parsed["WorkingDirectory"] == "/proj"


def test_plist_escapes_paths_with_spaces():
    xml = scheduler.build_plist(9, 0, python="/v/py",
                                main_py=Path("/a b/main.py"),
                                workdir=Path("/a b"), log_dir=Path("/a b/data"))
    parsed = plistlib.loads(xml.encode())  # would raise if XML were malformed
    assert "/a b/main.py" in parsed["ProgramArguments"]


def test_install_writes_plist(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))      # Path.home() reads $HOME
    info = scheduler.install(7, 30)
    assert Path(info["plist"]).exists()
    assert info["time"] == "07:30"
    assert "launchctl load" in info["load_cmd"]
    parsed = plistlib.loads(Path(info["plist"]).read_bytes())
    assert parsed["StartCalendarInterval"] == {"Hour": 7, "Minute": 30}
