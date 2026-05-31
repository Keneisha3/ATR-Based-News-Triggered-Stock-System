"""Install a macOS LaunchAgent so the daily run happens on its own.

Generates a LaunchAgent plist that runs `python main.py notify` each day at a
chosen time, writes it to ~/Library/LaunchAgents, and prints the command to load
it. launchd is used instead of cron on macOS because it survives logout and
catches up on runs missed during sleep. Paths with spaces work because
ProgramArguments is an argv array rather than a shell string.
"""

from __future__ import annotations

import sys
from pathlib import Path
from xml.sax.saxutils import escape

from . import config

LABEL = "com.atrnews.daily"


def _python_executable() -> str:
    """The venv's python. sys.executable is correct when run through the venv."""
    return sys.executable


def build_plist(hour: int, minute: int, *, label: str = LABEL,
                python: str | None = None, main_py: Path | None = None,
                workdir: Path | None = None, log_dir: Path | None = None) -> str:
    """Return the launchd plist XML for a daily `main.py notify` run."""
    python = python or _python_executable()
    main_py = main_py or (config.ROOT / "main.py")
    workdir = workdir or config.ROOT
    log_dir = log_dir or config.DATA_DIR

    args = [python, str(main_py), "notify"]
    args_xml = "\n".join(f"    <string>{escape(a)}</string>" for a in args)
    out_log = escape(str(log_dir / "notify.out.log"))
    err_log = escape(str(log_dir / "notify.err.log"))

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{escape(label)}</string>
  <key>ProgramArguments</key>
  <array>
{args_xml}
  </array>
  <key>WorkingDirectory</key>
  <string>{escape(str(workdir))}</string>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>{hour}</integer>
    <key>Minute</key><integer>{minute}</integer>
  </dict>
  <key>StandardOutPath</key>
  <string>{out_log}</string>
  <key>StandardErrorPath</key>
  <string>{err_log}</string>
  <key>RunAtLoad</key>
  <false/>
</dict>
</plist>
"""


def plist_path(label: str = LABEL) -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{label}.plist"


def install(hour: int, minute: int, *, label: str = LABEL) -> dict:
    """Write the LaunchAgent plist. Returns paths + the launchctl load command."""
    path = plist_path(label)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_plist(hour, minute, label=label))
    return {
        "plist": str(path),
        "load_cmd": f"launchctl load {path}",
        "unload_cmd": f"launchctl unload {path}",
        "time": f"{hour:02d}:{minute:02d}",
    }
