#!/usr/bin/env bash
# Set up the venv, install dependencies, then run main.py.
#
#   ./start.sh            # default command is "start"
#   ./start.sh report     # pass any main.py command through
#   ./start.sh report --refresh
#
# Safe to re-run.

set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"

echo "==> Setting up virtual environment (.venv) ..."
$PY -m venv .venv
./.venv/bin/python -m pip install --quiet --upgrade pip
echo "==> Installing dependencies ..."
./.venv/bin/python -m pip install --quiet -r requirements.txt

if [ "$#" -eq 0 ]; then
  set -- start
fi

echo "==> Running: python main.py $*"
echo
exec ./.venv/bin/python main.py "$@"
