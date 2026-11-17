#!/bin/bash
# Clash Royale Pro Coach Companion Launcher
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="$DIR:$PYTHONPATH"
"$PYTHON" frontend_app/companion/app.py "$@"
