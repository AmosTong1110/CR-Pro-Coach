#!/bin/bash
# CRbot launcher: local simulator + post-match AI coach
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$DIR"

if [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
else
    PYTHON="python3"
fi

export PYTHONPATH="$DIR:$PYTHONPATH"
exec "$PYTHON" -m frontend_app.qt.app "$@"
