"""Main entry point for CR-Pro-Coach desktop application.

Usage:
    python main.py
    # or:
    python -m frontend_app.qt.app
"""
import sys
from pathlib import Path

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frontend_app.qt.shell import main

if __name__ == "__main__":
    main()
