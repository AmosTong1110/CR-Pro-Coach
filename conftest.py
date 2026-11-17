"""Global pytest configuration for CR-Pro-Coach.

Includes exit handler to prevent PySide6/Shiboken macOS crash-reporter popup on Python exit.
"""
import os
import sys


def pytest_sessionfinish(session, exitstatus):
    """Exit cleanly via os._exit to prevent PySide6/Shiboken crash on macOS atexit."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exitstatus)
