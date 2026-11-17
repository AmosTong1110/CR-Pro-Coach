"""Global pytest configuration for CR-Pro-Coach.

Includes exit handler to prevent PySide6/Shiboken macOS crash-reporter popup on Python exit.
"""
import os
import sys


def pytest_unconfigure(config):
    """Exit cleanly via os._exit after all test summaries and reports are printed,
    bypassing buggy Shiboken C++ atexit destruction on macOS.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(config.option.exitstatus if hasattr(config.option, "exitstatus") else 0)
