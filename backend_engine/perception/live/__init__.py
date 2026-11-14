"""
Mode B: read-only companion for real Clash Royale games.

SAFETY CONTRACT (enforced by tests/test_mode_b.py::test_read_only_contract):
this package only *reads pixels* from a mirrored phone screen or recorded frames.
It never sends taps, clicks, keystrokes, ADB input or memory reads to the game, and
never imports libraries that can (pyautogui, pynput, Quartz event posting, adb shell input…).

Pipeline
    FrameSource.grab()                 screen region / recorded frames      (capture.py)
      -> Detector.detect(frame)        units & spells in pixels            (detector.py)
      -> ArenaCalibration.px_to_tile   pixels -> 18x32 tiles               (calibration.py)
      -> PlayTracker.update            detections -> PublicDeployEvent     (play_tracker.py)
      -> OppElixirTracker + BayesianDeckGuesser  (same code as the simulator)
      -> CompanionState for the overlay window                             (companion.py)
"""
