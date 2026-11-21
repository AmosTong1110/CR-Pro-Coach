# CRbot (CR-Pro-Coach): instructions for Claude

## Session start
- Read `docs/SESSION_HANDOFF.md` (short current state). Do not read `docs/HANDOFF_ARCHIVE.md` or
  `docs/CRBOT_UPGRADE_PLAN.md` whole: grep them for the section the task needs.
- Before ending a session that changed the project, update `docs/SESSION_HANDOFF.md`. Keep it short
  (under ~5 KB): replace stale lines instead of appending history; move long detail to the archive.

## Ending responses
- End major responses with the `AskUserQuestion` tool offering concrete next-step options.

## Working rules
- Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`; keep all tests green.
- Respect the constraints listed in `docs/SESSION_HANDOFF.md` (M1 8 GB budget, human-limited AI, determinism,
  English-only UI, snake_case card ids, scan downloaded model files, user privacy).
- Reply in the language of the user's latest message.
