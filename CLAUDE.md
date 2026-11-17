# CRbot (CR-Pro-Coach): instructions for Claude

## Session start
- Whenever a new session starts, or the user asks to continue work, **first read `docs/SESSION_HANDOFF.md`**
  to restore project context, then `docs/CRBOT_UPGRADE_PLAN.md` for the roadmap.
- Before ending a session that changed the project, update `docs/SESSION_HANDOFF.md`
  (completed work, progress %, open issues, next steps).

## Ending responses
- Always end major responses by invoking the `AskUserQuestion` tool with concrete next-step action buttons
  (if the app's own next-step box is used instead, give it the same concrete options).

## Working rules
- Run tests with `QT_QPA_PLATFORM=offscreen python3 -m pytest -q`; keep all tests green.
- Respect the constraints in `docs/SESSION_HANDOFF.md` §3: M1 8 GB budget, human-limited AI,
  determinism, English-only UI, snake_case card ids, scan downloaded model files, user privacy.
- Reply in the language of the user's latest message.
