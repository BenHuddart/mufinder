# MuFinder Agent Map

Use this file as the short route map. The detailed system of record lives in
the repository docs so future agent runs can discover the current product,
architecture, and quality expectations without relying on thread context.

- Start with `docs/agent_harness.rst` for the MuFinder development harness and
  the expected agent workflow.
- Use `docs/architecture.rst` for package boundaries and allowed dependency
  directions before moving code between layers.
- Use `docs/project_specification.rst` for product requirements and durable
  workflow contracts.
- Use `docs/quality_and_automation.rst` for local commands, CI gates, and
  targeted validation loops.
- Do not develop directly on `main`. Create a short-lived `feature/*` or
  `fix/*` branch before making repository changes, then open a PR into `main`.
- Use `docs/compatibility_policy.rst` before changing persisted names,
  compatibility aliases, or project-file migration behavior.
- Update feature docs in `docs/` whenever behavior changes; new functionality
  is incomplete until tests and documentation both describe it.
- Delete-key functionality in Qt widgets must support both
  `QKeySequence.StandardKey.Delete` and `Qt.Key.Key_Backspace`. On macOS, the
  keyboard key labelled Delete is commonly delivered as Backspace, so any
  future delete/remove/cancel/reset shortcut should bind Backspace to the same
  handler as Delete and include a focused regression test.
