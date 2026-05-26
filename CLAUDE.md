# MuFinder — Claude Code Guide

## Navigation

For detailed context, read these docs before making non-trivial changes:

- `docs/agent_harness.rst` — development harness and expected agent workflow
- `docs/architecture.rst` — package layers and allowed dependency directions
- `docs/project_specification.rst` — product requirements and workflow contracts
- `docs/quality_and_automation.rst` — local commands, CI gates, and validation loops
- `docs/compatibility_policy.rst` — compatibility aliases and expiry rules
- `docs/implemented_functionality.rst` — current feature surface

## Agent workflow

For non-trivial changes:

1. Read `AGENTS.md` and the relevant doc in `docs/`.
2. Identify the package layer that owns the behavior (`docs/architecture.rst`).
3. Update implementation, tests, and documentation together.
4. Run focused tests for the touched area.
5. Run the full quality gates before merging.

A change is **not complete** until tests and docs both describe the new behaviour.

## Package layers and dependency directions

Dependencies flow inward: `app` → `services/workflow/science` → `domain/castep_io/persistence`. Lower layers must not import higher layers.

| Layer | Responsibility |
|---|---|
| `mufinder.domain` | Dataclasses, enums, durable records. Must not import any other MuFinder package. |
| `mufinder.castep_io` | CASTEP parsing, writing, schema, presets. No UI, services, or science. |
| `mufinder.persistence` | JSON storage and codecs. No UI, services, workflow, or science. |
| `mufinder.science` | Numerical and crystallographic algorithms. No UI or services. |
| `mufinder.workflow` | Workflow expansion and rendering. No project mutation, runner state, or widgets. |
| `mufinder.services` | Project state, local/remote execution, scheduling. No PySide widgets. |
| `mufinder.app` | PySide widgets, dialogs, theme, main window. Delegates logic to lower layers. |

These boundaries are enforced by `tests/test_agent_harness.py`. If a new dependency direction is genuinely required, update `docs/architecture.rst` and the structural test in the same change.

## Quality gates

All changes must satisfy:

- Tests pass (`python -m pytest`)
- Coverage ≥ 85% (enforced by pytest-cov)
- Sphinx docs build with warnings as errors (`python -m sphinx -W -b html docs docs/_build/html`)

## Common local commands

```bash
# Install dev dependencies
python -m pip install -e ".[dev]"

# Full test suite with coverage
python -m pytest

# Targeted checks (fast, no coverage)
python -m pytest --no-cov \
  tests/test_castep_output_parser.py \
  tests/app/test_workflow_submission.py \
  tests/services/test_job_execution_controller.py \
  tests/services/test_local_runner.py \
  tests/app/test_submission_dialog.py \
  tests/app/test_castep_setup_dialog.py \
  tests/app/test_convergence_widget.py \
  tests/app/test_main_window_actions.py

# Harness checks (after changing AGENTS.md, docs, or package boundaries)
python -m pytest --no-cov tests/test_agent_harness.py

# Build docs
python -m sphinx -W -b html docs docs/_build/html
```

## Platform-specific input handling

Delete-key actions in Qt widgets must bind **both** `QKeySequence.StandardKey.Delete` and `Qt.Key.Key_Backspace`. On macOS the physical Delete key is delivered as Backspace, so any delete/remove/cancel/reset shortcut needs both bindings and a focused regression test.

## Workflow parameter surface

Shared CASTEP parameters must appear first in workflow configuration dialogs, in this stable order: `basis_precision`, `cut_off_energy`, `elec_energy_tol`, `kpoint_mp_grid`. Workflow-specific rows follow. Use the shared helper in `mufinder.app.workflow_parameters` rather than encoding a local ordering rule in individual widgets.

## Compatibility aliases

Before removing or renaming any persisted field, import alias, or public helper, check `docs/compatibility_policy.rst`. Every new shim needs a ledger entry there, including the schema version or migration event that will allow removal.

## Branch workflow

1. Day-to-day development on `develop`.
2. Manual preview builds from `develop` for distributable artifact validation.
3. Cut `release/vX.Y.Z` from `develop` once stable.
4. Release PR auto-merges into `main` once tests and docs build pass.
5. Publish workflow tags and releases from `main`.
6. `main` merges back into `develop`; release branch is deleted automatically.
