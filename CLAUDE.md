# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Navigation

For detailed context, read these docs before making non-trivial changes:

- `docs/agent_harness.rst` — development harness and expected agent workflow
- `docs/architecture.rst` — package layers and allowed dependency directions
- `docs/project_specification.rst` — product requirements and workflow contracts
- `docs/quality_and_automation.rst` — local commands, CI gates, and validation loops
- `docs/compatibility_policy.rst` — compatibility aliases and expiry rules
- `docs/implemented_functionality.rst` — current feature surface
- `docs/castep_keyword_catalog.rst` — how `castep_keywords.json` is discovered,
  generated, imported, and surfaced to users (read before touching catalog
  discovery in `castep_io/schema.py`, `castep_setup.py`, or
  `services/cluster_keyword_generator.py`)
- `docs/castep/kpoint_spacing_convention.md` — conventions and round-trip
  guarantees for the spacing↔grid mapping (read before touching
  `science/kpoint_grids.py`, `app/convergence_examiner.py`,
  `app/kpoint_convergence_widget.py`, or any "Use spacing" toggle)

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

## Workflow submission

Per-workflow submission logic lives in `mufinder.app.workflow_controllers`, not on `MainWindow`. Each workflow has a `WorkflowJobController` subclass owning its `submit → materialize → result → refresh` cycle, constructed from the shared `WorkflowControllerContext`, an optional per-controller deps dataclass (required, precisely-typed callables), and its own view. `tests/test_agent_harness.py` ratchets the `_on_submit_*` / `_materialize_*` handlers remaining on `MainWindow` (only the two cross-workflow dashboard handlers): the set may only shrink. Add a new workflow by writing a controller — never a new `MainWindow` submit/materialize handler. See `docs/architecture.rst` (Workflow controllers) for the full contract.

## Quality gates

All changes must satisfy:

- Tests pass (`python -m pytest -n 4 --dist load` locally; CI uses `-n auto`)
- Coverage ≥ 85% (enforced by pytest-cov)
- Sphinx docs build with warnings as errors (`python -m sphinx -W -b html docs docs/_build/html`)

## Common local commands

```bash
# Install dev dependencies (uv preferred; plain pip also works).
# The scientific-backends extra is required by the test suite (soprano, mulfc).
uv sync --extra dev --extra scientific-backends
# or: python -m pip install -e ".[dev,scientific-backends]"

# Run the GUI application
python -m mufinder.main

# Full test suite with coverage.
# Locally, cap parallelism at 4 workers and run in the foreground — each xdist
# worker loads Qt, and `-n auto` can exhaust the machine. Never run the full
# suite as a background process you are not watching. CI uses `-n auto`.
python -m pytest -n 4 --dist load

# Run a single test file or function
python -m pytest --no-cov tests/science/test_supercell.py
python -m pytest --no-cov tests/science/test_supercell.py::test_min_image_distance

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

## CASTEP rules

`castep_keywords.json` must never be committed; it must not appear in the repository. MuFinder discovers CASTEP metadata at runtime via environment variables (`MUFINDER_CASTEP_KEYWORDS_JSON`, `CASTEP_KEYWORDS_JSON`, `MUFINDER_CASTEP_ROOT`, `CASTEP_ROOT`).

Treat `docs/castep/keyword_catalog.yaml` as the authoritative contributor-facing source for implemented keywords and their formatting rules. Never invent CASTEP keywords, block names, allowed values, units, or syntax — leave gaps unimplemented and note them in the catalog instead.

The CASTEP I/O stack has a deliberate three-way split:
- `castep_io/schema.py` — discovers local metadata; defines the supported runtime subset
- `castep_io/validators.py` — enforces explicit validation
- `castep_io/cell_writer.py` / `castep_io/param_writer.py` — render only validated data

When extending CASTEP support: update docs and schema first, then validator/writer logic, then tests. Preserve exact CASTEP formatting conventions: uppercase keywords, deterministic ordering, `%BLOCK … %ENDBLOCK` delimiters.

## Platform-specific input handling

Delete-key actions in Qt widgets must bind **both** `QKeySequence.StandardKey.Delete` and `Qt.Key.Key_Backspace`. On macOS the physical Delete key is delivered as Backspace, so any delete/remove/cancel/reset shortcut needs both bindings and a focused regression test.

## Workflow parameter surface

Shared CASTEP parameters must appear first in workflow configuration dialogs, in this stable order: `basis_precision`, `cut_off_energy`, `elec_energy_tol`, `kpoint_mp_grid`. Workflow-specific rows follow. Use the shared helper in `mufinder.app.workflow_parameters` rather than encoding a local ordering rule in individual widgets.

## Compatibility aliases

Before removing or renaming any persisted field, import alias, or public helper, check `docs/compatibility_policy.rst`. Every new shim needs a ledger entry there, including the schema version or migration event that will allow removal.

## Branch workflow

The project follows GitHub Flow: `main` is the single long-lived branch.

1. Branch short-lived `feature/*` (or `fix/*`) branches off `main`. Aim to merge
   small features within about a week; longer-lived branches are allowed when a
   feature genuinely needs it.
2. Open a pull request **directly into `main`**. CI runs the full quality gates on
   every PR; merges are blocked until they pass. PR bodies must follow
   `.github/PULL_REQUEST_TEMPLATE.md` — `gh pr create --body` does not apply it
   automatically, so read it before writing the body. PR titles use Conventional
   Commits prefixes (`feat:`, `fix:`, …), which semantic-release parses for
   version bumps and the changelog.
3. Use the `Preview Build` workflow to produce downloadable artifacts from any
   branch for ad hoc validation, without publishing a release.
4. Release by pushing a `vX.Y.Z` tag. Branch protection prevents pushing the
   version bump commit directly to `main`, so the process is:
   ```
   git checkout main && git pull
   uvx "python-semantic-release>=10.5,<11" version   # bumps pyproject.toml + CHANGELOG.md, commits + tags locally
   git checkout -b release/vX.Y.Z                    # branch off the version bump commit
   git checkout main && git reset --hard origin/main  # restore local main
   git push origin release/vX.Y.Z                    # open a PR, wait for CI + merge
   git push origin vX.Y.Z                            # push the tag — triggers Release Publish
   ```
   The `Release Publish` workflow builds the Windows and macOS binaries and
   publishes the GitHub release from the tag.
