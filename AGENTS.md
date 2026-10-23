# MuFinder agent guide

This is the one instruction file for every coding agent. Claude Code imports it from
`CLAUDE.md`; Codex, Cursor and Copilot read it directly. Its rules are binding; linked
docs give the rationale. `tests/test_agent_harness.py` caps it at 140 lines and checks it.

## What MuFinder is

A PySide6 desktop application for muon stopping-site workflows with CASTEP. A project
starts from a host crystal structure and moves through supercells, convergence testing,
pristine relaxation, UEP screening, site generation, muon relaxation and analysis,
magnetic fields, quantum corrections and energy barriers. Jobs run locally or on a
cluster and are tracked in a job dashboard.

## Where to read first

| Need | Read |
|---|---|
| Package layers, shared homes, controller contract | `docs/architecture.rst` |
| Commands, CI gates, release steps | `docs/quality_and_automation.rst` |
| Job states and transitions | `docs/job_lifecycle.rst` |
| Persisted names, aliases, migrations | `docs/compatibility_policy.rst` |
| Product requirements | `docs/project_specification.rst` |
| CASTEP keyword catalog discovery and generation | `docs/castep_keyword_catalog.rst` |
| k-point spacing and grid round-trip rules | `docs/castep/kpoint_spacing_convention.rst` |
| Feature surface per workflow | `docs/user_guide/overview.rst` and the user guide |
| How this repository is organised for agents | `docs/agent_harness.rst` |

## Making a change

1. Branch `feature/*` or `fix/*` off `main`; never commit to `main`.
2. Find the owning layer and the concern's shared home in `docs/architecture.rst` first.
3. Change implementation, tests and docs together. A change is incomplete until tests and
   docs both describe the new behaviour.
4. Run the focused tests for the touched area, then the full gates before review.
5. Open a PR into `main` with a Conventional Commits title (`feat:`, `fix:`, `docs:`,
   `refactor:`, `chore:`). Write the body from `.github/PULL_REQUEST_TEMPLATE.md`;
   `gh pr create --body` does not apply it.
6. Before renaming or removing any persisted field, import alias or public helper, read
   `docs/compatibility_policy.rst` and add a ledger entry for every shim.

## Package layers

Dependencies flow inward: `app` → `services` / `workflow` / `science` → `domain` /
`castep_io` / `persistence`. Lower layers never import higher ones, and nothing outside
`mufinder.app` imports Qt widgets. Enforced by `tests/test_agent_harness.py`.

| Layer | Owns |
|---|---|
| `domain` | Dataclasses, enums, durable records. Imports no other MuFinder package. |
| `castep_io` | CASTEP parsing, writing, schema, presets, validation. |
| `persistence` | JSON storage and codecs. |
| `science` | Numerical and crystallographic algorithms. UI-free and service-free. |
| `workflow` | Expands resolved workflow state into `JobSpec` objects and rendered inputs. |
| `services` | Project state, local and cluster execution, scheduling. No widgets. |
| `app` | PySide widgets, dialogs, theme, main window. Delegates logic downward. |

## Reuse before you write

Every cross-workflow concern has exactly one home, listed under "Shared workflow
infrastructure" in `docs/architecture.rst`; read it before adding a helper, extend the
helper there and never copy a sibling workflow's implementation. The harness pins
duplication (`PINNED_DUPLICATED_WIDGET_METHODS`).

Rules:

- A new workflow is a `WorkflowJobController` subclass plus a widget built from the shared
  helpers. Never add `_on_submit_*` / `_materialize_*` handlers to `MainWindow`.
- If a shared helper lacks something a workflow needs, extend the helper in the same PR.
- A private method whose name and purpose already exist on another workflow widget or
  controller is a duplication. Move it to the shared home instead of copying it.
- Logic that computes, validates or mutates project state belongs in `services`,
  `workflow` or `science`, not in a widget or `MainWindow`.
- Reusable styling goes in `theme.py`, not in feature-local stylesheets.

## Standing rules

- **Delete keys.** Every delete, remove, cancel or reset shortcut binds both
  `QKeySequence.StandardKey.Delete` and `Qt.Key.Key_Backspace` (macOS delivers Delete as
  Backspace) and has a focused regression test.
- **Shared parameter order.** Workflow configuration dialogs show `basis_precision`,
  `cut_off_energy`, `elec_energy_tol`, `kpoint_mp_grid` first, through
  `app/workflow_parameters.py`; workflow-specific rows follow.
- **Plots.** Build figures with `create_plot_canvas`; wrap a plot that shares a dialog in
  `PlotFrame`; never call `subplots_adjust` or disable the layout engine; add every new
  plot dialog to `PLOT_DIALOGS` in `tests/app/test_plot_canvas_layout.py`.
- **Confirmations.** Every remove, cancel, reset, dismiss or delete action is confirmed
  first, from every entry point, with the safe choice as the default.
- **Startup.** Nothing heavyweight runs before the splash screen is shown.

## CASTEP rules

- `castep_keywords.json` is never committed. Metadata is discovered at runtime through
  `MUFINDER_CASTEP_KEYWORDS_JSON`, `CASTEP_KEYWORDS_JSON`, `MUFINDER_CASTEP_ROOT` and
  `CASTEP_ROOT`; see `docs/castep_keyword_catalog.rst`.
- `docs/castep/keyword_catalog.yaml` is the authoritative list of implemented keywords and
  their formatting rules. Never invent keywords, block names, values, units or syntax;
  leave a gap unimplemented and note it in the catalog.
- Keep the split: `castep_io/schema.py` discovers metadata and defines the supported
  subset, `castep_io/validators.py` validates, and `cell_writer.py` / `param_writer.py`
  render validated data only. Extend in that order: docs and schema, then validators and
  writers, then tests.
- Rendering is deterministic: uppercase keywords, fixed ordering, `%BLOCK` / `%ENDBLOCK`
  delimiters, stable scalar formatting. Raise descriptive errors; never ignore an
  unsupported or conflicting option.

## Tests

- Tests mirror `src/mufinder/` (`tests/app`, `tests/services`, `tests/science`, ...).
  Shared factories and fixtures live in `tests/support/` and `tests/conftest.py`; do not
  define a local copy of a helper that exists there.
- Write behaviour tests through public entry points. Do not add `*_coverage.py` files;
  close a coverage gap in the sibling test file.
- Every feature covers its normal path, its validation and failure modes, and its
  persistence behaviour.
- `tests/test_agent_harness.py` enforces layer boundaries (including no Qt widgets outside
  `app`), the `MainWindow` handler, duplicate-method and coverage-file ratchets, the
  plot-canvas rules, the shared-homes table and this file's pointers; run `make harness`
  after touching docs indexes, package boundaries, a workflow widget or this file.

## Docs

- `docs/` is a contract, not a changelog. Describe current behaviour: no "previously", no
  migration narrative, no ordinal history, no `MainWindow._*` method names.
- Plans and reviews live in GitHub issues, not in `docs/`.
- Sphinx builds with `-W`; every new page goes in a toctree in `docs/index.rst`.

## Commands

The `Makefile` is the executable definition; run its targets, not the raw commands.

```bash
make install                          # dev + scientific-backends extras (or: uv sync --extra dev --extra scientific-backends)
make test-fast FILES="tests/app/test_convergence_widget.py"   # focused tests, no coverage
make test                             # full suite, coverage >= 85 %; foreground, at most 4 workers
make lint                             # ruff check and ruff format --check
make format                           # ruff format; layout is the tool's, never hand-tuned
make docs                             # sphinx -W
make harness                          # tests/test_agent_harness.py
```
