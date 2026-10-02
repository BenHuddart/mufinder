# Multi-row selection: design and implementation plan

Status: design agreed 2026-10-02. All five phases implemented.

This plan makes actions on multi-row selections behave the same way in every
job table (Pristine Relaxation, Muon Relaxation, UEP, Cutoff and K-point
Convergence, Energy Barriers, Quantum Corrections) and in the Job Dashboard,
and makes that behaviour visible before the user clicks.

## Constraints

- Text inside table rows must not grow or change with the selection. The app
  has to stay usable on a 13-inch laptop, and several tables size columns to
  their contents. Counts, hints and state go outside the rows: styling,
  tooltips, a line below the table, the status bar, dialogs and menus.
- The Job Dashboard keeps its six action-button columns.
- All changes live in `mufinder.app`. No new `_on_submit_*` / `_materialize_*`
  handlers on `MainWindow` (ratcheted by `tests/test_agent_harness.py`).
- Every delete/remove/cancel/reset shortcut keeps both the Delete and
  Backspace bindings (`bind_delete_shortcuts`).

## Agreed behaviour

### 1. Targeting rule for in-row controls

Highlighted rows are what is affected.

- A control on a row that is part of a multi-selection targets the selection.
- A control on an unselected row first replaces the selection with that row,
  then acts on it alone (left-click now matches the existing right-click
  behaviour).
- Controls that only make sense for one row (View, Open Wizard, Continue
  relaxation) always act on their own row and never prompt. The UEP
  "open the wizard for just this row?" question is removed.
- Every value-setting control propagates. This adds Fix Cell in Pristine
  Relaxation, which today never propagates.
- After the action the selection is restored, so the highlight still matches
  what was changed (Windows collapses the selection when a cell widget is
  clicked).

### 2. The apply-to-selection prompt

The prompt stays, with three buttons:

| Button | Effect |
|---|---|
| All N selected rows | apply to every eligible selected row |
| This row only | apply to the clicked row |
| Cancel | do nothing; a combo returns to its previous value |

One wording template is used everywhere. It names the action and the count,
and reports ineligible rows: "Set supercell on 3 of 5 selected rows? 2 are
locked."

### 3. Destructive actions

Remove, cancel, reset, dismiss and delete always ask for confirmation, from
every entry point (in-row control, button below the table, Delete/Backspace,
right-click menu), including rows that were never submitted. The confirmation
states the count and what will be lost or cancelled.

In the Job Dashboard, Delete/Backspace on a selection shows one summary dialog
before anything happens:

```
Apply to 6 selected jobs?

  • Cancel 2 running jobs
  • Reset 1 failed job to ready
  • Delete 3 finished job records and their linked results

  [ ] Also clean the reset job's output directory

                              [ Cancel ]  [ Apply ]
```

Only the lines that apply to the selection are shown. Today running jobs are
cancelled with no confirmation before the first dialog appears.

### 4. Discoverability

- **Accent style.** While two or more rows are selected, the propagating
  controls in those rows get an accent style. Single-row controls stay plain.
- **Hint line.** Workflow tables show the count in their existing button row
  ("3 selected"). The dashboard shows a slim line only during multi-selection:
  "3 selected · row buttons apply to all selected".
- **Status bar.** After a multi-row action: "Supercell set on 3 rows, 1 locked
  row skipped".
- **Button row below each workflow table** has the same set: Configure
  Selected, Remove Selected, Run Local, Run on Cluster. Quantum Corrections
  omits Remove because its rows are derived from completed relaxations.
- **One right-click menu** on every job table, in this order: Run Local, Run
  on Cluster, Cancel, Inspect Failure, Reset, Configure…, Remove, then a
  separator and table-specific items (Continue relaxation…). Items that do
  not exist for a table are omitted. The dashboard gets a menu with Resubmit,
  Retry retrieval, Cancel, Inspect, Reset, Dismiss and Delete. No bulk button
  bar is added to the dashboard.

### 5. Run buttons

The label follows the selection: "Run All Locally" / "Run All on Cluster" with
nothing selected, "Run Selected Locally" / "Run Selected on Cluster"
otherwise. This replaces three current behaviours (run all, run the invisible
"current" row, disabled). The convergence tables' current-row fallback is
removed. A local "Run All" starts without an extra confirmation.

### 6. Mixed selections

Every action acts on the rows it applies to and reports how many were skipped.

- Prompts and confirmations state "3 of 5".
- Menu items are enabled when at least one selected row qualifies and show the
  count when partial: "Reset (2 of 5)".
- Run skips rows that are already running or complete and reports it in the
  status bar.
- An item is greyed out only when no selected row qualifies.

This replaces the all-or-nothing gate on right-click Reset
(`selection_supports_reset`).

### 7. Batch Configure

Today the dialog is seeded only with values the rows share, then its full
contents replace every row's own settings, so per-row differences are lost
without warning. New behaviour:

- Parameters that differ between the rows are shown as "mixed".
- On OK, only fields the user edited are written to the rows. Untouched
  fields keep their per-row values.
- Editing a mixed field clears its marker and writes the new value to all
  rows.

## Implementation phases

Each phase is a separate PR into `main` with its own tests and doc updates.

### Phase 1: targeting rule and prompt

- `app/selection_snapshot.py`
  - `resolve_target_keys` returns an empty tuple for Cancel and uses a
    three-button dialog built from one wording template (action label,
    eligible count, skipped count).
  - Snapshot widgets select their own row on press when it is not in the
    selection.
  - Add a helper to restore the snapshot selection after the action.
- Callers to update for the empty return and shared wording:
  `relaxation_widget.py`, `muon_relaxation_widget.py`, `uep_widget.py`,
  `convergence_widget.py`, `kpoint_convergence_widget.py`,
  `energy_barriers_widget.py`, `quantum_corrections_widget.py`,
  `main_window.py` (`_dispatch_dashboard_action`).
- Combos: on Cancel, refresh the row from stored state so the combo shows its
  previous value.
- `relaxation_widget.py`: Fix Cell becomes a `SelectionSnapshotComboBox` and
  propagates. The combo records the index it showed at press time, which is
  what Cancel restores.
- `uep_widget.py`: remove the wizard confirmation.
- Tests: update the `resolve_target_keys` tests for the three outcomes; add
  per-table tests for select-on-click, Cancel restoring a combo, and Fix Cell
  propagation.

### Phase 2: destructive confirmations

- Add a shared confirmation helper in `app/workflow_table_utils.py` (count,
  consequence text, Cancel default).
- Route Delete/Backspace, "Remove Selected" buttons and in-row Remove through
  it in all workflow tables. Generalise the cutoff table's
  running/progressed wording to the relaxation, UEP and Energy Barriers
  tables.
- `main_window.py`: replace `_on_dashboard_delete_key_requested` with the
  summary dialog. Split `_cancel_jobs`, `_reset_failed_jobs` and
  `_delete_dashboard_jobs` so their prompt-free cores can run after a single
  confirmation (`_apply_failed_job_reset` already exists for reset).
- Tests: one per entry point per table; dashboard summary dialog for mixed,
  single-kind and cancelled cases; Delete and Backspace both covered.

### Phase 3: discoverability, shared menu, Run labels

- `app/theme.py`: accent style driven by a dynamic property on cell widgets.
- `app/workflow_table_utils.py`
  - Helper that updates the accent property and hint text on selection
    change.
  - Generalise `open_reset_failed_context_menu` / `RowMenuAction` into the
    shared core menu; keep `RESET_FAILED_TABLE_TOOLTIP` wording in step.
- Each workflow widget: standard button row, count label, Run labels that
  follow the selection, shared menu wiring. Remove
  `_selected_or_current_cutoffs` / `_selected_or_current_grids` fallbacks.
- `main_window.py`: the dashboard dock's widget becomes a container holding
  the table and the hint line; enable a custom context menu that dispatches
  to the existing `_on_*_from_dashboard` handlers. Check tests that assume
  the dock's widget is the table.
- Tests: hint visibility and text, accent property on selected rows only,
  menu contents and order per table, Run label for empty and non-empty
  selections.

### Phase 4: mixed selections

- Replace `selection_supports_reset` with an eligible-subset calculation that
  returns eligible keys and a skipped count; rows without a key count as
  skipped.
- Apply the same calculation to every menu item, Run button and prompt.
- Status-bar reporting of skipped rows in the workflow controllers'
  submission paths.
- Tests: partial eligibility for each action; none-eligible greys out.

### Phase 5: mixed markers in the parameter dialog

- `app/param_dialog.py`, `app/param_form.py`, `app/param_widgets.py`: a
  per-field mixed state, cleared on edit; `collect_values` gains a way to
  return only edited fields.
- Each widget's `_common_params_for_rows` / `_open_*_parameter_dialog`: pass
  the set of differing parameter names, and merge edited fields into each
  row instead of replacing the row's mapping.
- Points needing care:
  - the mutually exclusive `kpoint_mp_grid` / `kpoint_mp_spacing` pair, where
    rows may differ in which one is active (read
    `docs/castep/kpoint_spacing_convention.md` first);
  - advanced keywords present on only some rows;
  - imported and locked rows, which are already excluded or read-only.
- Tests: differing values survive an unrelated edit; editing a mixed field
  writes to all rows; grid/spacing mixtures.

## Documentation to update

- `docs/implemented_functionality.rst`: multi-row behaviour, Run labels,
  shared menu.
- `docs/project_specification.rst`: the Delete/Backspace contract (currently
  line 84) and the confirmation requirement.
- `docs/job_lifecycle.rst`: reset from the row menu on mixed selections,
  dashboard summary dialog.
- Per-workflow pages that describe their tables (`uep_workflow.rst`,
  `energy_barriers.rst`, `quantum_corrections.rst`, `magnetic_fields.rst`).
- `docs/architecture.rst` if the shared menu helper changes the controller
  wiring contract.

## Current behaviour, for reference

| Table | In-row controls that propagate | That don't | Run with nothing selected | Right-click menu | Delete key |
|---|---|---|---|---|---|
| Job Dashboard | all six buttons | — | n/a | none | cancel + reset + delete in one keypress |
| Pristine Relaxation | Supercell, Configure, Remove | Fix Cell | all rows | Reset, Continue | removes, no confirm |
| Muon Relaxation | Supercell, Configure, Remove | — | all rows | Reset, Continue | removes, no confirm |
| UEP | Supercell, Configure, Remove | Open Wizard | all rows | Reset | removes, no confirm |
| Cutoff convergence | Configure, Remove | — | current or only row | Submit, Cancel, Inspect, Reset | confirms active/progressed |
| K-point convergence | Configure, Remove | — | current or only row | Reset | removes |
| Energy Barriers | Configure | View | disabled | Reset | removes, no confirm |
| Quantum Corrections | Configure | — | disabled | none | not bound |
