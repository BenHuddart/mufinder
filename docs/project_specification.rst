Project Specification Addendum
==============================

Purpose
-------

This addendum extends the MuFinder implementation specification with mandatory
requirements for testing, documentation, and quality automation.

Mandatory engineering requirements
----------------------------------

1. Every feature must include tests in ``tests/`` covering:

   - Normal execution path(s).
   - Input validation and failure modes.
   - Persistence behavior if the feature writes or reads project data.

2. Every feature must include documentation updates in ``docs/`` covering:

   - User-facing behavior.
   - Inputs, outputs, and constraints.
   - Any workflow dependencies introduced by the feature.

3. Changes are not considered complete unless both test and documentation
   gates pass in automation.

Definition of done
------------------

A feature is considered done only when all statements below are true:

- Code implementation is merged.
- Tests are present and passing.
- Coverage threshold is maintained.
- Sphinx documentation builds cleanly.
- Documentation reflects the final implemented behavior.

Startup Responsiveness Requirements
-----------------------------------

GUI startup must preserve prompt first-visible feedback:

- the splash screen must be shown as early as practical after ``QApplication`` setup
- heavyweight feature imports and non-critical startup initialization must occur only after splash exposure
- new startup functionality must preserve this ordering and must not introduce additional pre-splash blocking work

Governance
----------

This addendum is normative for all future development phases unless replaced
by a newer approved specification revision.

Calculation Job Architecture
----------------------------

The calculation job framework must follow these architectural constraints:

- Workflow expansion must produce resolved ``JobSpec`` objects.
- CASTEP input rendering must consume only resolved ``JobSpec`` objects.
- Workflow logic, template expansion, and variation handling must remain
   separate from CASTEP syntax rendering details.
- Validation and serialization must remain separate concerns.
- Default/override behavior must use one explicit precedence order in one
   implementation location.
- Future HPC execution metadata must accompany ``JobSpec`` objects and must
   not replace the scientific content represented by ``JobSpec``.

Local Execution Requirements
----------------------------

The local CASTEP execution path must satisfy the following requirements:

- Submission entry points must be available from both:

   - cutoff convergence table rows in ``ready`` state
   - shared job dashboard rows once jobs have entered the submission/queue lifecycle

- The cutoff convergence panel must expose a dedicated run action that persists
   the current convergence configuration and submits the selected cutoff rows, or every
   row when none is selected, replacing any separate pre-generation step. The action's
   label must state which of the two it will do.

- Workflow row tables for relaxation and convergence jobs must support
   removing the current row or selected rows via Delete and Backspace,
   not just via pointer-driven controls.

- Shared job dashboard rows must appear in the order their jobs were created,
  and that order must be the same after the project is closed and reopened.
  Changing a job's state, including resubmitting or resetting it, must not move
  its row.

- Shared job dashboard controls must be exposed as per-row action buttons,
  rather than requiring a context menu. A context menu may offer the same actions
  for the selection in addition.

- Text inside table rows must not change with the selection. Selection counts, hints
  and scope must be shown outside the rows so tables stay usable on a small screen.

- Every workflow job table must offer the same right-click row menu, in the same order,
  omitting only the actions that table does not have.

- Editing the parameters of several rows in one dialog must not change a parameter the
  user did not touch. Parameters whose values differ between the rows must be marked as
  such, must keep each row's own value unless edited, and must apply to every row once
  edited.

- An action on a selection that mixes rows it applies to with rows it does not must act
  on the applicable rows and report how many it left out, before the action where the
  control can show it (menu entries, prompts) and afterwards otherwise (the status bar
  after a submission). It must be unavailable only when no selected row qualifies.

- A control embedded in a table row must act on the whole selection when its
  row is part of a multi-row selection, after a prompt offering "all selected
  rows", "this row only" and "cancel". A control on a row outside the
  selection must first move the selection to that row and then act on it
  alone. Controls that are inherently single-row must not prompt.

- Local execution must run in background process management to avoid UI
   blocking.

- Multi-job submission must support sequential and parallel modes, with
   user-controlled concurrency.

- Job lifecycle state updates must be reflected in project job records and UI:

   - ready
   - queued
   - running
   - finished (completed, failed, cancelled)

- Configuring a cutoff sweep must persist the configuration snapshot without
  generating CASTEP input files until the user submits selected ready rows.

- Cutoff convergence rows with configured parameters must be shown in ``ready``
   state before submission, even when CASTEP input files have not yet been generated.

- Once a cutoff row is in ``ready`` (or any later runtime state), direct editing
   of that row's cutoff-energy value must be disabled.

- Completed jobs must leave the shared job dashboard after successful finish.

- Cancellation must be available for active local jobs.

- Removing a cutoff entry that already has associated files must warn the user and
   offer both:

    - permanent deletion of the associated files
    - moving those files into a separate archival subfolder outside the active workflow set

- Removing cutoff rows must ask one confirmation for the whole batch, stating how many
   rows have a submitted job and how many running jobs will be cancelled. When several
   removed rows have associated files, the file-handling prompt must offer
   ``Yes to All`` / ``No to All`` choices.

- Every action that removes, cancels, resets, dismisses or deletes must be confirmed
   before it takes effect, from every entry point, with the safe choice as the default.
   An action that covers several operations at once (Delete/Backspace on a mixed
   Job Dashboard selection) must present them in a single summary before any of them runs.

- Failed and cancelled job management must be available from the shared dashboard UI:

   - resubmit failed or cancelled jobs while reusing existing remote files so
     continuation runs can start from the same working directory
   - update the dashboard row with the replacement remote scheduler id after
     successful resubmission
   - inspect available ``*.err`` files for failed jobs
   - report contextual excerpts around failure lines from those ``*.err`` files
   - reset failed jobs to ``ready`` for rerun
   - optionally clean output files during reset while preserving generated input files

- Local execution configuration must include persisted MPI launcher path
   support (``mpirun``) with user path selection and discovery fallback.

- Output parsing must extract at least:

   - final energy, preferring ``NB est. 0K energy`` where available
   - maximum force magnitude
   - success/failure markers

- Cutoff convergence UI must display both final energy and maximum force
   per cutoff row when available from parsed output.

- The cutoff convergence UI must provide an examiner workflow that can:

   - evaluate convergence using multiple energy-analysis approaches
   - present user-facing method documentation in the GUI
   - persist analysis inputs and outputs so plots/results are restorable when the examiner is reopened

- Project-level defaults must be visible from the GUI, including cutoff-energy defaults.

- When a converged cutoff default is adopted from examiner output, persistence must include both:

   - the cutoff energy value
   - the approach/method metadata used to determine that cutoff

- Cutoff convergence energy/force columns must use sufficient numeric precision
   to allow comparison of small differences between converged rows.

- Status displays must use intuitive color cues for common states such as
  running, completed, failed, and cancelled.

- Closing a visible GUI session with an open project must prompt the user to
  save before exit.

Workflow Simulation-Cell Requirements
-------------------------------------

The workflow simulation-cell model must satisfy the following requirements:

- Workflow simulation-cell selection must be persisted independently per workflow stage.
- A workflow selection must support both:

   - host unit-cell mode
   - saved-supercell mode referencing a persisted derived structure snapshot

- Saved supercells must be persisted as normal ``Structure`` entities, not as a new entity type.
- Saved supercell structures must carry:

   - ``parent_structure_id`` pointing to the active host
   - ``properties["supercell_asset"]["matrix"]``
   - ``properties["supercell_asset"]["owner_workflow_stage"]``
   - ``properties["supercell_asset"]["source_asset_structure_id"]`` (nullable clone provenance)

- Cross-workflow supercell reuse must be clone-on-load:

   - selecting an asset owned by another workflow creates a current-workflow asset clone
   - clone creation may be deduplicated when an identical current-workflow host+matrix asset already exists

- Effective workflow input structure resolution must safely degrade:

   - invalid, missing, or host-mismatched asset selections must fall back to host unit cell
   - selection state normalization must be supported

- Convergence definition snapshots must freeze the resolved simulation cell:

   - resolved structure id stored in ``CalculationDefinition.input_structure_ids[0]``
   - mode/source provenance stored in calculation parameters

- Materialized workflow input files must render from resolved/frozen structure snapshots rather than from project-level supercell preview state.
- Existing ``Project.supercell_matrix`` persistence remains compatibility/read-only for host-preview defaults and must not govern workflow calculation structures.

- Legacy stale spin data handling must ensure spin-polarized workflows remain reproducible:

   - host spin extraction must fall back to magnetic moment vectors where needed
   - stale all-zero saved supercell spin arrays must be repairable from host magnetic data
