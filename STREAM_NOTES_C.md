# Stream C — Job-lifecycle hardening (recs #4, #8, and #2/#3 dashboard rendering)

Branch: `feature/jlh-stream-c` (off the foundation commits a278b2b / 940e365).

This file collects (a) the proposed documentation updates I was told NOT to
apply directly (docs are off-limits this round), and (b) the design decisions,
deviations, and cross-file consequences for the orchestrator/reviewer.

---

## 1. Summary of what landed

### REC #4 — Retry-retrieval action for failed/stopped cluster jobs
A failed or stopped cluster job that finished on the scheduler but whose output
retrieval/parse failed keeps its remote metadata. A new **Retry retrieval**
per-row dashboard action re-attempts retrieval from the same remote directory
without resubmitting a duplicate scheduler job.

### REC #8 — Dismiss action for terminal jobs
Failed / stopped / cancelled jobs can now be **Dismiss**ed from the active
dashboard. Dismissal is *not* a status change: it stamps `dismissed_at` and the
row is filtered out of the dashboard, while the job record, its runs, results,
and provenance all remain in the document.

### #2/#3 — Dashboard monitoring rendering
A new **Last Checked** column renders the relative age of `last_checked_at`
("just now" / "2 min ago" / "—" when None), with a tooltip carrying
`last_status_source` / `last_status_message`. Active cluster jobs whose cluster
is disconnected (or whose last check is older than the threshold) get a
" — stale" suffix in a warning colour and the staleness reason in the tooltip.
Staleness is computed via `services.job_monitoring.job_staleness` (not
reimplemented).

---

## 2. Exact new/changed public names + signatures

### `mufinder.services.project_service.ProjectService`
- `retry_job_retrieval(self, job: Job) -> None`
  Asserts FAILED/STOPPED → RETRIEVING via `_assert_transition`, sets
  `status=RETRIEVING`, clears `error_message`, preserves remote metadata. The
  failed/stopped-only precondition is enforced by the controller, NOT here (see
  §4 signal/transition note).
- `dismiss_job(self, job: Job) -> None`
  Requires status ∈ {FAILED, STOPPED, CANCELLED} and `dismissed_at is None`;
  stamps `dismissed_at = utc_now()`. Otherwise raises
  `ValueError("Only failed, stopped, or cancelled jobs can be dismissed.")`.
  Not a status transition.
- `record_job_dismissed(self, document, *, job) -> None`
  Appends a `ProvenanceRecord(action="job_dismissed", actor="user", …)`,
  mirroring `record_job_reset`'s shape.

### `mufinder.services.cluster_runner.ClusterJobRunner`
- `retrieve_outputs_for(self, job_id, *, remote_job_id, remote_directory,
  seed_name, local_output_dir, cluster_name="", retrieve_den_fmt=False,
  retrieve_out_cell=False, retrieve_phonon=False, retrieve_ts=False,
  task_farming_enabled=False, num_farms=None) -> None`
  (Re-)registers a `_TrackedClusterJob` from persisted metadata and runs one
  retrieval pass via the existing `_retrieve_results`. If not connected to the
  job's cluster it resolves to `failed` (see §4).

### `mufinder.app.cluster_controller.ClusterController`
- `retrieve_outputs_for(self, job_id, **kwargs) -> None`
  Async worker-thread dispatch (fire-and-forget, mirroring `refresh_queue`).

### `mufinder.services.job_execution_controller.JobExecutionController`
- `prepare_retry_retrieval(self, job_id: UUID) -> Job | None`
  Validates preconditions (failed/stopped, cluster target, remote id + dir
  present), calls `retry_job_retrieval`, stamps `last_checked_at=utc_now()`,
  `last_status_source="user"`, `last_status_message="Retry retrieval requested"`,
  and registers a **fresh** `CalculationRun` (status RETRIEVING) as the active
  run. Returns the job, or `None` when ineligible.

### `mufinder.app.main_window.MainWindow` (dashboard surface only)
- New column constants; jobs table is now **15 columns** (was 12): inserted
  `_JOB_LAST_CHECKED_COLUMN = 4` (after Status), `_JOB_RETRY_RETRIEVAL_COLUMN = 10`,
  and `_JOB_DISMISS_COLUMN = 14`.
- New handlers `_on_retry_retrieval_from_dashboard(job_ids)` /
  `_on_dismiss_from_dashboard(job_ids)` (deliberately NOT `_on_submit_*` /
  `_materialize_*`, following the cross-workflow dashboard-action precedent).
- Helpers: `_can_retry_retrieval(job)`, `_is_cluster_connected_for_job(job)`,
  `_cluster_retrieve_flags(job)` (static), `_format_relative_age(then, *, now)`
  (static), `_apply_last_checked_cell(item, job, *, now)`.

### Exact user-facing strings
- Dismiss guard: `"Only failed, stopped, or cancelled jobs can be dismissed."`
- Retry monitor message: `"Retry retrieval requested"` (`last_status_message`).
- Not-connected retry failure (runner): `"Cannot retry retrieval: not connected
  to the job's cluster."`
- Dashboard buttons: `"Retry retrieval"`, `"Dismiss"`.
- Status-bar: `"Retrying retrieval for N cluster job(s)"`,
  `"Dismissed N job(s) from the dashboard"`.
- `last_status_source` vocabulary added by this stream: only `"user"`.

---

## 3. Files touched
- `src/mufinder/services/project_service.py` — appended `retry_job_retrieval`,
  `dismiss_job`, `record_job_dismissed`. `_assert_transition`,
  `reconcile_orphaned_local_jobs`, and existing primitives untouched.
- `src/mufinder/services/cluster_runner.py` — appended `retrieve_outputs_for`.
  `_poll_queue`, `QueueCheckSnapshot`, `_TrackedClusterJob`, `submit`, and
  `_retrieve_results`' signature/behavior untouched (only called).
- `src/mufinder/app/cluster_controller.py` — appended `retrieve_outputs_for`.
- `src/mufinder/services/job_execution_controller.py` — appended
  `prepare_retry_retrieval`.
- `src/mufinder/app/main_window.py` — dashboard-table surface only (columns,
  `_refresh_jobs_table`, `_configure_jobs_table_columns`, dispatch + handlers,
  visibility filter, `_sync_cluster_tracking_from_document` refactor to share
  `_cluster_retrieve_flags`). Imports: added `UTC`, `QBrush`, and
  `services.job_monitoring.job_staleness`.
- `tests/services/test_project_service_transitions.py` — added
  `retry_job_retrieval` to `_PRIMITIVES` (did NOT add `dismiss_job`; see §4).
- `tests/app/test_main_window_actions.py` — appended retry/dismiss GUI tests.
- `tests/services/test_retry_retrieval.py` (new).
- `tests/services/test_job_dismissal.py` (new).
- `tests/app/test_dashboard_monitoring_columns.py` (new).
- `tests/app/test_main_window.py` — **one-line** cross-file consequence:
  `columnCount() == 12` → `15` (see §5).

---

## 4. Design decisions / deviations / discoveries

**Retry run bookkeeping — fresh RETRIEVING run.** The fail/stop path already
pops `_active_run_by_job` and marks the prior run terminal. Without a new active
run, `run_for_job` would fall back to that terminal FAILED/STOPPED run and the
eventual completion (`complete_job_with_result`) would overwrite it to COMPLETED.
`prepare_retry_retrieval` therefore calls `_create_active_run(document, job,
RETRIEVING)`, which increments `next_run_index` (a prior run_index=1 → the retry
run is run_index=2) and registers it as active. Verified against
`complete_job_with_result` / `fail_job` in `job_execution_controller.py`.

**Signal-legality of `_retrieve_results` from RETRIEVING.** Characterized
`_retrieve_results`' emitted sequence: `status_changed→RETRIEVING` first, then
one of `job_failed`+`FAILED` / `STOPPED` / `job_finished`+`COMPLETED`. The
pinned matrix allows RETRIEVING → {STOPPED, COMPLETED, FAILED, CANCELLED}, so
every branch is legal from the RETRIEVING state the primitive establishes. The
leading `RETRIEVING` re-emit is an inert no-op in
`mark_cluster_status_changed` (it only acts on `RUNNING→RETRIEVING`).

**Not-connected guard (prevents a new stuck-row class).** `_retrieve_results`
calls `_require_ssh()` *before* its try/except; `_require_ssh` raises
`ConnectionError` when disconnected. On the fire-and-forget worker thread that
raise would reach no GUI signal and strand the job (plus its fresh RETRIEVING
run) forever — exactly the failure class this initiative kills. So
`retrieve_outputs_for` pre-checks `is_connected` + cluster-match and, belt-and-
suspenders, wraps the `_retrieve_results` call, converting either to
`job_failed`+`status_changed(FAILED)` so `fail_job` resolves both job and run.

**`retry_job_retrieval` guards via `_assert_transition` ONLY.** It's in
`_PRIMITIVES`, whose generic test picks `allowed_source` = RUNNING (a legal
RETRIEVING source) and asserts the primitive succeeds there and raises the exact
`"Invalid job transition: …"` message elsewhere. A bespoke failed/stopped-only
`ValueError` inside the primitive would break both branches, so that
precondition lives in `JobExecutionController.prepare_retry_retrieval` and the
button eligibility instead. This matches the task wording ("asserts the
transition to RETRIEVING").

**`dismiss_job` is NOT in `_PRIMITIVES`.** It is not a status transition, so the
generic test's `assert job.status == target` cannot apply. Its guards +
idempotence live only in `test_job_dismissal.py`.

**Retry eligibility is a predicate, not a status set.** The dispatcher keyed
eligibility off a per-kind status set. Retry's real rule is failed/stopped AND
`remote_job_id` AND `remote_directory` AND connected-to-this-job's-cluster, so
`_dispatch_dashboard_action`'s `eligibility()` special-cases `retry_retrieval`
to `_can_retry_retrieval(job)`. `_is_cluster_connected_for_job` (casefold
comparison mirroring `_submit_jobs_on_cluster`) is reused in three places: the
button `enabled=`, the dispatcher predicate, and the `cluster_connected=`
argument to `job_staleness`.

**Stale marker is computed on dashboard rebuild, not by the runtime timer.**
`_refresh_active_job_runtime_cells` only ticks when a RUNNING/RETRIEVING job is
present AND the dock is visible, and stops otherwise. A stale job is typically
QUEUED-on-a-disconnected-cluster — precisely when that timer is stopped — so the
timer is NOT a reliable driver for staleness. The Last-checked cell (age +
stale marker) is therefore populated on each `_refresh_jobs_table` rebuild
(post-poll flush + debounce timer already rebuild the table). I deliberately did
NOT touch `_refresh_active_job_runtime_cells`; aging Last-checked there would be
a harmless bonus for already-ticking rows but is not the mechanism, and adding
it risked conflicting with that method's live-runtime bookkeeping.

**Shared retrieve-flags helper.** The per-calc-type `retrieve_den_fmt` /
`retrieve_out_cell` / `retrieve_phonon` / `retrieve_ts` flags were inline in
`_sync_cluster_tracking_from_document`. Extracted to the static
`_cluster_retrieve_flags(job)` so cluster re-attachment and retry retrieval
download identical optional outputs (drift here would silently skip
`.den_fmt`/`-out.cell`/`.phonon`/`.ts` on retry).

**No new keyboard shortcuts.** Both new actions are buttons only, sidestepping
the macOS Delete/Backspace dual-binding rule. The existing dashboard
Delete-key handler (`_on_dashboard_delete_key_requested`) was left unchanged and
does NOT route to dismiss.

**No un-dismiss UI (deliberate).** There is no "restore/undismiss" action this
pass. A dismissed job keeps all its data (record, runs, results, provenance) and
only `dismissed_at` is set, so it is recoverable **only at the data level**
(clearing `dismissed_at`). This is intentional per the task; flag for a future
pass if an un-dismiss affordance is wanted.

---

## 5. Cross-file consequence (acknowledged, not silently ignored)

Adding the Last-checked column + two action buttons necessarily changes the
jobs-table `columnCount` from 12 to 15. `tests/app/test_main_window.py`
(`test_main_window_jobs_table_includes_cluster_columns`, line ~2547) hard-asserts
`columnCount() == 12`. That file is outside my enumerated working set but is NOT
in the HARD-RULES prohibition list, and "tests pass" is a non-negotiable gate;
dashboard rendering for #2/#3 is explicitly my scope, and the column count *is*
dashboard rendering (other streams provide the data layer, not table columns).
I changed exactly that **one integer** (`12` → `15`) and nothing else in the
file, to minimize merge surface with streams A/B. Every other assertion in that
test survives (header `in` membership checks; `sectionResizeMode` checks on
symbolic constants that still exist). Orchestrator: this is the only edit outside
my working set.

---

## 6. Proposed documentation updates (apply after merge — docs off-limits this round)

### `docs/job_lifecycle.rst` — "Main-window integration" dashboard actions list
Replace the actions bullet list under "The dashboard displays active states plus
``failed``, ``stopped``, and ``cancelled``. Actions are enabled as follows:" with:

    - ``Resubmit``: ``failed`` and ``cancelled``.
    - ``Retry retrieval``: ``failed`` and ``stopped`` cluster jobs that still
      carry both a ``remote_job_id`` and ``remote_directory``, while the cluster
      runner is connected to that job's cluster. It re-attempts output retrieval
      from the existing remote directory (via
      ``ClusterJobRunner.retrieve_outputs_for``) instead of resubmitting a
      duplicate scheduler job, moving the job back to ``retrieving`` under a
      fresh ``CalculationRun``.
    - ``Cancel``: ``submitting``, ``queued``, ``running``, and ``retrieving``.
    - ``Inspect``: ``failed`` and ``stopped``.
    - ``Reset``: ``failed``, ``stopped``, and ``cancelled``.
    - ``Dismiss``: ``failed``, ``stopped``, and ``cancelled``. Dismissal hides
      the row from the active dashboard by stamping ``dismissed_at``; it is not a
      status transition, so the job record, runs, results, and provenance all
      remain in the document. There is no un-dismiss UI — a dismissed job is
      recoverable only by clearing ``dismissed_at`` at the data level.

Add to the same section (dashboard columns): a ``Last Checked`` column shows the
relative age of ``last_checked_at`` with a stale marker for active cluster jobs
whose cluster is disconnected or whose last check is older than the
``job_monitoring`` threshold, and a tooltip carrying ``last_status_source`` /
``last_status_message`` / the staleness reason.

Also add a bullet to the ``ProjectService`` primitives list:

    - ``retry_job_retrieval`` moves a ``failed`` or ``stopped`` job back to
      ``retrieving`` (the FAILED/STOPPED → RETRIEVING edges), clears
      ``error_message``, and preserves remote metadata. The failed/stopped-only
      and remote-metadata preconditions are enforced by
      ``JobExecutionController.prepare_retry_retrieval``, not the primitive.
    - ``dismiss_job`` stamps ``dismissed_at`` for a ``failed``/``stopped``/
      ``cancelled`` job (not a status transition; rejects already-dismissed
      jobs), and ``record_job_dismissed`` appends a ``job_dismissed`` provenance
      milestone.

### `docs/job_lifecycle.rst` — Recommendations #4 and #8
Mark both **(implemented)**:

    4. Add a retry-retrieval action for stopped/failed cluster jobs
       (implemented). A ``Retry retrieval`` dashboard action re-attempts output
       retrieval for a scheduler-completed cluster job that kept its remote
       metadata, avoiding an unnecessary duplicate scheduler job. It routes
       through ``JobExecutionController.prepare_retry_retrieval`` →
       ``ProjectService.retry_job_retrieval`` (FAILED/STOPPED → RETRIEVING) and
       ``ClusterJobRunner.retrieve_outputs_for``, which resolves a
       not-connected cluster to ``failed`` rather than stranding the job in
       ``retrieving``.

    8. Separate terminal history from the active dashboard (implemented). A
       ``Dismiss`` action stamps ``dismissed_at`` on a failed/stopped/cancelled
       job so ``_refresh_jobs_table`` filters it out of the active dashboard,
       while keeping the job record, runs, results, and provenance intact (a
       ``job_dismissed`` milestone is recorded). Recommendations #2/#3 monitor
       metadata are surfaced by the ``Last Checked`` column and stale marker.

### `docs/implemented_functionality.rst` — dashboard section (~lines 719-766)
Update the actions bullet (line ~720) and the "still need attention" paragraph:

    - Failed/cancelled-job resubmit, failed/stopped cluster-job retry-retrieval,
      failed-job inspection, reset, and terminal-job dismiss actions in the
      shared job dashboard.

Add to the runtime-behavior list:

    - A ``Last Checked`` dashboard column showing the relative age of the last
      queue check, with a stale marker (and tooltip reason) for active cluster
      jobs whose cluster is disconnected or whose last check is overdue.
    - Retrying output retrieval for a failed/stopped cluster job from the
      dashboard, reusing the existing remote directory without resubmitting a
      duplicate scheduler job.
    - Dismissing failed/stopped/cancelled jobs from the active dashboard
      (stamping ``dismissed_at``) while preserving their record, runs, results,
      and provenance; dismissed jobs are recoverable only at the data level.

In the closing paragraph, append after "…so they can be inspected, resubmitted,
or reset.": "Failed or stopped cluster jobs that kept their remote metadata can
also retry retrieval, and any terminal job can be dismissed to remove its row
from the active dashboard without deleting its data."

---

## 7. Test commands run (all green)

    PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov \
      tests/services/test_project_service_transitions.py \
      tests/services/test_retry_retrieval.py            # 112 passed

    PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov \
      tests/services/test_job_dismissal.py              # 11 passed

    PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov \
      tests/app/test_main_window_actions.py -k "retry_retrieval or dismiss"   # 4 passed

    PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov \
      tests/app/test_dashboard_monitoring_columns.py    # 4 passed

    PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov \
      tests/test_agent_harness.py                       # 6 passed

    # Broad focused sweep (regression guard for the column shift):
    PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov -n 4 --dist load \
      tests/app/test_main_window_actions.py tests/app/test_main_window.py \
      tests/app/test_dashboard_monitoring_columns.py tests/services/test_cluster_runner.py \
      tests/services/test_job_execution_controller.py \
      tests/services/test_project_service_transitions.py \
      tests/services/test_retry_retrieval.py tests/services/test_job_dismissal.py \
      tests/app/test_orphaned_local_job_load_path.py    # 441 passed
