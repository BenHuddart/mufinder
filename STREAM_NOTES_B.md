# Stream B — Job-lifecycle hardening (recommendations #6 and #7)

Branch: `feature/jlh-stream-b` (off the foundation commits `a278b2b` + `940e365`).

This file holds the **proposed** documentation edits for Stream B. Per the
stream rules, `docs/` was not edited directly; the maintainer should fold the
text below into the named files.

---

## 1. Summary of code changes

### REC #6 — fail fast on local process start errors
- `src/mufinder/services/local_runner.py`
  - `_on_process_error`: when the error is `QProcess.ProcessError.FailedToStart`,
    the runner now reports the failure immediately instead of waiting for a
    `finished` event Qt may never deliver. It pops the active entry, discards any
    pending cancel marker, calls `process.deleteLater()`, emits
    `job_failed(job_id, message)` + `status_changed(job_id, "failed")`, and starts
    the next queued job. Other `ProcessError` values keep the store-and-wait
    behaviour (the finish handler reports the outcome).
  - Because the active entry is popped, a late `finished` event for the same job
    hits the existing `active is None` early-return and is a no-op — no new
    double-report guard was needed.
  - New introspection accessors for the watchdog: `active_job_ids()` and
    `pending_job_ids()` (both `-> frozenset[UUID]`).

### REC #7 — active-job watchdog
- `src/mufinder/services/job_watchdog.py` (new, Qt-free):
  `detect_stuck_jobs(...) -> list[tuple[Job, str]]` and the two-tick
  `StuckJobWatchdog` wrapper.
- `src/mufinder/app/cluster_controller.py`: GUI-thread
  `tracked_active_job_ids() -> frozenset[UUID]` mirror.
- `src/mufinder/services/job_execution_controller.py`: appended
  `stop_stuck_job(job_id, message) -> LifecycleUpdate`.
- `src/mufinder/app/main_window.py`: 60 s `QTimer` + one handler
  `_on_job_watchdog_tick`.

### Exact new/changed public names and signatures
```
LocalJobRunner.active_job_ids() -> frozenset[UUID]
LocalJobRunner.pending_job_ids() -> frozenset[UUID]

ClusterController.tracked_active_job_ids() -> frozenset[UUID]

job_watchdog.detect_stuck_jobs(
    jobs: Iterable[Job],
    *,
    local_active_ids: frozenset[UUID],
    local_pending_ids: frozenset[UUID],
    cluster_tracked_ids: frozenset[UUID],
    cluster_connected: bool,
    now: datetime,
    connected_cluster_name: str | None = None,
    grace_seconds: int = DEFAULT_GRACE_SECONDS,   # 120
) -> list[tuple[Job, str]]

job_watchdog.StuckJobWatchdog(grace_seconds=DEFAULT_GRACE_SECONDS)
  .evaluate(jobs, *, local_active_ids, local_pending_ids, cluster_tracked_ids,
            cluster_connected, now, connected_cluster_name=None)
      -> list[tuple[Job, str]]
  .forget(job_id: UUID) -> None
  .reset() -> None

JobExecutionController.stop_stuck_job(job_id: UUID, message: str) -> LifecycleUpdate
```

### Exact message strings
- REC #6 (local process failed to start):
  `The CASTEP process failed to start: {detail}. Check the CASTEP executable path and file permissions.`
  where `{detail}` is `process.errorString()` when non-empty, else the enum name
  (`FailedToStart`).
- REC #7 local finding:
  `MuFinder's records say this job is {status}, but no local process is tracking it. The job watchdog stopped it; reset or resubmit it.`
- REC #7 cluster finding:
  `MuFinder's records say this job is {status}, but the connected cluster session is not tracking it. The job watchdog stopped it; reset or resubmit it.`
- REC #7 status-bar summary (shown when N > 0 jobs were stopped):
  `N job(s) were stopped by the job watchdog because no runner was tracking them.`

`last_status_source` for watchdog stops is the exact string `"watchdog"`.

### Files touched
- `src/mufinder/services/local_runner.py`
- `src/mufinder/services/job_watchdog.py` (new)
- `src/mufinder/services/job_execution_controller.py` (appended one method)
- `src/mufinder/app/cluster_controller.py`
- `src/mufinder/app/main_window.py` (timer block in `__init__`, one handler,
  one `closeEvent` timer stop)
- `tests/services/test_local_runner.py` (appended)
- `tests/services/test_job_watchdog.py` (new)
- `tests/app/test_job_watchdog_wiring.py` (new)
- `STREAM_NOTES_B.md` (this file)

---

## 2. Deviations and discoveries

### Cluster mirror signal coverage
The `ClusterController` mirror (`tracked_active_job_ids`) is maintained entirely
from signals/calls the controller already sees on the GUI thread:
- **Add**: any `status_changed(job_id, value)` where `value` is an active status
  (`submitting`/`queued`/`running`/`retrieving`). This covers `submit()` (emits
  `status_changed("queued")`) and every poll-loop `queued`/`running` transition.
- **Add**: `track_existing_job(job_id=..., status=<active>)` — the runner emits
  **no** signal on re-track, so the mirror is updated eagerly on the GUI thread
  here. Terminal statuses passed to `track_existing_job` are not added (the
  runner marks them terminal and never polls them).
- **Remove**: any `status_changed(job_id, value)` where `value` is terminal
  (`completed`/`failed`/`stopped`/`cancelled`). Verified against every emission
  site in `cluster_runner.py`: `cancel`, and each terminal branch of
  `_poll_queue` / `_retrieve_results`, all emit a terminal `status_changed`.
- **Clear**: `clear_tracked_jobs()` (project switch) — the only place the runner
  drops all tracked jobs.

**Gap / deliberate deviation — the mirror is NOT cleared on disconnect.**
The runner's `_tracked_jobs` survive a disconnect (`disconnect()` clears only
the SSH transport/scheduler/config, not the tracked-job dict), and an automatic
reconnect (`_on_cluster_connection_status_changed`) does **not** call
`_sync_cluster_tracking_from_document` (only the manual *Refresh Queue* path
does). If the mirror were cleared on disconnect, then ~2 minutes after any
reconnect the watchdog would see connected + active cluster job + empty mirror
and stop a live job — and two-tick confirmation would not save it, because an
empty mirror persists across ticks. The mirror therefore faithfully tracks the
runner's `_tracked_jobs` lifecycle (add/remove/clear) and is left intact across
disconnects. This is safe because the watchdog never flags cluster jobs while
disconnected (`cluster_connected=False` short-circuits the cluster rule).

### Cluster rule uses the connected cluster NAME, not a bare bool
The recommendation sketch named a `cluster_connected: bool`. That bool is kept
(it is the disconnect short-circuit), but the cluster rule additionally takes
`connected_cluster_name` and only flags a cluster job when the connected cluster
is the one *that job* targets — mirroring
`ClusterJobRunner._tracked_job_matches_connected_cluster` and the semantic
already pinned by `services.job_monitoring.job_staleness` (blank job target or
blank connected name = wildcard, case-insensitive compare). This prevents a
false positive when connected to cluster A while a tracked job targets cluster B.

### Two-tick confirmation lives in `job_watchdog.StuckJobWatchdog`
State is kept in the Qt-free service helper, not on `MainWindow`, so it is unit
testable without a window. `MainWindow` holds one instance and calls `evaluate`
each tick; it calls `forget(job_id)` after a successful stop and `reset()` when
there are no active jobs (early-return path).

### RETRIEVING local jobs are never flagged
Local runs never enter `retrieving` (only cluster retrieval does), so there is
no local-runner condition to check; the rule set intentionally omits it.

---

## 3. Tests run and results

All run as focused suites with:
`PYTHONPATH=<worktree>/src .venv/bin/python -m pytest --no-cov <files>`

- `tests/services/test_local_runner.py` — 20 passed (REC #6 fail-fast x3,
  non-fatal store-and-wait, id accessors, plus pre-existing).
- `tests/services/test_job_watchdog.py` — 28 passed (detection matrix, grace
  period, cluster disconnect/other-cluster, two-tick + recovery + forget/reset,
  controller stop path).
- `tests/app/test_job_watchdog_wiring.py` — 10 passed (cluster mirror add/remove/
  clear/disconnect-survival/track_existing_job).
- Regression suites: `tests/app/test_cluster_controller.py`,
  `tests/services/test_job_execution_controller.py`,
  `tests/services/test_cluster_runner.py`, `tests/services/test_job_monitoring.py`
  — 81 passed.
- `tests/app/test_main_window.py`, `tests/app/test_cluster_tracking_reattachment.py`
  — 58 passed (theme-teardown RuntimeWarnings are pre-existing, unrelated).
- `tests/test_agent_harness.py` — 6 passed (architecture boundaries + MainWindow
  handler ratchet unaffected).

---

## 4. Proposed documentation edits

### 4a. `docs/job_lifecycle.rst` — "Local runner behavior" subsection
Add a bullet to the existing "Local runner behavior" list (after the
non-zero-exit / parser-failure bullet) describing the fail-fast path:

> - A ``QProcess`` ``FailedToStart`` error (for example a missing or
>   non-executable CASTEP binary) is reported immediately as ``job_failed`` and
>   ``failed`` rather than waiting for a ``finished`` event that Qt may never
>   deliver. The active entry is cleaned up and the next queued job starts, so a
>   job cannot remain ``submitting`` behind a process that never launched. The
>   failure message is *"The CASTEP process failed to start: {detail}. Check the
>   CASTEP executable path and file permissions."* Other ``QProcess`` errors keep
>   the store-and-wait behaviour and are reported by the finish handler.

### 4b. `docs/job_lifecycle.rst` — new "Active-job watchdog" subsection
Add after "Main-window integration":

> Active-job watchdog
> ~~~~~~~~~~~~~~~~~~~~
>
> Orphaned-local reconciliation (recommendation 1) only covers the app-restart
> case. A running session can still leave a job active with nothing executing it:
> a local ``QProcess`` that died without a finish event, a cluster job the runner
> silently stopped tracking, or a queue entry that never reached a runner. A
> once-a-minute watchdog closes this gap.
>
> The detection core, ``mufinder.services.job_watchdog.detect_stuck_jobs``, is a
> pure, Qt-free function. Given the project's jobs plus runner-introspection
> snapshots, it flags:
>
> - a local ``queued`` job absent from both the local runner's pending and
>   active id sets;
> - a local ``running`` job absent from the local runner's active id set;
> - a local ``submitting`` job older than a 120-second grace period (by
>   ``submitted_at``; skipped when ``submitted_at`` is ``None``) and absent from
>   the pending and active id sets;
> - a cluster job in an active state whose id is not in the cluster controller's
>   tracked-id mirror, **only** when the connected cluster is the one that job
>   targets. When the cluster is disconnected, cluster jobs are never flagged —
>   that is the staleness marker's responsibility (recommendations 2–3), not the
>   watchdog's.
>
> Runner introspection is exposed by ``LocalJobRunner.active_job_ids()`` /
> ``pending_job_ids()`` and, on the GUI thread, by
> ``ClusterController.tracked_active_job_ids()`` — a mirror of the runner's
> non-terminal ``_tracked_jobs`` maintained from the ``status_changed`` stream
> plus ``track_existing_job``/``clear_tracked_jobs``. The mirror is deliberately
> retained across a disconnect (the runner keeps its tracked jobs across a
> disconnect, and the watchdog never flags cluster jobs while disconnected), so
> reconnecting cannot trigger a false positive.
>
> To ride out an in-flight runner signal that has not yet reached the document, a
> finding is acted on only when it is confirmed in **two consecutive**
> evaluations (``StuckJobWatchdog``). A confirmed finding is applied by
> ``JobExecutionController.stop_stuck_job``, which routes the transition through
> ``ProjectService.stop_job`` (the single authority for the ``-> stopped`` edge),
> closes out the job's active ``CalculationRun``, records a ``job_stopped``
> journal event, and stamps the monitor bookkeeping fields
> (``last_checked_at``, ``last_status_source = "watchdog"``,
> ``last_status_message``). Stopped rows remain visible and resettable, exactly
> like a walltime stop.
>
> ``MainWindow`` owns the 60-second ``QTimer`` (services must not create widgets)
> and a single handler that snapshots the runners, runs the watchdog, applies
> ``stop_stuck_job`` per confirmed finding, requests a coalesced dashboard flush,
> and shows *"N job(s) were stopped by the job watchdog because no runner was
> tracking them."* when any were stopped. The handler is cheap when no jobs are
> active.

### 4c. `docs/job_lifecycle.rst` — recommendations #6 and #7
Mark both as implemented, mirroring the "(implemented)" style of #1:

> 6. Fail fast on local process start errors (implemented).
>    ``LocalJobRunner._on_process_error`` now handles
>    ``QProcess.ProcessError.FailedToStart`` directly: it emits ``job_failed`` and
>    ``failed``, cleans up the active entry, and starts the next queued job, so a
>    job cannot remain ``submitting`` when Qt does not deliver a finish event. A
>    late ``finished`` event for that job is a no-op. Other ``ProcessError``
>    values keep the store-and-wait behaviour.

> 7. Add an active-job watchdog (implemented).
>    A once-a-minute watchdog (``mufinder.services.job_watchdog``) detects active
>    jobs with no runner tracking them — ``queued``/``running``/``submitting``
>    local jobs missing from the local runner's pending/active sets, and active
>    cluster jobs missing from the cluster controller's tracked-id mirror while
>    their cluster is connected — and, after two-tick confirmation, stops them via
>    ``JobExecutionController.stop_stuck_job`` (through ``ProjectService.stop_job``)
>    with a diagnostic message and ``last_status_source = "watchdog"``. See
>    "Active-job watchdog" above.

### 4d. `docs/implemented_functionality.rst`
In the dashboard paragraph (around the existing reconciliation sentence, ~line
761), append:

> During a running session, an active-job watchdog additionally stops jobs that
> are left active with no runner tracking them (a dead local process, an
> untracked active cluster job) and reports how many were stopped in the status
> bar; those rows also remain visible and resettable.

And in the local-execution status-transition list (~line 774), add:

> - Local jobs whose CASTEP process fails to start are failed immediately with a
>   diagnostic message, rather than remaining stuck in ``submitting``.
