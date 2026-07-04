# Stream A notes — job-lifecycle hardening (recs #2, #3, #5)

Implementation notes and **proposed documentation updates** for the polling /
stamping / grace-window work. Docs under `docs/` are deliberately untouched in
this stream; the orchestrator should fold the text below into the named
sections.

## Summary of behavior changes

1. **Rec #2 — persisted monitor metadata.**
   - `QueueCheckSnapshot` gained `observations_by_job_id: dict[UUID, str]`, a
     short human-readable statement of what the scheduler reported for each
     polled job that cycle. `checked_at` is now produced by
     `mufinder.domain.models.utc_now()` (it was already `datetime.now(UTC)`,
     which is what `utc_now()` wraps — no behavioral change, just the shared
     helper).
   - New `JobExecutionController.mark_queue_checked(snapshot) -> bool` stamps
     `last_checked_at` / `last_status_source="scheduler_poll"` /
     `last_status_message` on every observed job present in the document. No
     status transitions, no run mutations. Wired from the top of
     `MainWindow._on_cluster_queue_checked`, before the dashboard flush that
     persists the document.
   - Retrieval and local-runner outcomes stamp the same fields through the
     shared controller paths (`complete_job_with_result`, `fail_job`,
     `mark_started`, `mark_local_status_changed`). Because the local and
     cluster `job_finished`/`job_failed` handlers share one code path in
     `MainWindow`, the default source is **inferred from the job's execution
     target** (cluster ⇒ `"retrieval"`, local ⇒ `"local_runner"`); explicit
     `source=`/`status_message=` keywords override the inference (e.g. for the
     future watchdog).
   - `ProjectService.reconcile_orphaned_local_jobs` stamps
     `last_status_source="reconciliation"` and `last_status_message` equal to
     the orphaned-job stop message.

2. **Rec #5 — grace window for missing scheduler entries.**
   - A tracked job absent from the scheduler queue still triggers immediate
     retrieval, but *output-unavailable* download failures no longer fail the
     job on the first poll. A per-entry counter
     (`_TrackedClusterJob.missing_retrieval_attempts`) allows
     `MISSING_ENTRY_RETRIEVAL_ATTEMPTS = 3` consecutive failed download
     attempts; the entry stays non-terminal and retries on subsequent polls.
     The poll observation reads
     `"missing from scheduler queue; waiting for output (attempt k of 3)"`.
   - After the 3rd failed attempt the runner emits `job_failed` with exactly:
     > Job disappeared from the scheduler queue and its output could not be
     > retrieved after 3 checks. The scheduler history may have expired, the
     > queue status command may be filtered, or a different cluster may be
     > connected. Retry retrieval or resubmit.
   - Parse failures of successfully downloaded output still fail immediately
     (no grace window), as does the incomplete-output → stopped/failed logic.
   - The counter resets to 0 whenever the job reappears in the scheduler queue.
   - **No new transition edges were needed.** Signal sequence per missing-entry
     poll: `status_changed(retrieving)` (inert no-op for queued jobs; legal
     RUNNING→RETRIEVING otherwise), then on final failure `job_failed` +
     `status_changed(failed)` (QUEUED/RUNNING/RETRIEVING → FAILED are all
     legal), on success `job_finished` + `status_changed(completed)` (the
     completion path start/completes through existing primitives).

3. **Rec #3 — poll failure needs no job mutation (confirmed).**
   Characterized: the `_poll_queue` except-branch disconnects and emits
   `poll_failed` only; tracked entries keep their status/terminal/counter
   state and no job lifecycle signal is emitted. Staleness is derived by the
   dashboard (Stream C) from `last_checked_at` + disconnect state via
   `services.job_monitoring.job_staleness`. Pinned by
   `test_cluster_runner_poll_failure_leaves_tracked_jobs_untouched`.

## Known caveat (intentional, for the doc/reviewers)

Within a single poll cycle, a job that terminates via retrieval receives two
stamps: the retrieval stamp (`"retrieval"`, e.g. "output retrieved and
parsed") from the controller completion/failure path, then the poll stamp
(`"scheduler_poll"`, e.g. "missing from scheduler queue; output retrieved and
parsed") from `mark_queue_checked`, which runs when `queue_checked` fires at
the end of the same cycle. The final persisted message still describes the
outcome accurately; the source records the poll as the last observer. Pure
`"retrieval"` stamps survive for retrieval flows that happen outside a poll
cycle (e.g. Stream C's retry-retrieval action reusing the same controller
paths). Similarly, a scheduler-reported FAILED state briefly stamps
`"retrieval"/"output retrieval failed"` via the shared `fail_job` path before
the poll stamp corrects it to `"scheduler reports FAILED"` in the same cycle.

## Proposed text for docs/job_lifecycle.rst

### Section "Cluster runner behavior" — replace the "During polling" bullet list with:

```
During polling:

- Only non-terminal tracked jobs for the currently connected cluster are polled.
- Scheduler states are mapped to MuFinder states by the scheduler backend.
- ``queued`` and ``running`` emit status updates.
- ``cancelled`` marks the tracked job terminal and emits cancellation.
- ``stopped`` marks the tracked job terminal and emits ``stopped``.
- ``failed`` marks the tracked job terminal, retrieves available error files,
  emits ``job_failed``, and emits ``failed``.
- ``completed`` triggers retrieval of ``.castep`` plus optional ``.den_fmt``,
  ``-out.cell``, ``.phonon``, or ``.ts`` outputs. Retrieval or parsing failure
  becomes ``failed``. Parsed incomplete CASTEP output becomes ``stopped`` when
  it looks like walltime; otherwise it becomes ``failed``. Parsed completed
  output emits ``job_finished`` and ``completed``.
- A missing scheduler entry usually means completion, so retrieval is
  attempted immediately — but with a grace window: if the output cannot be
  downloaded (SFTP error, missing remote ``.castep``, connection hiccup), the
  job is *not* failed yet. The runner retries on up to
  ``MISSING_ENTRY_RETRIEVAL_ATTEMPTS`` (3) consecutive polls, keeping the
  entry non-terminal, and only then emits ``job_failed`` with a diagnostic
  explaining that scheduler history may have expired, the queue command may be
  filtered, or the wrong cluster may be connected. Parse failures of
  successfully downloaded output fail immediately. The attempt counter resets
  if the job reappears in the queue.
- Every poll emits a ``queue_checked`` snapshot carrying ``checked_at``
  (``utc_now()``), per-job elapsed runtimes, and per-job human-readable
  observations (``observations_by_job_id``) describing what the scheduler
  reported for each polled job (for example "scheduler reports RUNNING" or
  "missing from scheduler queue; waiting for output (attempt 1 of 3)").
  ``JobExecutionController.mark_queue_checked`` persists these observations as
  per-job monitor metadata: ``last_checked_at``, ``last_status_source``
  (``"scheduler_poll"``), and ``last_status_message``.
- Poll failure disconnects the runner, clears the connected state, and emits a
  warning through the UI. Tracked job state is left untouched — active cluster
  rows go stale rather than failing, and the dashboard derives staleness from
  ``last_checked_at`` plus the disconnect (``services.job_monitoring``).
```

### Section "Execution orchestration" — append after the existing bullets:

```
Monitor metadata (``last_checked_at``, ``last_status_source``,
``last_status_message``) is stamped by ``JobExecutionController`` as
bookkeeping (never a status transition), with a fixed source vocabulary:
``"scheduler_poll"`` (queue-poll observations via ``mark_queue_checked``),
``"retrieval"`` (cluster output retrieval outcomes), ``"local_runner"``
(local lifecycle events such as "queued locally", "process started",
"process completed", "process failed"), ``"reconciliation"`` (orphaned
local-job reconciliation on project open), plus ``"watchdog"`` and ``"user"``
reserved for future flows. The shared completion/failure paths infer the
source from the job's execution target; explicit ``source=`` /
``status_message=`` keyword overrides are available. Failure stamps hold a
short observation ("output retrieval failed"); the full diagnostic remains on
``job.error_message``.
```

### Recommendations section — mark implemented:

- Recommendation 2 ("Persist monitor metadata and last-observed status"):
  append "(implemented)" and describe: the fields exist on ``Job``, are
  decoded/persisted, are stamped from queue polls, retrieval outcomes, local
  runner events, and reconciliation, and are cleared by ``reset_failed_job``.
- Recommendation 5 ("Avoid treating every missing scheduler entry as
  completed without context"): append "(implemented)" and describe the
  3-attempt grace window, the exact diagnostic message, and the
  parse-failures-fail-immediately rule.
- Recommendation 3: the data path (stamping + untouched state on poll
  failure) is implemented by this stream; the dashboard stale marker is
  Stream C's rendering work (``services.job_monitoring.job_staleness``).

## Proposed text for docs/implemented_functionality.rst

Add to the cluster execution / job dashboard feature area:

```
- Per-job monitor metadata: every queue poll, retrieval outcome, local-runner
  lifecycle event, and project-open reconciliation stamps ``last_checked_at``,
  ``last_status_source``, and ``last_status_message`` on the job, so the
  dashboard can distinguish "still queued" from "not checked recently" and
  show what the scheduler last reported.
- Missing-scheduler-entry grace window: a cluster job that disappears from the
  scheduler queue is retried for up to three polls when its output cannot be
  downloaded yet, instead of failing on the first attempt; after three failed
  checks it fails with a diagnostic suggesting expired scheduler history, a
  filtered queue command, or a wrong-cluster connection, and offering retry or
  resubmission. Unreadable (downloaded but unparseable) output still fails
  immediately.
```

## Exact new/changed public names

- `mufinder.services.cluster_runner.MISSING_ENTRY_RETRIEVAL_ATTEMPTS = 3`
- `mufinder.services.cluster_runner.MISSING_SCHEDULER_ENTRY_FAILURE_MESSAGE`
- `QueueCheckSnapshot.observations_by_job_id: dict[UUID, str]` (defaulted, so
  existing two-field constructions keep working)
- `_TrackedClusterJob.missing_retrieval_attempts: int = 0`
- `ClusterJobRunner._retrieve_results(entry, *, missing_from_queue: bool = False) -> str`
  (private; now returns the observation string)
- `JobExecutionController.mark_queue_checked(snapshot: Any) -> bool`
- `JobExecutionController.mark_started(job_id, *, source: str | None = None, status_message: str | None = None)`
- `JobExecutionController.complete_job_with_result(job, *, result, warnings, convergence_achieved, source: str | None = None, status_message: str | None = None)`
- `JobExecutionController.fail_job(job_id, message, *, source: str | None = None, status_message: str | None = None)`
- `JobExecutionController._stamp_monitor_metadata(job, *, source, message)` (private helper)
