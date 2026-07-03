Job Lifecycle
=============

This document describes the current lifecycle for local and cluster jobs. It is
both a user-facing reference for dashboard behavior and a technical API contract
for the services that drive job execution.

User-facing summary
-------------------

MuFinder jobs move through the dashboard as calculations are submitted,
monitored, and finished:

``defined``
   The job has been created but has not entered the dashboard execution
   lifecycle yet. Submit it from the workflow table that created it.

``submitting``
   MuFinder is handing the job to a local process or a cluster scheduler.

``queued``
   The job has been accepted and is waiting to run.

``running``
   CASTEP is running locally or on the selected cluster.

``retrieving``
   A cluster job has finished on the scheduler and MuFinder is copying/parsing
   output files.

``completed``
   The calculation finished successfully. Completed jobs leave the dashboard.

``failed``
   The process, scheduler, retrieval, or parser reported a failure. Failed jobs
   remain in the dashboard so their diagnostics can be inspected. They can be
   resubmitted or reset.

``stopped``
   A cluster calculation appears to have stopped before producing a successful
   CASTEP completion marker, usually because of walltime or scheduler limits.
   Stopped jobs remain in the dashboard for inspection and reset.

``cancelled``
   The user requested cancellation for a queued or running job. Cancelled jobs
   remain in the dashboard and can be resubmitted or reset.

The dashboard shows jobs that still need attention: active jobs, failed jobs,
stopped jobs, and cancelled jobs. Completed jobs disappear after successful
result handling. ``Resubmit`` is available for failed and cancelled jobs.
Cluster resubmission reuses existing remote files when the job still records a
remote directory, so continuation runs can start from the previous working
directory.

Technical API behavior
----------------------

State authority
~~~~~~~~~~~~~~~

The persisted job state is stored on ``mufinder.domain.models.Job`` using the
``JobStatus`` enum. State transitions are validated by
``ProjectService._assert_transition``. The current transition table is:

.. list-table::
   :header-rows: 1

   * - Current state
     - Allowed next states
   * - ``defined``
     - ``submitting``, ``cancelled``
   * - ``submitting``
     - ``queued``, ``running``, ``stopped``, ``failed``, ``cancelled``
   * - ``queued``
     - ``running``, ``stopped``, ``failed``, ``cancelled``
   * - ``running``
     - ``retrieving``, ``stopped``, ``completed``, ``failed``, ``cancelled``
   * - ``retrieving``
     - ``stopped``, ``completed``, ``failed``, ``cancelled``
   * - ``stopped``
     - ``submitting``, ``cancelled``
   * - ``completed``
     - none
   * - ``failed``
     - ``submitting``, ``cancelled``
   * - ``cancelled``
     - ``submitting``

``ProjectService`` owns the primitive mutations:

- ``submit_job`` moves an eligible job to ``submitting``, stamps
  ``submitted_at``, clears ``started_at``/``completed_at``, and clears
  ``error_message``.
- ``mark_job_queued`` moves ``submitting`` to ``queued`` and may update
  ``remote_job_id``.
- ``start_job`` moves ``submitting`` or ``queued`` to ``running`` and stamps
  ``started_at``.
- ``mark_job_retrieving`` moves ``running`` to ``retrieving`` for remote output
  copy/parsing.
- ``complete_job`` moves ``running`` or ``retrieving`` to ``completed``, stamps
  ``completed_at``, and clears ``error_message``.
- ``stop_job`` moves an active job (``submitting``, ``queued``, ``running``, or
  ``retrieving``) to ``stopped`` and records a message. The ``submitting`` and
  ``queued`` source edges exist so orphaned local jobs left mid-submission when
  MuFinder closed can be reconciled on the next project open (see
  ``reconcile_orphaned_local_jobs`` below).
- ``fail_job`` moves an in-flight job to ``failed`` and records a message.
- ``cancel_job`` moves a cancellable job to ``cancelled``.
- ``reset_failed_job`` moves ``failed``, ``stopped``, or ``cancelled`` back to
  ``defined`` and clears remote/process metadata.

State authority and runner contract
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Job state transitions are owned **exclusively** by the ``ProjectService``
primitives listed above. The layering around them is deliberate and is codified
by the ``JobRunner`` protocol in ``mufinder.services.runner_protocol``:

- **ProjectService** performs every validated ``JobStatus`` transition. Nothing
  else writes ``Job.status``.
- **JobExecutionController** mutates only bookkeeping fields on the ``Job``
  (``execution_target``, ``num_processes``, and remote-metadata clears) and
  routes all status changes through the service primitives. For example, when
  the local runner reports ``status_changed(job_id, "queued")`` for a
  ``submitting`` job, ``mark_local_status_changed`` calls
  ``ProjectService.mark_job_queued``; a queued notification for a job that is not
  ``submitting`` is an inert no-op.
- **Runners never touch ``Job`` objects.** ``LocalJobRunner`` and
  ``ClusterJobRunner`` operate on their own in-memory tracking state and
  communicate solely by emitting signals keyed by job id (``job_started``,
  ``job_finished``, ``job_failed``, ``job_cancelled``, ``status_changed``). The
  controller translates those signals into ``ProjectService`` calls.

The two runners share the signal surface above and the identical
``cancel(job_id)`` method, but their ``submit`` error contracts are
**deliberately different** and are not to be unified:

- ``LocalJobRunner.submit`` **never raises** for bad input. It validates the
  request (CASTEP executable, ``mpirun`` for multi-core, job directory) and
  reports any problem by emitting ``job_failed`` (and
  ``status_changed`` -> ``failed``). ``submit_local_jobs`` still wraps the call
  defensively, but the runner's own contract is "validation failures are
  reported via ``job_failed``, not raised".
- ``ClusterJobRunner.submit`` **may raise** (``ConnectionError`` and similar)
  because it performs blocking SSH/SFTP work, and it returns the remote scheduler
  id on success. Its two callers convert the exception rather than letting it
  escape: ``JobExecutionController.submit_cluster_job`` catches it and returns a
  failed ``ClusterJobSubmissionOutcome`` (via ``fail_cluster_submission``), and
  ``ClusterController._run_submit_batch`` catches per job and emits
  ``job_submit_finished(job_id, None, str(exc))``.

Because the two ``submit`` methods take different arguments and return different
types, ``JobRunner`` models ``submit`` loosely and pins these per-implementation
contracts in its docstring; ``cancel`` is modeled precisely.

Execution orchestration
~~~~~~~~~~~~~~~~~~~~~~~

``JobExecutionController`` is the service facade used by the main window. It
keeps job records and ``CalculationRun`` records synchronized, records terminal
journal events, and tracks the active run id for each in-flight job.

Local submission uses ``submit_local_jobs``:

- The default eligible state is ``defined``. Dashboard resubmission passes
  ``{failed, cancelled}``.
- The controller resolves the job output directory and seed name before
  submission.
- The job is moved to ``submitting`` and its execution target is set to
  ``local``.
- Any stale remote id or remote directory is cleared for local execution.
- A new ``CalculationRun`` is created with status ``submitting``.
- ``LocalJobRunner.submit`` receives the process request.
- Submitted jobs are recorded in the project journal once at least one runner
  accepts the batch.

Cluster submission uses ``submit_cluster_job``:

- The controller accepts ``defined``, ``failed``, and ``cancelled`` jobs.
- The job is moved to ``submitting`` and assigned the selected cluster name,
  process count, and remote directory.
- A new ``CalculationRun`` is created with status ``submitting``.
- ``ClusterJobRunner.submit`` uploads or reuses files, submits to the scheduler,
  and returns the new remote scheduler id.
- On success, the job ``remote_job_id`` is replaced with the new scheduler id,
  the job is marked ``queued``, and the active run is marked ``queued``.
- On submission failure, the job is marked ``failed`` and the active run is
  completed as ``failed``. For continuation resubmissions, the previous remote
  context is preserved when possible so the user can try again.

Local runner behavior
~~~~~~~~~~~~~~~~~~~~~

``LocalJobRunner`` is an in-memory queue around ``QProcess``:

- Requests are validated for CASTEP executable, ``mpirun`` when multiple cores
  are requested, and the local job directory.
- Valid jobs enter a pending queue. If concurrency is saturated, the runner
  emits ``queued``.
- When a process starts, the runner emits ``job_started`` and ``running``.
- Cancelling a pending job emits ``job_cancelled`` and ``cancelled`` without
  starting a process.
- Cancelling an active job kills the process. The finish handler emits
  ``job_cancelled`` and ``cancelled`` when it sees the cancellation marker.
- Non-zero exit status, missing ``.castep`` output, parser failure, or a parsed
  unsuccessful CASTEP result emits ``job_failed`` and ``failed``.
- Successful parsing of a completed CASTEP output emits ``job_finished`` and
  ``completed``.

Local runner state is not persisted. If the application is closed while local
jobs are active, the runner performs a best-effort cancellation during shutdown,
but the durable project file may still contain active local statuses if that
state is not saved after cancellation.

Cluster runner behavior
~~~~~~~~~~~~~~~~~~~~~~~

``ClusterJobRunner`` is an SSH/scheduler facade with in-memory tracking.

All of its paramiko SSH/SFTP work is blocking, so it runs on a dedicated worker
thread owned by ``mufinder.app.cluster_controller.ClusterController``. The
controller mirrors the runner's public surface (the same signal names, methods,
and connection-state properties) and is what ``main_window`` holds, so the GUI
event loop is never blocked by polling, result retrieval, or connection setup.
Queue polling and output retrieval — the heaviest path, since a single
completed batch can download many files — therefore execute off the GUI thread.
The controller mirrors connection state (``is_connected``,
``connected_cluster_name``, ``connected_username``) on the GUI thread so callers
never touch the live transport across threads.

Connecting is also asynchronous (``connect_async`` / ``connect_finished``): the
multi-second SSH handshake runs on the worker behind a modal "Connecting…"
dialog instead of freezing the window. Host-key trust is handled by the GUI
retry loop in ``_connect_to_cluster_with_trust`` — on an unknown-host-key error
it prompts, saves the key, and reconnects, and the next attempt trusts it — so
no cross-thread coordination is needed.

Batch submission is asynchronous. The GUI computes remote targets, checks which
remote directories already exist (on the worker), resolves any overwrite prompts
up front, then prepares document state for each job
(``JobExecutionController.prepare_cluster_submission``) and hands the batch to
``ClusterController.submit_batch``. The worker uploads and submits each job,
emitting progress and per-job results that the GUI applies via
``finalize_cluster_submission``/``fail_cluster_submission``. A modal progress
dialog reports progress and offers a Cancel button; jobs prepared but not yet
submitted when cancelled are moved to ``cancelled``.

- ``connect_to_cluster`` creates the SSH connection and scheduler backend.
- ``submit`` validates the local output directory and remote seed name, creates
  the remote directory, uploads ``.cell``/``.param`` and ``run.sh`` unless
  ``reuse_existing_files`` is true, submits to the scheduler, stores a tracked
  job entry, emits ``queued``, and starts a one-minute poll timer.
- ``cancel`` sends the scheduler cancel command for a tracked non-terminal job,
  marks it terminal, and emits ``cancelled``.
- ``track_existing_job`` re-registers persisted remote jobs after project load
  or before a manual queue refresh. It requires a remote scheduler id, remote
  directory, local output directory, cluster name, and current active status.
- ``refresh_queue`` performs one poll immediately.

During polling:

- Only non-terminal tracked jobs for the currently connected cluster are polled.
- Scheduler states are mapped to MuFinder states by the scheduler backend.
- A missing scheduler entry is treated as ``completed`` so MuFinder attempts
  retrieval.
- ``queued`` and ``running`` emit status updates.
- ``cancelled`` marks the tracked job terminal and emits cancellation.
- ``stopped`` marks the tracked job terminal and emits ``stopped``.
- ``failed`` marks the tracked job terminal, retrieves available error files,
  emits ``job_failed``, and emits ``failed``.
- ``completed`` triggers retrieval of ``.castep`` plus optional ``.den_fmt`` or
  ``-out.cell`` outputs. Retrieval/parsing failure becomes ``failed``. Parsed
  incomplete CASTEP output becomes ``stopped`` when it looks like walltime;
  otherwise it becomes ``failed``. Parsed completed output emits
  ``job_finished`` and ``completed``.
- Poll failure disconnects the runner, clears the connected state, and emits a
  warning through the UI.

Main-window integration
~~~~~~~~~~~~~~~~~~~~~~~

The main window wires runner signals into ``JobExecutionController``:

- local and cluster ``job_finished`` both use the same completion path that
  creates workflow-specific results and then calls
  ``complete_job_with_result``.
- local and cluster ``job_failed`` both call ``fail_job``.
- local and cluster ``job_cancelled`` both call ``cancel_job``.
- cluster ``status_changed`` is used for ``queued``, ``running``,
  ``retrieving``, and ``stopped`` updates.

The per-job document mutation happens synchronously, but the **heavy dashboard
work is coalesced**: rebuilding the journal (whole provenance), the jobs table
(which recreates per-row action buttons), and the workflow views, plus the
full-document autosave. Lifecycle handlers call ``_request_dashboard_flush``
instead of refreshing and saving inline; the pending flush runs once after each
queue poll (``_on_cluster_queue_checked``), on a short debounce timer, and before
save/close/project-switch (``_flush_dashboard``). A burst of cluster jobs
finishing in one poll therefore rebuilds the views and writes the document once
rather than once per job, which previously froze the GUI for large batches.

The dashboard displays active states plus ``failed``, ``stopped``, and
``cancelled``. Actions are enabled as follows:

- ``Resubmit``: ``failed`` and ``cancelled``.
- ``Cancel``: ``submitting``, ``queued``, ``running``, and ``retrieving``.
- ``Inspect``: ``failed`` and ``stopped``.
- ``Reset``: ``failed``, ``stopped``, and ``cancelled``.

When a project is opened or refreshed, ``_sync_cluster_tracking_from_document``
re-registers persisted active cluster jobs that have enough metadata to poll:
non-local execution target, active status, remote job id, remote directory or an
inferable remote directory, and a local output directory.

Recommendations
---------------

The current design has a sensible central transition service, but several
failure modes can still leave jobs in the dashboard indefinitely. The following
changes are recommended.

1. Orphaned local-job reconciliation on project open (implemented).
   A local job left ``submitting``, ``queued``, ``running``, or ``retrieving``
   when MuFinder closes has no live process behind it on the next launch, yet
   would otherwise show as running forever. On every project open/switch,
   ``ProjectService.reconcile_orphaned_local_jobs`` selects active jobs whose
   ``execution_target`` is blank/empty or ``"local"`` (case-insensitively),
   moves each to ``stopped`` with the message *"MuFinder was closed while this
   job was in progress locally; reset or resubmit it."*, and closes out the
   job's latest non-terminal ``CalculationRun`` (``stopped`` with a
   ``completed_at`` stamp) to keep run and job status consistent. A
   ``job_stopped`` provenance milestone is recorded per reconciled job. The
   stopped rows remain visible and resettable, and the reset/resubmit action
   is unchanged. Reconciliation runs from ``MainWindow`` inside
   ``_refresh_ui_after_project_change`` — before the jobs table is rendered, so
   the user never sees a phantom running row — and ``_open_project_from_path``
   surfaces a status-bar message such as *"N local job(s) were stopped because
   MuFinder closed while they were in progress."* when any jobs are reconciled.

   Cluster jobs are deliberately left untouched by this path: they re-attach to
   the cluster runner instead, via
   ``MainWindow._sync_cluster_tracking_from_document`` (which re-registers
   active cluster jobs, including ``submitting``, that still carry a remote job
   id and directory). Remaining gaps for cluster jobs that lack enough metadata
   to re-register — missing remote id, remote directory, cluster config, or
   local output directory — are covered by recommendations 2–3 below and are
   not yet surfaced as explicit diagnostics.

2. Persist monitor metadata and last-observed status.
   Store fields such as ``last_checked_at``, ``last_status_source``, and
   ``last_status_message`` for cluster jobs. The dashboard can then distinguish
   "still queued" from "not checked since yesterday" and warn about stale rows.

3. Make disconnected cluster jobs visibly stale.
   A queue poll failure currently disconnects the runner and leaves active
   jobs in their last known state. The dashboard should show a stale/offline
   marker for active cluster jobs when their cluster is disconnected or their
   last queue check is older than a threshold.

4. Add a retry-retrieval action for stopped/failed cluster jobs.
   Retrieval or parsing failure currently converts a scheduler-completed job to
   ``failed``. If the remote output appeared late or a transient SSH failure
   occurred, resubmitting may be the wrong recovery. A ``Retry retrieval``
   action for jobs with remote metadata would avoid unnecessary duplicate
   scheduler jobs.

5. Avoid treating every missing scheduler entry as completed without context.
   The current cluster poller treats absent scheduler status as completion and
   attempts retrieval. That is a useful recovery path, but it can hide cases
   where the scheduler command is filtered, history expired, or the wrong
   cluster is connected. Prefer a short retrieval attempt window followed by a
   diagnostic state if outputs are unavailable.

6. Fail fast on local process start errors.
   ``LocalJobRunner`` records ``QProcess`` errors and relies on the finish path
   to emit a failure. Add targeted handling for ``FailedToStart`` and related
   startup errors so a job cannot remain ``submitting`` if Qt does not deliver a
   normal finish event.

7. Add an active-job watchdog.
   Periodically check active jobs for impossible combinations, such as
   ``queued`` local jobs with no pending runner entry, ``running`` local jobs
   with no active ``QProcess``, or active cluster jobs with no tracked runner
   entry. Convert these to a diagnosable terminal or attention state instead of
   leaving them as active forever.

8. Separate terminal history from the active dashboard.
   Failed, stopped, and cancelled jobs intentionally remain visible for
   recovery, but they can become clutter. Add an explicit ``Dismiss`` or
   ``Archive`` action for terminal jobs that have already been inspected or
   intentionally abandoned, while keeping provenance and run records intact.

9. Strengthen tests around stuck-row prevention.
   Add tests for project-open reconciliation, local restart with active jobs,
   cluster active job with missing metadata, poll failure stale UI, and
   ``QProcess`` startup failure. These should live beside the existing
   ``JobExecutionController``, ``LocalJobRunner``, ``ClusterJobRunner``, and
   main-window action tests.
