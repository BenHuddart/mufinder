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
result handling, and terminal jobs that have been ``Dismiss``-ed are hidden
without losing their records. ``Resubmit`` is available for failed and
cancelled jobs. Cluster resubmission reuses existing remote files when the job
still records a remote directory, so continuation runs can start from the
previous working directory, and failed or stopped cluster jobs that kept their
remote metadata can ``Retry retrieval`` instead of resubmitting.

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
  ``defined`` and clears remote/process metadata (including the monitor
  metadata and any ``dismissed_at`` stamp).
- ``retry_job_retrieval`` moves a ``failed`` or ``stopped`` job back to
  ``retrieving`` (the FAILED/STOPPED → RETRIEVING edges), clears
  ``error_message``, and preserves remote metadata. The failed/stopped-only
  and remote-metadata preconditions are enforced by
  ``JobExecutionController.prepare_retry_retrieval``, not the primitive.
- ``dismiss_job`` stamps ``dismissed_at`` for a ``failed``/``stopped``/
  ``cancelled`` job. It is not a status transition and rejects
  already-dismissed jobs; ``record_job_dismissed`` appends a ``job_dismissed``
  provenance milestone. The app layer nonetheless *derives* a submittable
  workflow-row status from the stamp — see
  :ref:`dismissed-rows-stay-resubmittable` — so the contract here is unchanged
  while a dismissed job stops being a dead end.

``reset_failed_job`` and ``dismiss_job`` accept exactly the same statuses, held
in the module constant ``TERMINAL_RECOVERABLE_JOB_STATUSES``. Keeping the two
identical is what makes the derived projection sound: anything the user can
dismiss, the app can also reset back to ``defined``.

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

Monitor metadata (``last_checked_at``, ``last_status_source``,
``last_status_message``) is stamped by ``JobExecutionController`` as
bookkeeping (never a status transition), with a fixed source vocabulary:
``"scheduler_poll"`` (queue-poll observations via ``mark_queue_checked``),
``"retrieval"`` (cluster output retrieval outcomes), ``"local_runner"``
(local lifecycle events such as "queued locally", "process started",
"process completed", "process failed"), ``"reconciliation"`` (orphaned
local-job reconciliation on project open), ``"watchdog"`` (active-job
watchdog stops), and ``"user"`` (dashboard retry-retrieval requests). The
shared completion/failure paths infer the source from the job's execution
target; explicit ``source=`` / ``status_message=`` keyword overrides are
available. Failure stamps hold a short observation ("output retrieval
failed"); the full diagnostic remains on ``job.error_message``.

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
- A ``QProcess`` ``FailedToStart`` error (for example a missing or
  non-executable CASTEP binary) is reported immediately as ``job_failed`` and
  ``failed`` rather than waiting for a ``finished`` event that Qt may never
  deliver. The active entry is cleaned up and the next queued job starts, so a
  job cannot remain ``submitting`` behind a process that never launched. The
  failure message is *"The CASTEP process failed to start: {detail}. Check the
  CASTEP executable path and file permissions."* Other ``QProcess`` errors
  keep the store-and-wait behaviour and are reported by the finish handler.
- Successful parsing of a completed CASTEP output emits ``job_finished`` and
  ``completed``. On the same completion callback the runner also parses the
  job's *secondary* outputs (relaxed ``-out.cell``, ``.phonon``, ``.ts``) once
  via :func:`~mufinder.services.job_artifacts.collect_job_artifacts` and delivers
  the resulting :class:`~mufinder.services.job_artifacts.ParsedJobArtifacts` as
  the third ``job_finished`` payload, so the GUI handler and the owning workflow
  controller never re-read those files (see :doc:`architecture`, Workflow
  controllers → Parse-once off the GUI thread).

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

**No GUI-thread call blocks on the worker.** Every GUI-initiated cluster
operation is queued onto the worker and returns immediately; its outcome is
delivered back to the GUI thread via a signal. This is essential precisely
because the worker also runs the serial queue poll — a single poll can spend
minutes downloading and parsing a whole batch of finished jobs — so a GUI call
that waited on the worker would freeze the window until that batch drained. The
controller uses two asynchronous styles:

- *Fire-and-forget with a purpose-built signal* for the high-level flows:
  ``connect_async`` / ``connect_finished``, ``check_remote_dirs`` /
  ``remote_dirs_checked``, and ``submit_batch`` with its progress signals, plus
  ``refresh_queue`` / ``cancel`` / ``track_existing_job`` (results arrive via the
  forwarded runner signals).
- *A request/completion bridge* for the low-level helpers and
  ``clear_tracked_jobs``: the op is queued on the worker and its outcome is
  delivered via ``ClusterController.operation_finished`` ``(request_id, kind,
  result, error)``, keyed by a per-call ``request_id`` and a ``kind``
  discriminator so multiple in-flight ops route to their own continuation. The
  completion fires on **every** exit path — success, a caught exception, or an
  exception escaping the queued task — so a UI element disabled while an op is
  pending is always re-enabled and never stuck. ``clear_tracked_jobs`` (run on
  project switch) additionally clears the GUI-thread tracked-id mirror eagerly and
  relies on worker FIFO ordering so the subsequent ``track_existing_job``
  re-registrations are never wiped by a late clear.

The single deliberate exception is ``ClusterController.shutdown``, which stops
the worker's poll ``QTimer`` on the worker thread with a blocking call at teardown
(the app is already closing); no responsiveness-sensitive path blocks.

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

SLURM memory requests
~~~~~~~~~~~~~~~~~~~~~

SLURM enforces a per-job cgroup memory limit, and a CASTEP task that exceeds it
is killed by the out-of-memory handler rather than failing gracefully. The
generated ``run.sh`` therefore supports an optional per-CPU memory request,
surfaced as **Memory per CPU (GB)** in the cluster submission dialog and
pre-seeded from a SLURM cluster profile's ``default_mem_per_cpu_gb``.

- The value is plumbed as ``mem_per_cpu_gb`` from
  ``ClusterSubmissionConfig`` through ``ClusterJobSubmission`` and
  ``ClusterJobRunner.submit`` into
  :func:`~mufinder.services.script_template.render_job_script`.
- ``render_job_script`` emits it as a ``#SBATCH --mem-per-cpu`` directive on
  the slurm path only, inserted after the last existing ``#SBATCH`` line so it
  stays inside the leading block ``sbatch`` parses. A template that already
  carries its own ``--mem``/``--mem-per-cpu``/``--mem-per-gpu`` directive is
  left untouched, so a hand-written request always wins.
- ``None`` (the default for every cluster profile) emits nothing, leaving the
  rendered script byte-for-byte as it was before the option existed and the
  partition's own memory default in force.
- SLURM rejects a fractional gigabyte value, so
  :func:`~mufinder.services.script_template.format_mem_per_cpu_directive`
  renders whole gigabytes as ``6G`` and anything else as whole megabytes
  (``7.5`` becomes ``7680M``).
- ``--mem-per-cpu`` is per *allocated CPU*. MuFinder submits pure-MPI CASTEP
  runs and never sets ``--cpus-per-task``, which SLURM defaults to 1, so one
  CPU is one MPI rank and the value is the per-rank memory ceiling. This is
  deliberately preferred over ``--mem`` (per node), which would have to be
  rescaled whenever node packing changed.

This is distinct from the addqueue ``memory_gb`` control ("Memory per core"),
which is passed as the ``addqueue -m`` argument rather than written into a
batch script. The two are mutually exclusive in the dialog: the per-CPU row is
hidden for addqueue clusters, and ``mem_per_cpu_gb`` is forced to ``None``
there.

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
  output emits ``job_finished`` and ``completed``. The secondary outputs
  (``-out.cell`` / ``.phonon`` / ``.ts``) are parsed once here on the worker
  thread via :func:`~mufinder.services.job_artifacts.collect_job_artifacts` and
  carried to the GUI as the third ``job_finished`` payload, so the completion
  handler and controllers never re-parse them.
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
instead of refreshing and saving inline; the pending flush runs on
``_flush_dashboard``, scheduled via ``_dashboard_flush_timer``, and also
before save/close/project-switch.

``_request_dashboard_flush`` combines two rules so both an isolated
completion and a multi-minute completion stream behave well:

- **Quiet-period debounce** (``_DASHBOARD_FLUSH_DEBOUNCE_MS`` = 300 ms): an
  isolated event flushes shortly after it arrives, giving the same prompt
  feel as before.
- **Minimum-interval floor** (``_DASHBOARD_MIN_FLUSH_INTERVAL_MS`` = 2000 ms):
  during a burst — e.g. a serial cluster retrieval where jobs complete a few
  seconds apart, each just outside the debounce window — flushes are bounded
  to roughly one every 2 seconds instead of one per completion. The armed
  deadline is only ever tightened (moved earlier), never pushed further into
  the future, so a steady stream of requests cannot re-arm the timer forever
  and starve the flush until the stream ends.

``_on_cluster_queue_checked`` stamps per-job monitor metadata
(``last_checked_at``/source/message) and then also goes through
``_request_dashboard_flush`` rather than flushing synchronously — it no
longer bypasses the debounce/rate-limit, since a queue poll can itself fire
every few seconds during a retrieval. A burst of cluster jobs finishing
across one or several polls therefore rebuilds the views and writes the
document at a bounded cadence rather than once per job, which previously
froze the GUI for large batches.

The dashboard displays active states plus ``failed``, ``stopped``, and
``cancelled``; rows whose job carries a ``dismissed_at`` stamp are filtered
out. A ``Last Checked`` column shows the relative age of ``last_checked_at``
with a stale marker for active cluster jobs whose cluster is disconnected or
whose last check is older than the ``services.job_monitoring`` threshold, and
a tooltip carrying ``last_status_source`` / ``last_status_message`` / the
staleness reason. Actions are enabled as follows:

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
- ``Reset``: ``failed``, ``stopped``, and ``cancelled``. The same action is
  also available per row from every workflow table (see
  :ref:`per-row-reset-affordance`), which is what keeps a non-dismissed
  failure recoverable without going through the dashboard.
- ``Dismiss``: ``failed``, ``stopped``, and ``cancelled``. Dismissal hides
  the row from the active dashboard by stamping ``dismissed_at``; it is not a
  status transition, so the job record, runs, results, and provenance all
  remain in the document. There is no un-dismiss *action* — the stamp is
  cleared only by ``reset_failed_job``, which the job's own workflow row now
  triggers on its next submit (see below).

.. _per-row-reset-affordance:

Every workflow table has a per-row reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A failed job's recovery must not depend on the jobs dashboard: the dashboard row
is easy to lose track of in a long project, and it disappears entirely once the
job is dismissed. Every workflow table therefore offers the same recovery action
on the row itself, via a right-click context menu carrying a single
``Reset Failed Jobs...`` entry.

The shared implementation lives in ``mufinder.app.workflow_table_utils``:

- ``RESETTABLE_ROW_STATUSES`` — the gate, ``{failed, stopped, cancelled}``,
  matching what ``ProjectService.reset_failed_job`` accepts. A *dismissed*
  terminal job is deliberately excluded: after ``effective_row_status`` its row
  already reads ``"ready"`` and its recovery path is the row's own ``Submit``
  (see :ref:`dismissed-rows-stay-resubmittable`).
- ``selection_supports_reset`` — enables the action only when the whole
  selection is resettable. ``_reset_failed_jobs`` prompts once for the entire
  batch, including the destructive output cleanup, so a mixed selection is
  refused rather than silently narrowed.
- ``bind_reset_failed_context_menu`` / ``open_reset_failed_context_menu`` —
  install and open the menu. Right-clicking a row outside the current selection
  retargets the selection first, so the action always describes the row under
  the cursor. Each widget supplies only a ``key_for_row`` (its own submission
  key: a cutoff, a k-point grid, a label, a charge-state-qualified key, or a
  path key) and a ``status_for_row``.
- ``RESET_FAILED_TABLE_TOOLTIP`` — the discoverability stamp. A single-entry
  menu that is greyed out except on failed rows is invisible on its own: a user
  with a stuck row has no reason to right-click. ``bind_reset_failed_context_menu``
  therefore sets this tooltip on every table it binds, unless the widget already
  set a tooltip of its own. Per-item tooltips are unaffected and still take
  precedence where a widget sets them. The energy-barriers diffusion-path table
  additionally carries the hint in its visible hint label.

``MainWindow`` connects each widget's ``reset_failed_requested`` signal to a
thin ``_on_reset_failed_from_*`` handler that resolves the keys through the
workflow's ``WorkflowJobController.jobs_for_keys`` and hands the jobs to the
shared ``_reset_failed_jobs``. Muon relaxation binds the menu once per
charge-state table, each capturing its own table, so a right-click acts on the
rows it was raised over rather than on whichever tab happens to be current.

After a reset, ``_refresh_views_after_job_reset`` repaints every workflow table.
``_refresh_jobs_table`` already covers the dashboard plus the convergence,
pristine, muon, and UEP tables; energy barriers and quantum corrections are
refreshed through their own wrappers, which no-op until their lazily-created
view exists.

Quantum corrections is the exception to the shape above. Its row stages a
singlepoint job and then a phonon job into one shared output directory, so its
recovery is a *stage* reset (``_on_reset_quantum_corrections_to_stage_one``,
which deletes both job records and clears the directory) rather than a job
reset. ``_quantum_can_reset_to_stage_one`` gates it: the row qualifies once it
has staged a job and has no job still running — either the phonon job exists and
is not active, or the row is still on stage 1 with a singlepoint job that has
finished, which includes a failed, stopped, or cancelled one. That last case is
the escape hatch; without it a stage-1 failure with no phonon job showed
``"failed"`` (so ``Submit`` was disabled) *and* was refused by the reset, leaving
the row permanently stuck.

.. _dismissed-rows-stay-resubmittable:

Dismissed rows stay resubmittable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dismissing a terminal job removes the only generic ``Resubmit``/``Reset``
controls it had, but its workflow table row still points at it through the
definition's persistent label→job-id mapping. Left alone, that row would project
``"failed"`` — not a member of ``READY_WORKFLOW_ROW_STATUSES`` — and its
``Submit`` would stay disabled forever, with no way back to ``defined``. The
row's own ``Reset Failed Jobs...`` action cannot rescue it either: that action
gates on the displayed status, and a dismissed row no longer reads as failed.

Dismissal is therefore read as *"failure acknowledged, this row is actionable
again"*, via two pieces in ``mufinder.app.workflow_submission``:

- ``effective_row_status(job)`` — the single place the rule lives. A job whose
  status is in ``RESUBMITTABLE_AFTER_DISMISSAL_STATUSES`` (failed, stopped,
  cancelled) **and** which carries a ``dismissed_at`` stamp projects
  ``"ready"``; every other job projects its own status value. Every workflow's
  job→row-status projection goes through it. A *non*-dismissed failed job is
  unaffected: it still shows ``"failed"``, its row still refuses to submit, and
  its recovery path is the row's own ``Reset Failed Jobs...`` action plus the
  dashboard ``Resubmit``/``Reset`` buttons that are still on screen.
- ``prepare_jobs_for_resubmission`` on ``WorkflowControllerContext`` — the lazy
  reset. ``WorkflowJobController.submit`` partitions the planned jobs and, for
  the dismissed ones only, calls this hook immediately before dispatch. The
  ``MainWindow`` implementation runs the same stale-output cleanup prompt the
  ``Reset`` action uses and then calls ``reset_failed_job`` per job, so the jobs
  reach the submit backends as ``defined`` and clear their ``{DEFINED}`` gate
  unchanged. Cancelling the prompt aborts the whole submission silently.

``submit`` then re-runs ``materialize`` for the plan's ready keys. Controllers
only rewrite a job's input directory when the job is ``defined``, and the first
materialize pass ran while building the plan — when the dismissed jobs were
still terminal — so without the second pass the row would resubmit against the
stale ``.cell``/``.param`` the failed run left behind (the cleanup prompt
deliberately preserves exactly those two extensions). The second pass makes a
dismissed-row resubmit produce the same inputs an explicit reset-then-submit
would.

Because both steps act on the very job objects the submission plan already
holds, the row keeps its existing job — no duplicate is created — and, unlike
the explicit ``Reset`` action, a reset cutoff-convergence job is deliberately
*not* detached from its ``job_ids_by_cutoff`` mapping (detaching would leave the
row bound to nothing and let a second submit mint a duplicate).

The projection is derived at read time, so no persisted state changes and
projects already stranded by an earlier dismissal heal on load.

When a project is opened or refreshed, ``_sync_cluster_tracking_from_document``
re-registers persisted active cluster jobs that have enough metadata to poll:
non-local execution target, active status, remote job id, remote directory or an
inferable remote directory, and a local output directory.

Active-job watchdog
~~~~~~~~~~~~~~~~~~~

Orphaned-local reconciliation (recommendation 1) only covers the app-restart
case. A running session can still leave a job active with nothing executing it:
a local ``QProcess`` that died without a finish event, a cluster job the runner
silently stopped tracking, or a queue entry that never reached a runner. A
once-a-minute watchdog closes this gap.

The detection core, ``mufinder.services.job_watchdog.detect_stuck_jobs``, is a
pure, Qt-free function. Given the project's jobs plus runner-introspection
snapshots, it flags:

- a local ``queued`` job absent from both the local runner's pending and
  active id sets;
- a local ``running`` job absent from the local runner's active id set;
- a local ``submitting`` job older than a 120-second grace period (by
  ``submitted_at``; skipped when ``submitted_at`` is ``None``) and absent from
  the pending and active id sets;
- a cluster job in an active state whose id is not in the cluster controller's
  tracked-id mirror, **only** when the connected cluster is the one that job
  targets. When the cluster is disconnected, cluster jobs are never flagged —
  that is the staleness marker's responsibility (recommendations 2–3), not the
  watchdog's.

Runner introspection is exposed by ``LocalJobRunner.active_job_ids()`` /
``pending_job_ids()`` and, on the GUI thread, by
``ClusterController.tracked_active_job_ids()`` — a mirror of the runner's
non-terminal ``_tracked_jobs`` maintained from the ``status_changed`` stream
plus ``track_existing_job``/``clear_tracked_jobs``. The mirror is deliberately
retained across a disconnect (the runner keeps its tracked jobs across a
disconnect, and the watchdog never flags cluster jobs while disconnected), so
reconnecting cannot trigger a false positive.

To ride out an in-flight runner signal that has not yet reached the document, a
finding is acted on only when it is confirmed in **two consecutive**
evaluations (``StuckJobWatchdog``). A confirmed finding is applied by
``JobExecutionController.stop_stuck_job``, which routes the transition through
``ProjectService.stop_job`` (the single authority for the ``-> stopped`` edge),
closes out the job's active ``CalculationRun``, records a ``job_stopped``
journal event, and stamps the monitor bookkeeping fields
(``last_checked_at``, ``last_status_source = "watchdog"``,
``last_status_message``). Stopped rows remain visible and resettable, exactly
like a walltime stop.

``MainWindow`` owns the 60-second ``QTimer`` (services must not create widgets)
and a single handler that snapshots the runners, runs the watchdog, applies
``stop_stuck_job`` per confirmed finding, requests a coalesced dashboard flush,
and shows *"N job(s) were stopped by the job watchdog because no runner was
tracking them."* when any were stopped. The handler is cheap when no jobs are
active.

Recommendations
---------------

This section began as a review of failure modes that could leave jobs in the
dashboard indefinitely. All nine recommendations have since been implemented;
each entry below records what shipped and where the behavior lives.

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
   id and directory). Cluster jobs that lack enough metadata to re-register —
   missing remote id, remote directory, cluster config, or local output
   directory — are surfaced by the staleness marker and the active-job
   watchdog (recommendations 2–3 and 7).

2. Persist monitor metadata and last-observed status (implemented).
   ``Job`` carries ``last_checked_at``, ``last_status_source``, and
   ``last_status_message`` (optional, ``None``-defaulted, persisted through the
   codec and cleared by ``reset_failed_job``). They are stamped from queue
   polls (``JobExecutionController.mark_queue_checked``), retrieval outcomes,
   local-runner lifecycle events, project-open reconciliation, watchdog stops,
   and user retry requests, with the fixed source vocabulary described under
   "Execution orchestration". The dashboard's ``Last Checked`` column renders
   the age plus a tooltip with the source and observation.

3. Make disconnected cluster jobs visibly stale (implemented).
   A queue poll failure still disconnects the runner and deliberately leaves
   job state untouched; staleness is derived, not stored.
   ``services.job_monitoring.job_staleness`` marks an active cluster job stale
   when its cluster is disconnected or its last check is older than the
   threshold (180 s), and the dashboard renders a " — stale" marker with the
   reason in the tooltip.

4. Add a retry-retrieval action for stopped/failed cluster jobs (implemented).
   A ``Retry retrieval`` dashboard action re-attempts output retrieval for a
   scheduler-completed cluster job that kept its remote metadata, avoiding an
   unnecessary duplicate scheduler job. It routes through
   ``JobExecutionController.prepare_retry_retrieval`` →
   ``ProjectService.retry_job_retrieval`` (the FAILED/STOPPED → RETRIEVING
   edges) and ``ClusterJobRunner.retrieve_outputs_for``, registers a fresh
   ``CalculationRun`` so the outcome cannot overwrite the prior terminal run,
   and resolves a not-connected cluster to ``failed`` rather than stranding
   the job in ``retrieving``.

5. Avoid treating every missing scheduler entry as completed without context
   (implemented).
   A missing scheduler entry still attempts retrieval immediately, but
   output-unavailable failures are retried across a three-poll grace window
   before the job is failed with a diagnostic naming the likely causes
   (expired scheduler history, filtered queue command, wrong cluster) and
   suggesting retry or resubmission. Parse failures of downloaded output fail
   immediately. See "During polling" above.

6. Fail fast on local process start errors (implemented).
   ``LocalJobRunner._on_process_error`` handles
   ``QProcess.ProcessError.FailedToStart`` directly: it emits ``job_failed``
   and ``failed``, cleans up the active entry, and starts the next queued job,
   so a job cannot remain ``submitting`` when Qt does not deliver a finish
   event. A late ``finished`` event for that job is a no-op. Other
   ``ProcessError`` values keep the store-and-wait behaviour.

7. Add an active-job watchdog (implemented).
   A once-a-minute watchdog (``mufinder.services.job_watchdog``) detects
   active jobs with no runner tracking them — ``queued``/``running``/
   ``submitting`` local jobs missing from the local runner's pending/active
   sets, and active cluster jobs missing from the cluster controller's
   tracked-id mirror while their cluster is connected — and, after two-tick
   confirmation, stops them via ``JobExecutionController.stop_stuck_job``
   (through ``ProjectService.stop_job``) with a diagnostic message and
   ``last_status_source = "watchdog"``. See "Active-job watchdog" above.

8. Separate terminal history from the active dashboard (implemented).
   A ``Dismiss`` action stamps ``dismissed_at`` on a failed/stopped/cancelled
   job so the dashboard filters the row out, while keeping the job record,
   runs, results, and provenance intact (a ``job_dismissed`` milestone is
   recorded). There is no un-dismiss UI; the stamp is cleared by
   ``reset_failed_job``, which the job's workflow row triggers on its next
   submit (see :ref:`dismissed-rows-stay-resubmittable`).

9. Strengthen tests around stuck-row prevention (implemented alongside 1–8).
   Project-open reconciliation, poll-failure inertness, the missing-entry
   grace window, ``QProcess`` startup failure, watchdog detection and wiring,
   retry-retrieval run bookkeeping, and dismissal visibility are all pinned by
   tests beside the existing ``JobExecutionController``, ``LocalJobRunner``,
   ``ClusterJobRunner``, and main-window action suites (see
   ``tests/services/test_job_watchdog.py``,
   ``tests/services/test_retry_retrieval.py``,
   ``tests/services/test_job_dismissal.py``, and
   ``tests/app/test_dashboard_monitoring_columns.py``).
