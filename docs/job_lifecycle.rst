Job Lifecycle
=============

This document describes the current lifecycle for local and cluster jobs. It is
both a user-facing reference for dashboard behavior and a technical API contract
for the services that drive job execution.

User-facing summary
-------------------

MuFinder jobs move through the ``Job Dashboard`` (the dock panel at the bottom
of the main window) as calculations are submitted, monitored, and finished:

``defined``
   The job has been created but has not entered the dashboard execution
   lifecycle yet. Submit it from the workflow table that created it (``Run All
   Locally`` / ``Run All on Cluster``, the ``Run Selected ...`` variants, or the
   row menu's ``Run Local`` / ``Run on Cluster``).

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
   remain in the dashboard so their diagnostics can be inspected with
   ``Inspect``. They can be resubmitted or reset.

``stopped``
   A cluster calculation appears to have stopped before producing a successful
   CASTEP completion marker, usually because of walltime or scheduler limits.
   Local jobs that were still active when MuFinder closed, and jobs stopped by
   the active-job watchdog, also end up here. Stopped jobs remain in the
   dashboard for inspection and reset.

``cancelled``
   The user requested cancellation for a queued or running job. Cancelled jobs
   remain in the dashboard and can be resubmitted or reset.

The dashboard shows jobs that still need attention: active jobs, failed jobs,
stopped jobs, and cancelled jobs. Each row lists the job's ``Label``,
``Workflow``, ``Calc Type``, ``Status``, ``Last Checked``, ``Cluster``,
``Remote Job ID``, ``Runtime``, and ``Processes``, followed by per-row
``Resubmit``, ``Retry retrieval``, ``Cancel``, ``Inspect``, ``Reset``, and
``Dismiss`` buttons that are enabled only when they apply (see the action rules
under "Main-window integration" below). Right-clicking selected rows offers the
same actions, each acting on the selected jobs it applies to, plus
``Delete...``. Completed jobs disappear after successful
result handling, and terminal jobs that have been ``Dismiss``-ed are hidden
without losing their records. ``Resubmit`` is available for failed and
cancelled jobs. Cluster resubmission reuses existing remote files when the job
still records a remote directory, so continuation runs can start from the
previous working directory, and failed or stopped cluster jobs that kept their
remote metadata can ``Retry retrieval`` instead of resubmitting. Reuse covers
the remote inputs and outputs only: the scheduler submit script is regenerated
on every submission, so an updated walltime, core count, or memory request
takes effect on resubmission.

Dashboard rows are listed in the order their jobs were created, and that order
survives closing and reopening the project. Each job records ``created_at``
when it is added to the project; unlike ``submitted_at`` it never changes, so
resubmitting, resetting, or continuing a job does not move its row. Loading a
project sorts jobs by ``(created_at, id)``; the job files themselves are named
by random UUID, so their on-disk order means nothing. Jobs created in quick
succession get strictly increasing timestamps (``next_job_created_at``), so a
batch keeps its order even on a coarse system clock. Jobs saved before the
field existed get it backfilled on load, from their first appearance in the
provenance log (keeping a batch's submission order) or else their earliest
lifecycle timestamp, and the next save persists it.

Pressing Delete or Backspace (or choosing ``Delete...``) with dashboard rows
selected applies the action that fits each job's state: active jobs
(``submitting``, ``queued``, ``running``, ``retrieving``) are cancelled,
failed, stopped and cancelled jobs are reset, and any other job has its records
deleted. One
summary dialog lists those actions with their counts before anything happens
and offers to clean the output directories of the jobs being reset; cancelling
it leaves every selected job untouched. Imported jobs cannot be reset and are
reported as left unchanged. A ``Cancel`` or ``Dismiss`` button on a single row
asks for confirmation; on a multi-row selection the "all selected jobs / this
job only / cancel" prompt serves as that confirmation.

The pristine- and muon-relaxation tables additionally offer a per-row
``Continue relaxation…`` action for ``completed``, ``failed``, ``stopped``,
or ``cancelled`` rows: it edits that row's parameters in place and resumes
the same job from its CASTEP checkpoint rather than starting over, so it is
the recovery path that keeps a run directory's ``.check``/``.castep`` files
intact (unlike ``Reset``, whose optional cleanup deletes them). See
:doc:`implemented_functionality` for the full user-facing description and
:ref:`job-continuation-primitive` below for the technical contract.

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
     - ``submitting``, ``retrieving``, ``cancelled``
   * - ``completed``
     - none (validated); ``defined`` via ``prepare_job_continuation``
   * - ``failed``
     - ``submitting``, ``retrieving``, ``cancelled``
   * - ``cancelled``
     - ``submitting``

Two primitives deliberately bypass ``_assert_transition`` and direct-assign
``defined``, which is a target the table above contains nowhere:
``reset_failed_job`` and ``prepare_job_continuation``. Each carries its own
status guard instead; both are described below.

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
- ``prepare_job_continuation`` moves ``completed``, ``failed``, ``stopped``, or
  ``cancelled`` back to ``defined`` for a continuation run, clearing the same
  run metadata as ``reset_failed_job`` **except** ``remote_directory``, which it
  preserves. See :ref:`job-continuation-primitive`.
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

.. _job-continuation-primitive:

Continuation: a job that keeps its remote directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The word "continuation" is used in two distinct senses in this document. The
older one, in the user-facing summary above, describes a *resubmission* that
happens to reuse existing remote files. This section is about the separate
service primitive ``ProjectService.prepare_job_continuation``, which exists so a
finished calculation can be run further from its own CASTEP checkpoint.

The primitive accepts the module constant ``CONTINUABLE_JOB_STATUSES`` —
``completed``, ``failed``, ``stopped``, ``cancelled`` — and direct-assigns
``defined``, clearing ``submitted_at``, ``started_at``, ``completed_at``,
``remote_job_id``, ``num_processes``, ``error_message``, the monitor metadata
(``last_checked_at``, ``last_status_source``, ``last_status_message``), and any
``dismissed_at`` stamp. It deliberately preserves ``remote_directory``.

That preserved field is the point. **A job that is ``defined`` and still carries
a non-null ``remote_directory`` is the durable marker for a continuation run**:
the cluster submission path reads exactly that combination and resubmits into
the existing remote directory — where the previous run's ``.check`` file lives —
rather than recomputing a target, testing whether it already exists, and
offering to delete it. No other code path produces the combination. Every writer
of ``remote_directory`` either sets it while moving a job to ``submitting``
(``JobExecutionController``), clears it for local submission (also
``JobExecutionController``), clears it on ``reset_failed_job``, or touches only
active jobs (``MainWindow._sync_cluster_tracking_from_document``).

``CONTINUABLE_JOB_STATUSES`` is a *third* status set, wider than the shared
reset/dismiss set, because a ``completed`` relaxation is continuable: running
more geometry-optimisation steps from a converged-enough checkpoint is a normal
outcome rather than a recovery from failure. This does not disturb the
reset/dismiss reasoning restated above. That argument depends only on
``reset_failed_job`` and ``dismiss_job`` accepting *the same* statuses as each
other, so that anything the user can dismiss the app can also return to
``defined``; both still read ``TERMINAL_RECOVERABLE_JOB_STATUSES``, and the
derived ``"ready"`` projection for dismissed rows is unchanged. Continuation is
a separate entry point with its own guard, and it also clears ``dismissed_at``,
so continuing a dismissed job makes its dashboard row visible again on the same
terms as a reset.

This is a deliberate divergence from the quantum-corrections precedent, where a
``completed`` stage-1 job is rewound by *deleting* the job records and clearing
the directory (``MainWindow._on_reset_quantum_corrections_to_stage_one``, gated
by ``_QUANTUM_STAGE_ONE_RESETTABLE_STATUSES`` — the same four statuses).
Deleting is right there because the row genuinely starts over. A relaxation
continuation must instead keep the job's identity: the job id is what binds the
workflow-table row to the calculation, and the job record is what carries the
output/remote directory holding the checkpoint the continuation depends on.
Deleting and re-creating would sever the row binding and abandon the directory.

``record_job_continuation`` appends the matching ``job_continuation`` provenance
milestone, recording the job label and whether a remote directory was preserved.
No persisted field is added, so no compatibility-ledger entry is required.

Re-completing a continued job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A continued job completes **more than once**, which changes what the completion
path is allowed to do.

*The derived structure is updated in place.* Both relaxation controllers'
``build_result`` create a ``"{job label} relaxed"`` structure on the document.
On a re-completion they do not create a second one: they locate the structure an
earlier completion of the *same job* produced and refresh its lattice, species,
and positions from the new ``-out.cell``, keeping its id. Nothing else would be
safe — a comparison table stores ``dft_structure_ids``, and downstream workflows
consume derived structures by id, so minting a replacement would orphan the
first ``Result`` and strand every existing reference on a superseded geometry.
Both ``Result`` records end up naming the same structure id, and
``ProjectService.validate_document`` reports no dangling references across the
cycle. The lookup key is ``Structure.properties["source_job_id"]``, with the
derived structures of the job's own earlier runs as a fallback for records
written before that property existed; the label is deliberately not a key,
because a muon-relaxation job is labelled with its site label alone and one
site's two charge states share it. The muon site's
``status``/``relaxation_run_id``/``relative_energy_eV`` are rewritten from the
same completion, so the site and the structure always describe the same run.

*A stale ``-out.cell`` fails the job.* A continuation re-uses its job directory,
so the previous run's relaxed structure is still sitting where the completion
path looks for it: a run that dies before writing a new one would have the old
geometry accepted as its result. Two guards cover this, each closing the other's
blind spot.

The local one is in ``MainWindow._on_local_job_finished``, alongside the
existing missing-file and parse-failure guards (all three are inline because
``ControllerResult`` has no failure channel): it fails a relaxation job whose
local ``-out.cell`` predates the current run's ``started_at``. The comparison
allows a few seconds of slack for filesystem mtime granularity and network-mount
clock offsets, and skips entirely when ``started_at`` is unset — a job the poller
never observed running fails open rather than failing on no evidence.

The remote one is in ``ClusterJobRunner._retrieve_results``, and it is what makes
the local guard meaningful for cluster jobs at all. ``sftp.get`` stamps a
downloaded file with the *download* time, so an unconditionally retrieved
``-out.cell`` always looks newer than ``started_at`` however old the remote file
is. Retrieval therefore orders the remote outputs against the ``{seed}.param``
uploaded at submission: both mtimes are on the cluster's own filesystem, so they
compare directly, up to ``REMOTE_STALE_OUTPUT_SLACK_SECONDS`` of mtime
granularity and node-clock drift. ``.param`` is the marker because it is
uploaded fresh on every submission that does not set ``reuse_existing_files``,
whatever the queue system — unlike ``run.sh``, which ``addqueue`` never receives.

- A remote ``-out.cell`` older than the marker is **not downloaded**, exactly as
  if it did not exist. The local copy keeps its previous-retrieval mtime, and
  the local guard above rejects it.
- A remote ``.castep`` older than the marker **aborts retrieval** as an
  output-unavailable condition (so a job merely missing from the scheduler queue
  still gets its usual grace window). This is the run that never started at all:
  the surviving ``.castep`` still ends with the previous run's success banner and
  would otherwise complete the job on the old geometry under parameters that
  were never applied. The local guard cannot catch it, because a run that never
  started has no ``started_at``.

Both comparisons fail open when no marker is available — a
``reuse_existing_files`` resubmission does not re-upload the ``.param``, and an
SFTP server may not report mtimes at all — which leaves the pre-existing
behaviour unchanged in those cases.

*Parser fields are not all per-run.* CASTEP appends to an existing ``.castep``,
so a continued job's output holds every restart. ``completed``,
``geom_converged`` and ``error_message`` are read from the last restart only;
``num_scf_cycles`` (a ``max()``) and ``warnings`` span the whole file. See the
``parse_castep_output`` docstring for the full split.

*What counts as completion.* A run is ``completed`` only once its last restart
reaches CASTEP's closing timing summary (``Total time``). ``Final energy``,
``NB est. 0K energy`` and ``Geometry optimization completed successfully`` do
not count: CASTEP prints each of them before the final forces and stress tensor.
Treating them as completion meant that a run killed while computing the stress
tensor counted as finished. That happens on a walltime or memory limit, a node
failure, or a transient ``squeue`` omission while the job is still running.
Such a run reached the convergence analysis with no stress. When the last
restart's parameter echo reports ``stress calculation : on``, that restart must
also have printed a stress tensor. An earlier restart's tensor does not count.
A cut-off output now resolves to ``failed`` with "CASTEP did not report
successful completion" and can be continued or resubmitted.

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
  the remote directory, uploads ``.cell``/``.param`` unless
  ``reuse_existing_files`` is true, regenerates and uploads ``run.sh`` on every
  submission on script-based queues so the scheduler resources match the current
  submission, submits to the scheduler, stores a tracked job entry, emits
  ``queued``, and starts a one-minute poll timer.
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
- The directive is applied on resubmission too, because ``run.sh`` is
  regenerated on every submission even when remote inputs are reused.
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
on the row itself, via the ``Reset Failed Jobs...`` entry of the shared
right-click row menu.

The shared implementation lives in ``mufinder.app.workflow_table_utils``:

- ``RESETTABLE_ROW_STATUSES`` — the gate, ``{failed, stopped, cancelled}``,
  matching what ``ProjectService.reset_failed_job`` accepts. A *dismissed*
  terminal job is deliberately excluded: after ``effective_row_status`` its row
  already reads ``"ready"`` and its recovery path is the table's own run actions
  (see :ref:`dismissed-rows-stay-resubmittable`).
- ``resettable_keys`` — picks, from the selected rows, the ones a reset applies
  to: status in ``RESETTABLE_ROW_STATUSES`` and a non-``None`` ``key_for_row``.
  The action is enabled when there is at least one and resets those rows only.
  ``_reset_failed_jobs`` prompts once for the batch, including the destructive
  output cleanup, so the selection must never be narrowed *silently*: for a
  mixed or partially keyed selection the menu entry carries the count
  (``Reset Failed Jobs... (2 of 5)``, from ``partial_count_suffix``), and the
  prompt that follows names the same number of jobs.
- ``bind_reset_failed_context_menu`` / ``open_reset_failed_context_menu`` —
  install and open the menu. Right-clicking a row outside the current selection
  retargets the selection first, so the action always describes the row under
  the cursor. Each widget supplies a ``key_for_row`` (its own submission
  key: a cutoff, a k-point grid, a label, a charge-state-qualified key, or a
  path key) and a ``status_for_row``.
- ``standard_row_actions`` — builds the rest of the shared menu around the reset
  entry (``Run Local``, ``Run on Cluster``, ``Cancel``, ``Inspect Failure``
  above it; ``Configure...``, ``Remove`` below), passed to the bind call as
  ``leading_actions`` / ``additional_actions``. A table leaves out an action by
  not supplying its callback, and a table with no failed-job reset (Quantum
  Corrections) passes ``emit_reset=None``.
- ``RESET_FAILED_TABLE_TOOLTIP`` — the fallback discoverability stamp, set by
  ``bind_reset_failed_context_menu`` on a table that has no tooltip of its own.
  The workflow tables set their own tooltip naming the whole menu. Per-item
  tooltips are unaffected and still take precedence where a widget sets them.
  The energy-barriers diffusion-path table additionally carries the hint in its
  visible hint label.

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
``"failed"`` (so it could not be run) *and* was refused by the reset, leaving
the row permanently stuck.

.. _dismissed-rows-stay-resubmittable:

Dismissed rows stay resubmittable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dismissing a terminal job removes the only generic ``Resubmit``/``Reset``
controls it had, but its workflow table row still points at it through the
definition's persistent label→job-id mapping. Left alone, that row would project
``"failed"`` — not a member of ``READY_WORKFLOW_ROW_STATUSES`` — and its
run actions would skip it forever, with no way back to ``defined``. The
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
