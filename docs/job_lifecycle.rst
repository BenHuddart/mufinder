Job Lifecycle
=============

This document is the technical contract for a job's state: what is stored,
which code may change it, and how it is decided from what can be seen of the
job. The user-facing behaviour of the Job Dashboard, the workflow job tables
and local and cluster execution is described in
:doc:`/getting_started/running_jobs`.

Two rules shape everything below:

1. MuFinder never ends a cluster job because it lost track of it. A cluster
   job it cannot confirm stays active and says why on its dashboard row.
2. Valid CASTEP output completes a job, however tracking went.

Job states
----------

A job (``mufinder.domain.models.Job``) stores one of four ``JobStatus``
values:

.. list-table::
   :header-rows: 1
   :widths: 14 36 50

   * - State
     - Meaning
     - Stored beside it
   * - ``defined``
     - Created, not submitted.
     - –
   * - ``active``
     - Submitted, not yet finished.
     - ``activity`` (``JobActivity``): ``submitting``, ``queued``,
       ``running`` or ``retrieving``, the phase the job was last seen in.
       Display only; it has no transitions of its own.
   * - ``completed``
     - Finished, its result recorded.
     - –
   * - ``ended``
     - Did not finish.
     - ``end_reason`` (``EndReason``): ``walltime``, ``out_of_memory``,
       ``node_failure``, ``castep_error``, ``output_unavailable``,
       ``interrupted``, ``submission``, ``cancelled``, or ``unknown`` when no
       cause was identified.

``activity`` is ``None`` unless the job is active and ``end_reason`` is
``None`` unless it ended. :ref:`job-failure-reasons` describes the causes.

Every job table shows one word for a job,
``services.job_monitoring.job_status_word``: an active job reads as its
activity; an ended job reads ``cancelled``, ``stopped`` when its reason is
``walltime`` or ``interrupted`` (in ``STOPPED_END_REASONS``), and ``failed``
otherwise; any other job reads as its state. The user guide calls ended jobs
as a group jobs that *did not finish*.

Transitions
~~~~~~~~~~~

``ProjectService._assert_transition`` allows these edges:

.. list-table::
   :header-rows: 1

   * - From
     - To
   * - ``defined``
     - ``active`` (submitted); ``ended``, only as ``cancelled``
   * - ``active``
     - ``completed``; ``ended``
   * - ``completed``
     - ``defined`` (a continuation)
   * - ``ended``
     - ``defined`` (a reset or a continuation); ``active`` (resubmitted, or
       found alive or with output)

An active job may move from any activity to any other.

Primitives
~~~~~~~~~~

``ProjectService`` owns every change of state:

- ``submit_job`` makes a ``defined`` or ``ended`` job ``active`` as
  ``submitting``, stamps ``submitted_at`` and clears ``started_at``,
  ``completed_at``, ``error_message`` and ``end_reason``.
- ``mark_job_active(job, activity, remote_job_id=None)`` records what an
  active job is doing and may replace its scheduler job ID. It stamps
  ``started_at`` when the job starts running. It also makes an ``ended`` job
  ``active`` again, clearing ``error_message``, ``end_reason`` and
  ``completed_at`` and keeping the remote metadata. A ``defined`` job becomes
  active only through ``submit_job``. Making an ended job active also clears
  ``dismissed_at``.
- ``complete_job`` makes an ``active`` job ``completed``, stamps
  ``completed_at`` and clears ``error_message``.
- ``end_job(job, reason, message=None)`` makes an ``active`` job ``ended``
  for ``reason``, stamps ``completed_at`` and records ``message`` as its
  error. A cancellation given no message keeps the one the job holds. A
  ``defined`` job can be ended only as ``cancelled``.
- ``reset_failed_job`` returns an ``ended`` job to ``defined`` and clears the
  run metadata: timestamps, scheduler job ID, remote folder, process count,
  error, reason, monitor metadata and ``dismissed_at``.
- ``prepare_job_continuation`` returns a ``completed`` or ``ended`` job
  (``CONTINUABLE_JOB_STATUSES``) to ``defined`` and clears the same metadata
  as a reset except ``remote_directory``; see
  :ref:`job-continuation-primitive`.
- ``dismiss_job`` stamps ``dismissed_at`` on an ``ended`` job that has none.
  It is not a change of state; ``record_job_dismissed`` appends the
  ``job_dismissed`` milestone. See :ref:`dismiss-only-hides`.

Reset and dismiss accept exactly the ``ended`` jobs, so anything the user can
dismiss, the app can also reset.

Who writes a job
~~~~~~~~~~~~~~~~

The layering is codified by the ``JobRunner`` protocol in
``mufinder.services.runner_protocol``:

- **ProjectService** performs every change of state. Nothing else writes
  ``Job.status``, ``Job.activity`` or ``Job.end_reason``.
- **JobExecutionController** writes only bookkeeping fields
  (``execution_target``, ``num_processes``, remote-metadata clears and the
  monitor metadata) and routes every change of state through the primitives,
  keeping the job's ``CalculationRun`` and the journal in step. A run stores
  only a state (``active``, ``completed`` or ``ended``) and is written in the
  same step as its job.
- **Runners never touch ``Job`` objects.** ``LocalJobRunner`` holds a process
  for each job and reports its lifecycle by signals keyed by job id
  (``job_started``, ``job_finished``, ``job_failed``, ``job_cancelled``,
  ``status_changed``). Its report is its own verdict and is applied directly:
  ``status_changed(job_id, "queued")`` for a job still ``submitting`` calls
  ``mark_job_active`` with ``queued`` through
  ``JobExecutionController.mark_local_status_changed``, and any other queued
  notification is a no-op. ``ClusterJobRunner`` holds no job and is not a
  ``JobRunner``: a check reports what the cluster shows as evidence
  (``job_checked``) and :func:`~mufinder.services.job_reconcile.reconcile`
  decides the state from it.

The two runners' ``submit`` error contracts are deliberately different.
``LocalJobRunner.submit`` never raises for bad input: it reports a problem by
emitting ``job_failed``. ``ClusterJobRunner.submit`` may raise, because it does
blocking SSH and SFTP work, and returns the scheduler job ID. Its callers
convert the exception: ``JobExecutionController.submit_cluster_job`` returns a
failed ``ClusterJobSubmissionOutcome`` through ``fail_cluster_submission``, and
``ClusterController`` emits ``job_submit_finished(job_id, None, message)``.

.. _job-reconcile:

Deciding a job's state from evidence
------------------------------------

``mufinder.services.job_reconcile.reconcile(job, evidence)`` is a pure
function: it reads no file, contacts no cluster and changes nothing, and the
same job and evidence always give the same decision. A caller gathers a
``JobEvidence`` and ``JobExecutionController.apply_evidence``, the one place a
decision is applied, applies the returned ``Decision``.

``JobEvidence`` holds what was seen of the job at ``checked_at``:

- ``process`` (local jobs): ``pending``, ``running`` or ``none`` (the local
  runner holds no process for the job). A process the runner sees end is
  reported by the runner itself and is not evidence.
- ``submitting`` (cluster jobs): a submit request has not answered yet.
- ``scheduler`` (cluster jobs): ``pending``, ``running``, ``completed``,
  ``cancelled``, ``timed_out``, ``failed``, ``not_listed`` (neither the queue
  nor accounting lists the job), ``no_job_id`` or ``unavailable`` (the queue
  command failed), with the scheduler's own name or the error in
  ``scheduler_detail`` and ``from_accounting`` for a state read from
  accounting.
- ``output``: ``not_looked``, ``complete`` (parses and reports completion;
  ``parsed`` and ``artifacts`` attached), ``incomplete`` (CASTEP's message in
  ``output_detail``; ``walltime_seen`` when the scheduler log shows a walltime
  kill), ``unreadable``, ``absent`` (no CASTEP output written by this run) or
  ``unreachable`` (the folder could not be read).
- ``remote_job_id`` and ``remote_directory``: the submission the evidence is
  about. Evidence about another submission returns ``None``.
- ``on_request``: the user asked for the check.

``Decision`` holds ``status`` and ``activity`` (the job's own when nothing
changes), the ``end_reason`` and ``message`` of a job it ends, an
``observation`` for the job's last-checked note, and a ``marker`` for an
active job that could not be confirmed. An ``ended`` decision with no reason
is diagnosed by the caller from the job's logs and is ``unknown`` when they
show nothing. ``wants_output(scheduler, on_request=...)`` says from the
scheduler state alone whether the output folder is to be read: never while the
job is listed as pending or running or the scheduler was not heard from, and
after a cancellation or with no scheduler job ID only on request.

In the tables, a decision is written as the status word it gives the job:
``queued``, ``running`` and ``retrieving`` are ``active`` with that activity,
and ``failed``, ``stopped`` and ``cancelled`` are ``ended``; the reason
follows the comma.

An active local job:

.. list-table::
   :header-rows: 1

   * - Evidence
     - Decision
   * - process ``pending``
     - ``queued``
   * - process ``running``
     - ``running``
   * - process ``none``, output ``complete``
     - ``completed``
   * - process ``none``, output not complete
     - ``stopped``, ``interrupted``

An active cluster job:

.. list-table::
   :header-rows: 1

   * - Evidence
     - Decision
   * - ``submitting``
     - no change
   * - scheduler ``unavailable``
     - no change, with a marker
   * - scheduler ``no_job_id``
     - no change, with a marker; ``completed`` when the evidence is
       ``on_request`` and the output is ``complete``
   * - scheduler ``pending``
     - ``queued``, whatever the job was doing
   * - scheduler ``running``
     - ``running``
   * - scheduler ``cancelled``, output not ``complete``
     - ``cancelled``
   * - scheduler ``completed``, ``timed_out`` or ``failed``, output
       ``not_looked``
     - ``retrieving``
   * - scheduler ``not_listed``, output ``not_looked``
     - no change
   * - output ``unreachable``
     - no change, with a marker
   * - output ``complete``
     - ``completed``
   * - scheduler ``timed_out``, output not complete
     - ``stopped``, ``walltime``
   * - scheduler ``failed``, output not complete
     - ``failed``, diagnosed; the message names the scheduler's state
   * - scheduler ``completed`` or ``not_listed``, output ``incomplete``
     - ``stopped``, ``walltime`` when ``walltime_seen``; otherwise ``failed``,
       diagnosed, with CASTEP's message
   * - scheduler ``completed`` or ``not_listed``, output ``unreadable``
     - ``failed``, ``output_unavailable``
   * - scheduler ``completed``, output ``absent``
     - ``failed``, ``output_unavailable``
   * - scheduler ``not_listed``, output ``absent``
     - no change, with the lost-track marker

An ``ended`` job changes only on evidence with ``on_request``: scheduler
``pending`` or ``running`` makes it ``queued`` or ``running`` again, and output
``complete`` completes it. Anything else leaves it as it is. ``defined`` and
``completed`` jobs are never changed.

So a cluster job is never ended for want of evidence (rule 1), and complete
output completes a job whatever the scheduler reports (rule 2). A job whose
cluster is not connected is not checked at all; ``job_staleness`` marks it.

.. _local-job-check:

Local job check
~~~~~~~~~~~~~~~

``JobExecutionController.check_local_jobs`` looks at every active local job
(an ``execution_target`` that is blank or ``"local"``, in any case).
``services.job_evidence.local_job_evidence`` reads the local runner's
``pending_job_ids()`` and ``active_job_ids()`` and, only when the runner holds
no process for the job, its output folder: ``<seed>_farm001.castep`` when a
task-farmed run wrote one, otherwise ``<seed>.castep``. A file last written
before the job's ``submitted_at`` belongs to an earlier run and counts as no
output.

- A job the runner holds stays active.
- A job with no process and complete output is returned as ``finished``: the
  main window records its result through the same completion path as a job the
  runner reported, and that completes it.
- Any other job with no process ends with ``interrupted`` and the message
  *"MuFinder has no local process for this job: it was closed while the job
  ran, or the process ended without reporting. Reset or resubmit it."* Its
  active run is closed, a ``job_stopped`` event is recorded and the monitor
  source is ``"reconciliation"``.

The check runs when a project is opened, before the jobs table is drawn, and
once a minute. The status bar says how many jobs were stopped. Cluster jobs
are not part of it: a cluster job with no process on this machine may still be
queued, running or finished on the cluster.

.. _cluster-checks:

Cluster checks
~~~~~~~~~~~~~~

``JobExecutionController.check_cluster_jobs`` builds a ``CheckTarget`` from the
document for every active job on the connected cluster (scheduler job ID,
remote folder, seed, local output folder, retrieval flags) and passes them to
``ClusterController.check_jobs``. Nothing is registered when a project opens.
A job is a target when it has a local output folder and a remote folder; one
with a scheduler job ID and no recorded folder gets the folder its cluster
profile would give it. A job still being submitted is left out, and so is a job
with no scheduler job ID unless ``Check now`` names it.

Checks are asked for when a project opens and once a minute, when a cluster
connects, from the status bar's queue button (which first reconnects the
clusters of active jobs when none is connected), and from ``Check now``. All
but ``Check now`` are dropped while an earlier check is still on the worker.
``Check now`` is queued, marks its targets ``on_request`` and may name ended
jobs as well as active ones.

On the worker, ``ClusterJobRunner.check_jobs``:

- skips targets of a cluster other than the connected one;
- asks the queue once for every target with a scheduler job ID (SLURM:
  ``squeue --all --jobs=<ids>``, so hidden or unavailable partitions are
  listed);
- looks up every job the queue does not list in accounting in one call
  (``sacct --jobs=<ids> --allocations``, through ``Scheduler.history``, using
  the profile's ``Accounting command path`` when set). Disabled accounting
  finds nothing, and any other ``sacct`` failure counts those jobs as not
  listed; neither disconnects;
- turns the scheduler's state into a ``SchedulerState`` through
  ``Scheduler.map_to_scheduler_state``. SLURM's ``PD``, ``CF``, ``RD``,
  ``RF``, ``RH``, ``RQ``, ``SE``, ``S`` and ``ST`` are ``pending``; ``R``,
  ``CG``, ``RS``, ``SI`` and ``SO`` are ``running``; ``CD`` is ``completed``;
  ``CA`` is ``cancelled``; ``TO`` is ``timed_out``; ``F``, ``NF``, ``OOM``,
  ``BF``, ``DL``, ``PR`` and ``RV`` are ``failed``. Accounting's long names map
  the same way, and a state the backend does not know is ``pending``;
- reads the remote folder of every job ``wants_output`` names:
  :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs` downloads
  the error files, scheduler log, ``.castep`` and the optional outputs the
  retrieval flags request, parses them, and returns the ``OutputState``. The
  secondary outputs are parsed once here and travel with the evidence. A folder
  that raises is ``unreachable`` and the pass goes on;
- emits ``job_checked(job_id, evidence)`` per job. A job the scheduler reports
  finished is reported twice, first with the scheduler's answer alone so it
  reads ``retrieving`` during the download, then with what its folder held;
- ends the pass with ``queue_checked`` (``checked_at``, elapsed runtimes, the
  ids checked).

When the queue command fails, every target is reported with scheduler
``unavailable``, the runner disconnects and ``poll_failed`` raises a warning.
No job changes state; active cluster rows go stale and the status bar reads
``Connection to <name> lost`` in red while that cluster has active jobs.

``JobExecutionController.apply_evidence`` then:

- drops evidence about a job being submitted or about another submission of
  it;
- applies a change through the primitives. An active job's activity follows
  the decision; ``running`` stamps the run's start. An ``ended`` decision
  closes the active run and records ``job_stopped`` for ``walltime`` or
  ``interrupted``, ``job_cancelled`` for ``cancelled`` and ``job_failed``
  otherwise;
- returns complete output as ``finished`` rather than completing the job: the
  main window records the result through the same completion path as a local
  job;
- makes an ended job that ``Check now`` finds listed again, or with complete
  output, active through ``mark_job_active`` under a *fresh*
  ``CalculationRun``, so the ended run keeps its outcome;
- stamps the monitor metadata with ``checked_at``, the source
  (``"scheduler_poll"``, or ``"retrieval"`` when the folder was read) and the
  decision's observation, for example *"scheduler reports RUNNING"*;
- keeps the decision's marker in memory for the dashboard. It is never saved;
  the next check rebuilds it.

Submission
----------

Local submission
~~~~~~~~~~~~~~~~

``JobExecutionController.submit_local_jobs`` accepts ``defined`` jobs, or the
``eligible_statuses`` it is given (dashboard resubmission passes ``ended``).
It resolves each job's output folder and seed, submits it (``active``,
``submitting``) with target ``local``, clears any remote metadata, creates an
active ``CalculationRun`` and hands the process request to
``LocalJobRunner.submit``. The batch is recorded in the journal once at least
one job is accepted; a job the runner refuses ends with ``submission``.

Cluster submission
~~~~~~~~~~~~~~~~~~

``services.cluster_submission_planner.plan_cluster_submission`` computes the
remote targets, has the worker check which remote folders exist, asks any
overwrite questions up front and builds a ``ClusterJobSubmission`` per job. A
job whose existence check raises or does not finish within three minutes is not
submitted, keeps its state and is listed in the "Cluster submission failed"
message. A folder a job submits into as it is (a continuation's preserved
``remote_directory``) is never deleted, and a start-again job whose own folder
is such a folder fails instead of resuming from the checkpoint it meant to
discard.

``JobExecutionController.prepare_cluster_submission`` accepts a ``defined`` or
``ended`` job, submits it, sets the cluster, process count and remote folder
and creates an active run. ``ClusterController.submit_batch`` uploads and
submits each job on the worker. ``finalize_cluster_submission`` records the
scheduler job ID and moves the job to ``queued``;
``fail_cluster_submission`` ends it with ``submission`` and closes its run.
``submit_cluster_job`` does the same synchronously for callers off the GUI
thread.

A modal dialog reports progress and offers Cancel: the job the worker has in
hand finishes submitting and its result is applied, and jobs prepared but not
yet submitted are cancelled. When the worker reports no progress for three
minutes the dialog gives the batch up through
``ClusterController.abandon_batch``, which stops the worker as Cancel does, and
cancels the jobs it has no result for. A job of that batch the worker submits
after all is cancelled on the scheduler through
``ClusterJobRunner.cancel_scheduler_job`` and the project row is left
``cancelled``; if that cancel fails, the window warns with the job and its
scheduler job ID.

From ``submit_batch`` until a job's result arrives ``ClusterController.is_submitting``
holds for it. Such a job has no scheduler job ID by design: checks leave it
out, evidence about it changes nothing, and it carries no staleness marker.

Retrieval flags
~~~~~~~~~~~~~~~

Which secondary outputs a job produces is one value,
:class:`~mufinder.services.job_artifacts.RetrievalFlags`: ``retrieve_den_fmt``
(UEP source jobs), ``retrieve_out_cell`` (pristine and muon relaxations),
``retrieve_phonon`` (quantum-phonon jobs) and ``retrieve_ts`` (transition-state
and NEB jobs), set by ``RetrievalFlags.for_calc_type``, plus
``task_farming_enabled`` and ``num_farms``, because a farmed run writes
``<seed>_farmNNN.<ext>``. ``submit_local_jobs`` takes a ``retrieval_for_job``
callback, ``ClusterJobSubmission`` carries a ``retrieval`` field, and a cluster
check carries the value on its ``CheckTarget``. ``LocalJobRunner`` keeps it with
the job's process for ``RetrievalFlags.collect`` and ignores
``retrieve_den_fmt``.

Runners
-------

Local runner behavior
~~~~~~~~~~~~~~~~~~~~~

``LocalJobRunner`` is an in-memory queue around ``QProcess``:

- A request is validated for the CASTEP executable, ``mpirun`` when several
  cores are requested, and the job folder.
- CASTEP starts with the user's environment. The thread-pool variables the
  entry point set for MuFinder itself (``OPENBLAS_NUM_THREADS``,
  ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``; see :doc:`/quality_and_automation`,
  Native thread pools) are removed; a value the user set is passed on.
- A valid job waits in a pending queue, and the runner emits ``queued`` when
  concurrency is saturated. Starting a process emits ``job_started`` and
  ``running``.
- Cancelling a pending job emits ``job_cancelled`` without starting it;
  cancelling a running one kills the process and the finish handler emits
  ``job_cancelled``.
- A non-zero exit, missing ``.castep`` output, a parser failure or an
  unsuccessful CASTEP result emits ``job_failed``. A ``QProcess``
  ``FailedToStart`` error is reported at once as ``job_failed`` with *"The
  CASTEP process failed to start: {detail}. Check the CASTEP executable path
  and file permissions."*, and the next queued job starts, so a job cannot stay
  ``submitting`` behind a process that never launched.
- A completed CASTEP output emits ``job_finished`` with the parsed result and
  the secondary outputs parsed once through
  :func:`~mufinder.services.job_artifacts.collect_job_artifacts`, so the
  completion handler never re-reads them (see :doc:`/architecture`, Workflow
  controllers → Parse-once off the GUI thread).

Runner state is not persisted. When the application closes with local jobs
active, the runner cancels them as best it can; the :ref:`local-job-check`
settles them when the project is next opened.

Cluster runner behavior
~~~~~~~~~~~~~~~~~~~~~~~

``ClusterJobRunner`` is an SSH and scheduler facade. It keeps the connection
and nothing about any job: every operation is told which job it is about. Its
paramiko work is blocking, so it runs on a worker thread owned by
``mufinder.app.cluster_controller.ClusterController``, which mirrors the
connection state (``is_connected``, ``connected_cluster_name``,
``connected_username``) on the GUI thread.

No GUI-thread call blocks on the worker. High-level flows are fire-and-forget
with a purpose-built signal (``connect_async`` / ``connect_finished``,
``check_remote_dirs`` / ``remote_dirs_checked``, ``check_jobs`` /
``job_checked`` and ``queue_checked``, ``cancel`` / ``job_cancelled``,
``submit_batch`` with its progress signals). Low-level helpers use
``ClusterController.operation_finished(request_id, kind, result, error)``,
which fires on every exit path so a control disabled while an operation is
pending is always re-enabled. The one blocking call is
``ClusterController.shutdown`` at teardown. Connecting runs behind a modal
"Connecting…" dialog; an unknown host key is prompted for, saved, and the
connection retried.

- ``connect_to_cluster`` creates the SSH connection and scheduler backend.
- ``submit`` validates the local folder and seed, creates the remote folder,
  uploads ``.cell`` and ``.param`` unless ``reuse_existing_files`` is set,
  regenerates ``run.sh`` on script-based queues on every submission, submits
  and returns the scheduler job ID.
- ``cancel(job_id, remote_job_id)`` sends the scheduler cancel command and
  emits ``job_cancelled``; with no scheduler job ID nothing is sent or
  emitted. ``cancel_scheduler_job(remote_job_id)`` sends it and emits nothing,
  for a job the project already shows ``cancelled``.
- ``check_jobs(targets)`` reports evidence; see :ref:`cluster-checks`.
- ``scheduler_job_is_active`` guards the deletion of an old run's folder before
  a resubmission. An ``sacct`` failure other than disabled accounting is an
  error there, and the folder is not deleted.

.. _slurm-memory-requests:

SLURM memory requests
~~~~~~~~~~~~~~~~~~~~~

SLURM enforces a per-job memory limit and kills a CASTEP task that exceeds it,
so the generated ``run.sh`` supports an optional per-CPU request, **Memory per
CPU (GB)** in the cluster submission dialog, seeded from a SLURM profile's
``default_mem_per_cpu_gb``.

- ``mem_per_cpu_gb`` travels from ``ClusterSubmissionConfig`` through
  ``ClusterJobSubmission`` and ``ClusterJobRunner.submit`` to
  :func:`~mufinder.services.script_template.render_job_script`, which writes
  ``#SBATCH --mem-per-cpu`` after the last ``#SBATCH`` line on the slurm path
  only. A template with its own ``--mem``, ``--mem-per-cpu`` or
  ``--mem-per-gpu`` is left alone. It applies on resubmission too, because
  ``run.sh`` is regenerated every time.
- ``None`` (every profile's default) writes nothing, so the partition's own
  default holds.
- :func:`~mufinder.services.script_template.format_mem_per_cpu_directive`
  writes whole gigabytes as ``6G`` and anything else as whole megabytes
  (``7.5`` becomes ``7680M``), since SLURM rejects fractional gigabytes.
- MuFinder runs pure-MPI CASTEP and never sets ``--cpus-per-task``, so one CPU
  is one MPI rank and the value is the per-rank ceiling; ``--mem`` (per node)
  would have to change with node packing.

The addqueue ``memory_gb`` control ("Memory per core") is passed as
``addqueue -m`` instead. The dialog shows one or the other, and
``mem_per_cpu_gb`` is ``None`` on addqueue clusters.

Continuation and completion
---------------------------

.. _job-continuation-primitive:

Continuation: a job that keeps its remote directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

"Continuation" here is the service primitive
``ProjectService.prepare_job_continuation``, which lets a finished calculation
run further from its own CASTEP checkpoint. (:doc:`/getting_started/running_jobs`
also uses the word for a resubmission that reuses remote files.)

The primitive accepts ``CONTINUABLE_JOB_STATUSES`` — ``completed`` and
``ended`` — and returns the job to ``defined``, clearing ``submitted_at``,
``started_at``, ``completed_at``, ``remote_job_id``, ``num_processes``,
``error_message``, ``end_reason``, the monitor metadata and ``dismissed_at``.
It keeps ``remote_directory``.

**A ``defined`` job with a ``remote_directory`` is the durable marker of a
continuation run**: the cluster submission path resubmits into that folder,
where the previous run's ``.check`` file lives, instead of computing a target
and offering to delete an existing one. No other path produces the
combination: every writer of ``remote_directory`` sets it while submitting,
clears it for a local submission or a reset, or touches only active jobs (a
cluster check filling in the folder of a job that recorded none).

A completed relaxation is continuable because running more
geometry-optimisation steps from its checkpoint is a normal outcome, not a
recovery. Continuing a dismissed job clears ``dismissed_at``, so its dashboard
row comes back as it would after a reset. The job keeps its identity, because
its id binds the workflow row and its record carries the folder holding the
checkpoint. ``record_job_continuation`` appends the ``job_continuation``
milestone with the label and whether a remote folder was kept.

.. _remote-checkpoint-probe:

Checkpoint probe
~~~~~~~~~~~~~~~~

The ``Continue relaxation…`` dialog reports whether the job's checkpoint is
there: found, not found, or could not be verified. The probe always runs in
the controller before the dialog is built.

- For a cluster job, ``ClusterJobRunner.remote_file_exists`` stats one remote
  path, and ``ClusterController.check_remote_files`` runs it on the worker for
  a batch of ``(job_id, remote_path)`` targets and emits
  ``remote_files_checked``.
- ``CheckpointProbeResult`` (``found()``, ``absent()``,
  ``unavailable(reason)``) carries a ``CheckpointProbeStatus``. Not connected,
  no remote folder recorded and SSH errors are ``unavailable``; a recorded
  folder that no longer exists is ``absent``.
- A local job is probed with a ``pathlib.Path`` check for ``{seed}.check``
  (``unavailable`` when its folder cannot be resolved).

The probe shares the SSH worker with the geometry-progress reads of running
relaxations, so a probe, check or submission never waits behind a large
output: ``SSHConnection.read_bytes`` reads a byte range in one pipelined batch
of at most ``max_bytes``, and ``ClusterJobRunner.read_geometry_progress`` reads
at most ``REMOTE_READ_CHUNK_BYTES`` (2 MiB) of one job per turn of the worker's
event loop and queues the rest behind whatever is waiting. A job's progress is
emitted through ``geometry_progress_read`` once read to the end. A disconnect,
a failed read or another project being opened ends that read with ``None``,
and the next refresh carries on from the bytes already read.

.. _re-completing-continued-job:

Re-completing a continued job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A continued job completes more than once.

*The derived structure is updated in place.* The relaxation controllers'
``build_result`` find the ``"{job label} relaxed"`` structure an earlier
completion of the same job produced and refresh its lattice, species and
positions from the new ``-out.cell``, keeping its id, because comparison tables
and downstream workflows refer to it by id. The key is
``Structure.properties["source_job_id"]``, with the derived structures of the
job's earlier runs as a fallback; the label is not a key, since one site's two
charge states share it. The muon site's ``status``, ``relaxation_run_id``,
``total_energy_eV`` and ``relaxed_fractional_position`` are rewritten from the
same completion. Removing the job's records returns the site to ``staged``.

*A stale ``-out.cell`` fails the job.* Two guards stop a run that died before
writing a new ``-out.cell`` from taking the previous run's geometry:

- Locally, the relaxation controllers' ``validate_artifacts`` fail a job whose
  ``-out.cell`` predates the run's ``started_at``, with a few seconds of
  slack; it is skipped when ``started_at`` is unset.
- On the cluster,
  :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs` compares
  remote outputs with the ``{seed}.param`` uploaded at submission (both on the
  cluster's clock, up to ``REMOTE_STALE_OUTPUT_SLACK_SECONDS``). An older
  ``-out.cell`` is not downloaded, so the local guard rejects the old copy; an
  older ``.castep`` is not parsed and counts as no output from this run.
  ``sftp.get`` stamps a download with the download time, which is why this
  comparison is made on the cluster. Both guards fail open when no marker is
  available (a ``reuse_existing_files`` resubmission, or a server that reports
  no mtimes).

*Parser fields are not all per-run.* CASTEP appends to its ``.castep``, so
``completed``, ``geom_converged`` and ``error_message`` come from the last
restart while ``num_scf_cycles`` and ``warnings`` span the file (see
``parse_castep_output``).

*What counts as completion.* A run is complete only once its last restart
prints CASTEP's closing timing summary (``Total time``); ``Final energy`` and
``Geometry optimization completed successfully`` come before the final forces
and stress. When the last restart's parameter echo reports ``stress
calculation : on``, that restart must also have printed a stress tensor. A
cut-off output ends the job with *"CASTEP did not report successful
completion"*.

*Failure markers.* A line of the last restart containing ``ERROR``,
``FAILED``, ``did not converge`` or ``panic`` marks the run as failed, except
inside a boxed CASTEP warning (between two rules of ``?`` under a ``Warning in
<routine>`` heading), whose heading line is kept in ``warnings``.

If building or storing a finished job's result raises while the job is still
active, ``JobExecutionController.fail_unrecorded_finish`` ends it with *"The
job's output was retrieved, but MuFinder could not record the result:
<error>"*, removes any result the attempt added and clears a run's reference
to it. ``Check now`` can then record it again from the cluster. A job that had
already completed keeps its result.

The dashboard and workflow rows
-------------------------------

Main-window integration
~~~~~~~~~~~~~~~~~~~~~~~

The main window wires runner signals into ``JobExecutionController``: local
``job_finished`` goes to the completion path that builds the workflow result
and calls ``complete_job_with_result``; local ``job_failed`` to ``fail_job``,
``status_changed`` to ``mark_local_status_changed`` and ``job_started`` to
``mark_started``; local and cluster ``job_cancelled`` to ``cancel_job``;
cluster ``job_checked`` to ``apply_evidence``, with a ``finished`` update going
to the same completion path. ``queue_checked`` ends a check, and the status bar
shows ``JobExecutionController.take_check_summary``: one changed job is named
(*"Cluster job stopped at its walltime limit: <label>"*), several are counted
by outcome (*"Checked 5 cluster jobs: 1 completed, 2 started running"*). A
check that changed nothing leaves the status bar alone.

Each change to a job is applied at once, but the heavy work after it —
refreshing the journal, the jobs table and the workflow views, and the
autosave — is coalesced: a quiet-period debounce
(``_DASHBOARD_FLUSH_DEBOUNCE_MS``, 300 ms) flushes an isolated event soon, and a
minimum interval (``_DASHBOARD_MIN_FLUSH_INTERVAL_MS``, 2 s) bounds a burst to
about one flush every two seconds. The deadline only ever moves earlier, so a
steady stream cannot postpone the flush forever. A pending flush also runs
before save, close and a project switch. The journal touches only the cards a
flush added or changed, and catches up when its page is shown.

Dashboard
~~~~~~~~~

The Job Dashboard lists ``active`` and ``ended`` jobs without a
``dismissed_at`` stamp. With ``Show dismissed`` ticked (above the table, not
saved) it lists the dismissed ones too, with a greyed label whose tooltip says
when the job was dismissed. The ``Status`` cell shows the status word, with the
reason for a failed or stopped job (``failed · SCF did not converge``,
``stopped · walltime``) and geometry progress for a running relaxation. Its
tooltip explains the failure and carries the last check's age, source and
message. An active cluster job that ``JobExecutionController.staleness``
(``services.job_monitoring.job_staleness``) gives a reason for gets a stale
icon, with the reason in the tooltip: its cluster is disconnected, its last
check left a marker, it has no scheduler job ID, or its last check is older
than 180 s. A job being submitted is never stale. The table is redrawn when
the cluster connection opens or closes.

Each row has one action button and a ``⋯`` menu:

.. list-table::
   :header-rows: 1

   * - Action
     - Applies to
   * - ``Cancel`` (the button for active jobs)
     - ``active``. The scheduler is asked only about a job with a scheduler job
       ID on the connected cluster (``can_cancel_on_cluster``); any other
       cluster job is cancelled in the project and the status bar says so. A
       job being submitted is cancelled through the worker.
   * - ``Inspect…`` (the button for failed and stopped jobs)
     - ``ended`` other than ``cancelled`` (the failed-job dialog, see
       :ref:`job-failure-reasons`), and running pristine and muon relaxations
       (their geometry progress, see :ref:`geometry-optimisation-progress`).
   * - ``Check now`` (the button when
       ``services.job_monitoring.has_unretrieved_cluster_output`` holds: an
       ended cluster job with a remote folder whose reason is
       ``output_unavailable`` or ``interrupted``)
     - ``JobExecutionController.can_check_now``: an active or ended cluster job
       with a remote folder on the connected cluster, not being submitted; an
       ended local job with an output folder. A local job is decided at once
       from its folder.
   * - ``Resubmit…`` (the button for cancelled jobs)
     - ``ended``. Relaxations first ask whether to continue or start again
       (see :ref:`resubmit-continue-or-start-again`).
   * - ``Reset…``
     - ``ended``. The clean-up prompt adds ``unretrieved_output_note`` with
       ``Resetting`` when ``has_unretrieved_cluster_output`` holds.
   * - ``Dismiss``
     - ``ended`` without a ``dismissed_at`` stamp. When ``has_unretrieved_cluster_output`` holds, the
       confirmation adds ``services.job_monitoring.unretrieved_output_note``:
       the output may still be on the cluster and ``Check now`` retrieves it.

Delete and Backspace cancel active jobs and dismiss the rest.

.. _per-row-reset-affordance:

Every workflow table has a per-row reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ended job's recovery does not depend on the dashboard: every workflow table
offers ``Reset Failed Jobs…`` in its shared right-click row menu, from
``mufinder.app.workflow_table_utils``:

- ``RESETTABLE_ROW_STATUSES`` — ``{failed, stopped, cancelled}``, the status
  words of an ended job, dismissed or not. The same rows can also be run
  directly (see :ref:`ended-rows-are-submittable`).
- ``resettable_keys`` — the selected rows a reset applies to: a resettable
  status and a key. A mixed or partly keyed selection shows its count in the
  entry (``Reset Failed Jobs... (2 of 5)``, from ``partial_count_suffix``),
  and the prompt that follows names the same number, so the selection is never
  narrowed silently.
- ``bind_reset_failed_context_menu`` / ``open_reset_failed_context_menu`` —
  install and open the menu; right-clicking outside the selection retargets it
  first. Each widget supplies ``key_for_row`` and ``status_for_row``.
- ``standard_row_actions`` — the rest of the menu (``Run Local``, ``Run on
  Cluster``, ``Cancel``, ``Inspect Failure`` above; ``Configure…``, ``Remove``
  below); a table leaves an action out by not supplying its callback.
- ``RESET_FAILED_TABLE_TOOLTIP`` — the fallback tooltip for a table with none
  of its own.

The main window resolves a widget's ``reset_failed_requested`` keys through the
workflow's ``WorkflowJobController.jobs_for_keys`` and hands the jobs to the
shared reset handler. Muon relaxation binds the menu once per charge-state
table. After a reset every workflow table is repainted.

.. _quantum-singlepoint-records:

Quantum Corrections binds each row to one phonon job and offers ``Reset Failed
Jobs…`` as a button beside the run buttons. The reset's folder clean-up refuses
a folder another job uses unless that job is reset in the same batch or is a
*directory companion* (``WorkflowJobController.directory_companions``): the
``castep_quantum_singlepoint`` job an older project ran in the row's seed
folder before its phonon job, unless it is still active. Cleaning the phonon
job's folder removes the singlepoint's outputs too, while its record stays in
the dashboard. Without clean-up the folder and any ``.check`` file are kept,
and ``CONTINUATION : DEFAULT`` resumes from the checkpoint on a local rerun.
The reset clears ``remote_directory``, so a cluster resubmission computes a new
target; a full continuation is not offered for Quantum Corrections.

A project that holds ``castep_quantum_singlepoint`` jobs loads them as records:
they stay in the dashboard, collect normally when they finish, and can be
dismissed or deleted. A row with a phonon job keeps it, whatever its state. A
row with only a singlepoint job reads ``ready``; its phonon job is written to a
folder of its own (the seed gains ``_phonon`` when the singlepoint's folder
still exists). A row whose singlepoint is still active shows that job's status
and refuses submission until the singlepoint finishes or is cancelled.

.. _ended-rows-are-submittable:
.. _dismiss-only-hides:

Ended rows are submittable; Dismiss only hides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A workflow row's status comes from its job's state alone, through
``effective_row_status(job)`` in ``mufinder.app.workflow_submission``: a row
with no job reads ``ready`` (or the workflow's own word for it), every other
row reads ``job_status_word(job)``. Every workflow's job-to-row projection goes
through it.

A row can be submitted when its status is in ``READY_WORKFLOW_ROW_STATUSES``:
``ready``, ``defined``, ``not generated``, and the three words of an ended job,
``failed``, ``stopped`` and ``cancelled``. ``Run All`` includes them. Their
inputs stay as the job ran them until a reset: ``workflow_row_is_locked`` keeps
failed and stopped rows read-only (a cancelled sweep or relaxation row stays
editable), and the Configure dialogs of Hyperfine Coupling, Quantum Corrections
and Energy Barriers edit only ``UNSUBMITTED_WORKFLOW_ROW_STATUSES`` rows
(``ready``, ``defined``, ``not generated``). The dashboard's ``Resubmit…``
reruns a job's own inputs, so an edit to an ended row would otherwise be
ignored, and a changed farm count would disagree with the job's ``NUM_FARMS``.

``prepare_jobs_for_resubmission`` on ``WorkflowControllerContext`` is the lazy
reset. ``WorkflowJobController.submit`` calls it for the plan's ended jobs just
before dispatch; the main window runs the same clean-up prompt as ``Reset``
(with the unretrieved-output note when it applies) and then
``reset_failed_job``, so the jobs reach the submit backends as ``defined``.
Cancelling the prompt aborts the submission. Imported jobs are never reset.

``submit`` then runs ``materialize`` again for the plan's ready keys, because a
controller rewrites a job's inputs only while it is ``defined``; without it the
row would resubmit the failed run's ``.cell`` and ``.param``. The row keeps its
job, and a reset cutoff-convergence job stays in its ``job_ids_by_cutoff``
mapping.

Dismissing an ended job only hides it from the Job Dashboard. Nothing else
reads ``dismissed_at``: the workflow row reads the same, and the job stays
resubmittable from it. Every change of state clears the stamp (reset,
continuation, and an ended job made active again by ``Check now``), so a job
that runs again is always listed. ``Show dismissed`` lists the hidden jobs
with their usual actions, so a dismissed job whose output is still on the
cluster can be checked again.

Job bindings across definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reconfiguring a workflow registers a new definition with its own job mapping,
so two definitions can bind the same row to different jobs. Before a controller
reads its active definition's bindings it copies them across from equivalent
definitions, choosing by job state (:mod:`mufinder.services.definition_mappings`;
see :doc:`/architecture`, Definition mappings and row removal): an ``active``
job beats a ``completed`` one, which beats an ``ended`` one, which beats a
``defined`` one; the two relaxation workflows rank a ``defined`` job above an
``ended`` one, so a re-staged row keeps its fresh job. A tie keeps the binding
the active definition holds, and dismissal does not change a job's rank.

What keeps a job from staying active
------------------------------------

No separate safeguard watches for stuck jobs. Each of these follows from
deciding a job's state from evidence:

- A local job with no process is completed from complete output and stopped
  otherwise, on project open and once a minute (:ref:`local-job-check`).
- ``LocalJobRunner`` reports a ``FailedToStart`` error at once, so a job cannot
  stay ``submitting`` behind a process that never launched.
- A cluster job is checked on project open, on connect and once a minute, and
  ends when the scheduler or its output says it ended (:ref:`cluster-checks`).
- A cluster job that cannot be confirmed (cluster disconnected, no scheduler
  job ID, listed nowhere with no output) stays active, shows why on its row,
  and can be cancelled in the project.
- An ended job can be checked again with ``Check now``; complete output
  completes it.

These are pinned by ``tests/services/test_job_reconcile.py``,
``tests/services/test_job_evidence.py``,
``tests/services/test_job_execution_controller.py``,
``tests/services/test_project_service_transitions.py``,
``tests/services/test_cluster_runner.py``,
``tests/app/test_cluster_job_checks.py`` and
``tests/app/test_job_check_wiring.py``.
