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
     - ``defined`` (run again: started again or continued); ``active``
       (found alive or with output)

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
  error, reason, monitor metadata and ``dismissed_at``. It is how a job starts
  again; no user action calls it on its own.
- ``prepare_job_continuation`` returns a ``completed`` or ``ended`` job
  (``CONTINUABLE_JOB_STATUSES``) to ``defined`` and clears the same metadata
  as ``reset_failed_job`` except ``remote_directory``; see
  :ref:`job-continuation-primitive`.
- ``dismiss_job`` stamps ``dismissed_at`` on an ``ended`` job that has none.
  It is not a change of state; ``record_job_dismissed`` appends the
  ``job_dismissed`` milestone. See :ref:`dismiss-only-hides`.

Running again and dismissing accept exactly the ``ended`` jobs, so anything
the user can dismiss, they can also run again from its row.

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
- ``on_request``: the user asked for the check (``Check now`` or ``Refresh
  Queue``), or the check that opens a project is re-reading a local job that
  ended as ``interrupted`` (see :ref:`local-job-check`).

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
``complete`` completes it. Anything else leaves it as it is. This holds for a
job the user cancelled as for any other ended job. ``defined`` and
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
  jobs controller records its result through the same completion path as a job the
  runner reported, and that completes it.
- Any other job with no process ends with ``interrupted`` and the message
  *"MuFinder has no local process for this job: it was closed while the job
  ran, or the process ended without reporting. Run it again from its row or the
  Job Dashboard."* Its
  active run is closed, a ``job_stopped`` event is recorded and the monitor
  source is ``"reconciliation"``.

The check runs when a project is opened, before the jobs table is drawn, and
once a minute. The status bar says how many jobs were stopped. The check that
opens a project (``check_local_jobs(include_interrupted=True)``) also reads the
folder of every local job that ended as ``interrupted``, dismissed or not, as
``Check now`` would, with its evidence marked ``on_request``: a CASTEP process
that outlived MuFinder may have finished the job since. Complete output from
that run makes the job active again under a fresh ``CalculationRun`` and is
recorded through the same completion path; output last written before the
job's ``submitted_at`` still counts as no output, and anything else leaves the
job as it is. Cluster jobs are not part of the local check: a cluster job with
no process on this machine may still be queued, running or finished on the
cluster.

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
with no scheduler job ID unless the user asked for the check.

Checks are asked for in three ways:

- *Automatically*, when a project opens, once a minute and when a cluster
  connects: the active jobs with a scheduler job ID, not ``on_request``. Such a
  check is dropped while an earlier check is still on the worker.
- *By ``Check now``*, through ``JobExecutionController.check_now``: the jobs it
  names, marked ``on_request``. The dashboard offers it only on ended cluster
  jobs whose output may still be on the cluster, and connects to their cluster
  first.
- *By the status bar's ``Refresh Queue``*, through
  ``JobExecutionController.refresh_cluster_jobs`` (after reconnecting the
  clusters of ``services.cluster_status.refreshable_cluster_names`` when none
  is connected): ``Check now`` for every job
  on the connected cluster. Every active job is a target, one with no
  scheduler job ID included, and so is every ended job ``can_check_now``
  accepts, dismissed or not, all marked ``on_request``. An ended job whose last
  submission was rejected (``submission``) is left out: it never reached the
  scheduler, so anything under its ID or in its folder belongs to an earlier
  run.

``Check now`` and ``Refresh Queue`` are queued behind a check already on the
worker, never dropped. A refresh bounds its cost: the scheduler is asked about
every target with a scheduler job ID in the pass's one queue call and one
accounting call, but the folder of an ended job whose output was already read
to a verdict is not read again. ``services.job_monitoring.cluster_output_judged``
decides that from what is stored: the job ended for ``walltime``,
``out_of_memory``, ``node_failure``, ``castep_error`` or ``unknown`` (reasons a
cluster job is given only from what its folder and logs held), its monitor
source is ``"retrieval"`` (the check that ended it read the folder) and its
result was not lost in recording (``result_not_recorded``). Its target
has ``read_output`` off, and such a job with no scheduler job ID is left out,
since there is nothing to ask. A job whose output could not be read
(``output_unavailable``), whose result could not be recorded, that MuFinder
lost track of (``interrupted``), that was cancelled, or whose folder was never
read is read again on every refresh.
``Check now`` on a single job always reads its folder.

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
- reads the remote folder of every job ``wants_output`` names whose target has
  ``read_output`` on:
  :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs` downloads
  the error files, scheduler log, ``.castep`` and the optional outputs the
  retrieval flags request, parses them, and returns the ``OutputState``. The
  secondary outputs are parsed once here and travel with the evidence. A folder
  that raises is ``unreachable`` and the pass goes on;
- emits ``job_checked(job_id, evidence)`` per job. A job the scheduler reports
  finished is reported twice, first with the scheduler's answer alone so it
  reads ``retrieving`` during the download, then with what its folder held;
- ends the pass with ``queue_checked`` (``checked_at``, elapsed runtimes, the
  ids checked, and ``refresh`` for a ``Refresh Queue`` pass). The Job Dashboard takes the elapsed runtimes of the jobs the
  pass checked and keeps every other job's from its own last check, so
  ``Check now`` on one job leaves the other running jobs' runtime cells
  counting from the scheduler's time. A running cluster job's runtime cell
  counts on from its elapsed time at the check that reported it; a time read
  before the job's latest submission belongs to an earlier run and is ignored.

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
  jobs controller records the result through the same completion path as a local
  job;
- makes an ended job that ``Check now`` or ``Refresh Queue`` finds listed
  again, or with complete output, active through ``mark_job_active`` under a
  *fresh* ``CalculationRun``, so the ended run keeps its outcome;
- stamps the monitor metadata with ``checked_at``, the source
  (``"scheduler_poll"``, or ``"retrieval"`` when the folder was read) and the
  decision's observation, for example *"scheduler reports RUNNING"*. On an
  ended job awaiting a check's answer (see :ref:`awaiting-answer`) only
  ``checked_at`` is written, unless the scheduler answered: a scheduler state
  other than ``unavailable`` or ``no_job_id``;
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
scheduler job ID. The worker makes that cancel as soon as the submit returns,
before it starts any batch queued behind, which may submit the same job again
into the same remote folder.

Each ``submit_batch`` call is a batch of its own: it returns a batch ID, and
``submit_batch_progress``, ``job_submit_finished`` and ``submit_batch_finished``
carry that ID first. Batches run one after another on the worker. Cancel and
``abandon_batch`` take the batch ID and stop that batch alone, so a batch given
up on never submits its remaining jobs, whatever is submitted after it, and its
signals never reach a later batch's dialog.

From ``submit_batch`` until a job's result arrives ``ClusterController.is_submitting``
holds for it, in whichever batch holds it. Such a job has no scheduler job ID by
design: checks leave it out, evidence about it changes nothing, and it carries
no staleness marker.

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
pending is always re-enabled (an operation still queued when the window closes
is skipped, with nothing to answer). Connecting runs behind a modal
"Connecting…" dialog; an unknown host key is prompted for, saved, and the
connection retried.

- ``connect_to_cluster`` creates the SSH connection and scheduler backend.
- ``submit`` validates the local folder and seed, creates the remote folder,
  uploads ``.cell`` and ``.param``, regenerates ``run.sh`` on script-based queues on every submission, submits
  and returns the scheduler job ID.
- ``cancel(job_id, remote_job_id)`` sends the scheduler cancel command and
  emits ``job_cancelled``; with no scheduler job ID nothing is sent or
  emitted. ``cancel_scheduler_job(remote_job_id)`` sends it and emits nothing,
  for a job the project already shows ``cancelled``.
- ``check_jobs(targets)`` reports evidence; see :ref:`cluster-checks`.
- ``scheduler_job_is_active`` guards the deletion of an old run's folder before
  a resubmission. An ``sacct`` failure other than disabled accounting is an
  error there, and the folder is not deleted.

Closing the window calls ``ClusterController.shutdown``, the one call that
waits for the worker, and it waits a bounded time at most:

1. Work still queued on the worker is skipped, and every submission batch
   stops before its next job. A job a batch had not reached reports
   ``job_submit_finished`` at once with the error *"MuFinder closed before
   submitting this job."*, so it ends with ``submission``.
2. One last task disconnects on the worker, which owns the connection, and
   quits the worker's event loop. The window waits
   ``SHUTDOWN_GRACE_SECONDS`` (2 s) for it, with the event loop running, so a
   result the worker delivers meanwhile is applied.
3. If the worker is still busy, ``ClusterJobRunner.abort_connection`` closes
   the SSH transports from the GUI thread (``SSHConnection.abort``; paramiko's
   ``Transport.close`` may be called from any thread). Every SSH call blocked
   on them then fails, and the worker reaches its disconnect; a connect fails
   at its next step and does not fall back from a key to a password. The
   window waits ``SHUTDOWN_ABORT_SECONDS`` (3 s) more.
4. A worker still running after that is given up on: the controller's signals
   are blocked and its thread is kept for the life of the process, since a
   ``QThread`` destroyed while it runs aborts the process.
   ``mufinder.main.run`` then writes the settings and ends the process with
   ``os._exit`` instead of Python's clean-up. A connect still opening its TCP
   connection has no transport to close yet, so it ends this way. So does a
   ``Test Connection`` or pre-flight connection check cancelled, or left by
   its closed dialog, before its connect returned: nothing interrupts that
   connect, and its thread stays referenced in
   ``mufinder.app.cluster_connection_check`` until it ends
   (``connection_attempts_running``).

A busy dialog, ``Closing the cluster connection…``, appears once the wait
lasts more than a moment (``mufinder.app.blocking_call.wait_for_event``).
While closing, a failed check and a failed late cancel of an abandoned batch's
job are not reported, and once the connection is closed under the worker the
evidence of a check still running is dropped, since a closed connection can
look like a missing file. The project is saved before the wait, and again
after it when a job's result or a cut-off submission's note arrived during it
and left something unsaved, unless the user chose ``Don't save``. Both saves
run on the GUI thread inside ``closeEvent`` and write atomically
(``ProjectStore.write_json``: a synced temporary file moved into place), and
``mufinder.main.run`` can call ``os._exit`` only after the event loop returns,
so the project file is complete before the process ends; the worker never
writes the project.

What an operation the close cut short leaves in the saved project:

- A check or download: the jobs keep what the last applied check gave them,
  and the check that opens the project next time looks again. A partly
  downloaded file is downloaded again.
- A geometry-progress read, a connect, a folder probe: nothing.
- A cancel: the job stays active until a check sees the scheduler's answer.
- A submission answered in time: recorded as usual. A batch step before the
  submit that the closed connection ended (deleting an old run's folder, or
  asking whether that run is still active): ``ended`` with ``submission`` and
  the error, since nothing reached the scheduler. A submit the closed
  connection ended, or one given up on, keeps its state with no scheduler
  job ID, since whether the scheduler took the job is unknown (rule 1).
  ``ClusterController.submission_cut_off`` reports it, and
  ``JobExecutionController.note_submission_cut_off`` records *"Its submission
  was cut off while MuFinder was closing; it may be running on <cluster>."* as
  its monitor message (``last_status_message``, source ``submission``), which
  the status tooltip shows. Its remote folder is kept.

  - A job still ``submitting`` stays so. On the next open it is not being
    submitted any more, so it gets the ``no_job_id`` marker (*"no scheduler
    job ID was recorded"*); a check with no scheduler job ID keeps the note
    beside the marker rather than replacing it, and a requested check reads
    its folder.
  - A job of a batch the dialog had already given up on is ``cancelled``
    already, and is not cancelled on the scheduler. The note goes in its
    ``error_message`` too, which checks leave alone, and the job awaits a
    check's answer (``cut_off_submission_unanswered``, see
    :ref:`awaiting-answer`), so its row offers ``Check now`` and
    ``Refresh Queue`` counts its cluster; either reads its folder and
    completes it from complete output.

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
run further from its own CASTEP checkpoint. ``Continue relaxation…`` uses it,
and so does running an ended job again when the user chooses ``Continue``
(see :ref:`ended-rows-are-submittable`).

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
clears it for a local submission or when the job starts again, or touches
only active jobs (a
cluster check filling in the folder of a job that recorded none).

A completed relaxation is continuable because running more
geometry-optimisation steps from its checkpoint is a normal outcome, not a
recovery. Continuing a dismissed job clears ``dismissed_at``, so its dashboard
row comes back as it does when the job starts again. The job keeps its identity, because
its id binds the workflow row and its record carries the folder holding the
checkpoint. ``record_job_continuation`` appends the ``job_continuation``
milestone with the label and whether a remote folder was kept.

.. _continued-relaxation-inputs:

What a continued relaxation writes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A continued relaxation keeps the inputs its run used, on both routes.
``JobsController._kept_row_inputs`` renders the row from its stored settings
(``row_settings`` on the relaxation controllers) and takes from the job's own
``.cell`` and ``.param`` the project defaults the row inherits rather than sets
(``adapter_common.inherited_project_default_names``: the cut-off, the k-point
grid and the functional, each with its either-or partners), through
``castep_io.input_diff.overlay_job_inputs``. A setting the row stores is taken
from the row; a project default adopted or changed since the run is not.

- Running the ended row again and answering ``Continue``: ``RunAgainPlan``
  carries these inputs, and ``keep_continued_inputs`` writes them over the ones
  the accept step rebuilt, before a cluster job's files are uploaded.
- ``Continue relaxation…``: ``ContinueRelaxationDialog`` receives them as
  ``kept_inputs`` and opens the inherited editors on the run's values. An
  unedited accept writes them unchanged. An edit (``edited_names``: a value
  changed from the one the editor opened with, an either-or choice, an advanced
  parameter added or removed) takes that keyword, and any keyword the edit
  changes in the rendering, from a fresh rendering of the dialog's values; the
  table compares the files on disk with exactly what is written. The row
  stores its own values with only the edits applied (``edited_parameters``),
  so a later start-again uses the row and the project's current defaults.
  Resetting an editor returns it to the value it opened with.

When the job's ``.cell`` or ``.param`` cannot be read, or the row cannot be
rendered, neither route continues it: ``Relaxation not continued`` says so and
nothing is submitted. ``keep_continued_inputs`` leaves a folder another job uses
as the accept step's rewrite does. A write that fails stops the dialog's continuation
before it is submitted; on the run-again route ``Relaxation inputs not kept``
reports it and the job runs with the inputs rebuilt from the row. Quantum Corrections phonon jobs that continue rebuild their
inputs from the row.

.. _remote-checkpoint-probe:

Checkpoint probe
~~~~~~~~~~~~~~~~

The ``Continue relaxation…`` dialog reports whether the job's checkpoint is
there: found, not found, or could not be verified, and the question asked when
an ended job runs again is asked only when one is found or cannot be verified.
The probe always runs in the controller before the dialog or question is
built.

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
  available (a server that reports no mtimes).

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
to it. The job reads ``failed`` with that message. The calculation itself
finished, so ``Check now`` or ``Refresh Queue`` can record it again from the
cluster, and the status tooltip says so (``services.job_monitoring.result_not_recorded``, which reads the
message's ``RESULT_NOT_RECORDED_PREFIX``); if recording fails again the job
stays failed with the new error. A job that had already completed keeps its
result.

The dashboard and workflow rows
-------------------------------

Jobs controller
~~~~~~~~~~~~~~~

``mufinder.app.jobs_controller.JobsController`` is where the application acts
on jobs. It carries out what the Job Dashboard, the workflow rows and the
workflow controllers ask for: submission, resubmission, Continue relaxation,
cancel, inspect, ``Check now`` and dismiss. It asks and reports through
dialogs and the status bar, and every change of state goes through
``JobExecutionController`` or ``ProjectService``. It also runs the check of
every active job on project open and every 60 s, and keeps the status bar's
cluster connection, ``Queue last checked`` and ``Refresh Queue`` current.
``services.cluster_status.cluster_status`` decides them: they show while the
project has a job a refresh can change (``is_refreshable_cluster_job``: an
active cluster job, or an ended one for which ``has_unretrieved_cluster_output``
holds, dismissed or not), and a refresh with no connection reconnects to the
clusters of those jobs. A dropped connection reads ``Connection to <name>
lost`` only for a cluster with active jobs; one with only ended jobs reads as
not connected, and staleness markers stay on active jobs.

The jobs controller wires runner signals into ``JobExecutionController``: local
``job_finished`` goes to the completion path that builds the workflow result
and calls ``complete_job_with_result``; local ``job_failed`` to ``fail_job``,
``status_changed`` to ``mark_local_status_changed`` and ``job_started`` to
``mark_started``; local and cluster ``job_cancelled`` to ``cancel_job``;
cluster ``job_checked`` to ``apply_evidence``, with a ``finished`` update going
to the same completion path. ``queue_checked`` ends a check and updates ``Queue
last checked``. The status bar shows ``JobExecutionController.take_check_summary``:
one changed job is named (*"Cluster job stopped at its walltime limit:
<label>"*, *"Completed cluster job <label>"*), several are counted by outcome
(*"Checked 5 cluster jobs: 1 completed, 2 started running"*), ended jobs a
refresh changed included. An automatic check that changed nothing leaves the
status bar alone; the pass of a ``Refresh Queue`` press (its snapshot's
``refresh``) that changed nothing reports *"Refreshed queue: checked 3 jobs at
12:34:56"*. A check that ends before the refresh's own pass does not answer
the press.

Each change to a job is applied at once, but the heavy work after it —
refreshing the journal, the jobs table and the workflow views, and the
autosave — is coalesced by the main window: a 300 ms quiet-period debounce
flushes an isolated event soon, and a 2 s minimum interval bounds a burst to
about one flush every two seconds. The deadline only ever moves earlier, so a
steady stream cannot postpone the flush forever. A pending flush also runs
before save, close and a project switch. The journal touches only the cards a
flush added or changed, and catches up when its page is shown.

Dashboard
~~~~~~~~~

The Job Dashboard (``mufinder.app.job_dashboard.JobDashboard``) lists ``active`` and ``ended`` jobs without a
``dismissed_at`` stamp. With ``Show dismissed`` ticked (above the table, not
saved, unticked whenever a project opens) it lists the dismissed ones too, with a greyed label whose tooltip says
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

Each row's ``Action`` cell has three fixed slots, so an action is always in
the same place down the table (``JobDashboard.slot_actions``):

.. list-table::
   :header-rows: 1

   * - Job state
     - Slot 1 (run)
     - Slot 2 (look)
     - Slot 3
   * - submitting, queued, retrieving
     - ``Cancel``
     -
     -
   * - running
     - ``Cancel``
     - ``Inspect…`` on a relaxation that reports geometry progress
     -
   * - failed, stopped
     - ``Resubmit…``
     - ``Inspect…``, or ``Check now`` when output may still be on the cluster
     - ``✕``
   * - cancelled
     - ``Resubmit…``
     - ``Check now`` when the scheduler was not told
     - ``✕``

A slot with nothing to offer is left blank, never greyed out: a job with no
workflow row, or an imported job, has no ``Resubmit…``, and a dismissed job
shown with ``Show dismissed`` has no ``✕``. Each slot's buttons share the width
of the widest label the slot can show (``RowSlotButton``, which keeps its space
when hidden), so the column's width does not change with the jobs' states. The
actions:

.. list-table::
   :header-rows: 1

   * - Action
     - Applies to
   * - ``Cancel``
     - ``active``. The scheduler is asked only about a job with a scheduler job
       ID on the connected cluster (``can_cancel_on_cluster``); any other
       cluster job is cancelled in the project, the status bar says so, and the
       job records that its scheduler was not told (see :ref:`awaiting-answer`).
       A job being submitted is cancelled through the worker.
   * - ``Inspect…``
     - ``ended`` other than ``cancelled`` (the failed-job dialog, see
       :ref:`job-failure-reasons`), and running pristine and muon relaxations
       (their geometry progress, see :ref:`geometry-optimisation-progress`).
   * - ``Check now``
     - ``services.job_monitoring.has_unretrieved_cluster_output``: an ended
       cluster job with a remote folder whose reason is ``output_unavailable``
       or ``interrupted``, that awaits a check's answer (a cut-off submission
       or a cancel its scheduler was not told about, see :ref:`awaiting-answer`),
       or whose result could not be recorded (``result_not_recorded``), whether
       or not its cluster is connected. The status tooltip says which
       (``check_now_reason``). It takes
       ``Inspect…``'s slot. ``JobsController`` first connects to each cluster
       ``JobExecutionController.check_now_waits_for_connection`` names, through
       the connect path ``Refresh Queue`` uses (asking for credentials when the
       profile needs them), then checks; a cluster it cannot reach is named in a
       warning and its jobs are left as they were. Active jobs and settled ended
       jobs are not offered it: the automatic checks and ``Refresh Queue`` cover
       them.
   * - ``Resubmit…``
     - ``ended``, MuFinder's own, and still held by a workflow row
       (``JobsController.can_resubmit``, through
       ``WorkflowJobController.key_for_job``). It is the row's Run: each job
       goes to its owning controller's ``submit`` for its row, on its own
       target, with the job's cluster preselected in the cluster submission
       dialog; a selection is submitted once per workflow and target. Inputs
       are rebuilt from the row (see :ref:`ended-rows-are-submittable`).
   * - Dismiss (``✕``, accessible name ``Dismiss <label>``)
     - ``ended`` without a ``dismissed_at`` stamp, confirmed first. When
       ``has_unretrieved_cluster_output`` holds, the confirmation adds
       ``services.job_monitoring.unretrieved_output_note``: the output may still
       be on the cluster and ``Check now`` or ``Refresh Queue`` retrieves it.

The table's context menu lists every action above plus ``Open in <stage>``,
each enabled when it applies to at least one selected job, with the partial
count (``Dismiss (2 of 3)``). Right-click, the context-menu key and Shift+F10
open it; from the keyboard it is about the row whose button has focus, or the
current row. Double-click opens the job's stage. Delete and Backspace cancel
active jobs and dismiss the rest.

.. _awaiting-answer:

Jobs awaiting a check's answer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two kinds of ended cluster job may still be queued or running on their
cluster, and say so in their monitor metadata, with no field of their own
(``services.job_monitoring.awaits_cluster_answer``):

- A job cancelled in the project only
  (``cancel_not_sent_to_cluster``). ``JobExecutionController.mark_cancel_not_sent``
  stamps the source ``CANCEL_NOT_SENT_SOURCE`` (``"cancelled_in_project"``)
  and the message ``CANCEL_NOT_SENT_MESSAGE``, leaving the time of the last
  check as it is. It is called for the project-only ``Cancel`` (no scheduler
  job ID, or the job's cluster not connected) and when the scheduler cancel of
  a submission that answered after its batch was abandoned fails, which also
  records that submission's scheduler job ID when the job holds none. A job
  cancelled while being submitted whose late cancel succeeds, and a job the
  scheduler confirms cancelled, are not stamped.
- A cancelled job whose submission closing MuFinder cut off
  (``cut_off_submission_unanswered``): its message starts with
  ``SUBMISSION_CUT_OFF_NOTE`` and its source is still
  ``SUBMISSION_CUT_OFF_SOURCE`` (``"submission"``).

``has_unretrieved_cluster_output`` holds for either with a remote folder: its
row offers ``Check now``, ``Refresh Queue`` stays shown for its cluster, the
dismiss and run-again confirmations carry ``unretrieved_output_note``, and the
status tooltip gives ``check_now_reason`` in place of the stamp's source and
message. Only the scheduler's answer settles the question: a job it lists as
pending or running is active again, and any other state, ``not_listed``
included, replaces the stamp with the check's own (``"scheduler_poll"`` or
``"retrieval"``), after which the job is an ordinary cancelled job. Complete
output completes it whatever the scheduler says. A check the scheduler does
not answer (``unavailable``, or ``no_job_id`` for a job with no scheduler job
ID) records only when it ran: a folder without complete output does not show
that the job is not queued, so a job with no scheduler job ID stays open until
complete output completes it or it runs again.

.. _workflow-row-menu:

Every workflow table has the same row menu
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ended job's recovery does not depend on the dashboard: its row runs it again
(see :ref:`ended-rows-are-submittable`). Every workflow table's right-click row
menu comes from ``mufinder.app.workflow_table_utils``:

- ``standard_row_actions`` — the shared core, in one order: ``Run Local``,
  ``Run on Cluster``, ``Cancel``, ``Inspect Failure``, ``Configure…``,
  ``Remove``. A table leaves an action out by not supplying its callback (the
  Quantum Corrections and Hyperfine Coupling rows come from completed
  relaxations, so they have no ``Remove``). Run, Cancel and Inspect act on the
  selected rows they apply to, and a mixed selection shows its count in the
  entry (``Cancel (2 of 5)``, from ``partial_count_suffix``).
- ``RowMenuAction`` — one entry; a table appends its own after the core
  (``Continue relaxation…`` and ``Export Relaxed Structure`` on the relaxation
  tables, and the job-folder entries of ``JobFilesMenu``).
- ``bind_row_context_menu`` / ``open_row_context_menu`` — install and open the
  menu; right-clicking outside the selection retargets it first. Each widget
  supplies ``key_for_row``.

The main window resolves a widget's ``cancel_requested`` and
``inspect_failed_requested`` keys through the workflow's
``WorkflowJobController.jobs_for_keys`` and hands the jobs to
``JobsController``. Muon relaxation binds the menu once per charge-state table.

.. _quantum-singlepoint-records:

Quantum Corrections binds each row to one phonon job. Starting it again clears
its folder under the ownership checks of ``JobsController._fresh_start_refusals``,
which refuse a folder another job uses unless that job is a *directory
companion* (``WorkflowJobController.directory_companions``): the
``castep_quantum_singlepoint`` job an older project ran in the row's seed folder
before its phonon job, unless it is still active. Clearing the phonon job's
folder removes the singlepoint's outputs too, while its record stays in the
dashboard. Continuing keeps the folder and its ``.check`` file, and
``CONTINUATION : DEFAULT`` resumes from the checkpoint.

A project that holds ``castep_quantum_singlepoint`` jobs loads them as records:
they stay in the dashboard, collect normally when they finish, and can be
dismissed or deleted. A row with a phonon job keeps it, whatever its state. A
row with only a singlepoint job reads ``ready``; its phonon job is written to a
folder of its own (the seed gains ``_phonon`` when the singlepoint's folder
still exists). A row whose singlepoint is still active shows that job's status
and refuses submission until the singlepoint finishes or is cancelled.

.. _ended-rows-are-submittable:
.. _dismiss-only-hides:

Ended rows are submittable and editable; Dismiss only hides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A workflow row's status comes from its job's state alone, through
``effective_row_status(job)`` in ``mufinder.app.workflow_submission``: a row
with no job, or whose job is ``defined`` (staged or prepared to run again,
not yet submitted), reads
``ready`` (or the workflow's own word for it); every other row reads
``job_status_word(job)``. Every workflow's job-to-row projection goes
through it.

A row can be submitted when its status is in ``READY_WORKFLOW_ROW_STATUSES``:
``ready``, ``defined``, ``not generated``, and the three words of an ended job,
``failed``, ``stopped`` and ``cancelled``. ``Run All`` includes them, and the
Job Dashboard's ``Resubmit…`` runs the same rows through the same ``submit``.

The same statuses are the configurable ones
(``CONFIGURABLE_WORKFLOW_ROW_STATUSES``): every run of an ended row, from the
row or from the dashboard, rebuilds its inputs from the row, so an edit takes
effect on the next run, a changed NEB or phonon farm count included. A
continued relaxation keeps the project defaults its run used
(see :ref:`continued-relaxation-inputs`).
``workflow_row_is_locked`` (the tables' inline cells and parameter dialogs) and
``workflow_row_state_is_configurable`` (the Configure dialogs of Hyperfine
Coupling, Quantum Corrections and Energy Barriers) both follow it; active rows
stay locked, and in Hyperfine Coupling and Quantum Corrections
``locked_configuration_reason`` says why. Energy Barriers also locks completed
rows, whose dialog opens as ``Inspect NEB Parameters``; its rows have
``Remove``.

.. _edited-completed-rows:

Quantum Corrections and Hyperfine Coupling keep a completed row configurable,
through their shared controller layer ``CompletedRowEditsController``. It
compares the row's stored parameters with the ``.cell`` and ``.param`` its job
ran with (``CompletedRowEditsController.parameters_changed_since_run``, through
``castep_io.input_diff.diff_job_inputs``, leaving out the keywords the workflow
writes itself); nothing about the edit is stored beyond the row's own
settings. A completed row whose job's ``.cell`` or ``.param`` is missing cannot
be compared and stays locked, and ``locked_configuration_reason`` says so. A
completed row whose parameters differ reads ``edited`` (``EDITED_ROW_STATUS``),
which is in ``READY_WORKFLOW_ROW_STATUSES``, and keeps its job ``completed``
and its result until it runs again. A change that makes a completed row
``edited`` is confirmed in the Configure step, with ``Cancel`` the default.
When an ``edited`` row is submitted,
``WorkflowJobController.completed_jobs_to_run_again`` names its job, and
``submit`` passes it to ``prepare_jobs_for_resubmission`` with the ended jobs:
it always starts again, without a further question. Once the submission is
accepted the backends clear its folder like any start-again job, and
``RunAgainPlan.prepare`` returns it to ``defined`` through
``JobExecutionController.reset_completed_jobs``, which removes the results of
its runs and records a ``job_reset`` milestone with ``removed_result``. A
cancelled submission leaves the job, its result and its files as they were.
A submission whose rows are all ``completed`` and unchanged stops in the
controller's ``pre_submit_check`` with a ``Nothing to run`` notice that says how
to run such a row again, rather than the backends' ``No eligible jobs``. A
Hyperfine Coupling result's coupling that ``Use in Magnetic Fields`` already
copied into a Magnetic Fields form is that form's input: neither the edit nor
the rerun changes it or the form's field results.

``prepare_jobs_for_resubmission`` on ``WorkflowControllerContext`` decides how
the plan's ended jobs run again and changes nothing. ``WorkflowJobController.submit``
calls it before dispatch; ``JobsController`` asks what it must and returns a
``RunAgainPlan`` (``mufinder.app.workflow_submission``): the jobs that continue
and the jobs that start again. ``None`` (the user cancelled) aborts the
submission silently.

- A job whose inputs resume from a checkpoint
  (``services.job_monitoring.job_resumes_from_checkpoint``: a relaxation or a
  Quantum Corrections phonon job, never an imported one) is probed for its
  ``{seed}.check`` (:ref:`remote-checkpoint-probe`). When at least one is found
  or cannot be verified, one question, ``Run again``, asks ``Continue``,
  ``Start again`` or ``Cancel``. It says which checkpoints were found or could
  not be checked. ``Continue`` is the default, unless a job may still have
  output on its cluster that was never retrieved
  (``has_unretrieved_cluster_output``): then the question adds
  ``unretrieved_output_note`` with ``Starting again`` and ``Cancel`` is the
  default.
- Every other job starts again. When none could continue but one may still have
  unretrieved output on its cluster, ``confirm_destructive`` asks ``Start
  again`` or ``Cancel`` (the default) with the same note; otherwise nothing is
  asked. The user never gets two boxes.

Jobs are created only once a submission is accepted and every refusal is
past, and the run-again plan is applied then too, so nothing is lost before.
``submit`` hands the plan to the backend as ``run_again``, with its accept step
as ``on_accepted`` (see :doc:`/architecture`, Jobs created on acceptance): the
new rows' jobs are only planned before the dialog, and ``on_accepted`` writes
and adopts them and rebuilds the run-again jobs' inputs from their rows, then
clears the start-again jobs' previous output, then applies the plan. A write
that fails there keeps the previous output. A cancelled or refused submission
creates no job, writes no folder and adds no "Configured …" Journal record.

- ``submit_jobs_locally`` calls ``on_accepted`` once its dialog is accepted.
  ``submit_jobs_on_cluster`` calls it once the dialog is accepted, the cluster
  connected and the plan settled (directory checks, overwrite prompts), just
  before the batch starts.
- Start-again jobs are the backends' fresh starts. Once their new inputs are
  written their local output folders are emptied except their ``.cell`` and
  ``.param``, under the ownership checks of
  ``JobsController._fresh_start_refusals`` (a directory companion does not
  count); a folder that cannot be cleared keeps its job out of the submission,
  with its inputs put back as they were.
- On its own cluster a start-again job keeps its own remote folder: the planner
  deletes it with no overwrite prompt (``fresh_delete``), but only once the
  scheduler confirms the previous run has finished. That is asked before
  anything is cleared (``ClusterController.check_previous_runs``, through
  ``ClusterJobRunner.scheduler_job_is_active``), and again on the worker just
  before the delete. A previous run still queued or running, or a scheduler
  that cannot be asked, keeps that job out of the submission with its folders
  and record as they were, and the failure is reported.
- Continuing jobs are submitted into their own remote folder as-is
  (``continue_job_ids`` in ``plan_remote_targets``).
- Then ``RunAgainPlan.prepare`` returns the jobs going ahead to ``defined``:
  ``JobExecutionController.continue_ended_jobs`` (``prepare_job_continuation``
  and its milestone) keeps a continuing job's folders and ``remote_directory``
  (see :ref:`job-continuation-primitive`); ``reset_ended_jobs`` resets a
  start-again job, its milestone recording the folder as cleaned. Then
  ``keep_continued_inputs`` writes back, over the inputs the accept step
  rebuilt from the rows, the inputs a continuing relaxation keeps (see
  :ref:`continued-relaxation-inputs`).

A cancelled submission dialog, a failed connection, a declined prompt or a
refused job therefore leaves the job ended, its files byte for byte, and the
journal without a ``job_reset`` or ``job_continuation`` entry; ``Inspect…``
still works on it.
A continued cluster job uploads its rebuilt inputs into its kept folder: the
fresh ``{seed}.param`` is the marker ``retrieve_job_outputs`` orders the
previous run's ``.castep`` and ``-out.cell`` against, so neither counts as this
run's output, and the previous ``-out.cell`` stays older than the run for the
relaxation controllers' ``validate_artifacts``. The row keeps its job, and a
cutoff-convergence job that starts again stays in its ``job_ids_by_cutoff``
mapping.
Imported jobs are never touched.

Dismissing an ended job only hides it from the Job Dashboard. Nothing else
reads ``dismissed_at``: the workflow row reads the same, and the job stays
runnable from it. Every change of state clears the stamp (starting again,
continuation, and an ended job made active again by ``Check now`` or
``Refresh Queue``), so a job that runs again is always listed. ``Show dismissed`` lists the hidden jobs
with their usual actions, so a dismissed job whose output is still on the
cluster can be checked again.

Job bindings across definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reconfiguring a workflow registers a new definition with its own job mapping (a
Run with the configuration unchanged keeps the active one; see
:doc:`/architecture`, Definition mappings and row removal), so two definitions can bind the same row to different jobs. Before a controller
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
  otherwise, on project open and once a minute (:ref:`local-job-check`). On
  project open a local job stopped as ``interrupted`` is completed when its
  folder holds complete output from its run.
- ``LocalJobRunner`` reports a ``FailedToStart`` error at once, so a job cannot
  stay ``submitting`` behind a process that never launched.
- A cluster job is checked on project open, on connect and once a minute, and
  ends when the scheduler or its output says it ended (:ref:`cluster-checks`).
- A cluster job that cannot be confirmed (cluster disconnected, no scheduler
  job ID, listed nowhere with no output) stays active, shows why on its row,
  and can be cancelled in the project.
- An ended job whose output may still be on the cluster can be checked again
  with ``Check now``, and ``Refresh Queue`` checks every ended job on its
  cluster; complete output completes it, and a
  job the scheduler lists again is active again.

These are pinned by ``tests/services/test_job_reconcile.py``,
``tests/services/test_job_evidence.py``,
``tests/services/test_job_monitoring.py``,
``tests/services/test_job_execution_controller.py``,
``tests/services/test_project_service_transitions.py``,
``tests/services/test_cluster_runner.py``,
``tests/app/test_cluster_controller.py``,
``tests/app/test_cluster_job_checks.py`` and
``tests/app/test_job_check_wiring.py``.
