Job Lifecycle
=============

This document is the technical API contract for the services that drive local
and cluster job execution: state authority, the runner contract, execution
orchestration and how a job's state is decided from what can be seen of it. The user-facing behaviour of the
Job Dashboard, the workflow job tables and local and cluster execution is
described in :doc:`/getting_started/running_jobs`.

Technical API behavior
----------------------

State authority
~~~~~~~~~~~~~~~

The persisted job state is stored on ``mufinder.domain.models.Job`` as one of
four ``JobStatus`` values: ``defined``, ``active``, ``completed`` and
``ended``. An ``active`` job also stores its ``activity`` (``JobActivity``:
``submitting``, ``queued``, ``running`` or ``retrieving``), the phase it was
last seen in, for display only and with no transitions of its own. An
``ended`` job stores its ``end_reason`` (``EndReason``): one of the causes in
:ref:`job-failure-reasons`, ``cancelled``, or ``unknown`` when none was
identified. ``services.job_monitoring.job_status_word`` gives the word every
job table shows: an active job's activity; for an ended job ``cancelled``,
``stopped`` when the reason is ``walltime`` or ``interrupted``, and ``failed``
otherwise; else the state. State transitions are validated by
``ProjectService._assert_transition``:

.. list-table::
   :header-rows: 1

   * - Current state
     - Allowed next states
   * - ``defined``
     - ``active`` (submitted); ``ended`` only as ``cancelled``
   * - ``active``
     - ``completed``, ``ended``
   * - ``completed``
     - ``defined`` (a continuation)
   * - ``ended``
     - ``defined`` (reset or continuation), ``active`` (resubmitted, or found
       alive or with output)

``ProjectService`` owns the primitive mutations:

- ``submit_job`` makes a ``defined`` or ``ended`` job ``active`` as
  ``submitting``, stamps ``submitted_at``, clears ``started_at``,
  ``completed_at``, ``error_message`` and ``end_reason``.
- ``mark_job_active(job, activity, remote_job_id=None)`` records what an
  active job is doing, from any activity to any other, and may update
  ``remote_job_id``. It stamps ``started_at`` when the job starts running. It
  also makes an ``ended`` job ``active`` again when evidence shows that it is
  alive or has output still to be recorded (see :ref:`job-reconcile`),
  clearing ``error_message``, ``end_reason`` and ``completed_at`` and keeping
  remote metadata. A ``defined`` job becomes active only through
  ``submit_job``.
- ``complete_job`` makes an ``active`` job ``completed``, stamps
  ``completed_at``, and clears ``error_message``.
- ``end_job(job, reason, message=None)`` makes an ``active`` job ``ended``
  for ``reason`` and records ``message`` as its error (a cancellation keeps
  the message the job holds when none is given). A ``defined`` job can be
  ended only as ``cancelled``.
- ``reset_failed_job`` returns an ``ended`` job to ``defined`` and clears
  remote/process metadata (including the monitor metadata and any
  ``dismissed_at`` stamp).
- ``prepare_job_continuation`` returns a ``completed`` or ``ended`` job to
  ``defined`` for a continuation run, clearing the same run metadata as
  ``reset_failed_job`` **except** ``remote_directory``, which it preserves.
  See :ref:`job-continuation-primitive`.
- ``dismiss_job`` stamps ``dismissed_at`` for an ``ended`` job. It is not a
  state change and rejects already-dismissed jobs; ``record_job_dismissed``
  appends a ``job_dismissed`` provenance milestone. The app layer
  nonetheless *derives* a submittable workflow-row status from the stamp — see
  :ref:`dismissed-rows-stay-resubmittable`.

``reset_failed_job`` and ``dismiss_job`` accept exactly the same jobs, the
``ended`` ones, so anything the user can dismiss, the app can also reset back
to ``defined``.

.. _job-continuation-primitive:

Continuation: a job that keeps its remote directory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The word "continuation" is used in two distinct senses in this document. The
older one, in :doc:`/getting_started/running_jobs`, describes a *resubmission* that
happens to reuse existing remote files. This section is about the separate
service primitive ``ProjectService.prepare_job_continuation``, which exists so a
finished calculation can be run further from its own CASTEP checkpoint.

The primitive accepts the module constant ``CONTINUABLE_JOB_STATUSES`` —
``completed`` and ``ended`` — and sets ``defined``, clearing ``submitted_at``, ``started_at``, ``completed_at``,
``remote_job_id``, ``num_processes``, ``error_message``, the monitor metadata
(``last_checked_at``, ``last_status_source``, ``last_status_message``), and any
``dismissed_at`` stamp. It deliberately preserves ``remote_directory``.

That preserved field is the point. **A job that is ``defined`` and still carries
a non-null ``remote_directory`` is the durable marker for a continuation run**:
the cluster submission path reads exactly that combination and resubmits into
the existing remote directory — where the previous run's ``.check`` file lives —
rather than recomputing a target, testing whether it already exists, and
offering to delete it. No other code path produces the combination. Every writer
of ``remote_directory`` either sets it while submitting a job
(``JobExecutionController``), clears it for local submission (also
``JobExecutionController``), clears it on ``reset_failed_job``, or touches only
active jobs (a cluster check filling in the folder of a job that recorded
none; see :ref:`cluster-checks`).

``CONTINUABLE_JOB_STATUSES`` is wider than what reset and dismiss accept,
because a ``completed`` relaxation is continuable: running more
geometry-optimisation steps from a converged-enough checkpoint is a normal
outcome rather than a recovery from failure. Continuation is a separate entry
point with its own guard, and it also clears ``dismissed_at``, so continuing a
dismissed job makes its dashboard row visible again on the same terms as a
reset.

A continuation keeps the job's identity rather than deleting and re-creating
the job: the job id is what binds the workflow-table row to the calculation,
and the job record is what carries the output/remote directory holding the
checkpoint the continuation depends on. Deleting and re-creating would sever
the row binding and abandon the directory.

``record_job_continuation`` appends the matching ``job_continuation`` provenance
milestone, recording the job label and whether a remote directory was preserved.
No persisted field is added, so no compatibility-ledger entry is required.

.. _remote-checkpoint-probe:

Checkpoint probe
~~~~~~~~~~~~~~~~

The ``Continue relaxation…`` dialog reports whether the job's checkpoint is
there: found, not found, or could not be verified. For a cluster job the probe
is built from remote-file primitives:

- ``ClusterJobRunner.remote_file_exists`` stats one remote path over the
  existing SSH connection (the file-level analogue of
  ``remote_directory_exists``).
- ``ClusterController.check_remote_files`` runs this on the worker thread for a
  batch of ``(job_id, remote_path)`` targets and emits ``remote_files_checked``
  with the paths that exist and any per-job errors, mirroring
  ``check_remote_dirs`` / ``remote_dirs_checked`` so the GUI thread awaits the
  result instead of blocking on SSH. The probe always runs in the controller
  before a dialog is constructed, never inside a widget's ``__init__``.
- ``CheckpointProbeStatus`` (``found`` / ``absent`` / ``unavailable``) and
  ``CheckpointProbeResult`` (built with the ``found()`` / ``absent()`` /
  ``unavailable(reason)`` constructors) carry the three-way result. Not
  connected, no remote directory recorded on the job, and SSH or transport
  errors all map to ``unavailable`` with a reason rather than stopping the
  dialog from opening. A recorded remote directory that no longer exists on the
  cluster reports ``absent``.

The probe shares the SSH worker with the geometry-progress reads of running
relaxations, which fetch each job's ``.castep`` output. So that the probe (or a
queue check, or a submission) never waits for a large accumulated output:

- ``SSHConnection.read_bytes`` requests a byte range in one pipelined SFTP batch
  and returns at most the ``max_bytes`` it is given.
- ``ClusterJobRunner.read_geometry_progress`` does one read of at most
  ``REMOTE_READ_CHUNK_BYTES`` (2 MiB) of one job per turn of the worker's event
  loop, then queues the rest of its work behind whatever is already waiting. A
  job's progress is emitted through ``geometry_progress_read`` once its output
  has been read to the end, one job per emission. A disconnect, a failed
  read or another project being opened ends that job's read with ``None``;
  after a disconnect or a failed read the next refresh carries on from the
  bytes already read.

A local job is probed with a plain ``pathlib.Path`` check for ``{seed}.check``
in its job directory, reporting ``found`` or ``absent`` (or ``unavailable``
when the job's output directory cannot be resolved).

.. _re-completing-continued-job:

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
``status``/``relaxation_run_id``/``total_energy_eV``/``relaxed_fractional_position``
are rewritten from the same completion, so the site and the structure always
describe the same run. Removing the job's records unlinks the site again: it
returns to ``staged`` and drops the energy and relaxed position it copied.

*A stale ``-out.cell`` fails the job.* A continuation re-uses its job directory,
so the previous run's relaxed structure is still sitting where the completion
path looks for it: a run that dies before writing a new one would have the old
geometry accepted as its result. Two guards cover this, each closing the other's
blind spot.

The local one is in both relaxation controllers' ``validate_artifacts``,
alongside the missing-file and parse-failure checks (see :doc:`/architecture`,
Result dispatch): it fails a relaxation job whose local ``-out.cell`` predates
the current run's ``started_at``. The comparison
allows a few seconds of slack for filesystem mtime granularity and network-mount
clock offsets, and skips entirely when ``started_at`` is unset — a job the poller
never observed running fails open rather than failing on no evidence.

The remote one is in :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs`,
and it is what makes
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
- A remote ``.castep`` older than the marker is **not parsed** and counts as
  no output from this run, exactly as if it did not exist (so a job the
  scheduler does not list is left as it is). This is the run that never started at all:
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

*Failure markers.* A line of the last restart containing ``ERROR``, ``FAILED``,
``did not converge`` or ``panic`` marks the run as failed, except inside a
boxed CASTEP warning: the lines between two rules of ``?`` under a
``Warning in <routine>`` heading. A phonon run of a molecular crystal can warn
that it "Failed to identify 3 eigenvectors with acoustic character" and then
finish normally; its heading line is kept in ``warnings``.

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
  the local runner reports ``status_changed(job_id, "queued")`` for an active
  job still ``submitting``, ``mark_local_status_changed`` calls
  ``ProjectService.mark_job_active`` with ``queued``; a queued notification for
  any other job is an inert no-op.
- **Runners never touch ``Job`` objects.** They communicate solely by
  emitting signals keyed by job id, and the controller translates those into
  ``ProjectService`` calls. ``LocalJobRunner`` holds a process for each job and
  reports its lifecycle (``job_started``, ``job_finished``, ``job_failed``,
  ``job_cancelled``, ``status_changed``); the ``JobRunner`` protocol describes
  that surface. ``ClusterJobRunner`` holds no job and is not a ``JobRunner``:
  a check reports what the cluster shows of each job as evidence
  (``job_checked``), and ``reconcile`` decides the state from it (see
  :ref:`job-reconcile`).

The two runners' ``submit`` error contracts are **deliberately different** and
are not to be unified:

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
``"scheduler_poll"`` (what a cluster check heard from the scheduler),
``"retrieval"`` (what a cluster check found in the job's remote folder),
``"local_runner"`` (local lifecycle events such as "queued locally", "process
started", "process completed", "process failed"), and ``"reconciliation"``
(the local job check stopping a job no process is running). The
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

Retrieval flags
~~~~~~~~~~~~~~~

Which secondary outputs a job produces is one value,
:class:`~mufinder.services.job_artifacts.RetrievalFlags`:

- ``retrieve_den_fmt`` (UEP source jobs), ``retrieve_out_cell`` (pristine and
  muon relaxations), ``retrieve_phonon`` (quantum-phonon jobs) and
  ``retrieve_ts`` (transition-state and NEB jobs), set from the calc type by
  ``RetrievalFlags.for_calc_type``;
- ``task_farming_enabled`` and ``num_farms``, the job's farm setting, because
  a farmed run writes ``<seed>_farmNNN.<ext>`` in place of ``<seed>.<ext>``.

Submission passes the value whole: ``submit_local_jobs`` takes a
``retrieval_for_job`` callback and ``ClusterJobSubmission`` carries a
``retrieval`` field. ``LocalJobRunner.submit`` takes the fields as keywords,
rebuilds the value and keeps it with the job's process, where it drives
``RetrievalFlags.collect``; it never downloads, so it ignores
``retrieve_den_fmt``. A cluster check carries the value whole on its
``CheckTarget``, where it drives the download of the optional files.

Local runner behavior
~~~~~~~~~~~~~~~~~~~~~

``LocalJobRunner`` is an in-memory queue around ``QProcess``:

- Requests are validated for CASTEP executable, ``mpirun`` when multiple cores
  are requested, and the local job directory.
- CASTEP starts with the user's environment. The thread-pool variables the
  entry point set for MuFinder itself (``OPENBLAS_NUM_THREADS``,
  ``OMP_NUM_THREADS``, ``MKL_NUM_THREADS``; see :doc:`/quality_and_automation`,
  Native thread pools) are removed; a value the user set is passed on.
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
  controller never re-read those files (see :doc:`/architecture`, Workflow
  controllers → Parse-once off the GUI thread).

Local runner state is not persisted. If the application is closed while local
jobs are active, the runner performs a best-effort cancellation during shutdown,
but the durable project file may still contain active local statuses if that
state is not saved after cancellation.

Cluster runner behavior
~~~~~~~~~~~~~~~~~~~~~~~

``ClusterJobRunner`` is an SSH and scheduler facade. It keeps the connection
and nothing about any job: every operation is told which job it is about.

All of its paramiko SSH/SFTP work is blocking, so it runs on a dedicated worker
thread owned by ``mufinder.app.cluster_controller.ClusterController``, which is
what ``main_window`` holds. The GUI event loop is therefore never blocked by a
check, a download or connection setup. The controller mirrors connection state
(``is_connected``, ``connected_cluster_name``, ``connected_username``) on the
GUI thread so callers never touch the live transport across threads.

**No GUI-thread call blocks on the worker.** Every GUI-initiated cluster
operation is queued onto the worker and returns immediately; its outcome is
delivered back to the GUI thread via a signal. This matters because the worker
also runs the checks, and one check can spend minutes downloading and parsing
a batch of finished jobs. The controller uses two asynchronous styles:

- *Fire-and-forget with a purpose-built signal* for the high-level flows:
  ``connect_async`` / ``connect_finished``, ``check_remote_dirs`` /
  ``remote_dirs_checked``, ``check_jobs`` / ``job_checked`` and
  ``queue_checked``, ``cancel`` / ``job_cancelled``, and ``submit_batch`` with
  its progress signals.
- *A request/completion bridge* for the low-level helpers: the op is queued on
  the worker and its outcome is delivered via
  ``ClusterController.operation_finished`` ``(request_id, kind, result,
  error)``, keyed by a per-call ``request_id`` and a ``kind`` discriminator so
  multiple in-flight ops route to their own continuation. The completion fires
  on **every** exit path — success, a caught exception, or an exception
  escaping the queued task — so a UI element disabled while an op is pending is
  always re-enabled and never stuck.

The single deliberate exception is ``ClusterController.shutdown``, which
disconnects on the worker thread with a blocking call at teardown (the app is
already closing); no responsiveness-sensitive path blocks.

Connecting is also asynchronous (``connect_async`` / ``connect_finished``): the
multi-second SSH handshake runs on the worker behind a modal "Connecting…"
dialog instead of freezing the window. Host-key trust is handled by the GUI
connect-with-trust retry loop — on an unknown-host-key error
it prompts, saves the key, and reconnects, and the next attempt trusts it — so
no cross-thread coordination is needed.

Batch submission is asynchronous. ``services.cluster_submission_planner``
(``plan_cluster_submission``) computes remote targets, has the worker check
which remote directories already exist, resolves any overwrite prompts up front
through the window's prompt dialogs, and builds the ``ClusterJobSubmission``
for each job; the GUI then prepares document state for each job
(``JobExecutionController.prepare_cluster_submission``) and hands the batch to
``ClusterController.submit_batch``. A job whose existence check raises (an SSH
or SFTP error, or the check not completing within three minutes) is not
submitted and keeps its state: it is listed in the "Cluster submission failed"
message, and no overwrite prompt is shown for it, because its directory may
hold a previous run the user was never asked to overwrite. A directory a job
in the batch submits into as-is (a continuation's preserved ``remote_directory``)
is never deleted, whatever another job's
overwrite answer was, and a start-again job whose own directory is such a
directory fails instead of resuming from the checkpoint it was meant to
discard. The worker uploads and submits each job,
emitting progress and per-job results that the GUI applies via
``finalize_cluster_submission``/``fail_cluster_submission``. A modal progress
dialog reports progress and offers a Cancel button: the job the worker has in
hand finishes submitting and its result is applied, and jobs prepared but not
yet submitted are moved to ``cancelled``. When the worker reports no progress
for three minutes the dialog stops waiting: it gives the batch up through
``ClusterController.abandon_batch``, which asks the worker to stop submitting
as Cancel does, and moves the jobs it has no result for to ``cancelled``. If
the worker then submits one of those jobs after all, the controller cancels
that scheduler job through ``ClusterJobRunner.cancel_scheduler_job`` and leaves
the project row as it is, so the cluster matches the ``cancelled`` row. If that
cancel fails, the window warns with the job and its scheduler job ID so it can
be cancelled on the cluster by hand; the project records nothing more.

From ``submit_batch`` until a job's result arrives (or the batch ends)
``ClusterController.is_submitting`` holds for the job. Such a job has no
scheduler job ID yet by design: checks leave it out, evidence about it changes
nothing, and it carries no staleness marker.

- ``connect_to_cluster`` creates the SSH connection and scheduler backend.
- ``submit`` validates the local output directory and remote seed name, creates
  the remote directory, uploads ``.cell``/``.param`` unless
  ``reuse_existing_files`` is true, regenerates and uploads ``run.sh`` on every
  submission on script-based queues so the scheduler resources match the current
  submission, submits to the scheduler and returns the scheduler job ID.
- ``cancel(job_id, remote_job_id)`` sends the scheduler cancel command and
  emits ``job_cancelled``. With no scheduler job ID there is nothing to cancel
  and nothing is emitted.
- ``cancel_scheduler_job(remote_job_id)`` sends the scheduler cancel command and
  emits nothing; it serves a job the project already shows ``cancelled``.
- ``check_jobs(targets)`` looks at each target and reports what it sees; see
  :ref:`cluster-checks`.

.. _slurm-memory-requests:

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

.. _cluster-checks:

Cluster checks
~~~~~~~~~~~~~~

A cluster job's state is decided from what a check sees of it, never from
whether anything was holding it. ``JobExecutionController.check_cluster_jobs``
builds a ``CheckTarget`` for every active job on the connected cluster from the
document (scheduler job ID, remote folder, seed, local output folder, retrieval
flags) and passes them to ``ClusterController.check_jobs``. Nothing is
registered when a project opens and nothing is cleared when another one does.
A job is a target when it has a local output folder and a remote folder; one
with a scheduler job ID and no recorded folder gets the folder its cluster
profile would have given it. A job still being submitted is left out. So is
a job with no scheduler job ID, unless ``Check now`` names it: there is no one
to ask about it, and nothing ties output in its folder to this run.

Checks are asked for:

- when a project is opened and once a minute, from the timer the main window
  owns (the same tick as the :ref:`local-job-check`);
- when a cluster connects;
- from the status bar's queue button, which first reconnects the clusters of
  active jobs when none of them is connected;
- from the dashboard's ``Check now`` action, for the jobs it names.

All but ``Check now`` are dropped while an earlier check is still on the
worker, so a slow download does not collect a queue of checks behind it.
``Check now`` is queued, marks its targets ``on_request``, and may name
``failed``, ``stopped`` and ``cancelled`` jobs as well as active ones.

On the worker, ``ClusterJobRunner.check_jobs``:

- Skips targets of a cluster other than the connected one.
- Asks the queue once for every target with a scheduler job ID. SLURM is
  asked with ``squeue --all --jobs=<ids>``; ``--all`` keeps jobs in hidden or
  unavailable partitions in the output, which ``squeue`` otherwise leaves out.
- Looks up every job the queue does not list in job accounting with
  ``sacct --jobs=<ids> --allocations`` (``Scheduler.history``), in one call.
  Accounting still knows a job that a site wrapper, a ``SQUEUE_*`` filter, or
  a multi-cluster setup hides, and one that ``squeue`` has already forgotten.
  Accounting uses the cluster profile's ``Accounting command path`` when set,
  otherwise command discovery in the non-interactive SSH shell. On a cluster
  whose accounting is disabled the lookup finds nothing, and when ``sacct``
  fails for any other reason those jobs count as not listed. Neither
  disconnects the runner. The same lookup guards the deletion of an old run's
  directory before a resubmission
  (``ClusterJobRunner.scheduler_job_is_active``). There, an ``sacct`` failure
  other than disabled accounting is an error, and the directory is not deleted.
- Turns the scheduler's answer into a ``SchedulerState``. SLURM's pending,
  configuring, held, requeued, suspended, and stopped (``SIGSTOP``) states
  (``PD``, ``CF``, ``RD``, ``RF``, ``RH``, ``RQ``, ``SE``, ``S``, ``ST``) are
  ``pending``; ``R``, ``CG``, ``RS``, ``SI``, and ``SO`` are ``running``;
  ``CD`` is ``completed``; ``CA`` is ``cancelled``; ``TO`` is ``timed_out``;
  and ``F``, ``NF``, ``OOM``, ``BF``, ``DL``, ``PR``, and ``RV`` are
  ``failed``. Accounting reports the long names (``PENDING``, ``CANCELLED by
  <uid>``), which map the same way. A state the backend does not know is
  ``pending``, so it never claims the job is running or finished. A job with
  no scheduler job ID is ``no_job_id``, and one listed nowhere is
  ``not_listed``.
- Looks at the remote folder of every job the scheduler no longer runs or
  cannot speak for (``wants_output``): ``completed``, ``timed_out``,
  ``failed`` and ``not_listed``, and ``cancelled`` and ``no_job_id`` only on
  request. :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs`
  downloads the error files and scheduler log, ``.castep`` and the optional
  ``.den_fmt``, ``-out.cell``, ``.phonon``, ``.ts`` or ``.magres`` outputs that
  the job's retrieval flags request (see Retrieval flags), parses them, and
  returns the ``OutputState`` with its detail. The secondary outputs are
  parsed once here on the worker thread via
  :func:`~mufinder.services.job_artifacts.collect_job_artifacts` and travel
  with the evidence, so the completion handler and controllers never re-parse
  them. A folder that raises is reported as ``unreachable`` and the pass goes
  on to the next job.
- Emits ``job_checked(job_id, evidence)`` for each job. A job the scheduler
  reports finished is reported twice: first with the scheduler's answer alone,
  so that it reads as ``retrieving`` during the download, then with what its
  folder held.
- Ends the pass with a ``queue_checked`` snapshot carrying ``checked_at``
  (``utc_now()``), per-job elapsed runtimes and the ids of the jobs checked.

When the queue command itself fails, every target is reported with scheduler
``unavailable``, the runner disconnects and ``poll_failed`` raises a warning
through the UI. A check that finds the transport already dead is handled the
same way. No job changes state: active cluster rows go stale instead, and the
status bar reads ``Connection to <name> lost`` in red while that cluster still
has active jobs (``services.cluster_status``), until a reconnect or a project
change.

On the GUI thread ``JobExecutionController.apply_evidence`` passes the evidence
to ``reconcile`` (the table in :ref:`job-reconcile`) and applies the decision:

- Evidence about a job that is being submitted, or about an earlier submission
  of the job (another scheduler job ID or folder), changes nothing.
- Every transition goes through the ``ProjectService`` primitives, with the
  job's run and journal kept in step. An active job's activity follows the
  decision whatever it was before; ``running`` stamps the run's start. An
  ``ended`` decision closes the active run and records ``job_stopped`` for
  ``walltime`` or ``interrupted``, ``job_cancelled`` for ``cancelled`` and
  ``job_failed`` otherwise; one with no reason is diagnosed from the
  downloaded logs, and is ``unknown`` when they show none.
- Complete output is returned as ``finished`` rather than completing the job:
  the main window records the result through the same completion path as a
  local job, and that completes it.
- An ``ended`` job that ``Check now`` finds listed again, or with complete
  output, is made active again through ``ProjectService.mark_job_active`` and
  gets a *fresh* ``CalculationRun``, so
  the ended run keeps its outcome and the new one records what follows.
- The monitor metadata is stamped with the evidence's ``checked_at``, the
  source (``"scheduler_poll"``, or ``"retrieval"`` when the folder was looked
  at) and the decision's observation, for example "scheduler reports RUNNING"
  or "missing from scheduler queue; scheduler accounting reports QUEUED".
- The decision's marker, if any, is kept in memory for the dashboard. It is
  never saved; the next check rebuilds it.

Two consequences are deliberate. A job that neither the queue nor accounting
lists, and whose folder holds no output from this run, is never ended: it
stays active with the marker *"lost track: the scheduler does not list job
<id> and its folder holds no output from this run"* until a check finds an
entry or output, or the user cancels it. And a job the scheduler reports
failed or timed out still completes when its CASTEP output is complete.

Main-window integration
~~~~~~~~~~~~~~~~~~~~~~~

The main window wires runner signals into ``JobExecutionController``:

- local ``job_finished`` uses the completion path that creates the
  workflow-specific result and then calls ``complete_job_with_result``.
- local ``job_failed`` calls ``fail_job``, local ``status_changed`` calls
  ``mark_local_status_changed`` and local ``job_started`` calls
  ``mark_started``.
- local and cluster ``job_cancelled`` both call ``cancel_job``.
- cluster ``job_checked`` calls ``apply_evidence``. A ``finished`` update goes
  to the same completion path as a local job.
- cluster ``queue_checked`` ends a check. The status bar then shows
  ``JobExecutionController.take_check_summary``: one line for every cluster
  job the check changed, described by the state it reached. One job is named
  (*"Cluster job stopped at its walltime limit: <label>"*); several are counted
  by outcome (*"Checked 5 cluster jobs: 1 completed, 2 started running"*). A
  check that changed nothing leaves the status bar alone, and one that failed
  discards what it had collected.

The per-job document mutation happens synchronously, but the **heavy dashboard
work is coalesced**: refreshing the journal, the jobs table (which recreates
per-row action buttons), and the workflow views, plus the full-document
autosave. The journal treats the provenance stream as append-only: a flush that
added no records leaves it alone, new records touch only the cards they add or
change (a completion updates its batch's card) rather than rebuilding every
card, and while the Journal page is not shown it is only marked stale and
catches up when it is shown. Lifecycle handlers *request* a dashboard flush
instead of refreshing and saving inline; the pending flush runs on a timer and
also before save, close and project switch.

The flush request combines two rules so both an isolated completion and a
multi-minute completion stream behave well:

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

Applying a job's evidence stamps its monitor metadata
(``last_checked_at``/source/message) and then also requests a flush rather
than flushing synchronously, so it is subject to the same debounce and rate
limit; evidence can itself arrive every few seconds during a retrieval. A
burst of cluster jobs finishing across one or several checks therefore rebuilds
the views and writes the document at a bounded cadence rather than once per
job, which keeps the GUI responsive for large batches.

The dashboard displays active states plus ``failed``, ``stopped``, and
``cancelled``; rows whose job carries a ``dismissed_at`` stamp are filtered
out. The ``Status`` cell's tooltip carries the relative age of
``last_checked_at``, ``last_status_source``, ``last_status_message`` and the
staleness reason, and the row gets a stale icon for an active cluster job
that ``JobExecutionController.staleness`` returns a reason for
(``services.job_monitoring.job_staleness``): its cluster is disconnected, its
last check left a marker, it has no scheduler job ID, or its last check is
older than 180 s. A job still being submitted is never stale. Each row has
one action button for its
state and a ``⋯`` menu; the button is ``Cancel`` for active jobs, ``Inspect…`` for
``failed`` and ``stopped``, ``Check now`` when
``services.job_monitoring.has_unretrieved_cluster_output`` holds (a failed or
stopped cluster job with a remote directory whose failure reason is
``output_unavailable`` or ``interrupted``) and ``can_check_now`` does, and
``Resubmit…`` for ``cancelled``. The table is redrawn when the cluster
connection opens or closes, since the button and the stale icon depend on it. Delete and Backspace cancel active jobs and dismiss the rest; they
change nothing else. Actions apply as follows:

- ``Resubmit``: ``failed``, ``stopped``, and ``cancelled`` (relaxations first ask
  whether to continue or start again; see :ref:`resubmit-continue-or-start-again`).
- ``Check now`` (``JobExecutionController.can_check_now``): an active,
  ``failed``, ``stopped`` or ``cancelled`` cluster job with a remote
  directory, while the cluster runner is connected to that job's cluster and
  the job is not being submitted; and a ``failed``, ``stopped`` or
  ``cancelled`` local job with an output folder. A cluster job is checked on
  the worker as described in :ref:`cluster-checks` and stays as it is until
  the evidence arrives. A local job is decided at once from its output folder.
  Complete output completes the job without a duplicate scheduler job.
- ``Cancel``: ``submitting``, ``queued``, ``running``, and ``retrieving``. The
  scheduler can be asked to cancel only a job with a scheduler job ID on the
  connected cluster (``can_cancel_on_cluster``). Any other cluster job is
  cancelled in the document directly, without contacting the cluster, and the
  status bar says so. A job still being submitted goes to the worker, where
  its request already is.
- ``Inspect``: ``failed`` and ``stopped`` (the failed-job dialog, see
  :ref:`job-failure-reasons`), and ``running``
  pristine and muon relaxations (the geometry-optimisation progress dialog; see
  :ref:`geometry-optimisation-progress`). On a mixed selection each job gets the
  view that fits it.
- ``Reset``: ``failed``, ``stopped``, and ``cancelled``. The same action is
  also available per row from every workflow table (see
  :ref:`per-row-reset-affordance`), which is what keeps a non-dismissed
  failure recoverable without going through the dashboard.
- ``Dismiss``: ``failed``, ``stopped``, and ``cancelled``. Dismissal hides
  the row from the active dashboard by stamping ``dismissed_at``; it is not a
  status transition, so the job record, runs, results, and provenance all
  remain in the document. There is no un-dismiss *action* — the stamp is
  cleared only by ``reset_failed_job``, which the job's own workflow row now
  triggers on its next submit (see below). When a job to dismiss satisfies
  ``has_unretrieved_cluster_output``, the confirmation adds
  ``services.job_monitoring.unretrieved_output_note``: its output may still be
  on the cluster, dismissing does not retrieve it, and ``Check now`` does.

If building or storing a finished job's result raises while the job is still
active,
``JobExecutionController.fail_unrecorded_finish`` fails it with the message
*"The job's output was retrieved, but MuFinder could not record the result:
<error>"*, removes any result the attempt added, clears a run's reference to
it, and the traceback is logged. The job is then an ordinary failed job:
``Check now`` records it again from the cluster. A job that had already
completed when the error was raised keeps its result.

.. _per-row-reset-affordance:

Every workflow table has a per-row reset
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A failed job's recovery must not depend on the jobs dashboard: the dashboard row
is easy to lose track of in a long project, and it disappears entirely once the
job is dismissed. Every workflow table therefore offers the same recovery action
on the row itself, via the ``Reset Failed Jobs…`` entry of the shared
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
  The shared reset handler prompts once for the batch, including the
  destructive output cleanup, so the selection must never be narrowed
  *silently*: for a
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
  above it; ``Configure…``, ``Remove`` below), passed to the bind call as
  ``leading_actions`` / ``additional_actions``. A table leaves out an action by
  not supplying its callback, and a table with no failed-job reset passes
  ``emit_reset=None``.
- ``RESET_FAILED_TABLE_TOOLTIP`` — the fallback discoverability stamp, set by
  ``bind_reset_failed_context_menu`` on a table that has no tooltip of its own.
  The workflow tables set their own tooltip naming the whole menu. Per-item
  tooltips are unaffected and still take precedence where a widget sets them.

``MainWindow`` connects each widget's ``reset_failed_requested`` signal to a
thin handler that resolves the keys through the workflow's
``WorkflowJobController.jobs_for_keys`` and hands the jobs to the shared reset
handler. Muon relaxation binds the menu once per
charge-state table, each capturing its own table, so a right-click acts on the
rows it was raised over rather than on whichever tab happens to be current.

After a reset the main window repaints every workflow table: the dashboard
refresh covers the convergence, pristine, muon and UEP tables, and energy
barriers and quantum corrections are refreshed through their controllers'
``refresh_view``, which no-ops until their lazily-created view exists.

.. _quantum-singlepoint-records:

Quantum corrections follows the same shape: each row is bound to one phonon
job, and its table also offers ``Reset Failed Jobs…`` as a button beside the
run buttons, emitting the same signal for the selected resettable rows. The
reset's directory clean-up refuses a directory another job uses unless that job
is reset in the same batch or is a *directory companion*
(``WorkflowJobController.directory_companions``). The Quantum Corrections
controller names one: the ``castep_quantum_singlepoint`` job an older project
ran in a row's seed directory before its phonon job, unless that singlepoint
is still active. Cleaning the phonon job's directory therefore also removes
the singlepoint's outputs, so the rerun is a fresh phonon job, while the
singlepoint's record stays in the Job Dashboard. Without clean-up the
directory, and any ``.check`` file in it, is kept, and the job's
``CONTINUATION : DEFAULT`` resumes from that checkpoint on a local rerun. The
reset clears ``remote_directory`` as for every job, so a cluster resubmission
recomputes its target; a full continuation into the preserved remote
directory (:ref:`job-continuation-primitive`) is not offered for Quantum
Corrections.

A project that holds ``castep_quantum_singlepoint`` jobs loads with them as
records: they stay in the Job Dashboard, collect normally when they finish,
and can be dismissed or deleted there. A row with a phonon job keeps it,
whatever its state, and a completed phonon job keeps its result. A row with
only a singlepoint job reads ``ready``; its phonon job is written to a
directory of its own (the seed gains a ``_phonon`` suffix when the
singlepoint's directory still exists), so it never starts from the
singlepoint's checkpoint. A row whose singlepoint is still submitting, queued,
running or being retrieved shows that job's status, and submitting it is
refused with an explanation until the singlepoint finishes or is cancelled.

.. _dismissed-rows-stay-resubmittable:

Dismissed rows stay resubmittable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Dismissing a terminal job removes the only generic ``Resubmit``/``Reset``
controls it had, but its workflow table row still points at it through the
definition's persistent label→job-id mapping. Left alone, that row would project
``"failed"`` — not a member of ``READY_WORKFLOW_ROW_STATUSES`` — and its
run actions would skip it forever, with no way back to ``defined``. The
row's own ``Reset Failed Jobs…`` action cannot rescue it either: that action
gates on the displayed status, and a dismissed row no longer reads as failed.

Dismissal is therefore read as *"failure acknowledged, this row is actionable
again"*, via two pieces in ``mufinder.app.workflow_submission``:

- ``effective_row_status(job)`` — the single place the rule lives. An
  ``ended`` job (failed, stopped, cancelled) that carries a ``dismissed_at``
  stamp projects ``"ready"``; every other job projects its status word
  (``job_status_word``). Every workflow's
  job→row-status projection goes through it. A *non*-dismissed failed job is
  unaffected: it still shows ``"failed"``, its row still refuses to submit, and
  its recovery path is the row's own ``Reset Failed Jobs…`` action plus the
  dashboard ``Resubmit…``/``Reset…`` actions that are still on screen.
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

Job bindings across definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reconfiguring a workflow registers a new definition with its own job mapping,
so two definitions can bind the same unchanged row to different jobs. Before a
controller reads its active definition's bindings it copies them across from
the equivalent definitions, choosing by job state
(:mod:`mufinder.services.definition_mappings`; see :doc:`/architecture`,
Definition mappings and row removal). An in-flight job (``submitting``,
``queued``, ``running``, ``retrieving``) beats a ``completed`` one, which beats
a ``failed``, ``stopped`` or ``cancelled`` one, which beats a ``defined`` one;
the two relaxation workflows instead rank a ``defined`` job above a failed,
stopped or cancelled one, so a re-staged row keeps its fresh job. A ``defined``
job is copied like any other: a row's page ``Run`` registers a new definition
before it submits, so a row that has not run yet keeps its job, seed and folder
instead of gaining another job each time. A tie
keeps the binding the active definition already holds, and dismissal does not
change a job's rank.

.. _local-job-check:

Local job check
~~~~~~~~~~~~~~~

A local job is reported on by the local runner, which holds its process. A job
the runner holds no process for has nothing left to report on it: MuFinder was
closed while it ran, or its process ended without a finish event.

``JobExecutionController.check_local_jobs`` looks at every active local job
(an ``execution_target`` that is blank or ``"local"``, case-insensitively). For
each one ``services.job_evidence.local_job_evidence`` reads the local runner's
``pending_job_ids()`` and ``active_job_ids()``, and, only when the runner holds
no process for the job, its output folder: ``<seed>_farm001.castep`` when a
task-farmed run wrote one, otherwise ``<seed>.castep``. A file last written
before the job's ``submitted_at`` belongs to an earlier run in the same folder
and counts as no output. The evidence goes to ``reconcile`` (see
:ref:`job-reconcile`) through ``JobExecutionController.apply_evidence``:

- A job the runner holds keeps its status.
- A job with no process and complete output is returned as ``finished``. The
  main window records its result through the same completion path as a job the
  runner reported, and the job becomes ``completed``.
- Any other job with no process is ended through ``ProjectService.end_job``
  with ``EndReason.INTERRUPTED`` (it reads ``stopped``) and the message *"MuFinder has no local
  process for this job: it was closed while the job ran, or the process ended
  without reporting. Reset or resubmit it."* Its latest run, if still active,
  is closed as ``stopped``, a ``job_stopped`` journal event is recorded, and
  the monitor metadata is stamped with ``last_status_source =
  "reconciliation"``. The stopped row stays visible and can be reset or
  resubmitted like a walltime stop.

The check runs when a project is opened, before the jobs table is drawn, so a
job with no process is never shown as running, and once a minute from a timer
the main window owns. The status bar reports how many jobs were stopped. Cluster
jobs are not part of this check: a cluster job with no process on this machine
may still be queued, running or finished on the cluster.

.. _job-reconcile:

Deciding a job's state from evidence
------------------------------------

``mufinder.services.job_reconcile.reconcile(job, evidence)`` is a pure function
that returns the state the evidence supports for a job. It reads no file,
contacts no cluster and changes nothing: a caller gathers a ``JobEvidence`` and
applies the returned ``Decision``. The same job and evidence always give the
same decision.

``JobEvidence`` holds what was seen of the job at ``checked_at``:

- ``process`` (local jobs): ``pending``, ``running`` or ``none`` (the local
  runner holds no process for the job). A process the runner sees end is
  reported by the runner itself and is not evidence.
- ``submitting`` (cluster jobs): a submit request has not answered yet.
- ``scheduler`` (cluster jobs): ``pending``, ``running``, ``completed``,
  ``cancelled``, ``timed_out``, ``failed``, ``not_listed`` (neither the queue
  nor accounting lists the job), ``no_job_id`` or ``unavailable`` (the queue
  command failed). ``scheduler_detail`` carries the
  scheduler's own state name or the error, and ``from_accounting`` marks a
  state read from accounting.
- ``output``: ``not_looked``, ``complete`` (parses and reports completion, with
  ``parsed`` and ``artifacts`` attached), ``incomplete`` (with CASTEP's message
  in ``output_detail`` and ``walltime_seen`` when the scheduler log shows a
  walltime kill), ``unreadable``, ``absent`` (no CASTEP output written by this
  run) or ``unreachable`` (the folder could not be read).
- ``remote_job_id`` and ``remote_directory``: the submission the evidence is
  about. Evidence about another submission of the job returns ``None``.
- ``on_request``: the user asked for the check.

``Decision`` holds the ``status`` and ``activity`` (the job's own when nothing
changes), the ``end_reason`` and ``message`` of a job it ends, an ``observation`` for
the job's last-checked note, and a ``marker`` for an active job that could not
be confirmed. A ``None`` reason on an ended job is diagnosed by the caller from
the job's logs. ``wants_output(scheduler, on_request=...)`` says, from the
scheduler state alone, whether the job's output folder is to be read: never
while the job is listed as pending or running or the scheduler was not heard
from, and after a cancellation or with no scheduler job ID only on request.

The callers are the :ref:`local-job-check`, the :ref:`cluster-checks` and the
dashboard's ``Check now`` action.

A local job in an active state:

.. list-table::
   :header-rows: 1

   * - Evidence
     - Decision
   * - process ``pending``
     - ``active``, ``queued``
   * - process ``running``
     - ``active``, ``running``
   * - process ``none``, output ``complete``
     - ``completed``
   * - process ``none``, output not complete
     - ``ended``, ``interrupted`` (reads ``stopped``)

A cluster job in an active state. A decision word is the status word the job
reads as: ``queued``, ``running`` and ``retrieving`` are ``active`` with that
activity, ``failed``, ``stopped`` and ``cancelled`` are ``ended``, and the
reason follows the comma.

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
     - ``queued``, whatever the job was doing (a requeued or suspended job
       reads ``queued``)
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
     - ``failed`` with the scheduler's state
   * - scheduler ``completed`` or ``not_listed``, output ``incomplete``
     - ``stopped`` (``walltime``) when ``walltime_seen``, otherwise ``failed``
       with CASTEP's message
   * - scheduler ``completed`` or ``not_listed``, output ``unreadable``
     - ``failed``, ``output_unavailable``
   * - scheduler ``completed``, output ``absent``
     - ``failed``, ``output_unavailable``
   * - scheduler ``not_listed``, output ``absent``
     - no change, with the lost-track marker

A cluster job is therefore never ended for want of evidence, and complete
output completes a job whatever the scheduler reports. A job whose cluster is
not connected is not checked at all; ``job_staleness`` marks it.

An ``ended`` job changes only on evidence with ``on_request``: scheduler
``pending`` or ``running`` makes it ``active`` again as ``queued`` or
``running``, and output ``complete`` completes it. Anything else leaves it
as it is. ``defined`` and ``completed`` jobs are never changed.

What keeps a job from staying active
------------------------------------

No separate safeguard watches for stuck jobs. Each of these follows from
deciding a job's state from evidence:

- A local job with no process is completed from complete output and stopped
  otherwise, on project open and once a minute (:ref:`local-job-check`).
- ``LocalJobRunner`` reports a ``QProcess`` ``FailedToStart`` error at once as
  ``job_failed``, so a job cannot stay ``submitting`` behind a process that
  never launched.
- A cluster job is checked on project open, on connect and once a minute, and
  ends when the scheduler or its output says it ended (:ref:`cluster-checks`).
- A cluster job that cannot be confirmed (cluster disconnected, no scheduler
  job ID, listed nowhere with no output) stays active, shows why on its
  dashboard row, and can be cancelled in the project. MuFinder never ends a
  cluster job because it lost track of it.
- A job that ended can be checked again with ``Check now``; complete output
  completes it.
- ``Dismiss`` stamps ``dismissed_at`` on a failed, stopped or cancelled job so
  the dashboard hides the row, while the job record, runs, results and
  provenance stay (see :ref:`dismissed-rows-stay-resubmittable`).

These are pinned by ``tests/services/test_job_reconcile.py``,
``tests/services/test_job_evidence.py``,
``tests/services/test_job_execution_controller.py``,
``tests/services/test_cluster_runner.py``,
``tests/app/test_cluster_job_checks.py`` and
``tests/app/test_job_check_wiring.py``.
