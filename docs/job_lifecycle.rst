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
   The job has been accepted and is waiting to run, including a cluster job
   SLURM holds, requeues, or is still configuring before it starts.

``running``
   CASTEP is running locally or on the selected cluster. When a queued cluster
   job starts, the status bar reports *"Cluster job started running: <label>"*.

``retrieving``
   A cluster job has finished on the scheduler and MuFinder is copying/parsing
   output files.

``completed``
   The calculation finished successfully. Completed jobs leave the dashboard.

``failed``
   The process, scheduler, retrieval, or parser reported a failure. Failed jobs
   remain in the dashboard so their diagnostics can be inspected with
   ``Inspect``. They can be resubmitted or reset. A cluster job that failed
   only because the scheduler stopped listing it is still watched: if SLURM
   later reports it queued, running, or completed, the job returns to the
   dashboard's active states and the status bar says that tracking resumed.

``stopped``
   A cluster calculation reached its walltime limit before producing a
   successful CASTEP completion marker: the scheduler reported ``TIMEOUT``, or
   the scheduler log downloaded with an incomplete output records a time-limit
   kill. Local jobs that were still active when MuFinder closed, and jobs stopped by
   the active-job watchdog, also end up here. Stopped jobs remain in the
   dashboard and can be inspected, resubmitted, or reset.

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
without losing their records. ``Resubmit`` is available for failed, stopped,
and cancelled jobs; relaxations ask first whether to continue from their
checkpoint or start again (see :ref:`resubmit-continue-or-start-again`).
Cluster resubmission reuses existing remote files when the job
still records a remote directory, so continuation runs can start from the
previous working directory, and failed or stopped cluster jobs that kept their
remote metadata can ``Retry retrieval`` instead of resubmitting. Reuse covers
the remote inputs and outputs only: the scheduler submit script is regenerated
on every submission, so an updated walltime, core count, or memory request
takes effect on resubmission.

.. _job-failure-reasons:

Why a job failed
~~~~~~~~~~~~~~~~

A failed or stopped row names its cause in the ``Status`` column, for example
``stopped · walltime`` or ``failed · memory``. Hovering the cell shows a
one-sentence explanation followed by the recorded failure message; a failure
with no recognised cause shows ``failed`` and its message, and points to
``Inspect``. ``Inspect`` opens with one line per job stating the cause, and its
details start each job with ``Cause:`` and, when a log shows it, the
``Evidence`` line and the file it came from, before the raw log excerpts.

The cause is recorded on ``Job.failure_reason`` when the job fails or stops.
``JobExecutionController.diagnose_failure`` reads the failure message and the
job's logs in its local output directory — the scheduler's ``*-<id>.out`` and
CASTEP's ``<seed>.err`` / ``<seed>[_farmNNN].NNNN.err``, which the cluster
runner downloads before reporting the failure — and
``mufinder.services.failure_diagnosis`` classifies them, highest priority first:

.. list-table::
   :header-rows: 1

   * - Label
     - Recorded when
   * - ``walltime``
     - The scheduler log records a time-limit kill (SLURM's ``DUE TO TIME
       LIMIT``, PBS's ``walltime … exceeded limit``) or the scheduler reported
       ``TIMEOUT``.
   * - ``memory``
     - The scheduler log records an ``oom-kill`` event or an exceeded memory
       limit, or the scheduler reported ``OUT_OF_MEMORY``.
   * - ``node fail``
     - The scheduler log records ``DUE TO NODE FAILURE``, or the scheduler
       reported ``NODE_FAIL``.
   * - ``CASTEP``
     - A CASTEP ``.err`` file has content; its first line is the evidence.
   * - ``no output``
     - The output could not be downloaded or parsed, or the job left the queue
       and its output never appeared.
   * - ``orphaned``
     - MuFinder was closed while a local job ran, or the active-job watchdog
       stopped a job no runner was tracking.
   * - ``submit``
     - The local runner or the scheduler rejected the submission.

A cluster job that stops is always recorded as ``walltime``: the cluster
runner reports ``stopped`` only for the scheduler's ``TIMEOUT`` state or for a
time-limit kill in the scheduler log of a run with incomplete output, so the
cause is known even when the scheduler log could not be downloaded.

A scheduler kill outranks a CASTEP error file, which outranks a retrieval
failure, so a job killed at its walltime before CASTEP wrote any output is
reported as ``walltime``, not ``no output``. Logs last modified before the job
was submitted are ignored, so a continuation in a reused directory is not
blamed on the previous run's errors. Jobs that failed before causes were
recorded show ``failed`` alone; ``Inspect`` still diagnoses them from their
logs. MuFinder never resubmits a failed job by itself: the cause tells the user
whether to raise the walltime or memory request, continue from the checkpoint,
or fix the input. A ``walltime`` row's tooltip ends with that next step:
``Resubmit`` (continuing or starting again, for a relaxation), with a longer
walltime if needed.

.. _resubmit-continue-or-start-again:

Resubmitting a relaxation: continue or start again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every relaxation ``.param`` contains ``CONTINUATION : DEFAULT``, so a relaxation
resubmitted into its own run directory resumes from the ``{seed}.check`` file
the previous run left there, and starts from its initial structure when there
is none. When ``Resubmit`` includes pristine or muon relaxations, MuFinder
therefore asks once for the whole selection before the submission dialog:

``Continue from the checkpoint``
   Resubmits in place, keeping the previous run's output. The dialog says how
   many of the relaxations have a checkpoint, have none (and will start from
   their initial structure), or could not be checked (a cluster job while its
   cluster is not connected).

``Start again from the initial structure``
   Deletes the previous run's output, checkpoint included, then runs the job's
   inputs from the start. On the job's own cluster the remote directory is
   deleted and the ``.cell``/``.param`` uploaded afresh, with no overwrite
   prompt; locally the output directory is cleared exactly as Reset's cleanup
   clears it (everything but the ``.cell``/``.param`` inputs, including
   ``{seed}-out.cell``). The deletion happens only once the submission dialog
   is accepted. A job whose local output cannot be cleared is not submitted,
   since it would resume from a checkpoint the user asked to discard. While
   ``Start again`` is selected, Enter cancels the dialog rather than
   resubmitting.

Starting again deletes only what belongs to the job:

- **Its directories must be its own.** Before the choice is shown, a
  relaxation is refused (neither started again nor continued, with a message
  naming the reason) when its local output folder
  is not strictly inside ``<project>/files/<stage>/`` or not named after the
  job's seed, when its remote directory is not named after the seed, or when
  another job uses the same local folder, the same remote directory on the same
  cluster, or a directory inside either. Reconfiguring a relaxation row can
  leave the old failed job and its replacement sharing one directory; without
  this check, starting the old job again would delete the replacement's
  output, and continuing it would run in the replacement's directory and
  overwrite its checkpoint. Two jobs in one selection sharing a directory are
  both refused. The same check runs again just before anything is deleted.
- **The previous run must have finished.** A stopped job may still be running:
  the active-job watchdog marks jobs this session lost track of as
  ``stopped · orphaned`` without cancelling them. Before deleting a remote
  directory, the submission worker asks the scheduler about the job's previous
  scheduler id and refuses (the submission fails, nothing is deleted) while it
  is still queued or running, or when the scheduler cannot be asked.
- **Symlinks are removed, never followed**, both in the local clean and in the
  remote delete, so a link inside a job directory cannot lead the delete into a
  shared directory it points at.
- **Only the exact ``{seed}-out.cell`` counts as output.** Every other
  ``.cell``/``.param`` is kept, so a seed that itself ends in ``-out`` keeps its
  input.
- A job selected to start again whose remote directory another job in the same
  submission is continuing in is dropped rather than submitted, since its
  checkpoint would survive.

The preselected choice follows the recorded causes. ``walltime``, ``memory``,
``node fail``, and ``orphaned`` leave a healthy run that was cut short or
under-resourced, so they default to continuing; a CASTEP error, a rejected
submission, a cancellation, or no recorded cause defaults to starting again.
Starting again deletes output, so a selection with mixed causes defaults to
continuing if any of them favours it. Other jobs in the selection (convergence,
UEP, Energy Barriers, Quantum Corrections) are resubmitted as before, and
imported relaxations are never offered a fresh start. ``Continue relaxation…``
in a relaxation table remains the way to change a row's parameters before
resuming.

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

Reset's optional cleanup deletes everything in a job's output folder except its
``.cell``/``.param`` inputs, so it applies the same ownership check as
:ref:`starting a relaxation again <resubmit-continue-or-start-again>`, limited to
the local folder (Reset never touches the cluster). The folder is kept, and the
job reset without cleaning, when it is outside ``<project>/files/<stage>/``, is
not named after the job, or is used by another job that is not being reset in
the same action. This is what protects a Quantum Corrections row: its phonon job
shares a seed directory, and the stage-1 checkpoint in it, with the singlepoint,
so resetting a failed phonon job keeps that directory while resetting both
stages together cleans it. The kept folders are listed in the "Directory
cleanup issues" message.

The pristine- and muon-relaxation tables additionally offer a per-row
``Continue relaxation…`` action for ``completed``, ``failed``, ``stopped``,
or ``cancelled`` rows: it edits that row's parameters in place and resumes
the same job from its CASTEP checkpoint rather than starting over, so it is
the recovery path that keeps a run directory's ``.check``/``.castep`` files
intact (unlike ``Reset``, whose optional cleanup deletes them). See
:ref:`continue-relaxation` for the user-facing description and
:ref:`job-continuation-primitive` below for the technical contract.

Job tables and selections
-------------------------

The workflow job tables (cutoff and k-point convergence, pristine and muon
relaxation, UEP, Energy Barriers, Quantum Corrections) and the Job Dashboard
share one rule for controls embedded in a row: highlighted rows are what is
affected.

- A control on a row that is part of a multi-row selection targets the
  selection. When at least two selected rows can take the action, MuFinder
  asks first, with three choices: apply to all selected (or eligible) rows,
  apply to this row only, or cancel. Cancelling changes nothing, and a
  drop-down returns to the value it showed before. When only one selected row
  can take it, the clicked row is acted on without a prompt.
- A control on a row outside the selection first makes that row the selection,
  then acts on it alone, without a prompt.
- The prompt uses the same wording in every table. It names the action and the
  number of rows, and when some selected rows cannot take the action (for
  example rows locked because their job has been queued) it states how many
  will be skipped.
- After the prompt is answered the table selection is set to the selected
  rows (``All``) or the clicked row (``This ... only``); cancelling restores the
  previous selection.
- Controls that propagate: the ``Supercell`` and ``Fix Cell`` drop-downs, the
  per-row ``Configure`` and remove buttons, and every Job Dashboard action
  button.
- Controls that only make sense for one row (``Open Wizard`` in the UEP table,
  ``View...`` in the Energy Barriers table) always act on their own row and
  never prompt.

Actions that remove, cancel, reset, dismiss or delete are always confirmed,
whichever way they are triggered (row control, button below the table,
right-click menu, or Delete/Backspace):

- Removing rows asks once for the whole batch, stating the number of rows and
  how many already have a submitted job (the cutoff and k-point tables also
  state how many jobs are still queued or running and will be cancelled). This
  includes rows that were never submitted. Relaxation removals may then ask
  what to do with each job's output files (delete or archive, with an
  apply-to-all choice); for cutoff rows with associated files the file-handling
  prompt offers ``Yes to All`` / ``No to All``.
- For a row control on a multi-row selection the three-way prompt is the
  confirmation, so the user is not asked twice. The exception is removing
  cutoff or k-point rows whose job is still queued or running: that prompt
  does not mention cancellation, so those removals are confirmed again with it
  stated.
- A single-row ``Cancel`` or ``Dismiss`` on the Job Dashboard, and ``Cancel``
  or ``Dismiss`` in a workflow table's or the dashboard's right-click menu, ask
  before acting. ``Reset`` asks its clean-up question, which can be cancelled.
- In the dedicated confirmations the safe choice is the default button and the
  one Escape selects, and the buttons are labelled with the action
  (``Remove 3 rows``, ``Keep running``) rather than Yes/No. In the three-way
  row-control prompt the default button is the all-selected choice.

What a click will affect is shown before the click, without changing any text
inside the rows, so tables stay usable on a small laptop screen:

- While two or more rows are selected, the row controls that will act on all of
  them are drawn with a heavier border. Single-row controls are not.
- Each workflow table shows ``N selected`` beside its buttons. The Job
  Dashboard shows ``N selected · row buttons apply to all selected`` above the
  table, only while two or more rows are selected.
- After a multi-row ``Supercell`` or ``Fix Cell`` change the status bar reports
  how many rows were changed and how many were skipped.
- The run buttons state their scope: ``Run All Locally`` / ``Run All on
  Cluster`` with nothing selected, ``Run Selected Locally`` / ``Run Selected on
  Cluster`` otherwise. With nothing selected every row is submitted; rows that
  are not ready are skipped by the submission step.
- The button row below each table offers the same set: ``Configure Selected
  Parameters...``, ``Remove Selected``, and the two run buttons. Quantum
  Corrections has no ``Remove Selected`` because its rows are derived from
  completed relaxations, and adds a ``Reset to Stage 1`` button.

Right-clicking selected rows opens the same menu in every workflow table, in
this order: ``Run Local``, ``Run on Cluster``, ``Cancel``, ``Inspect Failure``,
``Reset Failed Jobs...``, ``Configure...``, ``Remove``, then any table-specific
entries below a separator (``Continue relaxation…`` in the relaxation tables;
``Reset to Stage 1`` in Quantum Corrections, which has no failed-job reset or
``Remove``). Right-clicking a row outside the selection selects it first.
``Reset Failed Jobs...`` acts on the selected rows that are ``failed``,
``stopped`` or ``cancelled`` (see :ref:`per-row-reset-affordance`); Quantum
Corrections uses its stage-1 reset instead.

A selection may mix rows an action applies to with rows it does not (a running
row when resetting, a completed row when running). Every action then acts on
the rows it applies to and says how many it left out:

- Run, Cancel, Inspect and Reset in the row menu are offered when at least one
  selected row qualifies, act on those rows only, and carry the count when it
  is partial, for example ``Reset Failed Jobs... (2 of 5)``. An entry is greyed
  out only when no selected row qualifies. A row with no label yet has no job
  and counts as not qualifying.
- The run buttons and the menu's run entries submit the rows that are ready and
  the status bar reports the rest: ``Submitted 3 local job(s); skipped 2
  selected rows that were not ready``.
- The three-way prompt of a row control states the same split (``Set this
  supercell on 3 of 5 selected rows? 2 are not eligible and will be
  skipped.``).

The Job Dashboard's right-click menu offers ``Resubmit``, ``Retry
Retrieval``, ``Cancel``, ``Inspect``, ``Reset``, ``Dismiss`` and
``Delete...``. Each entry is enabled when it applies to at least one selected
job, acts on those jobs only, as the row buttons do, and carries the count when
it is partial (``Reset (2 of 3)``). ``Delete...`` does what Delete/Backspace
does. The row buttons remain the primary controls.

The shared implementation lives in ``mufinder.app.selection_snapshot`` (row
targeting and the three-way prompt) and ``mufinder.app.workflow_table_utils``
(``confirm_destructive``, ``standard_row_actions``, ``SelectionFeedback``).

Running jobs
------------

Every workflow job table runs its ``ready`` rows either on this computer or on
a configured cluster. A row's inputs are written when it is submitted, not when
it is configured: each run action first saves the workflow's current
configuration snapshot and then submits the rows.

Local execution
~~~~~~~~~~~~~~~

``Setup > CASTEP`` stores the local CASTEP executable path and the ``mpirun``
launcher path; ``mpirun`` is also discovered on ``PATH`` and in common install
prefixes. Multi-core local runs need a configured or discovered ``mpirun``;
without one, cores per job are limited to 1. Installing the CASTEP keyword
catalog from the same dialog is described in :doc:`castep_keyword_catalog`.

The local submission dialog sets:

- cores per job, with guidance from the logical CPU count;
- sequential execution, or parallel execution with a user-selected maximum
  concurrency;
- an oversubscription warning when cores per job multiplied by the
  concurrency exceeds the logical CPUs;
- a task-farming check that blocks ``OK`` when a row's farm count exceeds the
  cores per job.

Jobs run in the background through Qt ``QProcess``, so the UI never blocks,
inside the job directories created at submission time. A local job passes
through ``queued`` only when the parallel-concurrency limit is already full.
Status text in the dashboard and the workflow tables is colour-coded.

Cluster profiles and connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Setup > HPC Clusters`` stores cluster profiles globally, including built-in
SCARF and Redwood presets. The dialog lists the saved clusters on the left,
with ``Add``, ``Duplicate`` and ``Remove``, and edits the selected one in four
tabs: ``Connection``, ``Scheduler``, ``Queues`` and ``Keyword generation``.
Edits are kept per cluster while switching between them and are saved together
on ``OK``. ``Remove`` (or Delete/Backspace in the cluster list) asks first.
Cluster names must differ by more than letter case.

- A profile sets the hostname, username, queue system, default working
  directory, cores per node, default walltime and CASTEP executable path, plus
  a submission-script template path for SLURM profiles (blank uses the
  built-in SLURM template) and ``Default memory per core (GB)`` for
  ``addqueue`` profiles. Walltimes use SLURM's formats (``HH:MM:SS``,
  ``D-HH:MM:SS`` and the shorter forms) and are checked on save.
- A profile lists its queues (SLURM partitions or ``addqueue`` queues); see
  `Cluster queues`_.
- A SLURM profile may set a default per-CPU memory request, which pre-seeds the
  submission dialog. It is off by default, so a profile without it generates
  batch scripts with no memory directive (see `SLURM memory requests`_).
- Optional SSH jump-host settings serve clusters behind gateways, with a
  separate jump hostname, port, username, authentication method and key path.
- The Redwood preset targets ``glamdring.physics.ox.ac.uk`` through the Physics
  gateway ``bastion.physics.ox.ac.uk``. MuFinder prompts separately for the
  Physics/bastion and Glamdring credentials because those accounts can differ.
  The preset submits with ``/usr/local/shared/bin/addqueue``; a saved profile
  keeps whatever submit command path it was saved with.
- On macOS and Linux, eligible Redwood/Glamdring password-auth profiles can
  opt into read-only system keyring password lookup, so neither password is
  prompted for. MuFinder reads existing entries and never stores passwords.
  On macOS the entries live in the login Keychain and use the same services
  as ``glamdring-tools``: ``security add-generic-password -a
  <physics_username> -s bastion -w`` and ``security add-generic-password -a
  <glamdring_username> -s glamdring -w``. On Linux they live in the Secret
  Service keyring (GNOME Keyring or KWallet), are read with
  ``/usr/bin/secret-tool`` (Ubuntu package ``libsecret-tools``) and are
  created with ``secret-tool store --label=<label> service bastion account
  <physics_username>`` and ``secret-tool store --label=<label> service
  glamdring account <glamdring_username>``; the keyring must be unlocked in
  the desktop session. The setup dialog's ``?`` button shows the exact
  commands for the current platform. The lookup runs off the GUI thread, so a
  keyring waiting on an unlock prompt never freezes the window; if it takes
  noticeably long, a busy dialog appears whose ``Cancel`` button (or Escape)
  abandons the lookup without connecting. A lookup that fails, finds an empty
  password or does not finish within 120 seconds stops the connection and
  lists the missing entries with their setup commands. The persisted
  ``use_macos_keychain_passwords`` setting enables the lookup on both
  platforms.
- Scheduler commands (submit, queue, cancel) can be given explicit paths. A
  command without a configured path is located by probing the
  non-interactive SSH shell.
- SSH key authentication is primary, with password fallback.
- Unknown SSH host keys are never trusted automatically. MuFinder shows the
  key's SHA-256 fingerprint and asks before trusting it, for both the jump host
  and the target cluster host, and stores each trusted key in the user's
  settings; bastion trust records are stored separately from Glamdring trust
  records. A host key that differs from the trusted one stops the connection
  and shows the stored and presented fingerprints.
- Connection checks (the cluster setup ``Test Connection`` action and the
  pre-flight check before cluster keyword generation) run on a worker thread
  behind a cancellable progress dialog, so an unreachable or slow host never
  freezes the GUI; unknown-host-key prompts appear on the GUI thread and the
  check retries automatically once the key is trusted.

Cluster queues
~~~~~~~~~~~~~~

A cluster with several queues is one profile, not one profile per queue. The
``Queues`` tab lists the profile's queues in a table and picks the default
queue. Each queue has a name and optional overrides of the profile's values:
default walltime, maximum walltime, cores per node, memory (``Memory per CPU
(GB)`` for SLURM, ``Memory per core (GB)`` for ``addqueue``), CASTEP executable
and, for SLURM, the template script. A blank cell uses the profile's value from
the ``Scheduler`` tab; the maximum walltime has no profile value, and a queue
without one has no limit that MuFinder checks.

- Saving refuses (by ``queue_issues`` in
  ``mufinder.services.cluster_queues``) a queue without a name, a repeated
  name, a name that would need shell quoting (letters, digits, ``.``, ``_``
  and ``-``, with commas between several SLURM partitions), an unreadable
  walltime, a default walltime above the maximum (including one inherited
  from the profile), and a cores-per-node or memory value that is not
  positive. The dialog also
  refuses to switch to another cluster until the table is valid, so no cell
  is dropped unseen. Removing a queue asks first, from the button or
  Delete/Backspace. Renaming the default queue keeps it the default.
- ``ClusterConfig.for_queue`` resolves a queue's overrides onto its profile.
  The submission planner applies the chosen queue's template, cores per node
  and executable to every job of a submission, and keyword generation runs
  on the default queue with its overrides.
- A profile saved before queues existed, holding a single
  ``default_queue_name``, loads unchanged and shows that queue as one row with
  no overrides (see :doc:`compatibility_policy`).

Cluster submission and monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``Run All on Cluster`` / ``Run Selected on Cluster`` opens a dialog that
  chooses the cluster and overrides its runtime settings for this submission.
- ``Queue or partition`` lists the profile's queues for SLURM and ``addqueue``
  alike, starting on the default queue. Picking a queue fills cores per job,
  maximum runtime and memory from that queue; a hint shows the queue's
  maximum walltime, and says so when the job template sets its own partition.
  A name the profile does not list can be typed and is submitted as it is.
  The dialog refuses a walltime above the queue's maximum and a queue name
  that would need quoting. The last-used queue is remembered per project with
  the other last-used settings.
- MuFinder uploads the generated ``.cell`` / ``.param`` inputs and submits
  them to the cluster's scheduler. SLURM submission runs ``sbatch`` on a
  populated script template and captures the scheduler job id, with the
  chosen queue as an ``#SBATCH -p`` directive and an optional
  ``#SBATCH --mem-per-cpu`` request (see `SLURM memory requests`_). The
  partition is written into the script, never passed to ``sbatch``, and is
  left out when the template already names one with ``-p`` or
  ``--partition`` (as the built-in SCARF template does), so a hand-written
  partition wins. Redwood submission runs ``addqueue`` with the chosen queue
  as ``-q`` and the runtime controls ``Memory per core (GB)`` and
  ``Submission note``, in sbatch-compatible response mode so the job id parses
  reliably.
- Cluster submission refuses job labels containing anything other than letters,
  digits, ``.``, ``_`` and ``-``, because remote paths and scheduler commands
  use the label directly; the cluster runner checks each seed name again before
  submitting. Remote job directories are deleted recursively over SFTP, never
  through a remote shell command.
- Jump-host submissions show separate credential fields for the gateway and
  the target cluster. With the system keyring option enabled, submission and
  reconnect use the keyring entries instead of password prompts.
- When a matching cluster connection is already active, submission reuses it
  and does not ask for a password or passphrase again.
- Reconnecting reuses the password or passphrase cached from the last
  connection only while the profile keeps the same name, host, port and
  username for both the target and the jump host, and the jump host stays
  enabled or disabled; after any of them changes, the cached secret is never
  offered to the new endpoint.
- The dashboard shows each job's execution target (``Cluster``, which reads
  ``local`` for local jobs), remote job id, a live ``Runtime`` and the process
  count used at submission, with a connection-state indicator and a manual
  queue refresh. A manual refresh reconnects disconnected sessions for clusters
  that have active dashboard jobs.
- The queue is polled periodically, updating each job's last-checked time.
  SLURM polling tolerates stale or expired job ids (``Invalid job id
  specified``) by falling back to per-job status checks instead of failing the
  whole refresh.
- Connection, queue refresh, submission, cancellation and polling failures show
  a one-line summary, with the full error and recovery advice under
  ``Details``.

Outputs and failure diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After a run, MuFinder parses the ``.castep`` output and persists the parsed
data as a ``Result`` linked to the job's ``CalculationRun``. The parser
extracts:

- the completion and failure markers (see *What counts as completion* under
  `Re-completing a continued job`_);
- the final total energy, preferring, in order, the finite-basis-corrected
  total energy, the dispersion-corrected ``NB est. 0K`` energy, and the
  ``NB est. 0K energy``;
- the enthalpy, where available;
- the latest force and stress blocks, and the maximum force magnitude and
  maximum stress component magnitude;
- the SCF cycle count and wall time;
- warnings, error markers and geometry-convergence markers.

The convergence tables show the total energy and maximum force with enough
precision to compare small differences.

``Inspect`` on a failed job discovers the error files in its output directory,
``<seed>.err``, ``<seed>.NNNN.err`` and scheduler ``slurm-*.out`` files, and
shows context excerpts around the failure lines, after the cause summary
described in :ref:`job-failure-reasons`.

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
     - ``submitting``, ``queued``, ``running``, ``retrieving``, ``cancelled``
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
  ``retrieving``) to ``stopped`` and records a message and an optional
  ``failure_reason`` (see :ref:`job-failure-reasons`). The ``submitting`` and
  ``queued`` source edges exist so orphaned local jobs left mid-submission when
  MuFinder closed can be reconciled on the next project open (see
  ``reconcile_orphaned_local_jobs`` below).
- ``fail_job`` moves an in-flight job to ``failed`` and records a message and
  an optional ``failure_reason``.
- ``cancel_job`` moves a cancellable job to ``cancelled`` and clears
  ``failure_reason``. Every transition out of ``failed``/``stopped``
  (``submit_job``, ``complete_job``, ``reset_failed_job``,
  ``prepare_job_continuation``, ``retry_job_retrieval``,
  ``resume_reappeared_job``) clears it alongside ``error_message``.
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
- ``resume_reappeared_job`` moves a ``failed`` job to ``queued``, ``running``,
  or ``retrieving`` (the FAILED → QUEUED / RUNNING / RETRIEVING edges) when the
  scheduler lists it again; see :ref:`missing-scheduler-entry-recovery`. It
  clears ``error_message``, ``failure_reason`` and ``completed_at``, stamps
  ``started_at`` for ``running``, and preserves remote metadata. The
  missing-scheduler-entry precondition is enforced by
  ``JobExecutionController.resume_reappeared_cluster_job``, not the primitive.
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
active jobs (the cluster re-tracking that runs when a project is opened or
refreshed; see below).

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
the directory (the quantum-corrections stage reset, gated by the same four
statuses).
Deleting is right there because the row genuinely starts over. A relaxation
continuation must instead keep the job's identity: the job id is what binds the
workflow-table row to the calculation, and the job record is what carries the
output/remote directory holding the checkpoint the continuation depends on.
Deleting and re-creating would sever the row binding and abandon the directory.

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

A local job is probed with a plain ``pathlib.Path`` check for ``{seed}.check``
in its job directory, reporting ``found`` or ``absent`` (or ``unavailable``
when the job's output directory cannot be resolved).

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
``status``/``relaxation_run_id``/``total_energy_eV`` are rewritten from the
same completion, so the site and the structure always describe the same run.

*A stale ``-out.cell`` fails the job.* A continuation re-uses its job directory,
so the previous run's relaxed structure is still sitting where the completion
path looks for it: a run that dies before writing a new one would have the old
geometry accepted as its result. Two guards cover this, each closing the other's
blind spot.

The local one is in both relaxation controllers' ``validate_artifacts``,
alongside the missing-file and parse-failure checks (see :doc:`architecture`,
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
watchdog stops), and ``"user"`` (dashboard retry-retrieval requests). A job
resumed because the scheduler lists it again is stamped ``"scheduler_poll"``. The
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
``retrieval`` field. The runners take the fields as keywords:
``LocalJobRunner.submit``, ``ClusterJobRunner.submit``,
``track_existing_job`` and ``retrieve_outputs_for`` each accept them through
``**retrieval``, rebuild the value, and keep it on their tracking record,
where it drives both the download of the optional files and
``RetrievalFlags.collect``. The local runner never downloads, so it ignores
``retrieve_den_fmt``.

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
in the batch submits into as-is (a continuation's preserved ``remote_directory`` or a quantum
phonon job's stage-1 directory) is never deleted, whatever another job's
overwrite answer was, and a start-again job whose own directory is such a
directory fails instead of resuming from the checkpoint it was meant to
discard. The worker uploads and submits each job,
emitting progress and per-job results that the GUI applies via
``finalize_cluster_submission``/``fail_cluster_submission``. A modal progress
dialog reports progress and offers a Cancel button; jobs prepared but not yet
submitted when cancelled are moved to ``cancelled``. When the worker reports no
progress for three minutes the dialog stops waiting: it asks the worker to stop
submitting, as Cancel does, and moves the jobs it has no result for to
``cancelled``.

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
  directory, local output directory, cluster name, and current active status,
  or a ``failed`` status with ``watch_for_recovery`` for a job awaiting
  reappearance (see :ref:`missing-scheduler-entry-recovery`).
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

- Only tracked jobs for the currently connected cluster are polled: the
  non-terminal ones, and failed ones watched for reappearance.
- SLURM is polled with ``squeue --all --jobs=<ids>``. ``--all`` keeps jobs in
  hidden or unavailable partitions in the output, which ``squeue`` otherwise
  leaves out. Every job ``squeue`` does not list is then looked up in job
  accounting with ``sacct --jobs=<ids> --allocations``
  (``Scheduler.history``), in one call per poll. Accounting still knows a job
  that a site wrapper, a ``SQUEUE_*`` filter, or a multi-cluster setup hides,
  and one that ``squeue`` has already forgotten. On a cluster whose accounting
  is disabled the lookup finds nothing; if ``sacct`` fails for any other
  reason, the poll leaves those jobs unknown. Neither disconnects the runner.
  The same lookup guards the deletion of an old run's directory before a
  resubmission (``ClusterJobRunner.scheduler_job_is_active``). There, an
  ``sacct`` failure other than disabled accounting is an error, and the
  directory is not deleted.
- Scheduler states are mapped to MuFinder states by the scheduler backend.
  SLURM's pending, configuring, held, requeued, suspended, and stopped
  (``SIGSTOP``) states (``PD``, ``CF``, ``RD``, ``RF``, ``RH``, ``RQ``,
  ``SE``, ``S``, ``ST``) map to ``queued``; ``R``, ``CG``, ``RS``, ``SI``, and
  ``SO`` to ``running``; ``CD`` to ``completed``; ``CA`` to ``cancelled``;
  ``TO`` to ``stopped``; and ``F``, ``NF``, ``OOM``, ``BF``, ``DL``, ``PR``,
  and ``RV`` to ``failed``. Accounting reports the long names (``PENDING``,
  ``CANCELLED by <uid>``), which map the same way. A state the backend does
  not know maps to ``queued`` and is not terminal, so it never claims the job
  is running. A running job that SLURM suspends keeps its ``running`` status,
  because no transition leads from ``running`` back to ``queued``.
- A state read from accounting is acted on exactly like one ``squeue``
  reported, so a pending job hidden from ``squeue`` stays ``queued`` however
  long it waits and is never retrieved or failed for being missing. Its poll
  observation reads "missing from scheduler queue; scheduler accounting
  reports QUEUED".
- ``queued`` and ``running`` emit status updates. A job that moves from
  ``queued`` to ``running`` also emits ``job_started``, after its ``running``
  update; the main window reports it in the status bar.
- ``cancelled`` marks the tracked job terminal and emits cancellation.
- ``stopped`` and ``failed`` mark the tracked job terminal and download the
  available error files and scheduler log
  (:func:`~mufinder.services.cluster_retrieval.retrieve_error_files`), so a stop
  can be diagnosed like a failure. ``stopped`` then emits ``stopped``;
  ``failed`` emits ``job_failed`` and ``failed``.
- ``completed`` triggers one retrieval pass. The runner emits ``retrieving``,
  then :func:`~mufinder.services.cluster_retrieval.retrieve_job_outputs`
  downloads the error files, ``.castep`` and the optional ``.den_fmt``,
  ``-out.cell``, ``.phonon``, or ``.ts`` outputs that the job's retrieval flags
  request (see Retrieval flags), parses them, and returns a
  :class:`~mufinder.services.cluster_retrieval.RetrievalResult`. It emits
  nothing; the runner turns the result into signals. Retrieval or parsing
  failure becomes ``failed``. Parsed incomplete CASTEP output becomes
  ``stopped`` when the downloaded scheduler log shows a walltime kill;
  otherwise it becomes ``failed``. Parsed completed output emits
  ``job_finished`` and ``completed``. The secondary outputs (``-out.cell`` /
  ``.phonon`` / ``.ts``) are parsed once here on the worker thread via
  :func:`~mufinder.services.job_artifacts.collect_job_artifacts` and carried to
  the GUI as the third ``job_finished`` payload, so the completion handler and
  controllers never re-parse them.
- A job neither ``squeue`` nor accounting lists, or one accounting reports as
  completed, usually finished, so retrieval is attempted immediately — but
  with a grace window. When the output cannot be
  downloaded (SFTP error, missing remote ``.castep``, a ``.castep`` older than
  this submission, connection hiccup), the result's
  :class:`~mufinder.services.cluster_retrieval.RetrievalFailure` is
  ``retry_eligible`` and the job is *not* failed yet. The runner retries on up
  to ``MISSING_ENTRY_RETRIEVAL_ATTEMPTS`` (3) consecutive polls, keeping the
  entry non-terminal, and only then fails the job. A job accounting reports as
  completed fails with the retrieval error, which ``Retry retrieval`` can
  re-attempt. A job listed nowhere fails with a diagnostic explaining that
  scheduler history may have expired, the queue command may be filtered, or
  the wrong cluster may be connected, and is then watched for reappearance
  (see :ref:`missing-scheduler-entry-recovery`). Parse failures of
  successfully downloaded output are never retry-eligible and fail
  immediately. The attempt counter resets if the scheduler lists the job as
  unfinished again.
- Every poll emits a ``queue_checked`` snapshot carrying ``checked_at``
  (``utc_now()``), per-job elapsed runtimes, and per-job human-readable
  observations (``observations_by_job_id``) describing what the scheduler
  reported for each polled job (for example "scheduler reports RUNNING" or
  "missing from scheduler queue; waiting for output (attempt 1 of 3)").
  ``JobExecutionController.mark_queue_checked`` persists these observations as
  per-job monitor metadata: ``last_checked_at``, ``last_status_source``
  (``"scheduler_poll"``), and ``last_status_message``.
- Poll failure (``squeue`` itself failing) disconnects the runner, clears the
  connected state, and emits a warning through the UI. Tracked job state is left untouched — active cluster
  rows go stale rather than failing, and the dashboard derives staleness from
  ``last_checked_at`` plus the disconnect (``services.job_monitoring``).

.. _missing-scheduler-entry-recovery:

Recovering a job the scheduler lists again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A job failed with the missing-scheduler-entry diagnostic may only have been
hidden: still pending in a queue neither ``squeue`` nor accounting showed. The
cluster runner therefore keeps it as a *watched* entry: terminal, but still
polled, with no retrieval attempted. Its poll observation reads "missing from
scheduler queue; not in scheduler accounting; watching for the job to
reappear".

- When ``squeue`` or accounting lists a watched job as queued, running, or
  completed, the runner emits ``job_reappeared(job_id, status)`` once and
  stops watching it. Any other scheduler state (cancelled, failed, timed out)
  also ends the watch, and the job stays ``failed``.
- ``JobExecutionController.resume_reappeared_cluster_job`` applies the
  signal through ``ProjectService.resume_reappeared_job``: ``queued`` and
  ``running`` resume as reported, and ``completed`` resumes as
  ``retrieving``. It registers a *fresh* ``CalculationRun``, as ``Retry
  retrieval`` does, so the failed run keeps its outcome, and stamps
  ``"scheduler_poll"`` monitor metadata. The main window then re-registers the
  job with the runner (``track_existing_job`` with the resumed status), and
  the next poll carries on normally, retrieving a completed job's output.
- ``Refresh Queue`` while disconnected reconnects the clusters of watched jobs
  as well as those of active ones, so a project whose only cluster jobs await
  reappearance is still polled.
- Only a job for which
  ``job_execution_controller.awaits_scheduler_reappearance`` holds is resumed:
  a ``failed`` cluster job with a remote scheduler id, the
  missing-scheduler-entry message as its ``error_message``, and no
  ``dismissed_at`` stamp. A job the user has since reset, resubmitted, or
  dismissed is left alone.
- Each ``queue_checked`` snapshot lists the jobs polled only as watches
  (``watched_job_ids``). ``JobExecutionController.stale_reappearance_watches``
  picks out those whose job no longer awaits reappearance — reset, dismissed,
  continued, deleted, or resubmitted under a new scheduler id — and the main
  window ends their watches with ``ClusterJobRunner.stop_watching``, so they
  leave the poll after one more cycle. ``mark_queue_checked`` does not stamp a
  stale watch's observation onto its job.
- The same predicate selects the failed jobs that the project-open
  re-tracking (and a manual queue refresh) registers with
  ``watch_for_recovery``, so the watch survives closing and reopening
  MuFinder.

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
- cluster ``job_started`` calls ``mark_started`` and reports the start in the
  status bar.
- cluster ``job_reappeared`` calls ``resume_reappeared_cluster_job`` and
  re-registers the resumed job with the runner.

The per-job document mutation happens synchronously, but the **heavy dashboard
work is coalesced**: rebuilding the journal (whole provenance), the jobs table
(which recreates per-row action buttons), and the workflow views, plus the
full-document autosave. Lifecycle handlers *request* a dashboard flush
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

The cluster queue-poll handler stamps per-job monitor metadata
(``last_checked_at``/source/message) and then also requests a flush rather
than flushing synchronously, so it is subject to the same debounce and rate
limit; a queue poll can itself fire every few seconds during a retrieval. A
burst of cluster jobs finishing across one or several polls therefore rebuilds
the views and writes the document at a bounded cadence rather than once per
job, which keeps the GUI responsive for large batches.

The dashboard displays active states plus ``failed``, ``stopped``, and
``cancelled``; rows whose job carries a ``dismissed_at`` stamp are filtered
out. A ``Last Checked`` column shows the relative age of ``last_checked_at``
with a stale marker for active cluster jobs whose cluster is disconnected or
whose last check is older than the ``services.job_monitoring`` threshold, and
a tooltip carrying ``last_status_source`` / ``last_status_message`` / the
staleness reason. Actions are enabled as follows:

- ``Resubmit``: ``failed``, ``stopped``, and ``cancelled`` (relaxations first ask
  whether to continue or start again; see :ref:`resubmit-continue-or-start-again`).
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
thin handler that resolves the keys through the workflow's
``WorkflowJobController.jobs_for_keys`` and hands the jobs to the shared reset
handler. Muon relaxation binds the menu once per
charge-state table, each capturing its own table, so a right-click acts on the
rows it was raised over rather than on whichever tab happens to be current.

After a reset the main window repaints every workflow table: the dashboard
refresh covers the convergence, pristine, muon and UEP tables, and energy
barriers and quantum corrections are refreshed through their controllers'
``refresh_view``, which no-ops until their lazily-created view exists.

Quantum corrections is the exception to the shape above. Its row stages a
singlepoint job and then a phonon job into one shared output directory, so its
recovery is a *stage* reset (which deletes both job records and clears the
directory) rather than a job reset. Its gate is: the row qualifies once it
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

Job bindings across definitions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Reconfiguring a workflow registers a new definition with its own job mapping,
so two definitions can bind the same unchanged row to different jobs. Before a
controller reads its active definition's bindings it copies them across from
the equivalent definitions, choosing by job state
(:mod:`mufinder.services.definition_mappings`; see :doc:`architecture`,
Definition mappings and row removal). An in-flight job (``submitting``,
``queued``, ``running``, ``retrieving``) beats a ``completed`` one, which beats
a ``failed``, ``stopped`` or ``cancelled`` one, which beats a ``defined`` one;
the two relaxation workflows instead rank a ``defined`` job above a failed,
stopped or cancelled one, so a re-staged row keeps its fresh job. A muon
relaxation binding is copied only once its job has left ``defined``. A tie
keeps the binding the active definition already holds, and dismissal does not
change a job's rank.

When a project is opened or refreshed, the main window re-registers persisted
active cluster jobs that have enough metadata to poll with the cluster runner:
non-local execution target, active status, remote job id, remote directory or an
inferable remote directory, and a local output directory.

Active-job watchdog
~~~~~~~~~~~~~~~~~~~

Orphaned-local reconciliation (safeguard 1) only covers the app-restart
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
  that is the staleness marker's responsibility (safeguards 2–3), not the
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

Stuck-job safeguards
--------------------

Nine safeguards together ensure that no failure mode leaves a job in the
dashboard indefinitely. Each entry records the behavior and where it lives.

1. Orphaned local-job reconciliation on project open.
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
   is unchanged. Reconciliation runs from ``MainWindow`` while the UI is being
   refreshed for the new project, before the jobs table is rendered, so the
   user never sees a phantom running row, and opening a project surfaces a
   status-bar message such as *"N local job(s) were stopped because
   MuFinder closed while they were in progress."* when any jobs are reconciled.

   Cluster jobs are deliberately left untouched by this path: they re-attach to
   the cluster runner instead through the project-open re-tracking described
   above (which re-registers active cluster jobs, including ``submitting``, that
   still carry a remote job id and directory). Cluster jobs that lack enough metadata to re-register —
   missing remote id, remote directory, cluster config, or local output
   directory — are surfaced by the staleness marker and the active-job
   watchdog (safeguards 2–3 and 7).

2. Persist monitor metadata and last-observed status.
   ``Job`` carries ``last_checked_at``, ``last_status_source``, and
   ``last_status_message`` (optional, ``None``-defaulted, persisted through the
   codec and cleared by ``reset_failed_job``). They are stamped from queue
   polls (``JobExecutionController.mark_queue_checked``), retrieval outcomes,
   local-runner lifecycle events, project-open reconciliation, watchdog stops,
   and user retry requests, with the fixed source vocabulary described under
   "Execution orchestration". The dashboard's ``Last Checked`` column renders
   the age plus a tooltip with the source and observation.

3. Make disconnected cluster jobs visibly stale.
   A queue poll failure still disconnects the runner and deliberately leaves
   job state untouched; staleness is derived, not stored.
   ``services.job_monitoring.job_staleness`` marks an active cluster job stale
   when its cluster is disconnected or its last check is older than the
   threshold (180 s), and the dashboard renders a " — stale" marker with the
   reason in the tooltip.

4. Add a retry-retrieval action for stopped/failed cluster jobs.
   A ``Retry retrieval`` dashboard action re-attempts output retrieval for a
   scheduler-completed cluster job that kept its remote metadata, avoiding an
   unnecessary duplicate scheduler job. It routes through
   ``JobExecutionController.prepare_retry_retrieval`` →
   ``ProjectService.retry_job_retrieval`` (the FAILED/STOPPED → RETRIEVING
   edges) and ``ClusterJobRunner.retrieve_outputs_for``, registers a fresh
   ``CalculationRun`` so the outcome cannot overwrite the prior terminal run,
   and resolves a not-connected cluster to ``failed`` rather than stranding
   the job in ``retrieving``.

5. Avoid treating every missing scheduler entry as completed without context.
   A job ``squeue`` does not list is looked up in scheduler accounting first,
   so a hidden pending or running job keeps its state. A job listed nowhere
   still attempts retrieval immediately, but output-unavailable failures are
   retried across a three-poll grace window before the job is failed with a
   diagnostic naming the likely causes (expired scheduler history, filtered
   queue command, wrong cluster) and suggesting retry or resubmission. That
   job stays watched and resumes if the scheduler lists it again. Parse
   failures of downloaded output fail immediately. See "During polling" and
   :ref:`missing-scheduler-entry-recovery` above.

6. Fail fast on local process start errors.
   ``LocalJobRunner._on_process_error`` handles
   ``QProcess.ProcessError.FailedToStart`` directly: it emits ``job_failed``
   and ``failed``, cleans up the active entry, and starts the next queued job,
   so a job cannot remain ``submitting`` when Qt does not deliver a finish
   event. A late ``finished`` event for that job is a no-op. Other
   ``ProcessError`` values keep the store-and-wait behaviour.

7. Add an active-job watchdog.
   A once-a-minute watchdog (``mufinder.services.job_watchdog``) detects
   active jobs with no runner tracking them — ``queued``/``running``/
   ``submitting`` local jobs missing from the local runner's pending/active
   sets, and active cluster jobs missing from the cluster controller's
   tracked-id mirror while their cluster is connected — and, after two-tick
   confirmation, stops them via ``JobExecutionController.stop_stuck_job``
   (through ``ProjectService.stop_job``) with a diagnostic message and
   ``last_status_source = "watchdog"``. See "Active-job watchdog" above.

8. Separate terminal history from the active dashboard.
   A ``Dismiss`` action stamps ``dismissed_at`` on a failed/stopped/cancelled
   job so the dashboard filters the row out, while keeping the job record,
   runs, results, and provenance intact (a ``job_dismissed`` milestone is
   recorded). There is no un-dismiss UI; the stamp is cleared by
   ``reset_failed_job``, which the job's workflow row triggers on its next
   submit (see :ref:`dismissed-rows-stay-resubmittable`).

9. Tests around stuck-row prevention.
   Project-open reconciliation, poll-failure inertness, the missing-entry
   grace window, the accounting fallback and reappearance recovery, ``QProcess`` startup failure, watchdog detection and wiring,
   retry-retrieval run bookkeeping, and dismissal visibility are all pinned by
   tests beside the existing ``JobExecutionController``, ``LocalJobRunner``,
   ``ClusterJobRunner``, and main-window action suites (see
   ``tests/services/test_job_watchdog.py``,
   ``tests/services/test_retry_retrieval.py``,
   ``tests/services/test_job_dismissal.py``, and
   ``tests/app/test_dashboard_monitoring_columns.py``).
