Running and Monitoring Jobs
===========================

Every workflow submits its calculations as jobs. This page describes how jobs
move through the ``Job Dashboard``, how the workflow job tables act on selected
rows, and how jobs run on this computer or on an HPC cluster. The service
contract behind these behaviours is described in :doc:`/job_lifecycle`.

Job states and the Job Dashboard
--------------------------------

.. figure:: ../_generated/screenshots/job_dashboard.png
   :alt: The main window with the Job Dashboard docked below the Cutoff
         Convergence page, listing a failed, a stopped and a cancelled job.

   Failed, stopped and cancelled cutoff jobs on the ``Job Dashboard``.

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
   A running relaxation shows its geometry-optimisation progress (see
   :ref:`geometry-optimisation-progress`).

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
``Workflow``, ``Calc Type``, ``Status``, ``Progress``, ``Last Checked``, ``Cluster``,
``Remote Job ID``, ``Runtime``, and ``Processes``, followed by per-row
``Resubmit``, ``Retry retrieval``, ``Cancel``, ``Inspect``, ``Reset``, and
``Dismiss`` buttons that are enabled only when they apply (see the action rules
under "Main-window integration" in :doc:`/job_lifecycle`). Right-clicking
selected rows offers the same actions, each acting on the selected jobs it applies to, plus
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

.. _geometry-optimisation-progress:

Geometry-optimisation progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/geometry_progress_dialog.png
   :alt: The geometry progress window of a running BFGS relaxation at iteration 4,
         with its four criteria tabulated and plotted against their tolerances.

   A running pristine relaxation of silicon, read from a synthetic CASTEP output.

A running pristine or muon relaxation shows how far its geometry optimisation
has got in the narrow ``Progress`` column, as ``iteration · criteria met``: for
example ``14 · 3/4`` means CASTEP has finished iteration 14 and three of the
four convergence criteria it tests (``dE/ion``, ``|F|max``, ``|dR|max``,
``Smax``) are within tolerance. A run that fixes every ion or the cell tests
fewer criteria, so its count reads out of three or two. ``starting`` means the
run has echoed its parameters but not finished a step. The cell is empty for
every other job and status. Hovering it lists each criterion's value against
its tolerance, the step limit (``GEOM_MAX_ITER``) and, after a continuation,
which restart the figures belong to. ``Inspect`` on a running relaxation opens
a progress dialog with the same table and a plot of each criterion over the
iterations against its tolerance; it updates as new steps arrive and has a
``Refresh`` button.

The progress comes from the job's growing ``.castep``
(``<seed>_farm001.castep`` when the run is task-farmed).
``mufinder.castep_io.geometry_progress`` reads the convergence table CASTEP
prints after every step, from the most recent restart only: CASTEP counts
iterations from 0 again after a continuation. An iteration counts once its
table is complete, so a file read mid-write never shows a partial count.
``mufinder.services.job_progress`` reads the output as soon as a relaxation is
seen running, then every 60 seconds while the dashboard is visible, and when a
progress dialog opens or its ``Refresh`` is pressed (at most once every 10
seconds per job). Each read fetches only what CASTEP has appended since the
last one. A local job's file is read on a background thread;
a cluster job's file is read over SFTP on the cluster runner's SSH worker
thread, so neither blocks the window. A read that fails keeps the last progress
shown. Progress is never saved in the project.

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
:ref:`job-continuation-primitive` for the technical contract.

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
  completed relaxations, and adds ``Reset to Stage 1`` and ``Reset to Stage 2``
  buttons.

Right-clicking selected rows opens the same menu in every workflow table, in
this order: ``Run Local``, ``Run on Cluster``, ``Cancel``, ``Inspect Failure``,
``Reset Failed Jobs...``, ``Configure...``, ``Remove``, then any table-specific
entries below a separator (``Continue relaxation…`` in the relaxation tables;
``Reset to Stage 1`` and ``Reset to Stage 2`` in Quantum Corrections, which
has no failed-job reset or ``Remove``). Right-clicking a row outside the
selection selects it first. ``Reset Failed Jobs...`` acts on the selected rows
that are ``failed``, ``stopped`` or ``cancelled`` (see
:ref:`per-row-reset-affordance`); Quantum Corrections uses its stage resets
instead.

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

.. figure:: ../_generated/screenshots/local_submission_dialog.png
   :alt: The local submission dialog for six cutoff-convergence jobs, set to
         run one after another.

   The local submission dialog for a six-row cutoff sweep.

``Setup > CASTEP`` stores the local CASTEP executable path and the ``mpirun``
launcher path; ``mpirun`` is also discovered on ``PATH`` and in common install
prefixes. Multi-core local runs need a configured or discovered ``mpirun``;
without one, cores per job are limited to 1. Installing the CASTEP keyword
catalog from the same dialog is described in :doc:`/castep_keyword_catalog`.

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

.. figure:: ../_generated/screenshots/cluster_setup_dialog.png
   :alt: The Remote Cluster Connections dialog with the built-in SCARF profile
         selected on the Connection tab.

   ``Setup > HPC Clusters`` with the built-in SCARF profile selected.

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
  batch scripts with no memory directive (see :ref:`slurm-memory-requests`).
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
  no overrides (see :doc:`/compatibility_policy`).

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
  ``#SBATCH --mem-per-cpu`` request (see :ref:`slurm-memory-requests`). The
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
  :ref:`re-completing-continued-job`);
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
