Running and Monitoring Jobs
===========================

Every workflow turns its calculations into jobs. You start them from the job
table on a workflow page, on this computer or on an HPC cluster, and follow
them in the ``Job Dashboard``, the panel docked at the bottom of the main
window. This page shows how to submit jobs, read the dashboard, watch a
relaxation, recover from a failure and set up a cluster.

For developers: the service contract behind these behaviours (state authority,
runner contracts and the stuck-job safeguards) is in :doc:`/job_lifecycle`.

Submitting jobs
---------------

Each workflow job table runs its ``ready`` rows either on this computer or on a
configured cluster. A row's inputs are written when it is submitted, not when
it is configured: each run action first saves the workflow's current
configuration and then submits the rows.

To submit jobs from a workflow page:

1. Configure the rows in the job table.
2. Optionally select the rows to run. Highlighted rows are what is affected.
3. Click the run button below the table. With nothing selected the buttons read
   ``Run All Locally`` and ``Run All on Cluster`` and every row is submitted;
   rows that are not ready are skipped. With rows selected they read ``Run
   Selected Locally`` and ``Run Selected on Cluster``.
4. Fill in the submission dialog (see `Local submission`_ and
   `Cluster submission`_) and click ``OK``.

The same actions are in a row's right-click menu, which opens the same menu in
every workflow table, in this order: ``Run Local``, ``Run on Cluster``,
``Cancel``, ``Inspect Failure``, ``Reset Failed Jobs…``, ``Configure…``,
``Remove``, then any table-specific entries below a separator (``Continue
relaxation…`` and ``Export Relaxed Structure`` in the relaxation tables), and
last, below another separator, ``Open Containing Folder`` and ``Export
Inputs/Outputs…``. Quantum Corrections has no ``Remove``. Right-clicking a row
outside the selection selects it first. ``Reset Failed Jobs…`` acts on the
selected rows that are ``failed``, ``stopped`` or ``cancelled`` (see
:ref:`per-row-reset-affordance`).

The last two entries reach a job's files on disk, whatever its status, and
work for completed jobs, which the Job Dashboard does not list:

``Open Containing Folder``
   Opens the selected row's job folder in the file manager. It needs exactly
   one selected row whose job folder exists on this computer; a cluster job's
   folder appears once its outputs are retrieved.
``Export Inputs/Outputs…``
   Asks for a folder and copies the CASTEP input and output files of every
   selected row's job into it, one sub-folder per job named after its seed
   (``_2``, ``_3``… are added when two jobs share a seed). The files copied
   are ``.cell``, ``.param``, ``-in.cell``, ``.castep``, ``-out.cell``, ``-out.cif``,
   ``.geom``, ``.md``, ``.ts``, ``.bands``, ``.phonon``, ``.den_fmt`` and the
   ``.err`` and ``.NNNN.err`` error files, including task-farmed ``_farmNNN`` copies.
   Checkpoints (``.check``, ``.castep_bin``) are left out. Replacing files
   already in the folder is confirmed first, and files that cannot be copied
   are listed when the copy ends. A folder that already holds the jobs' own
   folders (such as ``files/pristine_relaxation``) is refused, since the files
   would be copied onto themselves.

When an entry does not apply, it is greyed out and its tooltip says why: the
row has no job yet, or its job has no local folder or no CASTEP files.

The button row below each table also offers ``Configure Selected
Parameters…`` and ``Remove Selected``. Quantum Corrections has no ``Remove
Selected`` because its rows are derived from completed relaxations, and adds a
``Reset Failed Jobs…`` button that acts like the menu entry.

Local submission
~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/local_submission_dialog.webp
   :alt: The local submission dialog headed 'Run 6 Cutoff Energy jobs', with
         cores per job, jobs at once and the cores the run uses.

   The local submission dialog for the six-row cutoff sweep of the CoF₂
   example (:doc:`/user_guide/examples/cof2`).

``Run All Locally`` and ``Run Selected Locally`` open ``Run Calculations
Locally``. Before the first local run, choose the CASTEP executable and the
``mpirun`` launcher in ``Setup > CASTEP``: ``CASTEP executable`` and ``MPI
launcher``, each with ``Browse…``, the latter also with ``Auto-detect``. Each
path is saved as soon as you finish editing it (see
:doc:`/getting_started/castep_setup`). ``mpirun`` is also discovered on ``PATH`` and in common
install prefixes. Multi-core local runs need a configured or discovered
``mpirun``; without one, cores per job are limited to 1. Installing the CASTEP
keyword catalog from the same dialog is described in
:doc:`/castep_keyword_catalog`.

The heading names the page the jobs come from and how many there are (``Run 6
Cutoff Energy jobs``); hold the pointer over it to list the jobs.

1. Set ``Cores per job``. The first local run in a project offers every
   physical core of this computer (every logical core where the physical count
   cannot be read): OpenMPI's ``mpirun`` refuses more processes than physical
   cores, so on a computer with two threads per core the logical count would
   not start. Later runs offer the count used last in that project, capped at
   this computer's physical cores the same way. Without ``mpirun`` the count is
   1 and is not remembered.
2. With more than one job, set ``Jobs at once``: 1 runs them one after
   another, a larger number runs that many side by side.
3. Read the line beneath: ``Uses 16 of 20 cores`` multiplies the two numbers
   and compares the total with this computer's logical cores. It turns to the
   warning colour when the total is larger, since jobs that share cores run
   slower. A task-farming check blocks the run button when a row's farm count
   exceeds the cores per job.
4. Click ``Run 6 jobs`` (the button gives the count).

Jobs run in the background, so the window never blocks, inside the job
directories created at submission time. A local job passes through ``queued``
only when the parallel-concurrency limit is already full. Status text in the
dashboard and the workflow tables is colour-coded.

Cluster submission
~~~~~~~~~~~~~~~~~~

``Run All on Cluster`` and ``Run Selected on Cluster`` open ``Run Calculations
on a Remote Cluster``. It chooses the cluster and overrides its runtime
settings for this submission.

1. Pick the ``Remote cluster``.
2. Pick a ``Queue or partition``. It lists the profile's queues for SLURM and
   ``addqueue`` alike, starting on the default queue. Picking a queue fills
   cores per job, maximum runtime and memory from that queue; a hint shows the
   queue's maximum walltime, and says so when the job template sets its own
   partition. A name the profile does not list can be typed and is submitted as
   it is. The dialog refuses a walltime above the queue's maximum and a queue
   name that would need quoting. The last-used queue is remembered per project
   with the other last-used settings.
3. Check ``Remote working directory`` and ``Username``, then set
   ``Processor cores per job`` and ``Maximum runtime`` (``HH:MM:SS``).
4. Set the memory request. For a SLURM cluster tick ``Memory per CPU (GB)`` to
   request memory per allocated CPU; it is off by default, so jobs are
   submitted with no memory directive (see :ref:`slurm-memory-requests`). For
   an ``addqueue`` cluster set ``Memory per core (GB)`` and, optionally,
   ``Submission note``.
5. Enter ``Cluster password or key passphrase``, and for a cluster behind a
   gateway ``Jump host password or key passphrase``, unless a matching
   connection is already active or the system keyring option supplies them (see
   `Cluster profiles and connections`_).
6. Click ``OK``.

MuFinder uploads the generated ``.cell`` / ``.param`` inputs and submits them to
the cluster's scheduler.

- On SLURM the chosen queue becomes an ``#SBATCH -p`` directive and the memory
  request an ``#SBATCH --mem-per-cpu`` line. The partition is written into the
  script and left out when the template already names one with ``-p`` or
  ``--partition`` (as the built-in SCARF template does), so a hand-written
  partition wins.
- The job template's ``$job_name`` is the job's CASTEP seed name, so a SLURM
  queue lists each job under its own label (``cutoff_convergence_ecut_400``,
  ``quantum_phonon_…``, ``barrier_…``), and a batch mixing workflows keeps each
  job's own name.
- On Redwood, MuFinder submits with ``addqueue``, using the chosen queue,
  ``Memory per core (GB)`` and ``Submission note``.
- Cluster submission refuses job labels containing anything other than letters,
  digits, ``.``, ``_`` and ``-``, because remote paths and scheduler commands
  use the label directly. Remote job directories are deleted recursively over
  SFTP, never through a remote shell command.
- When a matching cluster connection is already active, submission reuses it
  and does not ask for a password or passphrase again.

Working with selected rows
~~~~~~~~~~~~~~~~~~~~~~~~~~

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
- Controls that propagate: the ``Supercell`` drop-down, the ``Fix cell``
  checkbox, the per-row ``Configure`` and remove buttons, and every Job Dashboard
  action button.
- Controls that only make sense for one row (``Open Wizard`` in the UEP table,
  ``View…`` in the Energy Barriers table) always act on their own row and
  never prompt.

What a click will affect is shown before the click, without changing any text
inside the rows, so tables stay usable on a small laptop screen:

- While two or more rows are selected, the row controls that will act on all of
  them are drawn with a heavier border. Single-row controls are not.
- Each workflow table shows ``N selected`` beside its buttons. The Job
  Dashboard shows ``N selected · row buttons apply to all selected`` above the
  table, only while two or more rows are selected.
- After a multi-row ``Supercell`` or ``Fix cell`` change the status bar reports
  how many rows were changed and how many were skipped.

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

Actions that remove, cancel, reset, dismiss or delete are always confirmed,
whichever way they are triggered (row control, button below the table,
right-click menu, or Delete/Backspace):

- Removing rows asks once for the whole batch, stating the number of rows and
  how many already have a submitted job (the cutoff and k-point tables also
  state how many jobs are still queued or running and will be cancelled). This
  includes rows that were never submitted. Relaxation removals may then ask
  what to do with each job's output files (delete or archive, with an
  apply-to-all choice); for cutoff rows with associated files the file-handling
  prompt offers ``Delete files``, ``Move files`` and, for several rows,
  ``Delete all`` / ``Move all``.
- For a row control on a multi-row selection the three-way prompt is the
  confirmation, so you are not asked twice. The exception is removing cutoff or
  k-point rows whose job is still queued or running: that prompt does not
  mention cancellation, so those removals are confirmed again with it stated.
- A single-row ``Cancel`` or ``Dismiss`` on the Job Dashboard, and ``Cancel``
  or ``Dismiss`` in a workflow table's or the dashboard's right-click menu, ask
  before acting. ``Reset`` asks its clean-up question, which can be cancelled.
- Deleting a saved supercell, a comparison table, a magnetic field workspace,
  an analysis or dashboard jobs, and removing imported relaxations, are
  confirmed the same way.
- In every confirmation the safe choice is the default button and the one
  Escape selects, and the buttons are labelled with the action
  (``Delete supercell`` / ``Cancel``, ``Remove 3 rows``, ``Keep running``)
  rather than Yes/No. In the three-way row-control prompt the default button
  is the all-selected choice.
- Questions that change nothing destructive also name their answers:
  ``Save`` / ``Don't save`` / ``Cancel`` when closing a project, ``Switch
  host`` / ``Cancel`` for a host-structure mismatch, ``Change functional`` /
  ``Cancel`` when adopting a result converged with another functional, and
  ``Trust host key`` / ``Cancel`` for an unknown SSH host key.

The Job Dashboard
-----------------

.. figure:: ../_generated/screenshots/job_dashboard.webp
   :alt: The main window with the Job Dashboard docked below the Cutoff
         Convergence page, listing a failed, a stopped and a cancelled job.

   Failed, stopped and cancelled cutoff jobs of the CoF₂ example on the ``Job
   Dashboard``.

The ``Job Dashboard`` lists the jobs that still need attention: active jobs,
failed jobs, stopped jobs and cancelled jobs. Completed jobs leave it after
their results are handled, and a failed, stopped or cancelled job that you
``Dismiss`` is hidden without losing its records. Rows are listed in the order
their jobs were created, and that order survives closing and reopening the
project: resubmitting, resetting or continuing a job does not move its row.

Each row lists the job's ``Label``, ``Workflow``, ``Calc Type``, ``Status``,
``Progress``, ``Last Checked``, ``Cluster`` (``local`` for local jobs),
``Remote Job ID``, ``Runtime`` and ``Processes``, followed by per-row
``Resubmit``, ``Retry retrieval``, ``Cancel``, ``Inspect``, ``Reset`` and
``Dismiss`` buttons that are enabled only when they apply. The dashboard has no
filter controls; select rows to act on them, as in `Working with selected
rows`_.

Job states
~~~~~~~~~~

The ``Status`` column shows where each job is:

``defined``
   The job has been created but has not been submitted yet. Submit it from the
   workflow table that created it (see `Submitting jobs`_).

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
   A cluster job has finished on the scheduler and MuFinder is copying and
   parsing output files.

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
   kill. Local jobs that were still active when MuFinder closed, and jobs
   MuFinder lost track of, also end up here. Stopped jobs remain in the
   dashboard and can be inspected, resubmitted, or reset.

``cancelled``
   You cancelled a queued or running job. Cancelled jobs remain in the
   dashboard and can be resubmitted or reset.

Acting on jobs
~~~~~~~~~~~~~~

To act on one or more jobs, click a row button or select rows and right-click.
The right-click menu offers ``Resubmit``, ``Retry Retrieval``, ``Cancel``,
``Inspect``, ``Reset``, ``Dismiss`` and ``Delete…``. Each entry is enabled
when it applies to at least one selected job, acts on those jobs only, as the
row buttons do, and carries the count when it is partial (``Reset (2 of 3)``).
The row buttons remain the primary controls.

``Inspect``
   Shows why a failed or stopped job failed (see `When a job fails`_). On a
   running relaxation it opens the progress dialog (see `Watching a
   relaxation`_).

``Resubmit``
   Available for failed, stopped and cancelled jobs. Relaxations ask first
   whether to continue from their checkpoint or start again (see
   :ref:`resubmit-continue-or-start-again`). Cluster resubmission reuses
   existing remote files when the job still records a remote directory, so
   continuation runs can start from the previous working directory. The
   scheduler submit script is regenerated on every submission, so an updated
   walltime, core count, or memory request takes effect on resubmission.

``Retry retrieval``
   Failed or stopped cluster jobs that kept their remote metadata can fetch
   their output again instead of being resubmitted.

``Cancel``
   Cancels a queued or running job after asking for confirmation.

``Reset``
   Returns a failed, stopped or cancelled job to ``defined``, with an option to
   clean its output folder. See `Reset and its clean-up`_.

``Dismiss``
   Hides a failed, stopped or cancelled job from the dashboard, after asking
   for confirmation. Its records are kept.

``Delete…``
   Pressing Delete or Backspace with dashboard rows selected, or choosing
   ``Delete…``, applies the action that fits each job's state: active jobs
   (``submitting``, ``queued``, ``running``, ``retrieving``) are cancelled,
   failed, stopped and cancelled jobs are reset, and any other job has its
   records deleted. One summary dialog lists those actions with their counts
   before anything happens and offers ``Also clean the output directories of the
   jobs being reset``; cancelling it leaves every selected job untouched.
   Imported jobs cannot be reset and are reported as left unchanged.

A ``Cancel`` or ``Dismiss`` button on a single row asks for confirmation; on a
multi-row selection the "all selected jobs / this job only / cancel" prompt
serves as that confirmation.

Reset and its clean-up
~~~~~~~~~~~~~~~~~~~~~~

``Reset`` asks ``Clean and reset``, ``Reset, keep files`` (the default) or
``Cancel``. The clean-up deletes everything in a job's output folder
except its ``.cell``/``.param`` inputs, and applies the same ownership check as
:ref:`starting a relaxation again <resubmit-continue-or-start-again>`, limited
to the local folder (Reset never touches the cluster). The folder is kept, and
the job reset without cleaning, when it is outside ``<project>/files/<stage>/``,
is not named after the job, or is used by another job that is not being reset
in the same action. A Quantum Corrections phonon job is the one exception: the
singlepoint an older project ran in the same seed directory does not keep the
folder, so cleaning it lets the row rerun as a fresh phonon job. The kept
folders are listed in the "Directory cleanup issues" message.

The pristine- and muon-relaxation tables additionally offer a per-row
``Continue relaxation…`` action for ``completed``, ``failed``, ``stopped``, or
``cancelled`` rows: it edits that row's parameters in place and resumes the
same job from its CASTEP checkpoint rather than starting over, so it is the
recovery path that keeps a run directory's ``.check``/``.castep`` files intact
(unlike ``Reset``, whose optional clean-up deletes them). See
:ref:`continue-relaxation`.

Watching a relaxation
---------------------

.. _geometry-optimisation-progress:

Geometry-optimisation progress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/geometry_progress_dialog.webp
   :alt: The geometry progress window of a running BFGS relaxation at iteration 3,
         with its three criteria tabulated and plotted against their tolerances.

   A running pristine relaxation of CoF₂ (:doc:`/user_guide/examples/cof2`), read from a
   synthetic CASTEP output.

A running pristine or muon relaxation shows how far its geometry optimisation
has got in the narrow ``Progress`` column, as ``iteration · criteria met``: for
example ``14 · 3/4`` means CASTEP has finished iteration 14 and three of the
four convergence criteria it tests (``dE/ion``, ``|F|max``, ``|dR|max``,
``Smax``) are within tolerance. A run that fixes every ion or the cell tests
fewer criteria, so its count reads out of three or two. ``starting`` means the
run has echoed its parameters but not finished a step. The cell is empty for
every other job and status.

To see more detail:

1. Hover the ``Progress`` cell. The tooltip lists each criterion's value
   against its tolerance, the step limit (``GEOM_MAX_ITER``) and, after a
   continuation, which restart the figures belong to.
2. Select the running relaxation and click ``Inspect``. The progress dialog
   (``Geometry Progress - <label>``) shows the same table and a plot of each
   criterion over the iterations against its tolerance.
3. Leave the dialog open. It updates as new steps arrive, and its ``Refresh``
   button reads the output at once.

The progress comes from the job's growing ``.castep`` output
(``<seed>_farm001.castep`` when the run is task-farmed). Only the most recent
restart is read, because CASTEP counts iterations from 0 again after a
continuation. An iteration counts once its table is complete, so a file read
mid-write never shows a partial count. MuFinder reads the output as soon as a
relaxation is seen running, then every 60 seconds while the dashboard is
visible, and when a progress dialog opens or its ``Refresh`` is pressed (at most
once every 10 seconds per job). Each read fetches only what CASTEP has appended
since the last one, and neither a local nor a cluster read blocks the window. A
read that fails keeps the last progress shown. Progress is never saved in the
project.

When a job fails
----------------

.. _job-failure-reasons:

Why a job failed
~~~~~~~~~~~~~~~~

A failed or stopped row names its cause in the ``Status`` column, for example
``stopped · walltime`` or ``failed · memory``. Hovering the cell shows a
one-sentence explanation followed by the recorded failure message; a failure
with no recognised cause shows ``failed`` and its message, and points to
``Inspect``.

To read the diagnosis, select the job and click ``Inspect`` (``Inspect
Failure`` in a workflow table's right-click menu). The ``Failed job`` dialog
opens with a headline saying why the job did not finish:

1. the recognised cause, from the table below, when there is one;
2. otherwise the message the local runner or the scheduler recorded, for
   example ``CASTEP exited with status 1: SCF cycles did not converge``;
3. otherwise ``No cause found in the logs.``

Under the headline are the recorded message (when it is not the headline), the
log line that shows the cause and the file it came from (``From
slurm-4242.out: …``), and a note when the job's output folder no longer exists,
in which case the recorded message is all there is. ``Log excerpt`` expands the
context around the failure lines of each error file in the output folder
(``<seed>.err``, ``<seed>.NNNN.err`` and scheduler ``slurm-*.out`` files).

The dialog's buttons act on the job directly:

- ``Resubmit…`` runs the dashboard's ``Resubmit``, including its
  continue-or-start-again question for relaxations (see
  :ref:`resubmit-continue-or-start-again`).
- ``Continue relaxation…`` (a pristine or muon relaxation whose row can be
  continued) opens the row's ``Continue relaxation…`` dialog, to change its
  parameters before resuming.
- ``Open Folder`` (when the output folder exists) shows the folder in the file
  manager and keeps the dialog open.
- ``Close`` does nothing further.

With several jobs selected, the dialog is ``Failed jobs``: each headline is
followed by one line per job that has it, ``Log excerpt`` holds each job's
notes and excerpts under its name, and ``Resubmit…`` resubmits them all. ``Continue relaxation…`` and
``Open Folder`` act on one job, so they are offered only when one job is
selected.

The cause is recorded when the job fails or stops, from the failure message and
the job's logs in its local output directory: the scheduler's ``*-<id>.out``
and CASTEP's ``<seed>.err`` / ``<seed>[_farmNNN].NNNN.err``, which the cluster
runner downloads before reporting the failure. The causes, highest priority
first, are:

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
     - MuFinder was closed while a local job ran, or MuFinder lost track of a
       running job.
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
logs.

MuFinder never resubmits a failed job by itself: the cause tells you whether to
raise the walltime or memory request, continue from the checkpoint, or fix the
input. A ``walltime`` row's tooltip ends with that next step: ``Resubmit``
(continuing or starting again, for a relaxation), with a longer walltime if
needed.

.. _resubmit-continue-or-start-again:

Resubmitting a relaxation: continue or start again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every relaxation ``.param`` contains ``CONTINUATION : DEFAULT``, so a relaxation
resubmitted into its own run directory resumes from the ``{seed}.check`` file
the previous run left there, and starts from its initial structure when there
is none. When ``Resubmit`` includes pristine or muon relaxations, MuFinder
therefore asks once for the whole selection, in ``Resubmit relaxations``,
before the submission dialog.

Before that question, ``Resubmit`` checks what the submission needs. Local jobs
need a CASTEP executable: without one, ``CASTEP executable missing`` offers
``Open Setup > CASTEP…``, and the resubmission goes on once an executable is
set there. Cluster jobs need a configured cluster: without one, ``No configured
clusters`` offers ``Open Setup > HPC Clusters…`` in the same way. Cancelling
either leaves those jobs out and resubmits the rest. The keyword-catalog
warning (see :doc:`/castep_keyword_catalog`) also comes before the question;
cancelling it resubmits nothing.
The same prompts guard every local and cluster submission. Then:

1. Read the ``Causes`` line, which summarises the recorded causes.
2. Choose ``Continue from the checkpoint`` or ``Start again from the initial
   structure``.
3. Click ``Resubmit…`` and complete the submission dialog.

``Continue from the checkpoint``
   Resubmits in place, keeping the previous run's output. The dialog says how
   many of the relaxations have a checkpoint, have none (and will start from
   their initial structure), or could not be checked (a cluster job while its
   cluster is not connected).

``Start again from the initial structure``
   Deletes the previous run's output, checkpoint included, then runs the job's
   inputs from the start. On the job's own cluster the remote directory is
   deleted and the ``.cell``/``.param`` uploaded afresh, with no overwrite
   prompt; locally the output directory is cleared exactly as Reset's clean-up
   clears it (everything but the ``.cell``/``.param`` inputs, including
   ``{seed}-out.cell``). The deletion happens only once the submission dialog
   is accepted. A job whose local output cannot be cleared is not submitted,
   since it would resume from a checkpoint you asked to discard. While ``Start
   again`` is selected, Enter cancels the dialog rather than resubmitting.

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
  a job MuFinder lost track of is marked ``stopped · orphaned`` without being
  cancelled. Before deleting a remote directory, MuFinder asks the scheduler
  about the job's previous scheduler id and refuses (the submission fails,
  nothing is deleted) while it is still queued or running, or when the
  scheduler cannot be asked.
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
UEP, Energy Barriers, Quantum Corrections) run again from the start, and
imported relaxations are never offered a fresh start. ``Continue relaxation…``
in a relaxation table remains the way to change a row's parameters before
resuming.

Running on a cluster
--------------------

Cluster profiles and connections
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/cluster_setup_dialog.webp
   :alt: The Remote Cluster Connections dialog with the built-in SCARF profile
         selected on the Connection tab.

   ``Setup > HPC Clusters`` with the built-in SCARF profile selected.

To add or edit a cluster, choose ``Setup > HPC Clusters…``. Profiles are
stored globally, including built-in SCARF and Redwood presets.

1. Select a cluster in ``Saved clusters``, or click ``Add`` for a new one.
   ``Duplicate`` copies the selected cluster and ``Remove`` (or Delete or
   Backspace in the cluster list) deletes it after asking. Cluster names must
   differ by more than letter case.
2. Fill in the three tabs: ``Connection``, ``Scheduler`` and ``Queues``. Edits
   are kept per cluster while you switch between clusters.
3. Click ``Test Connection`` to check the settings.
4. Click ``OK`` to save all clusters together.

- A profile sets the ``Host name``, ``Username``, ``Queue system``, ``Default
  remote working directory``, ``Cores per node``, ``Default walltime`` and
  ``CASTEP executable`` path, plus a ``Template script`` for SLURM profiles and
  ``Default memory per core (GB)`` for ``addqueue`` profiles. Walltimes use
  SLURM's formats (``HH:MM:SS``, ``D-HH:MM:SS`` and the shorter forms) and are
  checked on save.
- The built-in SCARF and Redwood profiles ship without usernames: the empty
  ``Username`` field asks for yours (``your SCARF username``). A profile can be
  saved without one, but submission refuses a cluster that has no username, or
  no jump host username when it uses a jump host, and its message offers
  ``Open Setup > HPC Clusters…``, which opens that profile. A username typed in
  the submission dialog applies to that run.
- ``{username}`` in ``Default remote working directory`` stands for the
  username, so the SCARF profile's ``/work4/isis/{username}`` becomes
  ``/work4/isis/alice`` for the user ``alice``. The submission dialog leaves the
  directory empty until there is a username.
- ``Template script`` names the job script a SLURM profile uses: ``Built-in
  SCARF template``, ``Built-in SLURM template`` or the name of your file. ``Use
  custom file…`` picks a file and ``Use built-in`` goes back to the built-in
  template.
- ``Generate keywords using`` and ``Keyword shell init (optional)`` on the
  ``Scheduler`` tab set how MuFinder runs CASTEP on this cluster to build the
  keyword catalog (see :doc:`/getting_started/castep_setup`).
- A profile lists its queues (SLURM partitions or ``addqueue`` queues); see
  `Cluster queues`_.
- A SLURM profile may tick ``Request default memory per CPU (GB)``, which
  pre-seeds the submission dialog. It is off by default, so a profile without
  it generates batch scripts with no memory directive (see
  :ref:`slurm-memory-requests`).
- For clusters behind a gateway, tick ``Connect through an SSH jump host`` in
  the ``Jump host`` group. Its fields appear when the box is ticked: a separate
  host name, port, username, authentication method and SSH key path.
- The ``Scheduler commands`` group sets explicit paths for the submit, queue
  check and cancel commands. A command without a configured path is located by
  probing the non-interactive SSH shell.
- SSH key authentication is primary, with password fallback
  (``Authentication method``: ``SSH key`` or ``Password``).
- The Redwood preset targets ``glamdring.physics.ox.ac.uk`` through the Physics
  gateway ``bastion.physics.ox.ac.uk``. MuFinder prompts separately for the
  Physics/bastion and Glamdring credentials because those accounts can differ.
  The preset submits with ``/usr/local/shared/bin/addqueue``; a saved profile
  keeps whatever submit command path it was saved with.

Passwords and the system keyring
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On macOS and Linux, eligible Redwood/Glamdring password-auth profiles can tick
``Use macOS Keychain passwords`` (``Use system keyring passwords`` on Linux) to
use read-only system keyring lookup, so neither password is prompted for.
MuFinder reads existing entries and never stores passwords.

- On macOS the entries live in the login Keychain and use the same services as
  ``glamdring-tools``: ``security add-generic-password -a <physics_username> -s
  bastion -w`` and ``security add-generic-password -a <glamdring_username> -s
  glamdring -w``.
- On Linux they live in the Secret Service keyring (GNOME Keyring or KWallet),
  are read with ``/usr/bin/secret-tool`` (Ubuntu package ``libsecret-tools``)
  and are created with ``secret-tool store --label=<label> service bastion
  account <physics_username>`` and ``secret-tool store --label=<label> service
  glamdring account <glamdring_username>``; the keyring must be unlocked in the
  desktop session.
- The ``?`` button beside the checkbox shows the exact commands for the current
  platform.
- The lookup never freezes the window. If it takes noticeably long, a busy
  dialog appears whose ``Cancel`` button (or Escape) abandons the lookup
  without connecting. A lookup that fails, finds an empty password or does not
  finish within 120 seconds stops the connection and lists the missing entries
  with their setup commands.

Host keys and connection checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- Unknown SSH host keys are never trusted automatically. MuFinder shows the
  key's SHA-256 fingerprint and asks before trusting it, for both the jump host
  and the target cluster host, and stores each trusted key in your settings;
  bastion trust records are stored separately from Glamdring trust records. A
  host key that differs from the trusted one stops the connection and shows the
  stored and presented fingerprints.
- Connection checks (``Test Connection`` and the pre-flight check before
  cluster keyword generation) run behind a cancellable progress dialog, so an
  unreachable or slow host never freezes the window. Unknown-host-key prompts
  appear in front of you and the check retries automatically once the key is
  trusted.
- Jump-host submissions show separate credential fields for the gateway and
  the target cluster. With the system keyring option enabled, submission and
  reconnect use the keyring entries instead of password prompts.
- Reconnecting reuses the password or passphrase cached from the last
  connection only while the profile keeps the same name, host, port and
  username for both the target and the jump host, and the jump host stays
  enabled or disabled; after any of them changes, the cached secret is never
  offered to the new endpoint.

Cluster queues
~~~~~~~~~~~~~~

A cluster with several queues is one profile, not one profile per queue. On the
``Queues`` tab:

1. Click ``Add Queue`` and type a ``Name``. Select a queue and click ``Remove
   Queue`` (or press Delete or Backspace) to delete it; MuFinder asks first.
2. Optionally override the profile's values for that queue in the cells
   ``Default walltime``, ``Max walltime``, ``Cores per node``, memory
   (``Memory per CPU (GB)`` for SLURM, ``Memory per core (GB)`` for
   ``addqueue``), ``CASTEP executable`` and, for SLURM, ``Template script``. A
   blank cell uses the profile's value from the ``Scheduler`` tab; the maximum
   walltime has no profile value, and a queue without one has no limit that
   MuFinder checks.
3. Choose the ``Default queue``. Renaming the default queue keeps it the
   default.

Saving refuses a queue without a name, a repeated name, a name that would need
shell quoting (letters, digits, ``.``, ``_`` and ``-``, with commas between
several SLURM partitions), an unreadable walltime, a default walltime above the
maximum (including one inherited from the profile), and a cores-per-node or
memory value that is not positive. The dialog also refuses to switch to another
cluster until the table is valid, so no cell is dropped unseen.

The submission dialog applies the chosen queue's template, cores per node and
executable to every job of a submission, and keyword generation runs on the
default queue with its overrides. A profile saved with a single default queue
name shows that queue as one row with no overrides (see
:doc:`/compatibility_policy`).

Monitoring and retrieval
~~~~~~~~~~~~~~~~~~~~~~~~

- The dashboard shows each job's execution target (``Cluster``), remote job
  id, a live ``Runtime`` and the process count used at submission.
- While the open project has cluster jobs that are active (including one whose
  output is being retrieved) or that MuFinder is watching for the scheduler to
  list again, the status bar shows the connection state, when the queue was last
  checked (``Queue last checked``) and a ``Refresh Queue`` button. With no
  project, or with only local or finished jobs, these three are hidden.
- The connection state names the cluster. ``Not connected to SCARF`` in grey is
  the ordinary idle state, for example after reopening a project.
  ``Connected to SCARF`` is green. ``Connection to SCARF lost`` is red and means
  a connection dropped while jobs on that cluster are active: their status is
  not being updated until you reconnect.
- ``Refresh Queue`` checks the queue now. When no connection is open to a cluster
  with such jobs it reconnects first. The status bar reports the result
  (``Refreshed queue: checked 2 jobs at 12:34:56``, or that the check failed). If
  a job's connection has died unnoticed, the first refresh reports it as lost and
  the next one reconnects. The button is
  disabled, with a tooltip saying why, when no job has a remote job id to poll,
  such as while the only cluster job is still being submitted.
- The queue is polled periodically, updating each job's ``Last Checked`` time.
  SLURM polling tolerates stale or expired job ids (``Invalid job id
  specified``) by falling back to per-job status checks instead of failing the
  whole refresh.
- Connection, queue refresh, submission, cancellation and polling failures show
  a one-line summary, with the full error and recovery advice under
  ``Details``.
- When a cluster job finishes, MuFinder copies the output back (``retrieving``).
  A failed or stopped job that kept its remote metadata can use ``Retry
  retrieval`` instead of being resubmitted.

Outputs
~~~~~~~

After a run, MuFinder parses the ``.castep`` output and saves the parsed data
as the job's result. The parser extracts:

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
