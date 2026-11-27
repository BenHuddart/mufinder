Running and Monitoring Jobs
===========================

Every workflow turns its calculations into jobs. You start them from the job
table on a workflow page, on this computer or on an HPC cluster, and follow
them in the ``Job Dashboard``, the tray at the bottom of the main
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
   Selected Locally`` and ``Run Selected on Cluster``. A row is ready when it
   has not run yet (it reads ``ready``), or its job ``failed``, ``stopped`` or
   was ``cancelled``, whether or not you dismissed it from the Job Dashboard.
   Running a row whose job ended may first ask, once for all of them, whether
   to continue from the previous run or start again (see
   :ref:`resubmit-continue-or-start-again`); ``Cancel`` there submits nothing.
4. Fill in the submission dialog (see `Local submission`_ and
   `Cluster submission`_) and click ``OK``.

The same actions are in a row's right-click menu, which opens the same menu in
every workflow table, in this order: ``Run Local``, ``Run on Cluster``,
``Cancel``, ``Inspect Failure``, ``Configure…``, ``Remove``, then any
table-specific entries below a separator (``Continue relaxation…`` and
``Export Relaxed Structure`` in the relaxation tables), and last, below another
separator, ``Open Containing Folder`` and ``Export Inputs/Outputs…``. Quantum
Corrections and Hyperfine Coupling have no ``Remove``. Right-clicking a row
outside the selection selects it first (see :ref:`workflow-row-menu`). A row
whose job ended is run again with ``Run Local`` or ``Run on Cluster``, edited
with ``Configure…`` and removed with ``Remove``, as any other row.

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
Selected`` because its rows are derived from completed relaxations.

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
   For an ``addqueue`` cluster, ``Node placement`` sets how the cores spread
   across compute nodes (see :ref:`addqueue-node-placement`).
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
  ``Memory per core (GB)`` and ``Submission note``, and the node placement
  described below.
- Cluster submission refuses job labels containing anything other than letters,
  digits, ``.``, ``_`` and ``-``, because remote paths and scheduler commands
  use the label directly. Remote job directories are deleted recursively over
  SFTP, never through a remote shell command.
- When a matching cluster connection is already active, submission reuses it
  and does not ask for a password or passphrase again.

.. _addqueue-node-placement:

Node placement on addqueue clusters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Node placement`` appears below ``Processor cores per job`` for ``addqueue``
clusters only. It starts on ``Scheduler decides``, which sends the total alone
(``-n 128``) and leaves the spread across nodes to the scheduler.

The other entries split the total equally, one entry per number of processes
per node that divides the total exactly, largest first: for 128 cores,
``2 nodes × 64 per node`` sends ``-n 2x64`` and ``4 nodes × 32 per node`` sends
``-n 4x32``. The line under the list shows the ``-n`` value that will be sent.
Only splits the cluster's nodes can hold are listed. Redwood has 19 nodes that
run up to 32 processes and 28 that run up to 64, so a split of more than 32 per
node can use at most 28 nodes, one of 32 or fewer at most 47, and none can
exceed 64 per node. A cluster with no recorded node list offers every exact split.
Requesting 32 or fewer per node does not choose the node type; the scheduler
may still place them on larger nodes.

Changing ``Processor cores per job`` rebuilds the list and keeps the chosen
processes per node. If that number no longer fits the new total, the entry is
shown as ``(not possible)``, the line under it says why, and ``OK`` stays
disabled until another split is picked or the total fits again. MuFinder never
rounds the request. The choice is remembered with the other last-used
submission settings; a project saved before the option existed starts on
``Scheduler decides``. SLURM submissions are unchanged.

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
  row button (``Cancel``, ``Resubmit…``, ``Inspect…``, ``Check now`` and ``✕``).
- Controls that only make sense for one row (``Open Wizard`` in the UEP table)
  always act on their own row and never prompt.

What a click will affect is shown before the click, without changing any text
inside the rows, so tables stay usable on a small laptop screen:

- While two or more rows are selected, the row controls that will act on all of
  them are drawn with a heavier border. Single-row controls are not.
- Each workflow table shows ``N selected`` beside its buttons. The Job
  Dashboard shows ``N selected · row actions apply to all selected`` in its
  header strip, only while two or more rows are selected.
- After a multi-row ``Supercell`` or ``Fix cell`` change the status bar reports
  how many rows were changed and how many were skipped.

A selection may mix rows an action applies to with rows it does not (a
finished row when cancelling, a completed row when running). Every action then acts on
the rows it applies to and says how many it left out:

- Run, Cancel and Inspect in the row menu are offered when at least one
  selected row qualifies, act on those rows only, and carry the count when it
  is partial, for example ``Cancel (2 of 5)``. An entry is greyed
  out only when no selected row qualifies. A row with no label yet has no job
  and counts as not qualifying.
- The run buttons and the menu's run entries submit the rows that are ready and
  the status bar reports the rest: ``Submitted 3 local job(s); skipped 2
  selected rows that were not ready``.
- The three-way prompt of a row control states the same split (``Set this
  supercell on 3 of 5 selected rows? 2 are not eligible and will be
  skipped.``).

Actions that remove, cancel, dismiss or delete are always confirmed,
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
  before acting, as does Delete or Backspace on the dashboard. Running a job
  again asks before it deletes a previous run's output (see
  :ref:`resubmit-continue-or-start-again`).
- Deleting a saved supercell, a comparison table, a magnetic field workspace
  or an analysis, and removing imported relaxations, are confirmed the same way.
- In every confirmation the safe choice is the default button and the one
  Escape selects, and the buttons are labelled with the action
  (``Delete supercell`` / ``Cancel``, ``Remove 3 rows``, ``Keep running``)
  rather than Yes/No. In the three-way row-control prompt the default button
  is the all-selected choice.
- Questions that change nothing destructive also name their answers:
  ``Save`` / ``Don't save`` / ``Cancel`` when closing a project whose layouts or
  viewer choices changed since the last save, ``Switch
  host`` / ``Cancel`` for a host-structure mismatch, ``Change functional`` /
  ``Cancel`` when adopting a result converged with another functional, and
  ``Trust host key`` / ``Cancel`` for an unknown SSH host key.
- If a save fails while closing, for example on a read-only folder or a full
  disk, the reason is shown and the window stays open, whether or not closing
  asked first. The next close asks, so ``Don't save`` is always a way out. The
  file that could not be written keeps its earlier content. The exception is a
  project with only an upgrade or tidying from opening it to write: that closes
  without a message, and opening it next time does the same work again.

The Job Dashboard
-----------------

.. figure:: ../_generated/screenshots/job_dashboard.webp
   :alt: The main window with the Job Dashboard tray below the Cutoff
         Convergence page, listing a failed, a stopped and a cancelled job.

   Failed, stopped and cancelled cutoff jobs of the CoF₂ example on the ``Job
   Dashboard``.

The ``Job Dashboard`` lists the jobs that still need attention: active jobs,
failed jobs, stopped jobs and cancelled jobs. Completed jobs leave it after
their results are handled, and a failed, stopped or cancelled job that you
``Dismiss`` is hidden without losing its records. Tick ``Show dismissed`` in the
header strip to list the dismissed jobs as well (it starts unticked in each project
you open): their labels are greyed, the
tooltip says when each was dismissed, and their actions work as on any other
row. Rows are listed in the order
their jobs were created, and that order survives closing and reopening the
project: resubmitting or continuing a job does not move its row.

The tray's header strip
~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/job_dashboard_collapsed.webp
   :alt: The main window with the Job Dashboard collapsed to its header strip,
         which reads 1 failed, 1 stopped and 1 cancelled.

   The same jobs with the ``Job Dashboard`` collapsed to its header strip.

The strip along the top of the tray stays in view while the tray is shown. From
left to right it holds:

- the ``Job Dashboard`` button, which collapses the tray to the strip and expands
  it again. Ctrl+J (Cmd+J on macOS) does the same from anywhere in the window;
  expanding with the key moves focus into the table, and collapsing with focus in
  the table moves it to the button;
- a count of the listed jobs in each state, each with its state's glyph and word
  (``4 running``, ``12 queued``, ``2 failed``), in the order running, queued,
  submitting, retrieving, failed, stopped, cancelled. A state with no jobs is
  left out, and an empty dashboard reads ``No jobs``. The counts cover the jobs
  the dashboard lists, so dismissed jobs count only while ``Show dismissed`` is
  ticked. Each count is a toggle that filters the table by its state (see
  `Filtering by state`_);
- ``Show all``, while any count is ticked;
- the selection hint, while two or more rows are selected;
- ``Show dismissed``, which works whether the tray is collapsed or expanded.

Expanded, the tray is as tall as its rows need, and at least one row tall. It
takes at most 70 % of the height it shares with the page above it; beyond that
the table scrolls. It grows and shrinks as jobs join and leave the dashboard,
and follows the window as it is resized. Dragging the tray's top edge sets a
height of your own, kept until you next collapse or expand the tray.

When the rows need more room than the page above can give up, the page keeps
its size and its lower part is hidden behind the tray. The page does not
scroll: collapse the tray to see the rest of it. Collapsing the tray, or the rows
needing less room, shows the whole page again. Moving keyboard focus with Tab to
a control the tray hides collapses the tray, and a tray that comes to cover the
focused control takes focus to its ``Job Dashboard`` button, so focus is never out of sight. The window never changes size for the tray,
and its smallest size is the same collapsed or expanded. Double-clicking a row,
or ``Open in`` its stage, collapses the tray only while it hides part of the page.

Collapsed, the tray is the strip alone and cannot be dragged taller. Its rows
are out of sight, so the selection is cleared and Delete and Backspace do
nothing; the runtimes stop ticking until the tray is expanded, while the counts
keep up. MuFinder remembers whether the tray was collapsed, across projects and
restarts.

The tray always sits at the bottom of the window; it cannot be moved or float
on its own. ``View > Job Dashboard`` shows and hides it.

Filtering by state
~~~~~~~~~~~~~~~~~~

Click a count in the strip to list only the jobs in that state: ``2 failed``
lists the two failed jobs. Click more counts to list several states together;
the table lists the jobs in any ticked state. A ticked count has a border and a
filled background. Click it again to untick it, or click ``Show all``, which
shows after the counts while any is ticked, to untick them all and list every
job.

- The counts always cover every job the dashboard lists, ticked or not, so the
  strip still reads the whole dashboard while the table shows part of it.
- A ticked state whose last job leaves the dashboard stays in the strip as a
  ticked ``0 failed`` over an empty table, until you untick it or click ``Show
  all``. An unticked state with no jobs is left out.
- ``Show dismissed`` decides which jobs are listed, and the ticked counts narrow
  that list by state. Ticking or unticking ``Show dismissed`` keeps the ticked
  counts: with ``failed`` ticked, it adds or removes the dismissed failed jobs.
- A row the filter hides leaves the selection; the rows still shown stay
  selected, and a hidden row comes back unselected. Row buttons, the context
  menu and Delete and Backspace act only on the rows shown.
- The expanded tray fits the rows shown, and at least one row.
- Clicking a count while the tray is collapsed ticks it and expands the tray.
  Unticking a count or ``Show all`` leaves the tray as it is, and collapsing or
  expanding the tray keeps the ticked counts.
- The filter is not kept: every project opens with no count ticked, and so does
  MuFinder.

The table
~~~~~~~~~

Each row has these columns:

``Job``
   The job's label.

``Stage``
   The navigator's name for the page the job belongs to, such as ``Cutoff
   Energy``, ``Relaxations`` or ``Quantum Corrections``.

``Status``
   The job's state with its reason or progress beside it:
   ``failed · SCF did not converge``, ``stopped · walltime``, ``running · step
   12 of 100``. Hold the pointer over the cell for the full explanation and for
   when and how the status was last checked. A small clock on the row means the
   status of an active cluster job could not be confirmed; the tooltip says
   why. The cluster is not connected, the job has not been checked for a
   while, no scheduler job ID was recorded for it, or MuFinder lost track of
   it: the scheduler no longer lists the job and its folder on the cluster
   holds no output from this run. Such a job stays active, since it may still
   be queued or running, and MuFinder never ends it on its own. A job
   MuFinder lost track of is checked again every minute and completes as soon
   as its output appears. A job with no scheduler job ID is looked at only
   when you press ``Refresh Queue``, which completes it when the output in its
   folder is complete. ``Cancel`` ends either in the project.

``Where``
   ``local``, or the cluster and its scheduler job id (``SCARF #4127731``).

``Runtime``
   How long the job has run.

``Cores``
   The processes it runs on.

``Action``
   Every action that applies to the job, each in its own place (see `Acting
   on jobs`_). Return, Enter or Space press them; see `Using the dashboard
   from the keyboard`_ for how to reach them.

Double-click a row to show its stage's page with the job's row selected. The
counts in the header strip filter the table by state (see `Filtering by
state`_); select rows to act on them, as in `Working with selected rows`_.

Job states
~~~~~~~~~~

The ``Status`` column shows where each job is:

``ready``
   Shown in a workflow table for a row that has not run: it has no job yet, or
   its job has been created but not submitted. Submit it from that
   table (see `Submitting jobs`_). The Job Dashboard does not list these.

``submitting``
   MuFinder is handing the job to a local process or a cluster scheduler.

``queued``
   The job has been accepted and is waiting to run, including a cluster job
   SLURM holds, requeues, or is still configuring before it starts.

``running``
   CASTEP is running locally or on the selected cluster. When a queued cluster
   job starts, the status bar reports *"Cluster job started running: <label>"*
   once the check that saw it ends. A check that changes several jobs reports
   them together, for example *"Checked 5 cluster jobs: 1 completed, 1 stopped
   at its walltime limit"*.
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
   ``Inspect…``. They can be run again from their row or with ``Resubmit…``. A cluster job fails when
   the scheduler reports that it failed, or when the scheduler no longer runs
   it and its CASTEP output is incomplete, unreadable or missing. A job the
   scheduler reports as failed still completes when its CASTEP output is
   complete.

``stopped``
   A cluster calculation reached its walltime limit before producing a
   successful CASTEP completion marker: the scheduler reported ``TIMEOUT``, or
   the scheduler log downloaded with an incomplete output records a time-limit
   kill. A local job that was still active when MuFinder closed also ends up
   here, unless its CASTEP output is complete: then it is completed when the
   project is opened. When CASTEP outlived MuFinder and finishes later, the job
   is completed the next time the project is opened. Stopped jobs remain in the
   dashboard and can be inspected and resubmitted.

``cancelled``
   You cancelled a queued or running job. Cancelled jobs remain in the
   dashboard and can be resubmitted.

Failed, stopped and cancelled jobs are the jobs that did not finish. MuFinder
records why each one did not finish, and that reason decides which of the three
words its row shows.

Acting on jobs
~~~~~~~~~~~~~~

Each row's ``Action`` column shows every action that applies to the job, in
three slots that keep their places down the table: the first runs or stops the
job, the second looks at it, and the third dismisses it. A slot with nothing to
offer stays blank.

.. list-table::
   :header-rows: 1

   * - Job state
     - First slot
     - Second slot
     - Third slot
   * - ``submitting``, ``queued``, ``retrieving``
     - ``Cancel``
     -
     -
   * - ``running``
     - ``Cancel``
     - ``Inspect…`` on a relaxation that shows its progress
     -
   * - ``failed``, ``stopped``
     - ``Resubmit…``
     - ``Inspect…``
     - ``✕``
   * - ``failed`` or ``stopped`` on a cluster because its output could not be
       retrieved or its result recorded, or MuFinder lost track of it
     - ``Resubmit…``
     - ``Check now``
     - ``✕``
   * - ``cancelled``
     - ``Resubmit…``
     -
     - ``✕``
   * - ``cancelled`` on a cluster that was not told
     - ``Resubmit…``
     - ``Check now``
     - ``✕``

``Resubmit…`` is missing from a job that no workflow row holds any more, and
from an imported job. A dismissed job listed with ``Show dismissed`` keeps its
other buttons but has no ``✕``. Hold the pointer over ``✕`` to see what it
does.

Right-click a row for the same actions applied to the selection, followed by
``Open in <stage>``: each is enabled when it applies to at least one selected
job, acts on those jobs only, and carries the count when it is partial
(``Dismiss (2 of 3)``). The context-menu key and Shift+F10 open the same menu
for the current row.

``Cancel``
   Cancels an active job after asking for confirmation. A cluster job with
   no scheduler job ID, or whose cluster is not connected, is cancelled in the
   project only: the cluster is not contacted, and the status bar says so.
   Such a job may still be queued or running on the cluster, so its row keeps
   ``Check now``, its status tooltip says the cluster was not told, and
   ``Refresh Queue`` stays in the status bar. Once a check gets an answer the
   question is settled: a job the scheduler still lists returns to ``queued``
   or ``running``, where ``Cancel`` reaches the scheduler, and a job it no
   longer lists stays cancelled without ``Check now``. A job with no
   scheduler job ID cannot be asked about, so it keeps ``Check now`` until a
   check finds its complete output or you run it again. A job whose
   submission was cut off when MuFinder closed is treated the same way.

``Inspect…``
   Shows why a failed or stopped job failed (see `When a job fails`_). On a
   running relaxation it opens the progress dialog (see `Watching a
   relaxation`_).

``Check now``
   Offered in place of ``Inspect…`` on a failed or stopped cluster job whose
   output may still be on the cluster: it could not be retrieved, its result
   could not be recorded, or MuFinder lost track of the job. It is also
   offered on a cluster job cancelled in the project only, or whose
   submission was cut off when MuFinder closed, which may still be running
   there. The status tooltip says why the row offers it. MuFinder asks the scheduler about the job and, when
   the scheduler no longer runs it, fetches its output. When the job's cluster
   is not connected, it connects first, as ``Refresh Queue`` does, asking for
   your password if the cluster profile needs one; a cluster it cannot reach
   is named in a warning and the job is left as it was. The row changes when
   the answer arrives: a calculation that finished on the cluster is completed
   from the retrieved output, so its result is used without running it again,
   and a job the scheduler lists again returns to ``queued`` or ``running``. A
   check that finds nothing new leaves the job as it was. Active jobs are
   checked every minute, and ``Refresh Queue`` in the status bar checks every
   job on the cluster at once (see `Monitoring and retrieval`_).

``Resubmit…``
   Runs the job's workflow row again, as the row's ``Run`` does, on the job's
   own target: locally, or on its own cluster, which the submission dialog
   preselects. The inputs are rebuilt from the row, and when the job has a
   checkpoint MuFinder asks whether to continue from it or start again (see
   :ref:`resubmit-continue-or-start-again`). Available for failed, stopped and
   cancelled jobs whose row still holds them. The scheduler submit script is
   regenerated on every submission, so an updated walltime, core count, or
   memory request takes effect on resubmission.

``✕`` (Dismiss)
   Hides a failed, stopped or cancelled job from the dashboard, after asking
   for confirmation. Nothing else changes: its records are kept, its workflow
   row still reads ``failed``, ``stopped`` or ``cancelled``, and ``Show
   dismissed`` lists it again. The confirmation says when the job's output may
   still be on the cluster: dismissing does not retrieve it, so use ``Check
   now`` or ``Refresh Queue`` first to keep the result. A job that runs again,
   from its row, ``Resubmit…`` or a ``Check now`` or ``Refresh Queue`` that
   finds it, is listed again.

``Open in <stage>``
   In the right-click menu, and what a double-click on the row does: shows the
   stage's page with the job's row selected. When a job has no row on the page,
   the page still shows and the status bar says so. For a selection it needs
   the selected jobs to belong to one stage.

Pressing Delete or Backspace with dashboard rows selected cancels the active
jobs and dismisses the failed, stopped and cancelled ones, after one summary
that states both and names a lone job; refusing it leaves every selected job
untouched. The status bar then reports both. It never
runs a job again or deletes its records.

A ``Cancel`` or ``✕`` on a single row asks for confirmation; on a row inside
a multi-row selection the "all selected jobs / this job only / cancel" prompt
serves as that confirmation. Every row button acts this way on a selection.

Using the dashboard from the keyboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The table is one stop on the way round the window, however many rows it has. Tab
goes from the table to the current row's buttons, first slot to third, and then on
to the next control in the window; Shift+Tab comes back the same way. Tab skips a
blank slot. Shift+Tab from outside lands on the table, and Tab from there goes to
the row's buttons. Passing through leaves the selection as it is.

- With focus on the table, the arrow keys move the current row.
- With focus on a button, Up and Down move to the same button on the row above or
  below (or to that row's first button when its slot is blank), and Left and Right
  move between the row's buttons. Up and Down select a row outside the selection,
  and leave the selection alone on a row inside it.
- The context-menu key or Shift+F10 opens the right-click menu: for the row whose
  button has focus, or for the current row when the table has focus. A row outside
  the selection is selected first.
- Delete or Backspace cancels or dismisses the selected jobs, as described above.
- Ctrl+J (Cmd+J on macOS) collapses and expands the tray (see `The tray's header
  strip`_).
- In the header strip, Tab goes from the ``Job Dashboard`` button to each count
  from left to right, then ``Show all`` while it shows, ``Show dismissed`` and the
  table. Space ticks and unticks the count that has focus (see `Filtering by
  state`_). When the focused count or ``Show all`` leaves the strip, focus moves
  to the control beside it in the strip, the one to its right first, or to the
  ``Job Dashboard`` button when no count is left.

A job can leave the dashboard at any moment. Focus moves only if it was on that
job's row: it goes to the same button on the next row, or on the previous row if
that was the last, and to the table when no rows are left. Focus in a form
field or anywhere else stays where it is, and the selection does not change.

A job's own folder
~~~~~~~~~~~~~~~~~~

Every job writes in a folder of its own, under ``files/`` and named after its seed. A
new row whose folder another job already uses, an imported run's for example, gets
the next free name when it is submitted (``positivemuon_Site_1_2``), and its CASTEP
files are named to match. A row whose job has not run yet keeps that job when its
page's ``Run`` submits it again, so it stays in the folder it has and no second job is
made for it. A reconfigured row gets a new job
in its earlier job's folder. A folder that no job uses is written into as it stands.

An older project can hold two jobs in one folder. Opening it changes nothing, and
neither job's files or records change until you act. Finished jobs stay as they are,
and their results still read. A job waiting to run in such a folder is not started:
``Jobs not started`` names it and its folder under ``Details``, and the other jobs of
the submission go ahead. The page's ``Run``, the dashboard's ``Resubmit…`` and
``Continue relaxation…`` all refuse it this way, before anything is cleared.
Remove its row and add it again, and the new job gets a folder of its own. The folder stays,
because the other job uses it.

The pristine- and muon-relaxation tables additionally offer a per-row
``Continue relaxation…`` action for ``completed``, ``failed``, ``stopped``, or
``cancelled`` rows: it edits that row's parameters in place and resumes the
same job from its CASTEP checkpoint rather than starting over, so it is the
recovery path that keeps a run directory's ``.check``/``.castep`` files intact
(unlike ``Start again``, which deletes them). See
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
has got beside its status: ``running · step 14 of 100`` means CASTEP has finished
iteration 14 of the ``GEOM_MAX_ITER`` limit. ``running · starting`` means the run
has echoed its parameters but not finished a step.

To see more detail:

1. Hover the ``Status`` cell. The tooltip lists the convergence criteria CASTEP
   tests (``dE/ion``, ``|F|max``, ``|dR|max``, ``Smax``; a run that fixes every
   ion or the cell tests fewer) with each value against its tolerance and, after
   a continuation, which restart the figures belong to.
2. Click the row's ``Inspect…``. The progress dialog
   (``Geometry Progress - <label>``) shows the same table and a plot of each
   criterion over the iterations against its tolerance.
3. Leave the dialog open. It updates as new steps arrive, and its ``Refresh``
   button reads the output at once.

The progress comes from the job's growing ``.castep`` output
(``<seed>_farm001.castep`` when the run is task-farmed). Only the most recent
restart is read, because CASTEP counts iterations from 0 again after a
continuation. An iteration counts once its table is complete, so a file read
mid-write never shows a partial count. MuFinder reads the output as soon as a
relaxation is seen running, then every 60 seconds while the Job Dashboard is
shown (collapsed or expanded), and when a progress dialog opens or its ``Refresh`` is pressed (at most
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
``stopped · walltime`` or ``failed · memory``. A CASTEP error, or a failure with
no recognised cause, shows the first line of the recorded message instead
(``failed · SCF did not converge``). Hovering the cell shows a one-sentence
explanation followed by the recorded failure message.

To read the diagnosis, click the row's ``Inspect…`` (``Inspect Failure`` in a
workflow table's right-click menu). The ``Failed job`` dialog
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

- ``Resubmit…`` (when the job's workflow row still holds it) runs the
  dashboard's ``Resubmit…``: the row runs again with inputs rebuilt from it,
  after the continue-or-start-again question when the job has a checkpoint
  (see :ref:`resubmit-continue-or-start-again`).
- ``Continue relaxation…`` (a pristine or muon relaxation whose row can be
  continued) opens the row's ``Continue relaxation…`` dialog, to change its
  parameters before resuming.
- ``Open Folder`` (when the output folder exists) shows the folder in the file
  manager and keeps the dialog open.
- ``Close`` does nothing further.

With several jobs selected, the dialog is ``Failed jobs``: each headline is
followed by one line per job that has it, ``Log excerpt`` holds each job's
notes and excerpts under its name, and ``Resubmit…`` runs all their rows again. ``Continue relaxation…`` and
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
     - The output could not be parsed, or the scheduler reported the job
       finished and its folder holds no output from this run.
   * - ``lost track``
     - MuFinder was closed while a local job ran, or a local job's process
       ended without reporting. A cluster job with this cause may have finished
       on the cluster; ``Check now`` or ``Refresh Queue`` fetches its output.
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
recorded show ``failed`` alone; ``Inspect…`` still diagnoses them from their
logs.

MuFinder never resubmits a failed job by itself: the cause tells you whether to
raise the walltime or memory request, continue from the checkpoint, or fix the
input. A ``walltime`` row's tooltip ends with that next step: ``Resubmit…`` to
run its row again (continuing from its checkpoint or starting again, for a job
that can continue), with a longer walltime in the cluster submission dialog if
needed.

.. _resubmit-continue-or-start-again:

Running a job again: continue or start again
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A failed, stopped or cancelled job runs again from its workflow row: the row's
``Run`` (or ``Run All``), or the dashboard's ``Resubmit…``, which runs the same
row. The job's inputs are rebuilt from the row's current settings, so a row
whose job ended can be edited first, in its table or its ``Configure…``
dialog, and the edit takes effect on this run; a continued relaxation keeps the
project defaults its run used (see ``Continue`` below). The job keeps its
identity and its row.

Before that, ``Resubmit…`` checks what the submission needs. Local jobs need a
CASTEP executable: without one, ``CASTEP executable missing`` offers ``Open
Setup > CASTEP…``, and the resubmission goes on once an executable is set
there. Cluster jobs need a configured cluster: without one, ``No configured
clusters`` offers ``Open Setup > HPC Clusters…`` in the same way. Cancelling
either leaves those jobs out and resubmits the rest. The keyword-catalog
warning (see :doc:`/castep_keyword_catalog`) also comes first; cancelling it
resubmits nothing. The same prompts guard every local and cluster submission.

Relaxation inputs, and those of a Quantum Corrections phonon job, contain
``CONTINUATION : DEFAULT``, so such a job run in its own folder resumes from
the ``{seed}.check`` file its previous run left there. When at least one of
the jobs has a checkpoint, or its checkpoint could not be checked (a cluster
job while its cluster is not connected), MuFinder asks once, in ``Run again``:

``Continue`` (the default)
   Keeps the previous run's files, locally and on the cluster, and resumes
   from the checkpoint with the rebuilt inputs. A relaxation keeps the inputs
   its run used: the cut-off, k-point sampling and functional it took from the
   project's defaults stay as they were, and only settings stored on the row
   itself are taken from the row (see :ref:`continue-relaxation`); when its
   ``.cell`` or ``.param`` cannot be read it is not continued, and ``Relaxation
   not continued`` says so. A cluster job is submitted into
   its own remote folder, where its checkpoint is; the new ``.cell`` and
   ``.param`` are uploaded there, and the output of the previous run is never
   taken for this run's. A job whose checkpoint is missing starts from the
   beginning.

``Start again``
   Deletes the previous run's output, checkpoint included, and runs the job
   from the beginning. Its local output folder is cleared (see `What starting
   again clears`_). On its own cluster its remote
   folder is deleted and the inputs uploaded afresh, with no overwrite prompt,
   but only once the scheduler confirms the previous run has finished: a job
   the scheduler still lists as queued or running, or cannot be asked about,
   is not started again, keeps its folders and record, and is reported.

``Cancel``
   Submits nothing and changes no job.

The question says which jobs have a checkpoint and which could not be checked,
and how many other jobs in the selection start again whatever the answer.

Nothing is cleared or changed until the submission is accepted. The answer is
applied only after the submission dialog is accepted (and, for a cluster, the
cluster is connected and every overwrite question answered); cancelling the
submission dialog, or a submission refused before it starts, leaves every job
as it was: still failed, stopped or cancelled, with its files and its journal
unchanged.

With no checkpoint to continue from, the jobs start again without a question,
unless one of them may still have output on its cluster that was never
retrieved (it could not be fetched or its result recorded, MuFinder lost
track of it, or it was cancelled without the cluster being told). Starting
again does not retrieve it, so MuFinder asks first, with ``Cancel`` as the
default: ``Start again`` goes ahead; ``Cancel`` keeps the job so you can use
``Check now`` or ``Refresh Queue`` first. When the continue-or-start-again
question is asked anyway, the same warning is in it and ``Cancel`` is its
default, so there is only ever one question. A row's own ``Continue
relaxation…`` remains the way to change a relaxation's parameters while seeing
what changes (see :ref:`continue-relaxation`).

``Resubmit…`` is offered only for a job whose workflow row still holds it: a
job whose row was removed or reconfigured away, or an imported job, has no row
to run, so its row has no ``Resubmit…``. With jobs from
several workflows or targets selected, each workflow and target is submitted
in turn, and the cluster submission dialog preselects the job's own cluster.

What starting again clears
~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting a job again deletes everything in its local output folder except its
``.cell``/``.param`` inputs, which are then rebuilt from the row. The folder
must be the job's own: it is left alone, and the job is kept out of the
submission and reported with the reason, when it is outside
``<project>/files/<stage>/``, is a link or is reached through one, is not in
its own workflow and charge folder (a muon relaxation), is not named after the
job, or is used by another job. Running the job from such a folder would
resume from a checkpoint you chose to discard. A Quantum Corrections phonon job
is the one exception: the singlepoint an older project ran in the same seed
folder does not keep the folder, so clearing it lets the row rerun as a fresh
phonon job, while the singlepoint stays in the Job Dashboard as a record.
Imported jobs never run again, so their folders are never cleared.

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
  check, cancel and accounting (``sacct``) commands. A command without a configured path is located by
  probing the non-interactive SSH shell. Redwood uses
  ``/usr/local/shared/slurm/bin/sacct`` for accounting; a saved Redwood profile
  without an accounting-path setting receives this default. An explicitly saved
  empty path keeps command discovery enabled.
- SSH key authentication is primary, with password fallback
  (``Authentication method``: ``SSH key`` or ``Password``).
- The Redwood preset targets ``glamdring.physics.ox.ac.uk`` through the Physics
  gateway ``bastion.physics.ox.ac.uk``. MuFinder prompts separately for the
  Physics/bastion and Glamdring credentials because those accounts can differ.
  The preset submits with ``/usr/local/shared/bin/addqueue``; a saved profile
  keeps whatever submit command path it was saved with.
- The Redwood preset records its compute nodes (19 that run up to 32 processes
  and 28 that run up to 64) for :ref:`addqueue-node-placement`; a saved Redwood
  profile on that host without the setting receives it. The node list is not edited in
  ``Setup > HPC Clusters``. ``Cores per node`` stays the pre-filled total, not a
  limit.

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
  output is being retrieved), failed or stopped cluster jobs whose output may
  still be on the cluster because it could not be retrieved or its result
  recorded, or MuFinder lost track of them, or cluster jobs cancelled in the
  project only (dismissed ones included), the status bar shows the connection
  state, when the queue was last checked (``Queue last checked``) and a
  ``Refresh Queue`` button. With no project, or with only local jobs, completed
  jobs and jobs whose output has already been read, these three are hidden.
- The connection state names the cluster. ``Not connected to SCARF`` in grey is
  the ordinary idle state, for example after reopening a project.
  ``Connected to SCARF`` is green. ``Connection to SCARF lost`` is red and means
  a connection dropped while jobs on that cluster are active: their status is
  not being updated until you reconnect. A cluster with only failed or stopped
  jobs never reads as lost; it reads ``Not connected``.
- ``Refresh Queue`` checks the queue now. When no connection is open to a cluster
  with such jobs, active or with output left to fetch, it reconnects first. It does what ``Check now`` does for every
  job on the connected cluster: the active ones, including any with no scheduler
  job ID, and the failed, stopped and cancelled ones, dismissed or not. A job
  the scheduler lists again returns to ``queued`` or ``running``, and a job
  whose folder holds complete output from its run is completed and its result
  used, a job you cancelled included. The scheduler is asked about every job in
  one pass; a job's output is downloaded again only when it was not already
  read to a verdict, so a job whose retrieved output showed a CASTEP error or a
  walltime stop is not fetched again. A
  job whose submission was rejected is left out. A press while a check is
  running waits for it and then runs. The status bar reports what the refresh
  changed (``Checked 5 cluster jobs: 1 completed, 1 started running``), or
  ``Refreshed queue: checked 2 jobs at 12:34:56`` when nothing changed, or that
  the check failed; ``Queue last checked`` updates either way. If a job's
  connection has died unnoticed, the first refresh reports it as lost and the
  next one reconnects. The button is
  disabled, with a tooltip saying why, when no job has a remote job id to poll,
  such as while the only cluster job is still being submitted.
- The queue is polled periodically, updating when each job was last checked, shown
  in its ``Status`` tooltip. The periodic poll, like the check when a project
  opens or a cluster connects, looks at active jobs only and is skipped while
  another check is running.
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
