CASTEP Keyword Catalog
======================

MuFinder learns which keywords a CASTEP version understands from a *keyword
catalog*: a ``castep_keywords.json`` file distilled from CASTEP's own ``-help``
output. The catalog is CASTEP material, so it is never bundled with MuFinder and
must never be committed to this repository. Every installation therefore has to
obtain one, and the application is built to make that hard to miss and quick to
do.

Why it matters
--------------

Without a catalog MuFinder runs on the small starter catalog in
``docs/castep/keyword_catalog.yaml``. That is enough to write the inputs MuFinder
generates, but not to read everything CASTEP writes back:

- A full cell parse rejects keywords the active schema does not list, so an
  input ``.cell`` carrying a keyword outside the starter subset cannot be read
  for its settings. A relaxed ``-out.cell`` is read for its geometry only (see
  :doc:`castep/cell_syntax`), so the calculation keywords CASTEP echoes into it,
  such as ``QUANTISATION_AXIS`` or ``EXTERNAL_PRESSURE``, never block a
  workflow.
- Parameter dialogs offer only the starter subset of CASTEP parameters.
  ``Spin Polarized`` is in that subset for every workflow. ``SPIN`` (``Initial
  Spin``, the net spin of the whole cell) is not offered: MuFinder sets spins
  per atom with ``SPIN=`` tokens, CASTEP derives the net spin from them, and an
  explicit ``SPIN`` that disagrees with their sum stops the run. With a catalog
  loaded it can still be added from a dialog's advanced-parameter picker like
  any other keyword.
- Legacy-import parameter recovery is less complete (see :doc:`/legacy_import`).

Which catalog is used
---------------------

``mufinder.castep_io.schema`` picks the first usable file from this list:

1. The file named by the ``MUFINDER_CASTEP_KEYWORDS_JSON`` environment
   variable. It is the one override, intended for developers, and always wins;
   the catalog status says the catalog was set by this variable.
2. The catalog installed through ``Setup > CASTEP`` (see below).
3. The bundled starter catalog.

MuFinder does not search CASTEP source trees or any other location for a
catalog: only these three sources are used. A candidate that is missing,
unreadable, not JSON, or not a keyword catalog is skipped rather than fatal, so
a deleted or corrupted file can never stop the application starting. When the
override or the installed catalog is skipped, the reason is kept and shown to
the user. When the starter catalog
is used, ``mufinder.castep_io.starter_catalog`` translates its YAML entries,
merged with ``catalog_overrides``, into the same keyword specs a full catalog
produces. ``keyword_catalog_status()`` reports the origin, path, CASTEP version, keyword
counts, whether the starter catalog is active, and any such reason.

Installing a catalog
--------------------

.. figure:: _generated/screenshots/castep_setup_dialog.webp
   :alt: The Setup > CASTEP dialog with no keyword catalog loaded, offering to
         generate one or use an existing file.

   ``Setup > CASTEP`` before a keyword catalog is installed.

``Setup > CASTEP`` offers two routes, and both end the same way:

- **Generate Keyword Catalog** asks CASTEP to describe its keywords, either
  locally or on a configured cluster.
- **Use Existing File…** imports a ``castep_keywords.json`` the user already
  has.

The dialog keeps the keyword catalog apart from the optional ``CASTEP on this
computer`` paths, and states that the local executable and MPI launcher are
used only for local runs and local keyword generation. It shows which catalog
is in use and why a configured one was skipped. A line under the generation
target says which settings that target uses (this computer, or a cluster as a
scheduler job or on its login node), and a set-up cluster is preselected as the
target when no local executable is configured.

Either way the result goes through ``install_keyword_catalog``
(``mufinder.castep_setup``), which validates the file, writes MuFinder's own
copy into the per-user application-data directory, points the saved setup at
that copy, and rebuilds the running schema. The user never chooses a location,
the catalog takes effect immediately (no ``OK`` needed), and it cannot go missing
because a file elsewhere was moved. The copy does not track later changes to the
original; after a CASTEP upgrade, generate or import again. When the saved setup
names a complete catalog outside the application-data directory, startup
installs it the same way, during the splash screen's ``Loading CASTEP
configuration…`` step.

Validation (``mufinder.castep_io.keyword_catalog_file``) requires ``cell`` and
``param`` sections whose entries carry ``option_type``, ``level`` and
``docstring``, the sentinel keywords every CASTEP catalog documents
(``LATTICE_CART``, ``POSITIONS_FRAC``, ``SPECIES_POT``, ``TASK``,
``CUT_OFF_ENERGY``, ``XC_FUNCTIONAL``), and at least 40 ``cell`` and 150
``param`` keywords (CASTEP 26.11 has 118 and 449). A file that fails is rejected
with the reason and the catalog in use is left untouched. Catalogs are written
atomically, to a temporary file that is then moved into place.

Generation always writes to a scratch directory first, so a failed or partial
run cannot replace a working catalog.

Local generation wraps the CASTEP executable in the configured MPI launcher
(``mpirun -np 1``) when one is set. MPI-built binaries such as ``castep.mpi``
hang when the ``-help`` queries are run directly, so without the launcher the
progress stalls at 0%.

Cluster generation
------------------

CASTEP documents keywords one at a time: ``castep -help all`` lists the names
and ``castep -help <NAME>`` describes one, so a catalog needs several hundred
calls (567 for CASTEP 26.11). ``mufinder.services.cluster_keyword_generator``
issues them from a shell loop on the cluster that prints each description
between marker lines, and parses that output locally. The cluster needs only
``bash`` and CASTEP.

The scripts sent to the cluster are shell templates in
``mufinder/assets/shell/``: ``login_command.sh`` wraps each login-node command,
``help_loop.sh`` (with ``help_loop_timing.sh``) is the loop itself,
``job_help_loop.sh`` adapts it to a scheduler job, and ``keyword_job.sh`` with
one of the ``keyword_job_strategy_*.sh`` fragments is the job script.
``mufinder.services.cluster_keyword_scripts`` works out the values that go into
them (the cluster's setup, launcher and executable, shell-quoted, and the
markers and limits that the parser shares), and
``mufinder.services.shell_templates`` fills them in. Each template explains its
own logic in ``##`` comments, which are not sent to the cluster, and the tests
run ``bash -n`` over every script MuFinder can render.

Each cluster chooses where the loop runs. The choice is offered beside the
generation target in ``Setup > CASTEP`` (shown once a cluster is selected, and
remembered for that cluster) and as ``Generate keywords using`` in
``Setup > HPC Clusters``; both edit the same per-cluster setting:

- **Scheduler job** (the default, listed as ``Scheduler job (recommended)``)
  uploads one shell script, submits it, and reads back the dump it wrote. It is the safe choice where running CASTEP on a
  login node is forbidden or discouraged. Queue polling backs off from 2 to 15
  seconds. The script ends the dump with a completion marker; a dump without it
  (a job that hit its walltime, say) is reported as a failure rather than
  parsed.
- **Login node** lists the keywords, then describes them in batches of 50 per
  SSH command: about fourteen round trips for CASTEP 26.11, where one command
  per keyword needed 569, each paying for a login shell and the cluster's shell
  initialisation. Nothing waits in a queue, so it is quicker where it is
  permitted. Progress is reported and cancellation honoured between batches.

``addqueue`` clusters always use a scheduler job. Clusters saved before the
default changed keep the mode they were saved with.

What is inferred from the cluster's job settings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MuFinder already knows how to run CASTEP on a cluster: the job script template
used for every other job. Keyword generation reads the same template, so the
cluster does not have to be described twice.

The job runs on the cluster's default queue, with that queue's overrides of
the template, executable, walltime and memory applied first. On SLURM the
default queue is added to the script as ``#SBATCH -p``, and its memory per
CPU as ``#SBATCH --mem-per-cpu``, unless the template already names a
partition or a memory request.

- The *run line* is the last command in the template that mentions
  ``$castep_command`` or ``$seed_name``. Everything above it (scheduler
  directives such as partition and account, ``module load`` lines, other
  environment setup) is kept, with one process, one node, and a one hour
  walltime of its own rather than the cluster's default job walltime (the
  default queue's maximum instead, when that is shorter). The run
  line and anything after it are dropped.
- The *launcher* is whatever precedes the CASTEP executable on the run line
  (``srun -n 1``, for example). For ``$castep_command`` it is the launcher
  MuFinder itself builds for real jobs.
- The *executable* is the one the run line names if the template hard-codes it,
  otherwise the cluster's configured CASTEP executable.
- Inside the job the launcher is used as little as the cluster allows, because
  starting a job step costs seconds where starting CASTEP costs a fraction of
  one (on SCARF, launching every query through ``srun`` ran at about 20 queries
  a minute, roughly half an hour for a catalog). The script tries, in order:

  1. the bare executable, when it starts inside the allocation;
  2. a single job step that runs the whole loop with the bare executable inside
     it, when two consecutive starts inside one step both work;
  3. the launcher for every query, which mirrors how real jobs run and always
     applies.

  The first two are attempted only when ``timeout`` is available, and each is
  kept only if its dump is complete and at least nine queries in ten succeeded;
  otherwise the next one replaces it. ``timeout`` also bounds every query (two
  minutes), in both modes: a binary that answers once and then hangs on a later
  start costs a strategy a few minutes, because the loop gives up after three
  queries have been cut off, rather than holding the job to its walltime. ``keyword_generation.log`` in the job directory records which were tried.
  While the job runs, the dialog shows how many keywords have been described.
- ``addqueue`` clusters are handled differently, because ``addqueue`` starts
  the job script with ``srun --mpi=pmix``: the script is already a task of an
  MPI job step. A bare CASTEP there is that step's rank 0 rather than an
  independent process. Only its first start works, and when a later one fails
  in ``MPI_Init`` the scheduler cancels the whole step (seen on Redwood as
  ``STEP ... CANCELLED DUE TO TASK FAILURE`` four seconds into the job). So
  CASTEP is never started bare in the job script. Two strategies are tried:

  1. One ``mpirun -np 1`` runs the whole loop as a container. Inside it the
     loop removes the variables that tie a process to an MPI job (the
     PMIx/PMI rendezvous and the Slurm job-step variables), so each CASTEP
     starts as a standalone single process, as in a plain batch script, four at
     a time. A failure is confined to that one ``mpirun``. The attempt is
     limited to 15 minutes where ``timeout`` is available.
  2. ``mpirun -np 1`` for every query, one at a time, with no Open MPI
     overrides. This is proven on Redwood but slow there: 2.4 s a query, over
     twenty minutes for a catalog.
- On the login node the template's setup commands are run as well, and the
  launcher is not used.

The ``Keyword shell init (optional)`` setting in ``Setup > HPC Clusters`` is
therefore rarely needed: it adds commands to those read
from the template, and replaces the inferred setup on the login node. A cluster
with no usable template (the file is missing, or no run line was found) gets a
generic script with the scheduler defaults and ``srun -n 1`` as the fallback
launcher. ``addqueue`` clusters use the same generic script, submitted with the
cluster's queue name and memory settings and ``mpirun -np 1`` as the launcher.

Two things cannot be inferred: whether the site allows CASTEP on its login
node, which is why the scheduler is the default, and whether a bare MPI binary
starts there. If login-node generation fails, the error quotes the first failing
query; switch the cluster to a scheduler job.

Start-up time of the queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

How long generation takes is set almost entirely by how long CASTEP takes to
start, several hundred times over. Locally that is about 0.2 s. On SCARF it was
2.8 s, of which only 0.8 s was CPU time: an MPI-built CASTEP brings up its
network fabric before it prints help, and there Open MPI tried its OFI
transport, failed, and fell back. A serial catalog took about half an hour. The
loop therefore does two things, in both modes:

- It runs four queries at a time where that pays (``MUFINDER_PARALLEL`` sets
  the group size). Each query writes to its own file, and each group is printed
  whole and in order once it finishes. Whether it pays depends on the node, so
  the loop measures it: one query alone, then a group of four, and it drops to
  one at a time unless the group finishes in clearly less time than four in
  sequence. A group that overruns is abandoned at that point and its queries
  are run again in turn, so the trial costs a second or two at most. On
  SCARF queries mostly waited, and groups were much faster. On Redwood a lone
  query took 0.5 s, but four sharing the job's single CPU took several seconds
  between them. The measurement and the decision are written to the job log.
  The launcher-per-query strategy always runs one at a time, because a one-task
  allocation cannot run job steps side by side.
- It restricts Open MPI to its in-process transports (``OMPI_MCA_pml=ob1``,
  ``OMPI_MCA_btl=self``), which a single process needs nothing beyond, and
  tells it to yield the CPU while waiting (``OMPI_MCA_mpi_yield_when_idle=1``)
  so that queries sharing a core do not starve each other. The
  settings are kept only if CASTEP still answers ``--version`` with them, and
  other MPI libraries ignore them.

Individual ``-help`` calls that fail are tolerated, but the assembled catalog
must pass the same validation as an imported one.

Troubleshooting a failed scheduler job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When keyword generation through the scheduler fails, the error shown in
``Setup > CASTEP`` (open ``Details``) carries what is needed to diagnose it:

- the failure itself: the scheduler's rejection if the job could not be
  submitted, ``job failed`` if the scheduler reported it failed, ``ended before
  it had described every keyword`` if it ran but did not finish, or ``described
  only N of M keywords`` with the first failing query if CASTEP ran but most
  queries failed;
- the tail of ``keyword_generation.log``, including a ``mufinder: trying ...``
  line for each strategy attempted, the number of CPUs the job could use,
  whether the Open MPI fast-start settings were applied, and the wall, user and
  system time of one query (which shows whether a slow run is computing or
  waiting);
- the tail of every other file in the job directory, which is where the
  scheduler writes its own output (``slurm-<id>.out`` and the like) and where a
  job that died before reaching the keyword script explains itself;
- the job directory and scheduler job id. A failed job's directory
  (``.mufinder-keywords-<id>`` under the cluster's default working directory)
  is left in place so its files can be read; a successful or cancelled run
  removes it.

Earlier versions ran a Python helper script on the cluster for scheduler mode
and required a ``Keyword Python command`` cluster setting. That setting has been
removed; a stale value left in saved settings is ignored.

Keeping the user informed
-------------------------

Running without a catalog is allowed, never blocked, but it is made visible by
``mufinder.app.keyword_catalog_guard``:

- A permanent status-bar indicator shows the catalog's CASTEP version, or
  ``⚠ CASTEP keywords: limited built-in set``. Clicking it opens CASTEP setup.
  It also flags a configured catalog that was skipped.
- At startup, with no catalog, a prompt explains what is missing and offers
  ``Open CASTEP Setup…``. Ticking ``Don't show this again`` dismisses it
  permanently.
- Before a local or cluster submission, with no catalog, a warning offers to
  open CASTEP setup, submit anyway, or cancel. Choosing to submit anyway
  silences the warning for the rest of the session.
- The parameter dialog notes that not every CASTEP parameter is offered.
- An unknown-keyword parse error on the starter catalog says that no catalog is
  loaded and points at ``Setup > CASTEP``, instead of the contributor-facing
  advice to edit the schema.
