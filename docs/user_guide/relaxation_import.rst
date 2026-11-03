Importing CASTEP Relaxations
============================

MuFinder can import muon structural relaxations that CASTEP ran outside the
application, by hand or through another tool, so that those relaxed muoniated
structures can be used as inputs for Quantum Corrections and Energy Barriers
calculations. Only geometry optimisations with a muon in them are imported. The
design behind the importer is described in :doc:`/relaxation_import`.

Before you start
----------------

The importer needs:

- **An open project with an active host structure** that matches the
  calculations. The importer refuses to start until a host is loaded.
- **A folder holding the runs.** MuFinder searches it and every folder under it,
  so the runs can be kept however you like.

Each run needs these files, written by CASTEP with one seed name:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - File
     - What it gives
   * - ``<seed>.castep``
     - The run itself: whether it finished and converged, its total energy and
       the parameters it used.
   * - ``<seed>.cell``
     - The starting structure, which fixes the supercell and the muon's initial
       position.
   * - ``<seed>-out.cell``
     - The relaxed structure. CASTEP writes it when ``WRITE_CELL_STRUCTURE`` is
       on.
   * - ``<seed>.param`` (optional)
     - Parameters the ``.castep`` does not report.

MuFinder looks for the ``.cell``, ``-out.cell`` and ``.param`` beside the
``.castep`` first, then takes the closest file of that name elsewhere under the
folder, so a tree that keeps inputs and results apart also works. A file beside
another ``.castep`` of the same seed belongs to that run. ``Notes`` names any file
read from another folder. MuFinder does not guess: when two different files of
one name are equally close, or one file would serve several runs, the run is
listed as not importable, and the reason names the file.

Using the importer
------------------

1. **Load the host.** Open the project and make sure it has an active host
   structure that matches the calculations (see
   :doc:`/user_guide/host_structure`).
2. **Choose the folder.** Choose ``File > Import CASTEP Relaxations…`` and, in
   the folder dialog ``Select a folder containing CASTEP relaxations``, select
   the folder. MuFinder scans it on a background thread and reports progress in
   the status bar.
3. **Review the runs.** In the ``Import CASTEP Relaxations`` dialog (see `The
   import dialog`_), check which runs are ticked. ``Select converged`` and
   ``Select none`` adjust the ticks.
4. **Import.** Click ``OK``. The imported runs appear as read-only ``Imported
   <seed>`` rows in the ``Structural Relaxations`` table (``Muon Sites >
   Relaxations``).
5. **Use them downstream.** The imported rows can be used as sources in Quantum
   Corrections and as endpoints in Energy Barriers.

To undo an import, see `Removing an import`_.

The import dialog
-----------------

.. figure:: ../_generated/screenshots/relaxation_import_dialog.webp
   :alt: The Import CASTEP Relaxations dialog listing three CoF2 muon runs, two
         converged and ticked and one completed but unconverged.

   A hand-authored folder of three CoF₂ positive-muon relaxations, for the host
   of :doc:`/user_guide/examples/cof2`.

The dialog lists each ``.castep`` found with the columns ``Import``, ``Run``,
``Charge``, ``Status``, ``Supercell``, ``Energy (eV)``, and ``Notes``. ``Run`` is
the file's path under the chosen folder, without its suffix, shortened in the
middle when it is long, with the whole path in its tooltip. Converged runs are
pre-ticked; runs that cannot be imported are shown unticked and disabled, with
the reason in ``Notes``. ``Charge`` is empty for a run with no muon. A summary
above the table reports parameter disagreements, undetermined charge states and
anything the search could not read.

What can be imported
--------------------

- A run with no muon in its structure, such as a pristine reference or any other
  host-only run, is listed as ``No muon in this structure.`` and cannot be
  ticked. A muon is an ``H:mu`` atom, or a bare ``H`` when the cell holds exactly
  one that the host does not account for.
- A run of another CASTEP task, such as a single point or a phonon calculation,
  is listed with the task it ran and cannot be ticked.
- A run whose file name starts or ends with a space or a dot cannot name a job
  folder, so it is listed with that reason. Rename its files to import it.
- A run also cannot be imported when its CASTEP calculation did not complete, its
  ``.cell`` or ``-out.cell`` is missing or cannot be parsed, the two do not hold
  the same atoms, or its lattice is not an integer supercell of the host. An input cell with a keyword outside the
  loaded keyword catalog, such as ``EXTERNAL_PRESSURE``, still imports: its
  geometry is read, ``Notes`` says so, and a k-point grid that only the cell
  states, and the ``.castep`` does not report, is not imported.
- An unreadable ``.param`` is noted in ``Notes`` and does not stop the import: the
  ``.castep`` records what ran, and a ``.param`` only fills in what it leaves out.
- A completed but unconverged run can be ticked. It is imported as completed,
  with geometry convergence recorded as not achieved; nothing is imported as
  failed.
- The charge state comes from the total charge the ``.castep`` reports, or the
  ``.param`` if it does not: ``+1`` is a positive muon and ``0`` is muonium. An
  undetermined charge state, or a muon labelled with a bare ``H`` rather than
  ``H:mu``, is reported in the dialog but does not prevent the import.
- When the project has no active muon-relaxation definition, the import creates
  one and makes it active.
- Each imported row is labelled ``Imported <seed>``, with a suffix when that label
  is already in use: ``Imported <seed> (2)``. The table's ``Label`` column widens to fit it, and its
  tooltip gives it whole.
- A ``.castep`` that repeats another of the same name byte for byte, as a copied
  results folder does, counts once. The per-image ``<seed>_farmNNN.castep``
  outputs of a task farm are not runs.
- A folder the search cannot open, because it is locked or its path is too long
  for the system, is reported rather than passed over, since runs may be inside
  it. If nothing else was found, the message says so instead of reporting no
  outputs.
- Importing the same folder again creates new entries; runs already imported
  are not detected and skipped.

Working with imported rows
--------------------------

An imported row is read-only. It cannot be submitted, continued, edited in
place, or included in a batch parameter edit, and no dashboard reset or
directory cleanup deletes its files. Its parameter dialog shows the values the
run used, recovered from its ``.castep`` output, and downstream phonon and NEB
jobs inherit that run's cut-off energy and k-point grid.

What MuFinder writes
--------------------

Each imported run gets its own folder in the project:

::

    files/muon_relaxation/<charge_prefix>/<seed>/
        <seed>.cell               rendered: true initial structure plus the
        <seed>.param              effective parameters read from the .castep
        <seed>.castep             byte-exact copy
        <seed>-out.cell           byte-exact copy
        original_inputs/          byte-exact inputs + SHA-256 manifest

The seed name is kept. When a folder of that name already exists, because runs in
different folders share a seed or a folder is imported twice, the new job's seed
gets a numeric suffix (``<seed>_2``) and its files are named to match, so no run
overwrites another. ``original_inputs/`` keeps the input files' own names.

The ``.cell`` and ``.param`` files are rendered from the ``.castep`` record:
they describe the same atoms, with the k-point grid and cutoff that actually
ran. The rendered ``.cell`` contains the **initial** structure, not the relaxed
one, so that initial-versus-relaxed comparisons stay truthful.

Removing an import
------------------

1. Choose ``File > Remove Imported Relaxations…``.
2. In the ``Remove Imported Relaxations`` dialog, pick the ``Import batch`` to
   remove and click ``Remove…``. Each batch is named by its source folder and when
   it was imported, with the number of relaxations it holds.
3. Read the confirmation, which names what will be deleted. Its buttons are
   ``Remove <N> relaxations`` and ``Cancel``, and ``Cancel`` is the default.

A folder imported twice by an earlier version holds both batches' files in one job
folder. Removing one of those batches removes its rows and records and keeps the
files, and the confirmation says so; the folder is deleted when the last batch using
it is removed.

MuFinder refuses while anything still references an imported site, its run or
its structure; the message lists what still uses them. A relaxation whose saved
location is not its own folder, because it leads out of its charge folder or out of the
project, comes out with its records and leaves that location untouched, and the
confirmation says its files stay. MuFinder also refuses, and nothing goes, unless each
other folder is a real folder rather than a link, is the one the import recorded writing
to, and is used by no job other than another import of the same files. Otherwise it deletes the
rows, mapping entries, jobs, runs, results, structures, sites and copied files
as one unit, and records the removal in the project provenance and Journal. When some
of a batch's folders go and others stay, the status bar counts them
(``Removed 3 imported relaxations; deleted 2 folders; kept 1``), a box says why with
the names under ``Details``, and the Journal names each one.

A row of your own whose job folder is also used by an imported relaxation or another
job, or whose saved location is not its own folder, comes out without touching that
folder. MuFinder does not offer to delete or move it: the status bar counts it as kept,
and a box says why with the name under ``Details``. A folder that is a link keeps the
row until you remove the link.

The reference check covers quantum-corrections and energy-barrier definitions,
comparison tables, relaxation-analysis workspaces, and muon-sites summary
snapshots. Remove those references first: an imported relaxation owns the only
copy of its results, so removal refuses rather than leave a dangling reference.

Limitations
-----------

- Only muon geometry optimisations are imported; other calculations are not.
- Imported rows are read-only and cannot be re-run, continued or edited.
- A run's lattice must be an integer supercell of the active host to within a
  tenth of a percent, so a run built on a different relaxed host lattice is
  refused.
- Re-importing a folder does not detect runs already imported.
- An import cannot be removed while downstream calculations, comparison tables
  or analysis workspaces still reference it.
