Importing Legacy Relaxations
============================

MuFinder can import muon structural relaxations produced by the **legacy
MuFinder** (the Tkinter-era GitLab application) into a project, so that those
relaxed muoniated structures can be used as inputs for Quantum Corrections and
Energy Barriers calculations. Only muon structural relaxations are imported:
convergence sweeps, pristine relaxations, UEP, and dipolar-field data are not.
The design behind the importer is described in :doc:`/legacy_import`.

Before you start
----------------

The importer needs:

- **An open project with an active host structure** that matches the legacy
  calculations. The importer does not adopt the legacy folder's CIF as the
  host; it refuses to start until a host is loaded.
- **The legacy working folder**, the one containing ``input/``, ``run/``, and
  ``results/``.

Using the importer
------------------

1. **Load the host.** Open the project and make sure it has an active host
   structure that matches the legacy calculations (see
   :doc:`/user_guide/host_structure`).
2. **Choose the folder.** Choose ``File > Import Legacy Relaxations...`` and,
   in the folder dialog ``Select a legacy MuFinder working folder``, select
   the legacy working folder. MuFinder scans it on a background thread and
   reports progress in the status bar.
3. **Review the runs.** In the ``Import Legacy Relaxations`` dialog (see `The
   import dialog`_), check which runs are ticked. ``Select converged`` and
   ``Select none`` adjust the ticks.
4. **Import.** Click ``OK``. The imported runs appear as read-only ``Legacy Site
   <N>`` rows in the ``Structural Relaxations`` table (``Muon Sites >
   Relaxations``).
5. **Use them downstream.** The imported rows can be used as sources in Quantum
   Corrections and as endpoints in Energy Barriers.

To undo an import, see `Removing an import`_.

The import dialog
-----------------

.. figure:: ../_generated/screenshots/legacy_import_dialog.webp
   :alt: The Import Legacy Relaxations dialog listing three CoF2 muon runs, two
         converged and ticked and one completed but unconverged.

   A hand-authored legacy folder of three CoF₂ positive-muon relaxations.

The dialog lists each detected run with the columns ``Import``, ``Site``,
``Charge``, ``Status``, ``Supercell``, ``Energy (eV)``, and ``Notes``. Converged
runs are pre-ticked; runs that cannot be imported are shown unticked and
disabled, with the reason in ``Notes``. A summary above the table reports
parameter disagreements and undetermined charge states.

What can be imported
--------------------

- Legacy site 0 is the pristine reference, not a muon site, so it is skipped.
- A run cannot be imported when it has no ``.castep`` output, its CASTEP
  calculation did not complete, its input cell cannot be parsed, or its lattice
  is not an integer supercell of the host. Such runs are listed unticked and
  disabled, with the reason in ``Notes``.
- A completed but unconverged run can be ticked. It is imported as completed,
  with geometry convergence recorded as not achieved; nothing is imported as
  failed.
- An undetermined charge state, or a muon labelled with a bare ``H`` rather than
  ``H:mu``, is reported in the dialog but does not prevent the import.
- When the project has no active muon-relaxation definition, the import creates
  one and makes it active.
- Each imported site is labelled ``Legacy Site <N>`` from the site index in the
  legacy file name, with a suffix when that label is already in use.
- Importing the same folder again creates new entries; runs already imported
  are not detected and skipped.

Working with imported rows
--------------------------

An imported row is read-only. It cannot be submitted, continued, edited in
place, or included in a batch parameter edit, and no dashboard reset or
directory cleanup deletes its files. Its parameter dialog shows the values the
legacy run used, recovered from its ``.castep`` output, and downstream phonon
and NEB jobs inherit that run's cut-off energy and k-point grid.

What MuFinder writes
--------------------

Each imported run gets its own folder in the project:

::

    files/muon_relaxation/<charge_prefix>/<legacy_seed>/
        <legacy_seed>.cell        rendered: true initial structure plus the
        <legacy_seed>.param       effective parameters read from the .castep
        <legacy_seed>.castep      byte-exact copy
        <legacy_seed>-out.cell    byte-exact copy
        legacy_source/            byte-exact originals + SHA-256 manifest

The legacy seed name is kept, so the ``.castep`` file keeps the name it was
written with and imported job folders are easy to tell from MuFinder's own.

The ``.cell`` and ``.param`` files are rendered from the ``.castep`` record:
they describe the same atoms, with the k-point grid and cutoff that actually
ran. The rendered ``.cell`` contains the **initial** structure, not the relaxed
one, so that initial-versus-relaxed comparisons stay truthful.

Removing an import
------------------

1. Choose ``File > Remove Imported Relaxations...``.
2. In the ``Remove Imported Relaxations`` dialog, pick the ``Import batch`` to
   remove.
3. Read the confirmation, which names what will be deleted, and confirm. The
   default answer is ``No``.

MuFinder refuses while anything still references an imported site, its run or
its structure; the message lists what still uses them. Otherwise it deletes the
rows, mapping entries, jobs, runs, results, structures, sites and copied files
as one unit, and records the removal in the project provenance and Journal.

The reference check covers quantum-corrections and energy-barrier definitions,
comparison tables, relaxation-analysis workspaces, and muon-sites summary
snapshots. Remove those references first: an imported relaxation owns the only
copy of its results, so removal refuses rather than leave a dangling reference.

Limitations
-----------

- Only muon structural relaxations are imported; other legacy data is not.
- Imported rows are read-only and cannot be re-run, continued or edited.
- Runs without a completed ``.castep`` output, with an unparseable input cell,
  or whose lattice is not an integer supercell of the host cannot be imported.
- Re-importing a folder does not detect runs already imported.
- An import cannot be removed while downstream calculations, comparison tables
  or analysis workspaces still reference it.
