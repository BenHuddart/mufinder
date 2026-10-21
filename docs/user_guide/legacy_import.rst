Importing Legacy Relaxations
============================

MuFinder can import muon structural relaxations produced by the **legacy
MuFinder** (the Tkinter-era GitLab application) into a project, so that those
relaxed muoniated structures can be used as inputs for Quantum Corrections and
Energy Barriers calculations. Only muon structural relaxations are imported:
convergence sweeps, pristine relaxations, UEP, and dipolar-field data are not.
The design behind the importer is described in :doc:`/legacy_import`.

Using the importer
------------------

1. Open the project and make sure it has an active host structure that matches
   the legacy calculations. The import refuses to start without one.
2. Choose ``File > Import Legacy Relaxations...`` and select the legacy working
   folder (the one containing ``input/``, ``run/``, and ``results/``).
   MuFinder scans it on a background thread and reports progress in the status
   bar.
3. Review the ``Import Legacy Relaxations`` dialog. Each detected run is listed
   with ``Import``, ``Site``, ``Charge``, ``Status``, ``Supercell``,
   ``Energy (eV)``, and ``Notes`` columns. Converged runs are pre-ticked; runs
   that cannot be imported are shown unticked and disabled, with the reason in
   ``Notes``. ``Select converged`` and ``Select none`` adjust the ticks, and a
   summary above the table reports parameter disagreements and undetermined
   charge states.
4. Click ``OK``. The imported runs appear as read-only ``Legacy Site <N>`` rows in
   the ``Structural Relaxations`` table (``Muon Sites > Relaxations``) and can
   be used as sources in Quantum Corrections and as endpoints in Energy
   Barriers.

To undo an import, choose ``File > Remove Imported Relaxations...``, pick the
import batch, and confirm. Removal is refused while any downstream calculation,
comparison table, or analysis workspace still references the imported sites.

What can be imported
--------------------

- The project must have an active host structure that matches the legacy
  calculations. The importer does not adopt the legacy folder's CIF as the
  host; it refuses to start until a host is loaded.
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

An imported row is read-only. It cannot be submitted, continued, edited in
place, or included in a batch parameter edit, and no dashboard reset or
directory cleanup deletes its files. Its parameter dialog shows the values the
legacy run used, recovered from its ``.castep`` output, and downstream phonon
and NEB jobs inherit that run's cut-off energy and k-point grid.

Files on disk
-------------

::

    files/muon_relaxation/<charge_prefix>/<legacy_seed>/
        <legacy_seed>.cell        rendered: true initial structure plus the
        <legacy_seed>.param       effective parameters read from the .castep
        <legacy_seed>.castep      byte-exact copy
        <legacy_seed>-out.cell    byte-exact copy
        legacy_source/            byte-exact originals + SHA-256 manifest

The **legacy seed name is retained**. It keeps the ``.castep`` self-consistent
with its own filename, requires no content rewriting, and makes imported job
directories visibly distinct, which is a provenance feature rather than a cost.
The seed-name mapping already supports arbitrary seeds.

Rendering the ``.cell`` and ``.param`` from the ``.castep`` record is what makes
inheritance correct. It is not fabrication: the same atoms, with the grid and
cutoff that actually ran made explicit. The rendered ``.cell`` must contain the
**initial** structure, not the relaxed one, so that initial-versus-relaxed
comparisons stay truthful.

Removing an import
------------------

``File > Remove Imported Relaxations...`` removes one import batch. MuFinder
refuses while anything still references an imported site, its run or its
structure; otherwise it deletes the rows, mapping entries, jobs, runs, results,
structures, sites and copied files as one unit, and records the removal in the
project provenance and journal.

The reference check covers quantum-corrections and energy-barrier definitions,
comparison tables, relaxation-analysis workspaces, and muon-sites summary
snapshots. Remove those references first: an imported relaxation owns the only
copy of its results, so removal refuses rather than leave a dangling reference.
