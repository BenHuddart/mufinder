Convergence Testing
===================

Convergence Testing finds the plane-wave cutoff energy and the Monkhorst-Pack
k-point grid that every later calculation in the project uses. It is usually
the first set of CASTEP calculations you run on a new host. MuFinder runs a
sweep of single-point calculations over one setting at a time, plots the total
energy, maximum force, and (optionally) maximum stress against that setting,
and recommends the smallest value at which they stop changing. When you adopt
the recommendation it becomes a *project default*: the pristine-relaxation,
UEP, and muon-relaxation workflows then use it, together with the XC
functional the sweep was run with.

In the workflow navigator, ``Convergence Testing`` is a parent entry with two
children:

- ``Cutoff Energy`` opens the ``Cutoff Convergence`` page, which sweeps
  ``CUT_OFF_ENERGY``;
- ``K-Point Grid`` opens the ``K-Point Convergence`` page, which sweeps
  ``KPOINT_MP_GRID``.

Selecting ``Convergence Testing`` itself opens an overview page that summarizes
both project defaults (see `The overview page`_). The navigator marks the
parent complete once both a project default cutoff and a project default
k-point grid are set.

Before you start
----------------

Convergence Testing needs:

- **A host structure.** Load one first; every sweep runs on the host unit cell
  or on a supercell saved from it.
- **A supercell, if you want one.** By default the sweeps run on the host unit
  cell. To converge on a supercell instead, save it in the
  :doc:`/user_guide/supercells` first (or from the ``Manage Supercells...`` button on the
  sweep page). See `Limitations`_ before choosing a supercell.
- **A decision about the XC functional.** A converged cutoff or grid is only
  meaningful for the functional it was converged with. New sweep rows take the
  project default functional, so set it before you start: click ``Edit…`` on
  the ``Project defaults`` strip at the top of either sweep page and choose an
  ``XC functional``. With no project functional set, rows use the catalog
  default (``LDA``).
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`/castep_keyword_catalog`).

Running the workflow
--------------------

Converge the cutoff first, then the k-point grid: the k-point sweep takes the
adopted cutoff as its default ``CUT_OFF_ENERGY``.

1. **Open the cutoff sweep.** Select ``Cutoff Energy`` in the navigator (or
   click ``Open Cutoff Convergence`` on the overview page). Under
   ``Simulation Cell``, leave ``Single unit cell`` selected or choose a saved
   supercell from ``Saved supercells``.
2. **List the cutoffs.** Click ``Generate From Range...`` and set
   ``Start (eV)``, ``Stop (eV)``, and ``Step (eV)`` (300, 1400, and 100 eV by
   default), or click ``Add Cutoff`` to add rows one at a time. Double-click a
   value to edit it.
3. **Check the parameters.** Select the rows and click ``Configure Selected
   Parameters...``, or click one row's ``Configure...`` button. The defaults
   are listed in `Parameter dialogs`_. Tick ``Calculate Stress`` if you want
   stress to count towards convergence. The cutoff sweep writes no k-point
   setting by default (see `What MuFinder writes`_); add ``K-Point MP Grid``
   with ``+ Add Parameter`` if you want to control the sampling.
4. **Run the sweep.** Click ``Run All Locally`` or ``Run All on Cluster``. With
   rows selected the buttons read ``Run Selected Locally`` / ``Run Selected on
   Cluster`` and submit only those rows. The ``Status`` column then follows
   each job, and ``Total Energy (eV)`` and ``Max |Force| (eV/Ang)`` fill in as
   jobs complete.
5. **Examine the results.** When at least two rows have completed with both an
   energy and a force, click ``Examine Cutoff Convergence``. Set the
   thresholds, choose an ``Energy method``, and click ``Analyse
   Convergence`` (see `Examining the results`_).
6. **Adopt the cutoff.** Read the ``Energy result`` and ``Force result``
   lines, then click ``Set <value> eV as Project Default Cutoff``. The cutoff
   appears on the ``Project defaults`` strip and its plot is pinned on the
   overview page.
7. **Open the k-point sweep.** Select ``K-Point Grid`` in the navigator and
   pick the same simulation cell as the cutoff sweep.
8. **List the grids.** Click ``Generate From Spacing...`` and set ``Minimum
   spacing (1/Ang)`` (``0.02`` by default) to generate every distinct grid
   CASTEP produces between ``1 x 1 x 1`` and that spacing (see `Generating
   k-point grids`_), or click ``Add Grid`` to enter grids by hand.
9. **Check the parameters and run.** As in steps 3 and 4. The rows now use the
   adopted cutoff.
10. **Examine and adopt the grid.** Click ``Examine K-Point Convergence``, then
    ``Analyse Convergence``, then ``Set <grid> as Project Default K-Point
    Grid``.
11. **Check the overview.** Select ``Convergence Testing``: both cards show
    the adopted value, the functional, the method, and the criteria, with the
    pinned plots.

If you change the XC functional later, re-run both sweeps with the new
functional (see `Stale defaults`_).

The overview page
-----------------

.. figure:: ../_generated/screenshots/convergence_overview.png
   :alt: The Convergence Testing overview with a 600 eV cutoff and a 3 x 3 x 4
         grid adopted and their plots pinned.

   The overview after adopting a cutoff and a grid; the pinned sweeps are illustrative data.

The ``Convergence Testing`` page has two cards, ``Cutoff Convergence`` and
``K-Point Convergence``. Each card has an ``Open ...`` button that jumps to its
sweep page and lists:

- ``Project default`` — the current default cutoff or grid, or ``Not set``;
- ``Cutoff default`` (k-point card only) — the cutoff the grid sits alongside;
- ``Functional`` — the functional the value was converged with, or ``Not
  recorded``;
- ``Method`` — the energy method used, or ``Set manually`` for a value typed
  into the ``Project Defaults`` dialog;
- ``Criteria`` — the thresholds used.

Below the summary, each card shows a read-only copy of the examiner plot as it
was when the value was adopted, with the thresholds as ``dE``, ``dF``, and
(when used) ``dS`` badges. The pinned plot is replaced by a placeholder when
nothing has been adopted, when the default has since been edited or cleared by
hand, or when the active host structure has changed.

The sweep pages
---------------

.. figure:: ../_generated/screenshots/cutoff_convergence_page.png
   :alt: The Cutoff Convergence page with six cutoff rows from 300 to 800 eV
         ready to run.

   The ``Cutoff Convergence`` page with six configured cutoffs ready to run.

``Cutoff Convergence`` and ``K-Point Convergence`` share one layout.

``Project defaults`` strip
   The current ``XC``, ``Cutoff`` and ``K-points`` defaults. Hover a value for
   how it was established, or click ``Edit…`` to open the ``Project Defaults``
   dialog (see `Adopting a result as a project default`_).
``Simulation Cell``
   A summary of the cell the sweep will run on, a ``Saved supercells``
   drop-down (``Single unit cell`` plus every supercell saved for the active
   host), ``Manage Supercells...`` to open the :doc:`/user_guide/supercells`, and ``Use
   Unit Cell`` to return to the host unit cell. The cell is fixed for a job
   when the job is defined; changing the selection later does not change jobs
   already defined.
Sweep table (``Cutoff Sweep`` / ``K-Point Grid Sweep``)
   One row per sweep point:

   ``Cutoff Energy (eV)`` / ``K-Point Grid``
      The swept value. Double-click to edit. A grid is written as
      ``4 x 4 x 4``; any three positive integers separated by non-digits are
      accepted.
   ``Status``
      The row's job status, starting at ``ready``.
   ``Total Energy (eV)``
      The final total energy of the completed job.
   ``Max |Force| (eV/Ang)``
      The largest force on any atom in the completed job.
   ``Parameters``
      ``Configure...``, or ``Configured...`` once the row has its own values
      or has been submitted.
   (unlabelled)
      An ``x`` button that removes the row.
Row controls
   ``Add Cutoff`` / ``Add Grid``, ``Generate From Range...`` / ``Generate From
   Spacing...``, ``Remove Selected``, ``Configure Selected Parameters...``,
   and ``Examine Cutoff Convergence`` / ``Examine K-Point Convergence``.
   Right-click selected rows to run, cancel, inspect a failure, reset failed
   jobs, configure, or remove them. ``Delete`` (``Backspace`` on macOS)
   removes the selected rows.
``Run Jobs``
   ``Job name prefix`` (``cutoff_convergence`` or ``kpoint_convergence``) and
   the run buttons. Only rows that have not been submitted yet are run.

Once a row has been submitted, its value and its parameters are locked: the
value cell can no longer be edited and its dialog opens read-only as ``Inspect
Cutoff Parameters`` / ``Inspect K-Point Parameters``. A row whose job was
cancelled stays editable. Values must be unique within a sweep; MuFinder
refuses to run a sweep with a duplicate or non-numeric value.

A sweep's rows are restored from the saved form state when the project is
reopened, or from the active convergence definition when that state is
incomplete. Rows that use the default parameters keep their completed jobs
across a reload, so adding a cutoff and running submits only the new row. A
regenerated list starts as fresh ``ready`` rows and does not inherit the status
or results of an earlier definition.

Removing a row whose job is still queued or running cancels the job after a
confirmation. Removing a cutoff whose job directory contains files also asks
what to do with them: ``Yes`` deletes them, ``No`` (the default) moves them into
a ``removed_from_workflow`` folder, and ``Cancel`` keeps the cutoff.

Adding cutoffs
~~~~~~~~~~~~~~

``Add Cutoff`` adds 300 eV to an empty table, otherwise the last row's value
plus 20 eV. ``Generate From Range...`` opens ``Generate Cutoffs From Range``,
which adds every value from ``Start (eV)`` to ``Stop (eV)`` inclusive in steps
of ``Step (eV)``. New values are appended to the list (values already present
are not repeated) unless ``Replace existing cutoff list`` is ticked. The dialog
remembers its last values.

Generating k-point grids
~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/kpoint_convergence_page.png
   :alt: The K-Point Convergence page with the grids generated for rutile TiO2
         down to 0.07 1/Ang.

   Grids generated for rutile TiO2 with a minimum spacing of 0.07 1/Ang.

``Add Grid`` opens ``Add K-Point Grid``, pre-filled with the last row's grid.

``Generate From Spacing...`` opens ``Generate K-Point Grids``. MuFinder walks
CASTEP's own spacing-to-grid rule, ``n_i = floor(|b_i| / (2 pi s)) + 1``, from
large spacings down to ``Minimum spacing (1/Ang)``, using the reciprocal
lattice of the selected simulation cell, and adds every distinct grid it passes
through, from ``1 x 1 x 1`` up to the grid CASTEP would use at the minimum
spacing. Every generated grid is therefore one CASTEP can produce from some
``KPOINT_MP_SPACING``. In an anisotropic cell each axis steps up at its own
spacing, so the list can be longer than a simple ``n x n x n`` ramp. New grids
are appended unless ``Replace existing grid list`` is ticked.

MuFinder records which cell the grids were generated for. If you then select a
different simulation cell, the page shows *K-point grids were generated for a
different simulation cell. Regenerate grids before running new jobs.* Generate
the grids again with ``Replace existing grid list`` ticked.

Parameter dialogs
~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/parameter_dialog.png
   :alt: The parameter dialog of the 500 eV cutoff row, with the swept cutoff
         shown as set by the sweep.

   The ``Configure...`` dialog of the 500 eV row.

The row dialog follows the shared workflow layout, with the swept keyword
shown as ``Variable (set by sweep)``. ``+ Add Parameter`` adds any other
catalogued keyword. When several rows are configured together, a parameter
whose value differs between them is marked ``mixed`` and, if left alone, keeps
each row's own value.

Rows that have not been configured use these defaults:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Parameter
     - Cutoff sweep
     - K-point sweep
   * - ``Basis Precision`` / ``Cutoff Energy``
     - swept
     - project default cutoff; ``FINE`` basis precision when none is set
   * - ``SCF Energy Tolerance``
     - 1e-06 eV
     - 1e-06 eV
   * - ``K-Point MP Grid``
     - not written
     - swept
   * - ``XC Functional``
     - project default, else ``LDA``
     - project default, else ``LDA``
   * - ``Spin Polarized``
     - off
     - not written
   * - ``Max SCF Cycles``
     - 100
     - 100
   * - ``Calculate Stress``
     - off
     - off
   * - ``Metals Method``
     - ``DM``
     - not written
   * - ``Generate Symmetry``
     - on
     - on

The cutoff sweep also offers a ``Continuation`` row, empty by default.

What MuFinder writes
--------------------

Each row is one CASTEP ``SINGLEPOINT`` job. Its seed and folder are named
``<prefix>_ecut_<cutoff>`` (for example ``cutoff_convergence_ecut_300``) or
``<prefix>_kgrid_<a>x<b>x<c>`` (for example ``kpoint_convergence_kgrid_4x4x4``).
Folders are created in the project's ``files/convergence_cutoff`` and
``files/convergence_kpoint`` folders when the job is first submitted.

The ``.cell`` file contains the selected cell's ``LATTICE_CART`` and
``POSITIONS_FRAC`` blocks and ``SYMMETRY_GENERATE``. A k-point job adds its
``KPOINT_MP_GRID``. A cutoff job writes no k-point keyword unless you add one,
so CASTEP chooses the sampling itself. The ``.param`` file contains ``TASK :
SINGLEPOINT``, the swept or default ``CUT_OFF_ENERGY`` (or ``BASIS_PRECISION``),
and the other parameters from the table above. A default cutoff-sweep
``.param`` file reads::

   TASK : SINGLEPOINT
   XC_FUNCTIONAL : LDA
   CUT_OFF_ENERGY : 300 eV
   SPIN_POLARIZED : FALSE
   ELEC_ENERGY_TOL : 1e-06 eV
   MAX_SCF_CYCLES : 100
   CALCULATE_STRESS : FALSE
   METALS_METHOD : DM

Examining the results
---------------------

``Examine Cutoff Convergence`` and ``Examine K-Point Convergence`` open the
examiner dialog. It uses only rows with ``completed`` status, and opens with a
summary of how many completed points have energy, force, and stress values and
how many atoms are used for per-atom normalization.

``Convergence Criteria``
   ``Energy threshold (eV/atom)`` (default 0.001), ``Force threshold
   (eV/Ang)`` (default 0.05), and an optional ``Stress threshold (GPa)``
   (default 0.1, matching the usual ``GEOM_STRESS_TOL``). The force and
   stress thresholds cannot go below the precision CASTEP prints those
   quantities to, 0.00001 eV/Ang and 0.000001 GPa (``firstd_output_forces``
   and ``firstd_output_stress`` in CASTEP's ``Source/Functional/firstd.f90``):
   a finer threshold could only be met by values that print identically. A
   smaller value snaps to the minimum, and the field shows the value the
   analysis used. The energy threshold only has to be positive, since CASTEP
   prints energies to 13 significant figures. The stress box can only be
   ticked when every completed job reported a stress, that is, when every job
   was run with ``Calculate Stress`` on.
``Energy method``
   ``Delta-Threshold``, ``Derivative (dE/dEcut)``, or ``Asymptotic
   Extrapolation`` for a cutoff sweep; ``Info`` explains the selected method.
   A k-point sweep always uses ``Delta-Threshold``.
``Analyse Convergence``
   Runs the analysis on every completed point that has both an energy and a
   force (at least two are needed), writes the results into the text area, and
   redraws the plot.

The plot has an energy panel above a force panel, with the thresholds, the
value at which each criterion converged, and the recommended value marked.
When stress is used it is drawn on a second axis of the force panel. Hover a
point to see its value; on a k-point plot the tooltip also names the grid. For a k-point sweep the horizontal axis is the
``Number of k-points``, the product ``a x b x c`` of the grid divisions, on a
logarithmic scale; a second axis along the top marks round ``K-point spacing
(1/Ang)`` values at the matching positions.

The thresholds, method, results, and plot are saved with the project, so
reopening the examiner shows the last analysis.

How convergence is judged
~~~~~~~~~~~~~~~~~~~~~~~~~

Points are sorted by cutoff, or by number of k-points, and compared in
consecutive pairs.

- **Energy, Delta-Threshold.** For each pair, ``|E2 - E1|`` divided by the
  number of atoms. The energy is converged at the higher value of the *first*
  pair whose difference is at or below the energy threshold.
- **Energy, Derivative.** For each pair, ``|E2 - E1| / (Ecut2 - Ecut1)`` per
  atom, compared with the energy threshold divided by the mean cutoff step.
  For evenly spaced cutoffs this gives the same answer as Delta-Threshold.
- **Energy, Asymptotic Extrapolation.** Fits ``E(Ecut) = E_inf + A
  exp(-B Ecut)`` (needs at least four points) and reports the fitted ``E_inf``.
  The energy is converged at the first sampled cutoff whose fitted energy lies
  within the energy threshold per atom of ``E_inf``. The fit is rejected, with
  a suggestion to use Delta-Threshold, if it fails, if ``R^2`` is below 0.80,
  or if its per-atom RMS error exceeds five times the energy threshold.
- **Force.** For each pair, the change in maximum force, ``|F2 - F1|`` (not
  per atom). Converged at the higher value of the first pair at or below the
  force threshold.
- **Stress (optional).** The same test on the maximum stress, in GPa.

The recommendation (``Recommended cutoff`` or ``Recommended k-point grid``) is
the largest of the converged values, so the adopted setting satisfies every
criterion. Every criterion in use must converge: energy, force, and stress
when it is included. If any does not converge within the sweep there is no
recommendation, the adopt button stays disabled, and the results say which
criterion failed, for example *No recommendation: the force criterion did not
converge within the sweep.* Extend the sweep to larger cutoffs or denser grids,
or relax the threshold, and analyse again.

Because each test stops at the first small difference, a single accidentally
small step can trigger it. Check on the plot that the energy and force stay
flat beyond the recommended value.

For a k-point recommendation the result line also gives the grid's spacing,
``(spacing s 1/Ang)``. This is the largest ``KPOINT_MP_SPACING`` for which
CASTEP reproduces exactly that grid on the swept cell, so it can be copied into
a ``.cell`` file. ``1 x 1 x 1`` has no such spacing and is shown without one.
The convention, and its round-trip guarantee, are described in
:doc:`/castep/kpoint_spacing_convention`.

Adopting a result as a project default
--------------------------------------

After a successful analysis the adopt button names the value, for example
``Set 500 eV as Project Default Cutoff`` or ``Set 4 x 4 x 4 as Project
Default K-Point Grid``. Clicking it:

- sets the project default cutoff or grid, and records the energy method, the
  thresholds, the simulation cell, and the XC functional the sweep was run
  with; a grid is stored together with the lattice of the cell it was
  converged on, so later workflows can rescale it to their own cells;
- makes that functional the project default functional. If it differs from
  the current project functional MuFinder first asks *Change project
  functional?*; answering ``No`` adopts nothing. If the sweep's rows used
  different functionals, no functional is recorded and the project functional
  is left unchanged;
- pins the plot on the overview page and adds an entry to the Journal.

From then on, new rows in the pristine-relaxation, UEP, and muon-relaxation
workflows default to the adopted functional, cutoff, and grid; rows you have
configured with your own values keep them. The k-point sweep's own rows pick
up the adopted cutoff. For supercell rows MuFinder rescales the grid to keep
the same reciprocal-space sampling density.

The ``Project Defaults`` dialog (``Edit…`` on the strip) shows, for the cutoff
and the grid, the method, criteria, and converged-with functional, and lets you
change any of the three defaults by hand. The cutoff is entered in eV and the
grid as three positive integers (``4 4 4`` or ``4x4x4``); invalid input is
reported inline and the dialog stays open. A cutoff or grid typed in there is
recorded as ``Set manually``, with no criteria and no functional, does not
change the project functional, and removes the pinned plot from the overview
page. A grid typed in by hand is taken to be for the host unit cell and is
rescaled for supercell rows like an adopted one. Leaving a field blank clears
that default. The functional can also be chosen here before any convergence
test has run, or cleared with ``Not set``.

Stale defaults
~~~~~~~~~~~~~~

Converged values are kept when the project functional changes. If the cutoff
or grid was converged with a functional other than the current project
functional, the strip shows it in warning colour with that functional in
brackets, for example ``700 eV (PBE)``, and the ``Project Defaults`` dialog
says *Converged with PBE, but the project functional is LDA. Re-run the
convergence test to confirm this value.* Re-run the sweep with the new
functional and adopt the new result. Values set manually, and values whose
functional was not recorded, are never flagged.

Using the grid as a k-point spacing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pristine-relaxation, muon-relaxation, and UEP parameter dialogs show
``K-Point MP Grid`` and ``K-Point MP Spacing`` as a pair of mutually exclusive
rows with a radio button each; only the selected one is written. When you
select ``K-Point MP Spacing``, MuFinder pre-fills it with the spacing that
reproduces the project default grid (the same value as in the examiner's
recommendation line). A grid adopted from the examiner is recorded against
the cell the sweep ran on, and a grid typed into the ``Project Defaults``
dialog against the host unit cell, so the spacing is pre-filled in both cases.
No spacing is pre-filled for a ``1 x 1 x 1`` default.

Limitations
-----------

- Each sweep varies one setting with everything else fixed. The cutoff sweep
  uses CASTEP's own k-point sampling unless you set one, and the k-point sweep
  uses the project cutoff (or ``FINE`` basis precision), so check that the
  adopted pair is still converged if either changes substantially.
- Every point is a single-point calculation on the unrelaxed cell; there is no
  muon in the cell.
- The examiner normalizes energies by the number of atoms in the cell the
  sweep ran on, and the k-point examiner (and the overview plot, which reuses
  its analysis) computes spacings from that cell's lattice, whether it is the
  unit cell or a saved supercell.
- The pristine-relaxation, UEP, and muon-relaxation workflows rescale an
  adopted grid from the cell it was converged on to each row's cell at
  equivalent k-point density. A unit-cell row uses the grid exactly, unless the
  grid was converged on a supercell of that unit cell, in which case it is
  rescaled too. When the jobs are generated, one dialog summarises the
  adjustments: rows sharing a cell and a grid change share one line with a row
  count, at most five lines are listed, and Show Details lists every row.
- The k-point examiner places grids by their number of k-points, so two grids
  with the same product ``a x b x c`` (possible with grids entered by hand)
  cannot both be examined. It keeps one, preferring a grid with a completed
  result and then the one that samples the cell most uniformly, and names the
  grid it left out under ``Notes`` in the results.
- Convergence is judged from successive differences only; there is no
  check that later points stay within the threshold.
