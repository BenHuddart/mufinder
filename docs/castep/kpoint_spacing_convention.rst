K-point Spacing Convention
==========================

CASTEP accepts a Monkhorst-Pack sampling either as an explicit
``KPOINT_MP_GRID`` or as a ``KPOINT_MP_SPACING`` in 1/Ang from which it derives
the grid. MuFinder converts in both directions in
``mufinder.science.kpoint_grids`` and guarantees that every spacing it reports
for a grid reproduces that grid when written to a ``.cell`` file.

Throughout, ``|b_i|`` is the length of reciprocal lattice vector ``i``
including the factor of 2π, ``s`` is a spacing in 1/Ang and ``N_i`` is the grid
count along axis ``i``.

Spacing to grid
---------------

CASTEP derives the grid in the subroutine ``cell_kpoints_spacing_wrapped``
(``Source/Fundamental/cell.f90`` in the CASTEP source) as the minimum grid that
meets the spacing, plus one:

.. code-block:: text

   N_i = floor( |b_i| / (2π · s) ) + 1

``grid_from_spacing(lattice_vectors, spacing)`` uses the same formula, so its
result matches CASTEP exactly. When ``s`` comes from ``spacing_from_grid`` the
ratio ``|b_i| / (2π · s)`` is mathematically an integer, but two-step IEEE 754
division can land just below it (6.999…9991 instead of 7) and ``floor`` would
then drop a k-point. ``grid_from_spacing`` adds ``1e-9`` inside the ``floor``:
large enough to absorb rounding error, far too small to advance a genuinely
non-integer ratio. A non-positive or non-finite spacing raises ``ValueError``.

The formula is piecewise constant in ``s``. Axis ``i`` changes its count only at
the breakpoints ``s = |b_i| / (2π · n)`` for integer ``n``, so CASTEP reproduces
a grid ``(N_1, N_2, N_3)`` exactly for every spacing in the half-open range

.. code-block:: text

   ( max_i |b_i| / (2π · N_i),  min_i |b_i| / (2π · (N_i − 1)) ]

where an axis with ``N_i = 1`` contributes nothing to the upper bound. The
exclusive lower bound is the grid's physical k-point spacing; a value at that
bound gives one more k-point on the axis that attains it.

Grid to spacing
---------------

CASTEP has no grid-to-spacing conversion: ``KPOINT_MP_SPACING`` is only ever
user input. The spacing MuFinder shows for a grid is therefore its own
convention, chosen so that it can be copied into a ``.cell`` file.

``spacing_from_grid(lattice_vectors, grid)`` returns the inclusive upper bound
of the range above,

.. code-block:: text

   s = min_i |b_i| / (2π · (N_i − 1))   over axes with N_i ≥ 2

which is the largest ``KPOINT_MP_SPACING`` for which CASTEP reproduces the
grid. It raises ``ValueError`` for ``(1, 1, 1)``, which every spacing above the
cell's coarsest breakpoint produces, so no single value characterises
Gamma-only sampling, and for a grid with a non-positive count.

``kpoint_count_for_spacing(lattice_vectors, spacing)`` drops the ``floor`` to
give a smooth k-point count, ``∏_i (|b_i| / (2π · s) + 1)``, which agrees with
the grid's k-point count at the breakpoints. It is used only to place spacing
ticks on an axis of k-point counts.

Sweep grids
-----------

``generate_kpoint_grids(lattice_vectors, spacing)`` builds the k-point
convergence sweep. It probes ``grid_from_spacing`` at every breakpoint
``|b_i| / (2π · n)`` from coarse to fine, clamps each result to the grid of the
target spacing, and returns each distinct grid once, from ``(1, 1, 1)`` up to
that cap. Each axis therefore steps independently, following CASTEP's own
spacing curve; for a cubic cell the sweep is the isotropic ramp ``(1, 1, 1)``,
``(2, 2, 2)``, ….

The convention's guarantee is that, for every grid ``g`` other than
``(1, 1, 1)`` returned by ``generate_kpoint_grids``,

.. code-block:: text

   grid_from_spacing(lattice, spacing_from_grid(lattice, g)) == g

A grid that is not on CASTEP's curve, such as one entered by hand, still gets
a spacing from ``spacing_from_grid``, but CASTEP turns that spacing into a
different grid. Running the k-point sweep through ``Generate From Spacing...``
keeps every grid on the curve.

Where spacings appear
---------------------

- **Convergence examiner results.** A k-point recommendation reads
  ``Recommended k-point grid: ... (spacing s 1/Ang)``, with ``s`` from
  ``spacing_from_grid`` on the swept cell (see :doc:`/user_guide/convergence_testing`).
- **Per-grid annotations.** The examiner and the convergence overview label
  each grid with its spacing. ``(1, 1, 1)`` is never labelled, and a grid whose
  spacing equals the previous label (``math.isclose``, ``rel_tol=1e-9``) is
  skipped, so the labels decrease strictly. A run of equal spacings happens
  when one axis sets the upper bound while the others keep growing.
- **Overview snapshots.** The adopted k-point snapshot stored in the project
  carries per-point spacings. Before the overview displays it, the spacings are
  recomputed from the cell the sweep ran on (the host when the snapshot does
  not record one), so the display always follows the current formula. The
  stored snapshot is not rewritten.
- **Examiner spacing axis.** The secondary axis carries round spacing values
  placed with ``kpoint_count_for_spacing``, not per-grid spacings.
- **Configure-dialog defaults.** In the pristine relaxation, muon relaxation
  and UEP configure dialogs, rows set by grid have their spacing editor seeded
  from the project grid and its reference lattice through
  ``spacing_from_grid``, so switching the k-point choice to a spacing starts
  from a value that reproduces the grid. A ``1 x 1 x 1`` grid, one with no
  reference lattice, or a read-only row seeds nothing.

Supercells
----------

A spacing is never carried from one cell to another. Supercell grids come from
``equivalent_grid_for_supercell(reference_lattice, reference_grid,
supercell_lattice)``, which scales each axis count by
``|b_i(supercell)| / |b_i(reference)|`` and rounds up, so the supercell is never
sampled more coarsely than the reference grid.

Key modules
-----------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Module
     - Role
   * - ``mufinder.science.kpoint_grids``
     - ``grid_from_spacing``, ``spacing_from_grid``, ``generate_kpoint_grids``,
       ``kpoint_count_for_spacing``, ``equivalent_grid_for_supercell``.
   * - ``mufinder.app.convergence_examiner``
     - Per-grid spacing labels, snapshot spacing recomputation and the
       spacing axis ticks.
   * - ``mufinder.app.workflow_table_utils``, ``mufinder.app.workflow_row_widget``
     - ``SupercellColumn.suggested_spacing`` and the spacing default in the
       configure dialogs.
   * - ``mufinder.app.kpoint_convergence_widget``
     - ``Generate From Spacing...`` rows of the k-point sweep, through
       ``generate_kpoint_grids``.
   * - ``tests/science/test_kpoint_grids.py``
     - The round-trip, sweep and floating-point cases above, including
       hexagonal AlF\ :sub:`3` and cubic LiF checks.
