Convergence criteria
====================

This page is the reference for how the examiner dialog of
:doc:`/user_guide/convergence_testing` judges a completed sweep: the thresholds,
the criterion, the recommendation, the sweep-length check of a cutoff sweep, how
to read the results and plot, and how an adopted grid is used as a k-point
spacing. The same rules apply to the ``Cutoff Convergence`` and ``K-Point
Convergence`` sweeps unless a section says otherwise.

Thresholds
----------

The ``Convergence Criteria`` box of the examiner holds one threshold per
quantity:

.. list-table::
   :header-rows: 1
   :widths: 34 34 10 12 10

   * - Field
     - Tests the difference in
     - Default
     - Minimum
     - Maximum
   * - ``Energy threshold (eV/atom)``
     - total energy per atom
     - 0.001
     - 1e-12
     - 1
   * - ``Force threshold (eV/Ang)``
     - largest force on any atom
     - 0.05
     - 0.00001
     - 1
   * - ``Stress threshold (GPa)`` (optional)
     - largest absolute component of the stress tensor
     - 0.1
     - 0.000001
     - 100

The force and stress minimums are the precision CASTEP prints those quantities
to: a finer threshold could only be met by values that print identically. The
energy minimum only keeps the threshold positive and is not a precision: CASTEP
prints energies to 13 significant figures, so what can be resolved per atom depends
on the size of the total energy and the number of atoms. When you click ``Analyse
Convergence``, a value outside a field's range snaps to the nearest limit, and the
field shows the value the analysis used.

The stress default matches the usual ``GEOM_STRESS_TOL``. The stress box can only
be ticked when every completed job reported a stress, that is, when every job was
run with ``Calculate Stress`` on.

The criterion
-------------

The analysis uses every completed point that has both an energy and a force; at
least two are needed. The k-point examiner places grids by their number of
k-points, so of two grids with the same product ``a x b x c`` it keeps one and names
the other under ``Notes`` (see the limitations in
:doc:`/user_guide/convergence_testing`). Points are sorted by cutoff, or by number of k-points, and
every quantity is judged against the **finest point of the sweep**, the highest
cutoff or the densest grid:

- **Energy.** The difference from the finest point's total energy, divided by
  the number of atoms, is compared with the energy threshold.
- **Force.** The difference from the finest point's maximum force is compared
  with the force threshold.
- **Stress (optional).** The same test on the maximum stress, in GPa.

A quantity is converged at the first point from which it, and every finer point,
lies within the threshold of the finest point. The finest point is the reference,
so it is never converged on its own: a finer point must exist. A point that agrees
with the reference by accident does not count if a finer point lies outside the
threshold again. Each point's distance from the reference does not depend on its
neighbours, so a smaller step does not shrink the quantity being tested.

Per-atom normalisation
~~~~~~~~~~~~~~~~~~~~~~

Only the energy is normalised. The number of atoms is that of the cell the sweep
ran on, the unit cell or the saved supercell, and the examiner's opening summary
states it. Energy thresholds therefore carry over between cells of different
sizes. Forces and stresses are intensive and are compared as they are.

The recommendation
------------------

The recommendation (``Recommended cutoff`` or ``Recommended k-point grid``) is the
largest of the converged values, so the adopted setting satisfies every criterion.
Every criterion in use must converge: energy, force, and stress when it is
included.

If any does not converge within the sweep the recommendation reads ``none``, the
adopt button stays disabled, and the row says which criterion failed, for example
*the force criterion did not converge within the sweep*. Extend the sweep to larger
cutoffs or denser grids, or relax the threshold, and analyse again. A k-point
recommendation restored from a saved analysis whose k-point count no longer has a
grid in the sweep cannot be adopted either; its button stays disabled until you
analyse again.

The sweep-length check
----------------------

The finest cutoff is only the best answer the sweep has, so a sweep that stops too
early gives an optimistic recommendation. For a cutoff sweep the examiner therefore
fits

.. math::

   E(E_\text{cut}) = E_\infty + c\,E_\text{cut}^{\,m}

to the energies, with a free exponent :math:`m` between −20 and −0.5,
:math:`c \ge 0` and :math:`E_\infty` no higher than the lowest energy. This is the
power-law form CASTEP's own finite-basis-set correction assumes.

The fit starts from all the cutoffs and drops the lowest one at a time, down to the
four highest, until it reaches a run that passes every test:

- the energy does not rise between neighbours by more than a quarter of the energy
  threshold per atom;
- the fitted exponent is not at either bound;
- the RMS residual is within a quarter of the energy threshold per atom.

The ``Sweep length`` row then says:

- ``long enough`` when the finest cutoff is within the energy threshold of the
  fitted limit :math:`E_\infty`;
- ``too short`` when it is not, with the cutoff at which the fitted curve reaches
  the threshold, rounded up to a multiple of the sweep's last step: *extend the
  sweep to 1300 eV and analyse again*.

If no run passes, or the sweep has fewer than four cutoffs, the row reads ``limit
not estimated`` with the reason: fewer than four cutoffs, an energy that does not
fall steadily towards the finest cutoff, a fit that did not converge, a fit whose
exponent sits at a bound (*the decay is not a power law*), or one that does not
follow the points closely enough. Real CASTEP sweeps often wobble by a fraction of
a meV per atom, so this is common once a sweep has converged.

The sweep-length check never changes the recommendation and never blocks adoption;
it says whether the reference can be trusted.

K-point sweeps
~~~~~~~~~~~~~~

K-point sweeps get no fit: convergence with k-point sampling oscillates rather
than decaying smoothly, so there is no limit to extrapolate. The densest grid is
the reference, so extend a k-point sweep beyond the grid you want to certify.

Reading the results
-------------------

The results are a list of labelled rows, grouped by blank lines:

.. code-block:: text

   Thresholds          energy 0.001 eV/atom, force 0.05 eV/Ang
   Reference           800 eV, the finest cutoff in the sweep

   Energy              converged at 600 eV
                       within 0.00082 eV/atom of 800 eV, as is every finer cutoff
   Force               converged at 600 eV
                       within 0.023 eV/Ang of 800 eV, as is every finer cutoff

   Sweep length        long enough
                       800 eV is within 0.00001 eV/atom of the fitted limit

   Recommended cutoff  600 eV

The first line of each row is the result and the second explains it. A ``Stress``
row follows ``Force`` when stress is included, and a ``Notes`` row ends the list
when the sweep has caveats. A k-point sweep has no ``Sweep length`` row, names
grids rather than cutoffs (``converged at 4 x 4 x 6 (spacing 0.05 1/Ang)``), and
adds under ``Reference`` that a sweep should extend beyond the grid it is meant to
certify. A ``too short`` sweep length and a recommendation of ``none`` are shown in
the warning colour.

The plot
~~~~~~~~

The plot has an energy panel above a force panel, with the value at which each
criterion converged and the recommended value marked. The energy panel shows the
total energy of the cell, so the per-atom energy threshold is not drawn on it. The
force threshold is a horizontal line at its own value, while the criterion compares
each point's difference from the finest point with it, so the result rows, not the
line, give the verdict. When stress is used it is drawn on a second axis of the
force panel with its threshold, in the same way.

On a cutoff plot the energy panel also draws the power-law fit of the sweep-length
check over the cutoffs it used, and the fitted limit as a dotted line. For a
k-point sweep the horizontal axis is the ``Number of k-points``, the product
``a x b x c`` of the grid divisions, on a logarithmic scale; a second axis along the
top marks round ``K-point spacing (1/Ang)`` values at the matching positions. The
recommended grid is named in the energy panel's legend; the other grids are named
only in the tooltip.

Hover a point for its details: its value on a cutoff plot, its grid on a k-point
plot.

Using a grid as a k-point spacing
---------------------------------

For a k-point recommendation the result line also gives the grid's spacing,
``(spacing s 1/Ang)``. For a grid that CASTEP's spacing rule can produce on the swept
cell, which includes every grid from ``Generate From Spacing…``, this is the largest
``KPOINT_MP_SPACING`` that reproduces the grid, so it can be copied into a ``.cell``
file. A grid entered by hand may not be one the rule can produce; it still shows a
spacing, but CASTEP turns that spacing into a different grid. ``1 x 1 x 1`` has no
such spacing and is shown without one, as is a grid whose spacing equals that of the
next coarser grid in the sweep. The convention, and its round-trip guarantee, are
described in :doc:`/castep/kpoint_spacing_convention`.

The pristine-relaxation, muon-relaxation, and UEP parameter dialogs show ``K-Point
MP Grid`` and ``K-Point MP Spacing`` as a pair of mutually exclusive rows with a
radio button each; only the selected one is written. When you select ``K-Point MP
Spacing``, MuFinder pre-fills it with the spacing computed from the project default
grid, by the same rule as the examiner's recommendation line, so a default that
spacing cannot reproduce gives a different grid. A grid
adopted from the examiner is recorded against the cell the sweep ran on, and a grid
typed into the ``Project Defaults`` dialog against the host unit cell, so the
spacing is pre-filled in both cases. No spacing is pre-filled for a ``1 x 1 x 1``
default.
