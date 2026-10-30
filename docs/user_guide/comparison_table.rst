Generate Comparison Table
=========================

The comparison table checks how well a DFT-relaxed host reproduces the
experimental structure you started from. In ``Pristine Relaxation``, click
``Generate Comparison Table…`` to compare the project's active
(experimental) host structure against one or more completed pristine-relaxation
structures. MuFinder normalizes the cells into a common frame, matches atoms
under periodic boundary conditions, and reports the lattice parameters, the
displacements of the matched atoms and the space groups. The
comparison is a supercell-aware structural analysis, not a row-by-row text diff
of imported files.

Before you start
----------------

The comparison table needs:

- **A loaded host structure.** The table compares against the active host.
- **At least one completed pristine relaxation.** Run the jobs in ``Pristine
  Relaxation`` first (:doc:`/user_guide/structural_relaxation_analysis`; jobs
  are described in :doc:`/getting_started/running_jobs`). MuFinder reports which
  of the two is missing when you click the button.

Running the workflow
--------------------

1. **Open the page.** Select ``Pristine Relaxation`` in the navigator.
2. **Start the comparison.** Click ``Generate Comparison Table…``.
3. **Choose the relaxations.** In the ``Generate Comparison Table`` dialog, tick
   ``Include`` for one or more completed pristine-relaxation jobs. Each row
   shows the job's ``Label``, ``Supercell``, ``Status``, ``Total Energy (eV)``,
   and relaxed ``a (Ang)``, ``b (Ang)`` and ``c (Ang)`` lengths. Ticking nothing
   is refused.
4. **Name the table.** Edit the ``Comparison table label`` if needed (it
   defaults to ``Comparison <N>``) and click ``OK``.

   .. figure:: ../_generated/screenshots/comparison_job_dialog.webp
      :alt: The Generate Comparison Table dialog with one completed CoF2
            pristine relaxation ticked and its relaxed lattice lengths.

      ``Generate Comparison Table`` with the CoF₂ pristine relaxation ticked.
5. **Read the table.** MuFinder opens the new table, listed under ``Pristine
   Relaxation`` in the project tree. It is saved with the project.
6. **Rename or delete it.** The title at the top of an existing table stays
   editable: editing it updates the table's label in the project tree and is
   saved with the project. The ``Delete`` button at the top right deletes the
   table after a confirmation prompt, with ``Cancel`` as the default.
7. **Export it (optional).** Click ``Export CSV…`` (see `Export`_).

The comparison table
--------------------

.. figure:: ../_generated/screenshots/comparison_table.webp
   :alt: A comparison table of the CoF2 host against one relaxed cell, grouped into
         lattice, displacement and symmetry metrics.

   The CoF₂ host of :doc:`/user_guide/examples/cof2` compared with a synthetic
   pristine relaxation.

The table has a ``Metric`` column, an ``Experimental`` column for the reference
structure, and one column per selected DFT structure, headed ``DFT (<job
label>)``. Rows are grouped into three categories:

- ``Lattice parameters``: ``a``, ``b``, ``c`` (Å), ``α``, ``β``, ``γ`` (°) and
  the cell volume ``V`` (Å³). Each DFT cell shows the value with its change
  against experiment, such as ``4.7091 (+0.30 %)``.
- ``Atomic displacements``: the ``RMSD`` and the ``Max displacement`` of the
  matched atoms (Å).
- ``Symmetry``: the ``Space group`` of each structure, such as
  ``P4₂/mnm (136)``, and whether the DFT structure keeps the experimental one
  (``Same space group``).

Hovering over a metric name shows a one-line description, and ``About these
metrics`` above the table opens this page. When two or more DFT columns have a
numeric value for a metric, the column closest to experiment is shown in bold
green. A warning banner above the table lists problems found while comparing,
such as a failed supercell detection or a partial atom matching.

The collapsible ``DFT Parameter Differences`` section lists only those CASTEP
input parameters that differ across the selected pristine-relaxation jobs.

A saved table that holds metrics outside this list opens with the rows it
shares with the list and leaves the others out; generating the table again
fills in every row.

Interpreting the table
~~~~~~~~~~~~~~~~~~~~~~

1. Check the warnings for failed supercell detection or partial atom matching.
2. Read the lattice changes: a relaxation with a converged functional usually
   lands within a percent or two of experiment.
3. Read the RMSD and maximum displacement for the internal coordinates; a
   maximum displacement much larger than the RMSD points to one site that moved.
4. Check that the space group is kept. A change of space group means the
   relaxation broke the experimental symmetry.

Method
------

The comparison runs these steps:

1. normalizes the experimental and DFT cells into a common comparison frame,
2. matches atoms under periodic boundary conditions,
3. evaluates the lattice, displacement and space-group metrics.

Notation
~~~~~~~~

The equations below use the following symbols.

- :math:`L_{\mathrm{exp}}`, :math:`L_{\mathrm{DFT}}`: experimental and DFT
  direct-lattice matrices.
- :math:`\mathbf{a}`, :math:`\mathbf{b}`, :math:`\mathbf{c}`: direct-lattice
  basis vectors of the comparison cell.
- :math:`\mathbf{f}^{\mathrm{exp}}_i`, :math:`\mathbf{f}^{\mathrm{DFT}}_j`:
  fractional coordinates of atoms in the experimental and DFT structures.
- :math:`m(i)`: DFT atom matched to experimental atom :math:`i`.
- :math:`N_{\mathrm{match}}`: number of matched atom pairs.
- :math:`\mathbf{1}[\cdot]`: indicator function, equal to 1 when the condition
  is true and 0 otherwise.

Normalization and Atom Matching
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Supercell detection
^^^^^^^^^^^^^^^^^^^

MuFinder first tries to detect whether the DFT cell is an integer supercell of
the experimental cell. It forms the candidate matrix

.. math::

  M^* = L_{\mathrm{DFT}} L_{\mathrm{exp}}^{-1}

and compares it with the nearest integer matrix

.. math::

  M = \operatorname{round}(M^*).

The match residual is measured as

.. math::

  \varepsilon = \lVert M^* - M \rVert_F.

If :math:`\varepsilon \le 0.2` and :math:`\det(M) > 0`, the relation is
accepted as a valid supercell mapping. When :math:`\det(M) > 1`, the
experimental structure is tiled into that supercell before any other metrics
are computed. This prevents artificial disagreement caused solely by comparing
different cell sizes.

Periodic atom matching
^^^^^^^^^^^^^^^^^^^^^^

After normalization, atoms are matched species-by-species with a Hungarian
assignment. The pairwise cost between experimental atom :math:`i` and DFT atom
:math:`j` is the minimum-image Cartesian displacement

.. math::

  \Delta \mathbf{f}_{ij} = \mathbf{f}^{\mathrm{DFT}}_j - \mathbf{f}^{\mathrm{exp}}_i

.. math::

  d_{ij} = \left\lVert
  \left(\Delta \mathbf{f}_{ij} - \operatorname{round}(\Delta \mathbf{f}_{ij})\right)
  L_{\mathrm{DFT}}
  \right\rVert_2.

This minimum-image construction is the correct periodic-boundary-condition
analogue of a direct Cartesian distance. Matching separately within each
species block prevents, for example, Si atoms from being reassigned to O sites.

If exact one-to-one matching is impossible, MuFinder keeps the best partial
correspondence and records a warning in the comparison table.

Lattice Metrics
~~~~~~~~~~~~~~~

The lattice metrics quantify cell shape and size after normalization.

Lengths
^^^^^^^

The direct-lattice lengths are

.. math::

  a = \lVert \mathbf{a} \rVert_2,
  \qquad
  b = \lVert \mathbf{b} \rVert_2,
  \qquad
  c = \lVert \mathbf{c} \rVert_2.

Physically, these track axial dilation or contraction of the relaxed DFT cell
relative to experiment.

Angles
^^^^^^

The lattice angles are obtained from the usual dot-product relations,

.. math::

  \alpha = \cos^{-1}\left(
  \frac{\mathbf{b}\cdot\mathbf{c}}{\lVert \mathbf{b} \rVert_2 \lVert \mathbf{c} \rVert_2}
  \right),

.. math::

  \beta = \cos^{-1}\left(
  \frac{\mathbf{a}\cdot\mathbf{c}}{\lVert \mathbf{a} \rVert_2 \lVert \mathbf{c} \rVert_2}
  \right),

.. math::

  \gamma = \cos^{-1}\left(
  \frac{\mathbf{a}\cdot\mathbf{b}}{\lVert \mathbf{a} \rVert_2 \lVert \mathbf{b} \rVert_2}
  \right).

These respond primarily to shear and cell-shape distortion rather than pure
volumetric expansion.

Volume
^^^^^^

The cell volume is

.. math::

  V = |\det L|.

Large deviations often indicate over-relaxation, under-relaxation, or a true
change in the equilibrium cell size.

Changes relative to experiment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each DFT lattice value is followed by its signed percent change against the
experimental value,

.. math::

  \Delta x_{\%} = 100\,\frac{x_{\mathrm{DFT}} - x_{\mathrm{exp}}}{x_{\mathrm{exp}}},
  \qquad x \in \{a, b, c, \alpha, \beta, \gamma, V\}.

Values close to zero indicate good lattice agreement. Positive values indicate
expansion relative to experiment, while negative values indicate contraction.
When the DFT structure is a supercell, its lattice values are those of the
supercell and the change is taken against the experimental cell tiled to that
supercell.

Atomic-Displacement Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the atom mapping is fixed, each matched pair contributes a minimum-image
displacement magnitude :math:`d_i`.

RMSD
^^^^

The root-mean-square displacement is

.. math::

  \mathrm{RMSD} = \sqrt{\frac{1}{N_{\mathrm{match}}}
  \sum_{i=1}^{N_{\mathrm{match}}} d_i^2}.

This is the main summary of average structural drift at the atomic level.

Maximum displacement
^^^^^^^^^^^^^^^^^^^^

.. math::

  d_{\max} = \max_i d_i.

The maximum displacement is useful for spotting one strongly reconstructed
site inside an otherwise well-preserved structure.

Symmetry
~~~~~~~~

Space-group comparison
^^^^^^^^^^^^^^^^^^^^^^

MuFinder uses ``spglib`` to identify the international space-group number and
symbol for each structure; the ``Space group`` row shows both, with screw axes
written as subscripts (``P4_2/mnm`` as ``P4₂/mnm``). The ``Same space group``
row compares the detected group numbers of the normalized structures,

.. math::

  \mathrm{match}_{\mathrm{SG}} = \mathbf{1}\left[n_{\mathrm{exp}} = n_{\mathrm{DFT}}\right].

This is a compact crystallographic summary, but it should be interpreted with
the usual care: the result depends on the symmetry tolerance, 0.001 Å as for
every MuFinder symmetry analysis, and on the numerical quality of the input
structure.

Export
------

Use ``Export CSV…`` in the comparison view to export the displayed values for
external analysis or record-keeping.

Known limitations
-----------------

- Space-group comparison depends on ``spglib`` and on the symmetry tolerance.
- Large supercells increase the cost of the species-resolved Hungarian
  assignment.
