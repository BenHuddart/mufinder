Structure Comparison
====================

This page is the method behind the comparison table: how MuFinder puts an
experimental and a DFT structure into a common frame, matches their atoms, and
evaluates the lattice, displacement and space-group metrics. Generating and reading
a table is described under :ref:`pristine-comparison-table`.

The equations use :math:`L_{\mathrm{exp}}` and :math:`L_{\mathrm{DFT}}` for the
experimental and DFT direct-lattice matrices, whose rows are the lattice vectors.

Supercell detection
-------------------

MuFinder first tries to detect whether the DFT cell is an integer supercell of
the experimental cell. It forms the candidate matrix and its nearest integer matrix,

.. math::

  M^* = L_{\mathrm{DFT}} L_{\mathrm{exp}}^{-1},
  \qquad
  M = \operatorname{round}(M^*),

and measures the match residual

.. math::

  \varepsilon = \lVert M^* - M \rVert_F.

If :math:`\varepsilon \le 0.2` and :math:`\det(M) > 0`, the relation is
accepted as a valid supercell mapping. When :math:`\det(M) > 1`, the
experimental structure is tiled into that supercell before the atoms are matched,
which prevents artificial disagreement caused solely by comparing different cell
sizes.

Periodic atom matching
----------------------

After normalization, atoms are matched within each element with a Hungarian
assignment, so an atom is only matched to an atom of the same element. The element
is read from the species label: a CASTEP label such as ``Co:1`` or ``Co:2``, a site
label such as ``Co1`` and the bare symbol ``Co`` all name cobalt, so a relaxed
structure whose species carry spin or pseudopotential labels matches its unlabelled
host. Labels separate atoms in two cases only. A muon (``H:mu``, ``mu``, ``muon``
or ``X``) is matched only to a muon, never to a hydrogen atom of the host. A label
that names no element is matched only to the same label. Two labels of one element,
such as the ``Co:1`` and ``Co:2`` sublattices of an antiferromagnet, are matched
together, so an atom pairs with the nearest atom of its element whatever its label.

The pairwise cost between experimental atom :math:`i` and DFT atom :math:`j`, with
fractional coordinates :math:`\mathbf{f}^{\mathrm{exp}}_i` and
:math:`\mathbf{f}^{\mathrm{DFT}}_j`, is the Cartesian distance to the nearest
periodic image in the DFT lattice,

.. math::

  \Delta \mathbf{f}_{ij} = \mathbf{f}^{\mathrm{DFT}}_j - \mathbf{f}^{\mathrm{exp}}_i,
  \qquad
  d_{ij} = \min_{\mathbf{n} \in \mathbb{Z}^3}
  \left\lVert (\Delta \mathbf{f}_{ij} + \mathbf{n}) L_{\mathrm{DFT}} \right\rVert_2.

The minimum is found in a reduced basis of the lattice and over its neighbouring
cells, so it is the true nearest image in a hexagonal, monoclinic or otherwise
skewed cell. Folding each fractional component into :math:`[-1/2, 1/2]` alone is
exact only in an orthogonal cell: in a hexagonal cell with :math:`a = 4` Å, a
difference of (0.45, 0.55, 0) folds to a 3.12 Å vector, while its nearest image is
2.03 Å away.

If exact one-to-one matching is impossible, MuFinder keeps the best partial
correspondence and records a warning in the comparison table. The table also warns
when the mean matched displacement exceeds 2.5 Å, because the atom mapping may then
be unreliable.

Lattice parameters
------------------

``a``, ``b`` and ``c`` are the lengths of the lattice vectors; ``α`` is the angle
between ``b`` and ``c``, ``β`` between ``a`` and ``c``, and ``γ`` between ``a`` and
``b``; ``V`` is the absolute value of the determinant of the lattice matrix. All are
read per host cell. Each DFT value is followed by its signed
percent change against the experimental value,

.. math::

  \Delta x_{\%} = 100\,\frac{x_{\mathrm{DFT}} - x_{\mathrm{exp}}}{x_{\mathrm{exp}}}.

A positive change is an expansion relative to experiment and a negative change a
contraction.

.. _comparison-supercell-columns:

Supercell columns
~~~~~~~~~~~~~~~~~

When the DFT cell is a supercell of the host, :math:`L_{\mathrm{DFT}} \approx
M L_{\mathrm{exp}}` with :math:`\det(M) > 1`, its lattice rows are reduced to
the host cell before the parameters are read,

.. math::

  L_{\mathrm{host}} = M^{-1} L_{\mathrm{DFT}},

with :math:`M` the supercell matrix found by `Supercell detection`_. Each row of
:math:`L_{\mathrm{host}}` is a host-cell vector in the experimental setting. For
a diagonal :math:`M`, such as a 2 × 2 × 2 supercell, this divides
:math:`a`, :math:`b` and :math:`c` by the tiling and leaves the angles
unchanged. For any other :math:`M`, such as the
:math:`\sqrt{2} \times \sqrt{2}` cell with rows (1, 1, 0), (−1, 1, 0) and
(0, 0, 1), it gives the host cell that the supercell implies, averaged over its
tiles, so a strain along the host's first lattice vector shows in :math:`a`
alone rather than in the supercell's diagonal vectors. The volume is
:math:`V_{\mathrm{DFT}} / \det(M)`. The experimental column is the host's own
cell, so every column reads like with like, and the closest column is chosen
among the per-host-cell values. The displacement and symmetry rows still use
the supercell, with the host tiled to it.

A DFT cell that is the host cell in another setting (:math:`\det(M) = 1`, such
as its axes listed in another order) is read the same way, along the host's
axes, and has no note in the banner. A column whose supercell relation is not
found keeps its own lattice; the banner says so, and its lattice rows are not
per host cell.

RMSD and maximum displacement
-----------------------------

Once the atom mapping is fixed, each of the :math:`N_{\mathrm{match}}` matched pairs
contributes its displacement magnitude :math:`d_i`, the :math:`d_{ij}` of the pair. The two metrics are

.. math::

  \mathrm{RMSD} = \sqrt{\frac{1}{N_{\mathrm{match}}}
  \sum_{i=1}^{N_{\mathrm{match}}} d_i^2},
  \qquad
  d_{\max} = \max_i d_i.

The RMSD summarises how far the atoms moved on average, and the maximum displacement
picks out one strongly displaced site in an otherwise well-preserved structure.

Space group
-----------

MuFinder uses ``spglib`` to identify the international space-group number and
symbol for each structure; the ``Space group`` row shows both, with screw axes
written as subscripts (``P4_2/mnm`` as ``P4₂/mnm``). The ``Same space group``
row compares the detected group numbers of the normalized structures. The result
depends on the symmetry tolerance, 0.001 Å as for every MuFinder symmetry analysis,
and on the numerical quality of the input structure.
