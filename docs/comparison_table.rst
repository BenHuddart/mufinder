Generate Comparison Table
=========================

Overview
--------

The ``Generate Comparison Table...`` feature in ``Pristine Relaxation`` compares
the project's active (experimental) host structure against one or more completed pristine-relaxation
DFT structures. The comparison is not a direct row-by-row text diff of imported
files. It is a supercell-aware structural analysis pipeline that:

1. normalizes the experimental and DFT cells into a common comparison frame,
2. matches atoms under periodic boundary conditions,
3. evaluates lattice, displacement, bonding, topology, and symmetry metrics,
4. reports a heuristic overall similarity score alongside the raw metrics.

Each generated table is saved as a child item under ``Pristine Relaxation`` in
the project tree and persists across project save/load. The per-metric ``Info`` buttons in the
table expose a condensed version of the same method note documented here.

User Flow
---------

Prerequisites: a loaded host structure and at least one completed
pristine-relaxation job (MuFinder reports which is missing otherwise).

1. Open ``Pristine Relaxation``.
2. Click ``Generate Comparison Table...``.
3. In the dialog, tick ``Include`` for one or more completed pristine-relaxation
   jobs. Each row shows the job's ``Label``, ``Supercell``, ``Status``,
   ``Total Energy (eV)``, and relaxed ``a``/``b``/``c`` lengths.
4. Edit the ``Comparison table label`` if needed (it defaults to
   ``Comparison <N>``) and click ``OK``.
5. MuFinder opens the new table, listed under ``Pristine Relaxation`` in the
   project tree.

The title shown at the top of an existing comparison table remains editable
after creation. Editing the title updates the table's label in the project tree
and is saved with the project.

A table is deleted with the ``Remove`` button at the top right of the
comparison view, after a confirmation prompt.

Table Layout
------------

The rendered table uses metric rows and one column for the experimental
reference plus one column per selected DFT structure. Rows are grouped into
scientific categories:

- lattice parameters,
- lattice changes,
- atomic displacements,
- bonding,
- symmetry,
- overall similarity.

The collapsible ``DFT Parameter Differences`` section lists only those CASTEP
input parameters that differ across the selected pristine-relaxation jobs.

Notation
--------

The equations below use the following symbols.

- :math:`L_{\mathrm{exp}}`, :math:`L_{\mathrm{DFT}}`: experimental and DFT
  direct-lattice matrices.
- :math:`\mathbf{a}`, :math:`\mathbf{b}`, :math:`\mathbf{c}`: direct-lattice
  basis vectors of the comparison cell.
- :math:`\mathbf{f}^{\mathrm{exp}}_i`, :math:`\mathbf{f}^{\mathrm{DFT}}_j`:
  fractional coordinates of atoms in the experimental and DFT structures.
- :math:`m(i)`: DFT atom matched to experimental atom :math:`i`.
- :math:`N_{\mathrm{match}}`: number of matched atom pairs.
- :math:`r_{ij}`: periodic minimum-image Cartesian distance between atoms
  :math:`i` and :math:`j`.
- :math:`R_i`: ASE covalent radius for the species on site :math:`i`.
- :math:`\mathbf{1}[\cdot]`: indicator function, equal to 1 when the condition
  is true and 0 otherwise.

Normalization and Atom Matching
-------------------------------

Supercell detection
~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~

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
---------------

The lattice metrics quantify cell shape and size after normalization.

Lengths
~~~~~~~

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
~~~~~~

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
~~~~~~

The cell volume is

.. math::

  V = |\det L|.

Large deviations often indicate over-relaxation, under-relaxation, or a true
change in the equilibrium cell size.

Percent changes relative to experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The signed percent changes shown in the table are

.. math::

  \Delta a_{\%} = 100\,\frac{a_{\mathrm{DFT}} - a_{\mathrm{exp}}}{a_{\mathrm{exp}}},
  \qquad
  \Delta b_{\%} = 100\,\frac{b_{\mathrm{DFT}} - b_{\mathrm{exp}}}{b_{\mathrm{exp}}},

.. math::

  \Delta c_{\%} = 100\,\frac{c_{\mathrm{DFT}} - c_{\mathrm{exp}}}{c_{\mathrm{exp}}},
  \qquad
  \Delta V_{\%} = 100\,\frac{V_{\mathrm{DFT}} - V_{\mathrm{exp}}}{V_{\mathrm{exp}}}.

Values close to zero indicate good lattice agreement. Positive values indicate
expansion relative to experiment, while negative values indicate contraction.

Atomic-Displacement Metrics
---------------------------

Once the atom mapping is fixed, each matched pair contributes a minimum-image
displacement magnitude :math:`d_i`.

RMSD
~~~~

The root-mean-square displacement is

.. math::

  \mathrm{RMSD} = \sqrt{\frac{1}{N_{\mathrm{match}}}
  \sum_{i=1}^{N_{\mathrm{match}}} d_i^2}.

This is the main summary of average structural drift at the atomic level.
Because the square is taken before averaging, RMSD weights larger relaxations
more strongly than the simple mean displacement.

Mean and maximum displacement
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The other two displacement summaries are

.. math::

  \bar{d} = \frac{1}{N_{\mathrm{match}}}\sum_{i=1}^{N_{\mathrm{match}}} d_i,
  \qquad
  d_{\max} = \max_i d_i.

The mean displacement is less sensitive to a single outlier than RMSD, while
the maximum displacement is useful for spotting one strongly reconstructed site
inside an otherwise well-preserved structure.

Bonding Metrics
---------------

Bonding-derived metrics use a simple, explicit neighbour rule. Two atoms are
treated as bonded when

.. math::

  r_{ij} \le 1.3\,(R_i + R_j).

This is a heuristic coordination criterion based on ASE covalent radii. It is
transparent and deterministic, but it is still a model choice rather than a
universal physical law. The same rule is used for coordination numbers,
topology, and local bond-angle environments.

Nearest-neighbour bond-length changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each site,

.. math::

  r_i^{\mathrm{NN}} = \min_{j \ne i} r_{ij}.

The mean and maximum nearest-neighbour bond-length changes are then

.. math::

  \Delta r_{\mathrm{mean}} = \frac{1}{N_{\mathrm{match}}}
  \sum_{i=1}^{N_{\mathrm{match}}}
  \left| r_{m(i)}^{\mathrm{NN,DFT}} - r_i^{\mathrm{NN,exp}} \right|,

.. math::

  \Delta r_{\max} = \max_i
  \left| r_{m(i)}^{\mathrm{NN,DFT}} - r_i^{\mathrm{NN,exp}} \right|.

These are local-scale metrics. They are often more chemically intuitive than a
cell-wide volume change when the main relaxation is confined to one region.

Bond-angle changes
~~~~~~~~~~~~~~~~~~

For each site with at least two neighbours, MuFinder computes all unique
neighbour-neighbour angles and averages them to obtain a local mean angle

.. math::

  \bar{\theta}_i = \frac{1}{M_i}
  \sum_{(j,k) \in \mathcal{P}_i}
  \cos^{-1}\left(
  \frac{\mathbf{r}_{ij}\cdot\mathbf{r}_{ik}}{\lVert \mathbf{r}_{ij} \rVert_2 \lVert \mathbf{r}_{ik} \rVert_2}
  \right),

where :math:`\mathcal{P}_i` is the set of unique neighbour pairs around site
:math:`i` and :math:`M_i = |\mathcal{P}_i|`. The reported table value is the
mean absolute change over all matched sites that have valid local means in both
structures,

.. math::

  \Delta \theta_{\mathrm{mean}} = \frac{1}{N_{\theta}}
  \sum_i \left| \bar{\theta}_{m(i)}^{\mathrm{DFT}} - \bar{\theta}_i^{\mathrm{exp}} \right|.

This metric is informative for coordination-polyhedron distortions, but it is
less stable when a site has very low coordination.

Coordination-number changes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The coordination number of site :math:`i` is

.. math::

  \mathrm{CN}_i = \sum_{j \ne i} \mathbf{1}\left[r_{ij} \le 1.3\,(R_i + R_j)\right].

The table reports the number of matched sites whose coordination numbers differ
between experiment and DFT,

.. math::

  N_{\Delta \mathrm{CN}} = \sum_{i=1}^{N_{\mathrm{match}}}
  \mathbf{1}\left[\mathrm{CN}_i^{\mathrm{exp}} \ne \mathrm{CN}_{m(i)}^{\mathrm{DFT}}\right].

This is one of the clearest discrete indicators of bond breaking or bond
formation.

Symmetry and Topology
---------------------

Space-group comparison
~~~~~~~~~~~~~~~~~~~~~~

MuFinder uses ``spglib`` to identify the international space-group number and
symbol for each normalized structure. The comparison-table row is a Boolean
match on the detected group numbers,

.. math::

  \mathrm{match}_{\mathrm{SG}} = \mathbf{1}\left[n_{\mathrm{exp}} = n_{\mathrm{DFT}}\right].

This is a compact crystallographic summary, but it should be interpreted with
the usual care: the result depends on the chosen symmetry tolerance and on the
numerical quality of the input structure.

Topology preservation
~~~~~~~~~~~~~~~~~~~~~

Using the same neighbour rule as above, MuFinder builds an undirected bond-edge
set

.. math::

  E = \{(i,j) : r_{ij} \le 1.3\,(R_i + R_j)\}.

The experimental bond graph is mapped through the atom matching and compared
with the DFT bond graph restricted to matched atoms. The table reports whether
the mapped edge sets are exactly identical,

.. math::

  \mathrm{preserved}_{\mathrm{topo}} =
  \mathbf{1}\left[E_{\mathrm{exp}}^{\mathrm{mapped}} = E_{\mathrm{DFT}}^{\mathrm{matched}}\right].

This is stricter than a coordination-number comparison. Two structures can have
the same coordination counts but different bond connectivity.

Similarity Score
----------------

The ``Similarity score`` is a heuristic composite, intended as a quick ranking
aid rather than a replacement for the raw metrics.

First, MuFinder forms a lattice-agreement term from the mean absolute percent
change of :math:`a`, :math:`b`, and :math:`c`:

.. math::

  \Delta_{\mathrm{lat}} = \frac{1}{3}
  \sum_{x \in \{a,b,c\}} \left| \Delta x_{\%} \right|,
  \qquad
  S_{\mathrm{lat}} = \max\left(0, 1 - \frac{\Delta_{\mathrm{lat}}}{100}\right).

Second, it defines a displacement-agreement term using RMSD and the mean
experimental nearest-neighbour distance as a characteristic length scale:

.. math::

  \sigma = \left\langle r_i^{\mathrm{NN,exp}} \right\rangle,
  \qquad
  S_{\mathrm{disp}} = \exp\left(-\frac{\mathrm{RMSD}}{\max(\sigma, 10^{-8})}\right).

Third, it uses the fraction of matched sites whose coordination numbers are
preserved,

.. math::

  S_{\mathrm{bond}} = f_{\mathrm{coord}}.

The final score is

.. math::

  S = \min\left(
  1,
  \max\left(
  0,
  S_{\mathrm{lat}}^{0.35}
  S_{\mathrm{disp}}^{0.35}
  S_{\mathrm{bond}}^{0.30}
  + 0.1\,\mathbf{1}[n_{\mathrm{exp}} = n_{\mathrm{DFT}}]
  \right)
  \right).

Important details:

- the score is clipped to the interval :math:`[0, 1]`,
- the space-group match contributes a fixed bonus of ``0.1``,
- topology and bond-angle metrics are informative diagnostics but do not enter
  the score directly in the current implementation.

Interpreting the Table
----------------------

The most reliable reading strategy is usually:

1. check the warnings for failed supercell detection or partial atom matching,
2. inspect lattice changes and RMSD to judge global agreement,
3. inspect coordination and topology rows to detect chemistry-changing events,
4. use the similarity score only as a compact summary after the raw metrics have been reviewed.

In other words, a high similarity score is useful, but a preserved topology and
small RMSD are usually more physically interpretable than the score alone.

Architecture Notes
------------------

The implementation is intentionally layered:

- normalization in ``mufinder.science.supercell_matching``,
- atom matching in ``mufinder.science.atom_matching``,
- bonding and topology analysis in ``mufinder.science.bonding_analysis`` and
  ``mufinder.science.symmetry_analysis``,
- metric registry and evaluation in ``mufinder.science.structure_metrics``,
- orchestration in ``mufinder.science.structure_comparison``,
- rendering in ``mufinder.app.comparison_table_widget``.

The per-metric help shown by the GUI ``Info`` buttons is sourced from the same
metric-definition registry used to populate the comparison table.

Export
------

Use ``Export CSV...`` in the comparison view to export the displayed values for
external analysis or record-keeping.

Known Limitations
-----------------

- Space-group comparison depends on ``spglib`` and on the selected symmetry
  tolerance.
- Bonding, coordination, and topology metrics depend on the radius-based
  neighbour rule and can shift when a structure lies close to the cutoff.
- Bond-angle metrics are less stable for very low coordination numbers.
- Large supercells increase the cost of the species-resolved Hungarian
  assignment.
- The similarity score is intentionally heuristic and should not be interpreted
  as a universal or transferable materials descriptor.
