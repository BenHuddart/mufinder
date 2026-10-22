The muon in bcc iron
====================

Iron is the classic host for a ferromagnetic local field. The positive muon
stops in an interstitial site of the body-centred cubic lattice, and the field it
feels there is the sum of four terms that are each easy to state: a dipolar
field from the Fe moments, the Lorentz field, a contact field, and the
demagnetizing field of the sample. This page follows the
:doc:`/user_guide/magnetic_fields` workflow through that sum for the two
interstitial sites that have been proposed for the muon, the octahedral and the
tetrahedral site, and shows why the two terms that depend on the lattice
geometry cancel for a diffusing muon.

.. note::

   No calculation on this page ran CASTEP, and none needed to. The project is a
   host structure and a magnetic model, so every number on the page is computed
   by MuFinder from the stated model with its ``muesr`` field calculation, and
   is pinned by ``tests/science/test_magnetic_fields.py``. The contact field is
   the one input taken from the literature: the ``Hyperfine coupling`` is set so
   that it equals the field measured at low temperature, as described in `The
   contact field`_.

The host
--------

Iron is bcc, space group Im-3m, with a = 2.868 Å and one Fe at (0, 0, 0) and one
at (½, ½, ½) of the conventional cubic cell. It is a ferromagnet with a moment of
2.2 μB per Fe, here along c. The magnetisation is
:math:`\mu_0 M = 2.17` T.

.. figure:: ../../_generated/screenshots/fe_host_structure.webp
   :alt: The Host Structure page with bcc Fe loaded: two Fe in the atom table
         with Spin (hbar/2) 2.2 each, and the cell viewed down a with parallel
         spin arrows on both atoms.

   The bcc Fe cell, with a spin of 2.2 ħ/2 on each atom marking the
   ferromagnetic order.

The same 2.2 μB sets the magnetic model. ``Edit Magnetic Structure...`` holds one
collinear component, ``q = (0, 0, 0)``, applied to all atoms with the moment
``[0, 0, 2.2]``; the ``Magnetic Structure`` card on the page lists it.

The two candidate sites
-----------------------

The bcc lattice has two interstitial sites that can hold a light impurity. The
octahedral site (½, 0, 0), Wyckoff position 6b, sits between two Fe along an
axis and has six equivalent positions in the conventional cell. The tetrahedral
site (½, ¼, 0), position 12d, is surrounded by four Fe and has twelve. Neither
is a result of a CASTEP relaxation here, so each is entered as a custom
fractional position in ``Select Muon Site``.

.. figure:: ../../_generated/screenshots/fe_select_muon_site.webp
   :alt: The Select Muon Site dialog with the custom fractional position source
         and the octahedral site, fx 0.5, fy 0, fz 0.

   The octahedral site as a custom position. The tetrahedral site is
   (0.5, 0.25, 0).

Ticking ``Include equivalent sites`` on the page makes the calculation list
every symmetry-equivalent position of the chosen site. For a ferromagnet that
matters: the sites are equivalent by the lattice symmetry, but not by the
magnetic symmetry, since the moment along c picks out one direction.

The field at the muon
---------------------

The field at the muon is

.. math::

   \mathbf{B} = \mathbf{B}_\mathrm{dip} + \mathbf{B}_\mathrm{Lor} +
   \mathbf{B}_\mathrm{cont} + \mathbf{B}_\mathrm{demag},

as described in :doc:`/user_guide/magnetic_fields_theory`. With 2.2 μB per Fe,
:math:`\mu_0 M = 2.17` T. The Lorentz field, :math:`\mu_0 M / 3`, is +0.72 T. A
sample that is a sphere has a demagnetizing field of
:math:`-\mu_0 M / 3 = -0.72` T, which cancels it. The dipolar field is the sum
over the Fe moments around the site, and the contact field comes from the
coupling of the muon to the spin density at its position.

.. figure:: ../../_generated/screenshots/fe_magnetic_fields_page.webp
   :alt: The Magnetic Fields page for bcc iron at the octahedral site, listing six
         equivalent sites with Bz of -2.03636 and 0.741952 T, and the breakdown of
         the first: dipolar -0.926105 T, Lorentz 0.724304 T, contact -1.11 T,
         demagnetizing -0.724555 T.

   The octahedral site with every equivalent site and a sphere, computed by
   MuFinder from the model above. The first site's field along c is the sum of
   -0.926 T dipolar, +0.724 T Lorentz, -1.110 T contact and -0.725 T
   demagnetizing.

The breakdown shows the four terms for the selected site. The Lorentz and
demagnetizing terms are equal and opposite up to the 0.0003 T by which
``muesr``'s finite Lorentz sphere differs from :math:`\mu_0 M / 3`. That leaves
the dipolar and contact fields.

The dipolar field and the equivalent sites
------------------------------------------

The dipolar field at an interstitial site is not the same for every equivalent
position. Each site has a symmetry axis, the Fe-Fe axis through the octahedral
site, and its dipolar tensor is traceless and uniaxial about it. The field it
gives depends on the angle between that axis and the moment.

.. figure:: ../../_generated/screenshots/fe_equivalent_sites.webp
   :alt: The Calculation card for bcc iron with no demagnetizing field, listing
         six octahedral sites, four with Bz of -1.3118 T and two with 1.46651 T,
         and the breakdown of the third, whose dipolar term is 1.8522 T.

   The six octahedral sites without a demagnetizing field. The two whose Fe-Fe
   axis lies along the moment have a dipolar field of +1.85 T; the other four,
   -0.93 T.

At the octahedral sites the field along the moment is +1.85 T where the Fe-Fe
axis lies along c, two of the six sites, and -0.93 T where it lies along a or
b, the other four. At the tetrahedral sites it is -0.53 T for four of the twelve
and +0.26 T for the other eight. In both cases the weighted mean is zero, as
the traceless tensor requires. These are point-dipole sums, not measured
values: the ``muesr`` documentation's bcc iron example gives +0.265 T and
-0.530 T at the tetrahedral sites for the same model, and the octahedral
values follow from the same sum. A muon that is static at low temperature
sees one of these fields; a muon that hops between equivalent sites sees
their average.

.. figure:: ../../_generated/screenshots/fe_tetrahedral_sites.webp
   :alt: The Magnetic Fields page for bcc iron at the tetrahedral site, listing
         twelve equivalent sites with Bz of -0.847684 T for eight of them and
         about -1.636 T for four, and the breakdown of the first.

   The tetrahedral site with its twelve equivalent positions and a sphere,
   computed by MuFinder.

Which of these fields the muon feels depends on how long it stays on a site.
A muon that is static for the timescale of the measurement sees one of the
sublattice values. A muon that hops between the equivalent sites faster than
the dipolar field can dephase it sees only their average, and the average
dipolar field is zero. Grynszpan *et al.* (1979) measured the diffusion of the
muon in iron. What is left is the
contact field plus the Lorentz and demagnetizing terms, and the latter two
cancel for a spherical sample:

.. list-table::
   :header-rows: 1

   * - Term (T)
     - Octahedral
     - Tetrahedral
   * - Mean dipolar
     - 0
     - 0
   * - Lorentz
     - +0.7243
     - +0.7243
   * - Demagnetizing (sphere)
     - -0.7246
     - -0.7246
   * - Contact
     - -1.1100
     - -1.1100
   * - Mean total
     - -1.1103
     - -1.1103

The two sites give the same mean field, so a field that is averaged over the
sites cannot tell them apart.

The demagnetizing field
-----------------------

The cancellation is a property of a spherical sample. A thin plate magnetised
along its normal has :math:`N = 1` instead of :math:`\tfrac{1}{3}`, and the
demagnetizing field is the full :math:`-\mu_0 M`, so the field at the muon shifts.
``Infinite plate`` takes the plate normal along c, which is the direction of the
moment in this example.

.. figure:: ../../_generated/screenshots/fe_demagnetizing_shapes.webp
   :alt: The Calculation card for bcc iron with Infinite plate chosen, the
         demagnetizing term -2.17367 T and the first site's total -3.48547 T.

   An infinite plate normal to c. The demagnetizing term is -2.1737 T, so the
   first octahedral site's total is -3.4855 T instead of the sphere's -2.0364 T.

The model's moment assumes a single-domain sample. Iron in zero field is
multidomain, with a smaller bulk magnetisation, and for a real sample the
demagnetizing field depends on the measured value. Ticking ``Measured bulk
magnetisation`` takes it from magnetometry instead.

.. figure:: ../../_generated/screenshots/fe_measured_magnetisation.webp
   :alt: The Calculation card for bcc iron with a measured bulk magnetisation of
         2 muB per Fe along z ticked, the demagnetizing term -0.658686 T.

   A measured magnetisation of 2.0 μB per Fe along z, an illustrative value
   rather than a measurement. The sphere's demagnetizing term becomes -0.6587 T
   and the first site's total -1.9705 T.

The contact field
-----------------

The contact field at the muon is not computed from first principles by
MuFinder. It is the ``Hyperfine coupling`` times the moments of the muon's
nearest Fe, and the coupling is an input. Low-temperature measurements of the
muon's field in iron give -1.11 T, antiparallel to the magnetisation (Nishida
*et al.* 1977). The nearest two Fe carry 2.2 μB each, so a coupling of
-0.5045455 T/μB in the ``T/μB`` unit gives a contact field of -1.11 T, which is
what the pages above use.

Density functional theory predicts the coupling rather than taking it. For the
tetrahedral site Onuorah *et al.* (2018, 2019) find -1.25 T for a static muon
and -1.07 T once the muon's zero-point motion is included, the latter close to
the measured -1.11 T.

The site question
-----------------

Which interstitial site the muon occupies in iron has been argued both ways,
between the octahedral and the tetrahedral site. The low-temperature
measurements of Schmolz *et al.* (1986) are read with the muon at tetrahedral
sites, hopping between them. Density functional theory prefers the tetrahedral
site by 0.18 eV and finds the octahedral site harmonically unstable, though
the muon's zero-point energy stabilises it (Onuorah *et al.* 2019). The
calculations on this page give both sites the same averaged field, so they
cannot settle the question: it takes the unaveraged dipolar fields of a muon
that is static at low temperature, which differ between the two sites, or the
contact field and the energies of the two sites.

Saving the two calculations
---------------------------

``Save Workspace`` keeps a calculation as a workspace of its own. Saving the
octahedral and the tetrahedral calculation gives the two sites side by side in
the navigator, each with its results restored when the project is reopened.

.. figure:: ../../_generated/screenshots/fe_magnetic_workspaces.webp
   :alt: The project navigator with two Magnetic Fields workspaces, Octahedral
         site and Tetrahedral site, and the octahedral workspace's six equivalent
         sites on the page.

   The octahedral and tetrahedral workspaces under ``Magnetic Fields``.

References
----------

- N. Nishida, R. S. Hayano, K. Nagamine, T. Yamazaki, J. H. Brewer,
  D. M. Garner, D. G. Fleming, T. Takeuchi and Y. Ishikawa, Hyperfine field and
  diffusion of μ⁺ in Fe single crystals, Solid State Commun. **22**, 235
  (1977), https://doi.org/10.1016/0038-1098(77)90401-X.
- M. Schmolz, K. Döring, K. Fürderer, M. Gladisch, N. Haas, D. Herlach,
  G. Majer, J. Rosenkranz, W. Schäfer, L. Schimmele, W. Schwarz, A. Seeger and
  E. Yagi, Positive muons in iron: dipolar fields at tetrahedral sites and jump
  frequencies at low temperatures, Hyperfine Interact. **31**, 199 (1986),
  https://doi.org/10.1007/BF02401559.
- I. J. Onuorah, P. Bonfà and R. De Renzi, Muon contact hyperfine field in
  metals: a DFT calculation, Phys. Rev. B **97**, 174414 (2018),
  https://doi.org/10.1103/PhysRevB.97.174414.
- I. J. Onuorah, P. Bonfà, R. De Renzi, L. Monacelli, F. Mauri, M. Calandra and
  I. Errea, Quantum effects in muon spin spectroscopy within the stochastic
  self-consistent harmonic approximation, Phys. Rev. Materials **3**, 073804
  (2019), https://doi.org/10.1103/PhysRevMaterials.3.073804.
- R. Grynszpan, N. Nishida, K. Nagamine, R. S. Hayano, T. Yamazaki,
  J. H. Brewer and D. G. Fleming, Diffusion of positive muons in zone-refined
  iron below room temperature, Solid State Commun. **29**, 143 (1979),
  https://doi.org/10.1016/0038-1098(79)91025-1.
