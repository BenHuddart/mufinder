The muon in the helimagnet MnSi
===============================

MnSi is the textbook helimagnet. Its Mn moments do not point the same way from
one cell to the next but rotate about a propagation vector, in a helix whose
period of about 18 nm is some 40 times the lattice parameter and is not a whole
number of cells. A muon implanted in it therefore does not feel one field, or
even a handful of them, but a continuous distribution. This page follows the
:doc:`/user_guide/magnetic_fields` workflow through that case: the host and the
muon site, the helix built with the structure builder, the distribution of
fields it gives, how the numerical settings are checked, the commensurate
approximant that turns the distribution into a finite list, and the contact
term that fixes where the distribution sits.

.. note::

   No calculation on this page ran CASTEP, and none needed to. The project is a
   host structure and a magnetic model, so every number on the page is computed
   by MuFinder from the stated model with its ``muesr`` field calculation, and
   is pinned by ``tests/science/test_magnetic_fields.py``. The model is the
   plain helix: the refinement of Dalmas de Réotier *et al.* (2016), which adds
   a second Mn orbit, is not used. The ``Hyperfine coupling`` is the one input
   that is set rather than computed, as described in `The contact term`_.

The host and the muon site
--------------------------

MnSi has the cubic B20 structure, space group P2\ :sub:`1`\ 3, with
a = 4.558 Å. Mn and Si both sit at the 4a positions (x, x, x),
(-x+½, -x, x+½), (-x, x+½, -x+½) and (x+½, -x+½, -x), with x = 0.138 for Mn and
x = 0.845 for Si, so the cell holds four Mn and four Si. The structure is
chiral, and that chirality fixes the sense of the helix.

.. figure:: ../../_generated/screenshots/mnsi_host_structure.webp
   :alt: The Host Structure page with MnSi loaded: four Mn and four Si in the
         atom table with their fractional positions, and the cubic cell in the
         viewer.

   The B20 cell of MnSi: Mn at 4a with x = 0.138, Si at 4a with x = 0.845.

Amato *et al.* (2014) find the positive muon at another 4a position, with
x = 0.532, so at (0.532, 0.532, 0.532). It is not the result of a CASTEP
relaxation here, so it is entered as a custom fractional position in ``Select
Muon Site``. With ``Include equivalent sites`` ticked the calculation lists the
four positions of that orbit. For a helix they are not alike: one of them lies on
the propagation axis, along [111], and the other three are equivalent to each
other. They are called site I and sites II, III and IV below.

The helix
---------

Below about 29 K MnSi orders in a helix, first established by neutron
diffraction (Ishikawa *et al.*, 1976). The propagation vector is along [111]
with k = 0.035 Å\ :sup:`-1`, a period of about 18 nm, which is
q = (0.0147, 0.0147, 0.0147) in reciprocal lattice units, and the Mn moment is
0.385 μB at 5 K (Dalmas de Réotier *et al.*, 2016). The helix is left-handed, and
the moments turn in the plane perpendicular to [111].

The builder takes the vector in two steps. The propagation direction of
``Add Preset...`` is a crystal axis or a row of the q basis, and [111] is not an
axis, so q is first typed into row ``q1`` of the ``Propagation-Vector Basis``
panel, and the preset then propagates along that row.

.. figure:: ../../_generated/screenshots/mnsi_preset_helper.webp
   :alt: The Add Magnetic Preset dialog set to a left-handed helix along basis
         row q1 = (0.0147, 0.0147, 0.0147), moment 0.385 μB applied to the
         species Mn.

   The preset helper, with ``Helix`` along ``q1``, ``Left-handed``, a moment of
   0.385 μB and the species ``Mn`` as the atoms it applies to.

The result is one helix component on the four Mn. The checklist is complete
once the preview repeats the cell along every axis along which q is
incommensurate, and the structure type is ``incommensurate``.

.. figure:: ../../_generated/screenshots/mnsi_magnetic_structure_builder.webp
   :alt: The Magnetic Structure Builder with the MnSi helix: one component,
         Helix along q1 on the species Mn, a complete checklist, the q basis row
         q1 and the Mn spins of a 2 x 2 x 2 preview.

   The builder with the helix. Its Mn moment is the vector pair Re [0.314,
   -0.157, -0.157], Im [0, -0.272, 0.272], whose real and imaginary parts are
   perpendicular to [111] and to each other.

The sense of the rotation is the one thing worth checking, because the lattice
is chiral and the two senses give different fields at the muon. Along the
propagation axis a moment of a left-handed helix turns against the screw
sense, which a test checks from the moments of successive Mn along [111]. Sites
II to IV depend on it: the opposite sense gives a continuum from 86 mT to 209 mT
instead of the 90 mT to 207 mT below.

The field at the muon
---------------------

The field is the sum of the dipolar field from the Mn moments, the Lorentz
field and the contact field. The Lorentz field is zero for a helix. The sample has no demagnetizing field, so
the ``Demagnetizing field`` is ``None``. The dipolar field is a lattice sum, and
as the moment direction changes along the helix its value at the muon depends on
the phase of the helix at the muon.

That is why the field is a distribution. The phase of the helix at a given muon
is set by where the muon stopped relative to the origin of the helix, and since
the helix is incommensurate every phase occurs somewhere in the crystal, with
equal weight. ``Calculate Dipolar Field`` therefore samples the phase on a grid
and lists one row per site, showing the mean ``|B|`` of the distribution.

.. figure:: ../../_generated/screenshots/mnsi_magnetic_fields_page.webp
   :alt: The Magnetic Fields page for the MnSi helix: four distribution rows,
         site 1 equiv 1 with a mean of 0.0879129 T and the other three with
         0.154303 T, and the three convergence checks of the selected row.

   The four 4a sites of one magnetic domain, computed by MuFinder from the model
   above. Site I has a mean field magnitude of 0.0879 T. The other three have a mean of
   0.1543 T. The selected row's convergence checks are listed under the table.

Site I, on the axis, sees a circle: its field has the same magnitude at every
phase, 87.9 mT, an isolated line. Sites II to IV see an ellipse, so their
field magnitude runs between two limits, 90.3 mT and 206.8 mT, and spends most of its time
near them.

.. figure:: ../../_generated/screenshots/mnsi_field_distribution.webp
   :alt: The distribution plot of the four sites: a single line at 0.088 T, and
         three coinciding curves from 0.090 T to 0.207 T that rise sharply at
         both ends, with their sum.

   ``Open Distribution Plot`` with the four rows selected. The line at 88 mT is
   site I, which lies below the continuum's lower edge at 90 mT. The three
   continua coincide, and their sum has the two horns at 90 mT and 207 mT.

This is the two-horned lineshape of the muon spectrum of MnSi, with a separate
line just below it. Amato *et al.* (2014) predict, with no parameter fitted to
the lineshape, the line at 88.0 mT and a continuum from 90.0 mT to 206.5 mT, and
measure 91 mT, 96 mT and 207 mT at 5 K. The table puts the numbers side by side.

.. list-table::
   :header-rows: 1

   * - Edge
     - MuFinder (mT)
     - Predicted (mT)
     - Measured (mT)
   * - Site I (singleton)
     - 87.9
     - 88.0
     - 91
   * - Lower edge of II to IV
     - 90.3
     - 90.0
     - 96
   * - Upper edge of II to IV
     - 206.8
     - 206.5
     - 207

The computed gap between the line and the lower edge is 2.4 mT. The measured gap
is larger, 5 mT, and the plain helix cannot account for it. Dalmas de Réotier
*et al.* (2016) explain it by refining two Mn orbits, with a phase shift of
-2.04° of one of them, which gives the 4.8 mT gap. This example leaves that
refinement out.

Checking the sampling
---------------------

Every sampled distribution carries three convergence checks, listed under the
results table when a row is selected: halving the ``Lookup angles per q``,
halving the ``Distribution samples``, and the number of samples per occupied
histogram bin (see :ref:`Convergence Checks
<magnetic-fields-convergence-checks>`). For site II halving the lookup angles
or the samples shifts the distribution by 0.006 % and each occupied bin holds
64 samples, so all three pass with the defaults. The singleton passes the first
two at 0.2 %. A distribution that failed would name the setting to double.

The checks measure the numerical settings, not the physics, and they say nothing
about the model. For one wavevector the phase space is one-dimensional, so
``Auto`` sampling takes a grid of phases, which is deterministic: the same
calculation gives the same numbers, and the computed edges above are the same
on every run.

The commensurate approximant
----------------------------

The helix is incommensurate, but q = 0.0147 is close to 1/64. A magnetic cell of
64 host cells is a poorer description than the distribution, but it turns the
distribution into a finite list of fields, one for each of the 64 translations
of the muon site along the cell, which is useful for a comparison with a model
that needs discrete fields.

.. figure:: ../../_generated/screenshots/mnsi_calculation_settings.webp
   :alt: The Calculation Settings dialog with the approximant option ticked and
         a longest period of 64.

   The approximant needs ``Longest period`` 64, since 1/64 is the nearest
   fraction to 0.0147, 0.0009 r.l.u. away and inside the default tolerance of
   0.001. At the default period of 12 there is none and the page samples the
   distribution. With the period 64 but the option off, it samples the
   distribution too and names the approximant it found.

With ``Report finite fields for near-commensurate q`` on, the page lists the
64 fields at site II. They are the fields of a helix of q = (1/64, 1/64, 1/64),
whose period is 64 cells and which repeats over a 64 x 64 x 64 magnetic
supercell. The label of each row gives the translation ``T`` of the muon site, in
cells along a, that the row's field belongs to.

.. figure:: ../../_generated/screenshots/mnsi_approximant_fields.webp
   :alt: The Magnetic Fields page for the MnSi approximant at site II: 64 rows
         labelled site 1 + (n, 0, 0), and the term breakdown of the first, with
         a total of 0.118082 T.

   The 64 fields of the approximant at site II, computed by MuFinder. The
   first row's 0.1181 T is the sum of 0.0975 T dipolar, 0.0016 T Lorentz,
   0.1943 T contact and no demagnetizing field. The Lorentz term is a
   residue of the finite sphere, which does not average a helix to zero exactly.

Their magnitudes run from 90.0 mT to 207.3 mT, and the 64 phases sample the
same two-horned lineshape as the distribution, whose edges are 90.3 mT and
206.8 mT. The approximant is the coarse view of the same field: its q is 6 %
larger and its phases are 64 rather than 32,768, so its edges are within
0.5 mT of the distribution's.

The contact term
----------------

The dipolar field alone does not reproduce the lineshape: the same calculation
without the ``Hyperfine coupling`` gives 106.5 mT at site I and 47.8 mT to
104.4 mT at sites II to IV, nowhere near the measured fields. The contact field
is the ``Hyperfine coupling`` times the moments of the muon's nearest Mn. It
is applied in the sampled distribution at every phase, as it is in a
commensurate result, so the distributions above include it, and the
approximant's rows list it as the ``Contact`` term.

MuFinder does not compute the coupling. The one Amato *et al.* (2014) find from
transverse-field measurements, -0.9276 mol/emu, is -0.518 T/μB for a moment of
1 μB per Mn, and that gives 92.9 mT at site I and 95.3 mT to 211.7 mT at sites II
to IV, about 5 mT above the predicted edges. The calculation above uses
-0.505 T/μB, 2.5 % smaller, which puts all three edges within 0.3 mT of the
predicted ones. This is one parameter fitted to three numbers, so the
agreement of the edges is partly by construction. The coupling does more than
move the continuum: the contact field at the muon rotates with the helix
phase like the dipolar field, so it changes the semi-axes of the ellipse, and
with it the width. Without the contact term the continuum runs from 47.8 to
104.4 mT, 56.6 mT wide; with it, from 90.3 to 206.8 mT, 116.5 mT wide. What
the coupling does not change is the topology, which the dipolar sum, the
helix sense and the geometry fix: a circle at site I, so an isolated line, and
an ellipse at sites II to IV, so a two-horned continuum. The calculation
sums MuFinder's default of two nearest moments, which is not a setting on the
page, while the literature sums three; the three nearest Mn lie in one shell, and
summing any number from one to six changes the edges by 0.3 mT at most.

References
----------

- A. Amato, P. Dalmas de Réotier, D. Andreica, A. Yaouanc, A. Suter, G.
  Lapertot, I. M. Pop, E. Morenzoni, P. Bonfà, F. Bernardini and R. De Renzi,
  Understanding the μSR spectra of MnSi without magnetic polarons, Phys. Rev.
  B **89**, 184425 (2014),
  https://doi.org/10.1103/PhysRevB.89.184425.
- P. Dalmas de Réotier, A. Maisuradze, A. Yaouanc, B. Roessli, A. Amato, D.
  Andreica and G. Lapertot, Determination of the zero-field magnetic structure
  of the helimagnet MnSi at low temperature, Phys. Rev. B **93**, 144419 (2016),
  https://doi.org/10.1103/PhysRevB.93.144419.
- Y. Ishikawa, K. Tajima, D. Bloch and M. Roth, Helical spin structure in
  manganese silicide MnSi, Solid State Commun. **19**, 525 (1976),
  https://doi.org/10.1016/0038-1098(76)90057-0.
