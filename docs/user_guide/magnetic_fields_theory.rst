Magnetic Fields: Theory and Worked Examples
===========================================

This page gives the physics behind the :doc:`/user_guide/magnetic_fields`
page: how a magnetic structure is stored, how the local field at a muon site is
calculated, and worked examples whose numbers are checked by the test suite.
You do not need it to run a calculation; the page guide lists every control.

.. _magnetic-fields-representation:

Magnetic Structure Representation
---------------------------------

MuFinder stores magnetic structures using the propagation-vector formalism. A
magnetic model contains:

- up to three propagation-vector basis directions in reciprocal lattice units;
- user-facing components such as collinear moments, spin-density waves,
  helices, cycloids, and conical helices;
- complex Fourier coefficients for each component, stored as Cartesian moment
  vectors in Bohr magnetons;
- atom selectors describing where each component applies.

For an atom at fractional coordinate ``r`` and a unit-cell offset ``R``, each
component contributes a real moment from the Fourier coefficient ``S(q)`` as::

   m(r + R) = Re[ S(q) exp(-2 pi i q . (r + R)) ]

The complex coefficient therefore encodes both amplitude and phase. A purely
real coefficient gives a collinear cosine modulation. A coefficient with real
and imaginary parts along perpendicular axes gives a rotating structure such as
a helix or cycloid.


.. _magnetic-fields-theory:

Field Calculation Theory
------------------------

For commensurate structures, MuFinder can calculate a finite list of local
fields at the requested muon sites. The dipolar contribution is evaluated using
the ``muesr``/``muLFC`` engine. The stored magnetic model can also be exported as
atom-resolved Fourier coefficients in the format expected by ``muesr``.

Dipolar tensors are evaluated from the selected magnetic-ion positions using a
unit marker moment internally, because the tensor describes the linear coupling
between a magnetic moment and the local field at the muon. The tensor workflow
therefore depends on the chosen magnetic-ion set and lattice summation settings,
not on the detailed Fourier coefficients of a magnetic texture.

Moments follow one convention everywhere in MuFinder. Atom ``j`` in the cell at
lattice translation ``R`` carries

.. math::

   \mathbf{m}_j(\mathbf{R}) =
   \mathrm{Re}\left[\mathbf{F}_j\, e^{-2\pi i\, \mathbf{q} \cdot \mathbf{R}}\right],

where the atom-resolved coefficient :math:`\mathbf{F}_j` already contains the
phase :math:`e^{-2\pi i\, \mathbf{q} \cdot \mathbf{x}_j}` of the atom's
fractional position. ``R`` is measured from the home cell, the cell whose
fractional coordinates lie in ``[0, 1)`` and hold the muon site, so the field
reported for a site at ``r`` is the field at ``r``. ``muesr`` places each muon
in cell :math:`\lfloor N/2 \rfloor` of its :math:`N_a \times N_b \times N_c`
summation supercell and measures phases from the supercell origin, so MuFinder
multiplies every coefficient by
:math:`e^{2\pi i\, \mathbf{q} \cdot \lfloor \mathbf{N}/2 \rfloor}` before
handing it over. The ``Cells along`` setting therefore changes a result only
through the convergence of the lattice sum, never by moving the muon to a
different magnetic cell.

For incommensurate structures, a single muon site generally samples a
distribution of local fields rather than one finite set of symmetry-equivalent
values. MuFinder calculates the field distribution itself and stops there; it
does not currently simulate spectra from that distribution.

Before calling ``muesr`` for an incommensurate distribution, MuFinder checks the
atom-resolved complex Fourier coefficients. The fast ``muLFC`` incommensurate
engine assumes that each atom-level coefficient is circular-helical, meaning
that the real and imaginary parts are orthogonal and have equal magnitude.
General elliptical or linearly polarised terms are therefore decomposed into up
to two circular-helical components,
``F = w+ (e1 + i e2) + w- (e1 - i e2)``. Each component satisfies the backend
assumptions, and the resulting fields are summed linearly to preserve the
original magnetic structure.

.. _magnetic-fields-incommensurate-sampling:

Multi-q Incommensurate Sampling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

MuFinder follows the same phase-space strategy used by the corresponding
``mag2spectra`` field workflow, adapted to MuFinder's result objects and UI. The
important point is that a multi-q texture has independent phase coordinates, so
different propagation-vector basis rows must not be advanced with one locked
phase angle.

Each magnetic component stores a coefficient tuple ``q_coeff = (l, m, n)`` in
the magnetic model's propagation basis:

.. math::

   \mathbf{q}_\alpha =
   l_\alpha\mathbf{k}_1 +
   m_\alpha\mathbf{k}_2 +
   n_\alpha\mathbf{k}_3.

For every component, MuFinder first asks ``muesr`` for a one-dimensional
incommensurate lookup table using ``locfield(..., 'i', nangles=N)``. ``N`` is
the ``Lookup angles per q`` setting. Each lookup row is a dipolar, Lorentz, and
contact field contribution for one reduced phase angle in ``[0, 1)``.

MuFinder then identifies the active independent phase dimensions from the
actual incommensurate component wavevectors, not merely from populated q-basis
rows. Unused q-basis rows therefore do not add dimensions, and physically
dependent component wavevectors are sampled through one reduced phase axis with
the appropriate integer or fractional phase coefficient. At each sampled point
``Phi_n``, a component's own reduced phase is:

.. math::

   p_{\alpha,n} = \operatorname{mod}(\Phi_n \cdot \mathbf{c}_\alpha, 1),

where ``c_alpha`` is the component's coefficient in the reduced active phase
basis. MuFinder converts that phase to the nearest lookup-table index,

.. math::

   j_{\alpha,n} =
   \operatorname{round}(p_{\alpha,n} N) \bmod N,

and sums the component fields at those indices. This is what lets square and
hexagonal skyrmion-style multi-q structures sample the independent phase space
instead of behaving like their q vectors are artificially locked together.

The user-facing sampling controls are:

``Lookup angles per q``
   The lookup resolution ``N`` used for each component returned by ``muesr``.
   Increasing this value improves the angular fidelity of each component
   contribution. It is deliberately separate from the final number of sampled
   fields.

``Distribution samples``
   The target number of sampled fields in the final distribution. Monte Carlo
   sampling uses this value directly. Grid sampling chooses an equal resolution
   along each active phase axis without exceeding the target or the lookup
   resolution.

``Grid``
   Builds a deterministic Cartesian grid over the active independent phase
   dimensions. This is reproducible and is most useful for one- and two-q phase
   spaces when the requested distribution sample count is affordable.

``Monte Carlo``
   Draws random points in the active phase space and uses ``Distribution
   samples`` as the total number of phase samples. This keeps large phase spaces
   bounded, but the resulting histogram has statistical noise. The seed of
   the random draw is recorded with the result: it is the ``random_seed``
   option when one is given, and a freshly drawn seed otherwise.

``Auto``
   Chooses grid sampling for one or two active phase dimensions when the
   requested distribution sample count is below MuFinder's internal safety cap,
   and otherwise falls back to Monte Carlo sampling. This is the default because
   it gives deterministic low-dimensional distributions without accidentally
   constructing very large arrays.

The calculation parameters saved with an incommensurate result include the
chosen ``phase_sampling`` mode, the ``resolved_phase_sampling`` it ran as
(``grid`` or ``montecarlo``), the active ``independent_q_indices``, the
``phase_basis_rlu`` they span, the ``phase_sample_count`` actually used, the
``phase_sampling_seed`` of a Monte Carlo run (``None`` for a grid), the
``component_phase_coefficients`` of each lookup table in that basis, and the
histogram settings. The phase samples themselves are not saved: they follow
from these parameters, and
:func:`~mufinder.science.magnetic_fields.regenerate_phase_samples` rebuilds
them bit for bit, with the coefficients, without a new lattice sum.

.. _magnetic-fields-convergence-checks:

Convergence Checks
~~~~~~~~~~~~~~~~~~

Every incommensurate distribution carries three convergence checks, computed
from the lookup tables already returned by ``muesr`` without further backend
calls:

``Lookup angles per q``
   The distribution is recomputed from every other lookup angle, which is the
   lookup resolution ``N/2``. The check reports the Kolmogorov-Smirnov
   distance, the largest difference between the two cumulative distributions
   of ``|B|``, and passes when it is at most 1%. A failing check advises
   doubling ``Lookup angles per q``. The check runs only for an even number of
   lookup angles, since an odd number has no uniform half-resolution lookup.

``Distribution samples``
   The distribution is recomputed from half the phase samples: every other
   Monte Carlo sample, or every other grid point along each phase axis. It
   passes on the same 1% criterion and advises more distribution samples, or
   more lookup angles when a grid already uses every lookup angle along its
   axes.

``Histogram bins``
   The mean number of samples per occupied histogram bin, which must be at
   least 20. A failing check advises fewer bins, or more samples (more lookup
   angles for a grid limited by them). A single-valued distribution passes.

A structure whose sampled phase space collapses to a single phase reports only
the histogram check, and an odd ``Lookup angles per q`` omits the lookup-angle
check. After a calculation, the status line counts the checks
that need attention; selecting a distribution row lists each check with its
verdict and advice under the results table, for new and restored results.
The checks are saved with the result as ``convergence`` entries holding
``setting``, ``label``, ``converged``, ``metric`` and ``message``.

The defaults pass for a single-q distribution: a one-dimensional grid uses one
sample per lookup angle, so 32,768 lookup angles give 64 samples per
occupied bin of a 512-bin histogram. The lookup resolution adds little cost,
because ``muesr`` evaluates the lattice sum once per component and the lookup
angles only rotate that sum.

.. _magnetic-fields-commensurate-approximants:

Commensurate Approximants
~~~~~~~~~~~~~~~~~~~~~~~~~

An incommensurate wavevector is often close to a short-period fraction, for
example a measured ``q = (0.3334, 0, 0)`` near ``(1/3, 0, 0)``. MuFinder looks
for a commensurate approximant whenever it samples an incommensurate
distribution: every component of every wavevector is replaced by the fraction
``p/n`` with the smallest ``n`` up to ``Longest period`` that lies within
``Tolerance`` of it. When every component has such a fraction, the approximant
repeats over a magnetic supercell whose size along each axis is the least
common multiple of the denominators on that axis.

With ``Report finite fields for near-commensurate q`` off, the distribution is
sampled as usual and the status line names the approximant it found. With the
option on, MuFinder lists the finite set of fields that the
translation-equivalent copies ``r + T`` of the muon site see in that magnetic
supercell. The field at ``r + T`` is the commensurate lattice sum of the
approximant with every Fourier phase advanced by ``q . T``. ``muesr`` takes
these phases in units of :math:`2\pi`, adding them to ``q . R`` of each cell.
MuFinder lists one translation per distinct set of phases, so every listed
field stands for the same number of sites. Rows are labelled with their
translation, for example ``Custom site + (1, 0, 0)``, carry the full term
breakdown of a commensurate result, and give the muon position ``r + T`` in
host fractional coordinates. A structure without an approximant within the
limits keeps its sampled distribution. MuFinder finds the distinct phase sets
by stepping one unit translation at a time from the origin, so it never walks
the whole magnetic supercell, and it stops with an error as soon as it finds
more than 512.

Beyond the dipolar sum, the field calculation includes:

- the Lorentz field. It is part of the Lorentz-sphere evaluation of the
  dipolar sum, not an optional extra, so it is always included and reported as
  its own term (see `Field Terms`_);
- the contact hyperfine term, when a ``Hyperfine coupling`` is entered, from
  the moments of the contact shell set in ``Calculation Settings`` (see
  :ref:`Choose the contact shell
  <magnetic-fields-contact-shell>`);
- the demagnetizing field for commensurate finite-field calculations and
  incommensurate sampled distributions, exposed as none, sphere, infinite plate,
  advanced ellipsoid, or advanced rectangular prism;
- the muon-induced distortion correction of the dipolar and contact terms, when
  the selected site requests it (see :ref:`Apply muon-induced distortion corrections <magnetic-fields-distortion-corrections>`).

Dipolar tensors from selected magnetic-ion positions and the nuclear-relaxation
distortion correction are available through their own dialogs.

.. _magnetic-fields-field-terms:

Field Terms
~~~~~~~~~~~

Selecting a commensurate result row lists, under the results table, the terms
that sum to its field: ``Dipolar``, ``Lorentz``, ``Contact``,
``Demagnetizing``, ``Distortion correction`` (only for corrected rows) and
``Applied``, each as ``Bx (T)``, ``By (T)``, ``Bz (T)`` and ``|B| (T)``, followed
by their ``Total``. A newly calculated or restored commensurate result selects
its first row. With several rows selected, the first selected row is shown. A
saved result that stores only its total asks for a recalculation instead.

Selecting a distribution row lists the same terms without a separate distortion
correction, which a corrected distribution includes in its dipolar and contact
terms, under the headings ``Mean Bx (T)``, ``Mean By (T)``, ``Mean Bz (T)``
and ``RMS |B| (T)``: the mean vector of each term over the phase samples and
its root-mean-square magnitude :math:`\sqrt{\langle |\mathbf{B}_\text{term}|^2
\rangle}`. The means sum to the ``Total`` mean, and the ``Total`` RMS is that
of ``|B|``. Both columns are needed: over the phases of a helix the dipolar,
Lorentz and contact terms each average to nearly zero while every sample
carries a large field, and only the RMS shows its size. The demagnetizing and
applied terms are the same at every phase, so their mean is the vector and
their RMS its length. The selected row's convergence checks (see `Convergence
Checks`_) follow the table. A distribution saved without its terms shows its
checks and asks for a recalculation instead.

The results table and the term breakdown show fields to six significant
figures. A component no larger than 10\ :sup:`-12` T, or than 10\ :sup:`-9` of
its row's magnitude, is rounding noise from the lattice sum and shows as ``0``,
and no cell shows ``-0``. For a distribution row the magnitude is the mean
``|B|``. The saved result keeps the calculated values unrounded.


.. _magnetic-fields-worked-examples:

Worked Examples
---------------

The ferromagnet is bcc iron and the helix is MnSi. The antiferromagnet uses one
magnetic ion of 1 μB in a simple cubic cell with ``a = 3 Å``. All use the
default ``Calculation Settings`` (``100 x 100 x 100`` cells, Lorentz sphere
chosen by ``muesr``) except iron, which uses ``80 x 80 x 80`` cells. Positions
are fractional. Each number comes from the calculation and is pinned by
``tests/science/test_magnetic_fields.py``.

Ferromagnet
~~~~~~~~~~~

bcc iron has ``a = 2.868 Å`` with Fe at ``(0, 0, 0)`` and ``(1/2, 1/2, 1/2)``
and 2.2 μB per Fe along ``c``. The bulk magnetisation is
:math:`\mu_0 M = 2.174` T, so the Lorentz field :math:`\mu_0 M / 3` is 0.7246 T
along ``c``; MuFinder reports 0.7243 T because ``muesr`` counts the ions inside
a finite Lorentz sphere. With ``Demagnetizing field`` set to ``Sphere`` the
demagnetizing term is -0.7246 T, and the Lorentz and demagnetizing terms sum
to -0.0003 T, the finite-sphere difference. With ``Infinite plate`` it is
-2.1737 T, and the two sum to -1.4494 T. The ``80 x 80 x 80`` sum keeps the
cubic symmetry of the Lorentz sphere exactly, so field components that vanish by
symmetry print as ``0``; at ``100 x 100 x 100`` they carry about 10\ :sup:`-7` T
of numerical noise.

The two interstitial sites of the bcc lattice have a dipolar field that depends
on how the site's symmetry axis lies relative to the moment. Each site's field
along ``c`` is:

.. list-table::
   :header-rows: 1

   * - Site
     - Axis along ``c`` (T)
     - Axis along ``a`` or ``b`` (T)
     - Mean of the equivalent sites (T)
   * - Octahedral ``(1/2, 0, 0)`` (6b)
     - +1.85 (2 of 6 sites)
     - -0.93 (4 of 6)
     - 0
   * - Tetrahedral ``(1/2, 1/4, 0)`` (12d)
     - -0.53 (4 of 12)
     - +0.26 (8 of 12)
     - 0

The dipolar tensor of each site is traceless and uniaxial. At the octahedral
site ``(1/2, 0, 0)`` it is ``diag(0.8419, -0.4210, -0.4210)`` T/μB, with its axis
along ``a``, and at the tetrahedral site ``(1/2, 1/4, 0)`` it is
``diag(0.1194, -0.2387, 0.1194)`` T/μB, with its axis along ``b``. Multiplied by
the 2.2 μB moment and the ``c`` component of the moment direction, the ``zz``
element gives the field in the table. A muon that visits every equivalent site
of a sublattice sees none of it on average: the mean of the dipolar field over
the equivalent sites is zero.

A ``Hyperfine coupling`` of -0.5045455 T/μB gives a contact field of -1.11 T,
the value measured at low temperature, because ``muesr`` takes the contact field
from the moments of the muon's two nearest Fe. It points against the moment. At
the octahedral site with its axis along ``a``, the dipolar (-0.9261 T), Lorentz
(+0.7243 T) and contact (-1.1100 T) terms sum to -1.3118 T with no demagnetizing
field, and to -2.0364 T with ``Sphere``. A ``Measured bulk magnetisation`` of
2.0 μB per Fe changes the demagnetizing term of a sphere to -0.6587 T and that
total to -1.9705 T.

A-type antiferromagnet
~~~~~~~~~~~~~~~~~~~~~~

With ``q = (0, 0, 1/2)`` the ``ab`` layers alternate along ``c`` and the
Lorentz field vanishes. At the body centre the two neighbouring layers cancel,
so the field is zero. The face centre ``(1/2, 1/2, 0)`` sees -0.5493 T and the
edge centre ``(1/2, 0, 0)`` -0.8003 T along ``c``.

Incommensurate helix
~~~~~~~~~~~~~~~~~~~~

B20 MnSi (space group P2\ :sub:`1`\ 3, ``a = 4.558 Å``) has Mn and Si at the 4a
positions with ``x = 0.138`` and ``x = 0.845``. Below about 29 K the Mn moments
of 0.385 μB turn in the plane perpendicular to ``q = (0.0147, 0.0147, 0.0147)``
r.l.u., a left-handed helix of period about 18 nm that is incommensurate with the
lattice. The muon site is 4a with ``x = 0.532``, whose four positions in one
magnetic domain are site I on the propagation axis and sites II to IV, which are
equivalent to one another (Amato *et al.*, 2014; the worked example below gives the reference). The worked example
:doc:`/user_guide/examples/mnsi` builds the model step by step.

For one wavevector the sampled field traces an ellipse,
:math:`\mathbf{B}(\theta) = \cos\theta\,\mathbf{B}_0 + \sin\theta\,\mathbf{B}_{90}`,
so :math:`p(|B|)` runs between the two semi-axes and peaks at both ends. On the
axis the ellipse is a circle and the distribution a single line: site I gives
88 mT. Sites II to IV span 90 mT to 207 mT. These include a contact term, the
``Hyperfine coupling`` of -0.505 T/μB times the moments of the muon's nearest
Mn; the sampled distribution applies it at every phase just as a commensurate
field does. The dipolar field alone gives 107 mT at site I and 48 mT to 104 mT
at sites II to IV. The helix's sense matters at sites II to IV, which are not
on the axis: the right-handed helix of the same moment spans 86 mT to 209 mT,
while site I is unchanged.

.. figure:: ../_generated/screenshots/mnsi_field_distribution.webp
   :alt: The field-distribution plot of the four MnSi muon sites: a single line
         at 0.088 T and three equal continua from 0.090 T to 0.207 T, peaking at
         both ends, with their sum.

   ``Open Distribution Plot`` for the four 4a sites of the MnSi helix. The
   singleton is site I; the other three curves coincide.

Commensurate approximant
~~~~~~~~~~~~~~~~~~~~~~~~

The helix's ``q = 0.0147`` lies 0.0009 r.l.u. from 1/64, so with ``Longest
period`` 64 its approximant is ``q = (1/64, 1/64, 1/64)`` with a
``64 x 64 x 64`` magnetic supercell. ``Report finite fields for
near-commensurate q`` then lists 64 fields for each site, one per translation
``r + T`` with a distinct phase. At site II their magnitudes run from 90.0 mT
to 207.3 mT, the edges of the sampled distribution, and the 64 phases sample the
same two-horned lineshape. At the default period of 12 no fraction is within
0.001 r.l.u., so the page samples the distribution.

