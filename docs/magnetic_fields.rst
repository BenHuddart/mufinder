Magnetic Fields
===============

The ``Magnetic Fields`` workflow builds magnetic structures and calculates
local fields at candidate muon stopping sites. It needs a loaded host structure
and, for anything beyond a custom position, completed UEP or
structural-relaxation clustering results to supply muon sites.

A typical pass through the page is:

1. click ``Edit Magnetic Structure...`` in the ``Magnetic Structure`` card to
   open the ``Magnetic Structure Builder`` and define the magnetic model (see
   `Builder UI`_); optionally enter a ``Hyperfine coupling`` in ``Å⁻³``,
   ``T/μB``, or ``MHz/μB`` for the contact term;
2. choose a demagnetizing-field shape in the ``Demagnetizing Field`` card (see
   `Demagnetizing-Field Setup`_);
3. click ``Select Muon Site...`` in the ``Muon Site`` card (see
   `Muon-Site Selection`_), and tick ``Include equivalent sites`` to calculate
   all symmetry-equivalent positions of that site;
4. review ``Calculation Settings...`` and click ``Calculate Dipolar Field``.
   Results appear in the table below (``Site``, ``Bx (T)``, ``By (T)``,
   ``Bz (T)``, ``|B| (T)``, ``Result``); for an incommensurate result, select
   its row and click ``Open Distribution Plot`` to view the field-magnitude
   histogram, which ``Export Data...`` saves to a file;
5. optionally run ``Dipolar Tensor...`` or ``Nuclear Relaxation...`` from the
   same ``Calculation`` card.

The save button in the page header stores the current magnetic structure and
form settings as a named magnetic-field workspace. Saved workspaces appear as
children of ``Magnetic Fields`` in the project tree, can be renamed from their
page title, and can be deleted from their own page or by right-clicking their
project-tree node and choosing ``Delete Workspace...``.

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

MCIF Import
-----------

MuFinder imports magnetic metadata from ``.mcif`` files before any field or
spectral calculation is attempted. The importer follows the non-spectral MCIF
behaviour already used in mag2spectra:

- atom labels from ``_atom_site_label`` are preserved so magnetic-site labels
  can be mapped back to concrete unit-cell atom indices;
- propagation vectors are read from both legacy scalar tags such as
  ``_cell_wave_vector_x/y/z`` and MCIF triplet entries such as
  ``_magnetic_propagation_vector_kxkykz``;
- multiple imported propagation vectors are stored as
  ``propagation_vectors_rlu`` and mapped onto the model's first three
  propagation-vector basis rows;
- the first imported propagation vector is also mirrored to the legacy
  ``propagation_vector`` property used by older supercell-spin workflows;
- atom-site moments are read from dotted and underscored
  ``_atom_site_moment`` dialects, including fraction and uncertainty notation
  in numeric fields;
- moment labels are matched by exact atom label first, then by a unique
  ``label_...`` atom-label expansion.

When an MCIF-derived host is converted into a MuFinder magnetic model, each
labelled moment becomes an atom-targeted component. For finite propagation
vectors MuFinder stores a complex Fourier coefficient
``S(q) = m exp(2 pi i q.r_i)`` for atom ``i`` so that the model's evaluation
formula reconstructs the real MCIF unit-cell moment ``m`` at
``cell_offset=(0, 0, 0)``. This matches mag2spectra's MCIF reconstruction rule
while stopping at the field-distribution boundary: MuFinder does not import or
simulate spectra from MCIF files.

Builder UI
----------

The ``Magnetic Fields`` workflow page summarizes the accepted magnetic
structure and its ``Edit Magnetic Structure...`` button opens the ``Magnetic
Structure Builder`` dialog. The builder owns the component table, ``Preview
Supercell`` controls, atom selection, and live structure viewer:

- selecting no rows previews the full magnetic structure;
- selecting one or more rows previews only those components;
- diagonal preview supercell controls evaluate the magnetic moments outside the
  unit cell so longer-wavelength propagation vectors can be inspected;
- ``Build From Host Spins`` converts the current host-structure scalar spin
  metadata into the new propagation-vector model;
- ``Add Component`` creates a new propagation-vector component using a preset
  for collinear, spin-density-wave, helix, cycloid, or conical textures;
- ``Add Texture...`` creates one grouped multi-q texture assignment, currently
  supporting square and hexagonal skyrmion lattices as well as custom Fourier
  texture terms;
- ``Remove`` deletes selected components;
- ``Edit...`` reopens the selected component in the same preset-aware dialog so
  its label, target, q vector, phase, and texture parameters can be changed;
- both ``Delete`` and ``Backspace`` remove selected rows.

Components can currently be applied to all atoms, to atoms of a species, to a
symmetry-equivalent atom group, or to atoms selected in the builder preview. The
add-component dialog presents these choices as adaptive radio-button cards and
disables unavailable options.

Multi-q Textures
----------------

The texture builder follows the same formalism used internally by the
component builder, but it starts from an abstract spin texture before applying
that texture to selected atoms. A texture is a sum of complex Fourier terms:

.. math::

   \mathbf{m}(\mathbf{r}) =
   \Re\left[\sum_j \mathbf{S}_j
   \exp\left(-2\pi i\,\mathbf{q}_j\cdot\mathbf{r}\right)\right].

Each texture term stores ``q=(l,m,n)`` in a local propagation basis:

.. math::

   \mathbf{q}_j = l_j\mathbf{k}_1 + m_j\mathbf{k}_2 + n_j\mathbf{k}_3.

When the texture is accepted, MuFinder maps those local basis vectors onto the
magnetic model's three stored propagation-vector rows. This mapping is
transactional: a square skyrmion reserves two independent rows together, while a
hexagonal skyrmion reserves the two independent rows needed for
``(1,0,0)``, ``(0,1,0)``, and ``(-1,-1,0)``. The accepted texture appears as one
component row so that editing, preview selection, and atom targeting remain
intuitive.

The square skyrmion preset builds two helical terms with local wavevectors
``(1,0,0)`` and ``(0,1,0)``. The second physical q vector is generated by
rotating ``q1`` by 90 degrees in Cartesian reciprocal space about the supplied
core direction. The hexagonal preset builds three helical terms with local
wavevectors ``(1,0,0)``, ``(0,1,0)``, and ``(-1,-1,0)``; the second physical q
vector is generated by a 120 degree rotation. Bloch textures use in-plane
directions proportional to ``core x q``. Néel textures use directions
proportional to ``q``. The Fourier coefficients are always Cartesian moment
vectors in Bohr magnetons.

The texture dialog includes a 2D preview of the abstract texture in reduced
phase coordinates. The preview uses a repeated quiver plot: arrow direction
shows the in-plane moment component and arrow colour shows the out-of-plane
moment component, making skyrmion cores and helicity visible before the texture
is applied to atoms. If the texture has no out-of-plane component, the colour
scale falls back to moment magnitude. The main builder's 3D viewer then
previews the accepted texture on the actual host structure and selected
supercell. The preview canvas can be switched between light and dark modes
without changing the rest of the MuFinder interface, and the structure-viewer
backgrounds and axis triads follow the active application palette.

Moment Normalization and Pruning
--------------------------------

Linear multi-q superpositions can produce large spatial variations in
``|m|``. The texture builder therefore offers opt-in magnitude normalization
using a Fourier modulation step. MuFinder samples the texture on a regular
reduced-coordinate grid, estimates ``1 / |m|``, Fourier transforms that
correction, and convolves it with the original texture terms:

.. math::

   (\mathbf{S}_j,\mathbf{q}_j) \mapsto
   (\mathbf{S}_j f_\mathbf{G},\,\mathbf{q}_j+\mathbf{G}).

The result is an approximate constant-magnitude texture scaled to the amplitude
entered in the dialog. It usually contains more Fourier terms than the analytic
preset, so the dialog reports term counts and sampled ``|m|`` statistics before
and after normalization. Optional pruning can then keep terms above a
coefficient threshold or keep only the strongest ``N`` terms. Normalization is
never applied automatically; the accepted coefficients are exactly the
coefficients shown by the dialog diagnostics. The dialog includes an
information button in this section summarising the Fourier-modulation
algorithm, practical threshold choices, and the main caveats for near-zero
moment textures.

Muon-Site Selection
-------------------

The field-calculation page shows a compact summary of the selected muon site
and opens a dedicated selection dialog for choosing or editing it. The selected
site is normalized into one internal option containing a label, source,
fractional coordinate, optional Cartesian coordinate, and optional
distorted-structure reference. The available dialog sources are:

- UEP clustering results, using representative clustered UEP minima;
- structural-relaxation clustering results, using representative relaxed muon
  endpoints;
- a custom fractional coordinate in the active host cell.

Structural-relaxation cluster representatives can expose a distorted relaxed
structure. When that structure is available, the dialog enables the
``Include muon-induced displacements when available`` option. The dipolar-field
calculation does not yet apply ion-displacement corrections; the distorted
structure is currently used only by the ``Nuclear Relaxation...`` distortion
correction (see `Nuclear Relaxation`_).

Calculation
-----------

``Calculate Dipolar Field`` sends the selected magnetic structure and muon
site to the magnetic-field calculation helpers. Commensurate magnetic
structures produce finite vector field results. Incommensurate structures
produce sampled field-distribution summaries. The ``Calculation Settings...``
dialog exposes:

- ``Cells along`` ``a``, ``b``, and ``c``: the number of unit-cell repeats
  included in the ``muesr`` lattice summation, defaulting to
  ``100 x 100 x 100``;
- ``Lookup angles per q`` (default 4096), which sets the angular lookup
  resolution for each incommensurate Fourier component;
- ``Distribution samples`` (default 200,000), which caps or targets the number
  of sampled fields used for the final incommensurate distribution;
- ``Sampling method``, the incommensurate phase-space sampling method, with
  ``Auto`` (default), ``Grid``, and ``Monte Carlo`` options;
- ``Histogram bins`` (default 512), the number of bins used when summarising
  field magnitudes.

The sampling method affects only incommensurate field distributions. It does not
simulate muon spectra or time-dependent polarisation functions. ``muesr``/``muLFC``
environment errors are reported in the page status area rather than during
application startup.

Demagnetizing-Field Setup
-------------------------

The ``Demagnetizing Field`` card offers quick choices for the common limiting
cases and an ``Advanced shape`` option, configured through the ``Advanced...``
button, for shape-specific tensors:

``None``
   No demagnetizing-field correction is added.

``Sphere``
   Uses the isotropic tensor ``diag(1/3, 1/3, 1/3)``.

``Infinite plate``
   Uses the exact plate limit with demagnetizing factor 1 along the plate normal
   and 0 along the in-plane directions. The current simple card uses the
   crystal ``c`` axis as the plate normal.

``Advanced shape``
   ``Advanced...`` opens a dedicated shape window. The first implemented advanced shapes are
   ellipsoids and finite rectangular prisms. The window reports the tensor used
   by the calculation and renders the selected sample with a PyVista preview,
   including crystal-axis and magnetisation-direction arrows.

The advanced ellipsoid mode ports the analytical mag2spectra demagnetising
factor workflow. Spheres and infinite plates use exact limits, spheroids use
closed-form factors, and fully triaxial ellipsoids use the standard ellipsoid
integral. The principal axes are currently aligned with MuFinder's crystal-axis
frame.

The advanced rectangular-prism mode ports mag2spectra's analytical
Joseph–Schlömann rectangular-prism tensor (R. I. Joseph and E. Schlömann,
J. Appl. Phys. **36**, 1579 (1965), DOI: ``10.1063/1.1703091``) and averages it
over midpoint samples inside the prism. This avoids the resolution loss that a
single cubic Fourier grid would suffer for very thin plates or long needles. ``Fourier grid`` is
kept as the user-facing resolution control for the midpoint average; increasing
it improves the estimate but increases cost.

For field calculations, the selected tensor is inserted into the same correction
used by mag2spectra:

.. math::

   \mathbf{B}_\mathrm{demag} =
   -\frac{3}{2}\,N\,\mathbf{B}_\mathrm{Lor}.

This correction is a field-level contribution only. It does not simulate spectra
or time-dependent polarisation. For commensurate calculations it is applied once
to the finite Lorentz field. For incommensurate distributions, MuFinder first
combines the component Lorentz lookup tables at every sampled phase point, then
adds the demagnetizing correction point-by-point before applied fields and
histograms are formed.

Dipolar Tensors
---------------

The ``Dipolar Tensor...`` action opens a compact tensor workflow without adding
more controls to the main Magnetic Fields tab. A dipolar tensor calculation
requires a muon position and a set of magnetic-ion positions; it does not require
a particular magnetic structure or moment direction. MuFinder therefore reuses
the selected muon site from the Magnetic Fields page and lets the user choose
which host atoms should be treated as magnetic ions.

When the magnetic-structure builder already contains enabled components with
non-zero moments, the tensor dialog defaults to the atoms targeted by those
components. This gives the expected behaviour for ordinary workflows where the
magnetic ions have already been assigned in the structure builder. If no atoms
currently have assigned moments, MuFinder does not guess: the Calculate button
stays disabled until the user explicitly selects magnetic ions by element,
symmetry-equivalent site, all atoms, or atom index.

The result is shown inside the tensor dialog rather than the field results
table. Each selected or symmetry-equivalent muon site receives one ``3 x 3``
tensor. The dialog keeps the most recent tensor result for the current Magnetic
Fields workspace, and saved workspaces restore that latest result when reopened.
The displayed units can be switched between ``T/μB``, ``MHz/μB``, and ``Å⁻³``,
matching the hyperfine-coupling unit choices.

Only tensor-relevant numerical settings are shown in this workflow. The tensor
dialog reports the lattice-summation supercell used for the calculation, but it
does not expose incommensurate phase sampling, histogram bins, demagnetizing
field shape, or hyperfine coupling because those settings do not affect a
dipolar tensor. This keeps tensor coupling analysis separate from finite
dipolar-field and incommensurate field-distribution calculations while still
sharing the same host structure and muon-site selection.

Nuclear Relaxation
------------------

The ``Nuclear Relaxation...`` action estimates static nuclear dipolar
linewidths for the selected muon site. It is available whenever a host
structure and muon site are selected, and does not require an electronic
magnetic model. MuFinder uses a native implementation of the mag2spectra
second-moment workflow with bundled natural-abundance isotope data.

The supported modes are:

- zero-field single crystal;
- zero-field polycrystal;
- transverse-field single crystal;
- transverse-field polycrystal.

For single-crystal modes, the direction field is entered in crystal-axis
components and normalized before calculation. In TF this direction is the
applied-field axis. In ZF it is the initial muon-polarisation axis. Polycrystal
modes use analytical orientational averages and ignore the direction inputs.

The result dialog reports :math:`\sigma` in :math:`\mu\mathrm{s}^{-1}`,
:math:`\sigma^2` and :math:`M_2` in :math:`\mu\mathrm{s}^{-2}`, equivalent
field width :math:`\Delta B` in Tesla, convergence radius in Å, number of
nuclei summed, and element-resolved :math:`\sigma^2` contributions. For ZF
single-crystal calculations it also reports the perpendicular and parallel
widths :math:`\sigma_\perp^2` and :math:`\sigma_\parallel^2`.

All ZF widths use the Kubo-Toyabe per-component convention: :math:`\sigma^2`
is the variance :math:`\Delta^2 = \gamma_\mu^2 \langle B_i^2 \rangle` of a
single Cartesian field component, the quantity that enters the Kubo-Toyabe
polarisation function. In this convention the transverse single-crystal width
is :math:`\sigma_\perp^2 = \gamma_\mu^2 \langle B_x^2 + B_y^2 \rangle / 2` and
the longitudinal width is
:math:`\sigma_\parallel^2 = \gamma_\mu^2 \langle B_z^2 \rangle`, with
:math:`z` along the initial muon polarisation. The orientational (powder)
average of both single-crystal widths therefore equals the ZF polycrystal
width of R. S. Hayano *et al.*, Phys. Rev. B **20**, 850 (1979), DOI:
``10.1103/PhysRevB.20.850``, so single-crystal and
polycrystal results share one normalization.

If the selected muon site comes from a structural-relaxation cluster and the
representative relaxed structure is still available, the dialog can apply a
muon-induced distortion correction. MuFinder matches the active host to the
relaxed supercell, recentres pristine and distorted cells around the muon,
unwraps host-ion displacements across periodic boundaries, and adds the local
finite-cell correction to the converged pristine lattice sum:

.. math::

   \sigma^2_\mathrm{corrected} =
   \sigma^2_\mathrm{undistorted} +
   \sigma^2_\mathrm{distorted,sc} -
   \sigma^2_\mathrm{pristine,sc}.

The isotope table is bundled as MuFinder package data for reproducibility.
Natural abundances are used; isotope-enrichment editing and field-dependent TF
quadrupolar linewidth models are not part of this workflow. The optional
quadrupolar correction is available only for ZF modes.

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
   bounded, but the resulting histogram has statistical noise. MuFinder also
   carries an internal random-seed option for reproducible programmatic
   calculations.

``Auto``
   Chooses grid sampling for one or two active phase dimensions when the
   requested distribution sample count is below MuFinder's internal safety cap,
   and otherwise falls back to Monte Carlo sampling. This is the default because
   it gives deterministic low-dimensional distributions without accidentally
   constructing very large arrays.

The calculation parameters saved with an incommensurate result include the
chosen ``phase_sampling`` mode, the active ``independent_q_indices``, the
``phase_sample_count`` actually used, and the histogram settings. These fields
are intended to make convergence checks and later result persistence auditable.

Beyond the dipolar sum, the field calculation includes:

- the Lorentz field (always included in field calculations);
- the contact hyperfine term, when a ``Hyperfine coupling`` is entered;
- the demagnetizing field for commensurate finite-field calculations and
  incommensurate sampled distributions, exposed as none, sphere, infinite plate,
  advanced ellipsoid, or advanced rectangular prism.

Dipolar tensors from selected magnetic-ion positions and muon-induced distortion
corrections (nuclear relaxation only) are available through their own dialogs.

Dependency Handling
-------------------

``muesr`` and ``muLFC`` are optional scientific backends for periodic local-field
calculations. MuFinder imports these engines only when a calculation requires
them, so an environment without those packages raises a clear dependency error in
the Magnetic Fields workflow rather than failing during application startup. The
packaged desktop builds install these optional backends when they are available
for the target platform. On Windows, MuFinder follows the older project build
strategy and installs ``muLFC`` from the upstream Git source because PyPI does
not currently publish a Windows wheel or source distribution for that backend.
The import boundary also applies MuFinder's narrow compatibility shim for NumPy aliases
that current ``muesr`` releases still use internally.

Current Scope
-------------

The implemented workflow covers the propagation-vector data model, the
magnetic-structure builder and multi-q texture dialogs, commensurate local
fields, incommensurate field distributions, dipolar tensors, demagnetizing
corrections, and nuclear dipolar linewidths. It stops at field-level results:
MuFinder does not simulate muon spectra or polarisation functions, and the
dipolar-field calculation does not yet apply muon-induced displacement
corrections.
