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
2. click ``Select Muon Site...`` in the ``Muon Site`` card (see
   `Muon-Site Selection`_), and tick ``Include equivalent sites`` to calculate
   all symmetry-equivalent positions of that site;
3. in the ``Calculation`` card, choose a ``Demagnetizing field`` shape (see
   `Demagnetizing-Field Setup`_), review ``Calculation Settings...`` and click
   ``Calculate Dipolar Field``. Results appear in the table below (``Site``,
   ``Bx (T)``, ``By (T)``, ``Bz (T)``, ``|B| (T)``, ``Result``); a commensurate
   result row also lists its field terms (see `Field Terms`_), and for an
   incommensurate result, select its row and click ``Open Distribution Plot``
   to view the field-magnitude histogram, which ``Export Data...`` saves to a
   file;
4. optionally run ``Dipolar Tensor...`` or ``Nuclear Relaxation...`` from the
   same ``Calculation`` card.

.. figure:: _generated/screenshots/magnetic_fields_page.png
   :alt: The Magnetic Fields page with a ferromagnet's calculated field and its
         term breakdown.

   The ferromagnet of `Worked Examples`_ with a ``Sphere`` demagnetizing field.
   The selected result at the face centre ``(1/2, 1/2, 0)`` is broken down into
   its field terms.

The save button in the page header stores the current magnetic structure and
form settings as a named magnetic-field workspace. Saved workspaces appear as
children of ``Magnetic Fields`` in the project tree, can be renamed from their
page title, and can be deleted from their own page or by right-clicking their
project-tree node and choosing ``Delete Workspace...``.

Saved Results
-------------

The latest field, dipolar-tensor and nuclear-relaxation results of the
Magnetic Fields page and of each saved workspace are project results: each is
one ``Result`` record in the project, of type ``magnetic_field``,
``magnetic_dipolar_tensor`` or ``nuclear_relaxation``. The calculations run
inside MuFinder, so the records belong to no calculation run;
``data["owner"]`` is ``magnetic_fields`` for the page or the workspace id for
a saved workspace, and each owner has at most one record of each type.

- ``magnetic_field`` data holds ``kind`` (``commensurate`` or
  ``distribution``) and ``results``, one entry per site: a commensurate entry
  stores ``site_label``, ``muon_position``, ``total_T``, ``magnitude_T`` and its
  field terms (see `Field Terms`_), and a distribution entry stores
  ``fields_T``, the histogram, ``calculation_parameters`` and its
  ``convergence`` checks (see `Convergence Checks`_). The finite field list of
  a commensurate approximant (see `Commensurate Approximants`_) is stored as
  ``commensurate`` entries. ``site_source`` names the source of the muon site
  the fields were computed at (``uep_clustering``, ``relaxation_clustering``
  or ``custom``); ``uep_clustering`` marks the results as unrelaxed estimates
  (see `Muon-Site Selection`_). A record without it takes the source of the
  form's selected site.
- ``magnetic_dipolar_tensor`` data holds ``results``, the tensor dialog's rows.
- ``nuclear_relaxation`` data holds ``result``, the nuclear-relaxation result.

A new calculation replaces the owner's record of that type with a new record;
changing a setting that invalidates the field results, or clearing a dialog's
results, removes the record. Saving a workspace copies the page's results into
records the workspace owns, and deleting a workspace deletes its records. The
page's form settings stay in the project's UI state and do not repeat the
results.

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
Supercell`` controls, atom selection, a checklist and the live structure
viewer.

.. figure:: _generated/screenshots/magnetic_structure_builder.png
   :alt: The Magnetic Structure Builder showing a helix component, its
         checklist, q-basis table and a four-cell spin preview.

   The ``q = (0, 0, 1/4)`` helix of `Worked Examples`_, previewed over one
   whole magnetic cell (``1 x 1 x 4``).

The builder's parts are:

- the component table lists ``Enabled``, ``Component``, ``Type``,
  ``Apply to``, ``q`` and ``Moment`` for each component, top to bottom in the
  order the model stores them. The ``q`` column gives each distinct
  wavevector a component uses as its q-basis combination and its value in
  reciprocal lattice units, for example ``q1 = (0.25, 0, 0)`` or
  ``q1 − q2 = (0.25, -0.25, 0)``; a q = 0 term shows the vector alone;
- selecting no rows previews the full magnetic structure;
- selecting one or more rows previews only those components; when every
  selected component is disabled the preview shows no moments and says so;
- diagonal preview supercell controls evaluate the magnetic moments outside the
  unit cell so longer-wavelength propagation vectors can be inspected. Pressing
  ``Enter`` in a repeat box applies the value and keeps the builder open; Enter
  never presses ``Use Structure``;
- ``Build From Host Spins`` converts the current host-structure scalar spin
  metadata into the new propagation-vector model; when the builder already
  holds components it first asks for confirmation, because the conversion
  replaces them;
- ``Add Component`` creates a new propagation-vector component using a preset
  for collinear, spin-density-wave, helix, cycloid, or conical textures; its q
  comes from a typed vector or from the stored basis rows (see
  `Propagation-vector basis panel`_);
- ``Add Preset...`` builds a helix, conical, cycloid or spin-density-wave
  component from a propagation direction and the crystal axes (see
  `Preset helpers`_);
- ``Add Texture...`` creates one grouped multi-q texture assignment, currently
  supporting square and hexagonal skyrmion lattices as well as custom Fourier
  texture terms;
- ``Edit...`` reopens the selected component in the same preset-aware dialog so
  its label, target, q vector, phase, and texture parameters can be changed;
  double-clicking the ``Component`` cell renames a component in place;
- ``Duplicate`` inserts a copy of each selected component directly below its
  original. A copy keeps the original's target, Fourier terms, phase, enabled
  state and texture settings, shares its q-basis slots, and gets a
  ``(copy)`` label suffix; the copies are selected afterwards;
- ``Remove`` deletes the selected components after a confirmation whose
  default answer keeps them; ``Delete`` and ``Backspace`` do the same;
- ``Move Up`` and ``Move Down`` move the selected components one row, keeping
  their relative order and stopping at either end of the table. Order changes
  only how the builder lists components; the summed moments and fields do not
  depend on it;
- the ``Enabled`` checkbox switches a component off without deleting it. A
  disabled component keeps its row, shown in muted text, and is saved with the
  model, but it is left out of the preview, the structure type, the
  magnetic-ion defaults of the dipolar tensor dialog and every field
  calculation. Editing a component keeps its enabled state. When every
  component is disabled the preview and structure type are empty.

The ``Magnetic Structure`` card on the page reports how many components the
accepted model has and how many of them are disabled, and lists every enabled
component with its ``Type``, the atoms it applies to (``Apply to``), its ``q``
as a basis combination and r.l.u. vector, and its ``Moment``, using the same
descriptions as the builder table. The list shows up to eight rows and scrolls
beyond that. ``Calculate Dipolar Field`` stays disabled until at least one
component is enabled.

Components can currently be applied to all atoms, to atoms of a species, to a
symmetry-equivalent atom group, or to atoms selected in the builder preview. The
add-component dialog presents these choices as adaptive radio-button cards and
disables unavailable options.

Builder checklist
~~~~~~~~~~~~~~~~~

A ``Checklist`` panel above the component table ticks off the steps of
building a structure from the current model and preview; it is guidance only,
and nothing about it is saved. Each step is marked done (✓), still to do (○)
or a problem (⚠), with a one-line reason:

- ``Add components or build them from host spins`` is done once the model has
  an enabled component and no enabled component has zero moment. It is a problem when every
  component is disabled, or when an enabled component has zero moment, which
  it names;
- ``Set the propagation-vector basis`` is done when every row an enabled
  component uses holds a non-zero vector, or when every enabled component has
  q = 0 and no row is needed. A used row that holds no vector is a problem;
- ``Check the preview supercell`` compares the ``Preview Supercell`` repeats
  with the magnetic period along a, b and c, taken over every finite-q term of
  the enabled components. It is done when the preview shows at least one whole
  magnetic cell, or the largest preview (12 repeats) along an axis whose
  period is longer. It is a problem when a wavevector is incommensurate along
  an axis that the preview does not repeat, because the modulation cannot be
  seen; a repeat of two or more along every such axis completes the step;
- ``Use the structure`` is done when every other step is, and otherwise points
  at the flagged or unfinished steps.

Propagation-vector basis panel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every Fourier term stores its wavevector as a combination of the model's three
q-basis rows, ``q = l q1 + m q2 + n q3``. The ``Propagation-Vector Basis``
panel under the component table shows those rows:

- ``Row`` names the row, ``q1`` to ``q3``;
- ``q (r.l.u.)`` is the stored vector ``h, k, l``;
- ``Used by`` lists, in table order, every component with a term that
  references the row, marking disabled components ``(disabled)``, or says
  ``Unused``.

Selecting components in the component table shows the rows they use in bold,
with the row name in the accent colour, so a grouped texture's mapping onto the
stored basis can be read directly.

The add-component dialog shows the same rows, read-only, in its own
``Propagation-Vector Basis`` group, and its ``q from`` choice sets how the
component's q is stored:

- ``Enter a vector (r.l.u.)`` takes the typed ``q (r.l.u.)`` vector and
  assigns a row automatically: the vector reuses a row that already holds it,
  and otherwise fills a row that no other component uses. When every row is in
  use by other components, the builder refuses the new wavevector with a
  ``No q-basis slot available`` warning;
- ``qN = (h, k, l)``, offered for each row that holds a non-zero vector, stores
  the component on that row;
- ``Combination of basis rows`` shows ``Basis coefficients`` for ``q1``,
  ``q2`` and ``q3`` (rows without a vector stay disabled) and stores the
  component as that combination, for example ``q1 − q2``.

While a row or combination is chosen, the ``q (r.l.u.)`` fields show the
resulting vector read-only and a line beneath them gives the expression, such
as ``q = q1 − q2 = (-0.5, 0, 0.25)``. A combination with no coefficient, one
that uses a row without a vector, or one that sums to q = 0 is refused when
the dialog is accepted. A component stored on rows follows them when a row is
edited. ``Edit...`` reopens a component on the row or combination it is stored
as; a component whose only q is zero reopens on vector entry.

Double-clicking a ``q (r.l.u.)`` cell edits that row. The entry is three
numbers separated by commas or spaces, optionally in brackets. Components keep
their basis coefficients, so every component that uses the row changes
wavevector with it, and the ``q`` column, preview and structure type update at
once. An edit is refused with a warning, and the row keeps its vector, when:

- the entry is not three numbers;
- the row is used by a texture. A texture's moment directions are built from
  its q vectors, so the cell is read-only; edit the texture instead;
- the row is used by a helix, cycloid or conical component and the edit would
  turn any of that component's wavevectors. Those moment planes and cone axes
  are built around the q direction, so only the wavelength can change here: a
  component on a single row accepts any positive multiple of that row, and a
  component that combines rows, such as ``q1 + q2``, refuses any change to
  either of them;
- the row is used by a component and the new vector is zero;
- another row already holds the same vector.

An unused row accepts any vector. The edited basis is saved with the model when
the builder's ``Use Structure`` button is pressed.

Preset helpers
~~~~~~~~~~~~~~

``Add Preset...`` opens the ``Add Magnetic Preset`` dialog, which builds one
ordinary component from crystal-axis choices; the component can be edited like
any other afterwards. The dialog asks for:

- ``Structure``: ``Helix``, ``Conical``, ``Cycloid`` or ``Spin-density wave``;
- ``Propagation``: ``Along a*``, ``Along b*`` or ``Along c*`` with a
  ``Wavevector δ (r.l.u.)`` (default ``Along c*`` with δ = 0.25), or a stored
  basis row ``qN``, on which the component is then stored;
- a reference crystal axis, ``a``, ``b`` or ``c``. It defaults to the axis
  most nearly perpendicular to q, preferring a, then b, then c, and resets to
  that default when the ``Propagation`` choice changes, but not when only δ
  changes. For a helix and a conical
  structure the rotation plane is perpendicular to q and the moment at zero
  phase lies along the reference axis projected into that plane; a conical
  structure's cone axis is q. For a cycloid the rotation plane holds q and the
  reference axis. For a spin-density wave the moment is modulated along the
  reference axis;
- ``Handedness`` (not for a spin-density wave) and, for a conical structure,
  ``Cone angle (degrees)``, both with the same meaning as in the
  add-component dialog;
- ``Moment`` in ``μB``, ``Apply to`` (all atoms, an element, a
  symmetry-equivalent site or the preview selection) and a ``Label`` that
  follows the choices, such as ``Helix along c*``, until it is typed over.

A summary line describes the structure the choices give. Only a conical
structure reads the cone angle, so a hidden, invalid cone angle never blocks
another structure. A reference axis
parallel to q, a zero wavevector or a non-positive moment is reported there and
refused when the dialog is accepted. The new q is stored like a typed vector in
the add-component dialog, and the moments come from the same helix, cycloid,
conical and spin-density-wave builders.

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

- ``UEP clustering (unrelaxed estimate)``, using representative clustered UEP
  minima;
- ``Structural relaxation clustering``, using representative relaxed muon
  endpoints. By default this lists the clusters pinned to the ``Muon Sites``
  summary (:doc:`structural_relaxation_analysis`), in the summary's charge-state
  and energy order, with the ``Cluster`` column showing the summary label
  (``8c · cluster 1``). A live cluster matches a pin by its workspace and
  representative relaxation job, so re-clustering the analysis without
  re-pinning never relabels a different site. Tick ``Show every relaxed site`` to list every analysis
  workspace's clusters instead; the choice is saved with the project. With no
  pin, a notice and a ``Go to Muon Sites →`` button (which closes the dialog
  and opens that page) explain how to pin sites. Only analysis workspaces of
  the active host are listed;
- ``Custom fractional position``, a fractional coordinate in the active host
  cell.

Until a site is chosen, the page uses the lowest-energy pinned site (positive
muon first), else the first structural-relaxation cluster, else the first UEP
cluster, else a custom site at the origin. With no site chosen, the dialog
opens on ``Structural relaxation clustering`` when that source lists a site
(a pinned site, or any cluster with ``Show every relaxed site`` ticked), and on
UEP clustering otherwise.

UEP sites are the one exception to the rule that downstream workflows use
relaxed sites (see :doc:`project_specification`). A UEP minimum is unrelaxed,
and the relaxed muon site can sit some distance from it, so a field computed
there is a quick estimate made before any relaxation has run. While the UEP
source is chosen, the dialog shows a one-line note saying so, and the site
summary on the page repeats it. Fields computed at a UEP site are marked as
estimates wherever they are shown: the ``Result`` column reads, for example,
``field, unrelaxed estimate``, the results status line says the site is
unrelaxed, the distribution plot title ends in ``(unrelaxed estimate)``, its
``Export Data...`` file carries an ``# Estimate:`` header line, and the
project survey's local-field markers read ``≈`` before the magnitude. The
marker comes from the source saved with the result (see `Saved Results`_), so
it survives a reload.

Structural-relaxation cluster representatives can expose a distorted relaxed
structure. When that structure is available, the dialog enables the
``Correct the dipolar field for muon-induced displacements`` option; it is
disabled and cleared for every other site. The choice is stored with the
selected site, the site summary on the page notes it, and saved workspaces
restore it. When it is ticked, ``Calculate Dipolar Field`` applies the
correction described in `Muon-Induced Distortion Corrections`_. The
``Nuclear Relaxation...`` dialog has its own distortion option for the same
relaxed structure (see `Nuclear Relaxation`_).

Calculation
-----------

``Calculate Dipolar Field`` sends the selected magnetic structure and muon
site to the magnetic-field calculation helpers. Commensurate magnetic
structures produce finite vector field results. Incommensurate structures
produce sampled field-distribution summaries, or the finite field list of
their commensurate approximant when that option is on. The
``Calculation Settings...`` dialog exposes, under ``Numerical Parameters``:

- ``Cells along`` ``a``, ``b``, and ``c``: the number of unit-cell repeats
  included in the ``muesr`` lattice summation, defaulting to
  ``100 x 100 x 100``;
- ``Lookup angles per q`` (default 32,768), which sets the angular lookup
  resolution for each incommensurate Fourier component;
- ``Distribution samples`` (default 200,000), which caps or targets the number
  of sampled fields used for the final incommensurate distribution;
- ``Sampling method``, the incommensurate phase-space sampling method, with
  ``Auto`` (default), ``Grid``, and ``Monte Carlo`` options;
- ``Histogram bins`` (default 512), the number of bins used when summarising
  field magnitudes.

A note under these settings explains the convergence checks (see
`Convergence Checks`_). The ``Commensurate Approximants`` group holds:

- ``Report finite fields for near-commensurate q`` (default off), which lists
  the finite fields of a commensurate approximant instead of sampling a
  distribution when the structure has one;
- ``Tolerance`` (default 0.001 r.l.u.), the largest change of any wavevector
  component when it is replaced by a fraction;
- ``Longest period`` (default 12, at most 64), the largest denominator allowed
  for each wavevector component.

The settings summary on the ``Calculation`` card names the approximant
tolerance and period when the option is on.

The sampling method affects only incommensurate field distributions. It does not
simulate muon spectra or time-dependent polarisation functions. ``muesr``/``muLFC``
environment errors are reported in the page status area rather than during
application startup.

Muon-Induced Distortion Corrections
-----------------------------------

A muon displaces the host ions around it, so the dipolar field at the relaxed
site differs from the field computed for the pristine host. MuFinder corrects
the converged pristine lattice sum with the finite-cell change from those
displacements in the representative relaxed supercell:

.. math::

   \mathbf{B}_\mathrm{dip,corrected} =
   \mathbf{B}_\mathrm{dip}(\mathrm{pristine}) +
   \sum_{i \in \mathrm{cell}}
   \left[\mathbf{B}_i(\mathbf{r}_i + \mathbf{u}_i) - \mathbf{B}_i(\mathbf{r}_i)\right],

where :math:`\mathbf{r}_i` runs from the muon to the pristine site of host ion
:math:`i`, :math:`\mathbf{u}_i` is that ion's displacement relative to the
muon, and :math:`\mathbf{B}_i` is the point-dipole field of the moment the
magnetic model assigns to the pristine site. Every ion of the relaxed
supercell enters the sum; ions with no moment contribute nothing.

MuFinder pairs the relaxed supercell with the pristine host before it sums
anything, and reports the first check that fails:

- the relaxed lattice must be an integer supercell of the host lattice with a
  positive determinant, so the correction cell is finite;
- the relaxed structure must hold exactly one muon marker and, for every
  element, the host-atom count of that supercell of the pristine host;
- the selected site must lie within 0.5 Å of the relaxed muon after lattice
  translations, so the correction belongs to the site being calculated;
- every displaced ion is paired with the nearest pristine site of its
  element, and no two ions may share a pristine site.

The supercell is recentred on the muon and expressed in the pristine host
frame. Ion vectors start at the relaxed muon and pristine vectors at the
selected site, so a rigid drift of the relaxed cell cancels.

With ``Include equivalent sites``, each symmetry-equivalent site receives the
correction carried to it by the host symmetry operation that maps the
representative onto it: the displacement pattern is rotated with the
operation, and the moments are evaluated at the mapped pristine sites. The
magnetic order is not part of the host symmetry, so equivalent sites can
receive different corrections.

The correction is reported as its own component. Corrected rows read
``field (distortion-corrected)`` in the ``Result`` column, the vector total
includes the correction, and the saved result stores it as ``distortion_T``
beside the uncorrected ``dipolar_T``. The Lorentz, contact and demagnetizing
terms keep their pristine values. The correction applies to commensurate
structures only: an incommensurate field distribution or a commensurate
approximant with the option ticked stops with an error, and dipolar tensors
always use the pristine host.

Demagnetizing-Field Setup
-------------------------

The ``Demagnetizing field`` row at the top of the ``Calculation`` card offers
quick choices for the common limiting cases and an ``Advanced shape`` option,
configured through the ``Advanced...`` button, for shape-specific tensors:

``None``
   No demagnetizing-field correction is added.

``Sphere``
   Uses the isotropic tensor ``diag(1/3, 1/3, 1/3)``.

``Infinite plate``
   Uses the exact plate limit with demagnetizing factor 1 along the plate normal
   and 0 along the in-plane directions. This quick choice uses the crystal
   ``c`` axis as the plate normal.

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

MuFinder follows the ``muesr`` convention
(`muesr basic theory <https://muesr.readthedocs.io/en/latest/Intro.html>`_),
which splits the dipolar field of the whole sample into three terms:

.. math::

   \mathbf{B}_\mathrm{dip}' =
   \mathbf{B}_\mathrm{dip} + \mathbf{B}_\mathrm{Lor} + \mathbf{B}_\mathrm{demag}.

``Dipolar`` is the explicit sum over the ions inside the Lorentz sphere around
the muon. The ions outside it are treated as a magnetised continuum, whose field
at the muon comes from two surfaces:

- the surface of the Lorentz sphere gives the ``Lorentz`` field,
  :math:`\mathbf{B}_\mathrm{Lor} = \mu_0 \mathbf{M}_\mathrm{Lor} / 3`. ``muesr``
  takes :math:`\mathbf{M}_\mathrm{Lor}` as the sum of the moments inside the
  sphere divided by its volume;
- the surface of the sample gives the ``Demagnetizing`` field,

  .. math::

     \mathbf{B}_\mathrm{demag} = -\mu_0 N \mathbf{M}_\mathrm{meas},

  where :math:`\mathbf{M}_\mathrm{meas}` is the bulk magnetisation of the sample
  (S. J. Blundell, Contemp. Phys. **40**, 175 (1999), DOI:
  ``10.1080/001075199181521``).

``muesr`` leaves :math:`\mathbf{M}_\mathrm{meas}` to the user, because it
depends on the sample and the experiment. By default MuFinder takes it from the
magnetic model: it is the moment per unit cell of the uniform part of the order,
the Fourier components whose wavevector is a reciprocal lattice vector, divided
by the cell volume. Every other component averages to zero over the sample, so
an antiferromagnet or a helix has no demagnetizing field, whatever Lorentz field
the finite sphere leaves. This assumes the sample is a single magnetic domain,
or is saturated by an applied field. A multidomain ferromagnet in zero field has
a smaller bulk magnetisation, and the model value overestimates its
demagnetizing field.

The ``Measured bulk magnetisation`` row below the ``Demagnetizing field`` row
replaces the model value with a measured one, for example from magnetometry on
the same sample under the same conditions. Tick the box and enter the three
Cartesian components, ``x``, ``y`` and ``z`` in the frame of the host lattice
vectors, in Bohr magnetons per formula unit (μB/f.u.). The formula unit is the
reduced formula of the host's non-muon atoms, which the row names, for example
``μB per Fe2O3``; the cell holds :math:`Z` of them, the greatest common divisor
of its element counts. MuFinder then uses

.. math::

   \mathbf{M}_\mathrm{meas} = \frac{Z\,\mathbf{m}_\mathrm{f.u.}\,\mu_B}{V_\mathrm{cell}},

so a supercell of the host gives the same field as its primitive cell. A
magnetisation quoted per magnetic ion becomes μB/f.u. by multiplying by the
number of magnetic ions in the formula unit. An empty component counts as zero,
and a component that is not a finite decimal number stops the calculation with
an error. The measured value replaces the model value on the commensurate,
commensurate-approximant and incommensurate paths alike, and has no effect when
the demagnetizing field is ``None``. Unticking the box returns to the model value
and keeps the entered components. The form state saves the choice as
``use_measured_magnetization`` and the components as ``measured_magnetization``;
an incommensurate distribution records the value it used as
``measured_magnetization`` in its ``calculation_parameters``, or ``null`` when
it used the model value.

For a sphere, :math:`N = \tfrac{1}{3}`, so a single-domain sample's Lorentz and
demagnetizing fields cancel up to the difference between
:math:`\mathbf{M}_\mathrm{Lor}` and :math:`\mathbf{M}_\mathrm{meas}`. That
difference comes from counting the ions inside a finite sphere and shrinks as the
sphere grows; it is about 0.1 % of the Lorentz field in the
`Worked Examples`_. For an infinite plate magnetised along its normal,
:math:`N = 1` and the two terms sum to about
:math:`-2\mu_0 \mathbf{M}_\mathrm{meas} / 3`.

This correction is a field-level contribution only. It does not simulate spectra
or time-dependent polarisation. The bulk magnetisation does not depend on the
muon site or on the magnetic phase, so the demagnetizing field is the same for
every site of a commensurate calculation and for every sampled phase point of an
incommensurate distribution; it is added before applied fields and histograms
are formed.

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
matching the hyperfine-coupling unit choices. ``muesr`` returns the lattice sum
:math:`\sum (3\hat{\mathbf{r}}\hat{\mathbf{r}} - 1)/r^3` in ``Å⁻³``; MuFinder
multiplies it by :math:`\mu_0 \mu_B / (4\pi\,\text{Å}^3) = 0.9274` T/μB, so the
tensor applied to a moment in μB gives the dipolar field in tesla. The ``Å⁻³``
display uses the hyperfine-coupling convention,
:math:`\mathbf{B} = (2\mu_0/3)\,\mu_B A\,\mathbf{m}`, so it shows the contact
coupling that would give the same field rather than ``muesr``'s raw lattice sum.
Tensors saved in a project at schema 4 or earlier hold the raw ``Å⁻³`` sum
and are converted to T/μB when the project is loaded (see
:doc:`compatibility_policy`).

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
muon-induced distortion correction. MuFinder pairs the relaxed supercell with
the active host exactly as for the dipolar field, with the same checks (see
`Muon-Induced Distortion Corrections`_), places the displaced and pristine ions
around a muon at the centre of the correction cell, and adds the local
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
chosen ``phase_sampling`` mode, the ``resolved_phase_sampling`` it ran as
(``grid`` or ``montecarlo``), the active ``independent_q_indices``, the
``phase_sample_count`` actually used, and the histogram settings.

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
- the contact hyperfine term, when a ``Hyperfine coupling`` is entered;
- the demagnetizing field for commensurate finite-field calculations and
  incommensurate sampled distributions, exposed as none, sphere, infinite plate,
  advanced ellipsoid, or advanced rectangular prism;
- the muon-induced distortion correction for commensurate fields, when the
  selected site requests it (see `Muon-Induced Distortion Corrections`_).

Dipolar tensors from selected magnetic-ion positions and the nuclear-relaxation
distortion correction are available through their own dialogs.

Field Terms
~~~~~~~~~~~

Selecting a commensurate result row lists, under the results table, the terms
that sum to its field: ``Dipolar``, ``Lorentz``, ``Contact``,
``Demagnetizing``, ``Distortion correction`` (only for corrected rows) and
``Applied``, each as ``Bx (T)``, ``By (T)``, ``Bz (T)`` and ``|B| (T)``, followed
by their ``Total``. A newly calculated or restored commensurate result selects
its first row. With several rows selected, the first selected row is shown. A
saved result that stores only its total asks for a recalculation instead.
Incommensurate distributions are sampled as one combined field, so they show
their convergence checks (see `Convergence Checks`_) and a note instead of a
breakdown.

The results table and the term breakdown show fields to six significant
figures. A component no larger than 10\ :sup:`-12` T, or than 10\ :sup:`-9` of
its row's magnitude, is rounding noise from the lattice sum and shows as ``0``,
and no cell shows ``-0``. For a distribution row the magnitude is the mean
``|B|``. The saved result keeps the calculated values unrounded.

Worked Examples
---------------

The examples below use one magnetic ion of 1 μB in a simple cubic cell with
``a = 3 Å`` and the default ``Calculation Settings`` (``100 x 100 x 100``
cells, Lorentz sphere chosen by ``muesr``). Positions are fractional. Each
number comes from the calculation and is pinned by
``tests/science/test_magnetic_fields.py``.

Ferromagnet
~~~~~~~~~~~

With the moment along ``c`` and ``q = 0``, the magnetisation is
:math:`M = \mu_B / a^3`, and the Lorentz field
:math:`\mu_0 M / 3 = \mu_0 \mu_B / (3a^3)` is 0.1439 T along ``c``. MuFinder
reports 0.1437 T because ``muesr`` counts the ions inside a finite Lorentz
sphere. The bulk magnetisation gives :math:`\mu_0 M = 0.4316` T. With
``Demagnetizing field`` set to ``Sphere``, the demagnetizing term is -0.1439 T,
and the Lorentz and demagnetizing terms sum to -0.0002 T, the finite-sphere
difference. With ``Infinite plate`` it is -0.4316 T, and the two sum to
-0.2879 T.

.. list-table::
   :header-rows: 1

   * - Site
     - Dipolar ``Bz`` (T)
     - Lorentz ``Bz`` (T)
     - Total ``Bz`` (T)
   * - ``(1/2, 1/2, 1/2)``
     - 0.0000
     - 0.1437
     - 0.1437
   * - ``(1/2, 1/2, 0)``
     - -0.2977
     - 0.1438
     - -0.1538
   * - ``(1/2, 0, 0)``
     - -0.5166
     - 0.1438
     - -0.3728

Cubic symmetry cancels the dipolar sum at the body centre. A ``Hyperfine
coupling`` of 0.5 T/μB adds a contact field of 0.5 T along the moment there, for
a total of 0.6437 T.

The dipolar tensor at the face centre ``(1/2, 1/2, 0)`` is
``diag(0.1488, 0.1488, -0.2977)`` T/μB and at the edge centre ``(1/2, 0, 0)``
is ``diag(1.0332, -0.5166, -0.5166)`` T/μB. Each is traceless, and its ``zz``
element times 1 μB is the ferromagnet's dipolar field at that site.

A-type antiferromagnet
~~~~~~~~~~~~~~~~~~~~~~

With ``q = (0, 0, 1/2)`` the ``ab`` layers alternate along ``c`` and the
Lorentz field vanishes. At the body centre the two neighbouring layers cancel,
so the field is zero. The face centre ``(1/2, 1/2, 0)`` sees -0.5493 T and the
edge centre ``(1/2, 0, 0)`` -0.8003 T along ``c``.

Commensurate helix
~~~~~~~~~~~~~~~~~~

A helix with ``q = (0, 0, 1/4)`` and coefficient
:math:`\mathbf{F} = \hat{\mathbf{a}} + i\,\hat{\mathbf{b}}` turns the moment by
90° in the ``ab`` plane from one cell to the next along ``c``. Its magnetic
cell is four host cells long, so a muon site and its translations ``r + T``
can see different fields. ``Report finite fields for near-commensurate q``
lists them:

- on the fourfold axis at ``(1/2, 1/2, 0)`` every copy sees 0.2836 T, turning
  by 90° from one copy to the next;
- at the edge centre ``(1/2, 0, 0)`` the copies alternate between 1.1660 T
  along ``±a`` and 0.3637 T along ``±b``. Half of the muons see each magnitude.

Incommensurate helix
~~~~~~~~~~~~~~~~~~~~

Changing the wavevector to ``q = (0, 0, 0.2861)`` makes the helix
incommensurate, and ``Calculate Dipolar Field`` samples a distribution. For one
wavevector the sampled field traces an ellipse,
:math:`\mathbf{B}(\theta) = \cos\theta\,\mathbf{B}_0 + \sin\theta\,\mathbf{B}_{90}`,
so :math:`p(|B|)` runs between the two semi-axes and peaks at both ends. At
``(1/2, 1/2, 0)`` the fourfold axis makes the ellipse a circle and the
distribution a single line at 0.2816 T. At ``(1/2, 0, 0)`` it spans 0.3616 T to
1.1634 T, close to the two fields of the ``q = 1/4`` helix above.

.. figure:: _generated/screenshots/magnetic_field_distribution.png
   :alt: The field-distribution plot of the incommensurate helix at the edge
         centre, peaking at both ends of its range.

   ``Open Distribution Plot`` for the incommensurate helix at ``(1/2, 0, 0)``.

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
Every ``muesr`` import goes through one compatibility boundary,
``mufinder.science._muesr_compat.muesr_imports``, which keeps the released
``muesr`` working on every Python version MuFinder supports:

- it restores the NumPy ``complex`` and ``float`` aliases that ``muesr`` still
  uses internally;
- on Python 3.13 and later, where the standard library no longer ships ``cgi``,
  it supplies an empty ``cgi`` module while ``muesr`` is imported and removes it
  again afterwards. ``muesr`` imports ``cgi`` only for the HTML output of its
  bundled table printer, which MuFinder never calls;
- it silences the ``cgi`` deprecation warning that Python 3.12 raises during
  that import.

Current Scope
-------------

The implemented workflow covers the propagation-vector data model, the
magnetic-structure builder and multi-q texture dialogs, commensurate local
fields with optional muon-induced distortion corrections, incommensurate field
distributions with convergence checks, finite field lists of commensurate
approximants, dipolar tensors, demagnetizing corrections, and nuclear dipolar
linewidths. It stops at field-level results: MuFinder does not simulate muon
spectra or polarisation functions. Distortion corrections are not applied to
incommensurate field distributions, commensurate approximants or dipolar
tensors.
