Magnetic Fields
===============

The ``Magnetic Fields`` workflow builds magnetic structures and calculates
local fields at candidate muon stopping sites. Use it to define the magnetic
order of the host, pick a muon site, and read off the dipolar, Lorentz,
contact and demagnetizing fields there. The physics behind the numbers, with
worked examples, is on :doc:`/user_guide/magnetic_fields_theory`.

Before you start
----------------

- Load a host structure. Without one, the editing and calculation buttons on
  the page are disabled.
- For a muon site beyond a custom position, finish UEP clustering or
  structural-relaxation clustering first, because those results supply the
  sites you choose from (see `Choose the muon site`_).
- The calculations need the optional ``muesr`` and ``muLFC`` backends (see
  `Dependency handling`_).
- ``Calculate Dipolar Field`` stays disabled until the magnetic structure has
  at least one enabled component.

Calculate a field
-----------------

.. figure:: ../_generated/screenshots/fe_magnetic_fields_page.webp
   :alt: The Magnetic Fields page for bcc iron at the octahedral site, with the
         six equivalent sites' fields and the dipolar, Lorentz, contact and
         demagnetizing terms of the selected one.

   bcc iron (:doc:`/user_guide/examples/fe`) at the octahedral site
   ``(1/2, 0, 0)`` with every equivalent site, a contact field of -1.11 T and a
   ``Sphere`` demagnetizing field. The selected result is broken down into its
   field terms.

A typical pass through the page is:

1. In the ``Magnetic Structure`` card, click ``Edit Magnetic Structure...`` to
   open the ``Magnetic Structure Builder`` and define the magnetic model (see
   `Define the magnetic structure`_). Optionally enter a ``Hyperfine
   coupling`` in ``Å⁻³``, ``T/μB``, or ``MHz/μB`` for the contact term.
2. In the ``Muon Site`` card, click ``Select Muon Site...`` (see `Choose the
   muon site`_). Tick ``Include equivalent sites`` to calculate all
   symmetry-equivalent positions of that site.
3. In the ``Calculation`` card, choose a ``Demagnetizing field`` shape (see
   `Set the demagnetizing field`_) and review ``Calculation Settings...`` (see
   `Set the calculation parameters`_).
4. Click ``Calculate Dipolar Field``. Results appear in the table below
   (``Site``, ``Bx (T)``, ``By (T)``, ``Bz (T)``, ``|B| (T)``, ``Result``).
5. Read the result:

   - A commensurate result row also lists its field terms (see
     :ref:`Field Terms <magnetic-fields-field-terms>`): select the row to see
     them under the table.
   - An incommensurate result row lists the mean and RMS magnitude of each
     field term over the sampled phases, with its convergence checks: select
     the row to see them under the table. Click ``Open Distribution Plot`` to
     view the field-magnitude histogram. ``Export Data...`` in that window
     saves it to a file.

   .. figure:: ../_generated/screenshots/mnsi_distribution_breakdown.webp
      :alt: The Magnetic Fields page with the MnSi helix distribution at site
            II selected: a mean |B| of 0.154303 T, every term mean 0, RMS
            magnitudes of 0.0809557 T dipolar, 0.000360636 T Lorentz,
            0.194362 T contact and 0.159547 T in total, and three passing
            convergence checks.

      The distribution of the helix in :doc:`/user_guide/examples/mnsi` at
      site II. Every term averages to zero over the phase of the helix; the
      RMS column shows that the contact term is the largest.

6. Optionally run ``Dipolar Tensor...`` or ``Nuclear Relaxation...`` from the
   same ``Calculation`` card (see `Calculate a dipolar tensor`_ and `Calculate
   nuclear relaxation`_).

``Calculate Dipolar Field`` sends the selected magnetic structure and muon site
to the magnetic-field calculation helpers. Commensurate magnetic structures
produce finite vector field results. Incommensurate structures produce sampled
field-distribution summaries, or the finite field list of their commensurate
approximant when that option is on. ``muesr``/``muLFC`` environment errors are
reported in the page status area rather than during application startup.

Define the magnetic structure
-----------------------------

MuFinder stores a magnetic structure as a set of components, each a Fourier
term with a propagation vector, that apply to chosen atoms (the formalism is on
:doc:`/user_guide/magnetic_fields_theory`). You build the model in the
``Magnetic Structure Builder``, then accept it with ``Use Structure``.

The ``Magnetic Structure`` card on the page reports how many components the
accepted model has and how many of them are disabled, and lists every enabled
component with its ``Type``, the atoms it applies to (``Apply to``), its ``q``
as a basis combination and r.l.u. vector, and its ``Moment``, using the same
descriptions as the builder table. The list shows up to eight rows and scrolls
beyond that.

.. figure:: ../_generated/screenshots/cof2_magnetic_structure_builder.webp
   :alt: The Magnetic Structure Builder with two collinear q = 0 components on
         the two Co of CoF2, with moments of +2.64 and -2.64 muB along c.

   The collinear antiferromagnet of :doc:`/user_guide/examples/cof2` as one
   component per Co sublattice.

Import a structure from an MCIF file
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To start from a magnetic CIF, import the ``.mcif`` file as the host structure
(see :doc:`/user_guide/host_structure` for the import itself), then open the
builder. MuFinder imports magnetic metadata from ``.mcif`` files before any
field or spectral calculation is attempted:

- the magnetic symmetry operations and centerings are applied to the listed
  sites, so the host holds every atom of the magnetic cell and every image of a
  magnetic site carries its moment, reversed where the operation includes time
  reversal (see :doc:`/user_guide/host_structure`);
- atom labels from ``_atom_site_label`` are preserved so magnetic-site labels
  can be mapped back to concrete unit-cell atom indices; each generated atom
  keeps the label of the site it was generated from;
- propagation vectors are read from both legacy scalar tags such as
  ``_cell_wave_vector_x/y/z`` and MCIF triplet entries such as
  ``_magnetic_propagation_vector_kxkykz``;
- multiple imported propagation vectors are mapped onto the model's first three
  propagation-vector basis rows;
- atom-site moments are read from dotted and underscored
  ``_atom_site_moment`` dialects, including fraction and uncertainty notation
  in numeric fields, and converted to Cartesian vectors in μB:
  ``crystalaxis`` components lie along the unit lattice directions,
  ``m = Σ m_i a_i/|a_i|``, and ``cartn`` components are already Cartesian;
- moment labels are matched by exact atom label first, then by a unique
  ``label_...`` atom-label expansion.

When an MCIF-derived host is converted into a MuFinder magnetic model, each
atom with a moment becomes an atom-targeted component, whose metadata records
the coordinate system the file gave its components in. For finite propagation
vectors MuFinder stores a complex Fourier coefficient
``S(q) = m exp(2 pi i q.r_i)`` for atom ``i`` so that the model's evaluation
formula (see :doc:`/user_guide/magnetic_fields_theory`) reconstructs the real
MCIF unit-cell moment ``m`` at ``cell_offset=(0, 0, 0)``. MuFinder does not
import or simulate spectra from MCIF files.

Build components in the builder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/mnsi_magnetic_structure_builder.webp
   :alt: The Magnetic Structure Builder showing a left-handed helix component on
         the Mn atoms of MnSi, a checklist with every step ticked, the q-basis
         table and a 2 x 2 x 2 spin preview.

   The helix of :doc:`/user_guide/examples/mnsi`, a 0.385 μB Mn moment turning
   about q = (0.0147, 0.0147, 0.0147), previewed over ``2 x 2 x 2`` cells. Its
   checklist is complete, and the structure type is ``incommensurate``.

The ``Magnetic Structure Builder`` owns the component table, ``Preview
Supercell`` controls, atom selection, a checklist and the live structure
viewer. The component table lists ``Enabled``, ``Component``, ``Type``,
``Apply to``, ``q`` and ``Moment`` for each component, top to bottom in the
order the model stores them. The ``q`` column gives each distinct wavevector a
component uses as its q-basis combination and its value in reciprocal lattice
units, for example ``q1 = (0.25, 0, 0)`` or ``q1 − q2 = (0.25, -0.25, 0)``; a
q = 0 term shows the vector alone.

To build a structure:

1. Click ``Edit Magnetic Structure...`` on the page. If the host carries scalar
   spin metadata, click ``Build From Host Spins`` to convert it into the
   propagation-vector model. When the builder already holds components it first
   asks for confirmation, because the conversion replaces them.
2. Add components:

   - ``Add Component`` creates a new propagation-vector component using a
     preset for collinear, spin-density-wave, helix, cycloid, or conical
     textures; its q comes from a typed vector or from the stored basis rows
     (see `Set the propagation vectors`_).
   - ``Add Preset...`` builds a helix, conical, cycloid or spin-density-wave
     component from a propagation direction and the crystal axes (see `Use the
     preset helper`_).
   - ``Add Texture...`` creates one grouped multi-q texture assignment,
     currently supporting square and hexagonal skyrmion lattices as well as
     custom Fourier texture terms (see `Add a multi-q texture`_).

   Components can be applied to all atoms, to atoms of a species, to a
   symmetry-equivalent atom group, or to atoms selected in the builder preview.
   The add-component dialog presents these choices as adaptive radio-button
   cards and disables unavailable options.

3. Adjust the table:

   - ``Edit...`` reopens the selected component in the same preset-aware dialog
     so its label, target, q vector, phase, and texture parameters can be
     changed. Double-clicking the ``Component`` cell renames a component in
     place.
   - ``Duplicate`` inserts a copy of each selected component directly below its
     original. A copy keeps the original's target, Fourier terms, phase,
     enabled state and texture settings, shares its q-basis slots, and gets a
     ``(copy)`` label suffix; the copies are selected afterwards.
   - ``Remove`` deletes the selected components after a confirmation whose
     default answer keeps them; ``Delete`` and ``Backspace`` do the same.
   - ``Move Up`` and ``Move Down`` move the selected components one row,
     keeping their relative order and stopping at either end of the table.
     Order changes only how the builder lists components; the summed moments
     and fields do not depend on it.
   - Untick the ``Enabled`` checkbox to switch a component off without deleting
     it. A disabled component keeps its row, shown in muted text, and is saved
     with the model, but it is left out of the preview, the structure type, the
     magnetic-ion defaults of the dipolar tensor dialog and every field
     calculation. Editing a component keeps its enabled state. When every
     component is disabled the preview and structure type are empty.

4. Check the preview:

   - Selecting no rows previews the full magnetic structure. Selecting one or
     more rows previews only those components; when every selected component is
     disabled the preview shows no moments and says so.
   - The diagonal ``Preview Supercell`` controls evaluate the magnetic moments
     outside the unit cell so longer-wavelength propagation vectors can be
     inspected. Pressing ``Enter`` in a repeat box applies the value and keeps
     the builder open; Enter never presses ``Use Structure``.

5. Work through the checklist (see `Follow the builder checklist`_), then click
   ``Use Structure``.

Follow the builder checklist
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Set the propagation vectors
~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

.. figure:: ../_generated/screenshots/mnsi_q_basis.webp
   :alt: The Propagation-Vector Basis panel with row q1 holding 0.0147, 0.0147,
         0.0147 in bold and used by Helix along q1, and q2 and q3 unused.

   The panel with the helix of :doc:`/user_guide/examples/mnsi` selected: it
   uses ``q1`` only, and ``q2`` and ``q3`` are unused.

To choose a component's q, use ``Add Component`` (or ``Edit...``). The
add-component dialog shows the same rows, read-only, in its own
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

.. figure:: ../_generated/screenshots/mnsi_q_picker.webp
   :alt: The component dialog for a left-handed helix, with q from set to q1 =
         (0.0147, 0.0147, 0.0147) and the q fields shown read-only.

   ``Edit...`` on the helix reopens it on the row it is stored as: ``q from``
   reads ``q1 = (0.0147, 0.0147, 0.0147)``, and the component follows the row if
   the row is edited.

While a row or combination is chosen, the ``q (r.l.u.)`` fields show the
resulting vector read-only and a line beneath them gives the expression, such
as ``q = q1 − q2 = (-0.5, 0, 0.25)``. A combination with no coefficient, one
that uses a row without a vector, or one that sums to q = 0 is refused when
the dialog is accepted. A component stored on rows follows them when a row is
edited. ``Edit...`` reopens a component on the row or combination it is stored
as; a component whose only q is zero reopens on vector entry.

To change a stored row directly, double-click its ``q (r.l.u.)`` cell. The
entry is three numbers separated by commas or spaces, optionally in brackets.
Components keep their basis coefficients, so every component that uses the row
changes wavevector with it, and the ``q`` column, preview and structure type
update at once. An edit is refused with a warning, and the row keeps its
vector, when:

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

Use the preset helper
~~~~~~~~~~~~~~~~~~~~~

``Add Preset...`` opens the ``Add Magnetic Preset`` dialog, which builds one
ordinary component from crystal-axis choices; the component can be edited like
any other afterwards. In the dialog, set:

1. ``Structure``: ``Helix``, ``Conical``, ``Cycloid`` or ``Spin-density wave``.
2. ``Propagation``: ``Along a*``, ``Along b*`` or ``Along c*`` with a
   ``Wavevector δ (r.l.u.)`` (default ``Along c*`` with δ = 0.25), or a stored
   basis row ``qN``, on which the component is then stored.
3. A reference crystal axis, ``a``, ``b`` or ``c``. It defaults to the axis
   most nearly perpendicular to q, preferring a, then b, then c, and resets to
   that default when the ``Propagation`` choice changes, but not when only δ
   changes. For a helix and a conical structure the rotation plane is
   perpendicular to q and the moment at zero phase lies along the reference
   axis projected into that plane; a conical structure's cone axis is q. For a
   cycloid the rotation plane holds q and the reference axis. For a
   spin-density wave the moment is modulated along the reference axis.
4. ``Handedness`` (not for a spin-density wave) and, for a conical structure,
   ``Cone angle (degrees)``, both with the same meaning as in the
   add-component dialog.
5. ``Moment`` in ``μB``, ``Apply to`` (all atoms, an element, a
   symmetry-equivalent site or the preview selection) and a ``Label`` that
   follows the choices, such as ``Helix along c*``, until it is typed over.

.. figure:: ../_generated/screenshots/mnsi_preset_helper.webp
   :alt: The Add Magnetic Preset dialog set to a left-handed helix along basis
         row q1 = (0.0147, 0.0147, 0.0147), with the moment at zero phase along
         crystal axis a, 0.385 μB, applied to the species Mn.

   The helix of :doc:`/user_guide/examples/mnsi`. Its q has no crystal-axis
   choice, so it is typed into row ``q1`` of the basis first and the preset
   propagates along that row.

A summary line describes the structure the choices give. Only a conical
structure reads the cone angle, so a hidden, invalid cone angle never blocks
another structure. A reference axis parallel to q, a zero wavevector or a
non-positive moment is reported there and refused when the dialog is accepted.
The new q is stored like a typed vector in the add-component dialog, and the
moments come from the same helix, cycloid, conical and spin-density-wave
builders.

Add a multi-q texture
~~~~~~~~~~~~~~~~~~~~~

To add a skyrmion lattice or another multi-q texture, click ``Add Texture...``
in the builder. The ``Add Magnetic Texture`` dialog starts from an abstract spin
texture before applying it to selected atoms:

1. Enter a ``Label`` and the ``Amplitude`` in the ``Texture`` group.
2. Choose the ``Texture Type`` (``Hexagonal skyrmion``, ``Square skyrmion`` or
   ``Custom Fourier texture``), set the ``Propagation Basis`` and, for a
   skyrmion, the ``Skyrmion Parameters`` (for example Bloch or Neel). A custom
   texture takes its terms from the ``Custom Fourier Terms`` table, where
   ``Add Fourier Term`` adds a row.
3. Optionally normalise and prune the texture (see `Normalise a texture`_).
4. Choose what the texture applies to in ``Apply Texture To`` (``All atoms``,
   ``Element``, ``Equivalent site`` or ``Current selection``).
5. Check the ``Spin Texture Preview`` and accept the dialog.

The dialog's 2D preview shows the abstract texture in reduced phase
coordinates as a repeated quiver plot: arrow direction shows the in-plane
moment component and arrow colour shows the out-of-plane moment component,
making skyrmion cores and helicity visible before the texture is applied to
atoms. If the texture has no out-of-plane component, the colour scale falls
back to moment magnitude. The main builder's 3D viewer then previews the
accepted texture on the actual host structure and selected supercell. The
preview canvas can be switched between light and dark modes without changing
the rest of the MuFinder interface, and the structure-viewer backgrounds and
axis triads follow the active application palette.

The accepted texture appears as one component row so that editing, preview
selection, and atom targeting remain intuitive; ``Edit...`` reopens it in the
texture dialog.

How a texture is defined
^^^^^^^^^^^^^^^^^^^^^^^^

The texture builder follows the same formalism used internally by the
component builder. A texture is a sum of complex Fourier terms:

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
``(1,0,0)``, ``(0,1,0)``, and ``(-1,-1,0)``.

The square skyrmion preset builds two helical terms with local wavevectors
``(1,0,0)`` and ``(0,1,0)``. The second physical q vector is generated by
rotating ``q1`` by 90 degrees in Cartesian reciprocal space about the supplied
core direction. The hexagonal preset builds three helical terms with local
wavevectors ``(1,0,0)``, ``(0,1,0)``, and ``(-1,-1,0)``; the second physical q
vector is generated by a 120 degree rotation. Bloch textures use in-plane
directions proportional to ``core x q``. Néel textures use directions
proportional to ``q``. The Fourier coefficients are always Cartesian moment
vectors in Bohr magnetons.

Normalise a texture
^^^^^^^^^^^^^^^^^^^

Linear multi-q superpositions can produce large spatial variations in ``|m|``.
The ``Magnitude and Pruning`` group of the texture dialog therefore offers
opt-in magnitude normalization: tick ``Normalize moment magnitude``, then set
``Grid``, ``Threshold`` and, optionally, ``Pruning`` (``No pruning``,
``Coefficient threshold`` or ``Keep strongest N``, with ``Prune threshold`` or
``Keep N``). Normalization is never applied automatically; the accepted
coefficients are exactly the coefficients shown by the dialog diagnostics. The
information button in this group summarises the Fourier-modulation algorithm,
practical threshold choices, and the main caveats for near-zero moment
textures.

MuFinder samples the texture on a regular reduced-coordinate grid, estimates
``1 / |m|``, Fourier transforms that correction, and convolves it with the
original texture terms:

.. math::

   (\mathbf{S}_j,\mathbf{q}_j) \mapsto
   (\mathbf{S}_j f_\mathbf{G},\,\mathbf{q}_j+\mathbf{G}).

The result is an approximate constant-magnitude texture scaled to the amplitude
entered in the dialog. It usually contains more Fourier terms than the analytic
preset, so the dialog reports term counts and sampled ``|m|`` statistics before
and after normalization. Optional pruning can then keep terms above a
coefficient threshold or keep only the strongest ``N`` terms.

.. _magnetic-fields-muon-site:

Choose the muon site
--------------------

The ``Muon Site`` card shows a compact summary of the selected muon site. To
choose or edit it, click ``Select Muon Site...``, pick a ``Source`` in the
``Select Muon Site`` dialog and choose a row in the ``Site`` table (columns
``Site``, ``Source``, ``Cluster``, ``fx``, ``fy``, ``fz``), then click ``OK``.
The selected site is normalized into one internal option containing a label,
source, fractional coordinate, optional Cartesian coordinate, and optional
distorted-structure reference. The available sources are:

- ``UEP clustering (unrelaxed estimate)``, using representative clustered UEP
  minima;
- ``Structural relaxation clustering``, using representative relaxed muon
  endpoints. By default this lists the clusters pinned to the ``Muon Sites``
  summary (:doc:`/user_guide/structural_relaxation_analysis`), in the summary's charge-state
  and energy order, with the ``Cluster`` column showing the summary label
  (``8c · cluster 1``). A live cluster matches a pin by its workspace and
  representative relaxation job, so re-clustering the analysis without
  re-pinning never relabels a different site. Tick ``Show every relaxed site`` to list every analysis
  workspace's clusters instead; the choice is saved with the project. With no
  pin, a notice and a ``Go to Muon Sites →`` button (which closes the dialog
  and opens that page) explain how to pin sites;
- ``Custom fractional position``, a fractional coordinate (``fx``, ``fy``,
  ``fz``) in the active host cell.

.. figure:: ../_generated/screenshots/magnetic_muon_site_dialog.webp
   :alt: The Select Muon Site dialog listing pinned structural-relaxation
         clusters of CoF2, the 4c positive-muon cluster selected and the
         displacement correction ticked.

   Pinned CoF₂ sites in ``Select Muon Site``, with the distortion correction
   available for the selected representative.

For a host with no relaxation or UEP results, as in a literature example,
choose ``Custom fractional position`` and type the site's ``fx``, ``fy`` and
``fz``.

.. figure:: ../_generated/screenshots/fe_select_muon_site.webp
   :alt: The Select Muon Site dialog with the custom fractional position source
         and the octahedral site of bcc iron, fx 0.5, fy 0, fz 0.

   The octahedral interstitial site of bcc iron as a custom position.

Until a site is chosen, the page uses the lowest-energy pinned site (positive
muon first), else the first structural-relaxation cluster, else the first UEP
cluster, else a custom site at the origin. With no site chosen, the dialog
opens on ``Structural relaxation clustering`` when that source lists a site
(a pinned site, or any cluster with ``Show every relaxed site`` ticked), and on
UEP clustering otherwise.

UEP sites are estimates
~~~~~~~~~~~~~~~~~~~~~~~

UEP sites are the one exception to the rule that downstream workflows use
relaxed sites (see :doc:`/project_specification`). A UEP minimum is unrelaxed,
and the relaxed muon site can sit some distance from it, so a field computed
there is a quick estimate made before any relaxation has run. While the UEP
source is chosen, the dialog shows a one-line note saying so, and the site
summary on the page repeats it. Fields computed at a UEP site are marked as
estimates wherever they are shown: the ``Result`` column reads, for example,
``field, unrelaxed estimate``, the results status line says the site is
unrelaxed, the distribution plot title ends in ``(unrelaxed estimate)``, its
``Export Data...`` file carries an ``# Estimate:`` header line, and the
project survey's local-field markers read ``≈`` before the magnitude.
Dipolar tensors and nuclear relaxation computed at a UEP site are marked the
same way in their dialogs: the results show a note that they were computed at
an unrelaxed UEP site, each tensor's title ends in ``(unrelaxed estimate)``,
and the status line after a calculation says the results are estimates. The
marker comes from the source saved with the result (see `Save results and
workspaces`_), so it survives a reload, and a tensor or nuclear-relaxation
result keeps its marker after another muon site is chosen.

Correct for the relaxed structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Structural-relaxation cluster representatives can expose a distorted relaxed
structure. When that structure is available, the dialog enables the ``Correct
the dipolar field for muon-induced displacements`` option; it is disabled and
cleared for every other site. The choice is stored with the selected site, the
site summary on the page notes it, and saved workspaces restore it. When it is
ticked, ``Calculate Dipolar Field`` applies the correction described in
:ref:`Apply muon-induced distortion corrections
<magnetic-fields-distortion-corrections>`. The ``Nuclear Relaxation...`` dialog
has its own distortion option for the same relaxed structure (see `Calculate
nuclear relaxation`_).

.. _magnetic-fields-calculation-settings:

Set the calculation parameters
------------------------------

Click ``Calculation Settings...`` in the ``Calculation`` card to open the
``Calculation Settings`` dialog. Under ``Numerical Parameters`` it sets:

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
:ref:`Convergence Checks <magnetic-fields-convergence-checks>`). The
``Commensurate Approximants`` group holds:

- ``Report finite fields for near-commensurate q`` (default off), which lists
  the finite fields of a commensurate approximant instead of sampling a
  distribution when the structure has one (see :ref:`Commensurate Approximants
  <magnetic-fields-commensurate-approximants>`);
- ``Tolerance`` (default 0.001 r.l.u.), the largest change of any wavevector
  component when it is replaced by a fraction;
- ``Longest period`` (default 12, at most 64), the largest denominator allowed
  for each wavevector component.

The settings summary on the ``Calculation`` card names the approximant
tolerance and period when the option is on.

.. figure:: ../_generated/screenshots/mnsi_calculation_settings.webp
   :alt: The Calculation Settings dialog with 100 x 100 x 100 cells, 32768 lookup
         angles per q, 200000 distribution samples, Auto sampling, 512 histogram
         bins, and the approximant option ticked with tolerance 0.001 r.l.u. and
         longest period 64.

   The settings of the helix in :doc:`/user_guide/examples/mnsi`, whose q lies
   0.0009 r.l.u. from 1/64, so a period of 64 is needed for its approximant.

With the option on and an approximant found, ``Calculate Dipolar Field`` lists
the finite fields of the approximant, one row per translation of its magnetic
supercell with a distinct phase and labelled with that translation, each with
the full term breakdown of a commensurate result.

.. figure:: ../_generated/screenshots/mnsi_approximant_fields.webp
   :alt: The Magnetic Fields page listing 64 fields of the commensurate
         approximant q = (1/64, 1/64, 1/64) of the MnSi helix at site II, the
         first row's field of 0.118082 T and its dipolar, Lorentz, contact,
         demagnetizing and applied terms.

   The 64 fields of the approximant of the helix in
   :doc:`/user_guide/examples/mnsi` at site II, labelled ``site 1 + (n, 0, 0)``
   by the translation along ``a``. The table scrolls; the first row is
   selected and its terms are listed below it.

The sampling method affects only incommensurate field distributions. It does not
simulate muon spectra or time-dependent polarisation functions.

Set the demagnetizing field
---------------------------

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
   ``Advanced...`` opens the ``Advanced Demagnetizing Field`` window. Choose a
   ``Shape`` and its ``Parameters``, set the ``Orientation`` (see
   `Orient the sample`_), then click ``Use Shape``. The advanced shapes are
   ellipsoids (``Semi-axis a``, ``Semi-axis b``, ``Semi-axis c``) and finite
   rectangular prisms (``Length a``, ``Length b``, ``Length c``,
   ``Fourier grid`` and ``Padding``). The window reports the tensor used by
   the calculation and renders the sample with a PyVista preview, with arrows
   along the crystal axes ``a``, ``b`` and ``c`` and the magnetisation
   direction ``M``.

.. figure:: ../_generated/screenshots/fe_demagnetizing_shapes.webp
   :alt: The Calculation card for bcc iron with the Infinite plate shape chosen,
         a demagnetizing field of -2.17 T and a total of -3.49 T at the first
         site.

   An ``Infinite plate`` for bcc iron at the octahedral site: the plate normal
   is the ``c`` axis, along the moment, so the demagnetizing term is
   :math:`-\mu_0 M = -2.1737` T instead of the sphere's -0.7246 T.

The ellipsoid factors are exact. For semi-axes :math:`a`, :math:`b` and
:math:`c` along the sample axes,

.. math::

   N_a = \frac{abc}{3}\,R_D(b^2, c^2, a^2),

and cyclically for :math:`N_b` and :math:`N_c` (J. A. Osborn, Phys. Rev.
**67**, 351 (1945), DOI: ``10.1103/PhysRev.67.351``), where :math:`R_D` is
Carlson's symmetric elliptic integral of the second kind (B. C. Carlson, Numer.
Algorithms **10**, 13 (1995), DOI: ``10.1007/BF02198293``). The three factors
sum to one, a sphere gives :math:`\tfrac{1}{3}` along every axis, and spheroids
reproduce the prolate and oblate closed forms.

The rectangular-prism mode uses the analytical Joseph–Schlömann
rectangular-prism tensor (R. I. Joseph and E. Schlömann, J. Appl. Phys. **36**,
1579 (1965), DOI: ``10.1063/1.1703091``) and averages it over midpoint samples
inside the prism. This avoids the resolution loss that a single cubic Fourier
grid would suffer for very thin plates or long needles. ``Fourier grid`` is
kept as the user-facing resolution control for the midpoint average; increasing
it improves the estimate but increases cost.

Orient the sample
~~~~~~~~~~~~~~~~~

A real crystal is cut or grown with a face on some :math:`(hkl)` plane and an
edge along some :math:`[uvw]` direction, so the principal axes of its shape are
not, in general, the Cartesian axes of the host lattice. The ``Orientation``
group of the ``Advanced Demagnetizing Field`` window sets them:

``Aligned with crystal axes``
   The sample axes ``a``, ``b`` and ``c`` lie along Cartesian ``x``, ``y``
   and ``z`` of the host lattice vectors, and the tensor is diagonal. The
   ``Normal (h k l)`` and ``In-plane axis [u v w]`` rows are greyed out, since
   this choice ignores them; they keep their values for when you switch back.

``Crystal directions``
   The sample ``c`` axis is normal to the plane in ``Normal (h k l)``, taken
   through the reciprocal lattice as :math:`h\mathbf{a}^* + k\mathbf{b}^* +
   l\mathbf{c}^*`. The sample ``a`` axis lies along ``In-plane axis [u v w]``,
   :math:`u\mathbf{a} + v\mathbf{b} + w\mathbf{c}`, and ``b`` completes a
   right-handed set. Indices may be negative or fractional.

With crystal directions the tensor in the crystal frame is
:math:`R\,\mathrm{diag}(N_a, N_b, N_c)\,R^\mathsf{T}`, where the columns of the
rotation :math:`R` are the sample axes, so it has off-diagonal elements and the
demagnetizing field is no longer parallel to the magnetisation. The preview
shows the shape rotated against the crystal axes.

When ``[u v w]`` is not perpendicular to the normal, MuFinder keeps the normal
exactly and uses the component of ``[u v w]`` in the plane as the ``a`` axis;
the hint under the fields reports the angle between the two directions. A
direction parallel to the normal, a zero index triplet or a blank index is
reported there instead, and ``Use Shape`` stays disabled until it is fixed.
Orienting by crystal directions needs a host structure. Without one the
``Orientation`` group cannot be changed: a new sample stays aligned, and a sample
already oriented by crystal directions keeps them.

A sample whose rotation was saved without crystal directions, for example one
set up through the Python API, opens with ``Saved rotation`` selected, which
keeps that rotation; the summary under the ``Demagnetizing field`` row then ends
in ``rotated sample axes``. Choosing either of the other options replaces it.

The orientation is saved with the project as the Miller indices, so it follows
the host lattice. The summary under the ``Demagnetizing field`` row lists them,
for example ``Ellipsoid a=4, b=4, c=0.5; normal (1 1 0), in-plane [0 0 1]``.

.. figure:: ../_generated/screenshots/fe_oriented_demagnetizing_sample.webp
   :alt: The Advanced Demagnetizing Field window for bcc iron with an oblate
         ellipsoid of semi-axes 4, 4 and 0.5, normal (1 1 0) and in-plane axis
         [0 0 1], and a tensor with off-diagonal elements 0.37315.

   A disc of bcc iron cut with its faces on :math:`(110)` and its ``a`` axis
   along :math:`[001]`. The disc's large factor :math:`N_c = 0.8309` acts
   along :math:`[110]`, so the tensor has
   :math:`N_{xx} = N_{yy} = (N_b + N_c)/2 = 0.4577` and
   :math:`N_{xy} = (N_c - N_b)/2 = 0.3732`; along the moment, :math:`[001]`,
   it is the small in-plane factor 0.0846.

Use a measured bulk magnetisation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default MuFinder takes the bulk magnetisation from the magnetic model (see
`How the demagnetizing field is calculated`_). To use a measured value instead,
for example from magnetometry on the same sample under the same conditions:

1. Tick ``Measured bulk magnetisation`` in the row below the ``Demagnetizing
   field`` row.
2. Enter the three Cartesian components, ``x``, ``y`` and ``z`` in the frame of
   the host lattice vectors, in Bohr magnetons per formula unit (μB/f.u.). The
   formula unit is the reduced formula of the host's non-muon atoms, which the
   row names, for example ``μB per Fe2O3``; the cell holds :math:`Z` of them,
   the greatest common divisor of its element counts.

.. figure:: ../_generated/screenshots/fe_measured_magnetisation.webp
   :alt: The Calculation card for bcc iron with a measured bulk magnetisation of
         2 muB per Fe along z ticked, a sphere, and a demagnetizing field of
         -0.659 T.

   A measured magnetisation of 2.0 μB per Fe along ``z`` (an illustrative
   value for a sample short of saturation) replaces the model's 2.2 μB, so the
   sphere's demagnetizing term is -0.6587 T.

MuFinder then uses

.. math::

   \mathbf{M}_\mathrm{meas} = \frac{Z\,\mathbf{m}_\mathrm{f.u.}\,\mu_B}{V_\mathrm{cell}},

so a supercell of the host gives the same field as its primitive cell. A
magnetisation quoted per magnetic ion becomes μB/f.u. by multiplying by the
number of magnetic ions in the formula unit. An empty component counts as zero,
and a component that is not a finite decimal number stops the calculation with
an error. The measured value replaces the model value on the commensurate,
commensurate-approximant and incommensurate paths alike, and has no effect when
the demagnetizing field is ``None``. Unticking the box returns to the model value
and keeps the entered components. An incommensurate distribution records the
value it used with its calculation parameters, or none when it used the model
value.

How the demagnetizing field is calculated
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

For a sphere, :math:`N = \tfrac{1}{3}`, so a single-domain sample's Lorentz and
demagnetizing fields cancel up to the difference between
:math:`\mathbf{M}_\mathrm{Lor}` and :math:`\mathbf{M}_\mathrm{meas}`. That
difference comes from counting the ions inside a finite sphere and shrinks as the
sphere grows; it is under 0.1 % of the Lorentz field in the
:doc:`worked examples </user_guide/magnetic_fields_theory>`. For an infinite
plate magnetised along its normal, :math:`N = 1` and the two terms sum to about
:math:`-2\mu_0 \mathbf{M}_\mathrm{meas} / 3`.

This correction is a field-level contribution only. It does not simulate spectra
or time-dependent polarisation. The bulk magnetisation does not depend on the
muon site or on the magnetic phase, so the demagnetizing field is the same for
every site of a commensurate calculation and for every sampled phase point of an
incommensurate distribution; it is added before applied fields and histograms
are formed.

.. _magnetic-fields-distortion-corrections:

Apply muon-induced distortion corrections
-----------------------------------------

A muon displaces the host ions around it, so the dipolar and contact fields at
the relaxed site differ from the fields computed for the pristine host. To
correct for it:

1. Choose a site from ``Structural relaxation clustering`` whose representative
   relaxed structure is available (see `Correct for the relaxed structure`_).
2. Tick ``Correct the local field for muon-induced displacements`` in the
   ``Select Muon Site`` dialog.
3. Optionally tick ``Include equivalent sites`` on the page.
4. Click ``Calculate Dipolar Field``. Corrected rows read ``field
   (distortion-corrected)`` in the ``Result`` column, and the vector total
   includes the correction. The ``Distortion correction`` term appears in the
   breakdown beside the other field terms.

.. figure:: ../_generated/screenshots/cof2_magnetic_fields_page.webp
   :alt: The Magnetic Fields page for the CoF2 antiferromagnet at the relaxed 4c
         site: a distortion-corrected field along c, with dipolar and distortion
         correction terms in the breakdown.

   A distortion-corrected field at the CoF₂ 4c site; the relaxed structure is
   synthetic (see :doc:`/user_guide/examples/cof2`).

The correction applies to commensurate fields, to every translated row of a
commensurate approximant and to every phase sample of an incommensurate
distribution. A distribution has no separate distortion term: its correction is
included in the ``Dipolar`` and ``Contact`` terms of the breakdown, and its row
reads ``distribution mean (distortion-corrected)`` in the ``Result`` column, also
after the project is reopened. Dipolar
tensors always use the pristine host. The Lorentz and demagnetizing terms depend
on the bulk magnetisation, so they keep their pristine values.

MuFinder corrects the converged pristine lattice sum with the finite-cell change
from the ion displacements in the representative relaxed supercell:

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

For an order that is not commensurate with the relaxed supercell, each Fourier
component :math:`c` with coefficients :math:`\mathbf{F}_c` and propagation
vector :math:`\mathbf{q}_c` gives a complex change

.. math::

   \Delta\mathbf{B}_c = \sum_{i \in \mathrm{cell}}
   \left[\mathsf{D}(\mathbf{r}_i + \mathbf{u}_i) - \mathsf{D}(\mathbf{r}_i)\right]
   \mathbf{F}_{c,i}\, e^{-2\pi i\, \mathbf{q}_c \cdot \mathbf{L}_i},

where :math:`\mathsf{D}` is the point-dipole tensor and :math:`\mathbf{L}_i` the
lattice translation of ion :math:`i`'s pristine site, so that each ion carries the
moment :math:`\operatorname{Re}[\mathbf{F}\,e^{-2\pi i\,\mathbf{q}\cdot\mathbf{L}}]`
of its pristine site. Shifting the component's order by a phase
:math:`\theta_c` turns, as a phase sample of a distribution does, changes the
field by :math:`\operatorname{Re}[e^{-2\pi i \theta_c}\,\Delta\mathbf{B}_c]`.
MuFinder adds that change at the phase each lookup angle of the distribution
samples, so the convergence checks run on the corrected distribution. A
commensurate approximant row at :math:`\mathbf{r} + \mathbf{T}` takes the phase
:math:`\theta_c = \mathbf{q}_c \cdot \mathbf{T}`, and a commensurate field
takes :math:`\theta_c = 0`.

When a ``Hyperfine coupling`` :math:`A` is set, the contact term is corrected
too. The contact field is :math:`(2\mu_0/3)\,\mu_B A` times the
:math:`1/r^3`-weighted mean moment of the ``nnn`` nearest magnetic ions closer
than ``rcont``, ``muesr``'s contact rule, and the correction is that field with
the shell chosen from the displaced positions minus the field with the shell
chosen from the pristine sites. This is a rough correction: it tracks which ions
are in the shell and their distances through the weights, but a real hyperfine
coupling changes with the muon-ion distance in a way a single coupling cannot
describe. A coupling calculated for the relaxed structure, for example by DFT,
supersedes it. The relaxed supercell must hold the whole contact shell: when the
shell reaches past the largest sphere about the muon that the supercell holds,
the calculation stops and asks for a larger relaxed supercell. When ions with
different moments tie for the last place in the shell, ``muesr`` keeps an
arbitrary one of them, so the contact field is undefined there and the
calculation stops with a message naming the tied ions' distance.

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

.. _magnetic-fields-dipolar-tensors:

Calculate a dipolar tensor
--------------------------

The ``Dipolar Tensor...`` action opens a compact tensor workflow without adding
more controls to the main Magnetic Fields tab. A dipolar tensor calculation
requires a muon position and a set of magnetic-ion positions; it does not require
a particular magnetic structure or moment direction. MuFinder therefore reuses
the selected muon site from the Magnetic Fields page and lets you choose which
host atoms should be treated as magnetic ions.

1. Select a muon site on the page (see `Choose the muon site`_).
2. Click ``Dipolar Tensor...``.
3. In the ``Magnetic Ions`` group choose the magnetic ions: ``Atoms with
   assigned moments``, ``All atoms``, ``Element``, ``Equivalent site`` or
   ``Atom indices`` (counted from 1 in the host's atom order, like the
   magnetic-structure builder and the ``Equivalent site`` labels). When the
   magnetic-structure builder already contains
   enabled components with non-zero moments, the dialog defaults to the atoms
   targeted by those components. If no atoms currently have assigned moments,
   MuFinder does not guess: the ``Calculate`` button stays disabled until you
   explicitly select magnetic ions.
4. Optionally tick ``Include equivalent muon sites``.
5. Click ``Calculate``.
6. Read the ``Most Recent Result`` group and choose the ``Units``: ``T/μB``,
   ``MHz/μB``, or ``Å⁻³``, matching the hyperfine-coupling unit choices.

.. figure:: ../_generated/screenshots/dipolar_tensor_dialog.webp
   :alt: The Dipolar Tensor dialog with atom 1, the corner Co of CoF2,
         and its diagonal dipolar tensor at the 4c site in T per muB.

   The corner-sublattice tensor at the CoF₂ 4c site.

The result is shown inside the tensor dialog rather than the field results
table. Each selected or symmetry-equivalent muon site receives one ``3 x 3``
tensor. The dialog keeps the most recent tensor result for the current Magnetic
Fields workspace, and saved workspaces restore that latest result when reopened.

``muesr`` returns the lattice sum
:math:`\sum (3\hat{\mathbf{r}}\hat{\mathbf{r}} - 1)/r^3` in ``Å⁻³``; MuFinder
multiplies it by :math:`\mu_0 \mu_B / (4\pi\,\text{Å}^3) = 0.9274` T/μB, so the
tensor applied to a moment in μB gives the dipolar field in tesla. The ``Å⁻³``
display uses the hyperfine-coupling convention,
:math:`\mathbf{B} = (2\mu_0/3)\,\mu_B A\,\mathbf{m}`, so it shows the contact
coupling that would give the same field rather than ``muesr``'s raw lattice sum.

Only tensor-relevant numerical settings are shown in this workflow. The tensor
dialog reports the lattice-summation supercell used for the calculation, but it
does not expose incommensurate phase sampling, histogram bins, demagnetizing
field shape, or hyperfine coupling because those settings do not affect a
dipolar tensor. This keeps tensor coupling analysis separate from finite
dipolar-field and incommensurate field-distribution calculations while still
sharing the same host structure and muon-site selection.

Calculate nuclear relaxation
----------------------------

The ``Nuclear Relaxation...`` action estimates static nuclear dipolar
linewidths for the selected muon site. It is available whenever a host
structure and muon site are selected, and does not require an electronic
magnetic model. MuFinder uses a native implementation of the mag2spectra
second-moment workflow with bundled natural-abundance isotope data.

1. Select a muon site on the page, then click ``Nuclear Relaxation...``.
2. In the ``Mode`` group choose the ``Field regime`` (``Zero field`` or
   ``Transverse field``) and the ``Sample`` (``Polycrystal`` or ``Single
   crystal``). The four supported modes are zero-field single crystal,
   zero-field polycrystal, transverse-field single crystal and
   transverse-field polycrystal.
3. For a single crystal, enter the ``Crystal-axis direction`` as ``x``, ``y``
   and ``z`` components (default ``0, 0, 1``). They are normalized before
   calculation. In TF this direction is the applied-field axis. In ZF it is the
   initial muon-polarisation axis. Polycrystal modes use analytical
   orientational averages and ignore the direction inputs.
4. Optionally set ``Radius (Å)`` (the default is ``auto``) and ``Convergence
   rtol`` (default ``1e-6``).
5. For a ZF mode, optionally tick ``Include ZF quadrupolar correction``. The
   quadrupolar correction is available only for ZF modes.
6. If the selected site comes from a structural-relaxation cluster and the
   representative relaxed structure is still available, optionally tick
   ``Apply muon-induced distortion correction``. It is available when the
   selected relaxation-cluster site has a derived distorted structure.
7. Click ``Calculate``.

.. figure:: ../_generated/screenshots/nuclear_relaxation_dialog.webp
   :alt: The Nuclear Relaxation dialog with a zero-field polycrystal result at
         the CoF2 4c site, the distortion correction applied, and its Co and F
         contributions.

   A zero-field polycrystal width at the distortion-corrected CoF₂ 4c site.

The ``Results`` group reports :math:`\sigma` in :math:`\mu\mathrm{s}^{-1}`,
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

With the distortion correction ticked, MuFinder pairs the relaxed supercell
with the active host exactly as for the dipolar field, with the same checks (see
:ref:`Apply muon-induced distortion corrections
<magnetic-fields-distortion-corrections>`), places the displaced and pristine
ions around a muon at the centre of the correction cell, and adds the local
finite-cell correction to the converged pristine lattice sum:

.. math::

   \sigma^2_\mathrm{corrected} =
   \sigma^2_\mathrm{undistorted} +
   \sigma^2_\mathrm{distorted,sc} -
   \sigma^2_\mathrm{pristine,sc}.

The isotope table is bundled as MuFinder package data for reproducibility.
Natural abundances are used; isotope-enrichment editing and field-dependent TF
quadrupolar linewidth models are not part of this workflow.

Save results and workspaces
---------------------------

.. figure:: ../_generated/screenshots/fe_magnetic_workspaces.webp
   :alt: The project navigator listing two saved Magnetic Fields workspaces,
         Octahedral site and Tetrahedral site, with the octahedral workspace's
         restored field results.

   Two workspaces of bcc iron (:doc:`/user_guide/examples/fe`), one per
   candidate site, under ``Magnetic Fields`` in the navigator. The page shows
   the octahedral workspace with its restored results.

To keep a magnetic structure and its settings for later:

1. Click the save button in the page header (tooltip ``Save this magnetic-field
   workspace``). It stores the current magnetic structure and form settings as
   a named magnetic-field workspace.
2. Find the workspace as a child of ``Magnetic Fields`` in the project tree.
3. To rename it, edit its page title.
4. To delete it, use the delete button on its own page, or right-click its
   project-tree node and choose ``Delete Workspace...``.

The latest field, dipolar-tensor and nuclear-relaxation results of the Magnetic
Fields page and of each saved workspace are saved with the project and restored
when you reopen it. The calculations run inside MuFinder, so the results belong
to no calculation run. Each of the page and every workspace keeps one latest
result of each kind. A new calculation replaces the previous result of that
kind; changing a setting that invalidates the field results, or clearing a
dialog's results, removes it. Saving a workspace copies the page's results to
the workspace, and deleting a workspace deletes its results. The page's form
settings stay in the project's UI state and do not repeat the results.

Every result also remembers which source supplied its muon site (UEP
clustering, relaxation clustering or a custom position). A UEP source
marks the result as an unrelaxed estimate (see `UEP sites are estimates`_). A
result saved without a source takes the source of the form's selected site,
except dipolar-tensor and nuclear-relaxation results: those are kept when a
different muon site is chosen, so the selected site says nothing about where
they were computed, and a record without a source has none and is not marked as
an estimate.

For developers: the muon-site source is stored as ``site_source`` (``uep_clustering``,
``relaxation_clustering`` or ``custom``). The results are ``Result`` records of type ``magnetic_field``,
``magnetic_dipolar_tensor`` and ``nuclear_relaxation`` whose ``data["owner"]``
is ``magnetic_fields`` for the page or the workspace id for a saved workspace.
A ``magnetic_field`` record holds ``kind`` (``commensurate`` or
``distribution``) and one ``results`` entry per site: a commensurate entry
stores ``site_label``, ``muon_position``, ``total_T``, ``magnitude_T`` and its
field terms (the correction is stored as ``distortion_T`` beside the uncorrected
``dipolar_T``); a distribution entry stores ``fields_T``, the histogram,
``calculation_parameters``, its ``convergence`` checks, ``terms`` (each of
``dipolar``, ``lorentz``, ``contact``, ``demag`` and ``applied`` with its
``mean_T`` and ``rms_T``), ``total_rms_T``, and the constant ``applied_T`` and
``demag_T`` vectors; the finite field
list of a commensurate approximant is stored as ``commensurate`` entries. A
``magnetic_dipolar_tensor`` record holds ``results`` and a
``nuclear_relaxation`` record holds ``result``. The form state saves the
measured magnetisation as ``use_measured_magnetization`` and
``measured_magnetization``. See :doc:`/compatibility_policy` for the persisted
names.

Dependency handling
-------------------

``muesr`` and ``muLFC`` are optional scientific backends for periodic local-field
calculations. MuFinder imports these engines only when a calculation requires
them, so an environment without those packages raises a clear dependency error in
the Magnetic Fields workflow rather than failing during application startup. The
packaged desktop builds install these optional backends when they are available
for the target platform. On Windows, MuFinder follows the older project build
strategy and installs ``muLFC`` from the upstream Git source because PyPI does
not currently publish a Windows wheel or source distribution for that backend.
MuFinder keeps the released ``muesr`` working on every Python version it
supports.

Current scope
-------------

The implemented workflow covers the propagation-vector data model, the
magnetic-structure builder and multi-q texture dialogs, commensurate local
fields with optional muon-induced distortion corrections, incommensurate field
distributions with convergence checks, finite field lists of commensurate
approximants, dipolar tensors, demagnetizing corrections, and nuclear dipolar
linewidths. The page stops at field-level results; the library computes the
muon polarisation, its powder averages, its spectrum and its strong-collision
dynamics from them, and provides the reference relaxation functions (see
:doc:`/api/using_mufinder`). Distortion corrections are not applied to
incommensurate field distributions, commensurate approximants or dipolar
tensors.

.. toctree::
   :hidden:

   magnetic_fields_theory
