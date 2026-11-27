Host Structure
==============

Every project starts from a host crystal structure. The ``Host Structure``
panel loads it, shows it in a 3D viewer, and holds the per-atom spins and
propagation vector that the magnetic and spin-polarized workflows use. Every
other stage works on the active host chosen here.

Before you start
----------------

You need a crystal structure file for the material you are studying, in one of
these formats:

- CIF files (``.cif``);
- magnetic CIF files (``.mcif``);
- CASTEP cell files (``.cell``).

Nothing else has to exist first: the host is the first thing a project needs.

Importing a structure
---------------------

1. **Open the panel.** Select ``Host Structure`` in the project navigator.
2. **Load the file.** Click ``Load Structure…`` and choose the file.
3. **Check the result.** The facts card under ``Active host`` (formula, atom
   count, space group and lattice), the atom table and the 3D viewer show the
   structure that was read. The new structure becomes the active host.
4. **Add spins if you need them.** Edit the ``Spin (ħ/2)`` column in the
   atom table, and set the ``Propagation vector`` above it for magnetic work
   (see `Setting spins and the propagation vector`_).
5. **Switch hosts when you have several.** Choose another structure from
   ``Active host`` (see `Switching hosts`_).

An unsupported extension, invalid content or a missing file is reported with an
explicit error. :doc:`/user_guide/structure_files` lists what is read from each
format.

The host-structure panel
------------------------

.. figure:: ../_generated/screenshots/host_structure_page.webp
   :alt: The Host Structure panel with CoF2 loaded, its atom table with
         antiparallel Co spins and its 3D view down a.

   The ``Host Structure`` panel with the CoF₂ host of
   :doc:`/user_guide/examples/cof2` loaded, its two Co carrying
   spins of +3 and −3 ħ/2.

- ``Load Structure…`` imports another structure, and ``Supercell Editor…``
  opens the :doc:`/user_guide/supercells`.
- ``Active host`` switches between the imported and relaxed host structures
  (see `Switching hosts`_). The drop-down is as wide as its longest entry, and
  narrower in a narrow window, where its tooltip gives the whole entry. The
  line under it says that every later stage uses this structure, and that
  spins and saved supercells are kept per host.
- Under it, a card across the page holds the facts about the active host on
  one line, each a small caption over its value: first the structure's,
  ``Formula`` (of the whole cell, such as ``Co2F4``), ``Atoms``, ``Space
  group`` (``P4₂/mnm (136)``) and ``Source``, where the structure came from
  (``imports/CoF2.cif``); then, after a thin vertical line, the cell's: ``a``,
  ``b`` and ``c`` in Å, ``α``, ``β`` and ``γ`` in degrees and the ``Volume`` in
  Å³. Width to spare widens the gaps between the facts evenly, up to twice
  their usual size, and the rest is left at the right end. A value too long
  for its place, such as a long file path, is shortened in the middle first,
  with the whole value in its tooltip. When the facts still do not fit on one
  line, the cell's facts go on a second line and the vertical line goes;
  narrower still, the facts wrap one at a time. The values can be selected
  and copied, and a screen reader reads each with its caption.
  With no host, ``No host structure loaded.`` stands in its place as plain
  text, without the card's frame.

  - ``Atoms`` counts the atom table's rows. ``Formula`` reads each species
    label as its element in any case (``Co:1`` and ``fe`` count as Co and Fe)
    and leaves out a label that names no element.
  - ``Space group`` is the one spglib finds at the symmetry tolerance every
    stage uses (0.001 Å). A structure spglib cannot classify, such as one with
    two atoms on one site, has no ``Space group`` fact.
  - A relaxed host has no file, and its ``Source`` names the relaxation by its
    row on the :doc:`/user_guide/pristine_relaxation` page (``Pristine
    relaxation relax_1``), or by its job label when another relaxation's row
    has the same label. Any other structure without a file shows its label.

- ``Magnetic order`` heads the atom table. Under it, on one line on top of the
  table and with the table's edges, ``Propagation vector`` takes three
  components in boxes marked ``a*``, ``b*`` and ``c*`` inside them, and a help
  icon ends the line (see `Setting spins and the propagation vector`_). The
  table opens wide enough for that line as well as for its columns. Only in a
  window too narrow for both the table and the viewer do the boxes and the
  icon go under the label, and the boxes narrow.
- The help icon is a small muted question mark, framed only on hover. It shows
  what the propagation vector does as a tooltip. Clicking it, or pressing
  Space or Enter while it has the keyboard focus, opens the same text in a
  small bubble under it, whose text can be selected; Escape or a click
  elsewhere closes it.
- The atom table, with the ``Magnetic order`` rows above it, and the 3D viewer
  share a resizable splitter. The table opens wide enough for every column and
  for the propagation-vector line,
  and the viewer takes the rest of the window and never gets narrower than
  380 px. In a narrow window the table scrolls sideways instead. The last
  column takes any width the table has to spare, so the columns reach the
  table's right edge; when the rows need a vertical scrollbar it takes that
  room and every column stays whole.
- The atom table lists every atom in the structure, one row each, however
  large the cell: its element, its fractional coordinates ``x``, ``y`` and
  ``z`` along a, b and c (the column tooltips say so), and its editable spin
  in the ``Spin (ħ/2)`` column.
- The viewer is the shared :doc:`/user_guide/structure_viewer`. Selecting rows
  in the atom table selects the same atoms in the viewer, and the other way
  round.

Setting spins and the propagation vector
----------------------------------------

Spins are in units of ``ħ/2`` and are stored on the host structure; the
propagation vector ``k`` is in reciprocal-lattice units, each component between
−1 and 1. The magnetic and spin-polarized workflows use them (see
:doc:`/user_guide/magnetic_fields`).

- Type a number into an atom's ``Spin (ħ/2)`` cell to set its spin. Text
  that cannot be read as a number, such as ``abc``, is discarded and the cell
  keeps its previous spin.
- Set the three components of the ``Propagation vector`` k, along ``a*``,
  ``b*`` and ``c*``, under ``Magnetic order``. When a supercell is built from the host, each atom's
  spin is multiplied by cos(2π k·R), where R is the lattice translation of that
  atom's copy, so the supercell's spins (and the arrows in the Supercell
  Editor's preview) follow the magnetic order; the editor warns when a
  supercell does not fit ``k`` (see :doc:`/user_guide/supercells`). A saved
  supercell keeps the spins it was built with. Leave ``k`` at 0 0 0 when the loaded cell is
  already the magnetic cell: every copy then keeps the spins as typed.
- A ``.cell`` file's inline ``SPIN=...`` tokens on ``POSITIONS_FRAC`` rows
  become per-atom spins on import, and an MCIF file sets them from its moments
  (see :doc:`/user_guide/structure_files`).

This page owns the host's spins and propagation vector. Until a magnetic
structure is accepted in the ``Magnetic Structure Builder``, the
:doc:`/user_guide/magnetic_fields` page builds its model from them. Once one
is accepted, that page owns the model: a spin or propagation-vector edit here
changes only the host's spins and propagation vector, which the spin-polarized
CASTEP workflows still use, and leaves the accepted model as it is, even when
every spin and the vector are set to zero.

The viewer draws a spin arrow on each atom with a non-zero spin and shows the
arrows as soon as one spin is non-zero (see
:doc:`/user_guide/structure_viewer`).

Switching hosts
---------------

``Active host`` lists every structure that can be the active host: first each
imported structure (``Original — <label>``), by name, then the result of each
completed, converged pristine relaxation (``Relaxed — <row> (<date>)``, such as
``Relaxed — relax_1 (2026-11-26)``, named by its row as the Pristine Relaxation
page shows it, or by its job label when two relaxations' rows share a label),
newest first (see :doc:`/user_guide/pristine_relaxation`). Each
entry's tooltip says where it came from. Choosing another entry makes it the
active host, and every stage that follows uses it.

The navigator's ``Host Structure`` row names the active host on a second,
muted line, as ``Active host`` lists it but without the date
(``Relaxed — relax_1``, ``Original — CoF2``), so every page shows which host
the project is using. A name too long for the navigator is shortened with
``…``; the row's tooltip, and a screen reader, give the whole entry. Without a
host the row has one line.

When the project already holds work for the current host (a job configured on
it or on a cell built from it, such as the pristine relaxation that made a
relaxed host, a muon site generated for it, or a saved supercell built from
it), choosing another entry first asks ``Make '<entry>' the active host?``. The
question says that every later stage will use the new host and that spins, the
propagation vector and saved supercells are kept per host; when the current
host has sites, it adds that they stay in the project and are marked as
belonging to it. ``Switch host`` makes the change; ``Cancel``, the default,
keeps the current host and puts ``Active host`` back on it. In a project with
no such work the change is made at once.

The project's magnetic structure follows the new host: it becomes the magnetic
metadata that host carries, if any. The atom table shows the new host's own
spins. A relaxed pristine structure keeps the ``SPIN=`` values its relaxation
ran with, so a spin-polarized relaxation brings its spins along. The
change is recorded in the project provenance. While muon sites are being
generated for the current host the change is refused, before any question:
wait for the generation to finish, or cancel it, first.

What MuFinder stores
--------------------

On import MuFinder copies the source file into the project's ``imports/``
folder, makes the structure the active host by default, and records the import
in the project provenance (the :doc:`/user_guide/journal`). The original file
is never modified. :doc:`/user_guide/structure_files` lists what is read from
each format.

Limitations
-----------

- Some CIF and MCIF content cannot be imported or is not applied; see the
  limitations in :doc:`/user_guide/structure_files`.
- An imported host cannot be renamed or removed from the panel.

Background job updates preserve text while a field has focus or contains an
unfinished edit. Valid text commits through the field's usual editing action;
other fields continue to show refreshed project values. This also applies to
the Magnetic Fields page's applied field, measured magnetisation, hyperfine
coupling and workspace title, Relaxation Analysis workspace names and the
Energy Barriers maximum path length. The path length stays as typed until
Generate Paths uses it. Loading another project or host, and explicitly
resetting a form, fills the fields from that new state.

For developers
--------------

The panel is ``mufinder.app.host_structure_widget.HostStructureWidget``; the
import readers are described in :doc:`/user_guide/structure_files`, the viewer
in :doc:`/user_guide/structure_viewer`. See :doc:`/architecture`.

The facts come from ``structure_facts`` in
``mufinder.services.host_structure_facts``: the formula from ``cell_formula``
in ``mufinder.science.ase_bridge``, the space group from ``detect_space_group``
in ``mufinder.science.symmetry_analysis`` and the lattice from ``lattice_metrics`` in
``mufinder.science.structure_metrics``. The panel asks for them again only when
the structure's geometry changes, so a spin edit or a dashboard refresh does
not repeat the space-group search. ``ProjectService.host_structure_source``
words where the structure came from, and ``list_promotable_host_structures``
names a relaxed host by its row through ``job_row_label`` in
``mufinder.services.project_document``.

A spin typed in the atom table is stored by
``ProjectService.set_host_magnetic_configuration``. The host keeps the spins as
``atomic_spins`` and the propagation vector as ``propagation_vector``. While a
spin or the propagation vector is non-zero, the host's ``magnetic_structure``
and the project's are one record of ``propagation_vector`` and ``moments``, and
both are cleared when everything is zero. The record holds one moment along z
for each atom with a non-zero spin, with the atom's zero-based ``atom_index``,
a label of species and one-based atom number (``Co2``), its ``vector`` and
``coordinate_system: "cartesian"``.

A builder model (a dict with a ``components`` key, as ``is_built_magnetic_model``
in ``mufinder.domain.magnetic_properties`` reads it) is never replaced or
cleared by this. The host's ``magnetic_structure`` is left alone when it holds
one. The project's is left alone when it holds one, or when it is unset beside
a host's builder model, which is then the model in use. A spin record beside a
builder model still follows the spins: on the host, so the spin-polarized
workflows never fall back to spins the user has changed; on the project, as an
empty ``moments`` list when everything is zero, so the host's builder model
does not take the place of the spins on Magnetic Fields.

``magnetic_model_from_host_spins`` in ``mufinder.science.magnetic_structure``
builds the magnetic model from that record, one component per moment on the
atom its ``atom_index`` names. In a record without ``source: "mcif"``, a moment
with no ``atom_index`` (or a null one) whose label is exactly the species of one
of the host's atoms followed by that atom's one-based number (``Co2``; ``Co:12``
for a ``Co:1`` atom second in the list) goes on that atom, provided no other
atom's species and number read the same. Any other moment with no
``atom_index``, and every such moment in an MCIF record, goes on the atom at its
position in the list.
