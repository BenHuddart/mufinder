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
2. **Load the file.** Click ``Load Structure (.cif/.mcif/.cell)`` and choose the
   file.
3. **Check the result.** The atom table, the 3D viewer and the summary above
   them (label, source, atom count, species) show the structure that was read.
   The new structure becomes the active host.
4. **Add spins if you need them.** Edit the ``Spin (hbar/2)`` column in the
   atom table, and set ``Propagation vector (r.l.u.):`` for magnetic work (see
   `Setting spins and the propagation vector`_).
5. **Switch hosts when you have several.** Choose another structure from
   ``Active host structure:``. Every stage that follows uses the one selected
   there.

An unsupported extension, invalid content or a missing file is reported with an
explicit error.

The host-structure panel
------------------------

.. figure:: ../_generated/screenshots/host_structure_page.webp
   :alt: The Host Structure panel with CoF2 loaded, its atom table with
         antiparallel Co spins and its 3D view down a.

   The ``Host Structure`` panel with the CoF₂ host of
   :doc:`/user_guide/examples/cof2` loaded, its two Co carrying
   spins of +3 and −3 ħ/2.

- ``Active host structure:`` switches between imported host structures.
- The atom table and the 3D viewer sit side by side in a resizable splitter.
  A compact two-row summary (label, source, atom count, species) above them
  leaves horizontal room for the viewer.
- The atom table's ``Spin (hbar/2)`` column holds each atom's editable spin,
  and ``Propagation vector (r.l.u.):`` takes the propagation vector in
  reciprocal-lattice units.
- ``Supercell Editor...`` opens the :doc:`/user_guide/supercells`.

The viewer is an orthographic structure viewer that keeps the camera when
bonds, spins, slice planes, point markers or UEP isosurfaces are redrawn.

- Spins are drawn as 3D arrows with shafts and cones, centred on the atoms and
  coloured by sign for up and down moments. Spin display is off by default and
  turns on automatically when any edited or imported spin is non-zero; atoms
  are drawn translucent while spins are shown.
- Fitting the view accounts for the full atom spheres, not just their centres,
  so atoms at the left and right edges are not clipped.
- A crystallographic axis triad sits in a narrow column at the bottom left,
  leaving the main viewport to the structure. The viewer and triad colours
  follow the light or dark palette.
- Left-clicking empty space clears the atom selection.
- The controls fit narrow panes. The display toggles (``Cell``, ``Bonds``,
  ``Spins``) and the orientation controls (the compact ``a``, ``b`` and ``c``
  look-along buttons and ``Reset View``) share one line when there is room and
  wrap onto two when there is not. A long selection summary is shortened with
  an ellipsis, with the full text in its tooltip. The viewer needs roughly
  300 px of width, so plots beside it in the analysis dialogs get the width
  they need.

The same viewer is used across the workflows, for example in the supercell
editor, the UEP analysis window and the relaxation analysis workspaces.

Setting spins and the propagation vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spins are in units of ``hbar/2`` and are stored on the host structure; the
propagation vector is in reciprocal-lattice units. The magnetic and
spin-polarized workflows use them (see :doc:`/user_guide/magnetic_fields`). A ``.cell`` file's inline ``SPIN=...`` tokens on
``POSITIONS_FRAC`` rows become per-atom spins on import, and an MCIF file sets
them from its moments (see `What is read from each format`_).

What MuFinder stores
--------------------

On import MuFinder copies the source file into the project's ``imports/``
folder, makes the structure the active host by default, and records the import
in the project provenance (the :doc:`/user_guide/journal`). The original file
is never modified.

What is read from each format
-----------------------------

CIF
   CIF comments follow the CIF syntax rules. A comment beside a loop heading,
   and comment-only or blank lines between loop headings, are accepted. A
   ``#`` inside a quoted string, inside a semicolon text field or within an
   unquoted value such as ``C#1`` is kept as data, and LF, CRLF and CR line
   endings are all accepted. A quoted loop value that contains a space
   followed by ``#`` (``'C #1'``) is rejected.
MCIF
   Read like a CIF, plus the magnetic symmetry and magnetic metadata:

   - the magnetic operations and centerings
     (``_space_group_symop.magn_operation_xyz`` and
     ``_space_group_symop.magn_centering_xyz``, or the
     ``_space_group_symop_magn_operation.xyz`` spelling) are applied to the
     listed sites, so the structure holds the whole magnetic cell rather than
     only the sites the file lists. Every operation is combined with every
     centering; images of one site closer than 0.01 Å are one atom, and each
     atom keeps the label of the site it came from. A moment on a site is
     carried to each of its images, through the operation's ``mxmymz`` part
     when the file gives one and otherwise as an axial vector,
     ``m' = det(R) θ R m``, with θ = -1 for a time-reversed operation;
   - atom labels, single-vector legacy propagation-vector tags,
     ``_magnetic_propagation_vector_kxkykz`` entries, and atom-site moment
     vectors;
   - every imported propagation vector, kept in reciprocal-lattice units;
   - moment components converted to Cartesian vectors in μB: ``crystalaxis``
     components lie along the unit lattice directions, ``m = Σ m_i a_i/|a_i|``,
     and ``cartn`` components pass through. When the file gives
     ``_atom_site_moment.magnitude``, the converted moment must match it to
     within 1 % or 0.02 μB, whichever is larger;
   - moment labels mapped onto atoms by exact MCIF label match or by the
     unique ``label_...`` expansion of a site label.

   A ``.cif`` file that carries magnetic operations is read the same way. The
   import stops with a message when a magnetic operation cannot be read, when
   an image of one site lands on another site, when a moment disagrees with its
   stated magnitude, or when the file also carries ordinary symmetry
   operations that expand the sites before the magnetic ones are applied.

   When the imported structure becomes the active host, this metadata becomes
   the project's magnetic structure: MuFinder derives per-atom scalar spins
   from it and converts the moments into propagation-vector components, with
   the phase correction needed for the unit-cell moment at each labelled site
   to reproduce the imported moment. See :doc:`/user_guide/magnetic_fields`
   for how the magnetic structure is used.
CASTEP ``.cell``
   Only the lattice and position blocks are read, so a CASTEP ``-out.cell``
   imports whatever calculation keywords it carries. Inline ``SPIN=...``
   tokens on ``POSITIONS_FRAC`` rows become per-atom spins.

Limitations
-----------

- The viewer is orthographic only.
- A CIF with a quoted loop value containing a space followed by ``#`` cannot
  be imported (see `What is read from each format`_).
- Magnetic superspace-group operations
  (``_space_group_symop_ssg.operation_algebraic``), which incommensurate
  magnetic CIF files use, are not applied: such a file imports with only the
  sites it lists.

For developers
--------------

CIF and MCIF files are read with ASE-backed readers. As ASE takes a loop
heading only from a line holding that one tag, MuFinder hands ASE an in-memory
copy with the accepted loop-heading comments and blank lines removed; the
quoted-value limitation above comes from ASE's reader. ``.cell`` files are read
by MuFinder's own parser, ``mufinder.castep_io.cell_parser``. MCIF propagation
vectors are kept as ``propagation_vectors_rlu`` metadata, with the first also
mirrored to the legacy ``propagation_vector`` property. The viewer uses the
PyVista/VTK stack. See :doc:`/architecture`.
