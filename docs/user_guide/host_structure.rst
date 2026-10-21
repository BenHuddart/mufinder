Host Structure
==============

Every project starts from a host crystal structure. The ``Host Structure``
panel loads it, shows it in a 3D viewer, and holds the per-atom spins and
propagation vector that the magnetic and spin-polarized workflows use.

Importing a structure
---------------------

``Load Structure (.cif/.mcif/.cell)`` imports:

- CIF files (``.cif``);
- magnetic CIF files (``.mcif``), read through the CIF parser;
- CASTEP cell files (``.cell``).

CIF and MCIF files are parsed with ASE-backed readers. ``.cell`` files are read
by MuFinder's own CASTEP cell parser (``mufinder.castep_io.cell_parser``),
which reads only the lattice and position blocks, so a CASTEP ``-out.cell``
imports whatever calculation keywords it carries. Inline ``SPIN=...`` tokens on
``POSITIONS_FRAC`` rows become per-atom spins.

CIF comments follow the CIF syntax rules. A comment beside a loop heading, and
comment-only or blank lines between loop headings, are accepted: ASE's reader
takes a loop heading only from a line holding that one tag, so MuFinder hands
ASE an in-memory copy with those comments and blank lines removed from loop
heading lists. The source file is never modified. A ``#`` starts a comment only
where a token could start, so hashes inside quoted strings, inside semicolon
text fields and within unquoted values (``C#1``) are kept as data; MCIF
magnetic metadata is read with the same rules. LF, CRLF and CR line endings are
all accepted. ASE's reader still rejects a quoted loop value that contains a
space followed by ``#`` (``'C #1'``).

On import MuFinder copies the source file into the project's ``imports/``
folder, makes the structure the active host by default, and records the import
in the project provenance. An unsupported extension, invalid content or a
missing file is reported with an explicit error.

From an MCIF file MuFinder also extracts the magnetic metadata:

- atom labels, single-vector legacy propagation-vector tags,
  ``_magnetic_propagation_vector_kxkykz`` entries, and atom-site moment
  vectors;
- every imported propagation vector, kept in reciprocal-lattice units as
  ``propagation_vectors_rlu`` metadata, with the first also mirrored to the
  legacy ``propagation_vector`` property;
- moment labels mapped onto atom indices by exact MCIF label match or by the
  unique ``label_...`` expansion used by mag2spectra MCIF files.

When the imported structure becomes the active host, this metadata becomes the
project-level magnetic structure. MuFinder derives per-atom scalar spins from
it and stores them on the host structure, and converts the moments into
propagation-vector components with the phase correction needed for the
unit-cell moment at each labelled site to reproduce the imported moment. See
:doc:`/user_guide/magnetic_fields` for how the magnetic structure is used.

The host-structure panel
------------------------

.. figure:: ../_generated/screenshots/host_structure_page.png
   :alt: The Host Structure panel with rutile TiO2 loaded, its atom table and
         its 3D view.

   The ``Host Structure`` panel with rutile TiO2 loaded.

- ``Active host structure:`` switches between imported host structures.
- The atom table and the 3D viewer sit side by side in a resizable splitter.
  A compact two-row summary (label, source, atom count, species) above them
  leaves horizontal room for the viewer.
- The atom table's ``Spin (hbar/2)`` column holds each atom's editable spin,
  and ``Propagation vector (r.l.u.):`` takes the propagation vector in
  reciprocal-lattice units.
- ``Supercell Editor...`` opens the :doc:`/user_guide/supercells`.

The viewer is a PyVista/VTK structure viewer that keeps the camera when bonds,
spins, slice planes, point markers or UEP isosurfaces are redrawn. It is
orthographic only.

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
