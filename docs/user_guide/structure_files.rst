Structure Files
===============

This page is the reference for what MuFinder reads from each structure file it
imports on the :doc:`/user_guide/host_structure` panel: CIF (``.cif``),
magnetic CIF (``.mcif``) and CASTEP cell (``.cell``) files.

CIF
---

CIF comments follow the CIF syntax rules. A comment beside a loop heading,
and comment-only or blank lines between loop headings, are accepted. A
``#`` inside a quoted string, inside a semicolon text field or within an
unquoted value such as ``C#1`` is kept as data, and LF, CRLF and CR line
endings are all accepted. A quoted loop value that contains a space
followed by ``#`` (``'C #1'``) is rejected.

MCIF
----

An MCIF file is read like a CIF, plus the magnetic symmetry and magnetic
metadata:

- the magnetic operations and centerings
  (``_space_group_symop.magn_operation_xyz`` and
  ``_space_group_symop.magn_centering_xyz``, or the
  ``_space_group_symop_magn_operation.xyz`` and
  ``_space_group_symop_magn_centering.xyz`` spellings) are applied to the
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

In a file with several ``data_`` blocks, the symmetry and metadata come from
the block whose structure is imported: the last block that lists atom sites.
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
----------------

Only the lattice and position blocks are read, so a CASTEP ``-out.cell``
imports whatever calculation keywords it carries. Inline ``SPIN=...``
tokens on ``POSITIONS_FRAC`` rows become per-atom spins.

Limitations
-----------

- A CIF with a quoted loop value containing a space followed by ``#`` cannot
  be imported (see `CIF`_).
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
mirrored to the legacy ``propagation_vector`` property. See
:doc:`/architecture`.
