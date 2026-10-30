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
   ``Active host structure:`` (see `Switching hosts`_).

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

- ``Active host structure:`` switches between the imported and relaxed host
  structures (see `Switching hosts`_).
- ``Load Structure (.cif/.mcif/.cell)`` imports another structure, and
  ``Supercell Editor…`` opens the :doc:`/user_guide/supercells`.
- A compact two-row summary (label, source, atom count, species) sits above
  the atom table and the 3D viewer, which share a resizable splitter.
- The atom table lists every atom in the structure, one row each, however
  large the cell. Its ``Spin (hbar/2)`` column holds each atom's editable spin,
  and ``Propagation vector (r.l.u.):`` under it takes the propagation vector in
  reciprocal-lattice units.
- The viewer is the shared :doc:`/user_guide/structure_viewer`. Selecting rows
  in the atom table selects the same atoms in the viewer, and the other way
  round.

Setting spins and the propagation vector
----------------------------------------

Spins are in units of ``hbar/2`` and are stored on the host structure; the
propagation vector is in reciprocal-lattice units, each component between −1
and 1. The magnetic and spin-polarized workflows use them (see
:doc:`/user_guide/magnetic_fields`).

- Type a number into an atom's ``Spin (hbar/2)`` cell to set its spin. Text
  that cannot be read as a number, such as ``abc``, is discarded and the cell
  keeps its previous spin.
- Set the three components of ``Propagation vector (r.l.u.):``, along a*, b*
  and c*, for magnetic work.
- A ``.cell`` file's inline ``SPIN=...`` tokens on ``POSITIONS_FRAC`` rows
  become per-atom spins on import, and an MCIF file sets them from its moments
  (see :doc:`/user_guide/structure_files`).

The viewer draws a spin arrow on each atom with a non-zero spin and shows the
arrows as soon as one spin is non-zero (see
:doc:`/user_guide/structure_viewer`).

Switching hosts
---------------

``Active host structure:`` lists every structure that can be the active host:
first each imported structure (``Original — <label>``), by name, then the result
of each completed, converged pristine relaxation (``Relaxed — <job label>
(<date>)``), newest first (see :doc:`/user_guide/structural_relaxation_analysis`). Each
entry's tooltip says where it came from. Choosing another entry makes it the
active host, and every stage that follows uses it.

The project's magnetic structure follows the new host: it becomes the magnetic
metadata that host carries, if any. A relaxed pristine structure usually
carries no spins, so enter them again on the new host when you need them. The
change is recorded in the project provenance. While muon sites are being
generated for the current host the change is refused: wait for the generation
to finish, or cancel it, first.

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

For developers
--------------

The panel is ``mufinder.app.host_structure_widget.HostStructureWidget``; the
import readers are described in :doc:`/user_guide/structure_files`, the viewer
in :doc:`/user_guide/structure_viewer`. See :doc:`/architecture`.
