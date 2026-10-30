Supercell Editor
================

The ``Supercell Editor`` builds, previews and saves reusable supercells of the
host structure. A saved supercell is a generated structure snapshot derived
from a source structure, usually the active host structure. Workflow tabs can
then reuse that saved cell instead of asking you to recreate the same matrix in
several places. Supercells are optional: any calculation can run on the host
unit cell instead, but muon calculations usually need a cell large enough that
the muon does not interact with its own periodic images (see `Suggesting a
supercell`_).

Before you start
----------------

Load a host structure first (see :doc:`/user_guide/host_structure`): the editor
builds supercells from the active host.

Running the workflow
--------------------

1. **Open the editor.** Click ``Supercell Editor…`` on the ``Host Structure``
   page, or choose ``Manage Supercells…`` from a workflow's simulation-cell
   selector or row ``Supercell`` drop-down.
2. **Start a new cell.** Click ``New`` to start from the unit-cell matrix.
3. **Choose the scaling.** Set ``Nx``, ``Ny`` and ``Nz`` (1 to 64 each), or tick
   ``Advanced 3x3 matrix`` and enter a full 3×3 integer matrix. Alternatively,
   click ``Suggest…`` (see `Suggesting a supercell`_).
4. **Check the preview.** The preview, a :doc:`/user_guide/structure_viewer`,
   shows the generated structure and the propagated spin arrows, and the status
   line reports the resulting atom count. It updates on each edit of a ``Nx``,
   ``Ny``, ``Nz`` or matrix box. Opening the editor, selecting a saved cell, clicking ``New`` and applying a
   suggestion set the whole matrix at once and build the preview once.
5. **Name and save.** Give the supercell a clear name and click ``Save``. Click
   ``Save As`` to create another saved copy from the current matrix and name.
6. **Use it.** Pick it from the simulation-cell selector or the row
   ``Supercell`` drop-down of the workflow that should use it. If you opened
   the editor from a workflow, click ``Use Selected`` instead (see below).

The editor
----------

.. figure:: ../_generated/screenshots/supercell_editor.webp
   :alt: The Supercell Editor with a saved 2x2x2 CoF2 supercell selected and
         previewed.

   The editor with the saved 2x2x2 supercell of the CoF₂ example
   (:doc:`/user_guide/examples/cof2`) selected and its
   48-atom preview.

The ``Saved Supercells`` table on the left lists each saved cell with its
``Name``, ``Matrix``, ``Atoms`` count, and ``Usage`` (the configured
calculations and active workflow selections that refer to it). The controls
for the selected cell are ``New``, ``Save``, ``Save As``, ``Delete``,
``Suggest…``, ``Use Selected`` and ``Close``.

When the editor is opened from a workflow row or simulation-cell selector, a
``Use Selected`` button is shown; it returns the highlighted saved supercell to
that workflow (saving the current matrix first if nothing is selected). The
workflow selector refreshes immediately after the editor closes.

Suggesting a supercell
~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/supercell_suggestions.webp
   :alt: The supercell suggestions for CoF2: three diagonal cells, the
         3x3x3 cell highlighted.

   ``Suggest…`` for CoF₂, where no cell under 200 atoms reaches 10 Å.

``Suggest…`` proposes diagonal supercells ``(n₁, n₂, n₃)`` for the active
host. It lists cells whose minimum periodic-image distance is at least 8 Å,
each one improving on the cheaper options above it, in the columns
``Scaling``, ``Atoms``, ``Min image (Å)`` and ``≥ 10 Å``. The minimum image
distance is the shortest non-zero lattice translation of the supercell over all
``(m₁, m₂, m₃) ∈ {-1, 0, 1}³``, not simply the shortest edge. The dialog marks
as ``(Recommended)`` the cheapest cell that reaches the 10 Å rule of thumb
while staying under 200 atoms; when no such cell exists it highlights the best
cell under 200 atoms and warns. When the primitive cell already has 200 or
more atoms no search is run. ``Use This Supercell`` loads the highlighted
matrix into the editor; it is not saved until you choose ``Save``.

Editing saved supercells
~~~~~~~~~~~~~~~~~~~~~~~~

Selecting a saved supercell loads its name and matrix into the editor. Rename
it by changing the name and choosing ``Save``. Change the matrix and choose
``Save`` to regenerate the saved structure.

If a saved supercell is already frozen into a configured calculation, MuFinder
does not rewrite that calculation's structure snapshot. Matrix edits create a
new saved copy for future use instead. This keeps existing configured jobs and
their generated input structures reproducible.

Deleting saved supercells
~~~~~~~~~~~~~~~~~~~~~~~~~

Unused saved supercells can be deleted with ``Delete``. Deletion is disabled
for supercells already used by configured calculations. If a deleted supercell
was only an active workflow selection, MuFinder clears that selection back to
the single unit cell.

Using supercells in a workflow
------------------------------

Each workflow chooses its own simulation cell; there is no project-wide
selection that calculation inputs read.

- The cutoff and k-point convergence pages have a ``Simulation cell`` row: a
  drop-down with ``Single unit cell`` and the saved supercells of the active
  host, and ``Manage Supercells…`` to open the editor. Choosing ``Single unit
  cell`` returns to the host unit cell without deleting any saved supercell.
- The pristine-relaxation, muon-relaxation and UEP tables have a per-row
  ``Supercell`` drop-down with ``Unit cell``, the saved supercells and
  ``Manage Supercells…``.
- Saved supercells are project-wide and visible to every workflow for the same
  active host. Selecting one owned by another workflow clones it into the
  current workflow, reusing an existing current-workflow copy when one with the
  same host and matrix already exists, and the original saved cell stays as
  the source. Automatic labels combine the workflow stage and the matrix, for
  example ``convergence_cutoff 2x2x1``. UEP source jobs use the currently
  selected UEP source structure as their supercell context.

A configured calculation keeps the cell it was configured with: the cell is
frozen when the job is defined, with its expanded lattice, positions and spins,
so changing the selection later does not change jobs already defined.

What MuFinder stores
--------------------

Saved supercells are stored in the project and are listed in the editor's
``Saved Supercells`` table. A calculation's input files are rendered from the
frozen structure of the cell it was configured with, not from the live saved
supercell.

Limitations
-----------

- Diagonal scalings are limited to 1 to 64 along each axis.
- ``Suggest…`` proposes diagonal supercells only; use the advanced matrix mode
  for anything else.
- A saved supercell that a configured calculation uses cannot be deleted, and
  editing its matrix saves a new copy.

For developers
--------------

How these selections are stored in the project is described under "Supercell
selections" in :doc:`/architecture`. A convergence definition records
``simulation_cell_mode`` and ``supercell_source_workflow`` in its parameters,
and its first input structure is the selected supercell, or the host for
unit-cell mode.
