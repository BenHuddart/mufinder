Supercell Editor
================

MuFinder stores reusable supercells in each project. A saved supercell is a
generated structure snapshot derived from a source structure, usually the active
host structure. Workflow tabs can then reuse that saved cell instead of asking
you to recreate the same matrix in several places.

Opening the editor
------------------

.. figure:: ../_generated/screenshots/supercell_editor.png
   :alt: The Supercell Editor with a saved 2x2x2 rutile supercell selected and
         previewed.

   The editor with a saved 2x2x2 supercell selected and its 48-atom preview.

Load a host structure first: the editor builds supercells from the active
host. Then open the editor in one of three ways:

- select ``Supercell Editor`` in the project navigator and click
  ``Open Supercell Editor``;
- click ``Supercell Editor...`` in the ``Host Structure`` panel;
- choose ``Manage Supercells...`` from a workflow's simulation-cell selector
  or row ``Supercell`` drop-down.

When the editor is opened from a workflow row or simulation-cell selector, a
``Use Selected`` button is shown; it returns the highlighted saved supercell to
that workflow (saving the current matrix first if nothing is selected). The
workflow selector refreshes immediately after the editor closes.

The ``Saved Supercells`` table on the left lists each saved cell with its
``Name``, ``Matrix``, ``Atoms`` count, and ``Usage`` (the configured
calculations and active workflow selections that refer to it).

Creating and saving supercells
------------------------------

.. figure:: ../_generated/screenshots/supercell_suggestions.png
   :alt: The supercell suggestions for rutile TiO2: three diagonal cells, the
         2x2x4 cell highlighted.

   ``Suggest…`` for rutile TiO2, where no cell under 200 atoms reaches 10 Å.

Click ``New`` to start from the unit-cell matrix. The editor provides two
matrix entry modes:

- simple diagonal mode with ``Nx``, ``Ny``, and ``Nz`` values (1 to 64 each)
- advanced full ``3x3`` integer matrix mode, enabled with the
  ``Advanced 3x3 matrix`` checkbox

The preview updates the generated structure and propagated spin arrows before
the supercell is saved, and the status line reports the resulting atom count.
Give the supercell a clear name, then choose ``Save``. Use ``Save As`` to
create another saved copy from the current matrix and name.

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
------------------------

Selecting a saved supercell loads its name and matrix into the editor. Rename
it by changing the name and choosing ``Save``. Change the matrix and choose
``Save`` to regenerate the saved structure.

If a saved supercell is already frozen into a configured calculation, MuFinder
does not rewrite that calculation's structure snapshot. Matrix edits create a
new saved copy for future use instead. This keeps existing configured jobs and
their generated input structures reproducible.

Deleting saved supercells
-------------------------

Unused saved supercells can be deleted with ``Delete``. Deletion is disabled
for supercells already used by configured calculations. If a deleted supercell
was only an active workflow selection, MuFinder clears that selection back to
the single unit cell.

Workflow reuse
--------------

Saved supercells are available to compatible workflow tabs for the same source
structure. Selecting a supercell owned by another workflow creates the
workflow-specific snapshot MuFinder needs internally, while preserving the
original saved cell as the source. UEP source jobs use the currently selected
UEP source structure as their supercell context.

Workflow simulation cells
-------------------------

Each workflow chooses its own simulation cell; there is no project-wide
selection that calculation inputs read.

- The cutoff and k-point convergence pages have a ``Simulation Cell`` section:
  a summary of the active cell, a drop-down with ``Single unit cell`` and the
  saved supercells of the active host, ``Manage Supercells...`` to open the
  editor, and ``Use Unit Cell`` to return to the host unit cell without
  deleting any saved supercell.
- The pristine-relaxation, muon-relaxation and UEP tables have a per-row
  ``Supercell`` drop-down with ``Unit cell``, the saved supercells and
  ``Manage Supercells...``.
- Saved supercells are project-wide and visible to every workflow for the same
  active host. Selecting one owned by another workflow clones it into the
  current workflow, reusing an existing current-workflow copy when one with the
  same host and matrix already exists. Automatic labels combine the workflow
  stage and the matrix, for example ``convergence_cutoff 2x2x1``.

A convergence definition freezes the cell resolved when it is configured: its
first input structure is the selected supercell, or the host for unit-cell
mode, and its parameters record ``simulation_cell_mode`` and
``supercell_source_workflow``. The ``.cell`` files are rendered from that
frozen structure, so the expanded lattice, positions and spins are kept even if
the selection changes later.

How these selections are stored in the project is described under "Supercell
selections" in :doc:`/architecture`.
