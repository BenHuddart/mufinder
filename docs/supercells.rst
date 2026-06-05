Supercell Editor
================

MuFinder stores reusable supercells in each project. A saved supercell is a
generated structure snapshot derived from a source structure, usually the active
host structure. Workflow tabs can then reuse that saved cell instead of asking
you to recreate the same matrix in several places.

Opening the editor
------------------

Open ``Supercell Editor`` from the project navigator, from the host-structure
panel, or from workflow controls labelled ``Manage Supercells...``.

When the editor is opened from a workflow row or simulation-cell selector,
``Use Selected`` returns the highlighted saved supercell to that workflow. The
workflow selector refreshes immediately after the editor closes.

Creating and saving supercells
------------------------------

The editor provides two matrix entry modes:

- simple diagonal mode with ``Nx``, ``Ny``, and ``Nz`` values
- advanced full ``3x3`` integer matrix mode

The preview updates the generated structure and propagated spin arrows before
the supercell is saved. Give the supercell a clear name, then choose ``Save``.
Use ``Save As`` to create another saved copy from the current matrix and name.

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

Unused saved supercells can be deleted from the editor. Deletion is disabled
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
