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
2. **Start a new cell.** Click ``New`` under the saved supercells to start from
   the unit-cell matrix.
3. **Choose the scaling.** Set the three ``Scaling`` boxes, ``Nx × Ny × Nz`` (1
   to 64 each), or tick ``General 3×3 matrix`` and enter a full 3×3 integer
   matrix. Alternatively, click ``Suggest…`` (see `Suggesting a supercell`_).

   While ``General 3×3 matrix`` is ticked, the matrix sets the cell and the
   three scaling boxes are disabled. While the matrix is a plain scaling (zeros
   off the diagonal, positive on it) the disabled boxes show its diagonal;
   otherwise they keep their last values. Unticking it enables the boxes again
   and the cell follows them: a plain scaling stays the same cell, and any
   other matrix gives way to the boxes' scaling.

   A general matrix that is not a plain scaling is kept while the editor is
   open, so ticking the box again brings it back as it was, even after the
   scaling boxes have changed. Otherwise ticking starts the matrix from the
   current scaling, so the cell does not change. Selecting a saved cell,
   ``New`` and a suggestion replace a kept matrix. Selecting a saved cell ticks
   the box for a general matrix and unticks it for a plain scaling; a saved
   general matrix also sets the scaling boxes from its diagonal (at least 1
   each), which is the scaling unticking then returns to.
4. **Check the preview.** The facts card under the ``Supercell`` box states the
   previewed cell: its ``Matrix``, its ``Atoms`` count, its ``Image distance``
   (how far a muon is from its nearest periodic image, as ``Suggest…`` measures
   it) and its cell lengths ``a``, ``b`` and ``c``. The preview below it, a
   :doc:`/user_guide/structure_viewer`, shows the generated structure and the
   propagated spin arrows. Both update on each edit of a scaling or matrix box.
   Opening the editor, selecting a saved cell, clicking ``New`` and applying a
   suggestion set the whole matrix at once and build the preview once.
5. **Name and save.** Give the supercell a clear name and click ``Save As`` to
   add it to the saved supercells. ``Save`` is for a saved cell you have
   selected: it updates that cell, and it is disabled when no cell is selected.
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

The subtitle names the host the supercells belong to as the ``Host
Structure`` page's selector names it (``Supercells of Original — CoF2``).
Saved supercells are kept per host, so after a switch to another host the list
shows that host's cells.

The ``Saved Supercells`` table on the left lists each saved cell of that host,
in name order, with its ``Name``, ``Matrix``, ``Atoms`` count, and ``Usage``.
The ``Name`` column takes the spare width, so the columns reach the table's
right edge, and hovering over a name that does not fit shows it in full.
``Usage`` reads:

- ``Jobs in Cutoff Convergence`` when configured jobs of a workflow use the
  cell, so its matrix can no longer change;
- ``Selected by K-Point Convergence`` when a workflow's ``Simulation cell`` is
  this cell, for the jobs it configures next;
- ``Not selected`` otherwise.

When two or more workflows qualify, the cell gives their number (``Selected
by 2 workflows``); hovering over it names every workflow, with both the jobs
and the selections. The rows of the Pristine Relaxation, Muon Relaxations and
UEP pages pick their cell by its matrix, not as a saved cell, so ``Usage`` does
not count them; the tooltip says so.

``New`` and ``Delete``, under the table, act on the list: ``New`` starts a
cell and ``Delete`` deletes the selected one, as do the Delete and Backspace
keys while the table has the focus. ``Save``, ``Save As`` and ``Suggest…``, in
the ``Supercell`` box, act on the cell being edited. Messages about what was
done (``Saved 'CoF2 2x2x2' with 48 atoms``, ``Deleted 'CoF2 3x3x3'``) and
a matrix that cannot be generated appear in the status line at the bottom left,
beside ``Close``; the next change to the matrix, or choosing another saved
cell, clears them. When a cell cannot be generated, the facts card keeps its
``Matrix``.

Opened from the ``Host Structure`` page, the editor starts on the most recently
saved supercell, with that row selected and its name and matrix loaded. Opened
from a workflow, it starts on the cell that workflow is using. When no saved
cell has the starting matrix, as in a project with no saved cells, no row is
selected, the name is empty, and ``Save`` stays disabled until a row is selected.
An automatic name such as ``2x2x2 supercell`` follows the matrix as you change
it; a name you typed stays as it is.

When the editor is opened from a workflow row or simulation-cell selector, a
``Use Selected`` button is shown; it returns the highlighted saved supercell to
that workflow. It is disabled when no cell is selected and when the matrix has
been changed and not saved. The workflow selector refreshes immediately after
the editor closes.

Supercells and the propagation vector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When the host has a non-zero propagation vector ``k`` (see
:doc:`/user_guide/host_structure`), each copy of an atom in the supercell has
its spin multiplied by ``cos(2π k·R)``, with ``R`` the lattice translation of
its cell. The spins repeat across the supercell's boundary only when ``k·A`` is
a whole number for each supercell vector ``A``, written in host cell units (the
rows of the matrix). When the previewed cell does not fit, a warning line under
the facts card says so (``This supercell does not fit the propagation vector k =
(0.5, 0, 0).``), and its help icon explains it. With ``k = (0.5, 0, 0)`` a
``2x1x1`` cell fits and a ``3x1x1`` cell does not. Each component of ``k``
may be up to 0.001 from a value that fits, so a third typed as ``0.333`` fits a
tripled or a sixfold cell, and a ``k`` within 0.001 of zero fits every cell.
The warning does not stop you saving the cell.

Suggesting a supercell
~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/supercell_suggestions.webp
   :alt: The supercell suggestions for CoF2: three diagonal cells, the
         3x3x3 cell highlighted.

   ``Suggest…`` for CoF₂, where no cell under 200 atoms reaches 10 Å.

``Suggest…`` proposes diagonal supercells ``(n₁, n₂, n₃)`` for the active
host, each ``n`` at most 64. It searches the cells of up to 200 atoms or, when
none of them reaches 10 Å, up to the smallest cell that does, looking no
further than 1000 atoms. It lists cells whose minimum image distance is at
least 8 Å, each one improving on the cheaper options above it, in the columns ``Scaling`` (written
``2x2x3``, as the editor and the workflow pages write it), ``Atoms`` and ``Min
image (Å)``. The minimum image distance is the length of the cell's shortest
lattice translation, however sheared the host cell is, and matches the
editor's ``Image distance``. A tick after the distance marks a cell that
reaches 10 Å.
The dialog marks as ``(Recommended)``, and highlights, the cheapest cell that
reaches 10 Å while staying under 200 atoms. ``Use This Supercell`` loads the
highlighted matrix into the editor; it is not saved until you choose ``Save``.

A warning under the table says when that recommendation is not possible:

- When no cell under 200 atoms reaches 10 Å, the recommended cell is the best one
  under 200 atoms. When no searched cell reaches 10 Å at all, that cell is
  listed even when its images are closer than 8 Å.
- When every listed cell has 200 or more atoms, the recommended cell is the
  smallest that reaches 10 Å, and the warning gives its atom count.
- When the primitive cell already has 200 or more atoms, no search is run and
  the table holds the primitive cell alone; the warning says if it misses 10 Å.

**Why the images must be far apart.** A muon is implanted in the ultradilute
limit, so it never meets another muon, but a periodic calculation places a copy
of it in every neighbouring supercell (Huddart *et al.* 2022). In a small cell the
muon interacts with those images, which biases the energies of its stopping
sites. There are two couplings. A charged muon has a long-range Coulomb
interaction with its images, so it needs a larger cell than a neutral one
(Onuorah *et al.* 2025). The muon also displaces its neighbouring atoms, and in a
small cell the strain around one image reaches the next.

**Why the shortest translation, not the shortest edge.** The images sit at every
lattice translation of the supercell, and in a skewed cell a combination such as
``A₁ - A₂`` can be shorter than any edge. The minimum image distance is therefore
the shortest non-zero translation ``m₁A₁ + m₂A₂ + m₃A₃`` over all
``(m₁, m₂, m₃) ∈ {-1, 0, 1}³``. A cell with long edges can still fail the test.

**Where the numbers come from.** 10 Å is a rule of thumb, not a convergence test:
the direct test is whether the forces the muon exerts on the host atoms converge
as the cell grows (Onuorah *et al.* 2025). The limit of 200 atoms is MuFinder's own
estimate of what is affordable, because a relaxation runs once per candidate site
and each one costs more as the cell grows. A larger cell is still available
through the editor's matrix.

References:

- B. M. Huddart *et al.*, MuFinder: a program to determine and analyse muon
  stopping sites, Comput. Phys. Commun. **280**, 108488 (2022),
  https://doi.org/10.1016/j.cpc.2022.108488.
- I. J. Onuorah *et al.*, Automated computational workflows for muon spin
  spectroscopy, Digital Discovery **4**, 523 (2025),
  https://doi.org/10.1039/d4dd00314d.

Editing saved supercells
~~~~~~~~~~~~~~~~~~~~~~~~

Selecting a saved supercell loads its name and matrix into the editor. Rename
it by changing the name and choosing ``Save``. Change the matrix and choose
``Save`` to regenerate the saved structure. If another workflow has the cell
selected, MuFinder first asks, naming those workflows; ``Cancel`` keeps the saved
cell and your edits, and ``Save As`` adds a new cell instead. A cell that no other
workflow has selected is saved without asking.

If a saved supercell is already frozen into a configured calculation, MuFinder
does not rewrite that calculation's structure snapshot. Matrix edits create a
new saved copy for future use instead, and the status line names the copy. The
copy keeps a name you typed; if another saved cell has that name, as when you
leave the name unchanged, ``copy`` is added to it. This keeps existing configured
jobs and their generated input structures reproducible.

Deleting saved supercells
~~~~~~~~~~~~~~~~~~~~~~~~~

Unused saved supercells can be deleted with ``Delete`` under the table, or
with the Delete or Backspace key while the table has the focus; MuFinder asks
first, with ``Cancel`` as the default. Deletion is disabled for supercells
already used by configured calculations, and the keys then do nothing. If a deleted supercell
was only an active workflow selection, MuFinder clears that selection back to
the single unit cell.

Using supercells in a workflow
------------------------------

Each workflow chooses its own simulation cell; there is no project-wide
selection that calculation inputs read.

- The cutoff and k-point convergence pages have a ``Simulation cell`` row: a
  drop-down with ``Single unit cell`` and the saved supercells of the active
  host, and ``Manage Supercells…`` to open the editor. Choosing ``Single unit
  cell`` returns to the host unit cell without deleting any saved supercell. A
  selected supercell that stops being available falls back to the unit cell,
  with a warning line under the row until you choose a cell.
- The pristine-relaxation, muon-relaxation and UEP tables have a per-row
  ``Supercell`` drop-down with ``Unit cell``, the saved supercells and
  ``Manage Supercells…``.
- Saved supercells are project-wide and visible to every workflow for the same
  active host. Workflows that pick the same saved cell share it, so the library
  keeps one entry per cell, whichever workflow or page it was saved from. Saving
  a changed matrix on a shared cell that no configured calculation uses changes it
  for every workflow that has it selected, after asking (see `Editing saved
  supercells`_). UEP source jobs use the currently selected UEP source structure
  as their supercell context.

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
- ``Suggest…`` proposes diagonal supercells only; use ``General 3×3 matrix``
  for anything else. It does not check the propagation vector; the editor's
  warning does, once a suggestion is loaded.
- A saved supercell that a configured calculation uses cannot be deleted, and
  editing its matrix saves a new copy.

For developers
--------------

How these selections are stored in the project is described under "Supercell
selections" in :doc:`/architecture`. A convergence definition records
``simulation_cell_mode`` and ``supercell_source_workflow`` in its parameters,
and its first input structure is the selected supercell, or the host for
unit-cell mode.
