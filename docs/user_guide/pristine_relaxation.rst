Pristine Relaxation
===================

The optional ``Pristine Relaxation`` stage relaxes the host structure, or a
supercell of it, without a muon. It answers two questions before the muon
calculations start: do the functional and settings reproduce the experimental
structure, and what is the relaxed reference for later displacement analysis? For
a row that starts from the experimental structure, the lattice changes on the page
give the first answer at a glance. A comparison table adds the atomic displacements,
the space group and the differences between several relaxations.

Before you start
----------------

Pristine Relaxation needs:

- **A host structure.** Load one first.
- **A supercell, if you want one.** Each row relaxes the host unit cell or a
  supercell saved in the :doc:`/user_guide/supercells`.
- **Project defaults, ideally.** Generated jobs take the project default XC
  functional, cutoff energy and k-point grid, so adopt them first in
  :doc:`/user_guide/convergence_testing`.
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`/castep_keyword_catalog`).

Running the workflow
--------------------

1. **Open the page.** Select ``Pristine Relaxation`` in the navigator.
2. **Add a row.** Click ``Add Job``. Each row is one relaxation job, labelled
   ``relax_1``, ``relax_2`` and so on.
3. **Choose the cell.** In the row's ``Supercell`` drop-down, pick ``Unit
   cell`` or a saved supercell, or choose ``Manage Supercells…`` to open the
   :doc:`/user_guide/supercells`.
4. **Decide whether the cell may change.** The ``Fix cell`` checkbox is clear
   by default, so the lattice relaxes as well as the atoms. Tick it to relax the
   atomic positions in the starting cell, which a relaxation needs if it is to be
   the comparison host in :ref:`displacement-analysis`.
5. **Check the parameters.** Click the row's ``Configure…`` button, or select
   rows and click ``Configure Selected Parameters…``. The shared parameter
   dialog lists ``Basis Precision`` / ``Cutoff Energy``, ``SCF Energy
   Tolerance`` and ``K-Point MP Grid`` first.
6. **Run the jobs.** Set the ``Job name prefix`` (``pristine_relax`` by
   default) and click ``Run All Locally`` or ``Run All on Cluster``. With rows
   selected the buttons read ``Run Selected Locally`` / ``Run Selected on
   Cluster``. Follow the jobs in the ``Status`` column and in the job
   dashboard (see :doc:`/getting_started/running_jobs`).
7. **Read the relaxed lattice.** When a job completes, its ``a (Ang)``,
   ``b (Ang)`` and ``c (Ang)`` cells show each relaxed length and its change
   against the host, for example ``4.7091 (+0.30 %)``. A job whose CASTEP output
   reports incomplete geometry convergence is marked failed.
8. **Compare with experiment.** Click ``Generate Comparison Table…`` (see
   `The comparison table`_), or use a completed fixed-cell relaxation as the
   comparison host in :ref:`displacement-analysis`.

The Pristine Relaxation page
----------------------------

.. figure:: ../_generated/screenshots/pristine_relaxation_page.webp
   :alt: The Pristine Relaxation page with one completed variable-cell relaxation,
         its relaxed lattice lengths and their changes against the host.

   A pristine relaxation of CoF₂, from a synthetic CASTEP output
   (:doc:`/user_guide/examples/cof2`).

In a new project the ``Relaxation Jobs`` group holds a hint to add a job. Once
there is a job, its table has one row per job. Drag the edge of the ``Label`` or
``Supercell`` header to widen the column; the other columns fit their contents.

``Label``
   The job's name.
``Supercell``
   ``Unit cell`` or a saved supercell, plus ``Manage Supercells…``. A name too
   long for the column ends in ``…``. The drop-down's tooltip gives the full name
   and, when the row's cell changes the project k-point grid, the grid the project
   default gives the row.
``Fix cell``
   Ticked, the cell stays fixed; clear, it relaxes. MuFinder writes a fixed cell
   as ``FIX_ALL_CELL``.
``Status``
   The job's status, starting at ``ready``.
``a (Ang)``, ``b (Ang)``, ``c (Ang)``
   The relaxed lattice lengths once the job has completed, to five significant
   figures, each with its percent change against the host. The host is the
   structure the row relaxed from; a supercell row compares against the host
   tiled to the row's cell, so its lengths are those of the whole supercell. If
   that structure has since been removed from the project, the cell shows the
   length alone.
``Parameters``
   ``Configure…``, which opens the shared CASTEP parameter dialog for the
   row.

``Generate Comparison Table…`` sits beside ``Add Job`` under the table.

Generated jobs take the project default XC functional, cutoff energy and
k-point grid. For a supercell row MuFinder rescales the grid to keep the
reciprocal-space sampling density of the reference lattice the grid was
converged on: each component scales with the change in the matching reciprocal
axis and always rounds up, so the cell is never undersampled. For example, a
``4x4x4`` default becomes ``2x2x2`` for diagonal ``2x2x2`` and ``2x2x3``
supercells. When you run such rows the status bar adds how many of them were
rescaled to its report of the submission, ``K-point grid rescaled for 4 rows``.

``Generate Symmetry`` is on by default, so the ``.cell`` file carries
``SYMMETRY_GENERATE``. Unchecking it in the parameter dialog leaves the keyword
out for that row, and the choice is saved with the project.

A job that ran out of steps, or whose parameters you want to change, can carry on
from its checkpoint instead of starting over (see :ref:`continue-relaxation`). A
completed job's relaxed cell can be saved as a file with ``Export Relaxed Structure``
in its row menu (see :ref:`export-relaxed-structures`).

What MuFinder writes
--------------------

Each job runs in its own folder under the project's
``files/pristine_relaxation`` folder. On completion MuFinder retrieves the
``<seed>.castep`` and ``<seed>-out.cell`` outputs and reads the relaxed
geometry (lattice, species and fractional coordinates) and the run's per-atom
``SPIN=`` values from ``-out.cell``. Each successful job adds one relaxed
structure to the project, and continuing a job updates that structure in place.
The structure keeps the run's spins, so a spin-polarised calculation started
from it, such as UEP, site generation or a muon relaxation on the relaxed host,
starts from the same moments. These are the structures a comparison table
compares with the experimental host, and those from fixed-cell jobs can be the
comparison host in :ref:`displacement-analysis`.

.. _pristine-comparison-table:

The comparison table
--------------------

A comparison table checks how well one or more relaxed structures reproduce the
project's active host structure. It reads the lattice parameters, the
displacements of the matched atoms and the space groups, and it works across
supercells: the atoms of a supercell relaxation are compared with the host tiled
to the same cell, and its lattice parameters are read per host cell.
:doc:`/user_guide/comparison_table` gives the method.

Generating a table
~~~~~~~~~~~~~~~~~~

The table needs a loaded host structure and at least one completed pristine
relaxation; MuFinder reports which of the two is missing when you click the
button.

1. **Start the comparison.** Click ``Generate Comparison Table…``.
2. **Choose the relaxations.** The ``Generate Comparison Table`` dialog lists
   the completed pristine-relaxation jobs with ``Include`` ticked for each; clear
   any you want to leave out. Each row shows the job's ``Label``, ``Supercell``,
   ``Status``, ``Total Energy (eV)``, and relaxed ``a (Ang)``, ``b (Ang)`` and
   ``c (Ang)`` lengths. Leaving nothing ticked is refused.
3. **Name the table.** Edit the ``Comparison table label`` if needed (it
   defaults to ``Comparison <N>``) and click ``OK``.

   .. figure:: ../_generated/screenshots/comparison_job_dialog.webp
      :alt: The Generate Comparison Table dialog with one completed CoF2
            pristine relaxation ticked and its relaxed lattice lengths.

      ``Generate Comparison Table`` with the CoF₂ pristine relaxation ticked.
4. **Read the table.** MuFinder opens the new table, listed under ``Pristine
   Relaxation`` in the project tree. It is saved with the project.
5. **Rename or delete it.** The title at the top of an existing table stays
   editable: editing it updates the table's label in the project tree and is
   saved with the project. The ``Delete`` button at the top right deletes the
   table after a confirmation prompt, with ``Cancel`` as the default.
6. **Export it (optional).** Click ``Export CSV…`` to save the displayed values
   for external analysis or record-keeping.

Reading a table
~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/comparison_table.webp
   :alt: A comparison table of the CoF2 host against one relaxed cell, grouped into
         lattice, displacement and symmetry metrics.

   The CoF₂ host of :doc:`/user_guide/examples/cof2` compared with a synthetic
   pristine relaxation.

The table has a ``Metric`` column, an ``Experimental`` column for the reference
structure, and one column per selected DFT structure, headed ``DFT (<job
label>)``. Rows are grouped into three categories:

- ``Lattice parameters``: ``a``, ``b``, ``c`` (Å), ``α``, ``β``, ``γ`` (°) and
  the cell volume ``V`` (Å³) of the host cell. Each DFT cell shows the value
  with its change against experiment, such as ``4.7091 (+0.30 %)``. A DFT
  column that is a supercell of the host is shown per host cell, so it reads
  like the ``Experimental`` column (see :ref:`comparison-supercell-columns`).
- ``Atomic displacements``: the ``RMSD`` and the ``Max displacement`` of the
  matched atoms (Å).
- ``Symmetry``: the ``Space group`` of each structure, such as
  ``P4₂/mnm (136)``, and whether the DFT structure keeps the experimental one
  (``Same space group``).

Hovering over a metric name shows a one-line description, and ``About these
metrics`` above the table opens :doc:`/user_guide/comparison_table`. When two or
more DFT columns have a numeric value for a metric, the column closest to
experiment is shown in bold green; because every column is read per host cell, a
supercell column can be the closest. A warning banner above the table lists
problems found while comparing, such as a failed supercell detection or a partial
atom matching, and notes each supercell column, for example ``DFT (CoF2 2x2x2): 2
× 2 × 2 supercell; its lattice parameters are shown per host cell.``

The collapsible ``DFT Parameter Differences`` section lists only those CASTEP
input parameters that differ across the table's DFT columns. A selected job
whose relaxed structure has been removed from the project gets no column and
is left out of this comparison.

Read a table in this order:

1. Check the warnings for failed supercell detection or partial atom matching.
2. Read the lattice changes: a well-converged relaxation usually lands within a
   percent or two of experiment.
3. Read the RMSD and maximum displacement for the internal coordinates; a
   maximum displacement much larger than the RMSD points to one site that moved.
4. Check that the space group is kept. A change of space group means the
   relaxation broke the experimental symmetry.

A saved table shows the values computed when it was generated; opening it
does not recompute them, so generate the table again to read it with the
method on :doc:`/user_guide/comparison_table`. A saved table that holds metrics
outside this list opens with the rows it shares with the list and leaves the
others out; generating the table again fills in every row.

The page's lattice changes and the table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``a (Ang)``, ``b (Ang)`` and ``c (Ang)`` cells on the page and the lattice rows
of a table can show different numbers for the same job, because they answer
different questions:

- **The reference.** A row compares against the structure it relaxed from, and a
  table against the host that is active when you generate it.
- **The cell.** A supercell row shows the lengths of the whole supercell, and a
  table shows them per host cell.

Generate a table when you need the angles and volume, the atomic displacements,
the space group, the parameter differences between jobs, or a CSV file.

Limitations
-----------

- ``Analyse Displacements`` cannot use a free-cell pristine relaxation as the
  comparison host.
- Space-group comparison depends on ``spglib`` and on the symmetry tolerance.
- Large supercells increase the cost of the species-resolved Hungarian
  assignment.
