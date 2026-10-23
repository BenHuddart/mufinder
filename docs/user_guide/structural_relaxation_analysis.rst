Structural Relaxation Analysis
==============================

This page covers the stages that take a structure from a relaxed host to a
short list of candidate muon stopping sites. ``Pristine Relaxation`` relaxes the
host without a muon, so you can compare it with the experimental structure and
use it as a reference for displacement analysis. After the muon relaxations
have run under ``Muon Sites -> Relaxations``, an analysis workspace in
``Muon Sites -> Analysis`` groups their final muon positions into distinct
sites, assigns each site a Wyckoff position, and pins the result to the
``Muon Sites`` summary, which later workflows read. The last section describes
how to continue any relaxation from its checkpoint.

Pristine Relaxation
-------------------

The optional ``Pristine Relaxation`` stage relaxes the host structure, or a
supercell of it, without a muon.

Before you start
~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~

1. **Open the page.** Select ``Pristine Relaxation`` in the navigator.
2. **Add a row.** Click ``Add Job``. Each row is one relaxation job, labelled
   ``relax_1``, ``relax_2`` and so on.
3. **Choose the cell.** In the row's ``Supercell`` drop-down, pick ``Unit
   cell`` or a saved supercell, or choose ``Manage Supercells...`` to open the
   :doc:`/user_guide/supercells`.
4. **Decide whether the cell may change.** The ``Fix Cell`` drop-down reads
   ``No (relax)`` by default, so the lattice relaxes as well as the atoms.
   Choose ``Yes (fixed)`` to relax the atomic positions in the starting cell.
5. **Check the parameters.** Click the row's ``Configure...`` button, or select
   rows and click ``Configure Selected Parameters...``. The shared parameter
   dialog lists ``Basis Precision`` / ``Cutoff Energy``, ``SCF Energy
   Tolerance`` and ``K-Point MP Grid`` first.
6. **Run the jobs.** Set the ``Job name prefix`` (``pristine_relax`` by
   default) and click ``Run All Locally`` or ``Run All on Cluster``. With rows
   selected the buttons read ``Run Selected Locally`` / ``Run Selected on
   Cluster``. Follow the jobs in the ``Status`` column and in the job
   dashboard (see :doc:`/getting_started/running_jobs`).
7. **Read the relaxed lattice.** When a job completes, its ``a (Ang)``,
   ``b (Ang)`` and ``c (Ang)`` cells fill in. A job whose CASTEP output
   reports incomplete geometry convergence is marked failed.
8. **Compare with experiment.** Click ``Generate Comparison Table...`` (see
   :doc:`/user_guide/comparison_table`), or use a completed relaxation as the
   comparison host in `Displacement analysis`_.

The Pristine Relaxation page
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/pristine_relaxation_page.webp
   :alt: The Pristine Relaxation page with one completed variable-cell relaxation and
         its relaxed lattice lengths.

   A pristine relaxation of CoF₂, from a synthetic CASTEP output
   (:doc:`/user_guide/examples/cof2`).

The ``Relaxation Jobs`` table has one row per job:

``Label``
   The job's name.
``Supercell``
   ``Unit cell`` or a saved supercell, plus ``Manage Supercells...``.
``Fix Cell``
   ``Yes (fixed)`` or ``No (relax)``. MuFinder writes a fixed cell as
   ``FIX_ALL_CELL``.
``Status``
   The job's status, starting at ``ready``.
``a (Ang)``, ``b (Ang)``, ``c (Ang)``
   The relaxed lattice lengths, once the job has completed.
``Parameters``
   ``Configure...``, which opens the shared CASTEP parameter dialog for the
   row.

Generated jobs take the project default XC functional, cutoff energy and
k-point grid. For a supercell row MuFinder rescales the grid to keep the
reciprocal-space sampling density of the reference lattice the grid was
converged on: each component scales with the change in the matching reciprocal
axis and always rounds up, so the cell is never undersampled. For example, a
``4x4x4`` default becomes ``2x2x2`` for diagonal ``2x2x2`` and ``2x2x3``
supercells.

What MuFinder writes
~~~~~~~~~~~~~~~~~~~~

Each job runs in its own folder under the project's
``files/pristine_relaxation`` folder. On completion MuFinder retrieves the
``<seed>.castep`` and ``<seed>-out.cell`` outputs and reads the relaxed
geometry (lattice, species and fractional coordinates) from ``-out.cell``.
Each successful job adds one relaxed structure to the project. These are the
structures the :doc:`/user_guide/comparison_table` compares with the
experimental host, and they can be the comparison host in `Displacement
analysis`_.

Analysis workspaces
-------------------

An analysis workspace collects completed muon geometry optimisation results and
treats their final muon positions as an analysis set. A workspace is for either
positive muon or muonium calculations, and only takes relaxations from that
charge state.

Before you start
~~~~~~~~~~~~~~~~

An analysis workspace needs:

- **Completed, converged muon relaxations.** Generate sites and run their
  relaxations first (:doc:`/user_guide/site_generation`). The workspace takes
  the host structure that is active when you create it.
- **One charge state per workspace.** Rows from the ``Positive muon`` and
  ``Muonium`` tabs cannot be mixed; create one workspace for each.

Running the workflow
~~~~~~~~~~~~~~~~~~~~

1. **Select the relaxations.** In ``Muon Sites -> Relaxations``, open the
   charge-state tab you want and select completed, converged rows.
2. **Create the workspace.** Click ``Create Analysis Workspace from
   Selection``. In ``Create Analysis Workspace``, enter the ``Workspace name``
   (default ``positive muon analysis`` or ``muonium analysis``) and accept.
   Selecting rows from both charge states is refused.
3. **Add more structures if needed.** The workspace opens as a tab in ``Muon
   Sites -> Analysis`` and appears as a node under ``Analysis`` in the project
   tree. Click ``Add Structure to Analysis`` to add further completed
   relaxations of the same charge state, ticking ``Add`` for each. The dialog
   reports each relaxation's energy as ``dE vs workspace min (eV)``, so a
   negative value means that adding it would reset the workspace zero.
4. **Cluster the sites.** Click ``Clustering...``, choose a ``Method`` and
   click ``Apply Clustering`` (see `Clustering`_).
5. **Inspect distortions (optional).** Click ``Analyse Displacements`` (see
   `Displacement analysis`_).
6. **Pin the result.** Click ``Pin to Summary`` to copy the distinct sites to
   the ``Muon Sites`` summary page (see `The Muon Sites summary`_).
7. **Export (optional).** Click ``Export CIF...`` for the host with every
   muon position (see `Exporting to CIF`_), or ``Export Relaxed Structures``
   for the relaxed supercells themselves (see `Exporting relaxed structures`_).

An analysis made before every relaxation has finished goes stale once the full
set is available; delete it and make a new one (see `Deleting a workspace`_).

The workspace page
~~~~~~~~~~~~~~~~~~

.. figure:: ../_generated/screenshots/relaxation_analysis_workspace.webp
   :alt: An analysis workspace with ten CoF2 positive-muon relaxations clustered
         into 4c, 4d and 8i sites, beside a 3D preview of the host cell.

   Synthetic CoF₂ relaxations clustered into the three sites of
   :doc:`/user_guide/examples/cof2`.

Each workspace is a tab on the ``Muon Sites -> Analysis`` page. Selecting a
workspace's node in the project tree opens its tab, and switching tabs selects
the matching node. The node is marked complete once clustering has been
applied.

The header of each workspace has:

``Workspace``
   The workspace name. Edit it here; the project-tree node is renamed with it.
``Charge state``
   ``Positive muon`` or ``Muonium``, fixed when the workspace was created.
``Show shifted positions``
   Display option described under `Site ordering`_.
``Delete Analysis...``
   Removes the workspace (see `Deleting a workspace`_).
``Add Structure to Analysis``, ``Clustering...``, ``Analyse Displacements``, ``Pin to Summary``, ``Export CIF...``
   The steps above.
``Export Relaxed Structures``
   A drop-down that saves the relaxed supercells of the selected sites (see `Exporting
   relaxed structures`_). Select site rows in the table with Ctrl or Shift
   click.

Workspaces are saved with the project, so they are restored after a manual
project save and load. Creating or editing a workspace is not autosaved while
the application is running; deleting one is.

Clustering
~~~~~~~~~~

The ``Clustering...`` button opens the ``Clustering - <workspace>`` dialog,
which provides three methods, selected with ``Method``. Click ``Apply
Clustering`` to store the result on the workspace.

.. figure:: ../_generated/screenshots/relaxation_clustering_dialog.webp
   :alt: The Clustering dialog with Connected Components at 0.25 A: a graph of
         seven, one and two linked sites, the cluster-count plot and the
         clustered-sites table with 4c, 4d and 8i Wyckoff assignments.

   Connected-components clustering of the CoF₂ positive-muon workspace.

``Connected Components``
   Two relaxed muon endpoints belong to the same cluster when they are linked
   by a chain of endpoints whose symmetry-aware periodic distance is below the
   ``Distance tolerance (Å)``. The dialog shows the graph of subclusters and a
   plot of number of clusters against tolerance. Each manual tolerance
   evaluation adds a point to the plot. ``Scan Tolerances`` evaluates a
   regular tolerance grid so you can choose a stable plateau in the
   cluster-count curve deliberately.
``Hierarchical``
   Cuts a dendrogram at the threshold ``Hierarchical t``.
``K-means``
   Divides the endpoints into the number of clusters you request in
   ``Clusters k``.

The ``Wyckoff tolerance (Å)`` control sets the maximum allowed distance from
every anchored cluster member to the Wyckoff orbit assigned to its cluster (see
`Wyckoff assignment`_).

The clustered-sites table in the dialog reports each cluster's Wyckoff
assignment. Tooltips include the detected space group, the site-symmetry
symbol, and the maximum and mean fitted distances used to justify the
assignment. The structure viewer gets half the width of the dialog, or more
when its cluster summary table needs it to show the Wyckoff assignments in
full; the plot takes the rest. The split follows window resizes until you drag
the handle.

The connected-components graph scales with its canvas: every node stays inside
the plot and nodes never overlap, however small the window. Components are
packed into rows chosen to fit the canvas's shape and the packing is redone
when the window is resized. Nodes stop growing at about 22 pt in radius, so a
small graph on a large window stays readable, and components with more than 2.5
edges per node draw thinner, translucent edges so the edges of a dense
component can still be told apart.

Site ordering
~~~~~~~~~~~~~

Within a workspace, muon sites are ordered by increasing total energy. The
energy column is reported relative to the lowest-energy member of the
workspace. Final muon coordinates are folded back into the host unit cell.

``Show shifted positions`` applies a display-only symmetry anchoring step. It
chooses a representative site within each cluster and shows
symmetry-equivalent positions close to that representative where possible.
Changed coordinate cells are shown in blue. The saved relaxed structures are
not edited by this operation.

Wyckoff assignment
~~~~~~~~~~~~~~~~~~

Muon sites often relax close to high-symmetry interstitial positions. After
clustering, MuFinder assigns each cluster the highest-symmetry unoccupied
Wyckoff position of the host space group that is consistent with all members of
the cluster. The clustered-sites table shows this in the cluster header, for
example ``Cluster 1: (0.2500, 0.5000, 0.0000) 12c``.

The assignment is a tolerance-based fit rather than a claim that the relaxed
muon is exactly on the Wyckoff position. If no unoccupied special position
lies within the ``Wyckoff tolerance (Å)`` of every anchored cluster member,
MuFinder reports the assignment as unavailable rather than hiding a cluster
outlier behind the representative site. See `Method`_ for how the position is
fitted.

Displacement analysis
~~~~~~~~~~~~~~~~~~~~~

``Analyse Displacements`` opens an ``Analyse Displacements - <workspace>``
view of how each relaxed muon site distorts the host; clustering is not
required. Choose one relaxed muon site from the workspace (``Muon site``) and
a comparison host structure (``Comparison host``). The comparison structure may
be the imported host or a completed pristine relaxation, provided the pristine
relaxation was not run as a free-cell relaxation. Free-cell pristine
relaxations are disabled in the selector because cell-shape changes make
ion-by-ion displacement interpretation ambiguous.

MuFinder creates the comparison supercell needed to match the relaxed cell,
matches host atoms with periodic boundary conditions, and removes the implanted
muon from the relaxed structure before atom matching. The structure view shows
the relaxed structure with the comparison host atoms as transparent reference
points. Distances are measured from the site the muon relaxed to in the
relaxed supercell. The plot reports radial ion displacement relative to the
selected muon site against ion distance from the muon, in the style of Fig. 5(b) in the
MuFinder paper. Positive radial displacement means the matched ion moved away
from the muon.

.. figure:: ../_generated/screenshots/displacement_analysis_dialog.webp
   :alt: The Analyse Displacements dialog for a CoF2 4c site: the two nearest F
         moved in by about half an angstrom and the two nearest Co moved out.

   The F–μ–F distortion around the CoF₂ 4c site.

Deleting a workspace
~~~~~~~~~~~~~~~~~~~~

Click ``Delete Analysis...`` in a workspace's header, or right-click the
workspace's node in the project tree and choose ``Delete Analysis...``. MuFinder
asks for confirmation first, with ``Cancel`` as the default. If the deleted
workspace was the one showing, the Analysis page stays open.

Deleting a workspace discards only the workspace and its clustering. The
structural relaxations it used are kept, so they can be added to a new workspace
straight away. If the workspace's sites were pinned to the ``Muon Sites``
summary page, that pin is removed as well, and the confirmation says so. A pin
taken from another workspace is left alone. The deletion is saved to the
project immediately and cannot be undone.

The Muon Sites summary
----------------------

.. figure:: ../_generated/screenshots/muon_sites_summary.webp
   :alt: The Muon Sites page with three positive-muon sites (4c, 4d, 8i) and two
         muonium sites (8i, 4d) pinned.

   Both cards populated: the CoF₂ positive-muon and muonium workspaces, pinned.

The ``Muon Sites`` page (the parent node above ``Generation``, ``Relaxations``
and ``Analysis``) summarises your headline result: the distinct,
symmetry-inequivalent candidate stopping sites for each charge state. It has
one card for ``Positive muon`` and one for ``Muonium``, shown side by side.

To populate a card, open a clustered analysis workspace and click ``Pin to
Summary``. MuFinder copies the workspace's distinct clustered sites into the
card for the matching charge state, ranked by relative energy (lowest first),
and shows each site's relative energy, representative fractional position, and
Wyckoff label. A 3D preview renders the host structure with the pinned muon
sites. Pinning requires that clustering has been run, that the workspace was
built on the active host structure, and that its charge state is supported; the
button reports if there is nothing to summarise yet or the host does not match.
Click ``Unpin`` to clear a card. A pinned card is a snapshot, independent of the
live workspaces, and each pinned site remembers the relaxation that represents
its cluster.

Quantum Corrections, Energy Barriers and the Magnetic Fields muon-site dialog
list the pinned sites by default, labelled by Wyckoff label and cluster and
ranked by relative energy as on the summary. Each page has a ``Show every
relaxed site`` option that lists every eligible relaxed site instead; the
choice is saved per workflow. Pinning or unpinning updates those pages straight
away.

You can navigate from the summary straight into the live analysis:

- ``Open in Analysis →`` jumps to the workspace the card was pinned from. When
  a card is empty, this button instead reads ``Go to Analysis →`` and takes you
  to the Analysis tab so you can create and cluster a workspace.
- Clicking a site row in the ranked table, or clicking a muon in the 3D
  preview, opens the source workspace and reveals that specific cluster.

MuFinder validates a pinned card against the current host structure and hides
it if the host changes. If the source workspace or cluster no longer exists
when you follow a link, MuFinder opens the Analysis tab and notes that the
pinned analysis has changed.

Exporting to CIF
----------------

Both the live analysis and the pinned summary can export the host structure
together with muon positions as a CIF file. Muons are written as hydrogen atoms
with distinct ``Mu1``, ``Mu2``, ... ``_atom_site_label`` values. The file is
written in P1 with every atom listed explicitly, since muon sites generally
break the host's symmetry.

- In the ``Analysis`` tab, each workspace's ``Export CIF...`` button exports
  every relaxed muon position currently in that workspace: one CIF site per
  completed relaxation added to the workspace, whether or not clustering has
  been applied. Use this when you want the full set of raw relaxation endpoints
  rather than a clustered summary.
- On the ``Muon Sites`` summary page, each card's ``Export CIF…`` button
  exports only the pinned, distinct clustered sites (one representative per
  cluster), matching what the card's table and 3D preview show. It stays
  disabled until the card is populated.

The standard ``_atom_site`` loop carries no MuFinder-specific columns, because
an extra column can desynchronize the column count in some CIF readers (VESTA
among them) and corrupt the whole structure. Per-site relative energy and
Wyckoff label live in their own standalone loop below the atom list, keyed by
``_mufinder_muon_site_label``, which a reader that does not recognise it can
skip without touching atom parsing.

If a muon site is written closer than 0.5 Å to a host atom, or exactly on one,
MuFinder still writes the file but reports which sites and host atoms are
affected. The file is not malformed (each atom keeps full, independent
occupancy), but many external CIF viewers detect the overlap and render the
pair as a single merged, fractionally-occupied site (for example "Sr1/Mu1")
rather than two distinct atoms. Seeing this warning, or a merged site of this
kind in another program, usually indicates that the underlying analysis needs
another look (for example the wrong host or workspace charge state, or data
pinned or added from a mismatched structure) rather than a genuine muon site
landing on a host ion.

.. _export-relaxed-structures:

Exporting relaxed structures
----------------------------

A completed pristine or muon relaxation can be saved as the full relaxed
supercell that CASTEP wrote to ``{seed}-out.cell``: every relaxed host atom
plus, for a muon relaxation, the muon. Imported legacy relaxations export the
same way.

Both entry points offer the two formats as separate menu entries, ``CIF
(.cif)…`` and ``CASTEP cell (.cell)…``, so the format is chosen before any
dialog opens:

- In the ``Pristine Relaxation`` table or the ``Muon Sites -> Relaxations``
  table, right-click one row and choose ``Export Relaxed Structure`` and then
  a format.
- In an analysis workspace, click ``Export Relaxed Structures`` and choose a
  format from its drop-down. With one site row selected it saves that site.
  With several selected it saves those sites, and with none selected it saves
  every site in the workspace. Several sites are written to a folder you
  choose, one file per site, named after each relaxation (for example
  ``site_1_muonium_relaxed.cif``). Replacing existing files is confirmed first.

The save dialog suggests a name with the chosen extension, made from the
relaxation and its charge state, in the folder that holds the project. A name
typed without that extension gets it added.

``CIF (.cif)``
   Written in P1 with every atom listed. Host atoms are labelled
   by element and count (``Fe1``, ``Fe2``, ...). The muon is a hydrogen atom
   labelled ``Mu1``, so it stays distinct from any hydrogen in the host.
``CASTEP cell (.cell)``
   ``LATTICE_CART`` in Å and ``POSITIONS_FRAC``, with ``SPIN=`` values when the
   structure carries non-zero spins. The muon keeps its ``H:mu`` species
   label, so the file reopens in MuFinder and can seed a new CASTEP run.

Both formats start with comment lines that record the project, the structure,
the source job and its workflow, the charge state, the final energy in eV and
the export time. Both read back with the relaxation's lattice and fractional
positions, in MuFinder's structure loader or in viewers such as VESTA. A CIF
reads the muon back as hydrogen.

``Export Relaxed Structure`` is greyed out until the row's job has completed.
It is also disabled when the job completed without a relaxed structure, for
example because its ``-out.cell`` was missing or could not be parsed. Hover
over the disabled action to see which applies.

.. _continue-relaxation:

Continuing a relaxation from its checkpoint
-------------------------------------------

Every relaxation ``.param`` MuFinder generates contains ``CONTINUATION :
DEFAULT``, so CASTEP resumes from a matching ``{seed}.check`` file whenever one
is in the run directory. Use this to extend a relaxation that ran out of steps
or to change its parameters and carry on.

1. **Open the row menu.** In the ``Pristine Relaxation`` table or the ``Muon
   Sites -> Relaxations`` table, right-click a row whose job is ``completed``, ``failed``, ``stopped``
   or ``cancelled``, and choose ``Continue relaxation…``. The action is offered
   for one row at a time.
2. **Review the dialog.** ``Continue Relaxation`` opens with the row's current
   parameters in the shared CASTEP parameter editor, a live keyword-level diff
   of the old and new ``.param`` text under ``Parameter changes`` (added,
   changed and removed keywords), a checkpoint status line (checkpoint found,
   not found, or could not be verified), a note that the ``.cell`` is
   regenerated from the row's structure (this dialog does not edit atomic
   positions), and a summary naming the execution target and the directory that
   will be reused.
3. **Accept.** Click ``Continue``. The row keeps its job, which returns to
   ``defined``, and MuFinder rewrites the ``.cell`` and ``.param`` into the same
   directory, locally or on the cluster, then resubmits the job to its original
   execution target.

A local job goes back through local submission. A cluster job goes back into
its existing remote directory, with no existence prompt and nothing deleted, as
long as the cluster selected in the submission dialog is the job's original
cluster; choosing a different cluster uses the normal submission path, with a
new target directory and an existence check.

When the continued job completes again, its relaxed structure is refreshed in
place, so comparison tables and other references stay valid, and a muon site's
fields are refreshed from the same completion. A continued run that dies before
writing a new ``-out.cell`` fails rather than reusing the previous one. Each
continuation is recorded in the Journal.

The job dashboard's ``Resubmit`` also resumes from the checkpoint when you
choose ``Continue from the checkpoint`` (see :doc:`/getting_started/running_jobs`).
Neither route deletes the run directory's contents. ``Reset Failed Jobs...``
with its optional clean-up does: it deletes the local ``.check`` and
``.castep`` files and ``{seed}-out.cell``, keeping only the ``.cell`` and
``.param`` inputs, although it never cleans a folder another job still uses.

The regenerated ``.cell`` always holds the row's own structure: there is no
mode that writes the latest relaxed geometry into it, and MuFinder does not use
the CASTEP ``reuse`` keyword.

Limitations
-----------

- ``Analyse Displacements`` cannot use a free-cell pristine relaxation as the
  comparison host.
- A pinned card is tied to the host structure it was built on and is hidden
  when the active host changes.
- Wyckoff assignment is a tolerance-based fit; it reports no assignment rather
  than a position that one cluster member misses.

Method
------

Wyckoff fitting
~~~~~~~~~~~~~~~

MuFinder first anchors the cluster members in the same non-mutating way used
for shifted positions, then fits nearby special-position loci from the host
symmetry operations. This includes variable-coordinate special positions, not
only fixed high-symmetry points. Each candidate is labelled by generating its
full symmetry orbit and asking ``spglib`` for the corresponding Wyckoff letter
and multiplicity. Occupied host positions are rejected by checking whether the
candidate orbit overlaps any host atom within the occupancy tolerance.

The candidate special positions come from projecting the cluster onto the
fixed-point locus of every symmetry operation and of every pair of operations,
which is several thousand pairs for a high-symmetry cubic host. To keep the
clustering dialog responsive, MuFinder skips pairs that cannot produce a
solution, screens the rest in one vectorised pass, and projects exactly only
the pairs that can land within range of the cluster. It then asks ``spglib`` to
label candidates in rank order, so only the winning candidate is usually
labelled. None of this changes which site is chosen. Host symmetry operations
and per-cluster assignments are cached, so switching clustering methods reuses
the assignments of clusters that did not change.

Clustering methods
~~~~~~~~~~~~~~~~~~

``Connected Components`` builds a graph in which each relaxed muon endpoint is a
node. Two nodes are connected when their shortest symmetry-aware periodic
distance is below the chosen ``Distance tolerance (Å)``, and the clusters are
the connected components of that graph. This follows the graph-style treatment
described for relaxed sites in the MuFinder paper.

``Hierarchical`` and ``K-means`` use the Soprano and pymuon-suite inspired
feature representation of the UEP batch-search workflow. The free parameters
are the dendrogram threshold ``Hierarchical t`` and the requested number of
clusters ``Clusters k``.

The symmetry-aware distances do not depend on the tolerance, so ``Scan
Tolerances`` measures them once and reuses them at every grid point. Host
symmetry operations are cached per host, and each site's symmetry-equivalent
positions are compared in one vectorised pass. Near ties are re-measured
exactly, so the distances, cluster counts and anchored positions match a
site-by-site evaluation.

The graph sizes its nodes, their border rings, their numbers and its edges in
data units. Node size follows the closest pair of nodes, and large components
are spread so neighbouring nodes keep a consistent gap. Components are kept in
cluster order, with gaps tied to the node size.

Sources
~~~~~~~

This workflow acknowledges and follows the methods described in:

* B. M. Huddart et al., "MuFinder: A program to determine and analyse muon
  stopping sites", *Computer Physics Communications* **280**, 108488 (2022),
  DOI ``10.1016/j.cpc.2022.108488``. The connected-components clustering
  visualisation and radial displacement plot are based on the approaches shown
  in that work.
* L. Liborio, S. Sturniolo, and D. Jochym, "Computational prediction of muon
  stopping sites using ab initio random structure searching (AIRSS)",
  *J. Chem. Phys.* **148**, 134114 (2018), DOI ``10.1063/1.5024450``. The
  workspace follows the AIRSS-style practice of clustering relaxed muonated
  structures and selecting representative low-energy structures.
* The pymuon-suite project, especially the ``pm-muairss`` clustering workflow,
  which uses hierarchical and k-means clustering parameters
  ``clustering_hier_t`` and ``clustering_kmeans_k``. MuFinder reuses the same
  Soprano-inspired clustering interface conventions in this analysis view.
* The pymuon-suite ``pm-symmetry`` workflow, which uses symmetry analysis to
  report candidate high-symmetry Wyckoff points and whether they are occupied.
  MuFinder follows that idea for relaxed structural clusters, while fitting the
  nearest unoccupied Wyckoff orbit to all members of each cluster.

For developers
--------------

The CIF export is implemented by
:func:`mufinder.science.structure_io.export_structure_with_muon_sites`. The
continuation contract (status transition and the cleared and kept fields) is
described in
:ref:`job-continuation-primitive` in :doc:`/job_lifecycle`; the checkpoint probe
is :ref:`remote-checkpoint-probe`. Package layers and the shared workflow
infrastructure are described in :doc:`/architecture`.
