Structural Relaxation Analysis
==============================

The structural relaxation analysis workspace collects completed muon geometry
optimisation results and treats their final muon positions as an analysis set.
The usual order of work is:

1. In ``Muon Sites -> Relaxations``, select completed, converged rows from one
   charge-state tab and click **Create Analysis Workspace from Selection**.
   MuFinder asks for a workspace name (default ``positive muon analysis`` or
   ``muonium analysis``), records whether the workspace is for positive muon or
   muonium calculations, and only allows later additions from the same
   charge-state family. Selecting rows from both charge states is refused.
2. The workspace opens as a tab in ``Muon Sites -> Analysis`` and appears as a
   node under ``Analysis`` in the project tree; selecting the node opens its
   tab, and switching tabs selects the matching node. Use
   **Add Structure to Analysis** to add further completed relaxations; the
   workspace name can be edited in its ``Workspace`` field, which also renames
   its project-tree node.
3. Click **Clustering...**, choose a method, and click **Apply Clustering**
   (see `Clustering`_).
4. Optionally inspect distortions with **Analyse Displacements**, and click
   **Pin to Summary** to copy the distinct sites to the ``Muon Sites`` summary
   page.

Workspaces are stored in the project file under the saved UI state. They are
therefore restored after a manual project save and load. Creating or editing a
workspace is not autosaved while the application is running; deleting one is
(see `Deleting a Workspace`_). Each workspace's project-tree node follows the
selected tab and the workspace name, and is marked complete once clustering has
been applied.

The `Pristine Relaxation`_ stage and `Continuing a Relaxation from Its
Checkpoint`_ are described at the end of this page.

Deleting a Workspace
--------------------

An analysis made before every relaxation has finished goes stale once the full
set is available. Click **Delete Analysis...** in a workspace's header, or
right-click the workspace's node in the project tree and choose
**Delete Analysis...**, to remove it. MuFinder asks for confirmation first, with
**Cancel** as the default. If the deleted workspace was the one showing, the
Analysis page stays open.

Deleting a workspace discards only the workspace and its clustering. The
structural relaxations it used are kept, so they can be added to a new workspace
straight away. If the workspace's sites were pinned to the ``Muon Sites``
summary page (see `Muon Sites Summary`_), that pin is removed as well, and the
confirmation says so. A pin taken from another workspace is left alone. The
deletion is saved to the project immediately and cannot be undone.

Site Ordering
-------------

Within a workspace, muon sites are ordered by increasing total energy. The
energy column is reported relative to the lowest-energy member of the
workspace. The add-structure dialog also reports energies relative to the
current workspace minimum, so a negative value indicates that adding that
relaxation would reset the workspace zero. Final muon coordinates are folded
back into the host unit cell.

The **Show shifted positions** option applies a non-mutating symmetry anchoring
step for display. It chooses a representative site within each cluster and
shows symmetry-equivalent positions close to that representative where possible.
Changed coordinate cells are shown in blue. The saved relaxed structures are
not edited by this operation.

Wyckoff Assignment
------------------

Muon sites often relax close to high-symmetry interstitial positions. After
clustering, MuFinder assigns each cluster the highest-symmetry unoccupied
Wyckoff position of the host space group that is consistent with all members of
the cluster. The clustered-sites table shows this in the cluster header, for
example ``Cluster 1: (0.2500, 0.5000, 0.0000) 12c``.

The assignment is a tolerance-based fit rather than a claim that the relaxed
muon is exactly on the Wyckoff position. MuFinder first anchors the cluster
members in the same non-mutating way used for shifted positions, then fits
nearby special-position loci from the host symmetry operations. This includes
variable-coordinate special positions, not only fixed high-symmetry points.
Each candidate is labelled by generating its full symmetry orbit and asking
``spglib`` for the corresponding Wyckoff letter and multiplicity. Occupied host
positions are rejected by checking whether the candidate orbit overlaps any host
atom within the occupancy tolerance.

The candidate special positions come from projecting the cluster onto the
fixed-point locus of every symmetry operation and of every pair of operations,
which is several thousand pairs for a high-symmetry cubic host. To keep the
clustering dialog responsive, MuFinder skips pairs that cannot produce a
solution, screens the rest in one vectorised pass, and projects exactly only
the pairs that can land within range of the cluster. It then asks ``spglib``
to label candidates in rank order, so only the winning candidate is usually
labelled. None of this changes which site is chosen. Host symmetry operations
and per-cluster assignments are cached, so switching clustering methods reuses
the assignments of clusters that did not change.

The **Wyckoff tolerance (Å)** control in the clustering dialog sets the maximum
allowed distance from every anchored cluster member to the selected Wyckoff
orbit. If no unoccupied special position satisfies this all-members rule,
MuFinder reports the assignment as unavailable rather than hiding a cluster
outlier behind the representative site.

Clustering
----------

The **Clustering...** button opens the ``Clustering - <workspace>`` dialog,
which provides three methods, selected with ``Method``. Click
**Apply Clustering** to store the result on the workspace.

**Connected Components**
   MuFinder builds a graph in which each relaxed muon endpoint is a node. Two
   nodes are connected when their shortest symmetry-aware periodic distance is
   below the chosen ``Distance tolerance (Å)``. Clusters are the connected
   components of that graph. This follows the graph-style treatment described
   for relaxed sites in the MuFinder paper.

**Hierarchical**
   Hierarchical clustering uses the same Soprano and pymuon-suite inspired
   feature representation used by the UEP batch-search workflow. The free
   parameter is the dendrogram threshold, ``Hierarchical t``.

**K-means**
   K-means clustering also reuses the UEP batch-search feature representation.
   The free parameter is the requested number of clusters, ``Clusters k``.

For connected components, the dialog shows the graph of subclusters and a plot
of number of clusters against tolerance. Each manual tolerance evaluation adds a
point to the plot. **Scan Tolerances** evaluates a regular tolerance grid so a
stable plateau in the cluster-count curve can be chosen deliberately. The
symmetry-aware distances do not depend on the tolerance, so the scan measures
them once and reuses them at every grid point. Host symmetry operations are
cached per host, and each site's symmetry-equivalent positions are compared in
one vectorised pass. Near ties are re-measured exactly, so the distances,
cluster counts, and anchored positions match a site-by-site evaluation.

The connected-components graph sizes its nodes, their border rings, their
numbers and its edges in data units, so the graph scales with its canvas:
every node stays inside the plot and nodes never overlap, however small the
window. Node size follows the closest pair of nodes, and large components are
spread so neighbouring nodes keep a consistent gap. Components are packed into
rows chosen to fit the canvas's shape (a few long rows on a wide canvas, more
short rows on a tall one), kept in cluster order, with gaps tied to the node
size, and the packing is redone when the window is resized. Nodes stop growing
at about 22 pt in radius, so a small graph on a large window stays readable.
Components with more than 2.5 edges per node draw thinner, translucent edges,
so the edges of a dense component can still be told apart.

In the clustering window the structure viewer gets half the width, or more when
its cluster summary table needs it to show the Wyckoff assignments in full; the
plot takes the rest. The split follows window resizes until you drag the
handle.

The clustering summary table also reports the selected Wyckoff assignment for
each cluster. Tooltips include the detected space group, site-symmetry symbol,
and the maximum and mean fitted distances used to justify the assignment.

Displacement Analysis
---------------------

**Analyse Displacements** opens a site-level displacement view; clustering is
not required. Choose one relaxed muon site from the workspace (``Muon site``)
and a comparison host structure (``Comparison host``). The comparison structure may be the imported host or a completed
pristine relaxation, provided the pristine relaxation was not run as a free-cell
relaxation. Free-cell pristine relaxations are disabled in the selector because
cell-shape changes make ion-by-ion displacement interpretation ambiguous.

MuFinder creates the comparison supercell needed to match the relaxed cell,
matches host atoms with periodic boundary conditions, and removes the implanted
muon from the relaxed structure before atom matching. The structure view shows
the relaxed structure with the comparison host atoms as transparent reference
points. The plot reports radial ion displacement relative to the selected muon
site against ion distance from the muon, in the style of Fig. 5(b) in the
MuFinder paper. Positive radial displacement means the matched ion moved away
from the muon.

Muon Sites Summary
------------------

The **Muon Sites** page (the parent node above Generation, Relaxations, and
Analysis) is a summary of your headline result: the distinct, symmetry-
inequivalent candidate stopping sites for each charge state. It has one card for
**Positive muon** and one for **Muonium**, shown side by side.

To populate a card, open a clustered analysis workspace and choose **Pin to
Summary**. MuFinder copies the workspace's distinct clustered sites into the card
for the matching charge state, ranked by relative energy (lowest first), and
shows each site's relative energy, representative fractional position, and
Wyckoff label. A 3D preview renders the host structure with the pinned muon
sites. Pinning requires that clustering has been run, that the workspace was
built on the active host structure, and that its charge state is supported; the
button reports if there is nothing to summarise yet or the host does not match.
Use **Unpin** to clear a card. Pinned snapshots are saved in the project's UI
state under ``muon_sites_summary_snapshots``, keyed by charge state and
independent of the live workspaces. Each pinned site records the job id of its
cluster's representative relaxation.

Quantum Corrections, Energy Barriers and the Magnetic Fields muon-site dialog
list the pinned sites by default, labelled by Wyckoff label and cluster and
ranked by relative energy as on the summary. Each page has a **Show every
relaxed site** option that lists every eligible relaxed site instead; the
choice is saved per workflow under ``downstream_show_every_relaxed_site``.
Pinning or unpinning updates those pages straight away.

You can navigate from the summary straight into the live analysis:

* **Open in Analysis →** jumps to the workspace the card was pinned from. When a
  card is empty, this button instead reads **Go to Analysis →** and takes you to
  the Analysis tab so you can create and cluster a workspace.
* **Clicking a site row** in the ranked table, or **clicking a muon** in the 3D
  preview, opens the source workspace and reveals that specific cluster.

Because a pinned card is a snapshot, MuFinder validates it against the current
host structure and hides it if the host changes. If the source workspace or
cluster no longer exists when you follow a link, MuFinder opens the Analysis tab
and notes that the pinned analysis has changed.

Exporting to CIF
-----------------

Both the live analysis and the pinned summary can export the host structure
together with muon positions as a CIF file
(:func:`mufinder.science.structure_io.export_structure_with_muon_sites`).
Muons are written as hydrogen atoms with distinct ``Mu1``, ``Mu2``, ...
``_atom_site_label`` values. The file is written in P1 with every atom listed
explicitly, since muon sites generally break the host's symmetry.

The standard ``_atom_site`` loop carries no MuFinder-specific columns -
mixing one in was found to desynchronize the column count in some CIF
readers (observed with VESTA), silently shifting every atom's fields by one
and corrupting the whole structure. Per-site relative energy and Wyckoff
label instead live in their own standalone loop below the atom list, keyed by
``_mufinder_muon_site_label``, which a reader that does not recognise it can
skip entirely without touching atom parsing.

* In the **Analysis** tab, each workspace's **Export CIF...** button exports
  every relaxed muon position currently in that workspace - one CIF site per
  completed relaxation added to the workspace - regardless of whether
  clustering has been applied. Use this when you want the full set of raw
  relaxation endpoints rather than a clustered summary.
* On the **Muon Sites** summary page, each card's **Export CIF…** button
  exports only the pinned, distinct clustered sites (one representative per
  cluster), matching what the card's table and 3D preview show. It stays
  disabled until the card is populated.

If a muon site is written implausibly close to (or exactly coincident with) a
host atom - closer than 0.5 Å, well under any real muon-implantation bond
length - MuFinder still writes the file but reports which sites and host atoms
are affected. This file is not itself malformed (each atom keeps full,
independent occupancy), but many external CIF viewers detect the overlap and
render the pair as a single merged, fractionally-occupied site (for example
"Sr1/Mu1") rather than two distinct atoms. Seeing this warning, or a merged
site of this kind in another program, usually indicates that the underlying
analysis needs another look (e.g. the wrong host or workspace charge state, or
data pinned/added from a mismatched structure) rather than a genuine muon
site landing on a host ion.

Pristine Relaxation
-------------------

The optional ``Pristine Relaxation`` stage relaxes the host structure without a
muon. It is a table of jobs with explicit per-row labels:

- each row chooses its cell from a ``Supercell`` drop-down listing ``Unit
  cell``, the project's saved supercells, and ``Manage Supercells...``, which
  opens the :doc:`/user_guide/supercells`;
- each row has a ``Fix Cell`` control, written deterministically as CASTEP
  ``FIX_ALL_CELL``;
- each row's parameters are set in the shared CASTEP parameter dialog;
- generated jobs take the project default XC functional, cutoff energy and
  k-point grid. For a supercell row the grid is rescaled to keep the
  reciprocal-space sampling density of the reference lattice the grid was
  converged on, scaling each component by the change in the matching
  reciprocal axis and always rounding up so the cell is never undersampled.
  For example, a ``4x4x4`` default becomes ``2x2x2`` for diagonal ``2x2x2`` and
  ``2x2x3`` supercells;
- the table shows each row's status and, once relaxed, its ``a``, ``b`` and
  ``c`` lattice parameters.

On completion MuFinder retrieves the ``<seed>.castep`` and ``<seed>-out.cell``
outputs. A job whose CASTEP output reports incomplete geometry convergence is
marked failed. The relaxed geometry in ``-out.cell`` is parsed into lattice,
species and fractional coordinates, ``a``, ``b`` and ``c`` are stored in the
job's result, and each successful job adds one derived relaxed ``Structure`` to
the project. Those structures are what the :doc:`/user_guide/comparison_table` compares
with the experimental host, and they can be the comparison host in
`Displacement Analysis`_.

.. _continue-relaxation:

Continuing a Relaxation from Its Checkpoint
-------------------------------------------

Every relaxation ``.param`` MuFinder generates contains ``CONTINUATION :
DEFAULT``, so CASTEP resumes from a matching ``{seed}.check`` file whenever one
is in the run directory. The ``Pristine Relaxation`` and ``Muon Relaxation``
tables offer this as a per-row ``Continue relaxation…`` action for rows whose
job is ``completed``, ``failed``, ``stopped`` or ``cancelled``:

- The dialog opens with the row's current parameters in the shared CASTEP
  parameter editor, a live keyword-level diff of the old and new ``.param``
  text (added, changed and removed keywords), a checkpoint status line
  (checkpoint found, not found, or could not be verified; see
  :ref:`remote-checkpoint-probe`), a note that the ``.cell`` is regenerated
  from the definition's structure (this dialog does not edit atomic
  positions), and a summary naming the execution target and the directory that
  will be reused.
- Accepting it changes the row's parameters in place in the active definition.
  No new definition or job is created, so the row keeps its job. The job
  returns to ``defined`` with its ``remote_directory`` kept, and the ``.cell``
  and ``.param`` are rewritten into that same directory, locally or on the
  cluster.
- The job is resubmitted to its original execution target. A local job goes
  back through local submission. A cluster job goes back into its existing
  remote directory, with no existence prompt and nothing deleted, as long as
  the cluster selected in the submission dialog is the job's original cluster;
  choosing a different cluster uses the normal submission path, with a new
  target directory and an existence check.
- Each continuation is recorded as a ``job_continuation`` provenance entry.
- When the continued job completes again, its existing ``"{label} relaxed"``
  structure is refreshed in place, keeping its id, so comparison tables and
  other references stay valid, and a muon site's fields are refreshed from the
  same completion. A continued run that dies before writing a new
  ``-out.cell`` fails rather than reusing the previous one.

:ref:`job-continuation-primitive` in :doc:`/job_lifecycle` gives the technical
contract: the status transition, the cleared and kept fields, and why this
differs from the Quantum Corrections stage-1 reset.

The Job Dashboard's ``Resubmit`` also resumes from the checkpoint when you
choose ``Continue from the checkpoint`` (see
:ref:`resubmit-continue-or-start-again`). Neither route deletes the run
directory's contents. ``Reset Failed Jobs...`` with its optional clean-up does:
it deletes the local ``.check`` and ``.castep`` files and ``{seed}-out.cell``,
keeping only the ``.cell`` and ``.param`` inputs, although it never cleans a
folder another job still uses, such as the seed directory a Quantum
Corrections phonon job shares with its completed stage-1 singlepoint.

The regenerated ``.cell`` always holds the row's own structure: there is no
mode that writes the latest relaxed geometry into it, and MuFinder does not
use the CASTEP ``reuse`` keyword.

Sources
-------

This workflow acknowledges and follows the methods described in:

* B. M. Huddart et al., "MuFinder: A program to determine and analyse muon
  stopping sites", *Computer Physics Communications* **280**, 108488 (2022),
  DOI ``10.1016/j.cpc.2022.108488``. The connected-components clustering visualisation and
  radial displacement plot are based on the approaches shown in that work.
* L. Liborio, S. Sturniolo, and D. Jochym, "Computational prediction of muon
  stopping sites using ab initio random structure searching (AIRSS)",
  *J. Chem. Phys.* **148**, 134114 (2018), DOI ``10.1063/1.5024450``. The workspace follows the
  AIRSS-style practice of clustering relaxed muonated structures and selecting
  representative low-energy structures.
* The pymuon-suite project, especially the ``pm-muairss`` clustering workflow,
  which uses hierarchical and k-means clustering parameters
  ``clustering_hier_t`` and ``clustering_kmeans_k``. MuFinder reuses the same
  Soprano-inspired clustering interface conventions in this analysis view.
* The pymuon-suite ``pm-symmetry`` workflow, which uses symmetry analysis to
  report candidate high-symmetry Wyckoff points and whether they are occupied.
  MuFinder follows that idea for relaxed structural clusters, while fitting the
  nearest unoccupied Wyckoff orbit to all members of each cluster.
