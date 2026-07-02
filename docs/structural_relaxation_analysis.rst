Structural Relaxation Analysis
==============================

The structural relaxation analysis workspace collects completed muon geometry
optimisation results and treats their final muon positions as an analysis set.
Create a workspace from the Relaxations tab with **Create analysis workspace
from selection**. MuFinder asks for a workspace name, records whether the
workspace is for positive muon or muonium calculations, and only allows later
additions from the same charge-state family.

Workspaces are stored in the project file under the saved UI state. They are
therefore restored after a manual project save and load, but they are not
autosaved while the application is running.

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

The **Wyckoff tolerance (Å)** control in the clustering dialog sets the maximum
allowed distance from every anchored cluster member to the selected Wyckoff
orbit. If no unoccupied special position satisfies this all-members rule,
MuFinder reports the assignment as unavailable rather than hiding a cluster
outlier behind the representative site.

Clustering
----------

The **Clustering** dialog provides three methods.

**Connected Components**
   MuFinder builds a graph in which each relaxed muon endpoint is a node. Two
   nodes are connected when their shortest symmetry-aware periodic distance is
   below the chosen distance tolerance in Å. Clusters are the connected
   components of that graph. This follows the graph-style treatment described
   for relaxed sites in the MuFinder paper.

**Hierarchical**
   Hierarchical clustering uses the same Soprano and pymuon-suite inspired
   feature representation used by the UEP batch-search workflow. The free
   parameter is the dendrogram threshold.

**K-means**
   K-means clustering also reuses the UEP batch-search feature representation.
   The free parameter is the requested number of clusters.

For connected components, the dialog shows the graph of subclusters and a plot
of number of clusters against tolerance. Each manual tolerance evaluation adds a
point to the plot. **Scan Tolerances** evaluates a regular tolerance grid so a
stable plateau in the cluster-count curve can be chosen deliberately.

The clustering summary table also reports the selected Wyckoff assignment for
each cluster. Tooltips include the detected space group, site-symmetry symbol,
and the maximum and mean fitted distances used to justify the assignment.

Displacement Analysis
---------------------

**Analyse displacements** opens a site-level displacement view after clustering
has been applied. Choose one clustered muon site and a comparison host
structure. The comparison structure may be the imported host or a completed
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
summary**. MuFinder copies the workspace's distinct clustered sites into the card
for the matching charge state, ranked by relative energy (lowest first), and
shows each site's relative energy, representative fractional position, and
Wyckoff label. A 3D preview renders the host structure with the pinned muon
sites. Pinning requires that clustering has been run; the button reports if there
is nothing to summarise yet. Use **Unpin** to clear a card.

You can navigate from the summary straight into the live analysis:

* **Open in Analysis** jumps to the workspace the card was pinned from. When a
  card is empty, this button instead reads **Go to Analysis** and takes you to
  the Analysis tab so you can create and cluster a workspace.
* **Clicking a site row** in the ranked table, or **clicking a muon** in the 3D
  preview, opens the source workspace and reveals that specific cluster.

Because a pinned card is a snapshot, MuFinder validates it against the current
host structure and hides it if the host changes. If the source workspace or
cluster no longer exists when you follow a link, MuFinder opens the Analysis tab
and notes that the pinned analysis has changed.

Sources
-------

This workflow acknowledges and follows the methods described in:

* B. M. Huddart et al., "MuFinder: A program to determine and analyse muon
  stopping sites", 2022. The connected-components clustering visualisation and
  radial displacement plot are based on the approaches shown in that work.
* L. Liborio et al., "Computational prediction of muon stopping sites using ab
  initio random structure searching (AIRSS)", 2018. The workspace follows the
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
