Generation
==========

The ``Generation`` workflow (``Muon Sites -> Generation`` in the project
navigator) is the place where initial muon positions are committed as project
``MuonSite`` entries for later structural relaxation setup.  Generated rows
created here all enter the same downstream relaxation workflow, regardless of
whether they came from random sampling, element-attached heuristics, manual
coordinates, or clustered UEP analysis results.

A host structure must be loaded first; positions are generated in the active
host unit cell.  The typical order is:

1. choose a ``Method`` in ``Generation Setup``: ``Random generation``,
   ``Attach muon to element``, ``Custom coordinates``, or ``UEP clusters``;
2. set the method's parameters and click ``Generate Sites``;
3. optionally tidy the generated rows with ``Cluster Positions...`` or
   ``Remove Selected``;
4. select rows and click ``Create Jobs from Selected`` to stage relaxation
   jobs, then continue in ``Muon Sites -> Relaxations``.

Rows generated for a different host structure than the active one are dimmed
and marked ``(stale)``; regenerate them or switch the active host back.

Each generated site represents a separate relaxation input containing one
muon.  When this workflow refers to a ``muon-muon`` distance, it means the
spacing between different generated starting positions in the host cell.  This
is a diversity constraint used to avoid launching several relaxations from
starts that are likely to collapse to the same final site; it is not modelling
multiple muons in the same physical structure.

Random Initial Positions
------------------------

Random generation samples trial positions inside the active host cell and
accepts a trial position only when the proposed muon and, by default, all of
its symmetry-equivalent positions satisfy the configured distance constraints.
This follows the symmetry-aware initial-position algorithm described in the
MuFinder CPC paper:

- B. M. Huddart *et al.*, ``MuFinder: A program to determine and analyse muon
  stopping sites``, *Computer Physics Communications* **280**, 108488 (2022),
  DOI: ``10.1016/j.cpc.2022.108488``.

The random setup also sets ``Min muon-atom distance (Å)`` (default 1.0 Å) and
``Max attempts`` (default 20000), and offers two mutually exclusive modes
selected by radio buttons
attached directly to the corresponding input fields.  The active row uses the
MuFinder theme accent and enables only the field that currently controls the
calculation, so it is clear whether MuFinder is using the entered separation or
calculating one from the requested number of initial positions.

Muon-muon separation mode
~~~~~~~~~~~~~~~~~~~~~~~~~

``Muon-muon separation (Å)`` is the default mode (default 0.8 Å).  In this
mode the user chooses the minimum allowed distance between candidate starting
positions directly.
MuFinder then keeps trying random positions until the attempt limit is reached,
accepting every position that satisfies:

- the minimum periodic muon-muon distance from previously accepted candidates;
- the minimum periodic muon-atom distance from all host atoms;
- the same muon-muon and muon-atom constraints for the full symmetry-equivalent
  orbit of the trial position when symmetry handling is enabled.

This mode is useful when the desired candidate-start density is known, or when
the user wants to explore how a fixed diversity spacing changes the number of
generated starts in different supercells.  Because the attempt limit is the stop
criterion, the exact number of accepted candidates is an output of the method
rather than an input.

Each trial is first checked on its own. Host atoms and accepted orbits are
both symmetry-invariant, so a trial position that fails its own distance checks
fails as an orbit too. MuFinder therefore builds the symmetry-equivalent orbit
only for the few trials that pass. The distance checks run as vectorised passes
against all host atoms and accepted positions. Comparisons that land within
rounding error of a distance limit are re-measured exactly. A seeded run
therefore accepts the same positions as a position-by-position check. This keeps
the default 20000-attempt budget practical even for high-symmetry supercells.

Number-of-positions mode
~~~~~~~~~~~~~~~~~~~~~~~~

``Number of initial positions`` mode (default 24) is useful when the user
wants a target number of structural relaxations.  The user supplies the desired number of
initial positions, and MuFinder estimates the muon-muon separation expected to
produce that many accepted random starts.

The estimate is based on the same volume model used by the older MuFinder GUI.
For a trial muon-muon separation :math:`d_{\mu\mu}`, MuFinder treats
:math:`r_\mu = d_{\mu\mu}/2` as the radius of the volume reserved by one
candidate.  It estimates the free cell volume outside host-atom exclusion
spheres with effective radius

.. math::

   r_{\mathrm{host}} = \max(d_{\mu\mathrm{atom}} - r_\mu, 0),

where :math:`d_{\mu\mathrm{atom}}` is the minimum muon-atom distance.  The
expected number of random starts is then

.. math::

   N_{\mathrm{expected}} =
   f_{\mathrm{fill}}
   \frac{V_{\mathrm{free}}}
        {M_{\mathrm{sym}}\,(4\pi r_\mu^3 / 3)}.

Here:

- :math:`V_{\mathrm{free}}` is the estimated cell volume not excluded by host
  atoms;
- :math:`M_{\mathrm{sym}}` is the number of symmetry operations used to reserve
  symmetry-equivalent positions.  It is ``1`` if symmetry-equivalent random
  positions are not being considered or symmetry detection is unavailable;
- :math:`f_{\mathrm{fill}} = 0.350` is the empirical random-filling factor used
  by the previous MuFinder implementation.  It accounts for the fact that
  sequential random filling with excluded volumes does not pack space
  perfectly.

MuFinder numerically inverts this relationship to find the spacing expected to
match the requested count.  While count mode is selected, changing the requested
number of initial positions updates the disabled ``Muon-muon separation (Å)`` field
immediately as an estimated readout.  The generated positions then use the
normal symmetry-aware random placement algorithm with that spacing and the
requested count as the stopping criterion.  If the estimated spacing is
slightly too restrictive for the finite attempt budget, MuFinder retries up to
six times, reducing the spacing by 10% each time, before reporting that the
target count could not be reached.

When count mode finishes, the ``Muon-muon separation (Å)`` field is updated to the
estimated spacing that was actually used for the generated sites.  This
keeps the GUI state explicit and makes it easy to rerun the same setup in
separation mode if desired.

The generated-site metadata records:

- ``selection_mode`` as ``count`` or ``separation``;
- the muon-muon spacing actually used;
- the requested target count for count mode;
- the estimated spacing from the volume model;
- the muon-atom distance and symmetry-equivalent-position setting.

Practical Guidance
~~~~~~~~~~~~~~~~~~

Use separation mode when:

- comparing sampling density between related cells;
- reproducing a known random-start setup;
- intentionally generating as many starts as the constraints allow.

Use number-of-positions mode when:

- planning a fixed number of geometry optimizations;
- matching a computational budget;
- generating a comparable number of initial positions across several hosts.

The estimated spacing is a heuristic, not a guarantee about the true topology
of the potential-energy surface.  It depends on the free-volume approximation,
the symmetry multiplicity detected for the host, and the finite random attempt
budget.  Final site assignment still requires structural relaxation followed by
clustering or comparison of the relaxed endpoints.

Random and element-attached generation run on a background worker thread.
The generation tab shows accepted-site progress while trial positions are
being packed.  In
number-of-positions mode the progress maximum is the requested number of sites.
In separation mode MuFinder applies the same volume model in the forward
direction to predict the likely number of accepted sites; this prediction is
used only to scale the progress bar and is not written back to the setup
fields.  The label therefore shows only a running count such as
``Generated 6 sites`` rather than presenting the prediction as a target like
``Generated 6/13 sites``.
Generated rows are committed only after the worker completes successfully;
generation failures are reported in the same way as other blocked
site-generation requests.  If fewer sites are generated than requested, the
Journal records an ``Attention`` entry suggesting more attempts, smaller
distance limits, or a larger cell.

Creating Relaxation Jobs
------------------------

Rows in the generated-site table can be selected and staged with ``Create
Jobs from Selected``.  The ``Choose Charge State`` dialog asks which charge
state to create: ``Positive muon`` (the default), ``Muonium``, or
``Both charge states``.  Staging creates fixed-cell geometry-optimization rows
in the matching charge-state tab under ``Muon Sites -> Relaxations`` and
changes the corresponding ``MuonSite`` status from ``generated`` to
``staged``.  The initial position is carried forward as fractional coordinates
in the single host unit cell.  Choosing both charge states creates one row in
each tab for every selected initial position.

MuFinder writes CASTEP ``charge`` as ``+1`` for positive muon and ``0`` for
muonium, and keeps ``fix_all_cell`` enabled for these jobs.  Job names and
output directories are separated by charge state, for example
``files/muon_relaxation/positivemuon/positivemuon_<label>`` and
``files/muon_relaxation/muonium/muonium_<label>``.  If a selected submission
contains both charge states, MuFinder submits them as two batches so each group
gets an unambiguous runtime-configuration dialog and remote working directory.
The relaxation table shows the staged host-cell fractional coordinates under
``Initial`` and, once completed, the relaxed muon coordinates under ``Final``
and the energy under ``Total Energy (eV)``.  Final coordinates are folded back into the single host unit
cell when the CASTEP job was run in a supercell.

Structural Relaxations Tab
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Muon Sites -> Relaxations`` tab is the staging point for full CASTEP
geometry optimization runs that include one muon or muonium start per row.
Rows are organized into ``Positive muon`` and ``Muonium`` tabs.  The charge
state is chosen when jobs are staged rather than edited in the table, leaving
more table width for labels and supercell names.  Rows can share a displayed
site label when they live in different charge-state tabs; internally the
submission key is charge-qualified so selecting a positive-muon row does not
also submit the matching muonium row.

Per-row parameter dialogs expose the CASTEP controls that are safe for this
workflow.  The ``task``, ``charge``, ``fix_all_cell``, and
``write_cell_structure`` settings are locked by MuFinder: the tab always writes
geometry-optimization jobs, keeps the host cell fixed, writes the relaxed
``-out.cell`` file needed for result ingestion, and derives the CASTEP total
charge from the row's charge-state tab.  Project-default cutoff and k-point
settings are inherited unless a row has an explicit compatible override.
Supercell rows automatically receive an equivalent-density
``KPOINT_MP_GRID`` suggestion when a project default grid is available.
Use ``Configure...`` in a row's ``Parameters`` column for one row, or
``Configure Selected Parameters...`` for several.
Muonium is open-shell, so muonium rows run spin-polarised by default and their
dialog opens with ``Spin Polarized`` ticked; unticking it turns spin
polarisation off for that row. Positive-muon rows default to unticked.

Input files are materialized on submission, not when rows are first staged.
The generated seed name is the charge prefix plus a filesystem-safe version of
the row label, so spaces and punctuation are normalized to underscores.  For
example, ``Random Site 1`` becomes ``positivemuon_Random_Site_1`` for a
positive-muon row and ``muonium_Random_Site_1`` for a muonium row.  MuFinder
rejects configurations whose normalized seed names would collide within the
same charge-state directory.

The tab follows the project save boundary used elsewhere in MuFinder.  Staging
generated sites and saving a relaxation configuration update the in-memory
project workflow but do not write project files immediately.  Those staged rows
and definitions are written when the user chooses ``Save Project`` or accepts
the close-project save prompt.  Creating or deleting generated muon-site
entities, and materializing actual job records and input-file mappings during
submission, are project-file changes and are autosaved.

Element-Attached Positions
--------------------------

Choose ``Attach muon to element``.  Element-attached generation detects the
element symbols present in the active host structure and exposes them in the
``Element`` dropdown.  For each atom of the selected
element, MuFinder places the requested number of generated muon starts at the
configured ``Muon-element distance (Å)`` (default 1.0 Å) along directions derived from the local neighbour
geometry.  Directions are ranked by clearance from the surrounding
host structure and filtered by ``Min muon-atom distance (Å)``.

Element-attached sites are also separate one-muon relaxation starts.  The
simplified element-attached setup does not expose a muon-muon spacing field;
instead, ``Muons per atom`` (default 2) chooses how many of the best ranked directions are
retained for every matching atom in the active host.  If a future version adds
an additional spacing filter here, it should be understood as duplicate-start
suppression for nearby generated positions, not as a constraint between muons
that coexist in one structure.

This is intended as a chemically guided starting-position heuristic.  It is
especially useful when physical intuition suggests that the muon or muonium-like
state may be associated with a particular species.  The generated positions are
still only initial guesses and must be validated by relaxation.

Manual Positions
----------------

Choose ``Custom coordinates``.  Manual generation accepts either fractional
or Cartesian coordinates, selected with ``Coordinate system``; each click of
``Generate Sites`` adds one ``Manual Site`` row.  The
stored canonical representation is fractional coordinates in the active host
cell, with Cartesian coordinates stored for display and downstream setup.
Manual fractional fields use the same ``fX``, ``fY``, and ``fZ`` labels as the
host-structure and generated-site tables.  When Cartesian entry is selected, the
fields switch to ``X (Å)``, ``Y (Å)``, and ``Z (Å)``.

The generation widget previews the current manual position over the host
structure before it is committed.

UEP Cluster Sources
-------------------

UEP analysis no longer promotes sites directly from the UEP wizard.  Instead,
after running a UEP batch search and clustering the optimized points, users
return to ``Generation`` and choose the ``UEP clusters`` method.

The UEP cluster table lists one row per saved cluster representative, including:

- source job;
- cluster index and cluster size;
- stationary-point classification;
- classical energy, reported as ``E (eV)``;
- fractional coordinates.

Each row uses the optimized position of the cluster representative member, not
an averaged position or geometric centroid.  The representative is the member
selected by the UEP clustering summary as the lowest-energy member of that
cluster, using total energy when ZPE-corrected total energies are available and
classical UEP energy otherwise.  Individual non-representative cluster members
are retained in the UEP analysis state but are not shown in this version of the
generation table.

Selecting rows previews their positions over the host structure.  Clicking
``Generate Sites`` stores those rows as normal generated ``MuonSite``
objects for full structural relaxation.  MuFinder also stores source metadata
such as source job, cluster identity, representative member index,
classification, energies, and clustering settings, and prevents the same UEP
cluster row from being added twice.

Clustering Initial Positions Around an Anchor
---------------------------------------------

After one or more generated rows exist, selected rows can be normalized with
``Cluster Positions...``.  This action does not create or delete rows.  Instead,
it rewrites the selected rows so that each selected muon is represented by the
symmetry-equivalent fractional coordinate closest to a user-chosen anchor.

The motivation is to keep symmetry-equivalent starts together before full
structural relaxation.  In a symmetric host cell, two initial muon coordinates
may be different fractional representatives of the same physical site.  If
those representatives are launched as separate geometry optimizations, they can
relax to different realizations of the same symmetry-equivalent final site.
That is often awkward when comparing relaxed structures, assigning final-site
labels, or preparing a compact relaxation campaign.  Clustering the initial
positions around one anchor makes the generated table use a consistent
representative of each symmetry orbit before jobs are staged.

To use this tool:

- select one or more rows in the generated-site table;
- click ``Cluster Positions...``;
- enter a fractional anchor (``Manual anchor``), or choose one of the selected
  muon rows from the dialog dropdown to use its current fractional position as
  the anchor;
- confirm with the dialog's ``Cluster Positions`` button.

For each selected row, MuFinder asks spglib for the symmetry operations of the
active host structure, applies those operations to the row's current fractional
position, wraps the resulting orbit into the unit cell, and picks the orbit
member with the shortest periodic distance to the anchor.  The row's
fractional and Cartesian coordinates are updated in the generated table and a
provenance entry records the anchor and affected site labels.

This operation is most useful when:

- random generation has produced several candidates from related symmetry
  orbits and the user wants a common representative before staging jobs;
- UEP cluster representatives or manually entered positions are physically
  equivalent but appear in different parts of the unit cell;
- the user wants to compare subsequent relaxations without first disentangling
  equivalent-image choices in the output structures.

``Cluster Positions...`` is only a coordinate-representation cleanup.  It does
not cluster relaxed endpoints, merge generated rows, decide which final sites
are physically distinct, or replace post-relaxation analysis.  If symmetry
detection is unavailable for the host structure, MuFinder falls back to the
input position as the only representative, so the row may remain unchanged.

Generated Table Behavior
------------------------

The committed generated-site table is ordered with natural numeric sorting, so
``Random Site 2`` appears before ``Random Site 10``.  Selecting rows overlays
the selected generated positions in the host viewer.  Selected generated rows
can be clustered around an anchor with ``Cluster Positions...``, staged with
``Create Jobs from Selected``, or removed with ``Remove Selected`` or the
keyboard Delete key (Backspace on macOS) before relaxation jobs are
configured; only rows still marked as ``generated`` are deleted.  Press
Escape to clear the selection.  Older project files that used the legacy
``candidate`` status are normalized to ``generated`` when loaded.
