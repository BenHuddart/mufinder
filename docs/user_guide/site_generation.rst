Generation
==========

``Muon Sites -> Generation`` is where you choose the starting positions for
muon structural relaxations, and ``Muon Sites -> Relaxations`` is where you
configure and run those relaxations. On the Generation page you create one row
per candidate position, by random sampling, by attaching a muon to a chosen
element, by typing coordinates, or by taking clustered UEP results. You then
stage the rows as CASTEP relaxation jobs. Every generated row enters the same
downstream relaxation workflow, whichever method created it.

Each generated site represents a separate relaxation input containing one
muon. When this page refers to a ``muon-muon`` distance, it means the spacing
between different generated starting positions in the host cell. It is a
diversity constraint used to avoid launching several relaxations from starts
that are likely to collapse to the same final site; it does not model multiple
muons in the same physical structure.

Before you start
----------------

Generation needs:

- **A host structure.** Load one first; positions are generated in the active
  host unit cell. Rows generated for a different host structure than the active
  one are dimmed and marked ``(stale)``; regenerate them or switch the active
  host back.
- **For UEP cluster sources, a clustered UEP batch search.** See
  :doc:`/user_guide/uep_workflow`.
- **For running the relaxations, a working CASTEP setup.** Configure a local
  CASTEP executable or a cluster under ``Setup``, and load a keyword catalog
  (:doc:`/castep_keyword_catalog`). Converged defaults from
  :doc:`/user_guide/convergence_testing` are inherited by the relaxation rows.

Running the workflow
--------------------

1. **Choose a method.** Open ``Muon Sites -> Generation`` and choose a
   ``Method`` in ``Generation Setup``: ``Random generation``, ``Attach muon to
   element``, ``Custom coordinates``, or ``UEP clusters``.
2. **Set its parameters and generate.** Fill in the method's fields (see
   `Generation methods`_) and click ``Generate Sites``. The progress bar under
   the buttons follows the generation, and the new rows appear in the table.
3. **Check the rows.** Select rows in the table to overlay their positions on
   the host structure in the ``Preview Muon Sites`` viewer.
4. **Optionally tidy the rows.** Click ``Cluster Positions...`` to give
   symmetry-equivalent starts one common representative, or ``Remove
   Selected`` to delete rows you do not want (see `Tidying generated rows`_).
5. **Stage relaxation jobs.** Select rows and click ``Create Jobs from
   Selected``. In ``Choose Charge State``, pick ``Positive muon`` (the
   default), ``Muonium`` or ``Both charge states`` (see `Creating relaxation
   jobs`_).
6. **Run the relaxations.** Open ``Muon Sites -> Relaxations``, check each
   row's parameters and run the rows (see `Structural relaxations`_). Job
   submission and the ``Job Dashboard`` are described in
   :doc:`/getting_started/running_jobs`.

The Generation page
-------------------

.. figure:: ../_generated/screenshots/site_generation_page.webp
   :alt: The Generation page with element-attached and random candidate sites
         selected and marked in the preview.

   F-attached and random candidate sites in CoF₂ (:doc:`/user_guide/examples/cof2`),
   selected in the table.

The page has the ``Generation Setup`` group on the left, with the ``Method``
drop-down and the fields for the chosen method, the buttons ``Generate Sites``,
``Cluster Positions...``, ``Remove Selected`` and ``Create Jobs from
Selected``, a progress bar, and the generated-site table. The ``Preview Muon
Sites`` viewer on the right shows the host structure with the selected
positions overlaid. The table has the columns ``Site``, ``Method``, ``State``
and the fractional coordinates ``fX``, ``fY`` and ``fZ``.

The table is ordered with natural numeric sorting, so ``Random Site 2`` appears
before ``Random Site 10``. Selected rows can be clustered around an anchor with
``Cluster Positions...``, staged with ``Create Jobs from Selected``, or removed
with ``Remove Selected`` or the ``Delete`` key (``Backspace`` on macOS) before
relaxation jobs are configured; only rows still marked ``generated`` are
deleted. Press ``Escape`` to clear the selection.

Random and element-attached generation run in the background. The progress bar
shows accepted-site progress while trial positions are being packed. In
number-of-positions mode its maximum is the requested number of sites. In
separation mode the maximum is a prediction of the likely number of accepted
sites that only scales the bar, so the label shows a running count such as
``Generated 6 sites`` rather than a target like ``Generated 6/13 sites``.
Generated rows are committed only after generation completes successfully. If
fewer sites are generated than requested, the Journal records an ``Attention``
entry suggesting more attempts, smaller distance limits, or a larger cell.

Generation stops with an explicit error, shown in the generation tab, when the
distance constraints cannot be satisfied or the requested element is not in the
host structure.

Generation methods
------------------

Random initial positions
~~~~~~~~~~~~~~~~~~~~~~~~

Choose ``Random generation``. MuFinder samples trial positions inside the
active host cell and accepts a trial position only when the proposed muon and,
by default, all of its symmetry-equivalent positions satisfy the configured
distance constraints. The algorithm is described under `Method`_.

The random setup sets ``Min muon-atom distance (Å)`` (default 1.0 Å) and
``Max attempts`` (default 20000), and offers two mutually exclusive modes
selected by radio buttons attached directly to the corresponding input fields.
The active row uses the MuFinder theme accent and enables only the field that
currently controls the calculation, so it is clear whether MuFinder is using
the entered separation or calculating one from the requested number of initial
positions.

Muon-muon separation mode
^^^^^^^^^^^^^^^^^^^^^^^^^

``Muon-muon separation (Å)`` is the default mode (default 0.8 Å). You choose
the minimum allowed distance between candidate starting positions directly.

1. Select the ``Muon-muon separation (Å)`` radio button and enter the
   separation.
2. Set ``Min muon-atom distance (Å)`` and ``Max attempts``.
3. Click ``Generate Sites``.

MuFinder keeps trying random positions until the attempt limit is reached,
accepting every position that satisfies:

- the minimum periodic muon-muon distance from previously accepted candidates;
- the minimum periodic muon-atom distance from all host atoms;
- the same muon-muon and muon-atom constraints for the full symmetry-equivalent
  orbit of the trial position when symmetry handling is enabled.

This mode is useful when the desired candidate-start density is known, when you
want to reproduce a known random-start setup, when you want to compare sampling
density between related cells, or when you want to explore how a fixed
diversity spacing changes the number of generated starts in different
supercells. Because the attempt limit is the stop criterion, the exact number
of accepted candidates is an output of the method rather than an input.

Number-of-positions mode
^^^^^^^^^^^^^^^^^^^^^^^^

``Number of initial positions`` mode (default 24) is useful when you want a
target number of structural relaxations, for example to match a computational
budget or to generate a comparable number of positions across several hosts.

1. Select the ``Number of initial positions`` radio button and enter the
   count.
2. Set ``Min muon-atom distance (Å)`` and ``Max attempts``.
3. Click ``Generate Sites``.

MuFinder estimates the muon-muon separation expected to produce that many
accepted random starts (the estimate is described under `Method`_). While
count mode is selected, changing the requested number of initial positions
updates the disabled ``Muon-muon separation (Å)`` field immediately as an
estimated readout. The generated positions then use the normal symmetry-aware
random placement algorithm with that spacing and the requested count as the
stopping criterion. If the estimated spacing is slightly too restrictive for
the finite attempt budget, MuFinder retries up to six times, reducing the
spacing by 10% each time, before reporting that the target count could not be
reached.

When count mode finishes, the ``Muon-muon separation (Å)`` field is updated to
the estimated spacing that was actually used for the generated sites, which
makes it easy to rerun the same setup in separation mode if desired.

Each generated site remembers whether it came from ``count`` or ``separation``
mode, the muon-muon spacing actually used, the requested target count (count
mode), the estimated spacing from the volume model, the muon-atom distance and
the symmetry-equivalent-position setting.

The estimated spacing is a heuristic, not a guarantee about the true topology of
the potential-energy surface. It depends on the free-volume approximation, the
symmetry multiplicity detected for the host, and the finite random attempt
budget. Final site assignment still requires structural relaxation followed by
clustering or comparison of the relaxed endpoints.

Element-attached positions
~~~~~~~~~~~~~~~~~~~~~~~~~~

Use this when physical intuition suggests that the muon or muonium-like state is
associated with a particular species.

1. Choose ``Attach muon to element`` as the ``Method``.
2. Pick the species in the ``Element`` drop-down, which lists the element
   symbols present in the active host structure.
3. Set ``Muons per atom`` (default 2), ``Muon-element distance (Å)`` (default
   1.0 Å) and ``Min muon-atom distance (Å)`` (default 1.0 Å).
4. Click ``Generate Sites``.

For each atom of the selected element, MuFinder places ``Muons per atom``
generated starts at the ``Muon-element distance (Å)`` along directions derived
from the local neighbour geometry. Directions are ranked by clearance from the
surrounding host structure and filtered by ``Min muon-atom distance (Å)``, and
the best ranked directions are retained for every matching atom in the active
host. Each is a separate one-muon relaxation start; this setup has no muon-muon
spacing field. The method is a deterministic seeding heuristic that gives
chemically sensible starts near the selected species. The generated positions
are still only initial guesses and must be validated by relaxation.

Manual positions
~~~~~~~~~~~~~~~~

1. Choose ``Custom coordinates`` as the ``Method``.
2. Choose the ``Coordinate system``, ``fractional`` or ``cartesian``.
3. Enter the three coordinates. Fractional entry uses the ``fX``, ``fY`` and
   ``fZ`` labels also used in the host-structure and generated-site tables;
   Cartesian entry switches the fields to ``X (Å)``, ``Y (Å)`` and ``Z (Å)``.
   The viewer previews the current position over the host structure before it
   is committed.
4. Click ``Generate Sites``. Each click adds one ``Manual Site`` row.

MuFinder keeps the position as fractional coordinates in the active host cell,
with Cartesian coordinates stored for display and downstream setup.

UEP cluster sources
~~~~~~~~~~~~~~~~~~~

UEP analysis does not promote sites directly from the UEP wizard. After
running a UEP batch search and clustering the optimized points (see
:doc:`/user_guide/uep_workflow`):

1. Choose ``UEP clusters`` as the ``Method``.
2. In the table, select the clustered rows to add; the viewer previews their
   positions over the host structure.
3. Click ``Generate Sites``.

The table lists one row per saved cluster representative, with the columns
``Source`` (the source job), ``Cluster``, ``Size``, ``Class`` (the
stationary-point classification), ``E (eV)`` (the classical energy) and ``fX``,
``fY``, ``fZ``.

Each row uses the optimized position of the cluster representative member, not
an averaged position or geometric centroid. The representative is the member
selected by the UEP clustering summary as the lowest-energy member of that
cluster, using total energy when ZPE-corrected total energies are available and
classical UEP energy otherwise. Individual non-representative cluster members
are retained in the UEP analysis state but are not shown in this version of the
generation table.

Generating stores the selected rows as normal generated sites for full
structural relaxation. MuFinder also keeps the source job, cluster identity,
representative member index, classification, energies, and clustering settings
with each site, and prevents the same UEP cluster row from being added twice.

Tidying generated rows
----------------------

Clustering initial positions around an anchor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After one or more generated rows exist, selected rows can be normalized with
``Cluster Positions...``. This action does not create or delete rows. Instead,
it rewrites the selected rows so that each selected muon is represented by the
symmetry-equivalent fractional coordinate closest to an anchor you choose.

In a symmetric host cell, two initial muon coordinates may be different
fractional representatives of the same physical site. If those representatives
are launched as separate geometry optimizations, they can relax to different
realizations of the same symmetry-equivalent final site, which is awkward when
comparing relaxed structures, assigning final-site labels, or preparing a
compact relaxation campaign. Clustering the initial positions around one anchor
makes the table use a consistent representative of each symmetry orbit before
jobs are staged.

1. Select one or more rows in the generated-site table.
2. Click ``Cluster Positions...``.
3. In the ``Cluster Positions`` dialog, choose ``Manual anchor`` and enter the
   fractional anchor (``fX``, ``fY``, ``fZ``), or choose one of the selected
   muon rows from the ``Anchor`` drop-down to use its current fractional
   position as the anchor.
4. Confirm with the dialog's ``Cluster Positions`` button.

For each selected row, MuFinder asks spglib for the symmetry operations of the
active host structure, applies those operations to the row's current fractional
position, wraps the resulting orbit into the unit cell, and picks the orbit
member with the shortest periodic distance to the anchor. The row's fractional
and Cartesian coordinates are updated in the generated table and a provenance
entry records the anchor and affected site labels.

This is most useful when random generation has produced several candidates from
related symmetry orbits, when UEP cluster representatives or manually entered
positions are physically equivalent but appear in different parts of the unit
cell, or when you want to compare subsequent relaxations without first
disentangling equivalent-image choices in the output structures.

``Cluster Positions...`` is only a coordinate-representation cleanup. It does
not cluster relaxed endpoints, merge generated rows, decide which final sites
are physically distinct, or replace post-relaxation analysis. If symmetry
detection is unavailable for the host structure, MuFinder falls back to the
input position as the only representative, so the row may remain unchanged.

Creating relaxation jobs
------------------------

Rows in the generated-site table can be selected and staged with ``Create Jobs
from Selected``. The ``Choose Charge State`` dialog asks which charge state to
create: ``Positive muon`` (the default), ``Muonium``, or ``Both charge
states``. Staging creates fixed-cell geometry-optimization rows in the matching
charge-state tab under ``Muon Sites -> Relaxations`` and changes the
corresponding site's status from ``generated`` to ``staged``. The initial
position is carried forward as fractional coordinates in the single host unit
cell. Choosing both charge states creates one row in each tab for every
selected initial position. Once relaxed, each charge state's relaxation reaches
Quantum Corrections, Energy Barriers and the ``Muon Sites`` summary on its own:
the site records the relaxation of the charge state it was first relaxed in,
even after that relaxation is reset, and the other charge state's relaxation is
read from its row in the ``Relaxations`` page.

.. figure:: ../_generated/screenshots/charge_state_dialog.webp
   :alt: The Choose Charge State dialog for ten selected initial positions,
         with Both charge states chosen.

   ``Choose Charge State`` for the ten CoF₂ starts, which the example stages
   in both charge states.

MuFinder writes CASTEP ``charge`` as ``+1`` for positive muon and ``0`` for
muonium, and keeps ``fix_all_cell`` enabled for these jobs. Job names and
output directories are separated by charge state, for example
``files/muon_relaxation/positivemuon/positivemuon_<label>`` and
``files/muon_relaxation/muonium/muonium_<label>``. If a selected submission
contains both charge states, MuFinder submits them as two batches so each group
gets an unambiguous runtime-configuration dialog and remote working directory.

Structural relaxations
----------------------

.. figure:: ../_generated/screenshots/muon_relaxations_page.webp
   :alt: The Structural Relaxations page, Positive muon tab, with ten completed
         relaxations in a 2x2x2 CoF2 supercell, their initial and final positions
         and energies.

   Positive-muon relaxations in CoF₂, from synthetic outputs (see
   :doc:`/user_guide/examples/cof2`).

The ``Muon Sites -> Relaxations`` page (titled ``Structural Relaxations``) is
where you configure and run the full CASTEP geometry optimizations, one muon or
muonium start per row. Rows are organized into ``Positive muon`` and
``Muonium`` tabs. The charge state is chosen when jobs are staged rather than
edited in the table, leaving more table width for labels and supercell names.
Rows can share a displayed site label when they live in different charge-state
tabs; selecting a positive-muon row does not also submit the matching muonium
row.

.. figure:: ../_generated/screenshots/muonium_relaxations_page.webp
   :alt: The Structural Relaxations page, Muonium tab, with the same ten
         CoF2 starts relaxed as muonium in the supercell.

   The ``Muonium`` tab of the same project, the same starts staged with
   ``Both charge states``.

Running the relaxations
~~~~~~~~~~~~~~~~~~~~~~~

1. **Open the tab for the charge state** you staged, ``Positive muon`` or
   ``Muonium``.
2. **Check the parameters.** Click ``Configure...`` in a row's ``Parameters``
   column for one row, or select rows and click ``Configure Selected
   Parameters...`` for several. The ``task``, ``charge``, ``fix_all_cell``, and
   ``write_cell_structure`` settings are locked by MuFinder: the tab always
   writes geometry-optimization jobs, keeps the host cell fixed, writes the
   relaxed ``-out.cell`` file needed for result ingestion, and derives the
   CASTEP total charge from the row's charge-state tab. Project-default cutoff
   and k-point settings are inherited unless a row has an explicit compatible
   override. Supercell rows automatically receive an equivalent-density
   ``KPOINT_MP_GRID`` suggestion when a project default grid is available.
   Muonium is open-shell, so muonium rows run spin-polarised by default and
   their dialog opens with ``Spin Polarized`` ticked; unticking it turns spin
   polarisation off for that row. Positive-muon rows default to unticked.
3. **Run the rows.** Submit them locally or on a cluster with the run buttons
   (:doc:`/getting_started/running_jobs`). A running relaxation shows its
   geometry-optimisation progress.
4. **Read the results.** The table shows the staged host-cell fractional
   coordinates under ``Initial`` and, once completed, the relaxed muon
   coordinates under ``Final`` and the energy under ``Total Energy (eV)``.
   Final coordinates are folded back into the single host unit cell when the
   CASTEP job was run in a supercell.
5. **Analyse the relaxed sites.** Select rows and click ``Create Analysis
   Workspace from Selection`` to compare them (see
   :doc:`/user_guide/structural_relaxation_analysis`).

What MuFinder writes
~~~~~~~~~~~~~~~~~~~~

Input files are written when you submit, not when rows are first staged. The
seed name is the charge prefix plus a filesystem-safe version of the row label,
so spaces and punctuation are normalized to underscores. For example,
``Random Site 1`` becomes ``positivemuon_Random_Site_1`` for a positive-muon
row and ``muonium_Random_Site_1`` for a muonium row. MuFinder rejects
configurations whose normalized seed names would collide within the same
charge-state directory.

Saving
~~~~~~

Staging generated sites and saving a relaxation configuration update the
project in memory but do not write project files immediately. Those staged rows
and configurations are written when you choose ``Save Project`` or accept the
close-project save prompt. Creating or deleting generated muon sites, and
creating the job records and input files when you submit, are saved
automatically.

Method
------

Random placement
~~~~~~~~~~~~~~~~

Random generation follows the symmetry-aware initial-position algorithm
described in the MuFinder CPC paper:

- B. M. Huddart *et al.*, ``MuFinder: A program to determine and analyse muon
  stopping sites``, *Computer Physics Communications* **280**, 108488 (2022),
  DOI: ``10.1016/j.cpc.2022.108488``.

Each trial is first checked on its own. Host atoms and accepted orbits are both
symmetry-invariant, so a trial position that fails its own distance checks
fails as an orbit too. MuFinder therefore builds the symmetry-equivalent orbit
only for the few trials that pass. The distance checks run as vectorised passes
against all host atoms and accepted positions. Comparisons that land within
rounding error of a distance limit are re-measured exactly. A seeded run
therefore accepts the same positions as a position-by-position check. This keeps
the default 20000-attempt budget practical even for high-symmetry supercells.

Estimating the separation from a count
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The estimate used by number-of-positions mode is based on the same volume model
as the older MuFinder GUI. For a trial muon-muon separation
:math:`d_{\mu\mu}`, MuFinder treats :math:`r_\mu = d_{\mu\mu}/2` as the radius
of the volume reserved by one candidate. It estimates the free cell volume
outside host-atom exclusion spheres with effective radius

.. math::

   r_{\mathrm{host}} = \max(d_{\mu\mathrm{atom}} - r_\mu, 0),

where :math:`d_{\mu\mathrm{atom}}` is the minimum muon-atom distance. The
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
  symmetry-equivalent positions. It is ``1`` if symmetry-equivalent random
  positions are not being considered or symmetry detection is unavailable;
- :math:`f_{\mathrm{fill}} = 0.350` is the empirical random-filling factor used
  by the previous MuFinder implementation. It accounts for the fact that
  sequential random filling with excluded volumes does not pack space
  perfectly.

MuFinder numerically inverts this relationship to find the spacing expected to
match the requested count.

Related methods
~~~~~~~~~~~~~~~

For broader context on interstitial-site generation and defect-style candidate
enumeration, see the ``pymatgen-analysis-defects`` interstitial-generator
documentation and the ``pymuon-suite`` AIRSS and UEP workflow material
referenced in :doc:`/user_guide/uep_workflow`.

Limitations
-----------

- Generated positions are starting points only; final site assignment requires
  structural relaxation followed by clustering or comparison of the relaxed
  endpoints.
- The random seed used by the generators is internal; the generation setup does
  not expose it.
- ``Cluster Positions...`` only changes which symmetry-equivalent coordinate a
  row uses; it does not merge rows or judge which final sites are distinct.

For developers
--------------

Generated-site metadata records ``selection_mode`` as ``count`` or
``separation`` alongside the spacing and distances listed above. The package
layers, shared helpers and controller contract used by the generation and
relaxation workflows are described in :doc:`/architecture`.
