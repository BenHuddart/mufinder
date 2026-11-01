Energy Barriers
===============

Energy Barriers guides the user from two completed relaxed muon endpoints to a
climbing-image nudged elastic band (NEB) calculation seeded directly from those
endpoints. The workflow is intended for comparing two already-relaxed MuFinder
muon sites that belong to the same host and charge state.

Scientific intent
-----------------

This workflow answers a different question from site generation or relaxation:
not where the muon can stop, but how it moves between two already-established
local minima. MuFinder therefore treats Energy Barriers as an endpoint-driven
workflow. You begin from two completed relaxed sites, define a specific ordered
hop between them, and then refine the corresponding migration pathway with a
climbing-image NEB calculation built directly from that endpoint pair. No
separate transition-state search step is required.

Before a band has run, MuFinder shows the hop in the matching undistorted
pristine supercell, so the start and end muon positions are not mixed with host
distortions from two separately relaxed defect cells.

The barrier is classical: it holds no zero-point energy. The muon's zero-point
energy comes from :doc:`/user_guide/quantum_corrections`.

Before you start
----------------

Energy Barriers needs:

- **Two completed relaxed muon sites** of the same host and charge state. Run
  the structural relaxations under ``Muon Sites -> Relaxations`` first
  (:doc:`/user_guide/site_generation`), then cluster and pin them in
  :doc:`/user_guide/structural_relaxation_analysis`. The endpoint selectors
  list the distinct sites pinned to the ``Muon Sites`` summary (see `The
  Energy Barriers page`_). To use the same site twice, select it as both
  endpoints.
- **Project defaults, ideally.** The NEB rows start from the project's
  convergence defaults (:doc:`/user_guide/convergence_testing`) and are then
  overridden by the settings of the reactant endpoint's relaxation (see
  `Configuration UI`_).
- **A working CASTEP setup.** Configure a local CASTEP executable or a cluster
  under ``Setup``, and load a keyword catalog (:doc:`/castep_keyword_catalog`).

Running the workflow
--------------------

1. **Choose the endpoints.** Under the page title, pick the initial and final
   relaxed sites in the two ``Endpoints`` selectors.
2. **Cap the path length (optional).** Set ``Max length`` to limit the hop
   length for this generation pass, in Å. Leave it blank, shown as ``Auto``, to
   use MuFinder's automatic cutoff.
3. **Generate the paths.** Click ``Generate Paths`` to enumerate
   symmetry-distinct muon paths for that ordered pair only, including
   symmetry-equivalent hops of the same relaxed site when the same endpoint is
   selected twice. Each path becomes a row of the ``Paths`` table.
4. **Look at a hop (optional).** Select a row: ``Selected path`` below the table
   shows the hop.
5. **Remove paths you do not need.** Use ``Remove Selected``, the row menu's
   ``Remove`` entry, or the ``Delete`` / ``Backspace`` shortcuts.
6. **Configure the NEB jobs.** Select the paths and click ``Configure…`` (see
   `Configuration UI`_).
7. **Run them.** Click ``Run All Locally`` or ``Run All on Cluster``; the
   buttons read ``Run Selected ...`` when rows are selected. MuFinder builds
   and submits the NEB jobs (see :doc:`/getting_started/running_jobs`).
8. **Read the results.** When a job completes, its barriers fill its row, and
   selecting the row shows its band (see `Outputs and interpretation`_).

.. figure:: ../_generated/screenshots/gdru2si2_energy_barrier_paths.webp
   :alt: The Energy Barriers page with GdRu2Si2 site 1 and site 2 as endpoints.
         The paths table lists two site 1 to site 1 hops, along c and along
         [1 -1 0], ready, and the site 1 to site 2 hop, completed with 4 farms
         and barriers of 495, 213 and +283 meV. The hop along c is selected,
         and the pristine supercell below shows its arrow.

   The paths of the GdRu₂Si₂ worked example (:doc:`/user_guide/examples/gdru2si2`),
   with one hop that has not run selected.

The Energy Barriers page
------------------------

The whole workflow is on one page, from top to bottom:

``Endpoints``
  The initial and final relaxed sites, ``Max length`` and ``Generate Paths``.

``Paths``
  One row per hop, with its status and, once it has run, its barriers.

``Selected path``
  What the row selected in the table holds: the hop, or its finished band. With
  several rows selected it shows the first.

The endpoint selectors list the distinct sites pinned to the ``Muon Sites``
summary (:doc:`/user_guide/structural_relaxation_analysis`), in the summary's
energy rank, so each physical site is offered once. A pinned site is named by its
Wyckoff label and number (``16m · site 2``, the site the summary lists as
cluster 2), and any other by its own label. The energy above the lowest-energy
relaxed site with the same charge state and cell follows in meV
(``16m · site 2 (+257 meV)``); the lowest site shows none. The charge state appears
only when the list holds more than one. Tick ``Show every relaxed site`` to offer
every eligible relaxed site instead; the choice is saved with the project. When no
charge state has a pin the selectors are empty and a notice with ``Go to Muon
Sites →`` explains how to pin sites; the notice also names a charge state with no
pin and counts pinned sites that no longer resolve to an eligible relaxed site. A
site relaxed in both charge states is offered once in each, and its paths in one
charge state are kept apart from those in the other. A stored endpoint the
selectors no longer list is replaced by the first listed site (the final endpoint
by the first other one), so widening the list keeps the pair the selectors showed.
Paths already generated stay in the ``Paths`` table whichever list the selectors
show.

Path generation and assumptions
-------------------------------

MuFinder uses the pristine host symmetry and the two relaxed endpoint
structures to enumerate distinct muon paths. The current implementation:

- requires the two endpoint structures to share the same simulation cell;
- requires the same atom count and species ordering in both endpoints;
- requires exactly one muon in each endpoint structure;
- constructs a pristine supercell matching the relaxed endpoint cell;
- enumerates symmetry-equivalent muon start and end positions in that supercell;
- measures each candidate hop from a start position to the nearest periodic
  image of an end position, which is the shortest hop in hexagonal and other
  skewed cells too (the NEB intermediate image and the endpoint alignment use
  the same nearest image);
- optionally filters candidates by a user-specified maximum path length;
- discards zero-length self-maps before returning any same-site or cross-site
  path candidates;
- groups the candidate hops by full symmetry operations and keeps one hop per
  group.

Two hops are equivalent only when one symmetry operation maps the start of one
onto the start of the other and its end onto the other's end. When both
endpoints are the same relaxed site, a hop and its reverse are also
equivalent. Equal length alone is not enough: from the 4c site of CoF₂ two
families of 3.32 Å hops pass the nearest fluorine at very different
distances, and they are generated as two paths. Every hop in a group has the
same length.

The symmetry respects the host's collinear magnetic order. When the host
carries per-atom spins (``atomic_spins``) and no non-collinear moments, the
operations used are the crystal operations in the endpoint supercell that keep
the spin pattern or flip every spin together. A collinear spin-polarised
calculation without spin-orbit coupling gives the same energy for a pattern and
its global flip, so such an operation relates hops with the same barrier. Two
spins count as equal when they differ by at most 5 % of the largest spin, so
spins read from calculated populations (2.97 and −2.96) still form one
pattern. The
same operations give the symmetry-equivalent muon positions and align the
endpoints onto the chosen path when the NEB inputs are built. For a
two-sublattice antiferromagnet such as CoF₂ every crystal operation keeps or
flips the pattern, so the paths are those of the crystal. Where some operations
map the spins onto neither pattern (a ferrimagnet with unequal moments, or a
stripe order), hops that the crystal relates can be distinct, and positions
that the crystal relates but the magnetic order does not are not endpoints of
this relaxed site: relax such a site separately and choose it as the other
endpoint. Building the NEB inputs for a stored path whose endpoint is related to
the relaxed site only by such an operation stops with a message saying so.
Without spins, or with non-collinear moments, paths use the crystal
symmetry.

Below ``Generate Paths`` the page states which symmetry the paths from the
selected ``Initial`` endpoint use: "Paths use the crystal symmetry." or "Paths
use the magnetic order's symmetry, which is lower than the crystal symmetry."

Generating a pair again replaces its rows, with two exceptions that keep NEB
jobs and settings with the hop they were made for. A generated path
symmetry-equivalent to a stored one keeps that row's number and key; a stored
path no longer generated (for example after a shorter ``Max path length``, or
after the grouping changed) stays when it has an NEB job or saved settings.
Newly distinct paths take the next path numbers after those kept.

If no valid symmetry-distinct path can be constructed, the workflow does not
generate a row.

The ``Paths`` table has these columns:

- ``Hop``, the two sites named as in the endpoint selectors
  (``8g · site 1 → 16m · site 2``), with the charge state when the page holds more
  than one. Rows that would share a name also give the hop's direction in the
  host's axes when it tells them apart (``along c``, ``along [1 -1 0]``, with the
  components scaled to a largest value of 1 when they are no simple ratio);
- ``Length``;
- ``Farms``, the configured task-farm count (``Off`` when farming is disabled);
- ``Status``, the NEB job's status, in the colours of every workflow table;
- ``Forward``, ``Reverse`` and ``ΔE``, see `Outputs and interpretation`_. They
  read ``–`` until the band has finished.

Right-clicking selected rows offers the shared row actions (``Run Local``, ``Run on
Cluster``, ``Cancel``, ``Inspect Failure``, ``Reset Failed Jobs…``, ``Configure…``,
``Remove``) and ``Open Containing Folder`` for a row with a job, which holds the
CASTEP files; the ``.ts`` trajectory path is the tooltip of a finished row.

Generated path rows remain in the table after the endpoint selectors change, so
hops generated earlier stay available for inspection and submission while the user
prepares another pair.

A hop before its band
---------------------

Selecting a row that has no finished band shows, under ``Selected path``, the hop's
endpoints, length and charge state, and the undistorted pristine supercell that
matches the relaxed endpoint cell. Markers show the initial and final muon positions
and an arrow joins them, from the start to the nearest periodic image of the end, so
the migration direction is clear without relaxed host distortions. Atoms can hide
the arrow; ``Along:`` and dragging turn the view.

What MuFinder writes
--------------------

Each NEB job runs in its own folder under the project's
``files/energy_barriers_neb`` folder. For NEB jobs, MuFinder prepares aligned reactant and product structures for
the selected path and writes CASTEP inputs with
``TASK : TRANSITIONSTATESEARCH`` and ``TSSEARCH_METHOD : NEB`` locked by the
workflow. CASTEP additionally requires an intermediate image between the two
endpoints; MuFinder synthesises it with IDPP interpolation (falling back to a
minimum-image linear midpoint) and writes it into the generated ``.cell``
file alongside the product image.

Configuration UI
----------------

The heading of the configuration dialog names the hop as the endpoint selectors
do, with its length (``8g · site 1 → 16m · site 2 · 1.72 Å``); with
several paths selected it gives their number. Its rows start with the settings
shared by every workflow, then the three a band usually needs:

- the precision and cutoff controls, the electronic energy tolerance and the
  Monkhorst-Pack grid;
- ``Images`` (``TSSEARCH_MAX_PATH_POINTS``, default 11). It counts the
  intermediate images only, so the finished band holds two more: the reactant
  and the product. An even value is rounded up to the next odd number, because
  climbing image is locked on and CASTEP would otherwise add the image itself;
- ``Force tolerance (eV/Å)`` (``TSSEARCH_FORCE_TOL``), with CASTEP's own
  0.05 eV/Å default pre-filled;
- ``Task farms``, the CASTEP task-farm count (see `Task farming`_).

The exchange-correlation functional and the spin treatment follow, and then the
rows the workflow locks: ``TS Search Method`` (``NEB``) and ``Climbing image``
(``TRUE``), with ``TASK`` above the table.

The other NEB controls are behind the searchable ``Add Parameter`` flow, as is
the rest of the CASTEP surface: ``Spring constant (eV/Å²)``, ``Tangent``,
``NEB method``, ``Max iterations`` and ``Normed``. CASTEP uses its own default
for any of them that is not added, and a value set on a path stays in that
path's dialog as an added row.

.. figure:: ../_generated/screenshots/gdru2si2_neb_configuration.webp
   :alt: The Configure NEB Parameters dialog for the GdRu2Si2 site 1 to site 2
         path: 1000 eV, a 3 x 3 x 3 grid, 41 images, a force tolerance of
         0.08 eV/A, 4 task farms, PBE, spin polarised, and the locked TS search
         method and climbing image.

   The NEB settings of the GdRu₂Si₂ band. The 40 images entered are rounded up
   to 41, the count CASTEP runs with climbing image on.

Default values are layered. The project's convergence defaults
(``CUT_OFF_ENERGY`` and ``KPOINT_MP_GRID``) apply first; when the selected
path uses a supercell-sized relaxed endpoint, MuFinder remaps the inherited
k-point grid to preserve equivalent reciprocal-space sampling density relative
to the project's recommended reference lattice. The settings of the reactant
endpoint's relaxation then override them where its output ``.cell`` and
``.param`` are available: its k-point grid, cut-off energy, exchange-correlation
functional, and spin treatment (read from the source files as for
:doc:`/user_guide/quantum_corrections`), so the band is computed on the same numerical
footing as the structures it connects. A spin-polarised band writes the
reactant relaxation's per-atom ``SPIN=`` values on every image. Row-level configuration remains
authoritative, so a specific path can still override any of these without
affecting the rest of the project.

Task farming
------------

A NEB band is embarrassingly parallel across its images: each image's SCF is
independent within a sweep. CASTEP exposes this as *task farming*, enabled with
the ``NUM_FARMS`` parameter, which splits the MPI ranks into that many
independent farms and gives each a contiguous block of images. A sweep then
costs ``ceil(images / farms)`` SCFs instead of ``images``.

Task farming is configured **per path**, in the ``Task farms`` control of the
NEB configuration dialog, and defaults to 1 (disabled). It is a per-path
setting rather than a submission-time one because ``NUM_FARMS`` is written into
the job's ``.param`` when the job directory is generated, before any runtime is
chosen. Only NEB supports it: CASTEP resets ``NUM_FARMS`` to 1 for any other
``TSSEARCH_METHOD``.

Two rules govern the choice, and they differ in severity:

- **Farms must not exceed MPI ranks.** CASTEP stops immediately when they do.
  MuFinder therefore refuses to submit such a job, in both the local and cluster
  submission dialogs.
- **Ranks should divide evenly by farms.** Each farm receives
  ``floor(ranks / farms)`` ranks, and the remainder is assigned to no farm and
  idles for the whole run. MuFinder warns but allows this.

The relationship to the image count is an efficiency question only. Because
climbing-image NEB requires an odd ``TSSEARCH_MAX_PATH_POINTS``, a perfectly
balanced split needs an odd farm count that divides it. At MuFinder's default of
11 images, 2, 3, 4 or 6 farms all reach 92% balance (11 of the 12 farm-image
slots per sweep are busy), while 5 farms wastes over a quarter of the
allocation. The
configuration dialog shows the resulting split and balance live, so a poor
pairing is visible before submission. The image count itself is left entirely to
the user — it is a physical resolution choice, not a scheduling one.

.. figure:: ../_generated/screenshots/gdru2si2_neb_task_farms.webp
   :alt: The Task farms row of the NEB dialog set to 4, reading 41 images over
         4 farms, 11/10/10/10 per farm, 93% balance, and that the job needs at
         least 4 MPI ranks, ideally a multiple of 4.

   The farm split of the GdRu₂Si₂ band: 41 images over 4 farms.

The farm count can only be changed while a path has no NEB job yet, or while
its job is still in the ``defined`` state — the same rule that governs every
other NEB parameter. This matters more for farming than for the rest: the
configured count also tells MuFinder which output filenames to read, so it must
not drift away from the ``NUM_FARMS`` already written into a submitted job's
``.param``. To change it after a job has run, delete the job and rebuild it.

Outputs and interpretation
--------------------------

.. figure:: ../_generated/screenshots/gdru2si2_neb_analysis.webp
   :alt: The Energy Barriers page with the completed GdRu2Si2 site 1 to site 2
         path selected: a forward barrier of 495 meV, a reverse barrier of
         213 meV and a Delta E of +283 meV in the table, and below it the
         profile and the transition-state image 23 of 42 in the viewer.

   The real CASTEP band from GdRu₂Si₂ site 1 to site 2 (Huddart *et al.*
   2025), with its transition state in the viewer.

MuFinder keeps the generated CASTEP files of completed NEB jobs and the parsed
transition-state path data needed for later review. A finished path's row gives, in
whole meV:

- ``Forward``, the peak energy minus the initial image energy;
- ``Reverse``, the peak energy minus the final image energy;
- ``ΔE``, the final image energy minus the initial image energy, with its sign.

The barriers are classical (see `Scientific intent`_). They are taken from the
parsed CASTEP transition-state output. ``Reverse`` and ``ΔE`` are derived from the
stored per-image energies; a result without stored per-image energies shows only
``Forward``. A finished band whose path has gone, or whose endpoints are no longer
eligible, keeps a row of its own, named by its sites or its job.

Selecting a finished row shows its band under ``Selected path``, headed by the hop,
its length and the number of images. On the left is the reaction-coordinate profile
with labelled axes and a dashed annotation marking the forward barrier at the peak
image. Energies are plotted relative to the initial image (``E − E₀``) so the axis
stays readable despite the large absolute CASTEP energies. The plot is the primary
scrub surface: click or drag to select the nearest image, use the Left/Right arrow
keys to step, or use the ``◀ Image n / N ▶`` stepper under the plot. The readout
beside the heading gives the selected image's reaction coordinate and energy. The
structure viewer follows the selected image and preserves the current camera
orientation and zoom while scrubbing along the band; the view only resets when
another path is selected. CASTEP writes the ``.ts`` file in atomic units (Hartree,
Bohr and Hartree/Bohr); the parser converts every image to eV, Å and eV/Å on read,
so the viewer shows each image's cell and atoms at the same scale as every other
structure in the project.

``Pop Out Viewer`` moves the structure viewer into its own resizable window while it
stays linked to the page, which is useful for larger supercells on small screens.
The window floats above the main window and carries its own image stepper, so a band
can be scrubbed either from the page or from the popped-out viewer without hiding
it. Closing the window (or clicking ``Restore Viewer``) docks it back.

Because the workflow is anchored to a user-selected ordered endpoint pair, the
reported barrier is a barrier for that specific hop definition. It should not be
read as a complete diffusion network unless the relevant competing paths have
also been generated and compared.

The Energy Barriers stage counts as complete in the workflow navigator and on
the project home once the project holds at least one NEB result. Until then its
summary counts the stage's jobs, or reads ``No barrier calculations yet``.

Each completed NEB job adds an ``Energy barrier calculated`` entry to the
:doc:`/user_guide/journal`.

Current scope and limitations
-----------------------------

- Path generation is explicit and pair-scoped; MuFinder does not automatically
  generate path networks for all relaxed sites.
- Only endpoints with matching charge state are accepted in the page workflow.
- A hop before its band is drawn in the pristine supercell geometry, not the
  distorted relaxed endpoint structures.
- The workflow supports NEB generation and inspection on its page, but it is not a
  general-purpose reaction-network editor.
