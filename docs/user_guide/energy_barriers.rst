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

The path preview and inline inspector are designed to keep the science clear.
MuFinder shows the migration geometry in the matching undistorted pristine
supercell so the user can inspect the start and end muon positions without
mixing in host distortions from two separately relaxed defect cells.

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

1. **Open the page.** Select ``Energy Barriers`` in the navigator and stay on
   the ``Paths`` tab.
2. **Choose the endpoints.** Under ``Path Endpoints``, pick the ``Initial`` and
   ``Final`` relaxed structures.
3. **Cap the path length (optional).** Set ``Max path length (Å)`` to limit the
   hop length for this generation pass. Leave it blank, shown as ``Auto``, to
   use MuFinder's automatic cutoff.
4. **Generate the paths.** Click ``Generate Paths`` to enumerate
   symmetry-distinct muon paths for that ordered pair only, including
   symmetry-equivalent hops of the same relaxed site when the same endpoint is
   selected twice. Each path becomes a row of the ``Diffusion Paths`` table.
5. **Preview a path (optional).** Click a row's ``View…`` button to inspect
   the hop before submission.
6. **Remove paths you do not need.** Use ``Remove Selected``, the row menu's
   ``Remove`` entry, or the ``Delete`` / ``Backspace`` shortcuts.
7. **Configure the NEB jobs.** Select the paths and click ``Configure Selected
   Parameters…``, or use a row's ``Configure…`` button (see `Configuration
   UI`_).
8. **Run them.** Click ``Run All Locally`` or ``Run All on Cluster``; the
   buttons read ``Run Selected ...`` when rows are selected. MuFinder builds
   and submits the NEB jobs (see :doc:`/getting_started/running_jobs`).
9. **Inspect the results.** When jobs complete, open the ``NEB Analysis`` tab
   (see `Outputs and interpretation`_).

.. figure:: ../_generated/screenshots/gdru2si2_energy_barrier_paths.webp
   :alt: The Paths tab with GdRu2Si2 site 1 as the initial endpoint and site 2
         as the final one, two site 1 to site 1 paths ready and the site 1 to
         site 2 path completed with 4 task farms.

   The paths of the GdRu₂Si₂ worked example (:doc:`/user_guide/examples/gdru2si2`):
   two hops between copies of site 1 and the hop from site 1 to site 2.

The Energy Barriers page
------------------------

The page is divided into two tabs so the workflow remains usable on smaller
laptop screens while still separating path generation from NEB analysis:

``Paths``
  Select the initial and final relaxed endpoints (each entry shows the site's
  ``dE`` relative to the lowest-energy relaxed site with the same charge state
  and cell, as in Quantum Corrections), generate symmetry-distinct paths for
  that pair, preview a path inside the shared structure viewer, and
  configure, delete, or build the NEB job for each generated path.

``NEB Analysis``
  Inspect completed NEB results with a reaction-coordinate plot, an image
  stepper, a summary readout, and a synchronized structure viewer.

The endpoint selectors list the distinct sites pinned to the ``Muon Sites``
summary (:doc:`/user_guide/structural_relaxation_analysis`), labelled by Wyckoff label and
cluster (``8c · cluster 1``) in the summary's energy rank, so each physical
site is offered once. Tick ``Show every relaxed site`` above the selectors to
offer every eligible relaxed site instead; the choice is saved with the
project. When no charge state has a pin the selectors are empty and a notice
with ``Go to Muon Sites →`` explains how to pin sites; the notice also names a
charge state with no pin and counts pinned sites that no longer resolve to an
eligible relaxed site. A site relaxed in both charge states is offered once in
each, and its paths in one charge state are kept apart from those in the
other. A stored endpoint the selectors no longer list is
replaced by the first listed site (the final endpoint by the first other one),
so widening the list keeps the pair the selectors showed. Paths already
generated stay in the ``Diffusion Paths``
table whichever list the selectors show.

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

The ``Diffusion Paths`` table shows each generated path's endpoints
(``Reactant`` and ``Product``), ``Charge``, and ``Path Length``, its configured
task-farm count in the ``Farms`` column (``Off`` when farming is disabled), its
NEB job ``Status``, a per-row ``Configure…`` button in the ``Parameters``
column (it reads ``Configured…`` once the row has custom settings or its job
has been built), and a ``View…`` button.
Right-clicking selected rows offers the shared row actions (``Run Local``,
``Run on Cluster``, ``Cancel``, ``Inspect Failure``, ``Reset Failed
Jobs…``, ``Configure…``, ``Remove``).

Generated path rows remain in the table after the endpoint selectors change,
so hops generated earlier stay available for inspection and submission
while the user prepares another pair.

Path preview
------------

The ``View…`` action on a generated path opens a structure viewer based on
the undistorted pristine supercell that matches the relaxed endpoint cell.
MuFinder overlays the initial and final muon positions for the selected path
and can draw an arrow between them to make the intended migration direction
clear without showing relaxed host distortions. ``View…`` always opens the
viewer for the row whose button was clicked, regardless of the current table
selection.

.. figure:: ../_generated/screenshots/gdru2si2_path_preview.webp
   :alt: The path viewer with the GdRu2Si2 pristine 2x2x1 supercell viewed
         down a and the path arrow from site 1 to site 2.

   The GdRu₂Si₂ site 1 to site 2 hop in the pristine supercell.

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

The configuration dialog is intentionally curated around the most relevant
barrier-search controls first. For NEB rows, the default dialog surface
emphasizes:

- precision and cutoff controls;
- electronic energy tolerance;
- Monkhorst-Pack grid;
- the NEB image count (``TSSEARCH_MAX_PATH_POINTS``, default 11). It counts
  the intermediate images only, so the finished band holds two more: the
  reactant and the product. An even value is rounded up to the next odd
  number, because climbing image is locked on and CASTEP would otherwise add
  the image itself;
- NEB spring constant, tangent mode, method, maximum iterations, and
  normalization controls;
- the transition-state force tolerance (``TSSEARCH_FORCE_TOL``), shown in
  eV/Å with CASTEP's own 0.05 eV/Å default pre-filled;
- the CASTEP task-farm count (see `Task farming`_).

``TASK``, ``TSSEARCH_METHOD``, and ``TSSEARCH_NEB_CLIMBING`` (``TRUE``) are
locked by the workflow and cannot be edited.

The dialog also exposes the workflow-supported advanced parameter surface via
the searchable ``Add Parameter`` flow so expert users can opt into additional
CASTEP controls without overwhelming the default view.

.. figure:: ../_generated/screenshots/gdru2si2_neb_configuration.webp
   :alt: The Configure NEB Parameters dialog for the GdRu2Si2 site 1 to site 2
         path: 1000 eV, a 3 x 3 x 3 grid, 41 path points, a force tolerance of
         0.08 eV/A, PBE, spin polarised, the locked TS search method and
         climbing image, and 4 task farms.

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

Task farming changes the output filenames. CASTEP appends ``_farmNNN`` to the
seedname before the calculation starts, so a farmed run writes
``<seed>_farm001.castep`` … ``<seed>_farmNNN.castep`` and, for NEB,
``<seed>_farm001.ts`` — **no** unfarmed ``<seed>.ts`` is produced at all. Every
farm writes a complete, identical copy of the band, so MuFinder reads farm 001.
The ``<seed>.castep`` file that a farmed run still writes contains only the
header and timings, not the band or its convergence, which is why MuFinder
parses the farmed file for the job's real outcome.

Outputs and interpretation
--------------------------

.. figure:: ../_generated/screenshots/gdru2si2_neb_analysis.webp
   :alt: The NEB Analysis tab with the GdRu2Si2 site 1 to site 2 band: a
         forward barrier of 0.495 eV, a reverse barrier of 0.213 eV, 42 images,
         and the profile with the transition-state image 23 of 42 in the
         viewer.

   The real CASTEP band from GdRu₂Si₂ site 1 to site 2 (Huddart *et al.*
   2025), with its transition state in the viewer.

MuFinder keeps the generated CASTEP files of completed NEB jobs and the
parsed transition-state path data needed for later review. In the ``NEB Analysis``
tab, the completed-calculations table summarises each result with:

- the path described as ``initial → final`` site labels (falling back to the
  job label when the originating sites are no longer in the project);
- the charge state and path length of the originating hop;
- the forward barrier (peak energy minus the initial image energy);
- the reverse barrier (peak energy minus the final image energy);
- the energy difference ``ΔE`` between the final and initial images;
- the number of images in the band;
- a ``Status`` column carrying the producing job's status, coloured with the
  same per-status palette used across the workflow tables (the ``Status`` column
  in the ``Diffusion Paths`` table uses this same vocabulary and colouring).

The CASTEP trajectory (``.ts``) file path is not shown as a column; it is
available as a tooltip on each row, and right-clicking a row offers ``Open
Containing Folder`` for direct access to the raw CASTEP output.

The NEB path inspector below the table shows the reaction-coordinate profile
with labelled axes and a dashed annotation marking the forward barrier at the
peak image. Energies are plotted relative to the initial image (``E − E₀``)
so the axis stays readable despite the large absolute CASTEP energies. The
plot is the primary scrub surface: click or drag to select the nearest image,
use the Left/Right arrow keys to step, or use the ``◀ Image n / N ▶`` stepper
next to the plot. The structure viewer follows the selected image and
preserves the current camera orientation and zoom while scrubbing along the
band; the view only resets when a different NEB result is selected. CASTEP
writes the ``.ts`` file in atomic units (Hartree, Bohr and Hartree/Bohr); the
parser converts every image to eV, Å and eV/Å on read, so the viewer shows
each image's cell and atoms at the same scale as every other structure in the
project.

``Pop Out Viewer`` moves the structure viewer into its own resizable window
while it stays linked to the reaction-coordinate controls, which is useful
for larger supercells on small screens. The window floats above the main
window and carries its own image stepper, so the band can be scrubbed either
from the inspector or from the popped-out viewer without hiding it. Closing
the window (or clicking ``Restore Viewer``) docks it back into the inspector.
On narrow windows the inspector automatically stacks the plot controls above
the viewer so the viewer keeps a usable size.

The barrier values shown in the table are taken from the parsed CASTEP
transition-state output. Reverse barriers and ``ΔE`` are derived from the
stored per-image energies; a result without stored per-image energies shows
only the forward barrier. Barriers should be interpreted together
with the selected path definition and the endpoint structures used to
generate that path.

Because the workflow is anchored to a user-selected ordered endpoint pair, the
reported barrier is a barrier for that specific hop definition. It should not be
read as a complete diffusion network unless the relevant competing paths have
also been generated and compared.

The Energy Barriers stage counts as complete in the workflow navigator and on
the project home once the project holds at least one NEB result. Until then its
summary counts the stage's jobs, or reads ``No barrier calculations yet``.

Current scope and limitations
-----------------------------

- Path generation is explicit and pair-scoped; MuFinder does not automatically
  generate path networks for all relaxed sites.
- Only endpoints with matching charge state are accepted in the page workflow.
- The preview is based on the pristine supercell geometry, not the distorted
  relaxed endpoint structures.
- The workflow supports NEB generation and inline inspection, but it is not a
  general-purpose reaction-network editor.
