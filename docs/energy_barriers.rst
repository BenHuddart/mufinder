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

Workflow overview
-----------------

The project page is divided into two tabs so the workflow remains usable on
smaller laptop screens while still separating path generation from NEB
analysis:

``Paths``
  Select the initial and final relaxed endpoints, generate symmetry-distinct
  paths for that pair, preview a path inside the shared structure viewer, and
  configure, delete, or build the NEB job for each generated path.

``NEB Analysis``
  Inspect completed NEB results with a reaction-coordinate plot, an image
  stepper, a summary readout, and a synchronized structure viewer.

The intended user flow is:

- complete structural relaxation for the two sites of interest;
- choose the initial and final relaxed structures in the ``Paths`` tab;
- optionally set ``Max path length`` to cap the allowed hop length for that
  generation pass;
- click ``Generate Paths`` to enumerate symmetry-distinct muon paths for that
  ordered pair only, including symmetry-equivalent hops of the same relaxed
  site when the same endpoint is selected twice;
- optionally inspect a path with ``View...`` before submission;
- delete generated paths that are no longer relevant with ``Remove Selected
  Paths`` or the ``Delete`` / ``Backspace`` shortcuts;
- configure and build one or more NEB jobs, locally or on the cluster;
- inspect completed NEB paths in the ``NEB Analysis`` tab with the
  reaction-coordinate plot, image stepper, and synchronized structure viewer.

Path generation and assumptions
-------------------------------

MuFinder uses the pristine host symmetry and the two relaxed endpoint
structures to enumerate distinct muon paths. The current implementation:

- requires the two endpoint structures to share the same simulation cell;
- requires the same atom count and species ordering in both endpoints;
- requires exactly one muon in each endpoint structure;
- constructs a pristine supercell matching the relaxed endpoint cell;
- enumerates symmetry-equivalent muon start and end positions in that supercell;
- optionally filters candidates by a user-specified maximum path length;
- discards zero-length self-maps before returning any same-site or cross-site
  path candidates;
- groups equivalent paths by symmetry and keeps the shortest representative.

If no valid symmetry-distinct path can be constructed, the workflow does not
generate a row.

Generated path rows remain in the table after the endpoint selectors change,
so previously generated hops stay available for inspection and submission
while the user prepares another pair.

Path preview
------------

The ``View...`` action on a generated path opens a structure viewer based on
the undistorted pristine supercell that matches the relaxed endpoint cell.
MuFinder overlays the initial and final muon positions for the selected path
and can draw an arrow between them to make the intended migration direction
clear without showing relaxed host distortions. ``View...`` always opens the
viewer for the row whose button was clicked, regardless of the current table
selection.

NEB inputs
----------

For NEB jobs, MuFinder prepares aligned reactant and product structures for
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
- NEB spring, tangent, method, climbing-image, and normalization controls;
- transition-state force, displacement, and energy tolerances.

The dialog also exposes the workflow-supported advanced parameter surface via
the searchable ``Add Parameter`` flow so expert users can opt into additional
CASTEP controls without overwhelming the default view.

When the project defines default convergence settings, the NEB workflow
inherits ``CUT_OFF_ENERGY`` and ``KPOINT_MP_GRID`` from the project
automatically. Row-level configuration remains authoritative, so a specific
path can still override those defaults without affecting the rest of the
project. When the selected path uses a supercell-sized relaxed endpoint,
MuFinder remaps the inherited k-point grid to preserve equivalent
reciprocal-space sampling density relative to the project's recommended
reference lattice.

Outputs and interpretation
--------------------------

Completed NEB jobs persist the generated CASTEP files and the parsed
transition-state path data needed for later review. In the ``NEB Analysis``
tab, the completed-calculations table summarises each result with:

- the path described as ``initial → final`` site labels (falling back to the
  job label when the originating sites are no longer in the project);
- the charge state and path length of the originating hop;
- the forward barrier (peak energy minus the initial image energy);
- the reverse barrier (peak energy minus the final image energy);
- the energy difference ``ΔE`` between the final and initial images;
- the number of images in the band.

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
band; the view only resets when a different NEB result is selected.

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
persisted per-image energies; results recorded before per-image energies were
stored show only the forward barrier. Barriers should be interpreted together
with the selected path definition and the endpoint structures used to
generate that path.

Because the workflow is anchored to a user-selected ordered endpoint pair, the
reported barrier is a barrier for that specific hop definition. It should not be
read as a complete diffusion network unless the relevant competing paths have
also been generated and compared.

Current scope and limitations
-----------------------------

- Path generation is explicit and pair-scoped; MuFinder does not automatically
  generate path networks for all relaxed sites.
- Only endpoints with matching charge state are accepted in the page workflow.
- The preview is based on the pristine supercell geometry, not the distorted
  relaxed endpoint structures.
- The workflow supports NEB generation and inline inspection, but it is not a
  general-purpose reaction-network editor.
