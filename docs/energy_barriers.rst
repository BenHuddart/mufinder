Energy Barriers
===============

Energy Barriers guides the user from two completed relaxed muon endpoints to a
transition-state search and, optionally, a follow-on nudged elastic band (NEB)
calculation. The workflow is intended for comparing two already-relaxed MuFinder
muon sites that belong to the same host and charge state.

Scientific intent
-----------------

This workflow answers a different question from site generation or relaxation:
not where the muon can stop, but how it moves between two already-established
local minima. MuFinder therefore treats Energy Barriers as an endpoint-driven
workflow. You begin from two completed relaxed sites, define a specific ordered
hop between them, and then refine the corresponding migration pathway with a
transition-state search and, when needed, a higher-fidelity NEB calculation.

The path preview and inline inspector are designed to keep the science clear.
MuFinder shows the migration geometry in the matching undistorted pristine
supercell so the user can inspect the start and end muon positions without
mixing in host distortions from two separately relaxed defect cells.

Workflow overview
-----------------

The project page is divided into three tabs so the workflow remains usable on
smaller laptop screens while still separating path generation, TSS review, and
NEB analysis:

``Paths``
  Select the initial and final relaxed endpoints, generate symmetry-distinct
  paths for that pair, preview the path inside the shared structure viewer, and
  configure the transition-state search rows.

``Transition States``
  Review completed transition-state search jobs and configure any downstream NEB
  parameters for the paths that should be refined.

``NEB Analysis``
  Inspect completed NEB results with a synchronized reaction-coordinate plot,
  slider, summary readout, and structure viewer.

The intended user flow is:

- complete structural relaxation for the two sites of interest;
- choose the initial and final relaxed structures in the ``Paths`` tab;
- click ``Generate Paths`` to enumerate symmetry-distinct muon paths for that
  ordered pair only;
- optionally inspect a path with ``View...`` before submission;
- configure and submit one or more transition-state search rows;
- once a TSS job completes, configure and build NEB jobs from the rows that
  should be refined further;
- inspect completed NEB paths in the final tab with the reaction-coordinate
  slider and synchronized structure viewer.

Path generation and assumptions
-------------------------------

MuFinder uses the pristine host symmetry and the two relaxed endpoint
structures to enumerate distinct muon paths. The current implementation:

- requires the two endpoint structures to share the same simulation cell;
- requires the same atom count and species ordering in both endpoints;
- requires exactly one muon in each endpoint structure;
- constructs a pristine supercell matching the relaxed endpoint cell;
- enumerates symmetry-equivalent muon start and end positions in that supercell;
- groups equivalent paths by symmetry and keeps the shortest representative.

If no valid symmetry-distinct path can be constructed, the workflow does not
generate a row.

Path preview
------------

The ``View...`` action on a generated path opens a structure viewer based on
the undistorted pristine supercell that matches the relaxed endpoint cell.
MuFinder overlays the initial and final muon positions for the selected path
and can draw an arrow between them to make the intended migration direction
clear without showing relaxed host distortions.

Transition-state and NEB inputs
-------------------------------

For transition-state searches, MuFinder prepares aligned reactant and product
structures for the selected path and writes CASTEP inputs with
``TASK : TRANSITIONSTATESEARCH`` and ``TSSEARCH_METHOD : LSTQST`` locked by the
workflow.

For NEB jobs, MuFinder uses a completed transition-state result as the source,
then writes CASTEP inputs with ``TASK : TRANSITIONSTATESEARCH`` and
``TSSEARCH_METHOD : NEB``. The product image and, when available, the
intermediate image from the parsed transition-state file are written into the
generated ``.cell`` file.

Configuration UI
----------------

The configuration dialogs are intentionally curated around the most relevant
barrier-search controls first.

For transition-state search rows, the default dialog surface emphasizes:

- precision and cutoff controls;
- electronic energy tolerance;
- Monkhorst-Pack grid;
- LST/QST iteration and path-point limits;
- transition-state force, displacement, and energy tolerances.

For NEB rows, the default dialog surface emphasizes:

- precision and cutoff controls;
- electronic energy tolerance;
- Monkhorst-Pack grid;
- NEB spring, tangent, method, climbing-image, and normalization controls;
- transition-state force, displacement, and energy tolerances.

Both dialogs also expose the workflow-supported advanced parameter surface via
the searchable ``Add Parameter`` flow so expert users can opt into additional
CASTEP controls without overwhelming the default view.

Outputs and interpretation
--------------------------

Completed transition-state and NEB jobs persist the generated CASTEP files and
the parsed transition-state path data needed for later review. In the project
page, MuFinder surfaces:

- the transition-state or NEB file path;
- barrier height and image counts;
- the inline NEB profile plot, slider-linked image inspection, and structure
  viewer.

The barrier values shown in the tables are taken from the parsed CASTEP
transition-state output. They should be interpreted together with the selected
path definition and the endpoint structures used to generate that path.

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
- The workflow supports TSS and NEB generation and inline inspection, but it is
  not a general-purpose reaction-network editor.