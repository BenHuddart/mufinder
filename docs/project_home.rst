Project home
============

The project home is the page MuFinder shows when a project opens, and the page
the home button above the project navigator returns to. It answers two
questions: where the project stands, and what comes next. It has three parts:
the workflow chart, the field notes for one stage, and a survey of the host
cell. A line above the chart counts the complete stages and the jobs that are
active, and ``Open Journal`` opens the :doc:`journal`.

Workflow chart
--------------

The chart draws every workflow stage as a waypoint, laid out in the order the
work usually goes, with the navigator's Convergence Testing and Muon Sites
sections drawn as named regions. Lines between waypoints show how the stages
depend on each other:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Line
     - Meaning
   * - Solid
     - Required: the later stage needs the earlier one.
   * - Dashed
     - Recommended: the earlier stage sets project defaults the later one uses
       (convergence before UEP and site generation).
   * - Dotted
     - Optional route: one of several ways in, or a side trip.

The dependencies drawn are:

- Host Structure before the two convergence tests; every other stage also works
  on the active host.
- The Supercell Editor is an optional outpost: any calculation can run in a
  saved supercell.
- Pristine Relaxation is an optional detour after convergence. A relaxed
  structure feeds later stages by being made the active host on the Host
  Structure page; UEP can also use it as a source.
- UEP is an optional route into site generation, and an optional source of
  estimated sites for magnetic fields.
- Generation, then Relaxations, then Analysis.
- Magnetic Fields, Quantum Corrections and Energy Barriers all build on the
  distinct sites that Analysis finds.

A waypoint shows its stage's progress:

- a gold seal with a tick when the stage meets its completion rule (the same
  rule as the navigator's seal);
- a half-filled ring when the stage has started but is not complete, for
  example when it has jobs or saved workspaces;
- a dotted ring when nothing has happened there yet.

A blue badge counts a stage's active jobs (submitting, queued, running or
retrieving). A red flag marks the stage the user last worked on; the chart
opens with that stage selected. A line is drawn darker once the stage it leads
to has started.

Click a waypoint to select it; double-click it, or press Return, to open its
page. The arrow keys move the selection through the stages in workflow order.

Field notes
-----------

The field notes describe the selected stage: its status, a few facts about the
project (the host's formula and atom count, the default cutoff and k-point
grid, saved supercells, relaxations relaxed per charge state, analysis
workspaces and pinned distinct sites, barrier heights and so on), what the
stage is for, and an ``Open`` button that goes to its page. A stage that has
not started lists, under ``Needs``, the required earlier stages that have not
started either.

Survey of the cell
------------------

The survey draws the active host's unit cell looking down one lattice vector,
chosen with the ``a``, ``b`` and ``c`` buttons. Atoms on the cell faces appear
on every face they touch, atoms deeper in the cell are drawn fainter, and a key
names each element's colour. Hovering an atom or a marker shows what it is.

The survey follows the selected waypoint and shows that stage's results:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Stage
     - Shown
   * - UEP
     - The clustered UEP minima of each source job, as dashed circles.
   * - Generation
     - The candidate sites on the active host.
   * - Relaxations
     - Where each muon ended, with a line from where it started; rows still
       running or not yet submitted are shown faint at their starting point.
       One result set per charge state.
   * - Analysis
     - The distinct sites of each analysis workspace, sized by the number of
       relaxations they gather and labelled with their Wyckoff label. The
       workspace pinned to the Muon Sites page is marked ``(pinned)``.
   * - Quantum Corrections
     - The sites with a zero-point energy, labelled with it.
   * - Energy Barriers
     - The diffusion paths, labelled with their barrier height once computed.
   * - Magnetic Fields
     - The sites with a local-field result, labelled with ``|B|``, one result
       set per saved workspace.

When a stage has more than one result set, the arrows under the drawing step
through them. A stage with no positions of its own (Host Structure, Supercell
Editor, the convergence tests, Pristine Relaxation, or any stage not yet
started) shows the sites found so far, faded: the pinned distinct sites if
there are any, otherwise the most refined positions available.

Positions are always drawn in the active host's cell. UEP minima and diffusion
paths computed in a supercell are carried through Cartesian coordinates into
it. Results tied to a different host, for example candidate sites generated
before a relaxed pristine structure became the active host, are left out and
counted in a note.

The stage the user last worked on is saved with the project in UI state under
``project_home_stage``.
