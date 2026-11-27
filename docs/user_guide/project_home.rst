Project home
============

The project home is the page MuFinder shows when a project opens, and the page
``Project Home``, the first row of the project navigator, returns to. It reminds you, when
you come back to a project, what the project has found and where you stopped.
It describes the project; the navigator beside it lists the stages and shows
which are complete.

.. figure:: ../_generated/screenshots/project_home.webp
   :alt: The project home of the CoF2 example: the project's name and host line,
         where you left off, the pinned muon sites beside the cell survey,
         the other results and the recent activity.

   The project home of the CoF₂ example (:doc:`/user_guide/examples/cof2`) once
   its sites are pinned and a phonon calculation has run.

Before you start
----------------

Open or create a project (``File > New Project…`` or ``File > Open Project…``).
The home needs nothing else: it reads the project's current state, so it is
useful on an empty project and on a finished one.

Using the project home
----------------------

1. **Read where you left off.** The ``Where you left off`` card names the
   stage you last worked on and the day of that work, with a line of facts
   about it. Click ``Open <stage>`` (for example ``Open Quantum Corrections``)
   to go back to it.
2. **Check the sites found.** The ``Muon sites found`` table lists the sites
   pinned to the Muon Sites summary. Select a row to pick out that site in the
   survey of the cell; double-click it, or press Return or Enter, to open the
   Muon Sites summary.
3. **Look at the cell.** Turn the survey's 3D view with the mouse, or use the
   ``a``, ``b`` and ``c`` buttons beside ``Along:`` and ``Reset View`` (see
   `Survey of the cell`_).
4. **Read the rest.** ``Other results`` sums up the later workflows and
   ``Recent activity`` lists the newest Journal entries. ``Open Journal``, at
   the top right, opens the :doc:`/user_guide/journal`.
5. **Come back.** Select ``Project Home`` at the top of the project
   navigator at any time to return here.

What the page shows
-------------------

The sections read from top to bottom:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Section
     - Shown
   * - Header
     - The project's name; the active host's formula, space group and atom
       count; and the adopted cutoff and k-point grid, or ``not converged yet``
       when neither is adopted.
   * - Where you left off
     - The stage of the newest activity in the project: the newest job
       created, submitted, started or finished, or the newest Journal record
       of a workflow, whichever is later. The date is the day of that activity,
       written like ``24 Nov 2026``. A line of facts about
       the stage follows, the same facts the stage's own page summarises (the
       default cutoff and the tests run, relaxations relaxed per charge state,
       phonon results, barrier heights and so on). A project with no activity
       yet shows the first stage still to do, with no date: Host Structure on
       an empty project.
   * - Muon sites found
     - One row per site pinned to the Muon Sites summary, positive muon sites
       first and then muonium sites, each lowest ``dE`` first, with the site's
       Wyckoff label and its energy above the lowest site of its charge state.
       With nothing pinned, one sentence says how far the search on the active
       host has got, for example ``7 candidate sites, 4 of 14 relaxed``, in
       place of the table.
   * - Survey of the cell
     - The active host's cell with the pinned sites, or, with nothing pinned,
       the most refined positions found so far (see `Survey of the cell`_).
   * - Other results
     - One line joining the facts of Magnetic Fields, Polarisation, Hyperfine
       Coupling, Quantum Corrections and Energy Barriers, leaving out each one
       that has not started. Hidden while none has.
   * - Recent activity
     - The five newest Journal entries, newest first, each with its day and
       title. Hidden while the Journal is empty.

The sites, the other results and the recent activity read down the left of the
page and the survey takes the right. When the window is too short for the whole
page, the page scrolls.

Opening another project resets every section: no site stays selected, the
survey shows the new host from the default view and the page scrolls back to
the top. While the same project stays open, the page keeps up with it without
moving the survey's view.

Survey of the cell
------------------

The ``Survey of the cell`` shows the active host's unit cell in 3D, with a
muon sphere at each position found, in the muon colour the other 3D views use.
It is the structure viewer of the Host Structure page in a compact form: drag
with the right mouse button to rotate and with the middle button to pan, and
use the wheel to zoom. ``Cell``, ``Bonds`` and ``Spins`` show or hide the cell
edges, the bonds and the spin arrows; ``a``, ``b`` and ``c`` beside ``Along:``
look along a lattice vector; ``Reset View`` returns to the default view. The
axes in the lower left follow the view. The caption under the view names what
the spheres are.

The view fills in just after the rest of the page appears, so opening a project
is no slower for it. A project with no host structure shows a line asking for
one in its place. If the 3D view cannot be started, for example on a computer
without working OpenGL, the card says ``The 3D view is unavailable.`` and the
rest of the page works as usual.

With sites pinned, the survey shows each pinned site and puts a translucent
cyan halo around the site selected in the table; the muon sphere itself keeps
its size and colour. With nothing pinned it shows the most refined positions found so far, in this order of preference:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Positions
     - Shown
   * - Analysis
     - The distinct sites of an analysis workspace.
   * - Relaxations
     - Where each muon ended; rows still running or not yet submitted are
       shown at their starting point.
   * - Generation
     - The candidate sites on the active host.
   * - UEP
     - The UEP sites of a source job.

Positions are always shown in the active host's cell. UEP minima computed in a
supercell are carried through Cartesian coordinates into it. Results tied to a
different host, for example candidate sites generated before a relaxed pristine
structure became the active host, are left out and counted in a note.
