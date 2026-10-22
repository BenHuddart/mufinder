Quality and Automation
======================

Quality gates
-------------

All implemented and future functionality must satisfy these gates:

- Unit and integration tests must pass.
- Agent-harness structural tests must pass.
- Coverage must remain at or above 85 percent (enforced by pytest-cov).
- Sphinx documentation must build with warnings treated as errors.
- New functionality is incomplete until tests and docs are included.

Local commands
--------------

The ``Makefile`` is the single definition of the local commands; this page, the
README, the pull-request template and CI refer to its targets. Pass
``PYTHON=/path/to/python`` to use a specific interpreter.

- ``make install`` installs MuFinder editable with the ``dev`` and
  ``scientific-backends`` extras, both of which the test suite needs.
  ``uv sync --extra dev --extra scientific-backends`` is equivalent.
- ``make test`` runs the full suite with the 85 percent coverage gate on four
  ``pytest-xdist`` workers. Run it in the foreground and keep it at four
  workers or fewer: each worker loads Qt, and more can exhaust the machine.
  The gate is ``fail_under`` in ``[tool.coverage.report]``, the one place the
  threshold is set.
- ``make test-shard SHARD=n SHARDS=m`` runs group ``n`` of ``m`` groups of the
  suite (``pytest-split``) and writes that group's coverage data to
  ``.coverage.shard-n`` with no report and no gate. ``make coverage`` combines
  every ``.coverage.shard-*`` file and enforces the gate. CI runs the suite this
  way, as four shards.
- ``make test-fast`` runs focused tests without coverage. With no arguments it
  runs the targeted execution checks listed below; ``make test-fast
  FILES="tests/science/test_supercell.py"`` runs the files you name.
- ``make harness`` runs the agent-harness structural tests.
- ``make screenshots`` renders the documentation screenshots into
  ``docs/_generated/screenshots`` (see `Documentation screenshots`_).
- ``make docs`` renders the screenshots when they are out of date (see `When
  the renderer runs`_), then builds this Sphinx site with warnings treated as
  errors.
- ``make lint`` runs ``ruff check`` over ``src``, ``tests``, ``scripts`` and the API example scripts in ``docs/api/examples``.
- ``make quality`` runs ``lint``, ``test`` and ``docs``, the gates CI enforces.

To skip the Qt widget tests while iterating, select by marker with
``python -m pytest --no-cov -m "not gui"``. ``tests/conftest.py`` marks every
test under ``tests/app/`` or using the ``qapp`` fixture as ``gui`` and every
other test as ``unit``.

PySide test shutdown note
-------------------------

The pytest configuration in ``tests/conftest.py`` includes a final
``atexit`` process-exit hook. This is intentional: some macOS/PySide6 runs
can trigger a segmentation fault during late interpreter shutdown after tests
and coverage have already completed.

The hook records pytest's final exit status, flushes stdout/stderr, and exits
the process deterministically so CI and local runs report stable outcomes
without post-run crashes.

Test-suite performance
----------------------

CI runs on 4-vCPU runners. Keep these in place when changing the test setup:

- **Sharded CI runs.** CI splits the suite into four groups with
  ``pytest-split`` and runs each on its own runner, so a CI run takes about a
  quarter of the suite's run time plus one runner's setup. With no stored
  durations, ``--splitting-algorithm least_duration`` deals the tests out one
  at a time in test-ID order, which keeps the groups within a few seconds of
  each other; each group then runs in collection order. Coverage is
  measured per shard and combined before the gate, so the threshold applies to
  the whole suite exactly as in ``make test``.
- **Work-stealing workers.** ``make test`` and ``make test-shard`` run
  ``pytest-xdist`` with ``--dist worksteal``: an idle worker takes queued tests
  from a busy one, so a handful of slow tests at the end does not leave one
  worker running alone.
- **``uv`` installs.** CI installs from ``uv.lock`` with
  ``uv sync --frozen`` and a cached ``uv`` download store, which takes seconds
  where ``pip`` took close to a minute per job.

- **Incremental garbage collection between tests.** The autouse per-test Qt
  cleanup in ``tests/conftest.py`` runs ``gc.collect()`` after each test so
  leaked widgets are destroyed before the next one starts. A full collection
  walks every tracked object (around 200k once Qt, VTK, SciPy and matplotlib
  are imported), and paying that after every test would cost close to half of
  the suite's run time. The cleanup therefore freezes each collection's
  survivors with ``gc.freeze()``, so the next collection only scans what the
  next test allocated. The first teardown in each new test module unfreezes
  everything and does a full collection, so a reference cycle that outlives one
  test is still reclaimed.
- **``sys.monitoring`` coverage.** ``[tool.coverage.run]`` sets
  ``core = "sysmon"``, which measures line coverage with much less overhead
  than the default C tracer on Python 3.12+. Enabling branch coverage would
  silently fall back to the slower tracer on Python 3.12 and 3.13.
- **Slow-test report.** CI passes ``--durations=25``, so the slowest test
  phases are listed at the end of every CI shard log. Check that list when CI time
  creeps up; locally, ``make test PYTEST_ARGS="--no-cov --durations=25"``
  gives the same report.

Test layout
-----------

Tests mirror ``src/mufinder/``: each package has a test package of the same
name (``tests/app``, ``tests/domain``, ``tests/services``, ``tests/workflow``
and so on), and ``tests/setup`` holds the tests of the top-level modules
(``main``, ``cluster_setup``, ``branding`` and the version metadata). The
structural tests in ``tests/test_agent_harness.py``, the golden-input rendering
test and ``tests/conftest.py`` stay at the top level. The test packages have no
``__init__.py``, so pytest imports every test module by its file name: a test
file name must be unique across the whole ``tests`` tree. The one exception is
``tests/app/workflow_controllers``, which has an ``__init__.py`` so that its
modules can take the names of the controller modules they test
(``test_base.py``, ``test_uep.py``) without clashing with ``tests/workflow`` and
``tests/science``.

``tests/app/main_window`` holds the integration tests that drive
``MainWindow`` itself, one module per subject: the jobs dashboard, job
recovery, cluster and local submission, each workflow's actions, and the
project, structure and site-generation actions. A workflow controller's own
behaviour is tested through the controller in ``tests/app/workflow_controllers``,
which has one module per module of ``mufinder.app.workflow_controllers``; the
shared sweep behaviour of ``sweep_convergence`` is tested through the cutoff
controller in ``test_convergence_cutoff.py``. A window test covers what only the
window does, such as its signal wiring, its dialogs and the real submission
path.

``tests/docs`` checks the structure of the Sphinx sources and their citations,
never their wording. ``test_docs_build_inputs.py`` asserts that every page under
``docs/`` is reStructuredText (Sphinx has no Markdown parser configured, so a
``.md`` page would never be built), that every page is reachable from a toctree
in ``docs/index.rst``, and that each
literature DOI the user guides cite stays on its page; a new citation is added
to its ``CITED_DOIS`` table. Tests do not pin prose phrases, ``conf.py`` lines
or stylesheet values: whether a page describes the code correctly is checked in
review, and the ``-W`` build catches broken references and missing assets.

The Python API pages are build outputs. ``docs/api/index.rst`` holds a recursive
``autosummary`` over the library packages, and Sphinx writes one page per
module, class, function and exception into ``docs/api/generated`` (ignored by
git) from the templates in ``docs/_templates/autosummary``. Those pages are
reached through the ``autosummary`` directive rather than a toctree, so the
reachability check skips them. A docstring that is not valid reStructuredText
fails the ``-W`` build.

Pages that moved when the site was split into sections keep their old URLs:
``REDIRECTS`` in ``docs/conf.py`` maps each former document name to its new
page, and the HTML build writes a stub at the old name that forwards the
browser. ``tests/docs/test_docs_redirects.py`` checks that every target is a
current page and that no old name is still one. A page that moves adds a row.

Documentation screenshots
-------------------------

The documentation screenshots are build outputs: no image under ``docs/`` is
committed for them. ``scripts/render_docs_screenshots.py`` runs the
``scripts/docs_screenshots`` package, which writes each one to
``docs/_generated/screenshots`` (ignored by git) as ``<name>.webp``, and a page
shows it with a ``figure`` directive whose path is relative to the page, for
example ``../_generated/screenshots/<name>.webp`` from a page in
``docs/user_guide/``.

The renderer needs no display: Qt draws on its ``offscreen`` platform at twice
the logical size, and VTK renders through EGL. pyvistaqt's ``QtInteractor``
needs a native window, so the renderer gives the 3D viewers an off-screen
PyVista plotter that paints its frame instead. Settings and app data go to a
temporary directory, so a local run leaves the developer's own MuFinder
configuration alone.

The package
~~~~~~~~~~~

- ``render.py`` sets the Qt, VTK and settings environment when it is first
  imported, before anything imports Qt, so every module of the package that
  touches Qt imports it first. It holds what every scene shares: the
  off-screen viewer, the grab targets, ``render`` and its project cache,
  ``synthetic_runs``, ``magnetic_project`` and ``custom_site`` (the saved
  project of a host with a magnetic model and a Magnetic Fields form, and a
  custom muon site) and the scene steps more than one example uses.
- ``synthetic.py`` writes the CASTEP outputs the projects need: ``-out.cell``
  files, ``.castep`` results, a BFGS geometry-optimisation log
  (``bfgs_log_text``) with falling energies and forces, a model charge density
  of superposed Gaussian atomic shells (``den_fmt_text``) with the
  single-point output the UEP analysis reads beside it (``uep_castep_text``),
  a Gamma-point ``.phonon`` file for a linear X–μ–X complex embedded in a
  supercell (``triatomic_phonon_text``), NEB ``.ts`` bands and legacy-run
  outputs. It imports no Qt, so tests import it directly.
  ``bridged_diatomic_cell`` turns the two molecules nearest a muon to point at
  it, and ``bridged_diatomic_phonon_text`` writes a Gamma-point ``.phonon``
  file for that [X₂–μ–X₂] complex in a supercell of diatomic molecules.
- ``examples/`` holds one module per example project. Each defines its
  project builders, the scene steps only its scenes use, and ``SCENES``, a
  dictionary from screenshot name to ``Scene``.
- ``registry.py`` lists the example modules in render order as ``EXAMPLES``
  and merges their scenes into ``SCREENSHOTS``. A screenshot name belongs to
  exactly one example.
- ``__main__.py`` is the command line. With no arguments it renders every
  screenshot; ``--example NAME`` renders one example's scenes, names render
  those screenshots, ``--list`` prints the names it would render, and
  ``--output`` writes somewhere other than ``docs/_generated/screenshots``.
  ``--jobs N`` (default: four, or fewer CPUs) deals the screenshots out in
  registry order to ``N`` worker processes, each a run with names of its own
  share, so a project several workers need is built in each of them; ``--jobs
  1`` renders in one process.

Scenes
~~~~~~

Each entry of an example's ``SCENES`` maps a file name to a ``Scene``:

- a project builder, a function that writes a saved project into a scratch
  directory and returns its root;
- the navigator route to open (``None`` leaves the Project Home showing);
- an optional ``setup`` step that drives the window, such as clicking
  ``Calculate Dipolar Field``;
- the grab target: ``Window`` (the whole main window, optionally with the Job
  Dashboard shown), ``Page`` (one page widget, named by its ``MainWindow``
  attribute, or returned by a callable for a page kept per saved item, and
  laid out at a given logical size), ``Dock`` (a dock widget, by title),
  ``Dialog`` (the modal dialog an opener callable shows, matched by class
  name, so a dialog opened from another dialog can be grabbed, optionally
  grabbing only a ``region`` of it, such as one panel) or
  ``Floating`` (a non-modal window, such as the geometry progress or UEP
  clustering window, that an opener callable shows and returns). Each target
  takes an optional ``prepare`` step that receives the grabbed widget before
  the grab.

``render`` builds each project builder's project once per process, keyed on
the builder, and opens a fresh copy of it for every scene, so the scenes of an
example share one build and a scene that changes or saves its project leaves
the next scene's copy alone. A saved project stores its paths relative to its
root, so a copy opens as the original does. A builder that drives a
``MainWindow`` runs inside the renderer's ``QApplication`` like any scene.

The renderer opens the project in a real ``MainWindow`` with the status bar
hidden, since its keyword-catalog and cluster-connection warnings describe the
scratch settings rather than the page. A dialog is dismissed after its grab,
and a scene whose dialog never opens fails rather than hanging. Each window is
deleted once its scene is saved, so the next window's style sheet is not slowed
by widgets that are no longer shown.

Prefer ``Page`` grabs about 1000 to 1200 logical pixels wide: the theme scales
a figure to the content column, so a full-window grab at 1440 pixels makes its
text small. Keep ``Window`` for views about the window as a whole, such as the
Project Home and the Job Dashboard.

Images
~~~~~~

Each grab is written as lossless WebP (Qt's WebP writer at quality 100), so
the decoded pixels are exactly those grabbed, at about half the size of a PNG.
A ``build-finished`` hook in ``docs/conf.py`` post-processes the HTML pages:
every WebP image under ``_images/`` gets ``loading="lazy"`` and
``decoding="async"``, and, unless its figure sets a ``:width:`` or
``:height:``, explicit ``width`` and ``height`` attributes in CSS pixels, the
file's pixel size divided by the renderer's scale factor. The renderer fixes
``QT_SCALE_FACTOR`` at that factor rather than inheriting it from the
environment, and a test keeps the two constants equal. The browser reserves
each screenshot's space before it
loads and fetches only the screenshots near the viewport; the theme's
``max-width: 100%`` and ``height: auto`` still scale an image down to the
content column. Other images are left alone.

The CoF₂ example
~~~~~~~~~~~~~~~~

``examples/cof2.py`` holds every scene but those of the iron and manganese silicide magnetic models
and of the α-N₂ quantum corrections. Its
host is rutile CoF₂ with antiparallel spins on the two Co, ``PBE`` as the
project functional and a saved 2x2x2 supercell; :doc:`/user_guide/examples/cof2`
tells the story its scenes illustrate and cites the values its outputs are
tuned to.

``fixture_project`` is the configured-but-unsubmitted project of the setup
scenes: a cutoff sweep (300 to 800 eV) and a k-point sweep (the grids
``Generate From Spacing...`` gives down to 0.05 1/Ang), none submitted, and
ten candidate muon sites from ``ProjectService.generate_candidate_sites``,
one 1 Å from each fluorine and six random with a fixed seed. The builders make
the calls the application makes, including the sweep widgets' own form logic,
so the saved project is one the application could have written.
``converged_project`` adopts 600 eV and a 4x4x6 grid with illustrative sweep
data pinned on the Convergence Testing overview, and ``dashboard_project``
binds failed, stopped and cancelled jobs to three cutoff rows.

``results_project`` starts from the converged project, adds the
antiferromagnetic model, opens the project in a ``MainWindow`` and drives each
workflow as its buttons do: a spin-polarised pristine relaxation and its
comparison table, a UEP source row with a seeded batch search and its
clustering, the ten starts staged with ``Both charge states`` as positive-muon
and muonium relaxations in the 2x2x2 supercell, one analysis workspace per
charge state, clustered and pinned, the phonon job of the lowest positive-muon
site, the distortion-corrected 4c site on the Magnetic Fields page, and the 4c
to 4c paths on the Energy Barriers page with an NEB band along the first.
Jobs are submitted through ``JobExecutionController.submit_local_jobs`` with a
local runner whose ``submit`` writes the job's CASTEP outputs into its directory
instead of starting CASTEP; each job is then finished through the window's
local-runner completion handler, so result dispatch, stored results, derived
structures and relaxed sites are those a real run produces. A message box
during the build fails it rather than hanging it, except the k-point
rescaling notice the build's supercell rows are known to raise.

The outputs are synthetic and tuned so that the application's analysis of
them reproduces the cited values. The pristine relaxation ends on the host
cell with sub-percent changes. The UEP source job is a model density of Co²⁺
and F⁻ Gaussian shells, not a CASTEP density. Each start relaxes as a
positive muon to one of three clusters (``POSITIVE_SITES``) with the energies
and neighbour displacements of the MuFinder paper, and as muonium to the
nearer of two (``MUONIUM_SITES``). The phonon run embeds the F–μ–F modes of
Möller *et al.* in the relaxed supercell. The NEB band is illustrative, and its caption
says so. ``tests/science/test_magnetic_fields.py`` pins the CoF₂ field values
and ``tests/docs/test_docs_synthetic.py`` pins the phonon, density and
single-point outputs. The UEP batch search and its ZPE run in the scene's
thread with seeded random starts, so every build finds the same minima.

``running_relaxation_project`` stops the pristine relaxation part-way through
its output for the geometry progress window. Reopening a project ends its local
jobs, since no local runner survives the application, so that scene puts the
job back in the running state the local runner keeps and points the progress
monitor at its ``.castep``. The legacy-importer scene scans a hand-authored
legacy folder of three CoF₂ runs in the documented layout (see
:doc:`/legacy_import`, "Research data boundary") and drops the scan's notice
that no keyword catalog is installed, as the other scenes hide it.

The iron example
~~~~~~~~~~~~~~~~

``examples/fe.py`` holds the Magnetic Fields scenes of the bcc iron worked
example (:doc:`/user_guide/examples/fe`). Each builder from ``fe_project`` saves
the two-atom conventional cell with 2.2 μB per Fe along c, a Magnetic Fields
form with the contact coupling that gives -1.11 T, every equivalent site and a
spherical sample, and the octahedral or tetrahedral site as a custom position;
no CASTEP job is involved. The scenes run the real ``muesr`` calculation with an
80 x 80 x 80 lattice sum, which keeps the cubic symmetry of the Lorentz sphere
exactly, so components that vanish by symmetry print as ``0``. The results
tables are made tall enough to list every equivalent site, the
demagnetizing-shape, measured-magnetisation and equivalent-site scenes grab the
page's ``Calculation`` card alone, and the workspaces scene saves one workspace
per site and shows the octahedral one beside the navigator.

The manganese silicide example
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``examples/mnsi.py`` holds the Magnetic Fields scenes of the incommensurate
helix worked example (:doc:`/user_guide/examples/mnsi`). Each builder from
``mnsi_project`` saves the eight-atom B20 cell, the helix, and a Magnetic Fields
form with the muon at the 4a site with every equivalent site, no demagnetizing
field and the contact coupling that reproduces the edges of the published
prediction; no CASTEP job is involved. The helix is built by the ``Add Magnetic
Preset`` dialog itself, driven through the choices the page describes: a
left-handed helix of 0.385 μB Mn moments along the first q-basis row, which
holds q = (0.0147, 0.0147, 0.0147). The scenes show the builder with its
checklist, the ``Propagation-Vector Basis`` panel alone, the component dialog's
``q from`` choice, the preset helper opened over a q basis with no component yet,
the results page and the distribution plot of the four sites, the
``Calculation Settings`` dialog with the approximant option on, and the finite
fields of the approximant at site II. The distribution is sampled on a grid of
phases, which is deterministic, so every render gives the same pixels. The
approximant scene lists 64 fields and takes about a quarter of a minute.

The α-N₂ example
~~~~~~~~~~~~~~~~

``examples/alpha_n2.py`` holds the scenes of the solid nitrogen worked example
(:doc:`/user_guide/examples/alpha_n2`), which the Quantum Corrections pages
show. ``host_project`` saves the eight-atom Pa-3 cell, built from the
molecule centres and bond directions, with a saved 2x2x2 supercell, an adopted
1200 eV cutoff and 4x4x4 grid, and six random starts with a fixed seed.
``results_project`` drives that project in a ``MainWindow`` as the CoF₂
results project does: the starts staged as positive-muon relaxations in the
supercell, one analysis workspace, clustered and pinned, and the pinned site's
phonon row configured through its dialog (ultrasoft pseudopotentials, a
``1e-10`` eV SCF tolerance) and run. ``configured_project`` stops before the
phonon job is submitted, for the editable configuration dialog.

Each start relaxes to the nearest image of the (0, 1/4, 1/4) site, where
``bridged_diatomic_cell`` turns the two molecules either side to point at the
muon, 1.29 Å from their near nitrogens. The phonon run is
``bridged_diatomic_phonon_text``: the complex vibrates as the five-atom chain
spring model of ``tests/support/phonon_fixtures.py``, whose constants were
tuned to the harmonic numbers of Gomilšek *et al.*, so the window's analysis
gives their zero-point energy, widths, ``w1`` and ``w2``; every other molecule
vibrates on its own, its stretch in a band at 2330 to 2350 cm⁻¹ and its
librations and translations below 100 cm⁻¹. ``synthetic.py`` copies the chain
constants, and ``tests/docs/test_docs_synthetic.py`` keeps the copy equal to
the fixture's and pins the geometry and the harmonic numbers. The scenes show
the host, the clustered relaxations, the displacements of the complex, the
Muon Sites summary, the configuration dialog, the completed result, and the
three tabs of the analysis window with the help of ``Muon Coupling``.

When the renderer runs
~~~~~~~~~~~~~~~~~~~~~~

``make screenshots`` always renders every screenshot. A full render ends by
touching ``docs/_generated/screenshots/.stamp``; a render of named screenshots
or of one example does not, and neither does a render that fails. ``make
docs`` depends on the stamp, and the stamp on every file of
``scripts/docs_screenshots``, ``scripts/render_docs_screenshots.py`` and
``src/mufinder``, so ``make docs`` renders only when one of them is newer
than the stamp, and a fresh checkout, such as a CI runner, renders once. Delete the stamp to force a
render from ``make docs``.

Adding a screenshot
~~~~~~~~~~~~~~~~~~~

1. Add an entry to the ``SCENES`` of the example whose project it shows,
   reusing its project builders and the ``setup`` and ``prepare`` steps
   where they fit. A screenshot that needs a new project adds a builder to
   that example; one that needs a new kind of host adds an example module and
   lists it in ``registry.py``.
2. Render it alone, ``python scripts/render_docs_screenshots.py <name>``
   (``--list`` prints the names), and look at the image: it must not show an
   error, an empty table or a clipped control.
3. Reference it from a page with a ``figure`` directive, an ``:alt:`` text and
   a one-line caption.

``tests/docs/test_docs_screenshots.py`` runs the renderer in subprocesses,
since importing ``render.py`` configures Qt and VTK for the whole process. It
reads the registry's names with ``--list`` and fails if a page references an
image the registry does not render, the registry renders one no page uses, or
an example module under ``examples/`` is missing from the registry. It then
renders each example with ``--example`` in its own process, with a timeout
proportional to its scene count; that process shares the example's scenes
between its ``--jobs`` workers, and ``pytest-xdist`` spreads the examples
across test workers.
``tests/docs/test_docs_synthetic.py`` reads the ``synthetic.py`` outputs back
through the application's CASTEP and geometry-progress parsers, and
``tests/docs/test_docs_images.py`` builds a one-page Sphinx project with the
``docs/conf.py`` image hook and checks the attributes it writes.

Shared test helpers
-------------------

Test inputs that several modules build come from one place. Use these rather
than a module-local copy; a module that needs different values passes them at
the call site or binds them once with ``functools.partial``.

- ``tests/support/structures.py``: ``host_structure(...)`` builds a
  ``Structure`` from a lattice (cubic edge or explicit vectors), species and
  fractional positions, returning fresh lists on every call. Presets:
  ``si_pair_host`` (two Si atoms, 4.0 Å cube by default), ``silicon_oxide_host``
  (O-Si-O in a 10 Å cube) and ``hydrogen_host`` (one H atom).
- ``tests/support/symmetry_hosts.py``: hosts that exercise the symmetry code.
- ``tests/support/jobs.py``: ``make_job(...)`` builds a bare ``Job``, or, given
  ``service`` and ``document``, registers one through ``ProjectService.add_job``.
- ``tests/support/cluster_ssh.py``: ``FakeSSH`` stands in for an
  ``SSHConnection`` whose remote filesystem is a dict;
  ``TimestampedFakeSSH`` adds remote mtimes and ``OutputUnavailableSSH`` fails
  every download.
- ``tests/support/qt.py``: ``pump_until(predicate, timeout=...)`` spins the
  event loop until a queued signal has been delivered;
  ``find_tree_item_by_route`` and ``find_item_by_route`` locate project-tree
  items by route.
- ``tests/support/main_window.py``: ``open_project_with_host`` and
  ``window_with_host`` open a project with a host structure in a
  ``MainWindow``; ``select_route`` selects a project-tree route;
  ``patch_continuation_cluster_dialog``, ``patch_continuation_cluster_runner``
  and ``add_cluster_input_job`` stand in for the cluster submission dialog and
  SSH runner; ``stage_quantum_row`` stages an eligible Quantum Corrections
  row, optionally with the legacy singlepoint job an older project holds.
- ``tests/support/controllers.py``: ``RecordingBackend`` and ``make_context``
  stand in for the submit backends and the ``WorkflowControllerContext`` a
  workflow controller is built from; ``FakeRowState``, ``no_artifacts``,
  ``completed_castep_result``, ``muon_host_document`` and ``job_with_status``
  build the row states, parsed outputs, host project and jobs the controller
  tests share; ``FakeRelaxationView``, ``stub_pristine_deps`` and
  ``pristine_sync_definition`` stand in for the Pristine Relaxation widget,
  dependencies and definition.
- ``tests/conftest.py`` fixtures: ``main_window`` gives a fresh ``MainWindow``
  that the per-test Qt cleanup closes, and ``project_with_host`` gives
  ``(window, service, document)`` with the default host structure loaded.

GUI-responsiveness trace logging
---------------------------------

``mufinder.services.perf_trace.PerfTrace`` times named stages (via a
``with trace.stage("name"):`` block) and logs one summary line per traced
operation through the standard ``logging`` module. It wraps the main
window's coalesced dashboard flush (one stage per dashboard view, plus the
autosave) and ``ProjectService.save_project`` (total duration plus
``entity_files_written`` and ``entity_files_skipped`` counters), both of which
run on the GUI thread and have been observed to stall the app when many cluster
jobs finish in one burst. ``save_project`` skips rewriting entity files whose
content is unchanged since the last save/load (content-hash dirty tracking), so
a healthy mature project reports a large ``skipped`` count and a small
``written`` count per autosave.

Each summary logs at ``WARNING`` when any stage exceeds 50 ms or the total
exceeds 200 ms, and at ``DEBUG`` otherwise. To see the per-stage breakdown
during local debugging, enable DEBUG logging for the relevant modules, e.g.:

.. code-block:: python

   import logging
   logging.getLogger("mufinder.app.main_window").setLevel(logging.DEBUG)
   logging.getLogger("mufinder.services.project_service").setLevel(logging.DEBUG)

Targeted local execution checks
-------------------------------

The local CASTEP execution feature is validated by targeted tests that cover:

- setup persistence and mpirun discovery behavior
- submission-dialog validation for cores and concurrency
- cutoff-table context-menu submission wiring
- shared dashboard row-button actions for resubmit/cancel/inspect/reset
- shared workflow-row submission planning for convergence, relaxation, and UEP workflows
- shared job execution lifecycle orchestration for local and remote runners
- local runner queue/cancel behavior
- CASTEP output parsing for success/failure cases, including zero-K energy and force extraction

``make test-fast`` runs these targeted checks (the ``FILES`` default in the
``Makefile``) during focused iteration; ``make test`` runs the full
coverage-enabled suite before merge.

Agent harness checks
--------------------

MuFinder keeps repository knowledge and package boundaries executable so future
agent runs can work from local context rather than hidden conversation state.

Run the harness checks after changing ``AGENTS.md``, docs indexes, package
boundaries, or shared architecture:

.. code-block:: bash

   make harness

The harness tests currently verify:

- ``AGENTS.md`` stays within 140 lines and points to the durable docs it depends
  on; ``CLAUDE.md`` imports it.
- Every ``mufinder.*`` path in the shared-homes table (:doc:`/architecture`,
  Shared workflow infrastructure) names a module or package under
  ``src/mufinder``; the check reads the file system and imports nothing.
- The Sphinx index includes the agent harness, architecture, quality, and
  compatibility documents, and ``docs/architecture.rst`` names every layer
  package.
- Lower package layers do not import higher package layers
  (``FORBIDDEN_LAYER_IMPORTS``), and no layer package outside ``mufinder.app``
  imports ``PySide6.QtWidgets`` or ``PySide6.QtGui``.
- The ``_on_submit_*`` / ``_materialize_*`` handlers on ``MainWindow`` match
  the pinned set exactly; the set may only shrink (see
  :doc:`/architecture`, Workflow controllers).
- No module in ``mufinder.app`` calls ``subplots_adjust``, and the app modules
  that build a ``FigureCanvasQTAgg`` directly rather than through
  ``create_plot_canvas`` match a pinned set that may only shrink (see
  :doc:`/architecture`, Plot canvas policy).
- The private methods that two or more workflow widget modules define match
  a pinned name-to-count table; a count may only fall and no new shared name
  may appear (see :doc:`/architecture`, Shared workflow infrastructure).
- The ``*_coverage.py`` test files match a pinned list that may only shrink;
  coverage gaps are closed in the module's own test file.
- ``.github/PULL_REQUEST_TEMPLATE.md`` keeps its section headings and
  ``AGENTS.md`` still points to it.
- Source, test, and docs trees do not track Python bytecode artifacts that can
  confuse packaging and review.

Automation policy
-----------------

The repository follows GitHub Flow, and its automation is split across
continuous integration and tag-driven publishing:

- Pull requests into ``main`` (and pushes to ``main`` after merge) run the full
  quality gates:

  - ``make lint``: ``ruff check`` over ``src``, ``tests``, ``scripts`` and the API example scripts in ``docs/api/examples``
  - ``make test-shard SHARD=n SHARDS=4 PYTEST_WORKERS=auto`` on four runners,
    each distributed with ``pytest-xdist`` using ``--dist worksteal``, then
    ``make coverage`` in the required ``tests`` job, which combines the shards'
    coverage data and enforces the threshold. ``tests`` fails if any shard
    fails.
  - ``make docs``: the screenshots, rendered once on the fresh runner, then
    the ``sphinx`` HTML build with ``-W`` enabled. The job uploads nothing.

- ``Preview Build`` is a manual GitHub Actions workflow that builds Windows
  and macOS artifacts from any branch, tag, or SHA without publishing a release
  and without running the full suite.

- ``Release Publish`` is triggered by pushing a ``vX.Y.Z`` tag (it can also be
  dispatched manually against an existing tag). It derives the version from the
  tag, builds the Windows and macOS binaries, and publishes the GitHub release.
  Bump the version and stamp ``CHANGELOG.md`` locally with
  ``semantic-release version`` (for example
  ``uvx "python-semantic-release>=10.5,<11" version``) before pushing the tag.
  Branch protection blocks pushing that version-bump commit to ``main``, so it
  goes through a ``release/vX.Y.Z`` pull request first:

  .. code-block:: bash

     git checkout main && git pull
     uvx "python-semantic-release>=10.5,<11" version   # bumps pyproject.toml + CHANGELOG.md, commits + tags locally
     git checkout -b release/vX.Y.Z                    # branch off the version bump commit
     git checkout main && git reset --hard origin/main  # restore local main
     git push origin release/vX.Y.Z                    # open a PR, wait for CI + merge
     git push origin vX.Y.Z                            # push the tag, which triggers Release Publish

- ``Pages`` rebuilds this Sphinx site, screenshots included, with ``make docs``
  on every push to ``main`` and deploys it to GitHub Pages. GitHub Pages keeps
  its own copy of a deployed site, so once the deployment succeeds the
  ``cleanup`` job deletes the run's uploaded ``github-pages`` artifact; it needs
  ``actions: write`` for that and nothing else. A failed deployment keeps its
  artifact so the deploy job can be re-run, and the upload's one-day retention
  removes it otherwise.

The repository's default ``GITHUB_TOKEN`` permission for workflows is
read-only. A workflow that needs to write (``Pages``, ``Release Publish``)
declares the exact permissions it needs in a ``permissions`` block; ``CI``
declares ``contents: read``. No workflow runs a coding agent on GitHub; agents
work locally and open pull requests like any other contributor.

Merges should be blocked when either gate fails.

Branch workflow
---------------

Pull requests are restricted to repository collaborators through the
repository's *Collaborators only* pull-request setting. Anyone can open an
issue. ``CONTRIBUTING.md`` states this policy for outside readers, and the
``.github/ISSUE_TEMPLATE/`` forms structure bug reports and feature requests.

The expected branch lifecycle is:

1. Branch short-lived ``feature/*`` (or ``fix/*``) branches off ``main``. Small
   features should merge within about a week; longer-lived branches are allowed
   when a feature genuinely needs it.
2. Open a pull request directly into ``main`` and let CI run the full quality
   gates; merges are blocked until they pass. PR bodies follow
   ``.github/PULL_REQUEST_TEMPLATE.md``; GitHub only injects the template into
   the web form, so agents opening PRs with ``gh pr create --body`` must read
   the file first. PR titles carry a Conventional Commits prefix, which
   ``semantic-release`` parses to derive version bumps and the changelog.
3. Use ``Preview Build`` to produce distributable artifacts from any branch when
   validation is needed.
4. Release by pushing a ``vX.Y.Z`` tag from ``main``; the ``Release Publish``
   workflow builds the binaries and publishes the GitHub release.
