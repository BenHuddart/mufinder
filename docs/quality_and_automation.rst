Quality and Automation
======================

Quality gates
-------------

All implemented and future functionality must satisfy these gates:

- Unit and integration tests must pass.
- Lint and formatting checks must pass (``make lint``).
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
  way, as four shards. The shards deselect the tests marked ``screenshots``,
  which render the documentation screenshots, since CI renders those in its
  own jobs.
- ``make test-fast`` runs focused tests without coverage. With no arguments it
  runs the targeted execution checks listed below; ``make test-fast
  FILES="tests/science/test_supercell.py"`` runs the files you name.
- ``make harness`` runs the agent-harness structural tests.
- ``make screenshots`` renders the documentation screenshots into
  ``docs/_generated/screenshots`` (see `Documentation screenshots`_), and
  ``make screenshots-shard SHARD=n SHARDS=m`` renders share ``n`` of ``m``.
- ``make docs`` renders the screenshots when they are out of date (see `When
  the renderer runs`_), then builds this Sphinx site with warnings treated as
  errors.
- ``make lint`` runs ``ruff check`` and ``ruff format --check`` over ``src``,
  ``tests``, ``scripts``, the API example scripts in ``docs/api/examples``,
  ``docs/conf.py`` and ``packaging`` (see `Code formatting`_).
- ``make format`` applies ``ruff format`` to the same files.
- ``make quality`` runs ``lint``, ``test`` and ``docs``, the gates CI enforces.

To skip the Qt widget tests while iterating, select by marker with
``python -m pytest --no-cov -m "not gui"``. ``tests/conftest.py`` marks every
test under ``tests/app/`` or using the ``qapp`` fixture as ``gui`` and every
other test as ``unit``.

Code formatting
---------------

``ruff format`` owns the layout of every Python file that ``make lint`` checks.
It runs with its defaults and a line length of 100, set in ``[tool.ruff]`` in
``pyproject.toml``. Run ``make format`` before committing: ``make lint`` fails on
any file the formatter would change, and CI runs ``make lint``. Layout is not
hand-tuned and is not a review topic.

The ``dev`` extra pins ruff to one minor version, because the formatter's style
can change between minor releases and a different version would reformat files
that CI accepts. CI installs the exact version in ``uv.lock``; the pin keeps a
``pip`` install (``make install``) on the same minor version. Patch releases do
not change the formatter's style, so ``uv lock --upgrade-package ruff`` within
the pinned minor version can ride along with any change.

Moving to a new minor version is a pull request of its own:

1. Raise the range in the ``dev`` extra to the next minor version, for example
   ``ruff>=0.18,<0.19``, and run ``uv lock --upgrade-package ruff``.
2. Run ``make format``, then ``make lint``, and fix anything the new version's
   lint rules report.
3. Commit the version change, then the ``make format`` output alone in a
   ``style:`` commit.
4. If the formatter changed any file, add the ``style:`` commit's hash to
   ``.git-blame-ignore-revs`` in a third commit, and merge the pull request with
   a merge commit. If it changed nothing, squash as usual.

Open the pull request when few others are open: every branch cut before it
needs the update step at the end of this section.

The formatter removes the indentation that every line of a docstring after the
summary shares. An indented example keeps its indentation only when a line of
prose at the docstring's own indentation sits beside it, such as ``Usage:``
before an indented command.

``.git-blame-ignore-revs`` lists the commits that only reformat code. GitHub's
blame view skips them; to do the same locally, run this once per clone::

    git config blame.ignoreRevsFile .git-blame-ignore-revs

A pull request that reformats code is merged with a merge commit rather than
squashed, so the reformatting commit keeps the hash listed in
``.git-blame-ignore-revs``.

To bring a branch cut before a reformat up to date, run ``make format`` on the
branch and commit the result, then merge ``main``. Both sides then carry the
same layout, so the only conflicts left are real ones.

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
  the whole suite exactly as in ``make test``. Each shard uploads its
  ``.coverage.shard-n`` file as a ``coverage-shard-n`` artifact, and the
  ``tests`` job deletes them once the gate passes (see `Automation policy`_).
- **Work-stealing workers.** ``make test`` and ``make test-shard`` run
  ``pytest-xdist`` with ``--dist worksteal``: an idle worker takes queued tests
  from a busy one, so a handful of slow tests at the end does not leave one
  worker running alone.
- **``uv`` installs.** CI installs from ``uv.lock`` with
  ``uv sync --locked`` and a cached ``uv`` download store, which takes seconds
  where ``pip`` took close to a minute per job. ``--locked`` fails the job when
  ``uv.lock`` does not match ``pyproject.toml``, so a stale lock cannot merge;
  run ``uv lock`` and commit the result to fix it.

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
  supercell (``triatomic_phonon_text``) and legacy-run outputs. It imports no Qt, so tests import it directly.
- ``examples/`` holds one module per example project. Each defines its
  project builders, the scene steps only its scenes use, and ``SCENES``, a
  dictionary from screenshot name to ``Scene``.
- ``shards.py`` shares the screenshots out between render workers and CI
  shards (see `Sharing the render`_). It imports no Qt, so tests import it
  directly. ``durations.json`` beside it records each screenshot's seconds.
- ``watchdog.py`` gives each scene a time limit (``scene_deadline``). A scene
  that runs past it ends the renderer with exit status 3: stderr names the
  scene and holds the stack of every thread, and the renderer's scratch
  directory is removed first. A hang that holds the GIL is caught 30 seconds
  later by ``faulthandler``, which dumps the same stacks under a ``Timeout``
  heading and leaves the scratch directory behind. It imports no Qt, so tests
  import it directly.
- ``registry.py`` lists the example modules in render order as ``EXAMPLES``
  and merges their scenes into ``SCREENSHOTS``. A screenshot name belongs to
  exactly one example.
- ``__main__.py`` is the command line. With no arguments it renders every
  screenshot; ``--example NAME`` renders one example's scenes, names render
  those screenshots, ``--list`` prints the names it would render, and
  ``--output`` writes somewhere other than ``docs/_generated/screenshots``.
  ``--shard K/N`` narrows the selection to the ``K``-th of ``N`` shares.
  ``--jobs N`` (default: four, or fewer CPUs) splits the selection into ``N``
  shares the same way and renders each in a worker process, a run with names
  of its own share; ``--jobs 1`` renders in one process.
  ``--store-durations`` renders in one process and records each screenshot's
  seconds in ``durations.json``.
  ``--scene-timeout SECONDS`` (default: 300, at most 86400) is each scene's
  time limit, its project's build included; worker processes inherit it. Like the
  application, the renderer starts ``mufinder.app.gui_garbage_collector``, so
  Qt objects are freed on the GUI thread only.

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
  grabbing only a ``region`` of it, such as one panel or a rectangle around
  one row) or
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

The examples
~~~~~~~~~~~~

Five examples supply the screenshots, one module each, listed in render order.
Each page of :doc:`/user_guide/examples/cof2`, :doc:`/user_guide/examples/fe`,
:doc:`/user_guide/examples/mnsi`, :doc:`/user_guide/examples/alpha_n2` and
:doc:`/user_guide/examples/gdru2si2` tells the story its scenes illustrate and
cites the values they reproduce, and the workflow pages and the
:doc:`/getting_started/first_project` tutorial reuse the same scenes.

- ``cof2.py``: rutile CoF₂ through every stage from the host structure to the
  local field. Its CASTEP outputs are synthetic, tuned to published values. It
  holds every scene of the tutorial and of the project home, host structure,
  supercell, convergence, pristine relaxation, UEP, site generation,
  relaxation analysis, comparison table, legacy import, journal and
  job-running pages, plus the CoF₂ scenes of the Magnetic Fields page (the
  antiferromagnet, its dipolar tensor and its nuclear relaxation).
- ``fe.py``: bcc iron, the ferromagnet scenes of the Magnetic Fields page and
  its theory page.
- ``mnsi.py``: the MnSi helix, the incommensurate scenes of the Magnetic Fields
  page and its theory page.
- ``alpha_n2.py``: solid nitrogen, every scene of the Quantum Corrections page
  and its theory page, from real CASTEP outputs.
- ``gdru2si2.py``: GdRu₂Si₂, every scene of the Energy Barriers page, from
  real CASTEP outputs.

A scene belongs to exactly one module; to find the one behind a figure, look
the figure's name up in that module's ``SCENES`` table, or run
``python scripts/render_docs_screenshots.py --list --example <name>``.

The iron and MnSi projects run no CASTEP job: a host structure and a magnetic
model are enough for the real ``muesr`` field calculation the scenes make. The
CoF₂, α-N₂ and GdRu₂Si₂ ``results_project`` builders drive a ``MainWindow``
through the stages as its buttons do, with a local runner that writes each job's
outputs instead of starting CASTEP.

The CoF₂ example
^^^^^^^^^^^^^^^^

``examples/cof2.py`` holds every scene the other four do not. Its host is rutile CoF₂ with
antiparallel spins on the two Co, ``PBE`` as the project functional and a saved
2x2x2 supercell.

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
to 4c paths on the Energy Barriers page.
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
Möller *et al.* in the relaxed supercell. ``tests/science/test_magnetic_fields.py`` pins the CoF₂ field values
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
^^^^^^^^^^^^^^^^

``examples/fe.py`` holds the Magnetic Fields scenes of the bcc iron worked
example (:doc:`/user_guide/examples/fe`). Each builder from ``fe_project`` saves
the two-atom conventional cell with 2.2 μB per Fe along c, a Magnetic Fields
form with the contact coupling that gives -1.11 T, every equivalent site and a
spherical sample, and the octahedral or tetrahedral site as a custom position.
The scenes run the ``muesr`` calculation with an 80 x 80 x 80 lattice sum, which keeps the cubic symmetry of the Lorentz sphere
exactly, so components that vanish by symmetry print as ``0``. The results
tables are made tall enough to list every equivalent site, the
demagnetizing-shape, measured-magnetisation and equivalent-site scenes grab the
page's ``Calculation`` card alone, and the workspaces scene saves one workspace
per site and shows the octahedral one beside the navigator.

The manganese silicide example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``examples/mnsi.py`` holds the Magnetic Fields scenes of the incommensurate
helix worked example (:doc:`/user_guide/examples/mnsi`). Each builder from
``mnsi_project`` saves the eight-atom B20 cell, the helix, and a Magnetic Fields
form with the muon at the 4a site with every equivalent site, no demagnetizing
field and the contact coupling that reproduces the edges of the published
prediction. The helix is built by the ``Add Magnetic
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
^^^^^^^^^^^^^^^^

``examples/alpha_n2.py`` holds the scenes of the solid nitrogen worked example
(:doc:`/user_guide/examples/alpha_n2`), which the Quantum Corrections pages
show. Its outputs are real: the relaxed muon supercell and its Γ-point phonon
run from the data set of Gomilšek *et al.* (2023), kept unchanged under
``tests/fixtures/alpha_n2`` with a ``README.md`` naming their source. The data
set holds no log of the relaxation, so only that ``.castep`` is synthesised,
with ``castep_text`` around the dispersion-corrected energy of the undisplaced
cell; the phonon job gets the real ``.castep`` and ``.phonon``.

``host_project`` saves the eight-atom Pa-3 cell of the calculation, its edge
half the relaxed supercell's and its x from the bond of the molecules away from
the muon, with a saved 2x2x2 supercell, the paper's 1200 eV and 2x2x2 grid set
by hand as the project defaults, and one manual start at (0, 1/4, 1/4).
``results_project`` drives that project in a ``MainWindow`` as the CoF₂
results project does: the start staged as a positive-muon relaxation in the
supercell whose local runner writes the real relaxed cell, one analysis
workspace clustered and pinned, and the pinned site's phonon row configured
through its dialog (ultrasoft pseudopotentials, a ``1e-10`` eV SCF tolerance)
and run. ``configured_project`` stops before the phonon job is submitted, for
the editable configuration dialog. The 3D viewers of the displacements and the
phonon modes look down a, in which plane the [011] complex lies.
``tests/docs/test_docs_alpha_n2.py`` checks the fixtures against the data
set's checksums and pins the geometry, the displacements, the zero-point
energies, mode frequencies, widths and witnesses that the pages report. The
scenes show the host, the relaxation workspace, the displacements, the Muon
Sites summary, the configuration dialog, the completed result, and the three
tabs of the analysis window with the help of ``Muon Coupling``.

The GdRu₂Si₂ example
^^^^^^^^^^^^^^^^^^^^

``examples/gdru2si2.py`` holds the Energy Barriers worked example
(:doc:`/user_guide/examples/gdru2si2`), and unlike the other examples its
outputs are real: the relaxed supercells, the NEB band and the Γ-point phonons
of Huddart *et al.* (2025), kept under ``tests/fixtures/gdru2si2`` with a
``README.md`` naming their source. The NEB band is the data set's
``_farm001.ts`` trimmed to its header, its endpoints and the images of its last
iteration, the only images the ``.ts`` parser reads. Only the ``.castep`` logs
are synthesised, with ``castep_text`` around the real final energies.

``generation_project`` saves the ten-atom conventional cell with a
ferromagnetic spin on each Gd, a 2x2x1 supercell, 1000 eV and a 5x5x5 grid of
the supercell set by hand as the project defaults, and the random starts of the
paper's generation rules (0.5 Å between starts and their symmetry equivalents,
1.0 Å from atoms) with a fixed seed. ``results_project`` adds the two Table SI
sites as custom positions and drives the window as the CoF₂ results project
does: two spin-polarised positive-muon relaxations in the supercell whose
local runner writes the real relaxed cell of the nearer site, one analysis
workspace clustered and pinned, the phonon jobs of both sites with 4 farms and
the real ``.phonon`` files, and on the Energy Barriers page the paths from site
1 to itself up to 3 Å and from site 1 to site 2, the shortest of which is
configured in its NEB dialog (40 images, 4 farms, 0.08 eV/Å, 3x3x3) and run with
the trimmed band written as ``<seed>_farm001.ts``. ``configured_project``
stops before that NEB job is submitted, for the configuration dialog and its
``Task farms`` row, which a ``region`` rectangle grabs alone. The 3D viewers of
the path preview and the transition state look down a, where the muon stands
clear of the atom columns. ``tests/docs/test_docs_gdru2si2.py`` pins the
barrier, the image count, the zero-point energies, the witnesses and the site
positions that the page reports.

Sharing the render
~~~~~~~~~~~~~~~~~~

A screenshot's cost is its window, its drive and its grab, about three
seconds on a developer machine; the first screenshot of a project in a process
also builds that project. ``shards.partition`` weighs each screenshot by its
seconds in ``durations.json``, or by the median of the recorded ones when it
has none, and at least 0.1 s. It keeps the screenshots of one project
together, so each share builds a project once, and splits a project's
screenshots into consecutive pieces only when they alone weigh more than an
even share. The pieces go,
heaviest first, to the lightest share, and each share renders in registry
order. Only the ratios between the durations matter, so they are recorded
from a single-process run on any machine:
``python scripts/render_docs_screenshots.py --store-durations`` rewrites
every entry, and naming screenshots records only those and keeps the rest.

When the renderer runs
~~~~~~~~~~~~~~~~~~~~~~

``make screenshots`` always renders every screenshot. A full render ends by
touching ``docs/_generated/screenshots/.stamp``; a render of named screenshots
or of one example does not, and neither does a render that fails. ``make
docs`` depends on the stamp, and the stamp on every file of
``scripts/docs_screenshots``, ``scripts/render_docs_screenshots.py`` and
``src/mufinder``, so ``make docs`` renders only when one of them is newer
than the stamp, and a fresh checkout renders once. Delete the stamp to force a
render from ``make docs``.

In ``.github/workflows/ci.yml`` the screenshots render in four
``docs-screenshots`` jobs on separate runners, one per share, since a single
runner's four CPUs limit how fast its workers can render. Each job restores its
share of ``docs/_generated/screenshots`` from the GitHub Actions cache, keyed
on the share and on a hash of ``scripts/docs_screenshots``,
``scripts/render_docs_screenshots.py``, ``src/mufinder``, ``tests/fixtures`` and
``uv.lock``. On a miss it runs ``make screenshots-shard`` and saves the images
under that key; on a hit it installs nothing. Either way it uploads its share
as an artifact. The ``docs`` job waits for all four, downloads the shares into
``docs/_generated/screenshots`` and touches ``.stamp``, so ``make docs`` runs
Sphinx alone. A pull request that changes only pages therefore renders
nothing. ``tests/docs/test_docs_screenshots.py`` checks that the four shares
together render every screenshot exactly once. Once ``make docs`` passes, the
``docs`` job deletes the ``docs-screenshots`` artifacts (see `Automation policy`_).

A ``docs-screenshots`` job stops after 20 minutes and the Pages build after 30,
so a hang outside a scene, which ``--scene-timeout`` does not cover, fails the
run instead of holding a runner for hours.

The build job of ``.github/workflows/pages.yml`` restores the whole of
``docs/_generated/screenshots`` from one cache entry under the same hash. On a
hit it touches ``.stamp`` and ``make docs`` skips the render; on a miss ``make
docs`` renders every screenshot and the job saves them under that key.

Adding a screenshot
~~~~~~~~~~~~~~~~~~~

1. Add an entry to the ``SCENES`` of the example whose project it shows,
   reusing its project builders and the ``setup`` and ``prepare`` steps
   where they fit. A screenshot that needs a new project adds a builder to
   that example; one that needs a new kind of host adds an example module and
   lists it in ``registry.py``.
2. Render it alone and record its duration,
   ``python scripts/render_docs_screenshots.py --store-durations <name>``
   (``--list`` prints the names), and look at the image: it must not show an
   error, an empty table or a clipped control. Commit the updated
   ``durations.json``.
3. Reference it from a page with a ``figure`` directive, an ``:alt:`` text and
   a one-line caption.

``tests/docs/test_docs_screenshots.py`` runs the renderer in subprocesses,
since importing ``render.py`` configures Qt and VTK for the whole process. It
reads the registry's names with ``--list`` and fails if a page references an
image the registry does not render, the registry renders one no page uses, or
an example module under ``examples/`` is missing from the registry. It also
fails if ``durations.json`` does not list exactly the registry's screenshots,
in registry order, or if a CI share is empty. It then renders each example
with ``--example`` in its own process, with a timeout proportional to its
scene count; that process shares the example's scenes between its ``--jobs``
workers, and ``pytest-xdist`` spreads the examples across test workers. These
render tests carry the ``screenshots`` marker, which ``make test`` runs and
``make test-shard`` deselects. The renderer refuses a grab of 10 kB or less as
blank, so CI's render jobs catch one too.
``tests/docs/test_docs_screenshot_shards.py`` checks how ``shards.py`` splits
and balances the screenshots and merges recorded durations.
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

Native thread pools
-------------------

OpenBLAS, MKL and OpenMP start one worker thread per core unless told
otherwise. MuFinder's linear algebra is small (eigensolves of a few hundred
rows, matrix products in matching, symmetry and phonon analysis), and on a
many-core machine whose other cores are busy, for example with a local CASTEP
run, a full-size pool spends far longer waiting for its threads than
computing. ``mufinder.main.run`` therefore calls
``mufinder.native_threads.cap_native_thread_pools`` as its first step, before
anything imports NumPy (the libraries read the variables once, when they load):

- ``OPENBLAS_NUM_THREADS``, ``OMP_NUM_THREADS`` and ``MKL_NUM_THREADS`` are each
  set to ``4``;
- a variable the user's environment already sets to a non-empty value is kept;
- the cap is MuFinder's own: the local runner removes the variables it set
  before starting CASTEP (:doc:`/job_lifecycle`, Local runner behavior).

Setting environment variables is cheap, so nothing heavyweight runs before the
splash screen. A computation that needs more threads raises the limit locally
(for example with ``threadpoolctl.threadpool_limits``) instead of changing the
startup cap.

One 401 × 401 ``numpy.linalg.eigh`` (median of five) on a 20-core machine, with
other cores kept busy by spinning processes:

.. list-table::
   :header-rows: 1

   * - Pool size
     - Idle
     - 4 cores busy
     - 16 cores busy
     - 20 cores busy
   * - 20 (library default)
     - 7.7 ms
     - 0.98 s
     - 2.1 s
     - 2.5 s
   * - 4 (MuFinder's cap)
     - 6.3 ms
     - 9.2 ms
     - 18 ms
     - 0.16 s
   * - 1
     - 9.9 ms
     - 9.9 ms
     - 10 ms
     - 24 ms

Quantum Corrections' analysis window builds only the tab it opens on
(``Phonon Modes``) and refreshes its structure viewer once; the other tabs are
built the first time they are shown. Opening the window for the GdRu₂Si₂ site 1
phonon fixture (``tests/fixtures/gdru2si2``, 41 atoms; median of five opens,
construct, show and process events):

.. list-table::
   :header-rows: 1

   * - Build
     - Idle
     - 4 cores busy
   * - Every tab, default pools, two viewer refreshes
     - 1.59 s
     - 5.1 s
   * - Shown tab only, capped pools, one viewer refresh
     - 0.78 s
     - 0.78 s

What remains is the structure viewer: creating it and its first
``set_structure``.

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
- No module in ``mufinder.app`` calls ``QMessageBox.question`` or uses a
  ``QMessageBox.StandardButton.Yes`` / ``No`` button (``YesToAll`` and
  ``NoToAll`` included): confirmations go through ``confirm_destructive`` or
  ``confirm_choice`` in ``mufinder.app.workflow_table_utils``, whose buttons
  name their action.
- The private methods that two or more workflow widget modules define match
  a pinned name-to-count table; a count may only fall and no new shared name
  may appear (see :doc:`/architecture`, Shared workflow infrastructure).
- No module in ``src/mufinder`` or ``scripts`` other than
  ``mufinder.domain.constants`` types out a physical constant: a number within
  1e-4 (relative) of a CODATA constant or a conversion built from them fails the
  check, which names the constant to import. Powers of ten are skipped, because
  they double as tolerances and SI prefixes, and
  ``mufinder.persistence.migrations`` is allowed, because a migration must
  reproduce the value older projects were written with. Tests keep their own
  reference values, typed independently in ``tests/support/codata.py``.
- No module in ``src/mufinder`` or ``scripts`` defines, at module level, a name
  that ``mufinder.domain.constants`` defines or one of the names that held a
  constant before the home existed (``MU0``, ``MU0_OVER_4PI``, ``MU_B``,
  ``MU_N``, ``HBAR``, ``ANGSTROM_TO_M``, ...). This catches a redefinition the
  value check cannot see, such as ``MU0_OVER_4PI = 1e-7``.
- Only ``mufinder.domain.constants`` imports ``scipy.constants`` or ``ase.units``.
- The ``*_coverage.py`` test files match a pinned list that may only shrink;
  coverage gaps are closed in the module's own test file.
- ``.github/PULL_REQUEST_TEMPLATE.md`` keeps its section headings and
  ``AGENTS.md`` still points to it.
- Source, test, and docs trees do not track Python bytecode artifacts that can
  confuse packaging and review.

A SciPy release that adopts a newer CODATA set changes the constants, the muon
mass and gyromagnetic ratio written to CASTEP inputs, and every result built
from them. Moving to it is a change of its own: raise the ``scipy`` floor in
``pyproject.toml`` to that release, update ``tests/support/codata.py`` to the
new recommended values, regenerate the golden inputs with
``python -m tests.support.golden_inputs``, and state in the pull request that
rendered inputs change. Loosening the tests instead would hide the change.

Automation policy
-----------------

The repository follows GitHub Flow, and its automation is split across
continuous integration and tag-driven publishing:

- Pull requests into ``main`` (and pushes to ``main`` after merge) run the full
  quality gates:

  - ``make lint``: ``ruff check`` and ``ruff format --check`` (see `Code
    formatting`_)
  - ``make test-shard SHARD=n SHARDS=4 PYTEST_WORKERS=auto`` on four runners,
    each distributed with ``pytest-xdist`` using ``--dist worksteal``, then
    ``make coverage`` in the required ``tests`` job, which combines the shards'
    coverage data and enforces the threshold. ``tests`` fails if any shard
    fails.
  - ``make screenshots-shard SHARD=n SHARDS=4`` in four ``docs-screenshots``
    jobs, each skipped when its share is cached (see `When the renderer
    runs`_), then ``make docs`` in the required ``docs`` job: the ``sphinx``
    HTML build with ``-W`` enabled, over the shares the four jobs uploaded.
    ``docs`` fails if any share fails.

- ``Preview Build`` is a manual GitHub Actions workflow that builds Windows
  and macOS artifacts from any branch, tag, or SHA without publishing a release
  and without running the full suite.

- ``Release Publish`` is triggered by pushing a ``vX.Y.Z`` tag (it can also be
  dispatched manually against an existing tag). It derives the version from the
  tag, builds the Windows and macOS binaries, and publishes the GitHub release.
  Bump the version and stamp ``CHANGELOG.md`` locally with
  ``semantic-release version`` (for example
  ``uvx "python-semantic-release>=10.5,<11" version``) before pushing the tag.
  Its ``build_command`` runs ``uv lock --upgrade-package mufinder`` and stages
  ``uv.lock``, so the version-bump commit updates the lock's ``mufinder``
  version alongside ``pyproject.toml`` and leaves every dependency pin as it was.
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

Artifacts carry files between the jobs of one run and are not kept after it.
In ``CI`` the ``tests`` and ``docs`` jobs delete the ``coverage-shard-*`` and
``docs-screenshots-*`` artifacts through the composite action
``.github/actions/delete-run-artifacts``, which deletes the current run's
artifacts whose names match a regular expression. They delete only when they
succeed, so "Re-run failed jobs" can still download them. ``Release Publish``
deletes the installers once they are attached to the release, with its own
inline step: it checks out the release's source, which for an older tag does
not contain the composite action. An artifact that is not deleted, from a
failed run or from a pull request opened from a fork or by Dependabot (whose
token is read-only), expires under its ``retention-days``: one day for the CI
shards and three for preview builds.

The repository's default ``GITHUB_TOKEN`` permission for workflows is
read-only. A workflow that needs to write (``Pages``, ``Release Publish``)
declares the exact permissions it needs in a ``permissions`` block; ``CI``
declares ``contents: read``, and its ``tests`` and ``docs`` jobs add
``actions: write`` to delete their artifacts. No workflow runs a coding agent on GitHub; agents
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
