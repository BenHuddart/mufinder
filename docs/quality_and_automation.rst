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
  workers or fewer: each worker loads Qt, and more can exhaust the machine. CI
  runs ``make test PYTEST_WORKERS=auto``.
- ``make test-fast`` runs focused tests without coverage. With no arguments it
  runs the targeted execution checks listed below; ``make test-fast
  FILES="tests/science/test_supercell.py"`` runs the files you name.
- ``make harness`` runs the agent-harness structural tests.
- ``make docs`` builds this Sphinx site with warnings treated as errors.
- ``make lint`` runs ``ruff check`` over ``src`` and ``tests``.
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

CI runs on 2-vCPU runners, so the suite's total CPU time sets the length of
every CI run. Keep these in place when changing the test setup:

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
  phases are listed at the end of every CI log. Check that list when CI time
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
file name must be unique across the whole ``tests`` tree.

``tests/app/main_window`` holds the integration tests that drive
``MainWindow`` itself, one module per subject: the jobs dashboard, job
recovery, cluster and local submission, each workflow's actions, and the
project, structure and site-generation actions. A workflow controller's own
behaviour is tested through the controller in ``tests/app/test_workflow_controllers.py``;
a window test covers what only the window does, such as its signal wiring,
its dialogs and the real submission path.

``tests/docs`` checks the structure of the Sphinx sources and their citations,
never their wording. ``test_docs_build_inputs.py`` asserts that every page under
``docs/`` is reachable from a toctree in ``docs/index.rst`` and that each
literature DOI the user guides cite stays on its page; a new citation is added
to its ``CITED_DOIS`` table. Tests do not pin prose phrases, ``conf.py`` lines
or stylesheet values: whether a page describes the code correctly is checked in
review, and the ``-W`` build catches broken references and missing assets.

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
- ``tests/support/qt.py``: ``pump_until(predicate, timeout=...)`` spins the
  event loop until a queued signal has been delivered;
  ``find_tree_item_by_route`` and ``find_item_by_route`` locate project-tree
  items by route.
- ``tests/support/main_window.py``: ``open_project_with_host`` and
  ``window_with_host`` open a project with a host structure in a
  ``MainWindow``; ``select_route`` selects a project-tree route;
  ``patch_continuation_cluster_dialog``, ``patch_continuation_cluster_runner``
  and ``add_cluster_input_job`` stand in for the cluster submission dialog and
  SSH runner; ``stage_quantum_row_with_singlepoint`` stages a Quantum
  Corrections row whose stage-1 singlepoint has finished.
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
- Every ``mufinder.*`` path in the shared-homes table (:doc:`architecture`,
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
  :doc:`architecture`, Workflow controllers).
- No module in ``mufinder.app`` calls ``subplots_adjust``, and the app modules
  that build a ``FigureCanvasQTAgg`` directly rather than through
  ``create_plot_canvas`` match a pinned set that may only shrink (see
  :doc:`architecture`, Plot canvas policy).
- The private methods that two or more workflow widget modules define match
  a pinned name-to-count table; a count may only fall and no new shared name
  may appear (see :doc:`architecture`, Shared workflow infrastructure).
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

  - ``make test PYTEST_WORKERS=auto``: ``pytest`` with coverage threshold
    enforcement, distributed with ``pytest-xdist`` using ``--dist load``
  - ``make docs``: ``sphinx`` HTML build with ``-W`` enabled

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

- ``Pages`` rebuilds this Sphinx site (with ``-W``) on every push to ``main``
  and deploys it to GitHub Pages.

- ``Claude Code`` (``.github/workflows/claude.yml``) runs the Claude Code agent
  when an issue, pull-request comment, or review mentions ``@claude``. It is
  not a quality gate.

Merges should be blocked when either gate fails.

Branch workflow
---------------

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
