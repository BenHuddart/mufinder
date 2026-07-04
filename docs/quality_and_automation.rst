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

Install development dependencies (the ``scientific-backends`` extra is
required by the test suite):

.. code-block:: bash

   python -m pip install -e ".[dev,scientific-backends]"
   # or: uv sync --extra dev --extra scientific-backends

Run tests with coverage gate. Locally, cap parallelism at 4 workers and run
in the foreground — each xdist worker loads Qt, and ``-n auto`` can exhaust
the machine (CI uses ``-n auto``):

.. code-block:: bash

   python -m pytest -n 4 --dist load

Run only non-GUI tests during focused iteration:

.. code-block:: bash

   python -m pytest --no-cov -m "not gui"

Build docs with warning-as-error behavior:

.. code-block:: bash

   python -m sphinx -W -b html docs docs/_build/html

PySide test shutdown note
-------------------------

The pytest configuration in ``tests/conftest.py`` includes a final
``atexit`` process-exit hook. This is intentional: some macOS/PySide6 runs
can trigger a segmentation fault during late interpreter shutdown after tests
and coverage have already completed.

The hook records pytest's final exit status, flushes stdout/stderr, and exits
the process deterministically so CI and local runs report stable outcomes
without post-run crashes.

GUI-responsiveness trace logging
---------------------------------

``mufinder.services.perf_trace.PerfTrace`` times named stages (via a
``with trace.stage("name"):`` block) and logs one summary line per traced
operation through the standard ``logging`` module. It currently wraps
``MainWindow._flush_dashboard`` (one stage per dashboard view, plus the
autosave) and ``ProjectService.save_project`` (total duration and entity
file count), both of which run on the GUI thread and have been observed to
stall the app when many cluster jobs finish in one burst.

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

Run targeted checks during focused iteration:

.. code-block:: bash

   python -m pytest --no-cov \
     tests/test_castep_output_parser.py \
     tests/app/test_workflow_submission.py \
     tests/services/test_job_execution_controller.py \
     tests/services/test_local_runner.py \
     tests/app/test_submission_dialog.py \
     tests/app/test_castep_setup_dialog.py \
     tests/app/test_convergence_widget.py \
     tests/app/test_main_window_actions.py

Then run the full coverage-enabled suite before merge:

.. code-block:: bash

   python -m pytest -n 4 --dist load

Agent harness checks
--------------------

MuFinder keeps repository knowledge and package boundaries executable so future
agent runs can work from local context rather than hidden conversation state.

Run the harness checks after changing ``AGENTS.md``, docs indexes, package
boundaries, or shared architecture:

.. code-block:: bash

   python -m pytest --no-cov tests/test_agent_harness.py

The harness tests currently verify:

- ``AGENTS.md`` remains a short route map and points to durable docs.
- The Sphinx index includes the agent harness and architecture documents.
- Lower package layers do not import higher package layers.
- Source, test, and docs trees do not track Python bytecode artifacts that can
  confuse packaging and review.

Automation policy
-----------------

The repository follows GitHub Flow, and its automation is split across
continuous integration and tag-driven publishing:

- Pull requests into ``main`` (and pushes to ``main`` after merge) run the full
  quality gates:

  - ``pytest`` with coverage threshold enforcement, distributed with
    ``pytest-xdist`` using ``--dist load``
  - ``sphinx`` HTML build with ``-W`` enabled

- ``Preview Build`` is a manual GitHub Actions workflow that builds Windows
  and macOS artifacts from any branch, tag, or SHA without publishing a release
  and without running the full suite.

- ``Release Publish`` is triggered by pushing a ``vX.Y.Z`` tag (it can also be
  dispatched manually against an existing tag). It derives the version from the
  tag, builds the Windows and macOS binaries, and publishes the GitHub release.
  Bump the version and stamp ``CHANGELOG.md`` locally with
  ``semantic-release version`` before pushing the tag.

Merges should be blocked when either gate fails.

Branch workflow
---------------

The expected branch lifecycle is:

1. Branch short-lived ``feature/*`` (or ``fix/*``) branches off ``main``. Small
   features should merge within about a week; longer-lived branches are allowed
   when a feature genuinely needs it.
2. Open a pull request directly into ``main`` and let CI run the full quality
   gates; merges are blocked until they pass.
3. Use ``Preview Build`` to produce distributable artifacts from any branch when
   validation is needed.
4. Release by pushing a ``vX.Y.Z`` tag from ``main``; the ``Release Publish``
   workflow builds the binaries and publishes the GitHub release.
