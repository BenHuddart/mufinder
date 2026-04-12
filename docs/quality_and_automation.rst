Quality and Automation
======================

Quality gates
-------------

All implemented and future functionality must satisfy these gates:

- Unit and integration tests must pass.
- Coverage must remain at or above 85 percent (enforced by pytest-cov).
- Sphinx documentation must build with warnings treated as errors.
- New functionality is incomplete until tests and docs are included.

Local commands
--------------

Install development dependencies:

.. code-block:: bash

   python -m pip install -e ".[dev]"

Run tests with coverage gate:

.. code-block:: bash

   python -m pytest

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

Targeted local execution checks
-------------------------------

The local CASTEP execution feature is validated by targeted tests that cover:

- setup persistence and mpirun discovery behavior
- submission-dialog validation for cores and concurrency
- cutoff-table context-menu submission wiring
- shared dashboard row-button actions for submit/cancel/inspect/reset
- local runner queue/cancel behavior
- CASTEP output parsing for success/failure cases, including zero-K energy and force extraction

Run targeted checks during focused iteration:

.. code-block:: bash

   python -m pytest --no-cov \
     tests/test_castep_output_parser.py \
     tests/services/test_local_runner.py \
     tests/app/test_submission_dialog.py \
     tests/app/test_castep_setup_dialog.py \
     tests/app/test_convergence_widget.py \
     tests/app/test_main_window_actions.py

Then run the full coverage-enabled suite before merge:

.. code-block:: bash

   python -m pytest

Automation policy
-----------------

The repository automation is split across development, release preparation,
and final publishing:

- Pull requests into ``main`` run the full quality gates:

  - ``pytest`` with coverage threshold enforcement
  - ``sphinx`` HTML build with ``-W`` enabled

- ``Preview Build`` is a manual GitHub Actions workflow that builds Windows
  and macOS artifacts from ``develop`` or a ``release/*`` branch without
  publishing a release and without running the full suite.

- ``Prepare Release`` is a manual workflow that cuts ``release/vX.Y.Z`` from
  ``develop``, applies the semantic version bump, updates ``CHANGELOG.md``,
  opens the pull request to ``main``, and enables auto-merge with a merge
  commit so the release lands as soon as the required checks pass. It expects a
  repository variable named ``RELEASE_APP_ID`` and a repository secret named
  ``RELEASE_APP_PRIVATE_KEY`` so the workflow can mint a short-lived GitHub App
  installation token.

- Pushes to ``main`` publish releases only when ``pyproject.toml`` contains a
  version newer than the latest ``v*`` tag. The publish flow creates the tag,
  uploads Windows and macOS binaries, and then merges ``main`` back into
  ``develop`` to keep version metadata aligned between the two branches. Once
  that sync succeeds, the merged ``release/vX.Y.Z`` branch is deleted
  automatically.

Merges should be blocked when either gate fails.

Branch workflow
---------------

The expected branch lifecycle is:

1. Day-to-day development on ``develop``.
2. Manual preview builds from ``develop`` whenever distributable artifacts are
   needed for validation.
3. Cut ``release/vX.Y.Z`` from ``develop`` once the branch is stable.
4. Let the release PR auto-merge into ``main`` once the full tests plus docs
   build pass.
5. Let the publish workflow tag and release from ``main``.
6. Merge ``main`` back into ``develop`` immediately after publish.
7. Let the publish workflow delete the merged ``release/vX.Y.Z`` branch.
