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

The repository includes CI automation to run both quality gates for every
push and pull request:

- ``pytest`` with coverage threshold enforcement.
- ``sphinx`` HTML build with ``-W`` enabled.
- GitHub Pages deployment from ``main``.
- Tagged release builds for Windows and macOS artifacts.

Merges should be blocked when either gate fails.
