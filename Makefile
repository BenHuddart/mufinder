# The executable definition of MuFinder's local commands. README.md,
# docs/quality_and_automation.rst, the PR template and CI refer to these
# targets rather than spelling the commands out.

PYTHON ?= python

# The full suite runs in the foreground with at most 4 workers locally: each
# xdist worker loads Qt, and more workers can exhaust the machine. CI overrides
# this with PYTEST_WORKERS=auto.
PYTEST_WORKERS ?= 4
PYTEST_ARGS ?=

# Default focused set for test-fast: CASTEP output parsing, submission
# planning, job lifecycle, the local runner and the main submission dialogs.
# Override with FILES="tests/path/test_x.py ...".
FILES ?= \
	tests/castep_io/test_output_parser.py \
	tests/app/test_workflow_submission.py \
	tests/services/test_job_execution_controller.py \
	tests/services/test_local_runner.py \
	tests/app/test_submission_dialog.py \
	tests/app/test_castep_setup_dialog.py \
	tests/app/test_convergence_widget.py \
	tests/app/test_main_window_actions.py

.PHONY: install test test-fast harness docs lint quality

# Equivalent: uv sync --extra dev --extra scientific-backends
install:
	$(PYTHON) -m pip install -e ".[dev,scientific-backends]"

test:
	$(PYTHON) -m pytest -n $(PYTEST_WORKERS) --dist load $(PYTEST_ARGS)

test-fast:
	$(PYTHON) -m pytest --no-cov $(FILES)

harness:
	$(PYTHON) -m pytest --no-cov tests/test_agent_harness.py

docs:
	$(PYTHON) -m sphinx -W -b html docs docs/_build/html

# ruff is installed by the dev extra (added alongside the ruff configuration).
lint:
	$(PYTHON) -m ruff check src tests

quality: lint test docs
