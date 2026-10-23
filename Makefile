# The executable definition of MuFinder's local commands. README.md,
# docs/quality_and_automation.rst, the PR template and CI refer to these
# targets rather than spelling the commands out.

PYTHON ?= python

# The full suite runs in the foreground with at most 4 workers locally: each
# xdist worker loads Qt, and more workers can exhaust the machine. CI overrides
# this with PYTEST_WORKERS=auto. Workers steal queued tests from each other,
# so one slow test file does not leave a single worker finishing alone.
PYTEST_WORKERS ?= 4
PYTEST_ARGS ?=
PYTEST_XDIST = -n $(PYTEST_WORKERS) --dist worksteal

# CI splits the suite into SHARDS groups that run on separate runners.
# test-shard runs group SHARD and writes its coverage data to
# .coverage.shard-SHARD with no report and no gate; `make coverage` combines
# every shard's data and enforces the threshold from [tool.coverage.report].
SHARDS ?= 4
SHARD ?= 1

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
	tests/app/main_window

.PHONY: install test test-shard coverage test-fast harness screenshots screenshots-shard docs lint format quality

# Equivalent: uv sync --extra dev --extra scientific-backends
install:
	$(PYTHON) -m pip install -e ".[dev,scientific-backends]"

test:
	$(PYTHON) -m pytest $(PYTEST_XDIST) $(PYTEST_ARGS)

# The shards skip the tests that render the documentation screenshots: CI's
# docs-screenshots jobs render every one of them (see screenshots-shard).
test-shard:
	COVERAGE_FILE=.coverage.shard-$(SHARD) $(PYTHON) -m pytest $(PYTEST_XDIST) \
		--splits $(SHARDS) --group $(SHARD) --splitting-algorithm least_duration \
		-m "not screenshots" --cov-report= --cov-fail-under=0 $(PYTEST_ARGS)

coverage:
	$(PYTHON) -m coverage combine --keep
	$(PYTHON) -m coverage report

test-fast:
	$(PYTHON) -m pytest --no-cov $(FILES)

harness:
	$(PYTHON) -m pytest --no-cov tests/test_agent_harness.py

# The documentation screenshots are rendered from the application, not committed:
# scripts/render_docs_screenshots.py (the scripts/docs_screenshots package)
# writes them to docs/_generated/screenshots and touches its .stamp once every
# screenshot is written. `make screenshots` always renders; `make docs` renders
# only when the renderer or the application is newer than the stamp.
SCREENSHOT_STAMP = docs/_generated/screenshots/.stamp
SCREENSHOT_SOURCES = $(shell find scripts/docs_screenshots scripts/render_docs_screenshots.py \
	src/mufinder -type f -not -path '*/__pycache__/*' -not -name '*.pyc')

screenshots:
	$(PYTHON) scripts/render_docs_screenshots.py

# CI renders the screenshots as SHARDS shares on separate runners; this renders
# share SHARD and does not touch the stamp.
screenshots-shard:
	$(PYTHON) scripts/render_docs_screenshots.py --shard $(SHARD)/$(SHARDS)

$(SCREENSHOT_STAMP): $(SCREENSHOT_SOURCES)
	$(PYTHON) scripts/render_docs_screenshots.py

docs: $(SCREENSHOT_STAMP)
	$(PYTHON) -m sphinx -W -b html docs docs/_build/html

# ruff is installed by the dev extra, pinned to one minor version so local
# formatting matches CI. lint checks both the lint rules and the formatting;
# format rewrites the files in place.
RUFF_PATHS = src tests scripts docs/api/examples docs/conf.py packaging

lint:
	$(PYTHON) -m ruff check $(RUFF_PATHS)
	$(PYTHON) -m ruff format --check $(RUFF_PATHS)

format:
	$(PYTHON) -m ruff format $(RUFF_PATHS)

quality: lint test docs
