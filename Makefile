PYTHON ?= python

.PHONY: install-dev test docs quality

install-dev:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

docs:
	$(PYTHON) -m sphinx -W -b html docs docs/_build/html

quality: test docs
