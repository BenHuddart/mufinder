"""Golden-file rendering characterization tests.

These pin the exact rendered ``.cell`` / ``.param`` text produced from
representative inputs for every workflow, byte-for-byte, so later refactors of
the workflow / adapter / renderer layers cannot silently change CASTEP output.

The inputs are built by :mod:`tests.support.golden_inputs`; the same builders
generate the committed goldens under ``tests/data/golden_inputs/``. To
regenerate after an intentional change:

    python -m tests.support.golden_inputs

and eyeball the diff before committing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.support.golden_inputs import GOLDEN_DIR, iter_golden_cases

_CASES = list(iter_golden_cases())


@pytest.mark.parametrize(
    "workflow,job_name,cell_text,param_text",
    _CASES,
    ids=[f"{workflow}/{job_name}" for workflow, job_name, _c, _p in _CASES],
)
def test_rendered_inputs_match_golden(
    workflow: str, job_name: str, cell_text: str, param_text: str
) -> None:
    cell_golden = GOLDEN_DIR / workflow / f"{job_name}.cell"
    param_golden = GOLDEN_DIR / workflow / f"{job_name}.param"
    assert cell_golden.exists(), f"missing golden cell: {cell_golden} (run python -m tests.support.golden_inputs)"
    assert param_golden.exists(), f"missing golden param: {param_golden}"
    assert cell_text == cell_golden.read_text(encoding="utf-8")
    assert param_text == param_golden.read_text(encoding="utf-8")


def test_every_committed_golden_has_a_generating_case() -> None:
    # Guard against orphaned golden files that no builder produces (which would
    # silently stop being verified).
    expected = set()
    for workflow, job_name, _cell, _param in _CASES:
        expected.add(GOLDEN_DIR / workflow / f"{job_name}.cell")
        expected.add(GOLDEN_DIR / workflow / f"{job_name}.param")
    on_disk = set(Path(GOLDEN_DIR).rglob("*.cell")) | set(Path(GOLDEN_DIR).rglob("*.param"))
    assert on_disk == expected
