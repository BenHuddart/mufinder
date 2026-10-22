"""The worked examples on the ``Using MuFinder as a library`` page run.

``docs/api/using_mufinder.rst`` includes each script under ``docs/api/examples``
verbatim, so these tests run the scripts as a reader would: in a fresh process,
from the repository root (the parsing example reads a fixture by relative
path), against the installed package. Each test checks one stable line of the
script's output.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = PROJECT_ROOT / "docs" / "api" / "examples"

# Each script runs in well under a second; the limit only stops a hung run.
RUN_TIMEOUT_S = 60


def _run_example(name: str) -> list[str]:
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / name)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return [line.strip() for line in completed.stdout.splitlines()]


def test_parse_castep_output_example_prints_the_final_energy() -> None:
    lines = _run_example("parse_castep_output.py")

    assert "Final energy: -340.682656 eV" in lines
    assert "Geometry converged: True" in lines


def test_generate_muon_sites_example_is_reproducible_from_its_seed() -> None:
    lines = _run_example("generate_muon_sites.py")

    assert "Generated 5 sites" in lines
    assert "Random Site 1  0.6394 0.0250 0.2750" in lines


def test_dipolar_field_example_matches_the_documented_iron_ferromagnet() -> None:
    pytest.importorskip("muesr")
    pytest.importorskip("mulfc")

    lines = _run_example("dipolar_field.py")

    # docs/user_guide/magnetic_fields_theory.rst, "Ferromagnet", octahedral site.
    assert "Dipolar Bz: -0.9261 T" in lines
    assert "Lorentz Bz: 0.7243 T" in lines
    assert "Total Bz: -0.2018 T" in lines
