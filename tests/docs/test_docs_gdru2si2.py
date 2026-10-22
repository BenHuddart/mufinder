"""The numbers the GdRu2Si2 worked example reports, read from the paper's real CASTEP outputs.

``docs/user_guide/examples/gdru2si2.rst`` quotes the barrier, the image count,
the zero-point energies and the entanglement witnesses that MuFinder's parsers
give for the outputs under ``tests/fixtures/gdru2si2`` (Huddart et al., Phys.
Rev. B 111, 054440 (2025)). These tests pin them, so a parser change that moves
a number on the page fails here first.
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from mufinder.castep_io.cell_parser import parse_cell_file
from mufinder.castep_io.ts_parser import parse_transition_state_file
from mufinder.science.quantum_phonons import (
    muon_mass_warning,
    muon_zpe,
    parse_phonon_path,
    w1_entanglement_witness,
)

FIXTURES = Path(__file__).parents[1] / "fixtures" / "gdru2si2"
NEB = FIXTURES / "GdRu2Si2_12_neb.ts"
SUPERCELL_REPEATS = np.array([2.0, 2.0, 1.0])
_IMAGE_HEADER = re.compile(r"^\s*(REA|TST|PRO)\s+(\d+)\s+\S+\s*$", re.MULTILINE)


def test_the_trimmed_band_keeps_the_header_the_endpoints_and_one_iteration() -> None:
    text = NEB.read_text(encoding="utf-8")
    headers = _IMAGE_HEADER.findall(text)

    assert text.startswith(" TSTYPE TSConfirmation\n")
    assert [label for label, _ in headers].count("REA") == 1
    assert [label for label, _ in headers].count("PRO") == 1
    assert {iteration for label, iteration in headers if label == "TST"} == {"112"}


def test_the_neb_band_from_site_1_to_site_2() -> None:
    path = parse_transition_state_file(NEB)
    energies = np.array([image.energy for image in path.images])
    peak = int(np.argmax(energies))

    assert len(path.images) == 42
    assert [image.label for image in path.images] == ["REA", *["TST"] * 40, "PRO"]
    assert path.barrier_height_eV == pytest.approx(0.4952, abs=5e-5)
    assert round(path.barrier_height_eV, 2) == 0.50
    # Site 2, the band's end, sits 0.283 eV above site 1; the reverse barrier is 0.21 eV.
    assert energies[-1] - energies[0] == pytest.approx(0.2827, abs=5e-5)
    assert energies[peak] - energies[-1] == pytest.approx(0.2125, abs=5e-5)
    # The transition state is image 23 of 42, at reaction coordinate 0.627.
    assert peak == 22
    assert path.images[peak].reaction_coordinate == pytest.approx(0.6266, abs=5e-5)


@pytest.mark.parametrize(
    ("number", "conventional"),
    [(1, (0.50, 0.00, 0.07)), (2, (0.21, 0.21, 0.16))],
)
def test_the_relaxed_muons_sit_at_the_table_si_sites(number: int, conventional: tuple[float, ...]) -> None:
    cell = parse_cell_file(FIXTURES / f"GdRu2Si2_site{number}-out.cell")
    muon = np.asarray(cell.fractional_positions[list(cell.species).index("H:mu")], dtype=float)

    assert len(cell.species) == 41
    assert np.allclose(muon * SUPERCELL_REPEATS, conventional, atol=0.005)


@pytest.mark.parametrize(
    ("number", "modes_cm", "zpe_eV", "w1"),
    [
        (1, (1136.3, 3225.0, 4632.8), 0.558, -0.0079),
        (2, (2129.8, 4019.8, 4396.9), 0.654, -0.0082),
    ],
)
def test_the_phonons_of_each_site(number: int, modes_cm: tuple[float, ...], zpe_eV: float, w1: float) -> None:
    calculation = parse_phonon_path(FIXTURES / f"GdRu2Si2_site{number}.phonon")
    zpe = muon_zpe(calculation)

    # The three most muon-like modes are the three highest of the 123.
    assert [mode.mode_index for mode in zpe.modes] == [121, 122, 123]
    assert [mode.frequency_cm for mode in zpe.modes] == pytest.approx(modes_cm, abs=0.05)
    assert zpe.total_eV == pytest.approx(zpe_eV, abs=5e-4)
    # The paper's witnesses, -0.0079 and -0.0082, to the four places it quotes.
    assert w1_entanglement_witness(calculation) == pytest.approx(w1, abs=5e-5)


def test_the_phonon_runs_used_a_muon_mass_the_mass_check_accepts() -> None:
    calculation = parse_phonon_path(FIXTURES / "GdRu2Si2_site1.phonon")
    muon = calculation.atoms[calculation.atom_index_for_species("H:mu") - 1]

    # The runs set SPECIES_MASS to 0.1138 amu, 0.33 % above the muon mass, so
    # their frequencies are 0.16 % low; MuFinder's check allows 1 %.
    assert muon.mass_amu == pytest.approx(0.1138)
    assert muon_mass_warning(calculation) is None
