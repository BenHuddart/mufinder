from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.castep_io.magres import (
    MagresInputError,
    hyperfine_tensor_for_species,
    parse_magres_hyperfine,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "hyperfine"


def test_parses_muon_and_host_tensors_from_a_castep_run() -> None:
    # Cut from a CASTEP 26.11 run: bcc Fe 2x2x2 with H:mu at a tetrahedral site.
    tensors = parse_magres_hyperfine(FIXTURES / "fe_tetrahedral_muon.magres")

    assert [(tensor.species, tensor.index) for tensor in tensors] == [("Fe", 16), ("H:mu", 1)]
    muon = hyperfine_tensor_for_species(tensors, "H:mu")
    assert muon.isotropic_mhz == pytest.approx(324.2122)
    assert muon.coordinates_ang == pytest.approx((1.415, 0.708, 0.0))
    assert muon.tensor_mhz[1][1] == pytest.approx(461.8211)
    assert muon.tensor_mhz[0][0] == pytest.approx(255.4107)
    assert tensors[0].isotropic_mhz == pytest.approx(-1.6223)


def test_parses_isotropic_muonium() -> None:
    (muon,) = parse_magres_hyperfine(FIXTURES / "muonium_vacuum.magres")

    assert muon.isotropic_mhz == pytest.approx(-4628.4054)
    assert muon.tensor_mhz[2][2] == pytest.approx(-4628.4054)


def test_species_lookup_requires_exactly_one_match() -> None:
    tensors = parse_magres_hyperfine(FIXTURES / "fe_tetrahedral_muon.magres")

    with pytest.raises(MagresInputError, match="found no"):
        hyperfine_tensor_for_species(tensors, "H")


def test_missing_and_empty_files_are_reported(tmp_path: Path) -> None:
    with pytest.raises(MagresInputError, match="Failed to read"):
        parse_magres_hyperfine(tmp_path / "missing.magres")

    empty = tmp_path / "shielding.magres"
    empty.write_text("[magres]\n[/magres]\n", encoding="utf-8")
    with pytest.raises(MagresInputError, match="no hyperfine tensors"):
        parse_magres_hyperfine(empty)


def test_truncated_section_is_reported(tmp_path: Path) -> None:
    text = (FIXTURES / "muonium_vacuum.magres").read_text(encoding="utf-8")
    truncated = tmp_path / "truncated.magres"
    truncated.write_text(text.split("H:mu     1 Iso:")[0], encoding="utf-8")

    with pytest.raises(MagresInputError, match="H:mu 1 .* is incomplete"):
        parse_magres_hyperfine(truncated)
