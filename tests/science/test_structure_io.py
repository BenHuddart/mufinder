"""Tests for CIF, MCIF, and CELL structure loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

from mufinder.science.structure_io import StructureLoadError, load_structure_file


def test_load_cif_file(tmp_path: Path) -> None:
    atoms = Atoms(
        "SiO2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.3, 0.3, 0.3], [0.7, 0.7, 0.7]],
        cell=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        pbc=True,
    )
    path = tmp_path / "sample.cif"
    ase_write(path, atoms)

    loaded = load_structure_file(path)
    assert len(loaded.species) == 3
    assert loaded.properties["source_extension"] == ".cif"
    assert loaded.lattice_vectors[0][0] == pytest.approx(5.0)


def test_load_cell_file(tmp_path: Path) -> None:
    cell_text = """
%BLOCK LATTICE_CART
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 8.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0 SPIN=1.5
O 0.5 0.5 0.5 SPIN=-0.5
%ENDBLOCK POSITIONS_FRAC
""".strip()
    path = tmp_path / "sample.cell"
    path.write_text(cell_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert loaded.species == ["Si", "O"]
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])
    assert loaded.properties["atomic_spins"] == pytest.approx([1.5, -0.5])


def test_load_cell_file_avoids_ase_castep_fallback_warnings(tmp_path: Path, recwarn) -> None:
    path = tmp_path / "unit_sample.cell"
    path.write_text(
        """
%BLOCK LATTICE_CART
ang
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 8.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_ABS
ang
Si 0.0 0.0 0.0
O 4.0 4.0 4.0
%ENDBLOCK POSITIONS_ABS
""".strip(),
        encoding="utf-8",
    )

    loaded = load_structure_file(path)

    assert loaded.species == ["Si", "O"]
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])
    assert recwarn.list == []


def test_load_mcif_file(tmp_path: Path) -> None:
    mcif_text = """
data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 6
_cell_length_b 6
_cell_length_c 6
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_cell_wave_vector_x 0.0
_cell_wave_vector_y 0.0
_cell_wave_vector_z 0.5
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Si1 Si 0.0 0.0 0.0
  O1 O 0.25 0.25 0.25
loop_
    _atom_site_moment.label
    _atom_site_moment.crystalaxis_x
    _atom_site_moment.crystalaxis_y
    _atom_site_moment.crystalaxis_z
    Si1 0.0 0.0 1.0
    O1 0.0 0.0 -1.0
""".strip()
    path = tmp_path / "sample.mcif"
    path.write_text(mcif_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert loaded.species == ["Si", "O"]
    assert loaded.properties["source_extension"] == ".mcif"
    assert loaded.properties["magnetic_structure"]["propagation_vector"] == pytest.approx([0.0, 0.0, 0.5])
    assert loaded.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.5])
    assert loaded.properties["atomic_spins"] == pytest.approx([1.0, -1.0])
    moments = loaded.properties["magnetic_structure"]["moments"]
    assert len(moments) == 2
    assert moments[0]["label"] == "Si1"
    assert moments[0]["vector"] == pytest.approx([0.0, 0.0, 1.0])
    assert moments[0]["atom_index"] == 0


def test_load_mcif_preserves_richer_magnetic_metadata(tmp_path: Path) -> None:
    mcif_text = """
data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  Fe1_1 Fe 0.0 0.0 0.0
  Mn1 Mn 0.5 0.0 0.5
loop_
  _magnetic_propagation_vector_kxkykz
  '0, 0, 1/4'
  '1/2, 0, 0'
loop_
  _atom_site_moment_label
  _atom_site_moment_crystalaxis_x
  _atom_site_moment_crystalaxis_y
  _atom_site_moment_crystalaxis_z
  Fe1 0.0 0.0 1.25(5)
  Mn1 0.0 1/2 -2.0
""".strip()
    path = tmp_path / "rich.mcif"
    path.write_text(mcif_text, encoding="utf-8")

    loaded = load_structure_file(path)

    magnetic = loaded.properties["magnetic_structure"]
    assert magnetic["source"] == "mcif"
    assert magnetic["atom_labels"] == ["Fe1_1", "Mn1"]
    assert magnetic["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert len(magnetic["propagation_vectors_rlu"]) == 2
    assert magnetic["propagation_vectors_rlu"][0] == pytest.approx([0.0, 0.0, 0.25])
    assert magnetic["propagation_vectors_rlu"][1] == pytest.approx([0.5, 0.0, 0.0])
    assert loaded.properties["propagation_vector"] == pytest.approx([0.0, 0.0, 0.25])
    assert loaded.properties["atomic_spins"] == pytest.approx([1.25, -2.0])

    moments = magnetic["moments"]
    assert moments[0]["label"] == "Fe1"
    assert moments[0]["atom_index"] == 0
    assert moments[0]["atom_label"] == "Fe1_1"
    assert moments[1]["label"] == "Mn1"
    assert moments[1]["atom_index"] == 1
    assert moments[1]["vector"] == pytest.approx([0.0, 0.5, -2.0])
    assert magnetic["site_moments"]["Mn1"] == pytest.approx([0.0, 0.5, -2.0])


def test_unsupported_extension_and_missing_file(tmp_path: Path) -> None:
    unsupported = tmp_path / "bad.xyz"
    unsupported.write_text("placeholder", encoding="utf-8")

    with pytest.raises(StructureLoadError, match="Unsupported structure extension"):
        load_structure_file(unsupported)

    with pytest.raises(StructureLoadError, match="does not exist"):
        load_structure_file("/tmp/definitely_missing_file_12345.cif")
