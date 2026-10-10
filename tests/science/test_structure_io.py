"""Tests for CIF, MCIF and CASTEP CELL structure loading and CIF export."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write as ase_write

from mufinder.domain.models import Structure
from mufinder.science import structure_io as sio
from mufinder.science.structure_io import (
    CifMuonSite,
    StructureLoadError,
    export_structure_with_muon_sites,
    load_structure_file,
)

BOHR_IN_ANGSTROM = 0.529177210903

CUBIC_4 = """
%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART
"""

SI_AT_ORIGIN = """
%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _load_cell(tmp_path: Path, text: str):
    return load_structure_file(_write(tmp_path, "sample.cell", text))


# --------------------------------------------------------------------------
# Format dispatch and error normalisation
# --------------------------------------------------------------------------


def test_unsupported_extension_and_missing_file(tmp_path: Path) -> None:
    unsupported = _write(tmp_path, "bad.xyz", "placeholder")

    with pytest.raises(StructureLoadError, match="Unsupported structure extension"):
        load_structure_file(unsupported)

    with pytest.raises(StructureLoadError, match="does not exist"):
        load_structure_file(tmp_path / "missing.cif")

    with pytest.raises(StructureLoadError, match="does not exist"):
        load_structure_file(tmp_path / "missing.cell")


def test_load_cif_parse_failure_is_normalized(tmp_path: Path) -> None:
    path = _write(tmp_path, "broken.cif", "this is not valid cif content at all")

    with pytest.raises(StructureLoadError, match="Failed to parse structure file"):
        load_structure_file(path)


# --------------------------------------------------------------------------
# CIF
# --------------------------------------------------------------------------


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


def test_load_cif_without_moments_has_no_spin_properties(tmp_path: Path) -> None:
    atoms = Atoms(
        "Fe2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        pbc=True,
    )
    path = tmp_path / "plain.cif"
    ase_write(path, atoms)

    loaded = load_structure_file(path)

    assert "initial_magnetic_moments" not in loaded.properties
    assert "atomic_spins" not in loaded.properties
    assert loaded.properties["pbc"] == [True, True, True]
    assert loaded.properties["atom_count"] == 2


# --------------------------------------------------------------------------
# CASTEP CELL: geometry
# --------------------------------------------------------------------------


def test_load_cell_file_reads_spin_tokens(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        """
%BLOCK LATTICE_CART
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 8.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0 SPIN=1.5
O 0.5 0.5 0.5 SPIN=-0.5
%ENDBLOCK POSITIONS_FRAC
""",
    )

    assert loaded.species == ["Si", "O"]
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])
    assert loaded.properties == {
        "source_extension": ".cell",
        "atom_count": 2,
        "pbc": [True, True, True],
        "atomic_spins": pytest.approx([1.5, -0.5]),
    }


def test_load_cell_rows_without_spin_token_get_zero_spin(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4
        + """
%BLOCK POSITIONS_FRAC
Fe 0.0 0.0 0.0 SPIN=2.0
Fe 0.5 0.5 0.5
%ENDBLOCK POSITIONS_FRAC
""",
    )

    assert loaded.properties["atomic_spins"] == pytest.approx([2.0, 0.0])


def test_load_cell_without_spin_tokens_has_no_spins(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4
        + """
%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
O 0.5 0.5 0.5
%ENDBLOCK POSITIONS_FRAC
""",
    )

    assert loaded.species == ["Si", "O"]
    assert "atomic_spins" not in loaded.properties
    assert loaded.properties["atom_count"] == 2


def test_load_cell_keeps_fractional_positions_unwrapped(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4
        + """
%BLOCK POSITIONS_FRAC
Si 1.25 -0.25 0.0
%ENDBLOCK POSITIONS_FRAC
""",
    )

    assert loaded.fractional_positions == [pytest.approx([1.25, -0.25, 0.0])]


def test_load_cell_accepts_lowercase_block_names(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        """
%block lattice_cart
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%endblock lattice_cart
%block positions_frac
Si 0.0 0.0 0.0
%endblock positions_frac
""",
    )

    assert loaded.species == ["Si"]


def test_load_cell_lattice_abc_and_positions_abs(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        """
%BLOCK LATTICE_ABC
4.0 4.0 4.0
90.0 90.0 90.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_ABS
Si 0.0 0.0 0.0
O 2.0 2.0 2.0
%ENDBLOCK POSITIONS_ABS
""",
    )

    assert loaded.species == ["Si", "O"]
    assert loaded.lattice_vectors[0][0] == pytest.approx(4.0)
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])
    assert "atomic_spins" not in loaded.properties


def test_load_cell_positions_abs_wraps_into_the_cell(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4
        + """
%BLOCK POSITIONS_ABS
Si 5.0 -1.0 0.0
%ENDBLOCK POSITIONS_ABS
""",
    )

    assert loaded.fractional_positions == [pytest.approx([0.25, 0.75, 0.0])]


def test_load_cell_positions_abs_on_a_hexagonal_cell_face_wraps_to_zero(tmp_path: Path) -> None:
    # -b of a hexagonal cell sits a rounding error below fractional zero; it
    # must land on 0.0 rather than 1.0.
    loaded = _load_cell(
        tmp_path,
        """
%BLOCK LATTICE_ABC
3.0 3.0 5.0
90.0 90.0 120.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_ABS
Si 1.4999999999999993 -2.598076211353316 0.0
%ENDBLOCK POSITIONS_ABS
""",
    )

    position = loaded.fractional_positions[0]
    assert all(0.0 <= value < 1.0 for value in position)
    assert position == pytest.approx([0.0, 0.0, 0.0], abs=1e-12)


def test_load_cell_lattice_cart_and_positions_abs_with_ang_units(tmp_path: Path, recwarn) -> None:
    loaded = _load_cell(
        tmp_path,
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
""",
    )

    assert loaded.species == ["Si", "O"]
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])
    # The CELL reader never falls back to ASE's CASTEP reader and its warnings.
    assert recwarn.list == []


def test_load_cell_lattice_cart_with_bohr_units(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        """
%BLOCK LATTICE_CART
bohr
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
%ENDBLOCK LATTICE_CART
"""
        + SI_AT_ORIGIN,
    )

    assert loaded.lattice_vectors[0] == pytest.approx([BOHR_IN_ANGSTROM, 0.0, 0.0])


def test_load_cell_positions_abs_with_bohr_units(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4
        + f"""
%BLOCK POSITIONS_ABS
bohr
Si 0.0 0.0 0.0
O {2.0 / BOHR_IN_ANGSTROM!r} 0.0 0.0
%ENDBLOCK POSITIONS_ABS
""",
    )

    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.0, 0.0])


def test_load_cell_lattice_abc_with_unit_scaling(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        """
%BLOCK LATTICE_ABC
nm
0.4 0.4 0.4
90.0 90.0 90.0
%ENDBLOCK LATTICE_ABC
"""
        + SI_AT_ORIGIN,
    )

    assert loaded.lattice_vectors[0][0] == pytest.approx(4.0)


def test_load_cell_ignores_keywords_outside_the_geometry(tmp_path: Path) -> None:
    # Calculation keywords, including ones outside the bundled catalog, do not
    # stop a structure import.
    loaded = _load_cell(
        tmp_path,
        CUBIC_4
        + SI_AT_ORIGIN
        + """
FIX_VOL : false
KPOINT_MP_GRID 4 4 4
SOME_UNKNOWN_KEYWORD : 3
%BLOCK SPECIES_LCAO_STATES
Si 2
%ENDBLOCK SPECIES_LCAO_STATES
""",
    )

    assert loaded.species == ["Si"]


# --------------------------------------------------------------------------
# CASTEP CELL: malformed input
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        pytest.param(SI_AT_ORIGIN, "LATTICE_CART or LATTICE_ABC", id="missing-lattice"),
        pytest.param(CUBIC_4, "POSITIONS_FRAC or POSITIONS_ABS", id="missing-positions"),
        pytest.param(
            "%BLOCK LATTICE_CART\n1 0 0\n0 1 0\n%ENDBLOCK LATTICE_CART\n" + SI_AT_ORIGIN,
            "three lattice vectors",
            id="two-lattice-vectors",
        ),
        pytest.param(
            "%BLOCK LATTICE_CART\n1 0 0\n0 1 0\n0 0\n%ENDBLOCK LATTICE_CART\n" + SI_AT_ORIGIN,
            "LATTICE_CART",
            id="short-lattice-vector",
        ),
        pytest.param(
            "%BLOCK LATTICE_CART\nfurlong\n4 0 0\n0 4 0\n0 0 4\n%ENDBLOCK LATTICE_CART\n" + SI_AT_ORIGIN,
            "unsupported length unit 'furlong'",
            id="unsupported-unit",
        ),
        pytest.param(
            "%BLOCK LATTICE_ABC\n4 4 4\n%ENDBLOCK LATTICE_ABC\n" + SI_AT_ORIGIN,
            "lengths and angles rows",
            id="lattice-abc-one-row",
        ),
        pytest.param(
            "%BLOCK LATTICE_ABC\n4 4\n90 90 90\n%ENDBLOCK LATTICE_ABC\n" + SI_AT_ORIGIN,
            "LATTICE_ABC",
            id="lattice-abc-short-row",
        ),
        pytest.param(
            CUBIC_4 + "%BLOCK POSITIONS_FRAC\nSi 0.0 0.0\n%ENDBLOCK POSITIONS_FRAC\n",
            "POSITIONS_FRAC rows must contain species and three coordinates",
            id="short-positions-frac-row",
        ),
        pytest.param(
            CUBIC_4 + "%BLOCK POSITIONS_ABS\nSi 0.0 0.0\n%ENDBLOCK POSITIONS_ABS\n",
            "POSITIONS_ABS rows must contain species and three coordinates",
            id="short-positions-abs-row",
        ),
        pytest.param(
            CUBIC_4 + "%BLOCK POSITIONS_FRAC\nSi 0 0 0 SPIN=notanumber\n%ENDBLOCK POSITIONS_FRAC\n",
            "Invalid SPIN value 'notanumber'",
            id="invalid-spin",
        ),
        pytest.param(
            CUBIC_4 + "%BLOCK POSITIONS_FRAC\n%ENDBLOCK POSITIONS_FRAC\n",
            "POSITIONS_FRAC",
            id="empty-positions",
        ),
        pytest.param(
            CUBIC_4 + "%BLOCK POSITIONS_FRAC\nSi 0 0 0\n",
            "Unterminated block",
            id="unterminated-block",
        ),
    ],
)
def test_load_cell_rejects_malformed_geometry(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(StructureLoadError, match=match):
        _load_cell(tmp_path, text)


def test_load_cell_with_a_non_numeric_coordinate_is_a_load_error(tmp_path: Path) -> None:
    with pytest.raises(StructureLoadError, match="sample.cell.*could not convert string to float"):
        _load_cell(
            tmp_path,
            "%BLOCK LATTICE_CART\n4 0 x\n0 4 0\n0 0 4\n%ENDBLOCK LATTICE_CART\n" + SI_AT_ORIGIN,
        )


@pytest.mark.parametrize(
    ("text", "match"),
    [
        pytest.param(
            CUBIC_4 + "%BLOCK LATTICE_ABC\n5 5 5\n90 90 90\n%ENDBLOCK LATTICE_ABC\n" + SI_AT_ORIGIN,
            "both LATTICE_CART and LATTICE_ABC",
            id="both-lattice-blocks",
        ),
        pytest.param(
            CUBIC_4 + SI_AT_ORIGIN + "%BLOCK POSITIONS_ABS\nO 1 1 1\n%ENDBLOCK POSITIONS_ABS\n",
            "both POSITIONS_FRAC and POSITIONS_ABS",
            id="both-position-blocks",
        ),
        pytest.param(
            CUBIC_4 + SI_AT_ORIGIN + "%BLOCK POSITIONS_FRAC\nO 0.5 0.5 0.5\n%ENDBLOCK POSITIONS_FRAC\n",
            "Duplicate block POSITIONS_FRAC",
            id="repeated-block",
        ),
    ],
)
def test_load_cell_rejects_conflicting_geometry_blocks(tmp_path: Path, text: str, match: str) -> None:
    # CASTEP itself rejects these files, so the import does not pick one block.
    with pytest.raises(StructureLoadError, match=match):
        _load_cell(tmp_path, text)


# --------------------------------------------------------------------------
# CASTEP CELL: files written by CASTEP
# --------------------------------------------------------------------------


def test_load_cell_reads_a_castep_out_cell(tmp_path: Path) -> None:
    # CASTEP pads SPIN values after the '=', writes comments, muon species
    # labels and calculation keywords outside the bundled catalog.
    loaded = _load_cell(
        tmp_path,
        """
%BLOCK LATTICE_CART
ang
       5.430000000000000       0.000000000000000       0.000000000000000
       0.000000000000000       5.430000000000000       0.000000000000000
       0.000000000000000       0.000000000000000       5.430000000000000
%ENDBLOCK LATTICE_CART

%BLOCK CELL_CONSTRAINTS
       1       1       1
       0       0       0
%ENDBLOCK CELL_CONSTRAINTS

FIX_COM : false
%BLOCK IONIC_CONSTRAINTS
%ENDBLOCK IONIC_CONSTRAINTS

%BLOCK POSITIONS_FRAC
! relaxed positions
   Si            0.0000000000000000    0.0000000000000000    0.0000000000000000 SPIN=  1.000
   H:mu          0.2500000000000000    0.2500000000000000    0.2500000000000000 SPIN= -0.500
%ENDBLOCK POSITIONS_FRAC

FIX_VOL : false
%BLOCK SPECIES_MASS
          Si     28.0855000
        H:mu      0.1134289
%ENDBLOCK SPECIES_MASS

%BLOCK SPECIES_LCAO_STATES
          Si         2
%ENDBLOCK SPECIES_LCAO_STATES

SYMMETRY_TOL :        0.01000000 ang
""",
    )

    assert loaded.species == ["Si", "H:mu"]
    assert loaded.lattice_vectors[0] == pytest.approx([5.43, 0.0, 0.0])
    assert loaded.fractional_positions[1] == pytest.approx([0.25, 0.25, 0.25])
    assert loaded.properties["atomic_spins"] == pytest.approx([1.0, -0.5])


def test_load_cell_empty_spin_marker_means_no_spin(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4 + "%BLOCK POSITIONS_FRAC\nSi 0 0 0 SPIN=\n%ENDBLOCK POSITIONS_FRAC\n",
    )

    assert "atomic_spins" not in loaded.properties


def test_load_cell_positions_abs_accepts_a_muon_species_label(tmp_path: Path) -> None:
    loaded = _load_cell(
        tmp_path,
        CUBIC_4 + "%BLOCK POSITIONS_ABS\nSi 0 0 0\nH:mu 1 2 3\n%ENDBLOCK POSITIONS_ABS\n",
    )

    assert loaded.species == ["Si", "H:mu"]
    assert loaded.fractional_positions[1] == pytest.approx([0.25, 0.5, 0.75])


# --------------------------------------------------------------------------
# MCIF magnetic metadata
# --------------------------------------------------------------------------

_MCIF_HEADER = """
data_test
_symmetry_space_group_name_H-M 'P 1'
_cell_length_a 5
_cell_length_b 5
_cell_length_c 5
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
"""


def _mcif(tags: str = "", atoms: str = "Fe1 Fe 0.0 0.0 0.0\n  Mn1 Mn 0.5 0.5 0.5", loops: str = "") -> str:
    return (
        _MCIF_HEADER
        + tags
        + """
loop_
  _atom_site_label
  _atom_site_type_symbol
  _atom_site_fract_x
  _atom_site_fract_y
  _atom_site_fract_z
  """
        + atoms
        + "\n"
        + loops
    )


def _load_mcif(tmp_path: Path, text: str):
    return load_structure_file(_write(tmp_path, "sample.mcif", text))


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
"""
    loaded = _load_mcif(tmp_path, mcif_text)

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
    assert moments[0]["coordinate_system"] == "crystalaxis"


def test_load_mcif_preserves_richer_magnetic_metadata(tmp_path: Path) -> None:
    mcif_text = _mcif(
        atoms="Fe1_1 Fe 0.0 0.0 0.0\n  Mn1 Mn 0.5 0.0 0.5",
        loops="""
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
""",
    )

    loaded = _load_mcif(tmp_path, mcif_text)

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


def test_load_mcif_without_magnetic_metadata_has_no_magnetic_structure(tmp_path: Path) -> None:
    loaded = _load_mcif(tmp_path, _mcif())

    assert "magnetic_structure" not in loaded.properties
    assert "propagation_vector" not in loaded.properties
    assert loaded.properties["source_extension"] == ".mcif"


def test_load_mcif_reads_parent_propagation_vector_tags(tmp_path: Path) -> None:
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            tags="""
_parent_propagation_vector.kx 0.5
_parent_propagation_vector.ky 0.0
_parent_propagation_vector.kz 0.0
"""
        ),
    )

    assert loaded.properties["magnetic_structure"]["propagation_vectors_rlu"] == [[0.5, 0.0, 0.0]]


def test_load_mcif_reads_propagation_vector_loop_and_dedupes_repeats(tmp_path: Path) -> None:
    # The same vector given as a triplet tag, as x/y/z tags and in a loop is
    # recorded once; a comment after a value is ignored.
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            tags="""
_magnetic_propagation_vector_kxkykz '0, 0, 1/4'
_propagation_vector_x 0.0  # comment
_propagation_vector_y 0.0
_propagation_vector_z 0.25
""",
            loops="""
loop_
  _propagation_vector_kxkykz
  '0 0 0.25'
  '[0.5 0.5 0.5]'
""",
        ),
    )

    vectors = loaded.properties["magnetic_structure"]["propagation_vectors_rlu"]
    assert vectors == [pytest.approx([0.0, 0.0, 0.25]), pytest.approx([0.5, 0.5, 0.5])]


def test_load_mcif_reads_cartesian_moments(tmp_path: Path) -> None:
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            loops="""
loop_
  _atom_site_moment.label
  _atom_site_moment.cartn_x
  _atom_site_moment.cartn_y
  _atom_site_moment.cartn_z
  Fe1 0.0 0.0 3.0
"""
        ),
    )

    moment = loaded.properties["magnetic_structure"]["moments"][0]
    assert moment["coordinate_system"] == "cartesian"
    assert moment["vector"] == [0.0, 0.0, 3.0]
    assert loaded.properties["atomic_spins"] == pytest.approx([3.0, 0.0])


def test_load_mcif_skips_moments_with_unparsable_components(tmp_path: Path) -> None:
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            loops="""
loop_
  _atom_site_moment.label
  _atom_site_moment.crystalaxis_x
  _atom_site_moment.crystalaxis_y
  _atom_site_moment.crystalaxis_z
  Fe1 ? 0.0 1.0
  Mn1 abc 0.0 1.0
  Fe1 1/0 0.0 1.0
  Mn1 -3.0e-1 0.0 .
"""
        ),
    )

    assert "magnetic_structure" not in loaded.properties
    assert "atomic_spins" not in loaded.properties


def test_load_mcif_skips_moment_loops_missing_an_axis(tmp_path: Path) -> None:
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            loops="""
loop_
  _atom_site_moment.label
  _atom_site_moment.crystalaxis_x
  Fe1 1.0
"""
        ),
    )

    assert "magnetic_structure" not in loaded.properties


def test_load_mcif_labels_unlabelled_moments_by_row_and_uses_row_order(tmp_path: Path) -> None:
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            loops="""
loop_
  _atom_site_moment.crystalaxis_x
  _atom_site_moment.crystalaxis_y
  _atom_site_moment.crystalaxis_z
  0.0 0.0 1.0
  0.0 0.0 2.0
"""
        ),
    )

    moments = loaded.properties["magnetic_structure"]["moments"]
    assert [moment["label"] for moment in moments] == ["moment_1", "moment_2"]
    assert all("atom_index" not in moment for moment in moments)
    assert loaded.properties["atomic_spins"] == pytest.approx([1.0, 2.0])


def test_load_mcif_leaves_unmatched_and_ambiguous_moment_labels_unindexed(tmp_path: Path) -> None:
    loaded = _load_mcif(
        tmp_path,
        _mcif(
            atoms="Fe1_a Fe 0.0 0.0 0.0\n  Fe1_b Fe 0.5 0.5 0.5",
            loops="""
loop_
  _atom_site_moment.label
  _atom_site_moment.crystalaxis_x
  _atom_site_moment.crystalaxis_y
  _atom_site_moment.crystalaxis_z
  Fe1 0.0 0.0 1.0
  Zz9 0.0 0.0 2.0
""",
        ),
    )

    moments = loaded.properties["magnetic_structure"]["moments"]
    assert [moment["label"] for moment in moments] == ["Fe1", "Zz9"]
    assert all("atom_index" not in moment for moment in moments)


# --------------------------------------------------------------------------
# MCIF parser helpers with no public path to their edge cases
# --------------------------------------------------------------------------


def test_tokenize_cif_line_falls_back_on_unbalanced_quotes() -> None:
    assert sio._tokenize_cif_line("a 'b c") == ["a", "'b", "c"]


def test_parse_cif_tags_and_loops_skips_headerless_loop_and_keeps_bare_tag() -> None:
    tags, loops = sio._parse_cif_tags_and_loops("loop_\ndata_next\n_tag v\n_lonely")

    assert loops == []
    assert tags == {"_tag": "v", "_lonely": ""}


def test_extract_atom_labels_dedupes_tag_and_loop_labels() -> None:
    tags = {"_atom_site_label": "Si1"}
    loops = [
        {
            "headers": ["_atom_site_label"],
            "rows": [{"_atom_site_label": "Si1"}, {"_atom_site_label": "O1"}],
        }
    ]

    assert sio._extract_atom_labels(tags, loops) == ["Si1", "O1"]


def test_attach_moment_atom_indices_without_labels_is_a_passthrough() -> None:
    moments = [{"label": "Fe1", "vector": [0.0, 0.0, 1.0]}]

    assert sio._attach_moment_atom_indices(moments, atom_labels=[]) == moments
    assert sio._find_atom_label_index(["Fe1"], "") is None


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("0, 0, 1/4", [0.0, 0.0, 0.25]),
        ("0 0 0.25", [0.0, 0.0, 0.25]),
        ("[0.5 0.5 0.5]", [0.5, 0.5, 0.5]),
        ("", None),
        ("?", None),
        ("0 0", None),
        ("0 0 bad", None),
    ],
)
def test_parse_cif_triplet(token: str, expected: list[float] | None) -> None:
    assert sio._parse_cif_triplet(token) == (None if expected is None else pytest.approx(expected))


def test_spins_from_magnetic_structure_skips_malformed_moments() -> None:
    magnetic = {
        "moments": [
            "not-a-dict",
            {"atom_index": "x", "vector": [0, 0, 1]},
            {"atom_index": 99, "vector": [0, 0, 1]},
            {"atom_index": 0, "vector": [0, 0]},
            {"atom_index": 0, "vector": "nope"},
            {"atom_index": 0, "vector": [0, 0, 4.0]},
        ]
    }

    assert sio._spins_from_magnetic_structure(magnetic_structure=magnetic, atom_count=1) == pytest.approx([4.0])
    assert sio._spins_from_magnetic_structure(magnetic_structure={}, atom_count=2) is None
    assert sio._spins_from_magnetic_structure(magnetic_structure={"moments": []}, atom_count=2) is None


# --------------------------------------------------------------------------
# CIF export with muon sites
# --------------------------------------------------------------------------


def test_export_structure_with_muon_sites_writes_p1_cif_with_labels(tmp_path: Path) -> None:
    host = Structure(
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    sites = [
        CifMuonSite(label="Mu1", fractional_position=(0.25, 0.25, 0.25), relative_energy_ev=0.0, wyckoff_label="8c"),
        CifMuonSite(label="Mu2", fractional_position=(0.1, 0.0, 0.0), relative_energy_ev=0.12, wyckoff_label=None),
    ]
    output_path = tmp_path / "export.cif"

    export_structure_with_muon_sites(host, sites, output_path)

    text = output_path.read_text(encoding="utf-8")
    assert "_space_group_IT_number       1" in text
    assert "Si  Si1" in text
    assert "Si  Si2" in text
    assert "H   Mu1" in text
    assert "H   Mu2" in text

    # The _atom_site loop itself must stay standard (exactly the 7 usual
    # columns, one token per header per row): mixing MuFinder-specific
    # columns into it desynchronizes column counting in some readers
    # (observed with VESTA), shifting every atom's fields by one.
    atom_site_loop = text.split("loop_")[2]
    assert "_mufinder" not in atom_site_loop
    for line in ("Si  Si1", "Si  Si2", "H   Mu1", "H   Mu2"):
        row = next(candidate for candidate in atom_site_loop.splitlines() if candidate.strip().startswith(line))
        assert len(row.split()) == 7

    # Per-site metadata instead lives in its own standalone auxiliary loop.
    aux_loop = text.split("loop_")[3]
    assert "_mufinder_muon_site_label" in aux_loop
    assert "Mu1  0.00000  8c" in aux_loop
    assert "Mu2  0.12000  ?" in aux_loop

    reloaded = load_structure_file(output_path)
    assert reloaded.species == ["Si", "Si", "H", "H"]
    assert len(reloaded.fractional_positions) == 4


def test_export_structure_with_muon_sites_labels_are_per_species(tmp_path: Path) -> None:
    # A mixed-species host must number labels within each element (Mn1, O1..O2),
    # not by a single running index across every atom (Mn3, O4..O5).
    host = Structure(
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Sr", "Sr", "Mn", "O", "O"],
        fractional_positions=[
            [0.0, 0.0, 0.35],
            [0.0, 0.0, 0.65],
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],
            [0.0, 0.5, 0.0],
        ],
    )
    output_path = tmp_path / "export.cif"

    export_structure_with_muon_sites(
        host, [CifMuonSite(label="Mu1", fractional_position=(0.2, 0.2, 0.2))], output_path
    )

    text = output_path.read_text(encoding="utf-8")
    assert "Sr  Sr1" in text
    assert "Sr  Sr2" in text
    assert "Mn  Mn1" in text
    assert "O   O1" in text
    assert "O   O2" in text


def test_export_structure_with_muon_sites_warns_when_coincident_with_host_atom(tmp_path: Path) -> None:
    # A muon site landing on top of (or implausibly close to) a host atom
    # produces a valid-looking CIF that many viewers render as one merged,
    # fractionally-occupied site (e.g. "Sr1/Mu1"); this must be surfaced.
    host = Structure(
        lattice_vectors=[[3.79, 0.0, 0.0], [0.0, 3.79, 0.0], [0.0, 0.0, 12.5]],
        species=["Sr", "Mn"],
        fractional_positions=[[0.0, 0.0, 0.35], [0.0, 0.0, 0.0]],
    )
    sites = [
        CifMuonSite(label="Mu1", fractional_position=(0.0, 0.0, 0.35)),
        CifMuonSite(label="Mu2", fractional_position=(0.2, 0.2, 0.2)),
    ]
    output_path = tmp_path / "export.cif"

    warnings = export_structure_with_muon_sites(host, sites, output_path)

    assert len(warnings) == 1
    assert "Mu1" in warnings[0]
    assert "Sr1" in warnings[0]
    assert output_path.exists()


def test_export_structure_with_muon_sites_no_warning_for_plausible_separation(tmp_path: Path) -> None:
    host = Structure(
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
    )
    output_path = tmp_path / "export.cif"

    warnings = export_structure_with_muon_sites(
        host, [CifMuonSite(label="Mu1", fractional_position=(0.25, 0.25, 0.25))], output_path
    )

    assert warnings == []


def test_export_structure_with_muon_sites_creates_parent_directory(tmp_path: Path) -> None:
    host = Structure(
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    output_path = tmp_path / "nested" / "export.cif"

    export_structure_with_muon_sites(
        host, [CifMuonSite(label="Mu1", fractional_position=(0.5, 0.5, 0.5))], output_path
    )

    assert output_path.exists()
