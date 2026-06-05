"""Additional coverage for structure_io error paths and parser branches.

These tests exercise currently-uncovered functions, branches, and error
paths in ``mufinder.science.structure_io`` without duplicating the existing
suite in ``test_structure_io.py``. They cover CASTEP CELL parsing variants
(LATTICE_ABC, POSITIONS_ABS, unit scaling, malformed rows), the lightweight
CIF/MCIF tag/loop parser helpers, propagation-vector extraction variants,
moment/label matching, and the numeric token parsers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.science import structure_io as sio
from mufinder.science.structure_io import StructureLoadError, load_structure_file


# --------------------------------------------------------------------------
# load_structure_file: top-level resolution and error normalisation
# --------------------------------------------------------------------------


def test_resolve_ase_format_variants() -> None:
    assert sio._resolve_ase_format(".cif") == "cif"
    assert sio._resolve_ase_format(".mcif") == "cif"
    assert sio._resolve_ase_format(".cell") == "castep-cell"
    assert sio._resolve_ase_format(".xyz") is None


def test_load_cif_parse_failure_is_normalized(tmp_path: Path) -> None:
    path = tmp_path / "broken.cif"
    path.write_text("this is not valid cif content at all\n", encoding="utf-8")

    with pytest.raises(StructureLoadError, match="Failed to parse structure file"):
        load_structure_file(path)


def test_load_cif_without_moments_has_no_spin_properties(tmp_path: Path) -> None:
    from ase import Atoms
    from ase.io import write as ase_write

    atoms = Atoms(
        "Fe2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        pbc=True,
    )
    path = tmp_path / "plain.cif"
    ase_write(path, atoms)

    loaded = load_structure_file(path)
    # No magnetic moments => the magmom-derived properties are absent.
    assert "initial_magnetic_moments" not in loaded.properties
    assert "atomic_spins" not in loaded.properties
    assert loaded.properties["pbc"] == [True, True, True]
    assert loaded.properties["atom_count"] == 2


# --------------------------------------------------------------------------
# CASTEP CELL: LATTICE_ABC, units, and POSITIONS_ABS round-trips
# --------------------------------------------------------------------------


def test_load_cell_lattice_abc_and_positions_abs(tmp_path: Path) -> None:
    cell_text = """
%BLOCK LATTICE_ABC
4.0 4.0 4.0
90.0 90.0 90.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_ABS
Si 0.0 0.0 0.0
O 2.0 2.0 2.0
%ENDBLOCK POSITIONS_ABS
""".strip()
    path = tmp_path / "abc.cell"
    path.write_text(cell_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert loaded.species == ["Si", "O"]
    assert loaded.lattice_vectors[0][0] == pytest.approx(4.0)
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])
    # POSITIONS_ABS path yields no spins.
    assert "atomic_spins" not in loaded.properties


def test_load_cell_lattice_abc_with_unit_scaling(tmp_path: Path) -> None:
    # nm scale => lengths multiplied by 10 to angstrom.
    cell_text = """
%BLOCK LATTICE_ABC
nm
0.4 0.4 0.4
90.0 90.0 90.0
%ENDBLOCK LATTICE_ABC

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
""".strip()
    path = tmp_path / "abc_nm.cell"
    path.write_text(cell_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert loaded.lattice_vectors[0][0] == pytest.approx(4.0)


def test_load_cell_positions_abs_with_units(tmp_path: Path) -> None:
    cell_text = """
%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_ABS
ang
Si 0.0 0.0 0.0
O 2.0 2.0 2.0
%ENDBLOCK POSITIONS_ABS
""".strip()
    path = tmp_path / "abs_ang.cell"
    path.write_text(cell_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert loaded.fractional_positions[1] == pytest.approx([0.5, 0.5, 0.5])


def test_load_cell_lattice_cart_with_bohr_units(tmp_path: Path) -> None:
    cell_text = """
%BLOCK LATTICE_CART
bohr
1.0 0.0 0.0
0.0 1.0 0.0
0.0 0.0 1.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
H 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
""".strip()
    path = tmp_path / "bohr.cell"
    path.write_text(cell_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert loaded.lattice_vectors[0][0] == pytest.approx(sio._CELL_UNITS["bohr"])


# --------------------------------------------------------------------------
# CASTEP CELL: error paths
# --------------------------------------------------------------------------


def test_cell_missing_lattice_block(tmp_path: Path) -> None:
    cell_text = """
%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
""".strip()
    path = tmp_path / "nolat.cell"
    path.write_text(cell_text, encoding="utf-8")

    with pytest.raises(StructureLoadError, match="LATTICE_CART or LATTICE_ABC"):
        load_structure_file(path)


def test_cell_missing_positions_block(tmp_path: Path) -> None:
    cell_text = """
%BLOCK LATTICE_CART
4.0 0.0 0.0
0.0 4.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART
""".strip()
    path = tmp_path / "nopos.cell"
    path.write_text(cell_text, encoding="utf-8")

    with pytest.raises(StructureLoadError, match="POSITIONS_FRAC or POSITIONS_ABS"):
        load_structure_file(path)


def test_lattice_cart_too_few_vectors() -> None:
    with pytest.raises(StructureLoadError, match="three lattice vectors"):
        sio._parse_lattice_cart_block(["1.0 0.0 0.0", "0.0 1.0 0.0"])


def test_lattice_cart_vector_too_few_components() -> None:
    with pytest.raises(StructureLoadError, match="three components"):
        sio._parse_lattice_cart_block(["1.0 0.0 0.0", "0.0 1.0 0.0", "0.0 0.0"])


def test_lattice_abc_too_few_rows() -> None:
    with pytest.raises(StructureLoadError, match="lengths and angles rows"):
        sio._parse_lattice_abc_block(["4.0 4.0 4.0"])


def test_lattice_abc_wrong_count() -> None:
    with pytest.raises(StructureLoadError, match="three lengths and three angles"):
        sio._parse_lattice_abc_block(["4.0 4.0", "90 90 90"])


def test_positions_frac_too_few_columns() -> None:
    with pytest.raises(StructureLoadError, match="species and three coordinates"):
        sio._parse_positions_frac_block(["Si 0.0 0.0"])


def test_positions_frac_invalid_spin_value() -> None:
    with pytest.raises(StructureLoadError, match="Invalid SPIN value"):
        sio._parse_positions_frac_block(["Si 0.0 0.0 0.0 SPIN=notanumber"])


def test_positions_frac_without_spin_returns_none() -> None:
    species, positions, spins = sio._parse_positions_frac_block(
        ["Si 0.0 0.0 0.0", "O 0.5 0.5 0.5"]
    )
    assert species == ["Si", "O"]
    assert positions[1] == [0.5, 0.5, 0.5]
    assert spins is None


def test_positions_abs_too_few_columns() -> None:
    with pytest.raises(StructureLoadError, match="species and three coordinates"):
        sio._parse_positions_abs_block(["Si 0.0 0.0"])


def test_parse_length_unit_unsupported() -> None:
    with pytest.raises(StructureLoadError, match="unsupported length unit"):
        sio._parse_length_unit("furlong", block_name="LATTICE_CART")


def test_parse_length_unit_known() -> None:
    assert sio._parse_length_unit("ANG", block_name="LATTICE_CART") == 1.0


# --------------------------------------------------------------------------
# Block extraction
# --------------------------------------------------------------------------


def test_extract_castep_blocks_case_insensitive_and_strips() -> None:
    text = """
%block Foo_Bar
  line one
   line two
%endblock Foo_Bar
""".strip()
    blocks = sio._extract_castep_blocks(text)
    assert blocks == {"foo_bar": ["line one", "line two"]}


# --------------------------------------------------------------------------
# CIF/MCIF tag and loop parser
# --------------------------------------------------------------------------


def test_normalize_cif_line_strips_comments() -> None:
    assert sio._normalize_cif_line("# whole comment") == ""
    assert sio._normalize_cif_line("   ") == ""
    assert sio._normalize_cif_line("value # trailing") == "value"


def test_tokenize_cif_line_handles_unbalanced_quotes() -> None:
    # Unbalanced quote forces the shlex ValueError fallback to plain split.
    assert sio._tokenize_cif_line("a 'b c") == ["a", "'b", "c"]


def test_parse_cif_tags_and_loops_basic() -> None:
    text = """
data_x
_simple_tag value
loop_
  _atom_site_label
  _atom_site_fract_x
  Si1 0.0
  O1 0.25
""".strip()
    tags, loops = sio._parse_cif_tags_and_loops(text)
    assert tags["_simple_tag"] == "value"
    assert len(loops) == 1
    assert loops[0]["headers"] == ["_atom_site_label", "_atom_site_fract_x"]
    assert loops[0]["rows"][0] == {"_atom_site_label": "Si1", "_atom_site_fract_x": "0.0"}


def test_parse_cif_tags_and_loops_empty_loop_skipped() -> None:
    # A loop_ with no headers (immediately followed by data_) is skipped.
    text = "loop_\ndata_next\n_tag v"
    tags, loops = sio._parse_cif_tags_and_loops(text)
    assert loops == []
    assert tags["_tag"] == "v"


def test_parse_cif_tags_and_loops_bare_key_has_empty_value() -> None:
    tags, _ = sio._parse_cif_tags_and_loops("_lonely")
    assert tags["_lonely"] == ""


# --------------------------------------------------------------------------
# Atom labels and propagation vectors
# --------------------------------------------------------------------------


def test_extract_atom_labels_from_tag_and_loop_dedupes() -> None:
    tags = {"_atom_site_label": "Si1"}
    loops = [
        {
            "headers": ["_atom_site_label"],
            "rows": [{"_atom_site_label": "Si1"}, {"_atom_site_label": "O1"}],
        }
    ]
    assert sio._extract_atom_labels(tags, loops) == ["Si1", "O1"]


def test_extract_propagation_vectors_from_tag_triplet_and_xyz() -> None:
    tags = {
        "_magnetic_propagation_vector_kxkykz": "0, 0, 1/4",
        "_cell_wave_vector_x": "0.0",
        "_cell_wave_vector_y": "0.0",
        "_cell_wave_vector_z": "0.5",
    }
    vectors = sio._extract_propagation_vectors(tags, [])
    assert [0.0, 0.0, 0.25] in vectors
    assert [0.0, 0.0, 0.5] in vectors


def test_extract_propagation_vectors_parent_keys() -> None:
    tags = {
        "_parent_propagation_vector.kx": "0.5",
        "_parent_propagation_vector.ky": "0.0",
        "_parent_propagation_vector.kz": "0.0",
    }
    vectors = sio._extract_propagation_vectors(tags, [])
    assert vectors == [[0.5, 0.0, 0.0]]


def test_extract_propagation_vectors_from_loop() -> None:
    loops = [
        {
            "headers": ["_propagation_vector_kxkykz"],
            "rows": [{"_propagation_vector_kxkykz": "0 0 0.25"}],
        }
    ]
    vectors = sio._extract_propagation_vectors({}, loops)
    assert vectors == [[0.0, 0.0, 0.25]]


def test_extract_propagation_vectors_none_found() -> None:
    assert sio._extract_propagation_vectors({}, []) == []


# --------------------------------------------------------------------------
# Moment extraction and atom-index attachment
# --------------------------------------------------------------------------


def test_extract_moments_cartesian_coordinate_system() -> None:
    loops = [
        {
            "headers": [
                "_atom_site_moment.label",
                "_atom_site_moment.cartn_x",
                "_atom_site_moment.cartn_y",
                "_atom_site_moment.cartn_z",
            ],
            "rows": [
                {
                    "_atom_site_moment.label": "Fe1",
                    "_atom_site_moment.cartn_x": "0.0",
                    "_atom_site_moment.cartn_y": "0.0",
                    "_atom_site_moment.cartn_z": "3.0",
                }
            ],
        }
    ]
    moments = sio._extract_moments_from_loops(loops)
    assert moments[0]["coordinate_system"] == "cartesian"
    assert moments[0]["vector"] == [0.0, 0.0, 3.0]


def test_extract_moments_missing_axis_header_skipped() -> None:
    loops = [
        {
            "headers": ["_atom_site_moment.label", "_atom_site_moment.crystalaxis_x"],
            "rows": [{"_atom_site_moment.label": "Fe1"}],
        }
    ]
    assert sio._extract_moments_from_loops(loops) == []


def test_extract_moments_unparsable_value_skipped() -> None:
    loops = [
        {
            "headers": [
                "_atom_site_moment.crystalaxis_x",
                "_atom_site_moment.crystalaxis_y",
                "_atom_site_moment.crystalaxis_z",
            ],
            "rows": [
                {
                    "_atom_site_moment.crystalaxis_x": "?",
                    "_atom_site_moment.crystalaxis_y": "0.0",
                    "_atom_site_moment.crystalaxis_z": "1.0",
                }
            ],
        }
    ]
    assert sio._extract_moments_from_loops(loops) == []


def test_extract_moments_default_label_when_no_label_header() -> None:
    loops = [
        {
            "headers": [
                "_atom_site_moment.crystalaxis_x",
                "_atom_site_moment.crystalaxis_y",
                "_atom_site_moment.crystalaxis_z",
            ],
            "rows": [
                {
                    "_atom_site_moment.crystalaxis_x": "0.0",
                    "_atom_site_moment.crystalaxis_y": "0.0",
                    "_atom_site_moment.crystalaxis_z": "1.0",
                }
            ],
        }
    ]
    moments = sio._extract_moments_from_loops(loops)
    assert moments[0]["label"] == "moment_1"


def test_extract_moments_no_moment_header_skipped() -> None:
    loops = [{"headers": ["_atom_site_label"], "rows": []}]
    assert sio._extract_moments_from_loops(loops) == []


def test_attach_moment_atom_indices_no_labels_passthrough() -> None:
    moments = [{"label": "Fe1", "vector": [0.0, 0.0, 1.0]}]
    assert sio._attach_moment_atom_indices(moments, atom_labels=[]) == moments


def test_attach_moment_atom_indices_unmatched_label_keeps_moment() -> None:
    moments = [{"label": "Zz9", "vector": [0.0, 0.0, 1.0]}]
    mapped = sio._attach_moment_atom_indices(moments, atom_labels=["Fe1", "Mn1"])
    assert "atom_index" not in mapped[0]
    assert mapped[0]["label"] == "Zz9"


def test_find_atom_label_index_exact_prefix_and_none() -> None:
    labels = ["Fe1_1", "Mn1", "Fe2"]
    assert sio._find_atom_label_index(labels, "Mn1") == 1  # exact
    assert sio._find_atom_label_index(labels, "Fe1") == 0  # unique prefix Fe1_
    assert sio._find_atom_label_index(labels, "") is None  # empty
    assert sio._find_atom_label_index(["Fe1_a", "Fe1_b"], "Fe1") is None  # ambiguous


# --------------------------------------------------------------------------
# Header helpers
# --------------------------------------------------------------------------


def test_first_matching_header() -> None:
    assert sio._first_matching_header(["b", "c"], ["a", "c"]) == "c"
    assert sio._first_matching_header(["b"], ["a"]) is None


def test_first_header_with_suffix() -> None:
    assert sio._first_header_with_suffix(["_atom_site_moment.cartn_x"], [".cartn_x"]) == (
        "_atom_site_moment.cartn_x"
    )
    assert sio._first_header_with_suffix(["foo"], [".x"]) is None


# --------------------------------------------------------------------------
# Numeric token parsers
# --------------------------------------------------------------------------


def test_parse_cif_triplet_variants() -> None:
    assert sio._parse_cif_triplet("0, 0, 1/4") == pytest.approx([0.0, 0.0, 0.25])
    assert sio._parse_cif_triplet("0 0 0.25") == pytest.approx([0.0, 0.0, 0.25])
    assert sio._parse_cif_triplet("[0.5 0.5 0.5]") == pytest.approx([0.5, 0.5, 0.5])


def test_parse_cif_triplet_invalid() -> None:
    assert sio._parse_cif_triplet("") is None
    assert sio._parse_cif_triplet("?") is None
    assert sio._parse_cif_triplet("0 0") is None  # wrong arity
    assert sio._parse_cif_triplet("0 0 bad") is None  # non-numeric member


def test_parse_cif_float_variants() -> None:
    assert sio._parse_cif_float("1.25(5)") == pytest.approx(1.25)  # uncertainty
    assert sio._parse_cif_float("1/2") == pytest.approx(0.5)  # fraction
    assert sio._parse_cif_float("-3.0e1") == pytest.approx(-30.0)  # exponent
    assert sio._parse_cif_float(".") is None
    assert sio._parse_cif_float("abc") is None
    assert sio._parse_cif_float("1/0") is None  # zero division


# --------------------------------------------------------------------------
# spins_from_magnetic_structure
# --------------------------------------------------------------------------


def test_spins_from_magnetic_structure_uses_atom_index() -> None:
    magnetic = {
        "moments": [
            {"atom_index": 1, "vector": [0.0, 0.0, 2.5]},
            {"atom_index": 0, "vector": [0.0, 0.0, -1.0]},
        ]
    }
    spins = sio._spins_from_magnetic_structure(magnetic_structure=magnetic, atom_count=2)
    assert spins == pytest.approx([-1.0, 2.5])


def test_spins_from_magnetic_structure_skips_bad_entries() -> None:
    magnetic = {
        "moments": [
            "not-a-dict",
            {"atom_index": "x", "vector": [0, 0, 1]},  # bad index
            {"atom_index": 99, "vector": [0, 0, 1]},  # out of range
            {"atom_index": 0, "vector": [0, 0]},  # short vector
            {"atom_index": 0, "vector": "nope"},  # not a list
            {"atom_index": 0, "vector": [0, 0, 4.0]},  # good
        ]
    }
    spins = sio._spins_from_magnetic_structure(magnetic_structure=magnetic, atom_count=1)
    assert spins == pytest.approx([4.0])


def test_spins_from_magnetic_structure_no_moments_returns_none() -> None:
    assert sio._spins_from_magnetic_structure(magnetic_structure={}, atom_count=2) is None
    assert (
        sio._spins_from_magnetic_structure(
            magnetic_structure={"moments": []}, atom_count=2
        )
        is None
    )


def test_spins_from_magnetic_structure_index_fallback_no_atom_index() -> None:
    magnetic = {
        "moments": [
            {"vector": [0.0, 0.0, 1.0]},
            {"vector": [0.0, 0.0, 2.0]},
        ]
    }
    spins = sio._spins_from_magnetic_structure(magnetic_structure=magnetic, atom_count=2)
    assert spins == pytest.approx([1.0, 2.0])


# --------------------------------------------------------------------------
# Dedupe helpers
# --------------------------------------------------------------------------


def test_dedupe_strings() -> None:
    assert sio._dedupe_strings(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_dedupe_vectors_drops_near_duplicates() -> None:
    out = sio._dedupe_vectors([[0.0, 0.0, 0.25], [0.0, 0.0, 0.25 + 1e-15], [0.5, 0.0, 0.0]])
    assert out == [[0.0, 0.0, 0.25], [0.5, 0.0, 0.0]]


# --------------------------------------------------------------------------
# MCIF top-level: no magnetic metadata returns None branch
# --------------------------------------------------------------------------


def test_mcif_without_magnetic_metadata_has_no_magnetic_structure(tmp_path: Path) -> None:
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
  Si1 Si 0.0 0.0 0.0
  O1 O 0.25 0.25 0.25
""".strip()
    path = tmp_path / "plain.mcif"
    path.write_text(mcif_text, encoding="utf-8")

    loaded = load_structure_file(path)
    assert "magnetic_structure" not in loaded.properties
    assert loaded.properties["source_extension"] == ".mcif"


def test_extract_mcif_magnetic_structure_directly_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "empty.mcif"
    path.write_text("data_x\n_cell_length_a 5\n", encoding="utf-8")
    assert sio._extract_mcif_magnetic_structure(path) is None
