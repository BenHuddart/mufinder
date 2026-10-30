"""Tests for comprehensive CASTEP schema coverage and alias normalization."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mufinder.assets import starter_keyword_catalog_path
from mufinder.castep_io import (
    AtomicPosition,
    CellFile,
    KeywordAssignment,
    LatticeCart,
    ParamFile,
    PositionBlock,
    UnknownKeywordError,
    write_cell,
    write_param,
)
import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.schema import (
    BlockSchema,
    CatalogSourceInfo,
    KeywordSpec,
    SOURCE_KEYWORDS,
    canonical_keyword_name,
    get_keyword_spec,
    keyword_catalog_source_info,
    refresh_keyword_catalog,
    serialization_order,
)
from mufinder.castep_io.starter_catalog import (
    _starter_alias_map,
    _starter_parse_requirements,
    _starter_sequence,
)
from tests.support.keyword_catalogs import (
    catalog_entry,
    isolate_catalog_sources,
    refresh_from_catalog,
    write_complete_catalog,
)


def test_repo_does_not_vendor_castep_keyword_json_copy() -> None:
    assert not Path("src/mufinder/castep_io/base_catalog.json").exists()


def test_packaged_starter_catalog_matches_docs_copy() -> None:
    docs_catalog = Path("docs/castep/keyword_catalog.yaml")
    packaged_catalog = starter_keyword_catalog_path()

    assert packaged_catalog.name == "keyword_catalog.yaml"
    assert packaged_catalog.read_text(encoding="utf-8") == docs_catalog.read_text(encoding="utf-8")


def test_runtime_schema_covers_every_keyword_from_source_catalog() -> None:
    for file_kind, source_names in SOURCE_KEYWORDS.items():
        missing = [name for name in source_names if get_keyword_spec(file_kind, name) is None]
        assert missing == []


def test_aliases_canonicalize_to_single_runtime_keyword() -> None:
    assert canonical_keyword_name("param", "spin_polarised") == "spin_polarized"
    assert canonical_keyword_name("cell", "kpoints_mp_grid") == "kpoint_mp_grid"

    cell_spec = get_keyword_spec("cell", "kpoints_mp_grid")
    assert cell_spec is not None
    assert cell_spec.name == "kpoint_mp_grid"


def test_refresh_keyword_catalog_uses_updated_env_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    custom_catalog = tmp_path / "custom_keywords.json"
    custom_catalog.write_text(
        (
            "{\n"
            '  "castep_version": "test",\n'
            '  "cell": {"abc": {"option_type": "Text", "level": "Basic", "docstring": "Example"}},\n'
            '  "param": {"task": {"option_type": "Text", "level": "Basic", "docstring": "Example"}}\n'
            "}\n"
        ),
        encoding="utf-8",
    )

    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(custom_catalog))
    refresh_keyword_catalog()

    source = keyword_catalog_source_info()
    assert source is not None
    assert source.origin == "env_json:MUFINDER_CASTEP_KEYWORDS_JSON"
    assert source.path == custom_catalog


def test_refresh_keyword_catalog_falls_back_to_bundled_starter_catalog_when_missing(
    tmp_path: Path,
) -> None:
    original_candidate_sources = schema_module._candidate_catalog_sources

    try:
        schema_module._candidate_catalog_sources = lambda: (
            CatalogSourceInfo(path=tmp_path / "missing.json", origin="test_missing"),
        )

        source = refresh_keyword_catalog(allow_fallback=True)

        assert source is not None
        assert source.origin == "bundled_starter_catalog"
        assert source.path.name == "keyword_catalog.yaml"
        assert get_keyword_spec("param", "task") is not None
        assert get_keyword_spec("cell", "kpoints_mp_grid") is not None
    finally:
        schema_module._candidate_catalog_sources = original_candidate_sources
        refresh_keyword_catalog(allow_fallback=True)


def test_refresh_keyword_catalog_stays_strict_without_fallback_when_missing(
    tmp_path: Path,
) -> None:
    original_candidate_sources = schema_module._candidate_catalog_sources

    try:
        schema_module._candidate_catalog_sources = lambda: (
            CatalogSourceInfo(path=tmp_path / "missing.json", origin="test_missing"),
        )

        with pytest.raises(FileNotFoundError, match=r"Open Setup > CASTEP"):
            refresh_keyword_catalog()
    finally:
        schema_module._candidate_catalog_sources = original_candidate_sources
        refresh_keyword_catalog(allow_fallback=True)


def test_corrupt_configured_catalog_falls_back_and_reports_why(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """A corrupt configured catalog used to abort startup with JSONDecodeError."""

    corrupt = tmp_path / "castep_keywords.json"
    corrupt.write_text("{bad", encoding="utf-8")
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (CatalogSourceInfo(path=corrupt, origin="setup"),),
    )

    refresh_keyword_catalog(allow_fallback=True)

    status = schema_module.keyword_catalog_status()
    assert status.is_starter is True
    assert status.origin == "bundled_starter_catalog"
    assert status.castep_version == ""
    assert status.issue is not None and "is not valid JSON" in status.issue
    assert get_keyword_spec("param", "task") is not None


def test_setup_catalog_is_used_but_exported_env_var_overrides_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def _write(name: str, version: str) -> Path:
        path = tmp_path / name
        path.write_text(
            '{"castep_version": "%s", "cell": {}, '
            '"param": {"task": {"option_type": "String", "level": "Basic", "docstring": "Example"}}}'
            % version,
            encoding="utf-8",
        )
        return path

    managed = _write("managed.json", "25.1")
    override = _write("override.json", "26.1")
    monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    monkeypatch.setenv(schema_module.SETUP_KEYWORDS_JSON_ENV_VAR, str(managed))

    refresh_keyword_catalog()
    status = schema_module.keyword_catalog_status()
    assert (status.origin, status.path, status.castep_version) == ("setup", managed, "25.1")
    assert status.is_starter is False
    assert status.issue is None

    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(override))
    refresh_keyword_catalog()
    status = schema_module.keyword_catalog_status()
    assert status.origin == "env_json:MUFINDER_CASTEP_KEYWORDS_JSON"
    assert status.castep_version == "26.1"


def test_retired_catalog_sources_are_ignored_and_starter_catalog_is_used(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Only the override and the installed catalog are sources; nothing is discovered."""

    tree_catalog = Path("Test/Utilities/ReadTS/castep_keywords.json")
    castep_root = tmp_path / "castep-root"
    write_complete_catalog(castep_root / tree_catalog)
    home = tmp_path / "home"
    write_complete_catalog(home / "Source" / "CASTEP-26.1" / tree_catalog)
    legacy_json = write_complete_catalog(tmp_path / "legacy" / "castep_keywords.json")

    monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    monkeypatch.delenv(schema_module.SETUP_KEYWORDS_JSON_ENV_VAR, raising=False)
    monkeypatch.setenv("CASTEP_KEYWORDS_JSON", str(legacy_json))
    monkeypatch.setenv("MUFINDER_CASTEP_ROOT", str(castep_root))
    monkeypatch.setenv("CASTEP_ROOT", str(castep_root))
    monkeypatch.setenv("HOME", str(home))

    assert schema_module._candidate_catalog_sources() == ()

    refresh_keyword_catalog(allow_fallback=True)
    status = schema_module.keyword_catalog_status()
    assert status.is_starter is True
    assert status.origin == "bundled_starter_catalog"
    assert status.issue is None

    with pytest.raises(FileNotFoundError, match=r"Open Setup > CASTEP") as excinfo:
        refresh_keyword_catalog()
    assert "no catalog is configured" in str(excinfo.value)


def test_override_and_installed_catalog_are_tried_in_that_order(
    tmp_path: Path,
    monkeypatch,
) -> None:
    override = tmp_path / "override.json"
    installed = tmp_path / "installed.json"
    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(override))
    monkeypatch.setenv(schema_module.SETUP_KEYWORDS_JSON_ENV_VAR, str(installed))

    assert schema_module._candidate_catalog_sources() == (
        CatalogSourceInfo(path=override, origin="env_json:MUFINDER_CASTEP_KEYWORDS_JSON"),
        CatalogSourceInfo(path=installed, origin="setup"),
    )

    # The same file named twice is tried once, as the override.
    monkeypatch.setenv(schema_module.SETUP_KEYWORDS_JSON_ENV_VAR, str(override))
    assert schema_module._candidate_catalog_sources() == (
        CatalogSourceInfo(path=override, origin="env_json:MUFINDER_CASTEP_KEYWORDS_JSON"),
    )


def test_unknown_keyword_error_points_at_castep_setup_on_starter_catalog(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Regression: CASTEP writes QUANTISATION_AXIS into -out.cell after a Hubbard U run."""

    from mufinder.castep_io.cell_parser import CellParseError, parse_cell_text

    cell_text = (
        "%BLOCK lattice_cart\n5 0 0\n0 5 0\n0 0 5\n%ENDBLOCK lattice_cart\n"
        "%BLOCK positions_frac\nH 0 0 0 SPIN= 1.000\n%ENDBLOCK positions_frac\n"
        "QUANTISATION_AXIS :     0.000000    0.000000    1.000000\n"
    )
    monkeypatch.setattr(schema_module, "_candidate_catalog_sources", lambda: ())
    refresh_keyword_catalog(allow_fallback=True)

    with pytest.raises(CellParseError) as excinfo:
        parse_cell_text(cell_text)
    message = str(excinfo.value)
    assert message.startswith("Unsupported CASTEP cell keyword 'quantisation_axis'.")
    assert "Setup > CASTEP" in message
    assert "schema.py" not in message

    full = tmp_path / "castep_keywords.json"
    full.write_text(
        '{"castep_version": "26.11", "cell": {'
        '"quantisation_axis": {"option_type": "Real Vector", "level": "Basic", "docstring": "Axis"},'
        '"quantization_axis": {"option_type": "Real Vector", "level": "Basic", "docstring": "Axis"}'
        '}, "param": {}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (CatalogSourceInfo(path=full, origin="setup"),),
    )
    refresh_keyword_catalog()

    assert parse_cell_text(cell_text).atomic_spins == (1.0,)


def _stub_param_catalog(order: tuple[str, ...]) -> str:
    entries = ", ".join(
        f'"{name}": {{"option_type": "String", "level": "Basic", "docstring": "Example"}}'
        for name in order
    )
    return f'{{\n  "castep_version": "test",\n  "cell": {{}},\n  "param": {{{entries}}}\n}}\n'


def test_param_writer_follows_serialization_order_after_runtime_refresh(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """write_param must reflect a catalog order set by a *runtime* refresh, not a stale copy.

    Regression for the writers binding ``PARAM_SERIALIZATION_ORDER`` by value at import time:
    reconfiguring the CASTEP location at runtime reassigns the module global, and the writer
    must pick up the new order rather than the order discovered when it was first imported.
    """

    # Input order is the opposite of the first catalog order, so a stale by-value binding
    # (under which neither keyword is found and both fall through to write_param's
    # input-order tail) renders them in input order and fails the very first assertion.
    param_file = ParamFile(
        keywords=(
            KeywordAssignment("beta", "B"),
            KeywordAssignment("alpha", "A"),
        )
    )

    first = tmp_path / "first.json"
    first.write_text(_stub_param_catalog(("alpha", "beta")), encoding="utf-8")
    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(first))
    refresh_keyword_catalog()
    order = serialization_order("param")
    assert order.index("alpha") < order.index("beta")
    assert write_param(param_file).splitlines() == ["ALPHA : A", "BETA : B"]

    second = tmp_path / "second.json"
    second.write_text(_stub_param_catalog(("beta", "alpha")), encoding="utf-8")
    monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", str(second))
    refresh_keyword_catalog()
    order = serialization_order("param")
    assert order.index("beta") < order.index("alpha")
    assert write_param(param_file).splitlines() == ["BETA : B", "ALPHA : A"]


def test_cell_writer_follows_serialization_order_after_runtime_refresh(
    monkeypatch,
) -> None:
    """write_cell must consult the live serialization order after a runtime refresh."""

    cell_file = CellFile(
        lattice_cart=LatticeCart(
            vectors=((5.43, 0.0, 0.0), (0.0, 5.43, 0.0), (0.0, 0.0, 5.43)),
            unit="ang",
        ),
        positions_frac=PositionBlock(positions=(AtomicPosition("Si", (0.0, 0.0, 0.0)),)),
        kpoint_mp_grid=(4, 4, 4),
    )

    default_lines = write_cell(cell_file).splitlines()
    default_lattice = next(
        i for i, line in enumerate(default_lines) if line == "%BLOCK LATTICE_CART"
    )
    default_grid = next(
        i for i, line in enumerate(default_lines) if line.startswith("KPOINT_MP_GRID")
    )
    assert default_lattice < default_grid

    # Simulate the module-global reassignment that refresh_keyword_catalog performs, moving
    # kpoint_mp_grid ahead of lattice_cart. The autouse conftest fixture restores the global.
    reordered = (
        "kpoint_mp_grid",
        *(name for name in serialization_order("cell") if name != "kpoint_mp_grid"),
    )
    monkeypatch.setattr(schema_module, "CELL_SERIALIZATION_ORDER", reordered)

    reordered_lines = write_cell(cell_file).splitlines()
    reordered_lattice = next(
        i for i, line in enumerate(reordered_lines) if line == "%BLOCK LATTICE_CART"
    )
    reordered_grid = next(
        i for i, line in enumerate(reordered_lines) if line.startswith("KPOINT_MP_GRID")
    )
    assert reordered_grid < reordered_lattice


def test_starter_catalog_offers_spin_polarized_but_not_net_spin_in_every_workflow() -> None:
    from mufinder.castep_io.catalog_overrides import (
        ALL_WORKFLOWS,
        KEYWORD_OVERRIDES,
        KNOWN_WORKFLOWS,
    )
    from mufinder.castep_io.gui_metadata import keyword_visible_in_workflow

    _cell, param_specs, _source_names = schema_module._build_starter_schema()

    for workflow in KNOWN_WORKFLOWS:
        # Hyperfine Coupling always runs spin polarised, so it does not offer the choice.
        assert keyword_visible_in_workflow(
            param_specs["spin_polarized"], workflow_stage=workflow, include_advanced=True
        ) == (workflow != "hyperfine_coupling"), workflow
        # SPIN is the net spin of the cell; MuFinder sets spins per atom instead,
        # and CASTEP stops when an explicit SPIN disagrees with their sum.
        assert not keyword_visible_in_workflow(
            param_specs["spin"], workflow_stage=workflow, include_advanced=True
        ), workflow
    # With a full castep_keywords.json, SPIN stays addable but is never a default row.
    spin_override = KEYWORD_OVERRIDES["param"]["spin"]
    assert spin_override["advanced_workflows"] == ALL_WORKFLOWS
    assert not spin_override.get("basic_workflows")


# --- Keyword specs inferred from a CASTEP catalog ------------------------------


def _param(name: str) -> KeywordSpec:
    spec = get_keyword_spec("param", name)
    assert spec is not None
    return spec


def test_catalog_docstrings_populate_labels_examples_and_allowed_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docstring = (
        "First sentence. Second sentence.\n"
        "Allowed values: TRUE or FALSE, AUTO\n"
        "Default value: TRUE\n"
        "Example: FLAG : TRUE\n"
    )
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={},
        param={
            "xc_test_flag": catalog_entry("String", docstring),
            "test_free_text": catalog_entry(
                "String", "Allowed values: libxc-backed supported values; see docs"
            ),
        },
    )

    spec = _param("xc_test_flag")
    assert spec.label == "XC Test Flag"
    assert spec.description == "First sentence."
    assert spec.example == "FLAG : TRUE"
    assert spec.allowed_values == ("TRUE", "FALSE", "AUTO")
    assert spec.value_type == "enum"
    assert spec.default_value == "TRUE"
    assert _param("test_free_text").allowed_values == ()
    assert _param("test_free_text").value_type == "text"


def test_catalog_docstrings_set_typed_defaults_and_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={},
        param={
            "test_true": catalog_entry("Boolean (Logical)", "Default value: TRUE"),
            "test_false": catalog_entry("Boolean (Logical)", "Default value: false"),
            "test_enum": catalog_entry("String", "Allowed values: PBE, LDA\nDefault value: pbe"),
            "test_int": catalog_entry("Integer", "Default value: 42"),
            "test_physical": catalog_entry("Physical", "Default value: 0.05 eV"),
            "test_example_unit": catalog_entry("Physical", "Example: CUT_OFF_ENERGY : 340 eV"),
            "test_real": catalog_entry("Real", "Default value: 1e-6"),
            "test_text": catalog_entry("String", "Default value: text-value"),
            "test_workflow_text": catalog_entry("String", "Default value: set by workflow"),
        },
    )

    assert _param("test_true").default_value is True
    assert _param("test_false").default_value is False
    assert _param("test_enum").default_value == "PBE"
    assert _param("test_int").default_value == 42
    assert _param("test_physical").default_value == pytest.approx(0.05)
    assert _param("test_physical").unit == "eV"
    assert _param("test_example_unit").unit == "eV"
    assert _param("test_real").unit is None
    assert _param("test_real").value_type == "float"
    assert _param("test_text").default_value == "text-value"
    assert _param("test_workflow_text").default_value is None


def test_catalog_keyword_names_set_value_types_categories_and_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={
            "test_kpoint_grid": catalog_entry("Integer Vector"),
            "symmetry_test": catalog_entry("Defined"),
            "species_test": catalog_entry("Block"),
        },
        param={
            "geom_test_tol": catalog_entry("Physical"),
            "smearing_test": catalog_entry("Physical"),
            "md_test_temperature": catalog_entry("Physical"),
            "phonon_test_grid": catalog_entry("Integer Vector"),
            "test_general": catalog_entry("String"),
        },
    )

    grid = get_keyword_spec("cell", "test_kpoint_grid")
    flag = get_keyword_spec("cell", "symmetry_test")
    block = get_keyword_spec("cell", "species_test")
    assert grid is not None and (grid.kind, grid.value_type, grid.category) == (
        "scalar",
        "int_triplet",
        "kpoints",
    )
    assert flag is not None and (flag.kind, flag.value_type, flag.category) == (
        "flag",
        "bool",
        "symmetry",
    )
    assert block is not None and (block.kind, block.category) == ("block", "pseudopotentials")
    # A block with no curated layout keeps raw rows.
    assert block.block_schema is None

    assert _param("geom_test_tol").category == "geometry"
    assert _param("geom_test_tol").allowed_tasks == ("GEOMETRYOPTIMIZATION",)
    assert _param("smearing_test").category == "electronic"
    assert _param("md_test_temperature").allowed_tasks == ("MOLECULARDYNAMICS",)
    assert _param("phonon_test_grid").allowed_tasks == ("PHONON", "PHONON+EFIELD", "THERMODYNAMICS")
    assert _param("test_general").allowed_tasks == ()
    assert _param("test_general").category == "general"


def test_catalog_blocks_pick_up_curated_layouts_and_aliases_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={"positions_frac": catalog_entry("Block"), "lattice_abc": catalog_entry("Block")},
        param={
            "test_polarisation": catalog_entry(
                "String", "Default value: Determined from TEST_POLARIZATION"
            ),
            "test_polarization": catalog_entry("String"),
            "test_isation": catalog_entry("String"),
            "test_ization": catalog_entry("String"),
            "spin_polarised": catalog_entry("Boolean (Logical)"),
            "spin_polarized": catalog_entry("Boolean (Logical)"),
            "test_polarised": catalog_entry("String"),
            "test_polarized": catalog_entry("String"),
        },
    )

    positions = get_keyword_spec("cell", "positions_frac")
    lattice_abc = get_keyword_spec("cell", "lattice_abc")
    assert positions is not None and isinstance(positions.block_schema, BlockSchema)
    assert lattice_abc is not None and isinstance(lattice_abc.block_schema, BlockSchema)
    assert lattice_abc.block_schema.fixed_row_count == 2
    assert lattice_abc.block_schema.row_labels == ("Lengths", "Angles")

    assert canonical_keyword_name("param", "test_polarisation") == "test_polarization"
    assert canonical_keyword_name("param", "test_isation") == "test_ization"
    assert canonical_keyword_name("param", "spin_polarised") == "spin_polarized"
    assert canonical_keyword_name("param", "test_polarised") == "test_polarized"
    assert "test_polarisation" in _param("test_polarization").aliases


def test_refresh_skips_unusable_catalogs_and_reports_explicit_ones(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_catalog = tmp_path / "custom.json"
    custom_catalog.write_text(
        json.dumps(
            {
                "castep_version": "test",
                "cell": {"ABC": catalog_entry("Text")},
                "param": {"TASK": catalog_entry("String")},
            }
        ),
        encoding="utf-8",
    )
    # A file that is not a keyword catalog is skipped, not fatal: the next
    # candidate is used and the reason is kept for the UI.
    broken_catalog = tmp_path / "broken.json"
    broken_catalog.write_text(json.dumps({"cell": [], "param": {}}), encoding="utf-8")
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (
            CatalogSourceInfo(path=broken_catalog, origin="env_json:test"),
            CatalogSourceInfo(path=custom_catalog, origin="setup"),
        ),
    )

    refresh_keyword_catalog()

    status = schema_module.keyword_catalog_status()
    assert (status.origin, status.path, status.castep_version) == ("setup", custom_catalog, "test")
    assert get_keyword_spec("cell", "abc") is not None
    assert status.issue is not None
    assert "broken.json" in status.issue
    assert "no 'cell' keyword section" in status.issue

    corrupt_catalog = tmp_path / "corrupt.json"
    corrupt_catalog.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (
            CatalogSourceInfo(path=corrupt_catalog, origin="setup"),
            CatalogSourceInfo(path=tmp_path / "missing.json", origin="env_json:test"),
        ),
    )

    with pytest.raises(FileNotFoundError, match=r"Open Setup > CASTEP"):
        refresh_keyword_catalog()
    issue = schema_module.keyword_catalog_status().issue
    assert issue is not None
    assert "is not valid JSON" in issue
    assert "missing.json' no longer exists" in issue


def test_starter_catalog_entries_parse_requirement_strings_and_renamed_keywords() -> None:
    # The bundled starter catalog lists requirements as YAML lists and has no
    # renamed entries, so these spellings are exercised on the entry parsers.
    alias_map = {
        "spin_polarised": "spin_polarized",
        "calculate_polarisation": "calculate_polarization",
    }

    assert _starter_sequence("TASK=GEOMETRYOPTIMIZATION") == ("TASK=GEOMETRYOPTIMIZATION",)
    requires, requires_any_of, allowed_tasks = _starter_parse_requirements(
        "param",
        ["TASK=GEOMETRYOPTIMIZATION", "spin_polarised", "calculate_polarisation or spin_polarised"],
        alias_map=alias_map,
    )
    assert requires == ("spin_polarized",)
    assert requires_any_of == ("calculate_polarization", "spin_polarized")
    assert allowed_tasks == ("GEOMETRYOPTIMIZATION",)

    starter_aliases = _starter_alias_map(
        "param",
        [
            {"source_name": "calculate_polarisation", "canonical_name": "calculate_polarization"},
            {"source_name": "calculate_polarization", "canonical_name": "calculate_polarization"},
            {"source_name": "spin_polarised", "canonical_name": "spin_polarised"},
            {"source_name": "spin_polarized", "canonical_name": "spin_polarized"},
        ],
    )
    assert starter_aliases["calculate_polarisation"] == "calculate_polarization"
    assert starter_aliases["spin_polarised"] == "spin_polarized"


# --- Keywords kept out of Add Parameter ----------------------------------------

_HIDDEN_KEYWORDS = (
    ("cell", "unit_cell"),
    ("param", "num_farms"),
    ("param", "magres_task"),
    ("param", "devel_code"),
    ("param", "tssearch_lst_max_iter"),
)


@pytest.mark.parametrize(("file_kind", "name"), _HIDDEN_KEYWORDS)
def test_starter_catalog_marks_internal_keywords_hidden(file_kind: str, name: str) -> None:
    spec = get_keyword_spec(file_kind, name)

    assert spec is not None
    assert spec.user_visible is False


def test_keywords_are_user_visible_unless_an_override_hides_them() -> None:
    hidden = set(_HIDDEN_KEYWORDS)
    for file_kind in ("cell", "param"):
        for name in SOURCE_KEYWORDS[file_kind]:
            spec = get_keyword_spec(file_kind, name)
            assert spec is not None
            if (file_kind, spec.name) not in hidden:
                assert spec.user_visible, (file_kind, spec.name)


def test_full_catalog_applies_the_hidden_flag_without_adding_keywords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # This catalog lacks MAGRES_TASK and TSSEARCH_LST_MAX_ITER; their
    # visibility-only overrides must not invent them.
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={"unit_cell": catalog_entry("String")},
        param={
            "num_farms": catalog_entry("Integer"),
            "devel_code": catalog_entry("Block"),
            "iprint": catalog_entry("Integer"),
        },
    )

    assert _param("num_farms").user_visible is False
    assert _param("devel_code").user_visible is False
    assert _param("iprint").user_visible is True
    unit_cell = get_keyword_spec("cell", "unit_cell")
    assert unit_cell is not None and unit_cell.user_visible is False
    assert get_keyword_spec("param", "magres_task") is None
    assert get_keyword_spec("param", "tssearch_lst_max_iter") is None


def test_hidden_keywords_still_parse_and_write() -> None:
    from mufinder.castep_io.cell_parser import parse_cell_text
    from mufinder.castep_io.param_parser import parse_param_text

    parsed = parse_param_text(
        "NUM_FARMS : 2\nTSSEARCH_LST_MAX_ITER : 20\n"
        "%BLOCK DEVEL_CODE\nfoo=1\n%ENDBLOCK DEVEL_CODE\n"
    )
    names = {item.name for item in parsed.param_file.keywords}
    assert {"num_farms", "tssearch_lst_max_iter"} <= names
    assert [block.name for block in parsed.param_file.blocks] == ["devel_code"]
    assert parsed.unmapped == ()
    rendered = write_param(ParamFile(keywords=parsed.param_file.keywords))
    assert "NUM_FARMS : 2" in rendered

    cell = parse_cell_text(
        "%BLOCK LATTICE_CART\n5 0 0\n0 5 0\n0 0 5\n%ENDBLOCK LATTICE_CART\n"
        "%BLOCK POSITIONS_FRAC\nSi 0 0 0\n%ENDBLOCK POSITIONS_FRAC\n"
        "UNIT_CELL 0.0 0.0 0.0\n"
    )
    assert any(item.name == "unit_cell" for item in cell.cell_file.scalars)


def _refresh_with_retired_keywords(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={
            "bs_kpoint_mp_grid": catalog_entry("Integer Vector", level="Dummy"),
            "kpoint_mp_grid": catalog_entry("Integer Vector"),
        },
        param={
            "md_nose_t": catalog_entry(
                "Physical",
                "NB This is an OBSOLETE form - please use MD_ION_T instead.",
                level="Intermediate",
            ),
            "bs_max_iter": catalog_entry("Integer", level="Dummy"),
            # Naming another keyword as obsolete does not hide this one.
            "md_ion_t": catalog_entry(
                "Physical", "Replaces the OBSOLETE MD_NOSE_T.", level="Intermediate"
            ),
            "iprint": catalog_entry("Integer", "Replaces an obsolete verbosity flag."),
            "xc_vxc_deriv_epsilon": catalog_entry(
                "Real", "OBSOLETE: This keyword was once used.", level="Expert"
            ),
        },
    )


def test_full_catalog_hides_obsolete_and_dummy_level_keywords(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refresh_with_retired_keywords(tmp_path, monkeypatch)

    assert _param("md_nose_t").user_visible is False
    assert _param("xc_vxc_deriv_epsilon").user_visible is False
    assert _param("bs_max_iter").user_visible is False
    assert _param("md_ion_t").user_visible is True
    assert _param("iprint").user_visible is True
    dummy_cell = get_keyword_spec("cell", "bs_kpoint_mp_grid")
    assert dummy_cell is not None and dummy_cell.user_visible is False
    kpoint_grid = get_keyword_spec("cell", "kpoint_mp_grid")
    assert kpoint_grid is not None and kpoint_grid.user_visible is True


def test_obsolete_and_dummy_level_keywords_still_parse_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mufinder.castep_io.param_parser import parse_param_text

    _refresh_with_retired_keywords(tmp_path, monkeypatch)

    parsed = parse_param_text("MD_NOSE_T : 0.1 ps\nBS_MAX_ITER : 40\n")

    assert parsed.unmapped == ()
    assert {item.name for item in parsed.param_file.keywords} == {"md_nose_t", "bs_max_iter"}
    rendered = write_param(ParamFile(keywords=parsed.param_file.keywords))
    assert "MD_NOSE_T : 0.1 ps" in rendered
    assert "BS_MAX_ITER : 40" in rendered


def test_an_override_can_show_a_keyword_the_catalog_marks_hidden(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        schema_module.KEYWORD_OVERRIDES["param"], "bs_max_iter", {"user_visible": True}
    )

    _refresh_with_retired_keywords(tmp_path, monkeypatch)

    assert _param("bs_max_iter").user_visible is True


# --- CALCULATE_POLARIZATION is outside the starter catalog ----------------------


def test_starter_catalog_does_not_offer_calculate_polarization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    isolate_catalog_sources(monkeypatch)

    assert get_keyword_spec("param", "calculate_polarization") is None
    assert get_keyword_spec("param", "calculate_polarisation") is None
    assert "calculate_polarization" not in SOURCE_KEYWORDS["param"]


def test_starter_catalog_reads_calculate_polarization_as_unmapped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mufinder.castep_io.param_parser import parse_param_text

    isolate_catalog_sources(monkeypatch)

    parsed = parse_param_text("TASK : SINGLEPOINT\nCALCULATE_POLARIZATION : TRUE\n")

    assert [item.name for item in parsed.param_file.keywords] == ["task"]
    assert [item.canonical_name for item in parsed.unmapped] == ["calculate_polarization"]
    with pytest.raises(UnknownKeywordError, match="Setup > CASTEP"):
        write_param(
            ParamFile(keywords=(KeywordAssignment(name="calculate_polarization", value=True),))
        )


def test_full_catalog_keeps_calculate_polarization_and_its_english_spelling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docstring = "Specifies whether or not to perform a polarisation analysis."
    refresh_from_catalog(
        tmp_path,
        monkeypatch,
        cell={},
        param={
            "calculate_polarisation": catalog_entry("Boolean (Logical)", docstring),
            "calculate_polarization": catalog_entry("Boolean (Logical)", docstring),
        },
    )

    spec = _param("calculate_polarization")
    assert spec.user_visible is True
    assert spec.aliases == ("calculate_polarisation",)
    assert canonical_keyword_name("param", "calculate_polarisation") == "calculate_polarization"
