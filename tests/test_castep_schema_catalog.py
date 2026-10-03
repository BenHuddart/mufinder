"""Tests for comprehensive CASTEP schema coverage and alias normalization."""

from __future__ import annotations

import os
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
    write_cell,
    write_param,
)
import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.schema import (
    CatalogSourceInfo,
    SOURCE_KEYWORDS,
    canonical_keyword_name,
    get_keyword_spec,
    keyword_catalog_source_info,
    refresh_keyword_catalog,
    serialization_order,
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
    assert canonical_keyword_name("param", "calculate_polarisation") == "calculate_polarization"
    assert canonical_keyword_name("cell", "kpoints_mp_grid") == "kpoint_mp_grid"

    cell_spec = get_keyword_spec("cell", "kpoints_mp_grid")
    assert cell_spec is not None
    assert cell_spec.name == "kpoint_mp_grid"


def test_refresh_keyword_catalog_uses_updated_env_json(
    tmp_path: Path,
    monkeypatch,
) -> None:
    original_mufinder_env = os.environ.get("MUFINDER_CASTEP_KEYWORDS_JSON")
    original_castep_env = os.environ.get("CASTEP_KEYWORDS_JSON")

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

    if original_mufinder_env is None:
        monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    else:
        monkeypatch.setenv("MUFINDER_CASTEP_KEYWORDS_JSON", original_mufinder_env)
    if original_castep_env is None:
        monkeypatch.delenv("CASTEP_KEYWORDS_JSON", raising=False)
    else:
        monkeypatch.setenv("CASTEP_KEYWORDS_JSON", original_castep_env)
    refresh_keyword_catalog(allow_fallback=True)


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

        with pytest.raises(FileNotFoundError, match="requires a local CASTEP metadata file"):
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
            '"param": {"task": {"option_type": "String", "level": "Basic", "docstring": "Example"}}}' % version,
            encoding="utf-8",
        )
        return path

    managed = _write("managed.json", "25.1")
    override = _write("override.json", "26.1")
    monkeypatch.delenv("MUFINDER_CASTEP_KEYWORDS_JSON", raising=False)
    monkeypatch.delenv("CASTEP_KEYWORDS_JSON", raising=False)
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
    return (
        "{\n"
        '  "castep_version": "test",\n'
        '  "cell": {},\n'
        f'  "param": {{{entries}}}\n'
        "}\n"
    )


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
    default_lattice = next(i for i, line in enumerate(default_lines) if line == "%BLOCK LATTICE_CART")
    default_grid = next(i for i, line in enumerate(default_lines) if line.startswith("KPOINT_MP_GRID"))
    assert default_lattice < default_grid

    # Simulate the module-global reassignment that refresh_keyword_catalog performs, moving
    # kpoint_mp_grid ahead of lattice_cart. The autouse conftest fixture restores the global.
    reordered = (
        "kpoint_mp_grid",
        *(name for name in serialization_order("cell") if name != "kpoint_mp_grid"),
    )
    monkeypatch.setattr(schema_module, "CELL_SERIALIZATION_ORDER", reordered)

    reordered_lines = write_cell(cell_file).splitlines()
    reordered_lattice = next(i for i, line in enumerate(reordered_lines) if line == "%BLOCK LATTICE_CART")
    reordered_grid = next(i for i, line in enumerate(reordered_lines) if line.startswith("KPOINT_MP_GRID"))
    assert reordered_grid < reordered_lattice


def test_starter_catalog_offers_spin_polarized_but_not_net_spin_in_every_workflow() -> None:
    from mufinder.castep_io.catalog_overrides import ALL_WORKFLOWS, KEYWORD_OVERRIDES, KNOWN_WORKFLOWS
    from mufinder.castep_io.gui_metadata import keyword_visible_in_workflow

    _cell, param_specs, _source_names = schema_module._build_starter_schema()

    for workflow in KNOWN_WORKFLOWS:
        assert keyword_visible_in_workflow(
            param_specs["spin_polarized"], workflow_stage=workflow, include_advanced=True
        ), workflow
        # SPIN is the net spin of the cell; MuFinder sets spins per atom instead,
        # and CASTEP stops when an explicit SPIN disagrees with their sum.
        assert not keyword_visible_in_workflow(
            param_specs["spin"], workflow_stage=workflow, include_advanced=True
        ), workflow
    # With a full castep_keywords.json, SPIN stays addable but is never a default row.
    spin_override = KEYWORD_OVERRIDES["param"]["spin"]
    assert spin_override["advanced_workflows"] == ALL_WORKFLOWS
    assert not spin_override.get("basic_workflows")
