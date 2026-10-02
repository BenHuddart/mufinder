"""Targeted unit tests for CASTEP schema helper coverage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import mufinder.castep_io.schema as schema_module
from mufinder.castep_io.schema import (
    BlockSchema,
    CatalogSourceInfo,
    _apply_override,
    _auto_alias_map,
    _build_block_schema,
    _clean_allowed_values_segment,
    _extract_example,
    _first_sentence,
    _humanize,
    _infer_allowed_tasks,
    _infer_category,
    _infer_unit,
    _infer_value_type,
    _load_base_catalog,
    _parse_allowed_values,
    _parse_default_value,
    _starter_alias_map,
    _starter_default_value,
    _starter_kind_value_type,
    _starter_option_type,
    _starter_parse_requirements,
    _starter_resolve_name,
    _starter_sequence,
    _starter_source_order,
)


def test_schema_docstring_parsers_cover_common_cases() -> None:
    docstring = (
        "First sentence. Second sentence.\n"
        "Allowed values: TRUE or FALSE, AUTO\n"
        "Default value: TRUE\n"
        "Example: FLAG : TRUE\n"
    )

    assert _humanize("xc_functional") == "XC Functional"
    assert _first_sentence(docstring) == "First sentence."
    assert _extract_example(docstring) == "FLAG : TRUE"
    assert _clean_allowed_values_segment("A\n  B   C") == "A B C"
    assert _parse_allowed_values(docstring) == ("TRUE", "FALSE", "AUTO")
    assert _parse_allowed_values("Allowed values: libxc-backed supported values; see docs") == ()


def test_schema_default_and_unit_inference_cover_multiple_types() -> None:
    assert _parse_default_value("Default value: TRUE", value_type="bool", option_type="Boolean (Logical)") is True
    assert _parse_default_value("Default value: false", value_type="bool", option_type="Boolean (Logical)") is False
    assert _parse_default_value("Default value: pbe", value_type="enum", option_type="String") == "PBE"
    assert _parse_default_value("Default value: 42", value_type="int", option_type="Integer") == 42
    assert _parse_default_value("Default value: 0.05 eV", value_type="physical", option_type="Physical") == pytest.approx(0.05)
    assert _parse_default_value("Default value: text-value", value_type="text", option_type="String") == "text-value"
    assert _parse_default_value("Default value: set by workflow", value_type="text", option_type="String") is None
    assert _infer_unit("Example: CUT_OFF_ENERGY : 340 eV", "Physical") == "eV"
    assert _infer_unit("Default value: 1e-6 eV", "Physical") == "eV"
    assert _infer_unit("Default value: 1e-6", "Real") is None


def test_schema_inference_helpers_cover_categories_tasks_and_types() -> None:
    assert _infer_value_type("kpoint_mp_grid", "Integer Vector", ()) == ("scalar", "int_triplet")
    assert _infer_value_type("symmetry_generate", "Defined", ()) == ("flag", "bool")
    assert _infer_value_type("basis_precision", "String", ("FINE", "COARSE")) == ("scalar", "enum")
    assert _infer_value_type("unit_cell", "String", ()) == ("scalar", "text")
    assert _infer_category("cell", "species_pot") == "pseudopotentials"
    assert _infer_category("param", "geom_force_tol") == "geometry"
    assert _infer_category("param", "smearing_width") == "electronic"
    assert _infer_allowed_tasks("md_temperature") == ("MOLECULARDYNAMICS",)
    assert _infer_allowed_tasks("phonon_kpoint_mp_grid") == ("PHONON", "PHONON+EFIELD", "THERMODYNAMICS")
    assert _infer_allowed_tasks("task") == ()


def test_discover_and_load_base_catalog_handle_success_and_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    custom_catalog = tmp_path / "custom.json"
    custom_catalog.write_text(
        json.dumps(
            {
                "castep_version": "test",
                "cell": {"ABC": {"option_type": "Text", "level": "Basic", "docstring": "Example"}},
                "param": {"TASK": {"option_type": "String", "level": "Basic", "docstring": "Example"}},
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (CatalogSourceInfo(path=custom_catalog, origin="env_json:test"),),
    )

    payload = _load_base_catalog()

    assert schema_module.CATALOG_SOURCE_INFO.path == custom_catalog
    assert schema_module.CATALOG_ISSUE is None
    assert schema_module.CATALOG_CASTEP_VERSION == "test"
    assert payload["cell"][0]["name"] == "abc"
    assert payload["param"][0]["name"] == "task"

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
    payload = _load_base_catalog()
    assert schema_module.CATALOG_SOURCE_INFO.origin == "setup"
    assert payload["param"][0]["name"] == "task"
    assert "broken.json" in schema_module.CATALOG_ISSUE
    assert "no 'cell' keyword section" in schema_module.CATALOG_ISSUE

    corrupt_catalog = tmp_path / "corrupt.json"
    corrupt_catalog.write_text("{not json", encoding="utf-8")
    missing_catalog = tmp_path / "missing.json"
    monkeypatch.setattr(
        schema_module,
        "_candidate_catalog_sources",
        lambda: (
            CatalogSourceInfo(path=corrupt_catalog, origin="setup"),
            CatalogSourceInfo(path=missing_catalog, origin="env_json:test"),
            CatalogSourceInfo(path=tmp_path / "undiscovered.json", origin="auto_discovery"),
        ),
    )
    with pytest.raises(FileNotFoundError, match="requires a local CASTEP metadata file"):
        _load_base_catalog()
    assert "is not valid JSON" in schema_module.CATALOG_ISSUE
    assert "missing.json' no longer exists" in schema_module.CATALOG_ISSUE
    # Auto-discovery misses are routine, not something to report.
    assert "undiscovered.json" not in schema_module.CATALOG_ISSUE


def test_schema_override_and_alias_helpers_cover_manual_and_auto_paths() -> None:
    spec = schema_module.KeywordSpec(
        name="positions_frac",
        file="cell",
        kind="block",
        value_type="block",
        block_schema=None,
    )
    overridden = _apply_override(spec, {"label": "Fractional Positions"})

    assert overridden.label == "Fractional Positions"
    assert isinstance(overridden.block_schema, BlockSchema)

    entries = [
        {"name": "calculate_polarisation", "docstring": "Default value: Determined from CALCULATE_POLARIZATION"},
        {"name": "calculate_polarization", "docstring": "Example"},
        {"name": "spin_polarised", "docstring": "Example"},
        {"name": "spin_polarized", "docstring": "Example"},
    ]
    alias_map = _auto_alias_map("param", entries)
    assert alias_map["calculate_polarisation"] == "calculate_polarization"
    assert alias_map["spin_polarised"] == "spin_polarized"

    starter_entries = [
        {"source_name": "calculate_polarisation", "canonical_name": "calculate_polarization"},
        {"source_name": "calculate_polarization", "canonical_name": "calculate_polarization"},
        {"source_name": "spin_polarised", "canonical_name": "spin_polarised"},
        {"source_name": "spin_polarized", "canonical_name": "spin_polarized"},
    ]
    starter_aliases = _starter_alias_map("param", starter_entries)
    assert starter_aliases["calculate_polarisation"] == "calculate_polarization"
    assert starter_aliases["spin_polarised"] == "spin_polarized"


def test_starter_helper_functions_cover_requirements_defaults_and_orders() -> None:
    alias_map = {
        "spin_polarised": "spin_polarized",
        "calculate_polarisation": "calculate_polarization",
    }

    assert _starter_resolve_name("param", "spin_polarised", alias_map) == "spin_polarized"
    assert _starter_option_type("block", "block") == "Block"
    assert _starter_option_type("flag", "bool") == "Defined"
    assert _starter_option_type("scalar", "positive_float_with_optional_unit") == "Physical"
    assert _starter_kind_value_type({"kind": "flag", "value_type": "flag"}) == ("flag", "bool")
    assert _starter_kind_value_type({"kind": "scalar", "value_type": "text"}) == ("scalar", "text")
    assert _starter_sequence(["a", " b "]) == ("a", "b")
    assert _starter_sequence("TASK=GEOMETRYOPTIMIZATION") == ("TASK=GEOMETRYOPTIMIZATION",)

    requires, requires_any_of, allowed_tasks = _starter_parse_requirements(
        "param",
        ["TASK=GEOMETRYOPTIMIZATION", "spin_polarised", "calculate_polarisation or spin_polarised"],
        alias_map=alias_map,
    )
    assert requires == ("spin_polarized",)
    assert requires_any_of == ("calculate_polarization", "spin_polarized")
    assert allowed_tasks == ("GEOMETRYOPTIMIZATION",)

    assert _starter_default_value("int_triplet", [2, 2, 2]) == (2, 2, 2)
    assert _starter_default_value("bool", True) is True
    assert _starter_default_value("positive_float_with_optional_unit", 0.1) == pytest.approx(0.1)
    assert _starter_default_value("text", "value") == "value"
    assert _starter_source_order("param", "task", 100) == 0
    assert _starter_source_order("param", "write_formatted_density", 100) == 1
    assert _starter_source_order("param", "non_special_name", 5) > 5


def test_build_block_schema_handles_none_and_full_payload() -> None:
    assert _build_block_schema(None) is None

    schema = _build_block_schema(
        {
            "family": "table",
            "allow_unit": True,
            "default_unit": "ang",
            "fixed_row_count": 2,
            "min_rows": 1,
            "row_group_size": 2,
            "row_labels": ("A", "B"),
            "summary_mode": "matrix2",
            "columns": (
                {"name": "Species", "value_kind": "text"},
                {"name": "x", "value_kind": "float", "allowed_values": ("1", "2")},
            ),
        }
    )

    assert isinstance(schema, BlockSchema)
    assert schema.allow_unit is True
    assert schema.fixed_row_count == 2
    assert schema.columns[1].allowed_values == ("1", "2")
