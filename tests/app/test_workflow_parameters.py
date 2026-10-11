from __future__ import annotations

import mufinder.app.workflow_parameters as workflow_parameters_module
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import (
    COMMON_WORKFLOW_PARAMETER_NAMES,
    STRUCTURE_CONTROLLED_CELL_KEYWORDS,
    build_workflow_parameter_surface,
    GEOMETRY_PARAMETER_NAMES,
    clone_spec_for_workflow,
    drop_default_conflicts,
    drop_explicit_conflicts,
    geometry_relaxation_parameter_specs,
    kpoint_exclusive_hint,
    merge_species_options,
    reorder_workflow_parameter_specs,
    restore_structured_value,
    sanitize_parameter_mapping,
    seed_xc_functional_default,
    shared_parameter_values,
    spec_is_blocked_by_locked,
    structure_species_options,
    workflow_basic_parameter_names,
    workflow_default_params,
)
from mufinder.castep_io.gui_metadata import infer_task
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit


def _spec(
    name: str,
    *,
    file: str = "param",
    value_type: str = "enum",
    allowed_values: tuple[str, ...] = ("DEFAULT", "ALT"),
    basic_workflows: tuple[str, ...] = (),
    advanced_workflows: tuple[str, ...] = (),
    allowed_tasks: tuple[str, ...] = (),
) -> KeywordSpec:
    return KeywordSpec(
        name=name,
        file=file,
        kind="scalar",
        value_type=value_type,
        allowed_values=allowed_values,
        widget_hint="dropdown",
        basic_workflows=basic_workflows,
        advanced_workflows=advanced_workflows,
        allowed_tasks=allowed_tasks,
        default_value=allowed_values[0] if allowed_values else None,
    )


def test_build_workflow_parameter_surface_promotes_basic_names(monkeypatch) -> None:
    phonon_method = _spec(
        "phonon_method",
        allowed_values=("FINITEDISPLACEMENT", "DFPT"),
        allowed_tasks=("PHONON",),
    )
    xc_functional = _spec(
        "xc_functional",
        allowed_values=("PBE", "LDA"),
        advanced_workflows=("quantum_corrections",),
    )

    monkeypatch.setattr(
        workflow_parameters_module,
        "get_gui_keyword_specs_for_files",
        lambda **_kwargs: [xc_functional],
    )
    monkeypatch.setattr(
        workflow_parameters_module,
        "get_keyword_spec",
        lambda _file, name: phonon_method if name == "phonon_method" else None,
    )

    specs = build_workflow_parameter_surface(
        workflow_stage="quantum_corrections",
        basic_names=("phonon_method",),
    )

    assert [spec.name for spec in specs] == ["phonon_method", "xc_functional"]
    assert "quantum_corrections" in specs[0].basic_workflows
    assert "quantum_corrections" in specs[0].advanced_workflows
    assert "quantum_corrections" in specs[1].advanced_workflows


def test_kpoint_exclusive_hint_returns_spacing_when_all_rows_have_spacing() -> None:
    params = [{"kpoint_mp_spacing": 0.07}, {"kpoint_mp_spacing": 0.05}]
    assert kpoint_exclusive_hint(params) == "kpoint_mp_spacing"


def test_kpoint_exclusive_hint_returns_none_for_empty_sequence() -> None:
    assert kpoint_exclusive_hint([]) is None


def test_kpoint_exclusive_hint_returns_none_when_any_row_lacks_spacing() -> None:
    params = [{"kpoint_mp_spacing": 0.07}, {"kpoint_mp_grid": (4, 4, 4)}]
    assert kpoint_exclusive_hint(params) is None


def test_kpoint_exclusive_hint_returns_spacing_even_when_grid_also_present() -> None:
    # When project defaults inject kpoint_mp_grid alongside user-saved spacing
    # (as in the energy-barrier path), spacing presence is still the signal.
    params = [{"kpoint_mp_spacing": 0.07, "kpoint_mp_grid": (4, 4, 4)}]
    assert kpoint_exclusive_hint(params) == "kpoint_mp_spacing"


def test_workflow_basic_parameter_names_prefixes_common_parameters() -> None:
    names = workflow_basic_parameter_names("tssearch_lstqst_protocol", "tssearch_qst_max_iter")

    assert names[: len(COMMON_WORKFLOW_PARAMETER_NAMES)] == COMMON_WORKFLOW_PARAMETER_NAMES
    assert names[-2:] == ("tssearch_lstqst_protocol", "tssearch_qst_max_iter")


def test_reorder_workflow_parameter_specs_moves_common_parameters_to_shared_positions() -> None:
    specs = [
        _spec("tssearch_qst_max_iter", value_type="int"),
        _spec("kpoint_mp_grid", value_type="int_triplet"),
        _spec("cut_off_energy", value_type="number"),
        _spec("basis_precision"),
        _spec("elec_energy_tol", value_type="number"),
    ]

    reordered = reorder_workflow_parameter_specs(specs)

    assert [spec.name for spec in reordered] == [
        "basis_precision",
        "cut_off_energy",
        "elec_energy_tol",
        "kpoint_mp_grid",
        "tssearch_qst_max_iter",
    ]


def test_parameter_dialog_with_workflow_surface_shows_basic_and_allows_advanced(qapp, monkeypatch) -> None:
    task_spec = _spec(
        "task",
        allowed_values=("SINGLEPOINT", "PHONON"),
        basic_workflows=("quantum_corrections",),
    )
    phonon_method = _spec(
        "phonon_method",
        allowed_values=("FINITEDISPLACEMENT", "DFPT"),
        allowed_tasks=("PHONON",),
    )
    xc_functional = _spec(
        "xc_functional",
        allowed_values=("PBE", "LDA"),
        advanced_workflows=("quantum_corrections",),
    )

    monkeypatch.setattr(
        workflow_parameters_module,
        "get_gui_keyword_specs_for_files",
        lambda **_kwargs: [task_spec, xc_functional],
    )

    def _fake_get_keyword_spec(_file: str, name: str):
        if name == "phonon_method":
            return phonon_method
        return None

    monkeypatch.setattr(workflow_parameters_module, "get_keyword_spec", _fake_get_keyword_spec)

    specs = build_workflow_parameter_surface(
        workflow_stage="quantum_corrections",
        basic_names=("phonon_method",),
    )
    dialog = ParameterDialog(
        specs=specs,
        workflow_stage="quantum_corrections",
        locked_values={"task": "PHONON"},
    )

    assert "phonon_method" in dialog._rows
    assert "xc_functional" not in dialog._rows

    dialog.set_advanced_names({"xc_functional"})

    assert "xc_functional" in dialog._rows


def test_seed_xc_functional_default_prefers_project_then_schema_default() -> None:
    defaults: dict[str, object] = {"xc_functional": "LDA", "task": "SINGLEPOINT"}

    seed_xc_functional_default(defaults, "PBE", schema_default="LDA")
    assert defaults == {"xc_functional": "PBE", "task": "SINGLEPOINT"}

    seed_xc_functional_default(defaults, None, schema_default="LDA")
    assert defaults["xc_functional"] == "LDA"

    seed_xc_functional_default(defaults, None, schema_default=None)
    assert "xc_functional" not in defaults


# ---------------------------------------------------------------------------
# Batch edits of rows with differing values
# ---------------------------------------------------------------------------


def test_mixed_parameter_names_reports_differing_and_partly_absent_values() -> None:
    from mufinder.app.workflow_parameters import mixed_parameter_names

    rows = [
        {"task": "SINGLEPOINT", "xc_functional": "PBE", "max_scf_cycles": 100, "nextra_bands": 4},
        {"task": "GEOMETRYOPTIMIZATION", "xc_functional": "LDA", "max_scf_cycles": 100},
    ]

    # "task" is never reported; an absent value counts as different.
    assert mixed_parameter_names(rows) == {"xc_functional", "nextra_bands"}
    assert mixed_parameter_names(rows[:1]) == set()
    assert mixed_parameter_names([]) == set()
    assert mixed_parameter_names([{"a": None}, {}]) == {"a"}


def test_merge_batch_values_keeps_each_rows_untouched_mixed_values() -> None:
    from mufinder.app.workflow_parameters import merge_batch_values

    collected = {"task": "SINGLEPOINT", "max_scf_cycles": 200}
    untouched = {"xc_functional", "nextra_bands"}

    assert merge_batch_values(
        collected, {"xc_functional": "PBE", "nextra_bands": 4, "max_scf_cycles": 100}, untouched
    ) == {"task": "SINGLEPOINT", "max_scf_cycles": 200, "xc_functional": "PBE", "nextra_bands": 4}
    # A row that had no value for an untouched parameter stays without one.
    assert merge_batch_values(collected, {"xc_functional": "LDA"}, untouched) == {
        "task": "SINGLEPOINT",
        "max_scf_cycles": 200,
        "xc_functional": "LDA",
    }
    # Everything else follows the dialog, including a value the user removed.
    assert merge_batch_values({"task": "SINGLEPOINT"}, {"max_scf_cycles": 100}, set()) == {
        "task": "SINGLEPOINT"
    }


def test_merge_batch_advanced_keeps_untouched_rows_on_the_rows_that_showed_them() -> None:
    from mufinder.app.workflow_parameters import merge_batch_advanced

    assert merge_batch_advanced({"a", "b"}, {"b"}, {"b"}) == {"a", "b"}
    assert merge_batch_advanced({"a", "b"}, set(), {"b"}) == {"a"}
    assert merge_batch_advanced({"a"}, {"c"}, set()) == {"a"}


def _typed_spec(
    name: str,
    value_type: str,
    *,
    file: str = "param",
    unit: str | None = None,
    default_value: object | None = None,
    basic_workflows: tuple[str, ...] = (),
    mutually_exclusive_with: tuple[str, ...] = (),
) -> KeywordSpec:
    return KeywordSpec(
        name=name,
        file=file,
        kind="scalar",
        value_type=value_type,
        unit=unit,
        default_value=default_value,
        basic_workflows=basic_workflows,
        mutually_exclusive_with=mutually_exclusive_with,
    )


_CUTOFF = _typed_spec("cut_off_energy", "physical", unit="eV")
_GRID = _typed_spec("kpoint_mp_grid", "int_triplet", file="cell", mutually_exclusive_with=("kpoint_mp_spacing",))
_SPACING = _typed_spec("kpoint_mp_spacing", "physical", file="cell", unit="1/ang", mutually_exclusive_with=("kpoint_mp_grid",))
_SPECS_BY_NAME = {spec.name: spec for spec in (_CUTOFF, _GRID, _SPACING)}


def test_restore_structured_value_rebuilds_values_flattened_by_json() -> None:
    assert restore_structured_value(_SPECS_BY_NAME, "cut_off_energy", {"value": 500.0, "unit": "eV"}) == (
        ValueWithUnit(500.0, "eV")
    )
    assert restore_structured_value(_SPECS_BY_NAME, "kpoint_mp_grid", [4, 4, 2]) == (4, 4, 2)
    # A bare number picks up the spec's unit.
    assert restore_structured_value(_SPECS_BY_NAME, "cut_off_energy", 400) == ValueWithUnit(400.0, "eV")


def test_restore_structured_value_leaves_unknown_and_rejected_values_alone() -> None:
    assert restore_structured_value(_SPECS_BY_NAME, "not_a_keyword", [1, 2]) == [1, 2]
    # A grid with the wrong length is returned as stored rather than raising.
    assert restore_structured_value(_SPECS_BY_NAME, "kpoint_mp_grid", [4, 4]) == (4, 4)


def test_restore_structured_value_without_specs_reads_the_keyword_catalog(monkeypatch) -> None:
    lookups: list[tuple[str, str]] = []

    def fake_get_keyword_spec(file: str, name: str) -> KeywordSpec | None:
        lookups.append((file, name))
        return _GRID if (file, name) == ("cell", "kpoint_mp_grid") else None

    monkeypatch.setattr(workflow_parameters_module, "get_keyword_spec", fake_get_keyword_spec)

    assert restore_structured_value(None, "kpoint_mp_grid", [3, 3, 3]) == (3, 3, 3)
    assert lookups == [("param", "kpoint_mp_grid"), ("cell", "kpoint_mp_grid")]


def test_sanitize_parameter_mapping_drops_names_without_a_spec() -> None:
    stored = {"cut_off_energy": {"value": 500.0, "unit": "eV"}, "task": "SINGLEPOINT", "retired_keyword": 1}

    assert sanitize_parameter_mapping(stored, _SPECS_BY_NAME) == {"cut_off_energy": ValueWithUnit(500.0, "eV")}
    assert sanitize_parameter_mapping(stored, _SPECS_BY_NAME, keep=("task",)) == {
        "cut_off_energy": ValueWithUnit(500.0, "eV"),
        "task": "SINGLEPOINT",
    }


def test_sanitize_parameter_mapping_keep_unknown_passes_unknown_names_through() -> None:
    # Imported rows keep names the local keyword catalog does not know, unchanged,
    # while known names are still restored.
    stored = {"kpoint_mp_grid": [2, 2, 2], "retired_keyword": [1, 2]}

    assert sanitize_parameter_mapping(stored, _SPECS_BY_NAME, keep_unknown=True) == {
        "kpoint_mp_grid": (2, 2, 2),
        "retired_keyword": [1, 2],
    }


def test_sanitize_parameter_mapping_drop_removes_names_that_have_a_spec() -> None:
    lattice = _typed_spec("lattice_cart", "string", file="cell")
    specs = {**_SPECS_BY_NAME, "lattice_cart": lattice}

    cleaned = sanitize_parameter_mapping(
        {"lattice_cart": "ignored", "kpoint_mp_grid": [1, 1, 1]},
        specs,
        drop=STRUCTURE_CONTROLLED_CELL_KEYWORDS,
    )

    assert cleaned == {"kpoint_mp_grid": (1, 1, 1)}


def test_spec_is_blocked_by_locked_only_for_the_other_half_of_a_pair() -> None:
    assert spec_is_blocked_by_locked(_SPACING, {"kpoint_mp_grid"})
    assert not spec_is_blocked_by_locked(_GRID, {"kpoint_mp_grid"})
    assert not spec_is_blocked_by_locked(_CUTOFF, {"kpoint_mp_grid"})


def test_workflow_default_params_keeps_basic_unlocked_unskipped_defaults() -> None:
    stage = "pristine_relaxation"
    specs = [
        _typed_spec("cut_off_energy", "physical", unit="eV", default_value=300, basic_workflows=(stage,)),
        _typed_spec("max_scf_cycles", "int", default_value=100, basic_workflows=(stage,)),
        _typed_spec("nextra_bands", "int", default_value=4),  # advanced only
        _typed_spec("elec_energy_tol", "float", default_value=1e-5, basic_workflows=(stage,)),
        _typed_spec(
            "kpoint_mp_spacing",
            "float",
            default_value=0.05,
            basic_workflows=(stage,),
            mutually_exclusive_with=("kpoint_mp_grid",),
        ),
        _typed_spec("kpoint_mp_grid", "int_triplet", default_value=(1, 1, 1), basic_workflows=(stage,)),
    ]

    defaults = workflow_default_params(
        specs,
        workflow_stage=stage,
        locked_names={"kpoint_mp_grid"},
        skip_names=("elec_energy_tol",),
    )

    assert defaults == {
        "task": infer_task(stage),
        "cut_off_energy": ValueWithUnit(300.0, "eV"),
        "max_scf_cycles": 100,
    }


def test_drop_default_conflicts_keeps_one_value_per_exclusive_pair() -> None:
    values = {"kpoint_mp_grid": (2, 2, 2), "kpoint_mp_spacing": 0.05, "cut_off_energy": 300}

    assert drop_default_conflicts(values, _SPECS_BY_NAME) is values
    assert values == {"kpoint_mp_grid": (2, 2, 2), "cut_off_energy": 300}

    # A locked name always survives, and anything exclusive with it goes.
    values = {"kpoint_mp_grid": (2, 2, 2), "kpoint_mp_spacing": 0.05}
    assert drop_default_conflicts(values, _SPECS_BY_NAME, locked_names={"kpoint_mp_spacing"}) == {
        "kpoint_mp_spacing": 0.05
    }


def test_clone_spec_for_workflow_adds_the_stage_and_applies_default_overrides() -> None:
    spec = _typed_spec("geom_max_iter", "int", default_value=100, basic_workflows=("uep",))

    cloned = clone_spec_for_workflow(
        spec,
        workflow_stage="muon_relaxation",
        basic=True,
        advanced=False,
        default_overrides={"geom_max_iter": 300},
    )

    assert cloned.basic_workflows == ("uep", "muon_relaxation")
    assert cloned.advanced_workflows == spec.advanced_workflows
    assert cloned.default_value == 300
    # Adding a stage twice is a no-op, and without overrides the default stays.
    again = clone_spec_for_workflow(cloned, workflow_stage="muon_relaxation", basic=True, advanced=False)
    assert again.basic_workflows == ("uep", "muon_relaxation")
    assert again.default_value == 300


def test_drop_explicit_conflicts_lets_a_rows_own_choice_win() -> None:
    # The row chose spacing; the inherited grid default goes, the cutoff stays.
    values = {"kpoint_mp_grid": (2, 2, 2), "kpoint_mp_spacing": 0.05, "cut_off_energy": 300}

    assert drop_explicit_conflicts(values, _SPECS_BY_NAME, explicit_names={"kpoint_mp_spacing"}) is values
    assert values == {"kpoint_mp_spacing": 0.05, "cut_off_energy": 300}

    # Two explicit values are both kept: only defaults give way.
    values = {"kpoint_mp_grid": (2, 2, 2), "kpoint_mp_spacing": 0.05}
    drop_explicit_conflicts(values, _SPECS_BY_NAME, explicit_names={"kpoint_mp_grid", "kpoint_mp_spacing"})
    assert values == {"kpoint_mp_grid": (2, 2, 2), "kpoint_mp_spacing": 0.05}


def test_geometry_relaxation_parameter_specs_shape_the_relaxation_surface() -> None:
    specs = geometry_relaxation_parameter_specs(
        workflow_stage="muon_relaxation",
        excluded_names={"task", "fix_all_cell", "charge"},
        default_overrides={"geom_max_iter": 300},
    )
    by_name = {spec.name: spec for spec in specs}

    assert not {"task", "fix_all_cell", "charge"} & set(by_name)
    # The shared rows lead, as in every workflow dialog.
    common = [name for name in COMMON_WORKFLOW_PARAMETER_NAMES if name in by_name]
    assert common and [spec.name for spec in specs[: len(common)]] == common
    for name in ("kpoint_mp_grid", "kpoint_mp_spacing", *GEOMETRY_PARAMETER_NAMES):
        if name in by_name:
            assert "muon_relaxation" in by_name[name].basic_workflows, name
    assert by_name["geom_max_iter"].default_value == 300
    assert len(by_name) == len(specs)



def test_shared_parameter_values_keeps_what_every_row_agrees_on() -> None:
    rows = [
        {"xc_functional": "PBE", "max_scf_cycles": 80, "task": "SINGLEPOINT"},
        {"xc_functional": "PBE", "max_scf_cycles": 120},
        {"xc_functional": "PBE", "max_scf_cycles": 80},
    ]

    assert shared_parameter_values(rows) == {"xc_functional": "PBE"}
    assert shared_parameter_values(rows[:1]) == rows[0]
    assert shared_parameter_values(rows[:1]) is not rows[0]
    assert shared_parameter_values([]) == {}


def test_species_options_list_each_species_once_in_first_seen_order() -> None:
    from mufinder.domain.models import Structure

    first = Structure(label="a", species=["Fe", " O ", "Fe", ""], fractional_positions=[[0, 0, 0]] * 4)
    second = Structure(label="b", species=["H:mu", "O"], fractional_positions=[[0, 0, 0]] * 2)

    assert structure_species_options(first) == ("Fe", "O")
    assert structure_species_options(None) == ()
    assert merge_species_options([first, None, second]) == ("Fe", "O", "H:mu")
