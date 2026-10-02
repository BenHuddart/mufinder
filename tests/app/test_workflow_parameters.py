from __future__ import annotations

import mufinder.app.workflow_parameters as workflow_parameters_module
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import (
    COMMON_WORKFLOW_PARAMETER_NAMES,
    build_workflow_parameter_surface,
    kpoint_exclusive_hint,
    reorder_workflow_parameter_specs,
    seed_xc_functional_default,
    workflow_basic_parameter_names,
)
from mufinder.castep_io.schema import KeywordSpec


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
