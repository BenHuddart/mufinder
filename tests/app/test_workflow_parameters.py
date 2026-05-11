from __future__ import annotations

import mufinder.app.workflow_parameters as workflow_parameters_module
from mufinder.app.param_dialog import ParameterDialog
from mufinder.app.workflow_parameters import build_workflow_parameter_surface
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