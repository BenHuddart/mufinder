"""Characterisation tests for the workflow configuration codecs.

Each codec stores a workflow's configuration in
``CalculationDefinition.parameters["configuration"]``. The expected payloads
below pin the stored shape exactly, and each config survives a trip through
the JSON persistence codec unchanged.
"""

from __future__ import annotations

import json
from uuid import UUID

import pytest

from mufinder.app.convergence_types import ConvergenceConfig
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
)
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.app.workflow_config_codecs import (
    batch_row_payload,
    deserialize_convergence_config,
    deserialize_muon_relaxation_config,
    deserialize_pristine_relaxation_config,
    deserialize_uep_config,
    normalized_supercell_matrix,
    num_farms_from_extras,
    row_settings_payload,
    serialize_convergence_config,
    serialize_muon_relaxation_config,
    serialize_pristine_relaxation_config,
    serialize_uep_config,
    store_workflow_row_configuration,
    workflow_row_configuration,
    workflow_row_configuration_exists,
)
from mufinder.castep_io.types import ValueWithUnit
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import CalculationDefinition
from mufinder.persistence.codec import to_jsonable

_MATRIX = ((2, 0, 0), (0, 2, 0), (0, 0, 1))
_MATRIX_PAYLOAD = [[2, 0, 0], [0, 2, 0], [0, 0, 1]]
_SITE_ID = UUID("6f1c2d3e-4a5b-4c6d-8e7f-0123456789ab")
_BATCH_ID = UUID("0a1b2c3d-4e5f-4a6b-9c7d-8e9fa0b1c2d3")


def _params() -> dict[str, object]:
    return {
        "cut_off_energy": ValueWithUnit(value=500.0, unit="eV"),
        "kpoint_mp_grid": (4, 4, 4),
        "xc_functional": "PBE",
    }


def _through_json(payload: dict[str, object]) -> object:
    return json.loads(json.dumps(to_jsonable(payload)))


def _uep_config() -> UepConfig:
    return UepConfig(
        name_prefix="uep_source",
        rows=(
            UepSourceRow(
                label="uep_a",
                supercell_matrix=_MATRIX,
                parameter_values=_params(),
                added_advanced=("spin_polarized",),
            ),
            UepSourceRow(label="uep_b"),
        ),
    )


def _pristine_config() -> PristineRelaxationConfig:
    return PristineRelaxationConfig(
        name_prefix="relax",
        rows=(
            PristineRelaxationRow(
                label="cell_a",
                fix_all_cell=True,
                supercell_matrix=_MATRIX,
                parameter_values=_params(),
                added_advanced=("geom_method",),
            ),
            PristineRelaxationRow(label="cell_b"),
        ),
    )


def _muon_config() -> MuonRelaxationConfig:
    return MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=_SITE_ID,
                label="site_1",
                initial_fractional_position=(0.25, 0.5, 0.75),
                charge_state="muonium",
                supercell_matrix=_MATRIX,
                parameter_values=_params(),
                added_advanced=("spin",),
                imported=True,
                import_batch_id=_BATCH_ID,
            ),
            MuonRelaxationRow(
                site_id=_SITE_ID,
                label="site_2",
                initial_fractional_position=(0.0, 0.0, 0.5),
            ),
        )
    )


def _convergence_config() -> ConvergenceConfig:
    return ConvergenceConfig(
        sweep_parameter="kpoint_mp_grid",
        sweep_values=((2, 2, 2), (4, 4, 4)),
        shared_params=_params(),
        overrides={frozenset({(4, 4, 4)}): {"xc_functional": "LDA"}},
        name_prefix="kp_sweep",
    )


def _cutoff_config() -> ConvergenceConfig:
    return ConvergenceConfig(
        sweep_values=(300.0, 400.0),
        shared_params={"xc_functional": "PBE"},
        overrides={frozenset({400.0}): {"xc_functional": "LDA"}},
        name_prefix="cutoff_convergence",
    )


def test_uep_config_stored_shape_and_round_trip() -> None:
    payload = serialize_uep_config(_uep_config())

    assert payload == {
        "name_prefix": "uep",
        "rows": [
            {
                "label": "uep_a",
                "supercell_matrix": _MATRIX_PAYLOAD,
                "parameter_values": _params(),
                "added_advanced": ["spin_polarized"],
            },
            {"label": "uep_b", "supercell_matrix": None, "parameter_values": {}, "added_advanced": []},
        ],
    }
    expected = UepConfig(name_prefix="uep", rows=_uep_config().rows)
    assert deserialize_uep_config(payload) == expected
    assert deserialize_uep_config(_through_json(payload)) == expected


def test_pristine_relaxation_config_stored_shape_and_round_trip() -> None:
    payload = serialize_pristine_relaxation_config(_pristine_config())

    assert payload == {
        "name_prefix": "relax",
        "rows": [
            {
                "label": "cell_a",
                "fix_all_cell": True,
                "supercell_matrix": _MATRIX_PAYLOAD,
                "parameter_values": _params(),
                "added_advanced": ["geom_method"],
            },
            {
                "label": "cell_b",
                "fix_all_cell": False,
                "supercell_matrix": None,
                "parameter_values": {},
                "added_advanced": [],
            },
        ],
    }
    assert deserialize_pristine_relaxation_config(payload) == _pristine_config()
    assert deserialize_pristine_relaxation_config(_through_json(payload)) == _pristine_config()


def test_muon_relaxation_config_stored_shape_and_round_trip() -> None:
    payload = serialize_muon_relaxation_config(_muon_config())

    assert payload == {
        "rows": [
            {
                "site_id": str(_SITE_ID),
                "label": "site_1",
                "initial_fractional_position": [0.25, 0.5, 0.75],
                "charge_state": "muonium",
                "supercell_matrix": _MATRIX_PAYLOAD,
                "parameter_values": _params(),
                "added_advanced": ["spin"],
                "imported": True,
                "import_batch_id": str(_BATCH_ID),
            },
            {
                "site_id": str(_SITE_ID),
                "label": "site_2",
                "initial_fractional_position": [0.0, 0.0, 0.5],
                "charge_state": "positive_muon",
                "supercell_matrix": None,
                "parameter_values": {},
                "added_advanced": [],
                "imported": False,
                "import_batch_id": None,
            },
        ]
    }
    assert deserialize_muon_relaxation_config(payload) == _muon_config()
    assert deserialize_muon_relaxation_config(_through_json(payload)) == _muon_config()


@pytest.mark.parametrize(
    ("config", "expected_payload"),
    [
        (
            _convergence_config(),
            {
                "name_prefix": "kp_sweep",
                "sweep_parameter": "kpoint_mp_grid",
                "sweep_values": [[2, 2, 2], [4, 4, 4]],
                "cutoff_energies": [],
                "shared_params": _params(),
                "overrides": [{"cutoffs": [[4, 4, 4]], "params": {"xc_functional": "LDA"}}],
            },
        ),
        (
            _cutoff_config(),
            {
                "name_prefix": "cutoff_convergence",
                "sweep_parameter": "cut_off_energy",
                "sweep_values": [300.0, 400.0],
                "cutoff_energies": [300.0, 400.0],
                "shared_params": {"xc_functional": "PBE"},
                "overrides": [{"cutoffs": [400.0], "params": {"xc_functional": "LDA"}}],
            },
        ),
    ],
    ids=["kpoint", "cutoff"],
)
def test_convergence_config_stored_shape_and_round_trip(
    config: ConvergenceConfig,
    expected_payload: dict[str, object],
) -> None:
    payload = serialize_convergence_config(config)

    assert payload == expected_payload
    assert deserialize_convergence_config(payload) == config
    assert deserialize_convergence_config(_through_json(payload)) == config


def test_deserialize_skips_unreadable_rows_and_matrices() -> None:
    payload = {
        "name_prefix": "  ",
        "rows": [
            "not a row",
            {"label": "   "},
            {"label": "tuple_matrix", "supercell_matrix": ((2, 0, 0), (0, 2, 0), (0, 0, 2))},
            {"label": "bad_matrix", "supercell_matrix": [[1, 0], [0, 1], [0, 0]]},
            {"label": "legacy", "params": {"xc_functional": "PBE"}, "added_advanced": "spin"},
        ],
    }

    config = deserialize_pristine_relaxation_config(payload)

    assert config == PristineRelaxationConfig(
        name_prefix="pristine_relax",
        rows=(
            PristineRelaxationRow(label="tuple_matrix"),
            PristineRelaxationRow(label="bad_matrix"),
            PristineRelaxationRow(label="legacy", parameter_values={"xc_functional": "PBE"}),
        ),
    )
    assert deserialize_uep_config({"rows": "nope"}) is None
    assert deserialize_muon_relaxation_config([]) is None


def test_muon_deserialize_skips_rows_without_site_or_position() -> None:
    payload = {
        "rows": [
            {"site_id": "not-a-uuid", "label": "a", "initial_fractional_position": [0, 0, 0]},
            {"site_id": str(_SITE_ID), "label": "b", "initial_fractional_position": [0, 0]},
            {
                "site_id": str(_SITE_ID),
                "label": "c",
                "fractional_position": [0.1, 0.2, 0.3],
                "params": {"xc_functional": "PBE"},
                "import_batch_id": "not-a-uuid",
            },
        ]
    }

    config = deserialize_muon_relaxation_config(payload)

    # ``params`` is not read for muon rows, and an unreadable batch id is dropped.
    assert config == MuonRelaxationConfig(
        rows=(MuonRelaxationRow(site_id=_SITE_ID, label="c", initial_fractional_position=(0.1, 0.2, 0.3)),)
    )


def test_convergence_deserialize_rejects_unreadable_sweeps() -> None:
    assert deserialize_convergence_config({"sweep_parameter": "smearing_width"}) is None
    assert deserialize_convergence_config({"sweep_parameter": "kpoint_mp_grid", "sweep_values": [[1, 2]]}) is None
    assert deserialize_convergence_config({"cutoff_energies": [300.0], "shared_params": []}) is None

    config = deserialize_convergence_config(
        {
            "cutoff_energies": [300.0],
            "sweep_values": [999.0],
            "overrides": [{"cutoffs": ["x"], "params": {}}, {"sweep_values": [300.0], "params": {"task": "X"}}],
        }
    )

    assert config == ConvergenceConfig(
        sweep_values=(300.0,),
        overrides={frozenset({300.0}): {"task": "X"}},
        name_prefix="cutoff_convergence",
    )


def test_normalized_supercell_matrix() -> None:
    assert normalized_supercell_matrix(None) is None
    assert normalized_supercell_matrix(((2.0, 0, 0), (0, 2, 0), (0, 0, 1))) == _MATRIX  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        normalized_supercell_matrix(((1, 0), (0, 1), (0, 0)))  # type: ignore[arg-type]


def test_uep_config_reads_the_legacy_prefix_as_uep() -> None:
    config = deserialize_uep_config({"name_prefix": "uep_source", "rows": []})

    assert config is not None
    assert config.name_prefix == "uep"


def _definition_with_rows(row_settings: object) -> CalculationDefinition:
    return CalculationDefinition(
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type="castep_quantum_phonon_set",
        parameters={"configuration": {"row_settings": row_settings}},
    )


def test_workflow_row_configuration_restores_values_and_splits_extras() -> None:
    definition = _definition_with_rows(
        {
            "site": {
                "parameter_values": {"cut_off_energy": {"value": 500.0, "unit": "eV"}},
                "added_advanced": ["spin_polarized", 3],
                "num_farms": 4,
                "pseudopotential_family": "ultrasoft",
            },
            "legacy": {"params": {"xc_functional": "PBE"}},
        }
    )

    params, advanced, extras = workflow_row_configuration(definition, "site")

    assert params == {"cut_off_energy": ValueWithUnit(500.0, "eV")}
    assert advanced == ("spin_polarized",)
    assert extras == {"num_farms": 4, "pseudopotential_family": "ultrasoft"}
    assert workflow_row_configuration(definition, "legacy")[0] == {"xc_functional": "PBE"}
    assert workflow_row_configuration(definition, "missing") == ({}, (), {})
    assert workflow_row_configuration(None, "site") == ({}, (), {})
    assert workflow_row_configuration(_definition_with_rows([]), "site") == ({}, (), {})


def test_workflow_row_configuration_exists_ignores_blank_extras() -> None:
    definition = _definition_with_rows({"blank": {"pseudopotential_family": " "}, "farmed": {"num_farms": 2}})

    assert workflow_row_configuration_exists(definition, "blank") is False
    assert workflow_row_configuration_exists(definition, "farmed") is True
    assert workflow_row_configuration_exists(definition, "missing") is False


@pytest.mark.parametrize(("raw", "expected"), [(3, 3), ("2", 2), (0, 1), ("lots", 1), (None, 1)])
def test_num_farms_from_extras_is_at_least_one(raw: object, expected: int) -> None:
    assert num_farms_from_extras({"num_farms": raw}) == expected
    assert num_farms_from_extras({}) == 1


def test_row_settings_payload_drops_task_and_sorts_advanced() -> None:
    payload = row_settings_payload(
        {"task": "SINGLEPOINT", "cut_off_energy": ValueWithUnit(500.0, "eV")},
        ("spin_polarized", "basis_precision", "spin_polarized"),
        {"num_farms": 2},
    )

    assert payload == {
        "parameter_values": {"cut_off_energy": to_jsonable(ValueWithUnit(500.0, "eV"))},
        "added_advanced": ["basis_precision", "spin_polarized"],
        "num_farms": 2,
    }
    assert row_settings_payload({"task": "SINGLEPOINT"}, (), {}) is None


def test_batch_row_payload_keeps_a_rows_own_untouched_mixed_values() -> None:
    definition = _definition_with_rows(
        {
            "explicit": {
                "parameter_values": {"max_scf_cycles": 80, "xc_functional": "LDA"},
                "added_advanced": ["nextra_bands"],
            },
            # "derived" has no stored value for max_scf_cycles: it takes
            # it from its source defaults and must keep doing so.
            "derived": {"parameter_values": {"xc_functional": "LDA"}, "added_advanced": []},
        }
    )
    payload = {
        "parameter_values": {"xc_functional": "PBE"},
        "added_advanced": ["nextra_bands"],
        "num_farms": 2,
    }
    untouched = {"max_scf_cycles", "nextra_bands"}

    explicit = batch_row_payload(definition, "explicit", payload, untouched)
    derived = batch_row_payload(definition, "derived", payload, untouched)

    assert explicit == {
        "parameter_values": {"xc_functional": "PBE", "max_scf_cycles": 80},
        "added_advanced": ["nextra_bands"],
        "num_farms": 2,
    }
    assert derived == {
        "parameter_values": {"xc_functional": "PBE"},
        "added_advanced": [],
        "num_farms": 2,
    }
    # With nothing untouched the dialog's payload is stored as it is.
    assert batch_row_payload(definition, "explicit", payload, set()) is payload
    # A row with nothing of its own and an empty dialog stores nothing.
    assert batch_row_payload(definition, "missing", None, {"xc_functional"}) is None


def test_store_workflow_row_configuration_adds_and_removes_rows() -> None:
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.ENERGY_BARRIERS,
        calc_type="castep_nudged_elastic_band_set",
        parameters={"configuration": "unreadable"},
    )

    store_workflow_row_configuration(definition, "path", {"num_farms": 3})
    assert definition.parameters["configuration"] == {"row_settings": {"path": {"num_farms": 3}}}

    store_workflow_row_configuration(definition, "path", None)
    assert definition.parameters["configuration"] == {}


