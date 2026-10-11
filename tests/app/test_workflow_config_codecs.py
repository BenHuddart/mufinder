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
from mufinder.app.main_window import MainWindow
from mufinder.app.relaxation_types import (
    MuonRelaxationConfig,
    MuonRelaxationRow,
    PristineRelaxationConfig,
    PristineRelaxationRow,
)
from mufinder.app.uep_types import UepConfig, UepSourceRow
from mufinder.castep_io.types import ValueWithUnit
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
    payload = MainWindow._serialize_uep_config(None, _uep_config())

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
    assert MainWindow._deserialize_uep_config(None, payload) == expected
    assert MainWindow._deserialize_uep_config(None, _through_json(payload)) == expected


def test_pristine_relaxation_config_stored_shape_and_round_trip() -> None:
    payload = MainWindow._serialize_pristine_relaxation_config(None, _pristine_config())

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
    assert MainWindow._deserialize_pristine_relaxation_config(None, payload) == _pristine_config()
    assert MainWindow._deserialize_pristine_relaxation_config(None, _through_json(payload)) == _pristine_config()


def test_muon_relaxation_config_stored_shape_and_round_trip() -> None:
    payload = MainWindow._serialize_muon_relaxation_config(None, _muon_config())

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
    assert MainWindow._deserialize_muon_relaxation_config(None, payload) == _muon_config()
    assert MainWindow._deserialize_muon_relaxation_config(None, _through_json(payload)) == _muon_config()


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
    payload = MainWindow._serialize_convergence_config(config)

    assert payload == expected_payload
    assert MainWindow._deserialize_convergence_config(None, payload) == config
    assert MainWindow._deserialize_convergence_config(None, _through_json(payload)) == config
