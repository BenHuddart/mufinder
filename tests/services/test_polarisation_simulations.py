"""Tests for polarisation simulations of stored magnetic-field results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mufinder.science.magnetic_fields import MagneticFieldDistributionResult
from mufinder.science.nuclear_relaxation import calculate_nuclear_width_tensors
from mufinder.science.polarisation import PolarisationError, calculate_polarisation
from mufinder.science.polarisation_simulation import PolarisationSettings, simulate_polarisation
from mufinder.services.magnetic_field_results import (
    MAGNETIC_FIELD_PAGE_OWNER,
    UEP_SITE_SOURCE,
    polarisation_result,
    remove_magnetic_field_results,
    store_magnetic_field_results,
)
from mufinder.services.polarisation_simulations import (
    SimulationSource,
    field_results_for_source,
    field_sources,
    polarisation_simulation,
    polarisation_simulations,
    remove_polarisation_simulation,
    run_polarisation_simulation,
    simulation_owner,
    stale_sources,
    store_polarisation_simulation,
)
from mufinder.services.project_service import ProjectService
from mufinder.services.project_summary import MAGNETIC_FIELD_WORKSPACES_KEY
from tests.support.controllers import muon_host_document
from tests.support.field_results import field_result

ZERO = (0.0, 0.0, 0.0)
SETTINGS = PolarisationSettings(end_time_us=4.0, time_points=401)


def _field(total, *, label: str = "site 1", applied=ZERO, position=(0.25, 0.25, 0.25)):
    return field_result(total, applied, label, position)


def _commensurate(*results, site_source: str | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "kind": "commensurate",
        "results": [result.to_json() for result in results],
    }
    if site_source:
        value["site_source"] = site_source
    return value


def _store_fields(document, owner: str, value: dict[str, object]) -> None:
    store_magnetic_field_results(document, owner, {"magnetic_field_results": value})


def _add_workspace(document, workspace_id: str, title: str) -> None:
    document.ui_state.setdefault(MAGNETIC_FIELD_WORKSPACES_KEY, []).append(
        {"id": workspace_id, "title": title}
    )


@pytest.fixture
def document(tmp_path: Path):
    document, _host = muon_host_document(ProjectService(), tmp_path)
    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, _commensurate(_field((0.0, 0.0, 0.01))))
    _add_workspace(document, "ws-1", "Site B")
    _store_fields(
        document,
        "ws-1",
        _commensurate(
            _field((0.02, 0.0, 0.0), label="B1"),
            _field((0.0, 0.02, 0.0), label="B2"),
            site_source=UEP_SITE_SOURCE,
        ),
    )
    return document


def test_field_sources_list_the_page_then_each_workspace(document) -> None:
    _add_workspace(document, "ws-empty", "No fields yet")

    sources = field_sources(document)

    assert [source.owner for source in sources] == [MAGNETIC_FIELD_PAGE_OWNER, "ws-1"]
    page, workspace = sources
    assert page.label == "Magnetic Fields · 1 field at site 1"
    assert not page.estimate
    assert page.applied_T == ZERO
    assert workspace.site_labels == ("B1", "B2")
    assert workspace.estimate
    assert workspace.label == "Site B · 2 fields at B1 (unrelaxed estimate)"


def test_a_distribution_source_decodes_to_distribution_results(document) -> None:
    distribution = MagneticFieldDistributionResult(
        site_label="helix",
        muon_position=(0.5, 0.5, 0.5),
        fields_T=np.array([[0.0, 0.0, 0.01], [0.0, 0.01, 0.0]]),
        histogram_centres_T=np.array([0.01]),
        histogram_counts=np.array([2.0]),
        applied_T=ZERO,
    )
    _store_fields(
        document,
        MAGNETIC_FIELD_PAGE_OWNER,
        {"kind": "distribution", "results": [distribution.to_json()]},
    )

    (result,) = field_results_for_source(document, MAGNETIC_FIELD_PAGE_OWNER)

    assert isinstance(result, MagneticFieldDistributionResult)
    assert field_sources(document)[0].label == "Magnetic Fields · 1 distribution at helix"


def test_a_malformed_field_record_is_not_offered(document) -> None:
    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, {"kind": "commensurate", "results": [{}]})

    assert [source.owner for source in field_sources(document)] == ["ws-1"]
    with pytest.raises(PolarisationError, match="malformed"):
        field_results_for_source(document, MAGNETIC_FIELD_PAGE_OWNER)


def test_a_simulation_weights_each_source_and_shares_it_among_its_equivalents(document) -> None:
    record = run_polarisation_simulation(
        document,
        title="  ZF  ",
        settings=SETTINGS,
        sources=[
            SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "", weight=1.0),
            SimulationSource("ws-1", "", "", weight=3.0),
        ],
    )

    fields = [
        _field((0.0, 0.0, 0.01)),
        _field((0.02, 0.0, 0.0), label="B1"),
        _field((0.0, 0.02, 0.0), label="B2"),
    ]
    expected = calculate_polarisation(fields, SETTINGS.times_us, site_weights=(1.0, 1.5, 1.5))
    np.testing.assert_allclose(record.simulation.static.polarisation, expected.polarisation)
    assert record.title == "ZF"
    assert record.estimate
    assert [source.title for source in record.sources] == ["Magnetic Fields", "Site B"]
    assert stale_sources(document, record) == []


def test_nuclear_widths_are_summed_at_each_field_position(tmp_path: Path) -> None:
    document, host = muon_host_document(ProjectService(), tmp_path)
    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, _commensurate(_field((0.05, 0.0, 0.0))))

    record = run_polarisation_simulation(
        document,
        title="widths",
        settings=PolarisationSettings(
            sigma_from_nuclear_widths=True, end_time_us=4.0, time_points=401
        ),
        sources=[SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "")],
    )

    expected = simulate_polarisation(
        record.settings,
        [_field((0.05, 0.0, 0.0))],
        lattice_vectors=host.lattice_vectors,
        nuclear_width_tensors=[calculate_nuclear_width_tensors(host, (0.25, 0.25, 0.25))],
    )
    np.testing.assert_allclose(record.simulation.static.polarisation, expected.static.polarisation)
    assert record.simulation.static.parameters["sigma_nuclear"] == "auto"


@pytest.mark.parametrize(
    ("sources", "message"),
    [
        ([], "at least one"),
        ([SimulationSource("ws-missing", "", "Gone")], "Gone has no field results"),
        ([SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "", weight=0.0)], "positive weight"),
        ([SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "", weight=-1.0)], ">= 0"),
    ],
)
def test_a_simulation_refuses_unusable_sources(document, sources, message: str) -> None:
    with pytest.raises(PolarisationError, match=message):
        run_polarisation_simulation(document, title="x", settings=SETTINGS, sources=sources)


def test_a_simulation_needs_a_host(tmp_path: Path) -> None:
    document = ProjectService().create_project(tmp_path / "p.mufinder", "P")

    with pytest.raises(PolarisationError, match="host"):
        run_polarisation_simulation(
            document,
            title="x",
            settings=SETTINGS,
            sources=[SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "")],
        )


def test_simulations_are_stored_listed_replaced_and_removed(document) -> None:
    source = [SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "")]
    first = run_polarisation_simulation(document, title="A", settings=SETTINGS, sources=source)
    second = run_polarisation_simulation(document, title="B", settings=SETTINGS, sources=source)
    stored = store_polarisation_simulation(document, first)
    store_polarisation_simulation(document, second)

    assert store_polarisation_simulation(document, first) is stored
    assert [record.title for record in polarisation_simulations(document)] == ["A", "B"]
    # The static trace stays readable through the record-owner helper.
    assert polarisation_result(document, simulation_owner(first.simulation_id)) is not None

    again = run_polarisation_simulation(
        document,
        title="A, damped",
        settings=PolarisationSettings(end_time_us=4.0, time_points=401, lambda_z_per_us=0.5),
        sources=source,
        simulation_id=first.simulation_id,
    )
    replaced = store_polarisation_simulation(document, again)
    assert replaced.id != stored.id
    assert replaced.created_at == stored.created_at
    assert [record.title for record in polarisation_simulations(document)] == ["A, damped", "B"]

    remove_polarisation_simulation(document, first.simulation_id)
    assert polarisation_simulation(document, first.simulation_id) is None
    assert [record.title for record in polarisation_simulations(document)] == ["B"]
    remove_polarisation_simulation(document, first.simulation_id)


def test_a_simulation_survives_a_save_and_reload(document) -> None:
    service = ProjectService()
    record = run_polarisation_simulation(
        document,
        title="dynamic",
        settings=PolarisationSettings(
            end_time_us=4.0,
            time_points=401,
            collision_rate_per_us=1.0,
            padding_factor=2,
            polycrystalline=True,
        ),
        sources=[SimulationSource("ws-1", "", "")],
    )
    store_polarisation_simulation(document, record)
    service.save_project(document)

    (restored,) = polarisation_simulations(service.load_project(document.project_root))

    assert restored.simulation_id == record.simulation_id
    assert restored.settings == record.settings
    assert restored.sources == record.sources
    assert restored.site_source == UEP_SITE_SOURCE
    assert restored.simulation.dynamic is not None
    np.testing.assert_allclose(
        restored.simulation.spectrum.amplitudes, record.simulation.spectrum.amplitudes
    )


def test_recalculated_or_removed_fields_make_a_simulation_stale(document) -> None:
    record = run_polarisation_simulation(
        document,
        title="both",
        settings=SETTINGS,
        sources=[
            SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", ""),
            SimulationSource("ws-1", "", ""),
        ],
    )
    store_polarisation_simulation(document, record)

    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, _commensurate(_field((0.0, 0.0, 0.03))))
    remove_magnetic_field_results(document, "ws-1")

    stale = stale_sources(document, record)
    assert [source.owner for source in stale] == [MAGNETIC_FIELD_PAGE_OWNER, "ws-1"]
    # The simulation itself is kept.
    assert polarisation_simulation(document, record.simulation_id) is not None


def test_a_malformed_simulation_record_is_skipped(document) -> None:
    record = run_polarisation_simulation(
        document,
        title="x",
        settings=SETTINGS,
        sources=[SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "")],
    )
    stored = store_polarisation_simulation(document, record)
    stored.data["settings"] = {"time_points": "many"}

    assert polarisation_simulations(document) == []
    assert polarisation_simulation(document, record.simulation_id) is None


def test_a_source_listed_twice_is_refused(document) -> None:
    with pytest.raises(PolarisationError, match="more than once"):
        run_polarisation_simulation(
            document,
            title="x",
            settings=SETTINGS,
            sources=[
                SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", ""),
                SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", ""),
            ],
        )


def test_nuclear_widths_are_summed_once_per_muon_position(tmp_path: Path, monkeypatch) -> None:
    from mufinder.services import polarisation_simulations as module

    document, _host = muon_host_document(ProjectService(), tmp_path)
    translations = [_field((0.0, 0.0, 0.01 * (n + 1)), label=f"t{n}") for n in range(5)]
    _store_fields(document, MAGNETIC_FIELD_PAGE_OWNER, _commensurate(*translations))
    calls: list[tuple[float, ...]] = []
    real = module.calculate_nuclear_width_tensors

    def counting(host, position, **kwargs):
        calls.append(tuple(position))
        return real(host, position, **kwargs)

    monkeypatch.setattr(module, "calculate_nuclear_width_tensors", counting)

    run_polarisation_simulation(
        document,
        title="widths",
        settings=PolarisationSettings(
            sigma_from_nuclear_widths=True, end_time_us=4.0, time_points=401
        ),
        sources=[SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "")],
    )

    assert calls == [(0.25, 0.25, 0.25)]


def test_a_simulation_decodes_only_the_sources_it_uses(document, monkeypatch) -> None:
    from mufinder.services import polarisation_simulations as module

    decoded: list[str] = []
    real = module.MagneticFieldResult.from_json

    def counting(payload):
        decoded.append(str(payload.get("site_label")))
        return real(payload)

    monkeypatch.setitem(module._FIELD_KINDS, "commensurate", counting)

    run_polarisation_simulation(
        document,
        title="page only",
        settings=SETTINGS,
        sources=[SimulationSource(MAGNETIC_FIELD_PAGE_OWNER, "", "")],
    )

    assert decoded == ["site 1"]


@pytest.mark.parametrize("samples", [None, 3, "fields", []])
def test_a_distribution_with_malformed_samples_is_not_offered(document, samples) -> None:
    _store_fields(
        document,
        MAGNETIC_FIELD_PAGE_OWNER,
        {"kind": "distribution", "results": [{"site_label": "x", "fields_T": samples}]},
    )

    assert [source.owner for source in field_sources(document)] == ["ws-1"]
