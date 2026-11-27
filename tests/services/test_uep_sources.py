"""Tests for the structures a UEP workflow can run on and the page's selection."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectService
from mufinder.services.uep_sources import (
    select_uep_source_structure,
    selected_uep_source_structure,
    store_uep_form_state,
    uep_source_structures,
)

from tests.support.controllers import muon_host_document


def _relaxed(host: Structure, label: str, *, parent=None, origin="pristine_relaxation"):
    return Structure(
        label=label,
        lattice_vectors=host.lattice_vectors,
        species=host.species,
        fractional_positions=host.fractional_positions,
        origin=origin,
        parent_structure_id=host.id if parent is None else parent,
    )


def test_source_structures_are_the_host_then_its_relaxations_by_label(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    beta = _relaxed(host, "beta")
    alpha = _relaxed(host, "Alpha")
    for structure in (
        beta,
        alpha,
        _relaxed(host, "other host", parent=uuid4()),
        _relaxed(host, "supercell", origin="supercell"),
    ):
        service.add_structure(document, structure)

    assert uep_source_structures(document) == [host, alpha, beta]


def test_source_structures_need_an_active_host(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    document.project.host_structure_id = None

    assert uep_source_structures(document) == []
    assert selected_uep_source_structure(document) is None


def test_selection_defaults_to_the_host_and_keeps_a_saved_choice(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    relaxed = _relaxed(host, "relaxed")
    service.add_structure(document, relaxed)

    assert selected_uep_source_structure(document) is host

    document.ui_state["uep_form"] = {"name_prefix": "uep"}
    select_uep_source_structure(document, str(relaxed.id))
    assert document.ui_state["uep_form"] == {
        "name_prefix": "uep",
        "selected_source_structure_id": str(relaxed.id),
    }
    assert selected_uep_source_structure(document) is relaxed

    select_uep_source_structure(document, "")
    assert document.ui_state["uep_form"] == {"name_prefix": "uep"}
    assert selected_uep_source_structure(document) is host


def test_selection_falls_back_when_the_saved_id_is_unusable(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)

    for saved in ("not-a-uuid", str(uuid4()), 7):
        document.ui_state["uep_form"] = {"selected_source_structure_id": saved}
        assert selected_uep_source_structure(document) is host
    document.ui_state["uep_form"] = "not a form"
    assert selected_uep_source_structure(document) is host


def test_storing_the_form_keeps_a_choice_the_picker_cannot_show(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    relaxed = _relaxed(host, "relaxed")
    service.add_structure(document, relaxed)
    select_uep_source_structure(document, str(relaxed.id))

    # An empty picker, or one still listing another project's structures.
    for reported in (None, str(uuid4())):
        store_uep_form_state(
            document, {"name_prefix": "next", "selected_source_structure_id": reported}
        )
        assert document.ui_state["uep_form"] == {
            "name_prefix": "next",
            "selected_source_structure_id": str(relaxed.id),
        }
        assert selected_uep_source_structure(document) is relaxed

    # A structure the picker offers is the page's choice.
    state = {"name_prefix": "next", "selected_source_structure_id": str(host.id)}
    store_uep_form_state(document, state)
    assert document.ui_state["uep_form"] == state
    assert selected_uep_source_structure(document) is host


def test_storing_the_form_never_stores_a_structure_the_page_does_not_offer(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    other_hosts_relaxation = _relaxed(host, "other host's", parent=uuid4())
    service.add_structure(document, other_hosts_relaxation)

    # Nothing stored, or a stored choice that is no longer offered: no choice is stored.
    for previous in (
        {},
        "not a form",
        {"selected_source_structure_id": str(other_hosts_relaxation.id)},
    ):
        for reported in (None, str(uuid4()), str(other_hosts_relaxation.id)):
            document.ui_state["uep_form"] = previous
            store_uep_form_state(
                document, {"name_prefix": "uep", "selected_source_structure_id": reported}
            )
            assert document.ui_state["uep_form"] == {
                "name_prefix": "uep",
                "selected_source_structure_id": None,
            }
            assert selected_uep_source_structure(document) is host

    # A form without the key gains none.
    document.ui_state["uep_form"] = {}
    store_uep_form_state(document, {"name_prefix": "uep"})
    assert document.ui_state["uep_form"] == {"name_prefix": "uep"}


def test_a_stored_structure_of_another_host_is_not_the_selection(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    relaxed = _relaxed(host, "relaxed")
    service.add_structure(document, relaxed)
    select_uep_source_structure(document, str(relaxed.id))
    assert selected_uep_source_structure(document) is relaxed

    new_host = _relaxed(host, "new host", origin="imported")
    service.add_structure(document, new_host)
    document.project.host_structure_id = new_host.id

    assert selected_uep_source_structure(document) is new_host
