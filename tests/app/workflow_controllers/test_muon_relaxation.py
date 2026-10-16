"""Tests for :class:`MuonRelaxationController`.

Covers the charge-state-qualified row keys, the pre-submit site check, the
per-charge-state submission groups, ``build_result`` on a real ``-out.cell``,
row parameter edits, continuation, ``validate_artifacts`` and definition
maintenance.
"""

from __future__ import annotations

import copy
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job
from mufinder.services.job_artifacts import ParsedJobArtifacts, collect_job_artifacts
from mufinder.services.project_service import ProjectDocument, ProjectService

from mufinder.app.workflow_controllers.muon_relaxation import (
    MuonRelaxationController,
    MuonRelaxationDeps,
)
from mufinder.app.workflow_config_codecs import (
    deserialize_muon_relaxation_config,
    serialize_muon_relaxation_config,
)

from tests.support.controllers import (
    FakeRowState,
    RecordingBackend,
    job_with_status,
    make_context,
    muon_host_document,
)


# --------------------------------------------------------------------------- #
# MuonRelaxationController
# --------------------------------------------------------------------------- #
#
# The first migration exercising the two base hooks beyond their defaults:
# ``pre_submit_check`` (staged-site/host validation runs BEFORE any submission
# plan is built) and ``submission_groups`` (a mixed job list is split into
# per-charge-state groups, each dispatched separately through the base template
# loop). ``build_result`` carries the richest result logic so far — muon
# position payload (supercell + folded host coordinates), derived structure,
# and the sanctioned source-site mutation — driven below with a REAL
# ``-out.cell`` fixture containing an ``H:mu`` muon.


class _FakeMuonRelaxationView:
    """Minimal MuonRelaxationWidget stand-in for controller tests.

    ``collect_state``/``set_form_state`` store the form state verbatim; see
    :class:`FakeRelaxationView`.
    """

    def __init__(self, form_state: dict | None = None) -> None:
        self.runtime_states: list[dict] = []
        self.form_state: dict = form_state if form_state is not None else {"rows": []}

    def set_runtime_states(self, states) -> None:
        self.runtime_states.append(dict(states))

    def collect_state(self) -> dict:
        return copy.deepcopy(self.form_state)

    def set_form_state(self, state: dict) -> None:
        self.form_state = copy.deepcopy(state)


_MUON_OUT_CELL_TEXT = """
%BLOCK LATTICE_CART
ang
8.0 0.0 0.0
0.0 8.0 0.0
0.0 0.0 4.0
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
H:mu 0.3 0.4 0.5
%ENDBLOCK POSITIONS_FRAC
"""


def _real_coerce_fractional_triplet(value):
    """Coerce exactly as MainWindow._coerce_fractional_triplet does."""

    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None


def _stub_muon_deps(**overrides) -> MuonRelaxationDeps:
    from mufinder.app.relaxation_adapter import muon_charge_state_prefix

    base = dict(
        deserialize_muon_relaxation_config=lambda _payload: None,
        serialize_muon_relaxation_config=serialize_muon_relaxation_config,
        sync_active_muon_relaxation_dashboard_jobs=lambda _definition: None,
        muon_relaxation_form_config=lambda: None,
        muon_relaxation_row_signature=lambda row: (row.label, row.charge_state),
        muon_relaxation_row_key=(
            lambda row: f"{muon_charge_state_prefix(row.charge_state)}:{row.label}"
        ),
        handle_site_host_mismatch=lambda _mismatch: False,
        muon_relaxation_definition_and_row_for_job=lambda _job: (None, None),
        coerce_fractional_triplet=_real_coerce_fractional_triplet,
    )
    base.update(overrides)
    return MuonRelaxationDeps(**base)


def _muon_job(
    service: ProjectService, document: ProjectDocument, label: str, calc_type: str
) -> Job:
    return service.add_job(
        document,
        label=label,
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type=calc_type,
    )


def test_muon_controller_job_for_label_checks_charge_state_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters={"rows": []},
        output_directory=document.project_root / "files" / "muon_relaxation",
    )
    positive = _muon_job(service, document, "Site 1", "castep_positivemuon_relaxation")
    muonium = _muon_job(service, document, "Site 2", "castep_muonium_relaxation")
    pristine = _muon_job(service, document, "Site 3", "castep_pristine_relaxation")
    definition.parameters["job_ids_by_label"] = {
        "positivemuon:Site 1": str(positive.id),
        # A charge-qualified key bound to the wrong charge state's job.
        "muonium:Site 1": str(positive.id),
        # A bare legacy key accepts any muon-relaxation calc type.
        "Site 2": str(muonium.id),
        "Site 3": str(pristine.id),
    }
    controller = MuonRelaxationController(
        make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )

    assert controller.active_definition() is definition
    assert controller.job_for_label(definition, "Site 1", "positive_muon") is positive
    assert controller.job_for_label(definition, "Site 1", "muonium") is None
    assert controller.job_for_label(definition, "Site 2", "positive_muon") is muonium
    assert controller.job_for_label(definition, "Site 2") is muonium
    assert controller.job_for_label(definition, "Site 3") is None


def test_muon_controller_handles_both_charge_state_calc_types(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    controller = MuonRelaxationController(
        make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    positive = _muon_job(service, document, "s1", "castep_positivemuon_relaxation")
    muonium = _muon_job(service, document, "s2", "castep_muonium_relaxation")
    pristine = service.add_job(
        document,
        label="p",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    assert controller.handles_job(positive) is True
    assert controller.handles_job(muonium) is True
    # Pristine relaxation is owned by its own controller, not this one.
    assert controller.handles_job(pristine) is False


def test_muon_controller_ready_predicate_matches_bare_and_qualified_labels() -> None:
    controller = MuonRelaxationController(
        make_context(), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    states = {
        "muonium:Site 1": FakeRowState("ready"),
        "positivemuon:Site 2": FakeRowState("failed"),
    }
    # Charge-qualified keys match their own state only.
    assert controller.is_ready(states, "muonium:Site 1") is True
    assert controller.is_ready(states, "positivemuon:Site 2") is False
    # A bare label matches any charge-qualified key ending in ":{label}".
    assert controller.is_ready(states, "Site 1") is True
    assert controller.is_ready(states, "Site 2") is False
    # "defined" and "not generated" also count as ready (unlike the sweep predicate).
    assert controller.is_ready({"s": FakeRowState("defined")}, "s") is True
    assert controller.is_ready({"s": FakeRowState("not generated")}, "s") is True


def test_muon_controller_refresh_view_without_document_clears_rows() -> None:
    view = _FakeMuonRelaxationView()
    controller = MuonRelaxationController(
        make_context(document=None), _stub_muon_deps(), view
    )
    controller.refresh_view()
    assert view.runtime_states == [{}]


def test_muon_controller_pre_submit_check_blocks_submission(tmp_path: Path) -> None:
    """An unresolved staged-site/host mismatch silently refuses the submission:
    no submission plan is built, no jobs are materialized or dispatched, and no
    'no ready jobs' dialog appears."""

    from mufinder.domain.models import CalculationDefinition, MuonSite, Structure

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    other = Structure(
        label="Other",
        lattice_vectors=[[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(document, other)
    # The staged site was generated for a different host than the active one.
    site = MuonSite(
        label="Site 1",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=other.id,
    )
    document.sites[site.id] = site
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    mismatches: list[object] = []
    form_config_calls: list[bool] = []
    deps = _stub_muon_deps(
        handle_site_host_mismatch=lambda mismatch: (mismatches.append(mismatch), False)[1],
        muon_relaxation_form_config=lambda: (form_config_calls.append(True), None)[1],
    )
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1"], target="local")

    assert len(mismatches) == 1  # the host check ran and surfaced the mismatch
    assert local.calls == []  # nothing dispatched
    assert infos == []  # the refusal is silent (no "no ready jobs" dialog)
    assert form_config_calls == []  # no submission plan was even built
    assert document.jobs == {}  # nothing materialized


def test_muon_controller_pre_submit_check_passes_when_sites_match_host(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Site 1",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    mismatches: list[object] = []
    deps = _stub_muon_deps(
        handle_site_host_mismatch=lambda mismatch: (mismatches.append(mismatch), False)[1],
    )
    local = RecordingBackend()
    infos: list[tuple[str, str]] = []
    context = make_context(
        document=document, project_service=service, local_backend=local, infos=infos
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1"], target="local")

    # No mismatch dialog; the check passed and the plan ran (and found no
    # configured rows, hence the standard no-ready dialog).
    assert mismatches == []
    assert infos == [("No ready jobs", MuonRelaxationController.no_ready_message)]
    assert local.calls == []


def test_muon_controller_submission_groups_split_by_charge_state(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    controller = MuonRelaxationController(
        make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    pos_a = _muon_job(service, document, "a", "castep_positivemuon_relaxation")
    mu_b = _muon_job(service, document, "b", "castep_muonium_relaxation")
    pos_c = _muon_job(service, document, "c", "castep_positivemuon_relaxation")

    # Mixed list splits into positive-muon group first, then muonium, no extra
    # kwargs. ``submission_groups`` no longer dedupes by id — the submission
    # coordinator's plan already does that before this hook runs — so a duplicate
    # job id passed straight to this hook is preserved within its charge-state
    # group rather than collapsed.
    groups = controller.submission_groups([pos_a, mu_b, pos_c, pos_a])
    assert groups == [([pos_a, pos_c, pos_a], {}), ([mu_b], {})]

    # A single-charge-state list yields one group (no empty groups).
    assert controller.submission_groups([mu_b]) == [([mu_b], {})]


def _muon_two_row_submission_fixture(service: ProjectService, tmp_path: Path):
    """Document + config with one positive-muon and one muonium row, wired for
    real materialization through expand_muon_relaxation_jobs."""


    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
    from mufinder.domain.models import MuonSite

    document, host = muon_host_document(service, tmp_path)
    site1 = MuonSite(
        label="Site 1",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    site2 = MuonSite(
        label="Site 2",
        status="staged",
        initial_fractional_position=[0.4, 0.1, 0.2],
        source_structure_id=host.id,
    )
    document.sites[site1.id] = site1
    document.sites[site2.id] = site2

    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=site1.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="positive_muon",
            ),
            MuonRelaxationRow(
                site_id=site2.id,
                label="Site 2",
                initial_fractional_position=(0.4, 0.1, 0.2),
                charge_state="muonium",
            ),
        )
    )
    output_root = document.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters={"configuration": {}},
        output_directory=output_root,
    )

    deps = _stub_muon_deps(
        deserialize_muon_relaxation_config=lambda _payload: config,
        muon_relaxation_form_config=lambda: config,
    )
    return document, definition, deps, output_root


def test_muon_controller_materialize_writes_charge_prefixed_mappings_and_persists(
    tmp_path: Path,
) -> None:
    import dataclasses

    service = ProjectService()
    document, definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    jobs = controller.materialize(["muonium:Site 2"])

    assert [job.calc_type for job in jobs] == ["castep_muonium_relaxation"]
    assert [job.label for job in jobs] == ["Site 2"]
    mapping = definition.parameters["job_ids_by_label"]
    seed_mapping = definition.parameters["job_seed_names_by_label"]
    assert mapping == {"muonium:Site 2": str(jobs[0].id)}
    assert set(seed_mapping) == {"muonium:Site 2"}
    # Rendered inputs were written under the charge-state root.
    seed = seed_mapping["muonium:Site 2"]
    job_dir = output_root / "muonium" / seed
    assert (job_dir / f"{seed}.cell").exists()
    assert (job_dir / f"{seed}.param").exists()
    # The document was persisted with the muon-specific failure message.
    assert persists == [
        {"failure_message": "Muon relaxation jobs were generated, but autosave failed."}
    ]


def test_muon_controller_submit_local_splits_groups_across_backend_calls(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1", "muonium:Site 2"], target="local")

    assert cluster.calls == []
    # One backend call per charge-state group: positive muon first, then muonium.
    assert len(local.calls) == 2
    first_group, first_extras = local.calls[0]
    second_group, second_extras = local.calls[1]
    assert first_extras == {} and second_extras == {}
    assert [job.calc_type for job in first_group] == ["castep_positivemuon_relaxation"]
    assert [job.label for job in first_group] == ["Site 1"]
    assert [job.calc_type for job in second_group] == ["castep_muonium_relaxation"]
    assert [job.label for job in second_group] == ["Site 2"]
    # Both jobs were recorded under charge-prefixed mapping keys.
    mapping = definition.parameters["job_ids_by_label"]
    assert set(mapping) == {"positivemuon:Site 1", "muonium:Site 2"}


def _muon_parameter_edit_fixture(service: ProjectService, tmp_path: Path):
    """One site staged in BOTH charge states under the same label.

    The two rows share a label, so only the charge-qualified key distinguishes
    them; this is the fixture that proves an edit reaches exactly one row.
    """

    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow
    from mufinder.domain.models import MuonSite

    document, host = muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Site 1",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    config = MuonRelaxationConfig(
        rows=(
            MuonRelaxationRow(
                site_id=site.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="positive_muon",
                parameter_values={"geom_max_iter": 120},
            ),
            MuonRelaxationRow(
                site_id=site.id,
                label="Site 1",
                initial_fractional_position=(0.1, 0.2, 0.3),
                charge_state="muonium",
                parameter_values={"geom_max_iter": 400},
            ),
        )
    )
    output_root = document.project_root / "files" / "muon_relaxation"
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters=serialize_muon_relaxation_config(config),
        output_directory=output_root,
    )
    job = _muon_job(service, document, "Site 1", "castep_positivemuon_relaxation")
    definition.parameters["job_ids_by_label"] = {"positivemuon:Site 1": str(job.id)}

    view = _FakeMuonRelaxationView(serialize_muon_relaxation_config(config))

    deps = _stub_muon_deps(
        deserialize_muon_relaxation_config=deserialize_muon_relaxation_config,
        muon_relaxation_form_config=lambda: deserialize_muon_relaxation_config(
            view.collect_state()
        ),
        muon_relaxation_row_signature=lambda row: (
            tuple(sorted(row.parameter_values.items())),
            tuple(row.added_advanced),
        ),
    )
    return document, definition, job, deps, view, output_root


def _muon_edit_controller(service, document, deps, view, output_root, persists):
    import dataclasses

    context = dataclasses.replace(
        make_context(
            document=document,
            project_service=service,
            definition_output_directory=lambda _defn: output_root,
        ),
        persist_ui_state=lambda **kwargs: persists.append(dict(kwargs)),
    )
    return MuonRelaxationController(context, deps, view)


def test_muon_controller_update_row_parameters_touches_only_the_charge_state_row(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    definitions_before = set(document.calculations)
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )

    assert set(document.calculations) == definitions_before
    assert document.calculations[definition.id] is definition
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    assert controller.jobs_for_keys(["positivemuon:Site 1"]) == [job]

    config = deserialize_muon_relaxation_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 500}
    # The muonium row shares the label and is left untouched.
    assert by_charge["muonium"].parameter_values == {"geom_max_iter": 400}

    form_by_charge = {row["charge_state"]: row for row in view.form_state["rows"]}
    assert form_by_charge["positive_muon"]["parameter_values"] == {"geom_max_iter": 500}
    assert form_by_charge["muonium"]["parameter_values"] == {"geom_max_iter": 400}
    assert document.ui_state["muon_relaxation_form"] == view.form_state
    assert persists == [
        {
            "failure_message": (
                "Muon relaxation parameters were updated, but autosave failed."
            )
        }
    ]


def test_muon_controller_update_row_parameters_rejects_bare_label(
    tmp_path: Path,
) -> None:
    """A bare label is ambiguous when two charge states share it, so it is
    refused rather than applied to whichever row matches first."""

    service = ProjectService()
    document, definition, _job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    before = copy.deepcopy(definition.parameters["configuration"])
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.update_row_parameters("Site 1", {"geom_max_iter": 500}) is False

    assert definition.parameters["configuration"] == before
    assert persists == []


def test_muon_controller_update_row_parameters_edits_the_second_charge_state_row(
    tmp_path: Path,
) -> None:
    """The mirror reaches the muonium row (second in both the definition and the
    form) without disturbing the bound positive-muon row."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert (
        controller.update_row_parameters("muonium:Site 1", {"geom_max_iter": 700})
        is True
    )

    config = deserialize_muon_relaxation_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 120}
    assert by_charge["muonium"].parameter_values == {"geom_max_iter": 700}
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    form_by_charge = {row["charge_state"]: row for row in view.form_state["rows"]}
    assert form_by_charge["positive_muon"]["parameter_values"] == {"geom_max_iter": 120}
    assert form_by_charge["muonium"]["parameter_values"] == {"geom_max_iter": 700}


def test_muon_controller_update_row_parameters_without_definition_or_config(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []

    no_definition = _stub_muon_deps()
    controller = _muon_edit_controller(
        service, document, no_definition, view, output_root, persists
    )
    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is False
    )

    unparseable = _stub_muon_deps(
        deserialize_muon_relaxation_config=lambda _payload: None,
    )
    controller = _muon_edit_controller(
        service, document, unparseable, view, output_root, persists
    )
    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is False
    )
    assert persists == []


def test_muon_controller_update_row_parameters_without_the_row_in_the_form(
    tmp_path: Path,
) -> None:
    """With no form row to mirror into, the caller's own values are stored; see
    the pristine equivalent."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    view.form_state = {"rows": []}
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )

    config = deserialize_muon_relaxation_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 500}
    assert by_charge["muonium"].parameter_values == {"geom_max_iter": 400}
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    assert view.form_state["rows"] == []
    assert "muon_relaxation_form" not in document.ui_state


def test_muon_controller_materialize_after_edit_rewrites_same_directory(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    assert controller.materialize(["positivemuon:Site 1"]) == [job]
    seed_mapping_before = dict(definition.parameters["job_seed_names_by_label"])
    seed = seed_mapping_before["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed / f"{seed}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )
    job.status = JobStatus.DEFINED

    assert controller.materialize(["positivemuon:Site 1"]) == [job]

    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    # A changed seed name would relocate the directory and orphan the preserved
    # remote directory.
    assert definition.parameters["job_seed_names_by_label"] == seed_mapping_before
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered


def test_muon_controller_build_states_keeps_job_after_parameter_edit(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, _definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    job.status = JobStatus.COMPLETED
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = controller.build_states()
    assert before["positivemuon:Site 1"].job_id == job.id

    assert (
        controller.update_row_parameters("positivemuon:Site 1", {"geom_max_iter": 500})
        is True
    )

    after = controller.build_states()
    assert after["positivemuon:Site 1"].job_id == job.id
    assert after["positivemuon:Site 1"].status == JobStatus.COMPLETED.value


def test_muon_controller_row_is_continuable_uses_charge_qualified_key(
    tmp_path: Path,
) -> None:
    """Finding 8: gating reads the job's real status, never displayed text —
    a dismissed FAILED job (which *displays* "ready") is still continuable.
    The charge-qualified key (finding 7) is what actually resolves to the
    right row's job here; see ``test_muon_controller_continue_job_rejects_bare_label``
    for the strict charge-qualification check ``continue_job`` performs (a
    bare label happens to still resolve to *a* job through the pre-existing
    ``jobs_for_keys`` label-matching fallback, so ``row_is_continuable`` alone
    is not that safety net)."""

    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, _definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )

    job.status = JobStatus.RUNNING
    assert controller.row_is_continuable("positivemuon:Site 1") is False

    for status in (
        JobStatus.COMPLETED,
        JobStatus.FAILED,
        JobStatus.STOPPED,
        JobStatus.CANCELLED,
    ):
        job.status = status
        assert controller.row_is_continuable("positivemuon:Site 1") is True

    job.status = JobStatus.FAILED
    job.dismissed_at = utc_now()
    assert controller.row_is_continuable("positivemuon:Site 1") is True

    # The muonium row shares the label but has no bound job of its own.
    assert controller.row_is_continuable("muonium:Site 1") is False


def test_muon_controller_render_row_param_text_previews_without_persisting(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, _job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    text = controller.render_row_param_text(
        "positivemuon:Site 1", {"geom_max_iter": 777}, ()
    )

    assert "GEOM_MAX_ITER : 777" in text
    assert definition.parameters["configuration"] == before
    assert controller.render_row_param_text("unknown:Site 1", {}, ()) == ""


def test_muon_controller_continue_job_returns_to_defined_and_rewrites_directory(
    tmp_path: Path,
) -> None:
    from mufinder.domain.models import utc_now

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    controller.materialize(["positivemuon:Site 1"])
    seed_mapping_before = dict(definition.parameters["job_seed_names_by_label"])
    seed = seed_mapping_before["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed / f"{seed}.param"
    assert "GEOM_MAX_ITER : 120" in param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.execution_target = "myhpc"
    job.remote_directory = "/scratch/user/site1"
    job.remote_job_id = "999"
    job.error_message = "boom"
    job.dismissed_at = utc_now()

    continued = controller.continue_job("positivemuon:Site 1", {"geom_max_iter": 500})

    assert continued is job
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/site1"
    assert job.remote_job_id is None
    assert job.error_message is None
    assert job.dismissed_at is None

    assert list(document.jobs) == [job.id]
    assert definition.parameters["job_ids_by_label"] == {
        "positivemuon:Site 1": str(job.id)
    }
    # A changed seed name would relocate the directory and orphan the
    # preserved remote directory.
    assert definition.parameters["job_seed_names_by_label"] == seed_mapping_before
    rendered = param_path.read_text(encoding="utf-8")
    assert "GEOM_MAX_ITER : 500" in rendered
    assert "GEOM_MAX_ITER : 120" not in rendered

    continuation = next(
        record for record in document.provenance if record.action == "job_continuation"
    )
    assert continuation.job_id == job.id
    assert continuation.parameters["remote_directory"] == "/scratch/user/site1"


def test_muon_controller_continue_job_rejects_non_continuable_job_without_mutating(
    tmp_path: Path,
) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    job.status = JobStatus.RUNNING
    assert controller.continue_job("positivemuon:Site 1", {"geom_max_iter": 500}) is None

    assert job.status == JobStatus.RUNNING
    assert definition.parameters["configuration"] == before
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_muon_controller_continue_job_rejects_bare_label(tmp_path: Path) -> None:
    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    before = copy.deepcopy(definition.parameters["configuration"])

    job.status = JobStatus.FAILED
    assert controller.continue_job("Site 1", {"geom_max_iter": 500}) is None

    assert job.status == JobStatus.FAILED
    assert definition.parameters["configuration"] == before


def test_muon_controller_continue_job_returns_none_when_materialize_fails(
    tmp_path: Path,
) -> None:
    """Muon counterpart of the pristine "materialize can fail silently" check
    above (audit finding 1)."""

    service = ProjectService()
    document, definition, job, deps, view, output_root = _muon_parameter_edit_fixture(
        service, tmp_path
    )
    persists: list[dict] = []
    controller = _muon_edit_controller(
        service, document, deps, view, output_root, persists
    )
    controller.materialize(["positivemuon:Site 1"])
    seed_mapping_before = dict(definition.parameters["job_seed_names_by_label"])
    seed = seed_mapping_before["positivemuon:Site 1"]
    param_path = output_root / "positivemuon" / seed / f"{seed}.param"
    original_rendered = param_path.read_text(encoding="utf-8")

    job.status = JobStatus.FAILED
    job.remote_directory = "/scratch/user/site1"
    controller.materialize = lambda keys: []

    result = controller.continue_job("positivemuon:Site 1", {"geom_max_iter": 500})

    assert result is None
    config = deserialize_muon_relaxation_config(definition.parameters["configuration"])
    by_charge = {row.charge_state: row for row in config.rows}
    assert by_charge["positive_muon"].parameter_values == {"geom_max_iter": 500}
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == "/scratch/user/site1"
    assert param_path.read_text(encoding="utf-8") == original_rendered
    assert not any(record.action == "job_continuation" for record in document.provenance)


def test_muon_controller_submit_cluster_reaches_cluster_backend(tmp_path: Path) -> None:
    service = ProjectService()
    document, _definition, deps, output_root = _muon_two_row_submission_fixture(
        service, tmp_path
    )
    local = RecordingBackend()
    cluster = RecordingBackend()
    context = make_context(
        document=document,
        project_service=service,
        local_backend=local,
        cluster_backend=cluster,
        definition_output_directory=lambda _defn: output_root,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    controller.submit(["positivemuon:Site 1", "muonium:Site 2"], target="cluster")

    assert local.calls == []
    assert [
        [job.calc_type for job in group] for group, _extras in cluster.calls
    ] == [["castep_positivemuon_relaxation"], ["castep_muonium_relaxation"]]


def test_muon_controller_build_result_records_position_structure_and_site_mutation(
    tmp_path: Path,
) -> None:
    """build_result with a REAL -out.cell containing an H:mu muon contributes
    the lattice + folded-position payload, creates the derived structure, and
    mutates the source site (the sanctioned document side effect)."""

    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    row = MuonRelaxationRow(
        site_id=site.id,
        label="Shared Site",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="muonium",
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition

    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # A REAL -out.cell fixture (2x2x1 supercell of the 4 A host) on disk, fed
    # through the controller's parse + position-extraction helpers.
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(_MUON_OUT_CELL_TEXT, encoding="utf-8")

    deps = _stub_muon_deps(
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
    )
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
    artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    result = controller.build_result(job, parsed, run, artifacts)

    # Relaxed supercell lattice payload.
    assert result.payload["lattice_a"] == pytest.approx(8.0)
    assert result.payload["lattice_b"] == pytest.approx(8.0)
    assert result.payload["lattice_c"] == pytest.approx(4.0)
    # Muon position: raw supercell fraction, and folded back to host coordinates
    # through the row's 2x2x1 supercell matrix.
    assert result.payload["muon_final_fractional_position_supercell"] == pytest.approx(
        [0.3, 0.4, 0.5]
    )
    assert result.payload["muon_final_fractional_position_host"] == pytest.approx(
        [0.6, 0.8, 0.5]
    )
    # No output_files contributed — the call site already passes them generically.
    assert result.output_files == []

    # Exactly one derived structure created on the document, with the right
    # origin, parentage, and the muon present.
    assert len(result.derived_structure_ids) == 1
    derived = document.structures[result.derived_structure_ids[0]]
    assert derived.origin == "muon_relaxation"
    assert derived.parent_structure_id == host.id
    assert derived.label == "Shared Site relaxed"
    assert derived.properties["source_job_id"] == str(job.id)
    assert derived.properties["source_workflow_stage"] == WorkflowStage.MUON_RELAXATION.value
    assert any(str(s).strip().casefold() == "h:mu" for s in derived.species)

    # Sanctioned document side effect: the source site is marked relaxed and
    # linked to this run with the job's total energy and host-cell final
    # position; its initial position is left alone.
    assert site.status == "relaxed"
    assert site.relaxation_run_id == run.id
    assert site.total_energy_eV == pytest.approx(-12.0)
    assert site.relaxed_fractional_position == pytest.approx([0.6, 0.8, 0.5])
    assert site.initial_fractional_position == pytest.approx([0.1, 0.2, 0.3])


_MUON_OUT_CELL_TEXT_CONTINUED = """
%BLOCK LATTICE_CART
ang
8.2 0.0 0.0
0.0 8.2 0.0
0.0 0.0 4.1
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
H:mu 0.35 0.45 0.55
%ENDBLOCK POSITIONS_FRAC
"""


def _muon_row_and_definition(document, host, site, charge_state: str):
    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition

    row = MuonRelaxationRow(
        site_id=site.id,
        label=site.label,
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state=charge_state,
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition
    return row, definition


def test_muon_controller_recompletion_updates_structure_and_site_to_new_run(
    tmp_path: Path,
) -> None:
    """A continued muon job's re-completion refreshes the same derived structure
    and re-points the site's ``relaxation_run_id``/``total_energy_eV`` at the
    new run, so the site and the structure never describe different runs."""

    from mufinder.domain.models import MuonSite, Result

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    row, definition = _muon_row_and_definition(document, host, site, "muonium")

    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{job.label}-out.cell"

    deps = _stub_muon_deps(
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
    )
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    def complete(run_index: int, cell_text: str, total_energy: float):
        run = CalculationRun(job_id=job.id, run_index=run_index)
        document.runs[run.id] = run
        out_cell.write_text(cell_text, encoding="utf-8")
        artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
        parsed = CastepResult(completed=True, total_energy=total_energy, warnings=())
        controller_result = controller.build_result(job, parsed, run, artifacts)
        result = Result(
            calculation_run_id=run.id,
            result_type=job.calc_type,
            derived_structure_ids=list(controller_result.derived_structure_ids),
        )
        document.results[result.id] = result
        run.result_id = result.id
        return run, result, controller_result

    run_one, first_result, _first = complete(1, _MUON_OUT_CELL_TEXT, -12.0)
    first_id = first_result.derived_structure_ids[0]
    structure_count = len(document.structures)
    assert site.relaxation_run_id == run_one.id

    run_two, second_result, second = complete(2, _MUON_OUT_CELL_TEXT_CONTINUED, -12.5)

    # Stable structure id, no duplicate, both Results converge on it.
    assert second_result.derived_structure_ids == [first_id]
    assert len(document.structures) == structure_count
    assert first_result.derived_structure_ids == second_result.derived_structure_ids

    derived = document.structures[first_id]
    assert derived.lattice_vectors == [[8.2, 0.0, 0.0], [0.0, 8.2, 0.0], [0.0, 0.0, 4.1]]
    assert derived.fractional_positions[-1] == pytest.approx([0.35, 0.45, 0.55])
    assert derived.parent_structure_id == host.id
    # The refresh replaces the geometry as a set: species and positions must not
    # be able to drift out of step with each other.
    assert len(derived.species) == len(derived.fractional_positions)
    assert derived.species[-1].strip().casefold() == "h:mu"

    # Payload follows the new run, and so do the site fields.
    assert second.payload["muon_final_fractional_position_supercell"] == pytest.approx(
        [0.35, 0.45, 0.55]
    )
    assert site.status == "relaxed"
    assert site.relaxation_run_id == run_two.id
    assert site.relaxation_run_id != run_one.id
    assert site.total_energy_eV == pytest.approx(-12.5)
    assert site.relaxed_fractional_position == pytest.approx([0.7, 0.9, 0.55])
    assert service.validate_document(document) == []


def test_muon_controller_recompletion_keeps_charge_states_separate(
    tmp_path: Path,
) -> None:
    """One site's two charge states are two jobs with the SAME ``job.label`` and
    two derived structures with the same ``"{label} relaxed"`` label. Each job's
    re-completion must refresh its own structure — the reason the lookup keys on
    the job id and never on the label."""

    from mufinder.domain.models import MuonSite

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site

    positive_row, definition = _muon_row_and_definition(document, host, site, "positive_muon")
    muonium_row, _ = _muon_row_and_definition(document, host, site, "muonium")

    positive_job = _muon_job(
        service, document, "Shared Site", "castep_positivemuon_relaxation"
    )
    muonium_job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    assert positive_job.label == muonium_job.label

    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    out_cell = output_dir / f"{positive_job.label}-out.cell"
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )

    def controller_for(row):
        deps = _stub_muon_deps(
            muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
        )
        return MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    def complete(controller, job, run_index: int, cell_text: str):
        run = CalculationRun(job_id=job.id, run_index=run_index)
        document.runs[run.id] = run
        out_cell.write_text(cell_text, encoding="utf-8")
        artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
        parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
        return controller.build_result(job, parsed, run, artifacts)

    positive_controller = controller_for(positive_row)
    muonium_controller = controller_for(muonium_row)

    positive_id = complete(
        positive_controller, positive_job, 1, _MUON_OUT_CELL_TEXT
    ).derived_structure_ids[0]
    muonium_id = complete(
        muonium_controller, muonium_job, 1, _MUON_OUT_CELL_TEXT
    ).derived_structure_ids[0]
    assert positive_id != muonium_id
    assert (
        document.structures[positive_id].label == document.structures[muonium_id].label
    )

    # Continue only the positive-muon job: its structure moves, the muonium
    # structure is untouched.
    second_positive = complete(
        positive_controller, positive_job, 2, _MUON_OUT_CELL_TEXT_CONTINUED
    )
    assert second_positive.derived_structure_ids == [positive_id]
    assert document.structures[positive_id].lattice_vectors == [
        [8.2, 0.0, 0.0],
        [0.0, 8.2, 0.0],
        [0.0, 0.0, 4.1],
    ]
    assert document.structures[muonium_id].lattice_vectors == [
        [8.0, 0.0, 0.0],
        [0.0, 8.0, 0.0],
        [0.0, 0.0, 4.0],
    ]


def test_muon_controller_build_result_fails_loudly_on_muonless_cell(tmp_path: Path) -> None:
    """The success-path invariant is self-contained: if build_result is ever
    reached with a relaxed cell that lacks the muon (validate_artifacts' job to
    prevent), it raises a descriptive error instead of a deep TypeError."""

    from mufinder.app.relaxation_types import MuonRelaxationRow
    from mufinder.domain.models import CalculationDefinition, MuonSite

    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    site = MuonSite(
        label="Shared Site",
        status="staged",
        initial_fractional_position=[0.1, 0.2, 0.3],
        source_structure_id=host.id,
    )
    document.sites[site.id] = site
    row = MuonRelaxationRow(
        site_id=site.id,
        label="Shared Site",
        initial_fractional_position=(0.1, 0.2, 0.3),
        charge_state="muonium",
        supercell_matrix=((2, 0, 0), (0, 2, 0), (0, 0, 1)),
    )
    definition = CalculationDefinition(
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_muon_relaxation",
        parameters={},
        input_structure_ids=[host.id],
        input_site_ids=[site.id],
    )
    document.calculations[definition.id] = definition
    job = _muon_job(service, document, "Shared Site", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    document.runs[run.id] = run

    # Same fixture with the muon line removed: extraction returns None.
    muonless_text = "\n".join(
        line for line in _MUON_OUT_CELL_TEXT.splitlines() if "h:mu" not in line.casefold()
    )
    output_dir = tmp_path / "job_out"
    output_dir.mkdir()
    (output_dir / f"{job.label}-out.cell").write_text(muonless_text, encoding="utf-8")

    deps = _stub_muon_deps(
        muon_relaxation_definition_and_row_for_job=lambda _job: (definition, row),
    )
    context = make_context(
        document=document,
        project_service=service,
        job_output_directory=lambda _job: output_dir,
    )
    controller = MuonRelaxationController(context, deps, _FakeMuonRelaxationView())

    parsed = CastepResult(completed=True, total_energy=-12.0, warnings=())
    # The muonless cell still parses (lattice + Si present), so the carrier's
    # ``relaxed_cell``/``lattice_abc`` are populated — build_result proceeds past
    # the empty-carrier guard and only then raises on the missing H:mu position.
    artifacts = collect_job_artifacts(output_dir, job.label, want_out_cell=True)
    with pytest.raises(ValueError, match="relaxed cell containing"):
        controller.build_result(job, parsed, run, artifacts)
    # No partial side effects: site untouched, no derived structure added.
    assert site.status == "staged"
    assert site.relaxation_run_id is None


def test_muon_controller_validate_artifacts_requires_the_muon(tmp_path: Path) -> None:
    service = ProjectService()
    document, _host = muon_host_document(service, tmp_path)
    job = _muon_job(service, document, "Site 1", "castep_muonium_relaxation")
    run = CalculationRun(job_id=job.id)
    controller = MuonRelaxationController(
        make_context(document=document), _stub_muon_deps(), _FakeMuonRelaxationView()
    )
    parsed = CastepResult(completed=True, total_energy=-1.0, geom_converged=True)

    muonated_dir = tmp_path / "muonated"
    muonated_dir.mkdir()
    (muonated_dir / f"{job.label}-out.cell").write_text(_MUON_OUT_CELL_TEXT, encoding="utf-8")
    muonless_dir = tmp_path / "muonless"
    muonless_dir.mkdir()
    (muonless_dir / f"{job.label}-out.cell").write_text(
        "\n".join(
            line for line in _MUON_OUT_CELL_TEXT.splitlines() if "h:mu" not in line.casefold()
        ),
        encoding="utf-8",
    )

    assert controller.validate_artifacts(
        job, parsed, run, collect_job_artifacts(muonated_dir, job.label, want_out_cell=True)
    ) is None
    assert controller.validate_artifacts(
        job, parsed, run, collect_job_artifacts(muonless_dir, job.label, want_out_cell=True)
    ) == "Expected H:mu muon position was not found in Site 1-out.cell."
    # The shared relaxation checks run first.
    assert controller.validate_artifacts(
        job, parsed, run, ParsedJobArtifacts(cell_missing=True)
    ) == "Expected relaxed structure output was not found: Site 1-out.cell"
    assert controller.validate_artifacts(
        job, CastepResult(completed=True, geom_converged=False), run, ParsedJobArtifacts()
    ) == "Geometry optimization did not satisfy CASTEP convergence criteria."


# --------------------------------------------------------------------------- #
# Definition maintenance: sync_mappings and remove_rows
# --------------------------------------------------------------------------- #
#
# Every controller syncs and removes rows through
# ``mufinder.services.definition_mappings``; these tests pin what each workflow
# hands it: row keys and their legacy spellings, row signatures, the priority
# order, seed names and the active-definition key.


def _muon_sync_rows(site_id: UUID) -> dict[str, object]:
    from mufinder.app.relaxation_types import MuonRelaxationConfig, MuonRelaxationRow

    rows = tuple(
        MuonRelaxationRow(
            site_id=site_id,
            label=label,
            initial_fractional_position=(0.25, 0.25, 0.25),
            charge_state=charge_state,
        )
        for label, charge_state in (("Site 1", "positive_muon"), ("Site 2", "muonium"))
    )
    return serialize_muon_relaxation_config(MuonRelaxationConfig(rows=rows))


def _muon_sync_controller(document: ProjectDocument) -> MuonRelaxationController:
    return MuonRelaxationController(
        make_context(document=document),
        _stub_muon_deps(deserialize_muon_relaxation_config=deserialize_muon_relaxation_config),
        _FakeMuonRelaxationView(),
    )


def test_muon_controller_sync_mappings_reads_legacy_keys_and_carries_seed_names(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    site_id = uuid4()
    output = document.project_root / "files" / "muon_relaxation"
    source, other_source, target = (
        service.register_muon_relaxation_definition(
            document, input_structure_id=host.id, parameters=_muon_sync_rows(site_id), output_directory=output
        )
        for _ in range(3)
    )

    def job(status: JobStatus, calc_type: str) -> str:
        return job_with_status(
            service, document, status, stage=WorkflowStage.MUON_RELAXATION, calc_type=calc_type
        )

    completed = job(JobStatus.COMPLETED, "castep_positivemuon_relaxation")
    defined = job(JobStatus.DEFINED, "castep_muonium_relaxation")
    wrong_charge = job(JobStatus.COMPLETED, "castep_positivemuon_relaxation")
    source.parameters["job_ids_by_label"] = {
        # A bare key from before rows were charge-qualified.
        "Site 1": completed,
        # A job of the other charge state is not carried over...
        "muonium:Site 2": wrong_charge,
    }
    # ...nor is a job that never left ``defined``.
    other_source.parameters["job_ids_by_label"] = {"muonium:Site 2": defined}
    source.parameters["job_seed_names_by_label"] = {"Site 1": "  site1_seed  "}

    _muon_sync_controller(document).sync_mappings(target)

    assert target.parameters["job_ids_by_label"] == {"positivemuon:Site 1": completed}
    assert target.parameters["job_seed_names_by_label"] == {"positivemuon:Site 1": "site1_seed"}


def test_muon_controller_row_key_matcher_crosses_key_spellings() -> None:
    controller = MuonRelaxationController(make_context(), _stub_muon_deps(), _FakeMuonRelaxationView())

    matches = controller.row_key_matcher({"Site 1", "muonium:Site 2"})

    assert matches("positivemuon:Site 1")
    assert matches("Site 1")
    assert matches("Site 2")
    assert matches("muonium:Site 2")
    assert not matches("positivemuon:Site 2")
    assert not matches(3)


def test_muon_controller_remove_rows_drops_job_and_seed_bindings(tmp_path: Path) -> None:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    definition = service.register_muon_relaxation_definition(
        document,
        input_structure_id=host.id,
        parameters=_muon_sync_rows(uuid4()),
        output_directory=document.project_root / "files" / "muon_relaxation",
        job_ids_by_label={"positivemuon:Site 1": "1", "muonium:Site 2": "2"},
    )
    definition.parameters["job_seed_names_by_label"] = {"Site 1": "seed", "muonium:Site 2": "seed2"}

    assert _muon_sync_controller(document).remove_rows(["Site 1"]) == ()

    rows = definition.parameters["configuration"]["rows"]
    assert [(row["label"], row["charge_state"]) for row in rows] == [("Site 2", "muonium")]
    assert definition.parameters["job_ids_by_label"] == {"muonium:Site 2": "2"}
    assert definition.parameters["job_seed_names_by_label"] == {"muonium:Site 2": "seed2"}
