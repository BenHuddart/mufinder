"""Energy-barriers (NEB) workflow controller.

Owns the submit → materialize → result → refresh cycle for the CASTEP
nudged-elastic-band (NEB) diffusion-path workflow. Each selected diffusion path
runs as a single NEB band job whose endpoints are the relaxed muon sites; the
intermediate images are seeded by IDPP interpolation in
:meth:`EnergyBarriersController.materialize`.

Four features distinguish this controller from the others:

- **Source-key provenance.** Jobs are keyed by their *diffusion-path source key*
  (not a site label or sweep point). :meth:`materialize` writes
  ``job_ids_by_label`` keyed by source key and stamps each expanded spec with
  ``provenance={"source_key": key}``, then reverse-maps ``spec_by_key`` to attach
  the created job to the right key.
- **IDPP intermediate interpolation.** :meth:`materialize` calls the science-layer
  ``prepare_transition_state_structures`` to align the reactant/product endpoints
  into the pristine host, then ``idpp_intermediate_structure`` to seed the band's
  middle image, before expanding through the workflow-layer
  ``expand_nudged_elastic_band_jobs``.
- **``.ts`` transition-state result payload.**
  :meth:`EnergyBarriersController.build_result` parses the completed ``.ts`` file
  and contributes ``transition_state_file``, ``barrier_height_eV``, ``num_images``,
  ``image_labels``, ``reaction_coordinates``, and ``image_energies_eV``. Following
  the success-path-only convention, the failure guards (missing/unparseable
  ``.ts``) stay inline in ``MainWindow._on_local_job_finished`` and abort the job
  before this controller's ``build_result`` runs; ``build_result`` therefore
  re-parses on the validated success path. Both the NEB and the
  transition-state-search calc types are registered on :attr:`calc_types` so
  :meth:`handles_job` is coherent across the ``.ts``-producing pair (materialize
  only creates NEB jobs today, but the inline guard and this payload handle both
  identically); ``build_result`` returns an empty
  :class:`~mufinder.app.workflow_controllers.base.ControllerResult` when the
  ``.ts`` output cannot be resolved (mirroring the inline guard's abort).

- **Lazily-created view (sanctioned constructor-shape deviation).** Unlike the
  other controllers, whose views exist at ``MainWindow.__init__``, the
  energy-barriers view is created *lazily* — only when the workflow route is first
  opened (``_ensure_energy_barriers_view``). But the controller must still be
  constructed and registered in ``__init__`` regardless, because a completed NEB
  job must dispatch through ``merge_controller_result`` even if the view was never
  opened. Therefore this controller takes a
  ``view_getter: Callable[[], EnergyBarriersWidget | None]`` instead of a view
  instance (the ONLY sanctioned constructor-shape deviation across the seven
  migrations), and :meth:`refresh_view` no-ops when the getter returns ``None`` —
  mirroring exactly how the former ``MainWindow._refresh_energy_barriers_view``
  handled the not-yet-created view. Signal wiring to ``controller.submit`` happens
  at lazy view creation, as before; result dispatch and refresh both work before
  the view exists (``build_result`` reads the ``.ts`` via deps and touches no view;
  ``refresh_view`` no-ops).

The host-mismatch pre-submit guard, the definition/source/config/row helpers, and
the completed-candidate/result-row helpers stay on ``MainWindow`` (they are shared
with the path-generation and configuration surfaces that are NOT migrating) and
are reached via :class:`EnergyBarriersDeps`, following the shared-context +
per-controller-deps + view pattern documented in
:mod:`mufinder.app.workflow_controllers.base`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationRun, Job, Structure
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.science.energy_barriers import (
    EnergyBarrierError,
    idpp_intermediate_structure,
    prepare_transition_state_structures,
)
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    expand_nudged_elastic_band_jobs,
)
from mufinder.workflow.types import InputTemplate

from mufinder.app.energy_barriers_widget import (
    EnergyBarriersWidget,
    EnergyBarrierEndpointRow,
    EnergyBarrierNebResultRow,
    EnergyBarrierSourceRow,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
)


@dataclass(frozen=True)
class EnergyBarriersDeps:
    """Energy-barriers-specific handles, late-bound to ``MainWindow`` helpers.

    These helpers stay on ``MainWindow`` because non-migrating code still calls
    them: the definition/source/candidate/config helpers are shared with the
    path-generation, source-removal, and result-configuration surfaces (which are
    NOT migrating — this is the last submit/materialize/result migration);
    ``handle_structure_host_mismatch`` / ``handle_site_host_mismatch`` are shared
    across every host-guarded workflow; the endpoint-row / max-path-length /
    selected-endpoint helpers back the path-generation UI the controller now
    renders; ``neb_barrier_summaries`` and the completed-candidate / result-row
    helpers back the result table; and ``dedupe_jobs`` /
    ``charge_state_label_for_job`` / ``workflow_row_configuration_exists`` /
    ``project_convergence_input_defaults`` are shared across workflows. The
    source-decoding helpers (``uuid_from_source`` / ``float_from_source`` /
    ``fractional_triplet_from_source`` / ``fractional_position_block_from_structure``)
    and ``energy_barrier_neb_row_configuration_with_source_defaults`` are shared
    with the same non-migrating surfaces. ``input_template_from_keyword_values``
    wraps the shared module-level keyword→template helper, and
    ``neb_barrier_summaries`` / ``muon_charge_state_label`` wrap module-level
    formatting helpers (the controller must not import ``main_window``). The
    controller reaches them only through these required callables. Workflow-specific
    dependencies like these never go in the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`;
    cross-workflow job/definition readers (``definition_output_directory``,
    ``job_output_directory``, ``job_seed_name``) live there instead.
    """

    # Definition / source / candidate helpers
    active_energy_barrier_neb_definition: Callable[[], object | None]
    build_energy_barrier_neb_states: Callable[[], Mapping[str, object]]
    energy_barrier_neb_definitions: Callable[[], list[object]]
    ensure_energy_barrier_neb_definition: Callable[[list[str]], object | None]
    energy_barrier_neb_source_candidates: Callable[[], list[object]]
    eligible_energy_barrier_source_candidates: Callable[[], list[object]]
    job_for_energy_barrier_neb_source_key: Callable[[object, str], Job | None]
    energy_barrier_neb_row_configuration_with_source_defaults: Callable[
        ..., tuple[dict[str, object], tuple[str, ...], dict[str, object]]
    ]
    # Reads the per-row CASTEP task-farm count out of the row's configuration
    # extras (stored by the NEB configure dialog, not a CASTEP keyword row).
    num_farms_from_extras: Callable[[Mapping[str, object]], int]
    # Cheap extras-only read for the sources table. The full
    # ``*_with_source_defaults`` variant re-reads the reactant's output files to
    # inherit its cut-off/k-points, which is far too heavy for a per-row refresh.
    energy_barrier_neb_row_extras: Callable[[object, str], Mapping[str, object]]
    # Pre-submit host guard (host mismatch is shared with every host-guarded workflow)
    handle_structure_host_mismatch: Callable[[object], bool]
    handle_site_host_mismatch: Callable[[object], bool]
    # Source-value decoding
    uuid_from_source: Callable[[Mapping[str, object], str], UUID | None]
    float_from_source: Callable[[Mapping[str, object], str], float | None]
    fractional_triplet_from_source: Callable[
        [Mapping[str, object], str], tuple[float, float, float]
    ]
    fractional_position_block_from_structure: Callable[[Structure], object]
    # Shared workflow helpers
    dedupe_jobs: Callable[[list[Job]], list[Job]]
    project_convergence_input_defaults: Callable[[], object]
    input_template_from_keyword_values: Callable[[Mapping[str, object]], InputTemplate]
    workflow_row_configuration_exists: Callable[[object, str], bool]
    charge_state_label_for_job: Callable[[Job | None], str]
    # View-refresh helpers (endpoint options + result table)
    energy_barrier_endpoint_rows: Callable[[], list[EnergyBarrierEndpointRow]]
    selected_energy_barrier_endpoint_keys: Callable[[], tuple[str, str]]
    energy_barrier_max_path_length_text: Callable[[], str]
    completed_energy_barrier_neb_candidates: Callable[[], list[object]]
    energy_barrier_neb_sources_by_job_id: Callable[[], dict[UUID, dict[str, object]]]
    energy_barrier_neb_result_path_label: Callable[
        [object, Mapping[str, object] | None], str
    ]
    neb_barrier_summaries: Callable[[Mapping[str, object]], tuple[str, str, str]]
    muon_charge_state_label: Callable[[str], str]


class EnergyBarriersController(WorkflowJobController):
    """Controller for the energy-barriers (NEB) diffusion-path workflow."""

    calc_types = frozenset(
        {NUDGED_ELASTIC_BAND_CALC_TYPE, TRANSITION_STATE_SEARCH_CALC_TYPE}
    )
    no_ready_message = (
        "Only NEB paths that have not yet been submitted can be submitted."
    )

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: EnergyBarriersDeps,
        view_getter: Callable[[], EnergyBarriersWidget | None],
    ) -> None:
        super().__init__(context)
        self._deps = deps
        # Sanctioned constructor-shape deviation: the energy-barriers view is
        # created lazily, so this controller takes a getter rather than a view
        # instance. Everything else about the controller (construction and
        # registration in ``__init__``) is unchanged, so completed NEB jobs
        # dispatch through ``build_result`` even when the view was never opened.
        self._view_getter = view_getter

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # mirror the former ``_on_submit_*_neb_from_energy_barriers`` handlers. There is
    # no submission_groups override: NEB jobs submit as a single group.

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the selected paths' endpoint structures/sites against the host.

        Behavior-preserving move of ``MainWindow._check_energy_barrier_neb_host``.
        NEB can be submitted before any NEB definition exists, so this validates
        the actual endpoint structures/sites of the selected paths rather than a
        possibly-``None`` definition. Returns ``True`` if submission should proceed
        (no mismatch, or the user resolved it by switching hosts); ``False`` if the
        user cancelled or the mismatch could not be resolved. A refusal is silent.
        """

        document = self._context.document()
        if document is None:
            return True
        candidates = {
            candidate.key: candidate
            for candidate in self._deps.energy_barrier_neb_source_candidates()
        }
        structure_ids: list[UUID] = []
        site_ids: list[UUID] = []
        for key in keys:
            candidate = candidates.get(str(key).strip())
            if candidate is None:
                continue
            for field in ("reactant_structure_id", "product_structure_id"):
                value = self._deps.uuid_from_source(candidate.source, field)
                if value is not None and value not in structure_ids:
                    structure_ids.append(value)
            for field in ("reactant_site_id", "product_site_id"):
                value = self._deps.uuid_from_source(candidate.source, field)
                if value is not None and value not in site_ids:
                    site_ids.append(value)
        if structure_ids:
            struct_mismatch = (
                self._context.project_service.validate_structures_match_active_host(
                    document, structure_ids
                )
            )
            if struct_mismatch is not None:
                return self._deps.handle_structure_host_mismatch(struct_mismatch)
        if site_ids:
            site_mismatch = self._context.project_service.validate_sites_match_active_host(
                document, site_ids
            )
            if site_mismatch is not None:
                return self._deps.handle_site_host_mismatch(site_mismatch)
        return True

    def build_states(self) -> Mapping[str, object]:
        return self._deps.build_energy_barrier_neb_states()

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        str_keys = [str(key) for key in keys]
        for definition in self._deps.energy_barrier_neb_definitions():
            for key in str_keys:
                job = self._deps.job_for_energy_barrier_neb_source_key(definition, key)
                if job is None or job.id in seen:
                    continue
                selected.append(job)
                seen.add(job.id)
        return selected

    def materialize(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        str_keys = [str(key) for key in keys]
        definition = self._deps.ensure_energy_barrier_neb_definition(str_keys)
        if definition is None:
            return []
        output_root = self._context.definition_output_directory(definition)
        host_id = document.project.host_structure_id
        host = document.structures.get(host_id) if host_id is not None else None
        if output_root is None or host is None:
            return []

        candidates_by_key = {
            candidate.key: candidate
            for candidate in self._deps.energy_barrier_neb_source_candidates()
        }
        source_candidates_by_key = {
            candidate.key: candidate
            for candidate in self._deps.eligible_energy_barrier_source_candidates()
        }
        source_defaults_cache: dict[UUID, dict[str, object]] = {}
        configs: list[NudgedElasticBandJobConfig] = []
        for key in str_keys:
            candidate = candidates_by_key.get(key)
            if candidate is None:
                continue
            reactant_id = self._deps.uuid_from_source(
                candidate.source, "reactant_structure_id"
            )
            product_id = self._deps.uuid_from_source(
                candidate.source, "product_structure_id"
            )
            reactant_site_id = self._deps.uuid_from_source(
                candidate.source, "reactant_site_id"
            )
            product_site_id = self._deps.uuid_from_source(
                candidate.source, "product_site_id"
            )
            reactant_run_id = self._deps.uuid_from_source(
                candidate.source, "reactant_run_id"
            )
            product_run_id = self._deps.uuid_from_source(
                candidate.source, "product_run_id"
            )
            reactant_job_id = self._deps.uuid_from_source(
                candidate.source, "reactant_job_id"
            )
            product_job_id = self._deps.uuid_from_source(
                candidate.source, "product_job_id"
            )
            reactant_structure = (
                document.structures.get(reactant_id) if reactant_id is not None else None
            )
            product_structure = (
                document.structures.get(product_id) if product_id is not None else None
            )
            if reactant_structure is None or product_structure is None:
                continue
            try:
                prepared = prepare_transition_state_structures(
                    pristine_structure=host,
                    reactant_structure=reactant_structure,
                    product_structure=product_structure,
                    reactant_anchor_fractional=self._deps.fractional_triplet_from_source(
                        candidate.source, "path_start_fractional"
                    ),
                    product_anchor_fractional=self._deps.fractional_triplet_from_source(
                        candidate.source, "path_end_fractional"
                    ),
                )
                reactant_spec = structure_to_spec(prepared.reactant)
                product_block = self._deps.fractional_position_block_from_structure(
                    prepared.product
                )
                intermediate_structure = idpp_intermediate_structure(
                    prepared.reactant, prepared.product
                )
                intermediate_block = self._deps.fractional_position_block_from_structure(
                    intermediate_structure
                )
            except (EnergyBarrierError, StructureBridgeError, ValueError):
                continue
            values, _advanced_names, extras = (
                self._deps.energy_barrier_neb_row_configuration_with_source_defaults(
                    definition,
                    key,
                    target_lattice=reactant_structure.lattice_vectors,
                    source_candidate=source_candidates_by_key.get(key),
                    source_defaults_cache=source_defaults_cache,
                )
            )
            # ``num_farms`` is dropped alongside the workflow-locked ``task``: the
            # farm count comes from the row's extras (below), which is also what
            # decides the expected output filenames and the submission rank check.
            # A keyword-row copy would override the base template and silently
            # disagree with both. The configure dialog blocks the row, so this
            # only bites on configurations persisted by other paths.
            values = {
                name: value
                for name, value in values.items()
                if name not in {"task", "num_farms"}
            }
            try:
                overrides = self._deps.input_template_from_keyword_values(values)
            except ValueError:
                continue
            configs.append(
                NudgedElasticBandJobConfig(
                    source=NudgedElasticBandSource(
                        label=f"{str(candidate.source.get('label', '')).strip() or candidate.key}_neb",
                        reactant_structure=reactant_spec,
                        product_positions_frac=product_block,
                        intermediate_positions_frac=intermediate_block,
                        product_atomic_spins=reactant_spec.atomic_spins,
                        intermediate_atomic_spins=reactant_spec.atomic_spins,
                        source_site_ids=tuple(
                            value
                            for value in (reactant_site_id, product_site_id)
                            if value is not None
                        ),
                        source_relaxation_run_ids=tuple(
                            value
                            for value in (reactant_run_id, product_run_id)
                            if value is not None
                        ),
                        source_relaxation_job_ids=(reactant_job_id, product_job_id),
                        path_length_ang=self._deps.float_from_source(
                            candidate.source, "path_length_ang"
                        ),
                        provenance={"source_key": key},
                    ),
                    num_farms=self._deps.num_farms_from_extras(extras),
                    cell_overrides=overrides.cell,
                    param_overrides=overrides.param,
                )
            )
        if not configs:
            return []
        try:
            specs = expand_nudged_elastic_band_jobs(
                jobs=configs,
                name_prefix="barrier",
                global_defaults=self._deps.project_convergence_input_defaults(),
            )
        except (ValueError, StructureBridgeError):
            return []

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_key = {
            str(spec.provenance.get("source_key", "")): spec for spec in specs
        }
        created: list[Job] = []
        for key in str_keys:
            spec = spec_by_key.get(key)
            existing = self._deps.job_for_energy_barrier_neb_source_key(definition, key)
            if existing is not None:
                if existing.status == JobStatus.DEFINED and spec is not None:
                    write_job_directory(spec, output_root)
                created.append(existing)
                continue
            if spec is None:
                continue
            write_job_directory(spec, output_root)
            job = self._context.project_service.add_job(
                document,
                label=spec.name,
                workflow_stage=spec.workflow_stage,
                calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE,
            )
            mapping[key] = str(job.id)
            created.append(job)

        jobs = self._deps.dedupe_jobs(created)
        if jobs:
            self.refresh_view()
            self._context.persist_ui_state(
                failure_message=(
                    "Energy Barriers NEB jobs were generated, but autosave failed."
                )
            )
        return jobs

    # --- result contract ---

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the ``.ts`` transition-state payload for a completed NEB job.

        The inline ``_on_local_job_finished`` guard has already verified the ``.ts``
        file exists and parses (aborting the job on a missing file or
        ``TransitionStateParseError``), so the carried ``artifacts.transition_state``
        is present on this success path. The file is parsed **once** on the runner
        worker thread; this controller consumes the carried path with no second
        disk read. When the parsed path is absent (defensive), an empty
        :class:`~mufinder.app.workflow_controllers.base.ControllerResult` is
        returned. The ``.ts`` file itself is **not** contributed to
        ``output_files``: the call site already passes ``_output_files_for_job(job)``.

        Both the NEB and transition-state-search calc types are owned so
        :meth:`handles_job` is coherent across the ``.ts``-producing pair; the
        payload is identical for either.
        """

        transition_state = artifacts.transition_state
        if transition_state is None:
            return ControllerResult()
        ts_path = artifacts.ts_path
        if ts_path is None:
            output_dir = self._context.job_output_directory(job)
            seed_name = self._context.job_seed_name(job)
            ts_path = None if output_dir is None else output_dir / f"{seed_name}.ts"
        payload: dict[str, object] = {
            "transition_state_file": str(ts_path) if ts_path is not None else "",
            "barrier_height_eV": transition_state.barrier_height_eV,
            "num_images": len(transition_state.images),
            "image_labels": [image.label for image in transition_state.images],
            "reaction_coordinates": [
                image.reaction_coordinate for image in transition_state.images
            ],
            "image_energies_eV": [image.energy for image in transition_state.images],
        }
        return ControllerResult(payload=payload)

    # --- view contract ---

    def refresh_view(self) -> None:
        """Re-render the energy-barriers view from current state.

        Behavior-preserving move of ``MainWindow._refresh_energy_barriers_view``.
        The whole body moves (endpoint options *and* NEB sources *and* NEB
        results): this is the last migration, so there is no separate
        path-generation controller for the endpoint options to live in — they
        render into the same
        :class:`~mufinder.app.energy_barriers_widget.EnergyBarriersWidget` this
        controller now owns. The endpoint *handlers* stay on ``MainWindow`` and
        call the thin ``_refresh_energy_barriers_view`` delegate.

        No-ops when the view getter returns ``None`` (the view was never created),
        mirroring exactly how the former ``MainWindow._refresh_energy_barriers_view``
        handled the not-yet-created view — this is what lets result dispatch and
        materialize run safely before the workflow route is ever opened.
        """

        view = self._view_getter()
        if view is None:
            return
        document = self._context.document()
        if document is None:
            view.set_endpoint_options([])
            view.set_sources([])
            view.set_neb_results([])
            return

        endpoint_rows = self._deps.energy_barrier_endpoint_rows()
        reactant_key, product_key = self._deps.selected_energy_barrier_endpoint_keys()
        view.set_endpoint_options(
            endpoint_rows,
            reactant_key=reactant_key,
            product_key=product_key,
            max_path_length_text=self._deps.energy_barrier_max_path_length_text(),
        )

        # One unified table: each generated path carries its NEB status + configuration.
        source_rows: list[EnergyBarrierSourceRow] = []
        neb_states = self._deps.build_energy_barrier_neb_states()
        neb_definition = self._deps.active_energy_barrier_neb_definition()
        for candidate in self._deps.eligible_energy_barrier_source_candidates():
            source_rows.append(
                EnergyBarrierSourceRow(
                    key=candidate.key,
                    reactant=candidate.reactant.site.label
                    or candidate.reactant.job.label,
                    product=candidate.product.site.label or candidate.product.job.label,
                    charge_state=self._deps.charge_state_label_for_job(
                        candidate.reactant.job
                    ),
                    path_length=f"{candidate.path.length_ang:.4g} Å",
                    status=getattr(neb_states.get(candidate.key), "status", "ready")
                    or "ready",
                    farms=_farm_count_label(
                        self._deps.num_farms_from_extras(
                            self._deps.energy_barrier_neb_row_extras(
                                neb_definition, candidate.key
                            )
                        )
                    ),
                    configured=self._deps.workflow_row_configuration_exists(
                        neb_definition, candidate.key
                    ),
                )
            )
        view.set_sources(source_rows)

        neb_result_rows: list[EnergyBarrierNebResultRow] = []
        neb_sources_by_job_id = self._deps.energy_barrier_neb_sources_by_job_id()
        for candidate in self._deps.completed_energy_barrier_neb_candidates():
            result = candidate.result
            data = result.data if isinstance(result.data, dict) else {}
            forward, reverse, delta = self._deps.neb_barrier_summaries(data)
            source = neb_sources_by_job_id.get(candidate.job.id)
            path_length_ang = (
                self._deps.float_from_source(source, "path_length_ang")
                if source is not None
                else None
            )
            raw_charge = (
                str(source.get("charge_state", "")).strip() if source is not None else ""
            )
            neb_result_rows.append(
                EnergyBarrierNebResultRow(
                    key=candidate.key,
                    label=self._deps.energy_barrier_neb_result_path_label(
                        candidate, source
                    ),
                    charge_state=(
                        self._deps.muon_charge_state_label(raw_charge)
                        if raw_charge
                        else ""
                    ),
                    path_length="" if path_length_ang is None else f"{path_length_ang:.4g} Å",
                    barrier_forward=forward,
                    barrier_reverse=reverse,
                    delta_energy=delta,
                    images=str(data.get("num_images", "")),
                    status=candidate.job.status.value,
                    transition_state_file=str(data.get("transition_state_file", "")),
                )
            )
        view.set_neb_results(neb_result_rows)


def _farm_count_label(num_farms: int) -> str:
    """Render a configured farm count for the sources table."""

    count = max(1, int(num_farms))
    return "Off" if count == 1 else str(count)


__all__ = ["EnergyBarriersController", "EnergyBarriersDeps"]
