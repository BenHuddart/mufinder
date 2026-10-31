"""Energy-barriers (NEB) workflow controller.

Owns the nudged-elastic-band (NEB) diffusion-path workflow. Rows are the paths
generated between two relaxed muon sites; each runs as a single NEB band job
whose endpoints are those sites.

- **Paths.** :meth:`EnergyBarriersController.generate_paths` enumerates the
  distinct paths between the selected endpoints into the Energy Barriers form
  state; :meth:`EnergyBarriersController.remove_rows` drops paths from the form
  state and from every definition.
- **Source-key provenance.** Jobs are keyed by their *path source key*.
  :meth:`EnergyBarriersController.materialize` writes ``job_ids_by_label`` keyed by
  source key and stamps each expanded spec with ``provenance={"source_key": key}``.
- **IDPP intermediate.** :meth:`EnergyBarriersController.materialize` aligns the
  endpoints into the pristine host with ``prepare_transition_state_structures``
  and seeds the band's middle image with ``idpp_intermediate_structure`` before
  expanding through ``expand_nudged_elastic_band_jobs``.
- **``.ts`` result.** :meth:`EnergyBarriersController.validate_artifacts` fails a
  job whose ``.ts`` is missing or unparseable;
  :meth:`EnergyBarriersController.build_result` contributes
  ``transition_state_file``, ``barrier_height_eV``, ``num_images``,
  ``image_labels``, ``reaction_coordinates`` and ``image_energies_eV``. Both the
  NEB and the transition-state-search calc types are registered on
  :attr:`calc_types`, so :meth:`handles_job` covers the ``.ts``-producing pair.
- **Lazily created view.** The energy-barriers view is created only when its
  route is first opened, but the controller is built and registered at startup
  so a completed NEB job dispatches through it regardless. It therefore takes a
  ``view_getter: Callable[[], EnergyBarriersWidget | None]`` instead of a view,
  and :meth:`refresh_view` does nothing while the getter returns ``None``.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import UUID

from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.ts_parser import (
    TransitionStateImage,
    TransitionStateParseError,
    parse_transition_state_file,
)
from mufinder.castep_io.types import AtomicPosition, PositionBlock, ValueWithUnit
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Result, Structure
from mufinder.services.definition_mappings import (
    remove_generated_sources,
    remove_source_keys_from_definition,
)
from mufinder.services.job_artifacts import ParsedJobArtifacts
from mufinder.services.muon_sites_summary import (
    ENERGY_BARRIERS_SCOPE,
    DownstreamSiteScope,
    downstream_site_scope,
    set_show_every_relaxed_site,
)
from mufinder.services.project_service import ProjectDocument
from mufinder.services.source_candidates import (
    ENERGY_BARRIERS_FORM_STATE_KEY,
    EnergyBarrierNebCandidate,
    EnergyBarrierSourceCandidate,
    QuantumSourceCandidate,
    charge_state_for_job,
    charge_state_label_for_job,
    completed_energy_barrier_neb_candidates,
    eligible_energy_barrier_source_candidates,
    eligible_quantum_source_candidates,
    energy_barrier_form_state,
    energy_barrier_max_path_length_text,
    energy_barrier_neb_result_path_label,
    energy_barrier_neb_source_candidates,
    energy_barrier_neb_sources_by_job_id,
    energy_barrier_path_preview_structure,
    energy_barrier_path_symmetry_uses_magnetic_order,
    energy_barrier_source_label,
    energy_barrier_source_site_labels,
    enumerate_energy_barrier_source_candidates,
    float_from_source,
    fractional_triplet_from_source,
    keep_stored_path_keys,
    quantum_source_candidate_by_key,
    quantum_source_relative_energies,
    selected_energy_barrier_endpoint_keys,
    serialize_energy_barrier_neb_sources,
    serialize_energy_barrier_source_candidates,
    source_relaxation_parameter_values,
    uuid_from_source,
)
from mufinder.science.energy_barriers import (
    EnergyBarrierError,
    idpp_intermediate_structure,
    prepare_transition_state_structures,
)
from mufinder.science.site_generation import lattice_cart_to_frac
from mufinder.domain.charge_states import muon_charge_state_label
from mufinder.workflow import StructureBridgeError, structure_to_spec, write_job_directory
from mufinder.workflow.energy_barriers import (
    NUDGED_ELASTIC_BAND_CALC_TYPE,
    NUDGED_ELASTIC_BAND_SET_CALC_TYPE,
    TRANSITION_STATE_SEARCH_CALC_TYPE,
    TRANSITION_STATE_SEARCH_SET_CALC_TYPE,
    NudgedElasticBandJobConfig,
    NudgedElasticBandSource,
    expand_nudged_elastic_band_jobs,
    merge_serialized_sources,
    replace_serialized_sources_for_pair,
    strip_legacy_energy_barrier_params,
)
from mufinder.workflow.types import InputTemplate

from mufinder.app.adapter_common import (
    build_workflow_settings_template,
    project_convergence_parameter_values,
)
from mufinder.app.energy_barriers_widget import (
    EnergyBarrierEndpointRow,
    EnergyBarrierNebResultRow,
    EnergyBarrierPreviewImageRow,
    EnergyBarriersWidget,
    EnergyBarrierSourceRow,
)
from mufinder.app.relaxation_types import RelaxationRunState
from mufinder.app.workflow_config_codecs import (
    batch_row_payload,
    num_farms_from_extras,
    row_settings_payload,
    store_workflow_row_configuration,
    workflow_row_configuration,
    workflow_row_configuration_exists,
)
from mufinder.app.workflow_controllers.base import (
    ControllerResult,
    WorkflowControllerContext,
    WorkflowJobController,
    confirm_active_host,
    dedupe_jobs,
)
from mufinder.app.workflow_dialogs import SourceConfigureRequest, SourceConfigureResult
from mufinder.app.workflow_parameters import (
    build_workflow_parameter_surface,
    merge_species_options,
    workflow_basic_parameter_names,
)
from mufinder.app.workflow_submission import effective_row_status, workflow_row_state_is_ready
from mufinder.app.workflow_table_utils import format_energy_ev

# The NEB dialog's default rows after the shared ones: the two NEB settings that
# are routinely changed. The dialog adds the task-farm row below them. The other
# NEB keywords stay behind Add Parameter, and climbing image is locked on (see the
# dialog's locked_values): a barrier calculation needs the top image driven onto
# the saddle, so it is not user-editable here.
_KEYWORD_NAMES = workflow_basic_parameter_names(
    "tssearch_max_path_points",
    "tssearch_force_tol",
)
# A value stored for one of these stays in the dialog as an added row, including
# one the dialog would not keep by itself (Normed off) or that differs between
# the selected paths.
_ADD_PARAMETER_NEB_KEYWORDS = (
    "tssearch_neb_spring_constant",
    "tssearch_neb_tangent_mode",
    "tssearch_neb_method",
    "tssearch_neb_max_iter",
    "tssearch_neb_normed",
)
# Default NEB image count (TSSEARCH_MAX_PATH_POINTS, intermediate images only).
# Odd, as climbing-image NEB requires. 11 gives a finer band than the CASTEP minimum: a smoother barrier
# path and a lower residual force at the climbing image (7 images left a
# ~0.08 eV/Ang plateau just above the force tol).
_DEFAULT_IMAGES = 11
# Editable NEB force-convergence tolerance (TSSEARCH_FORCE_TOL). The default
# mirrors CASTEP's own (0.05 eV/Ang via GEOM_FORCE_TOL); it is exposed so a
# stubborn band whose barrier has converged but whose max force component
# plateaus above 0.05 can be loosened. The keyword's canonical unit is H/BOHR;
# it is surfaced in eV/Ang, the unit the .castep output reports.
_FORCE_TOL_UNIT = "eV/ang"
_DEFAULT_FORCE_TOL = ValueWithUnit(0.05, _FORCE_TOL_UNIT)
_PATH_SYMMETRY_NOTE_CACHE_SIZE = 64
PATH_SYMMETRY_CRYSTAL_NOTE = "Paths use the crystal symmetry."
PATH_SYMMETRY_MAGNETIC_NOTE = (
    "Paths use the magnetic order's symmetry, which is lower than the crystal symmetry."
)


@dataclass(frozen=True)
class EnergyBarriersDeps:
    """The window-side handles the energy-barriers controller needs.

    Each one shows a dialog: the host-mismatch handlers ask whether to switch
    the active host (shared with every host-guarded workflow) and
    ``show_warning`` reports a rejected input. Everything else the controller
    reads from the shared
    :class:`~mufinder.app.workflow_controllers.base.WorkflowControllerContext`
    or the service layer.
    """

    handle_structure_host_mismatch: Callable[[object], bool]
    handle_site_host_mismatch: Callable[[object], bool]
    show_warning: Callable[[str, str], None]


@dataclass(frozen=True)
class PathPreview:
    """What the path viewer shows for one generated path."""

    structure: Structure
    start_fractional: tuple[float, float, float]
    end_fractional: tuple[float, float, float]
    title: str


class EnergyBarriersController(WorkflowJobController):
    """Controller for the energy-barriers (NEB) diffusion-path workflow."""

    calc_types = frozenset({NUDGED_ELASTIC_BAND_CALC_TYPE, TRANSITION_STATE_SEARCH_CALC_TYPE})
    stage = WorkflowStage.ENERGY_BARRIERS
    definition_calc_type = NUDGED_ELASTIC_BAND_SET_CALC_TYPE
    no_ready_message = "Only NEB paths that have not yet been submitted can be submitted."

    def __init__(
        self,
        context: WorkflowControllerContext,
        deps: EnergyBarriersDeps,
        view_getter: Callable[[], EnergyBarriersWidget | None],
    ) -> None:
        super().__init__(context)
        self._deps = deps
        self._view_getter = view_getter
        self._path_symmetry_notes: dict[str, str] = {}

    # --- paths ---

    def candidates(self) -> list[EnergyBarrierSourceCandidate]:
        """Return the generated paths whose endpoint records still exist."""

        return eligible_energy_barrier_source_candidates(self._context.document())

    def neb_candidates(self) -> list[EnergyBarrierNebCandidate]:
        """Return every generated path as an NEB source, sorted by key."""

        return energy_barrier_neb_source_candidates(self._context.document())

    def path_definitions(self) -> list[CalculationDefinition]:
        """Return every definition that can hold generated paths.

        The transition-state-search sets an older configuration stored paths
        in, in project order, then this workflow's NEB definitions.
        """

        document = self._context.document()
        if document is None:
            return []
        legacy = [
            definition
            for definition in document.calculations.values()
            if definition.workflow_stage == WorkflowStage.ENERGY_BARRIERS
            and definition.calc_type == TRANSITION_STATE_SEARCH_SET_CALC_TYPE
        ]
        return legacy + self.definitions()

    def site_scope(self) -> DownstreamSiteScope:
        """Return which relaxed sites the endpoint pickers list."""

        return downstream_site_scope(self._context.document(), ENERGY_BARRIERS_SCOPE)

    def endpoint_rows(
        self, scope: DownstreamSiteScope | None = None
    ) -> list[EnergyBarrierEndpointRow]:
        """Return the relaxed sites offered as initial and final endpoints.

        By default these are the sites pinned to the Muon Sites summary, labelled
        by Wyckoff label and cluster in energy-rank order; with "Show every
        relaxed site" they are every eligible relaxed site.
        """

        candidates = eligible_quantum_source_candidates(self._context.document())
        if scope is None:
            scope = downstream_site_scope(
                self._context.document(), ENERGY_BARRIERS_SCOPE, candidates
            )
        relative = quantum_source_relative_energies(candidates)
        return [
            EnergyBarrierEndpointRow(
                key=candidate.key,
                label=_endpoint_label(scope, candidate),
                charge_state=charge_state_label_for_job(candidate.job),
                source_job=candidate.job.label,
                relative_energy=_format_endpoint_relative_energy(relative.get(candidate.key)),
            )
            for candidate in scope.scoped_candidates(candidates)
        ]

    def set_show_every_site(self, enabled: bool) -> None:
        """Offer every eligible relaxed site as an endpoint (``True``) or only the pinned ones."""

        document = self._context.document()
        if document is None:
            return
        set_show_every_relaxed_site(document, ENERGY_BARRIERS_SCOPE, enabled)
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="The Energy Barriers endpoint list was updated, but autosave failed."
        )

    def select_endpoints(self, reactant_key: str, product_key: str) -> None:
        """Remember the endpoint pair the form selects."""

        document = self._context.document()
        if document is None:
            return
        state = energy_barrier_form_state(document)
        state["selected_reactant_key"] = reactant_key
        state["selected_product_key"] = product_key
        document.ui_state[ENERGY_BARRIERS_FORM_STATE_KEY] = state
        self.refresh_view()

    def generate_paths(
        self, reactant_key: str, product_key: str, max_path_length_text: str = ""
    ) -> None:
        """Generate the distinct paths between two relaxed sites into the form state.

        The endpoints must both be chosen and share a charge state, and a
        maximum path length, when given, must be positive. The new paths replace
        those previously generated for the same endpoint pair, except that a path
        generated again keeps its key and a stored path with an NEB job or saved
        settings stays (:func:`keep_stored_path_keys`).
        """

        document = self._context.document()
        if document is None:
            return
        if not reactant_key or not product_key:
            self._context.show_info(
                "Select endpoints",
                "Choose initial and final relaxed structures before generating paths.",
            )
            return
        reactant = quantum_source_candidate_by_key(document, reactant_key)
        product = quantum_source_candidate_by_key(document, product_key)
        if reactant is None or product is None:
            return
        if charge_state_for_job(reactant.job) != charge_state_for_job(product.job):
            self._context.show_info(
                "Charge-state mismatch",
                "Initial and final structures must use the same muon charge state.",
            )
            return
        try:
            max_distance_ang = _parse_max_path_length(max_path_length_text)
        except ValueError as exc:
            self._deps.show_warning("Invalid path length", str(exc))
            return
        candidates = enumerate_energy_barrier_source_candidates(
            document,
            reactant_key,
            product_key,
            max_distance_ang=max_distance_ang,
        )
        state = energy_barrier_form_state(document)
        state["selected_reactant_key"] = reactant_key
        state["selected_product_key"] = product_key
        if max_distance_ang is None:
            state.pop("max_path_length_ang", None)
        else:
            state["max_path_length_ang"] = max_distance_ang
        stored = state.get("generated_sources", [])
        # A stored path with a job or saved settings stays even when this pass no
        # longer generates it: its job and results are durable.
        definitions = self.path_definitions()
        stored_keys = {
            str(source.get("key", "")).strip()
            for source in (stored if isinstance(stored, list) else ())
            if isinstance(source, dict)
        }
        durable_keys = {
            key
            for key in stored_keys
            if key
            and any(
                self.job_for_key(definition, key) is not None
                or workflow_row_configuration_exists(definition, key)
                for definition in definitions
            )
        }
        candidates = keep_stored_path_keys(document, candidates, stored, durable_keys=durable_keys)
        state["generated_sources"] = replace_serialized_sources_for_pair(
            stored,
            reactant_site_id=str(reactant.site.id),
            product_site_id=str(product.site.id),
            new_sources=serialize_energy_barrier_source_candidates(candidates),
            charge_state=charge_state_for_job(reactant.job),
            keep_keys=durable_keys,
        )
        document.ui_state[ENERGY_BARRIERS_FORM_STATE_KEY] = state
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="Energy Barriers paths were generated, but autosave failed."
        )

    def path_symmetry_note(self, reactant_key: str) -> str:
        """Return which symmetry groups the paths from ``reactant_key``, for the page.

        Empty when no eligible initial endpoint is chosen or its supercell cannot
        be matched to the host.
        """

        document = self._context.document()
        host_id = document.project.host_structure_id if document is not None else None
        host = document.structures.get(host_id) if document is not None and host_id else None
        reactant = quantum_source_candidate_by_key(document, reactant_key) if reactant_key else None
        if host is None or reactant is None:
            return ""
        # The note rebuilds the endpoint supercell, so it is computed once per host
        # and endpoint structure rather than on every refresh.
        cache_key = repr(
            (
                id(document),
                reactant.structure.id,
                host.id,
                host.lattice_vectors,
                host.species,
                host.fractional_positions,
                *(
                    host.properties.get(name)
                    for name in ("atomic_spins", "magnetic_structure", "propagation_vector")
                ),
            )
        )
        note = self._path_symmetry_notes.get(cache_key)
        if note is None:
            uses_magnetic_order = energy_barrier_path_symmetry_uses_magnetic_order(
                document, reactant_key
            )
            if uses_magnetic_order is None:
                note = ""
            elif uses_magnetic_order:
                note = PATH_SYMMETRY_MAGNETIC_NOTE
            else:
                note = PATH_SYMMETRY_CRYSTAL_NOTE
            if len(self._path_symmetry_notes) >= _PATH_SYMMETRY_NOTE_CACHE_SIZE:
                self._path_symmetry_notes.clear()
            self._path_symmetry_notes[cache_key] = note
        return note

    def path_preview(self, key: str) -> PathPreview | None:
        """Return what the path viewer shows for path ``key``.

        ``None`` for an unknown path. Raises ``ValueError`` with a user-facing
        reason when the pristine supercell cannot be built.
        """

        candidate = next(
            (candidate for candidate in self.candidates() if candidate.key == key), None
        )
        if candidate is None:
            return None
        return PathPreview(
            structure=energy_barrier_path_preview_structure(self._context.document(), candidate),
            start_fractional=candidate.path.start_fractional,
            end_fractional=candidate.path.end_fractional,
            title=f"{energy_barrier_source_label(candidate)} path preview",
        )

    def show_neb_preview(self, key: str) -> None:
        """Show the band of completed NEB result ``key`` in the view's preview panel.

        Each image's energy is reported relative to the initial image: absolute
        CASTEP energies are ~10⁵ eV, so differences vanish in the display.
        """

        view = self._view_getter()
        if view is None:
            return
        candidate = next(
            (
                candidate
                for candidate in completed_energy_barrier_neb_candidates(self._context.document())
                if candidate.key == key
            ),
            None,
        )
        if candidate is None:
            view.clear_neb_preview()
            return
        ts_path = _transition_state_file_path(candidate.result)
        if ts_path is None or not ts_path.exists():
            view.clear_neb_preview(title=f"{candidate.job.label} preview unavailable")
            return
        try:
            transition_state = parse_transition_state_file(ts_path)
        except TransitionStateParseError:
            view.clear_neb_preview(title=f"{candidate.job.label} preview unavailable")
            return
        initial_energy = transition_state.images[0].energy
        view.set_neb_preview(
            title=f"{candidate.job.label} ({len(transition_state.images)} images)",
            image_rows=[
                EnergyBarrierPreviewImageRow(
                    label=image.label,
                    reaction_coordinate=f"{image.reaction_coordinate:.6g}",
                    energy=f"{image.energy - initial_energy:+.4g} eV",
                    reaction_coordinate_value=image.reaction_coordinate,
                    energy_value=image.energy,
                )
                for image in transition_state.images
            ],
            structures=[
                self._image_structure(candidate.job.label, image)
                for image in transition_state.images
            ],
        )

    def _image_structure(self, job_label: str, image: TransitionStateImage) -> Structure:
        # The parser already converts the ``.ts`` geometry from Bohr to Å, so the
        # lattice and positions feed ``Structure`` (Å throughout) unchanged.
        lattice = [list(vector) for vector in image.lattice_vectors]
        document = self._context.document()
        return Structure(
            label=f"{job_label} {image.label}",
            origin="energy_barriers_neb_preview",
            lattice_vectors=lattice,
            species=list(image.species),
            fractional_positions=[
                list(lattice_cart_to_frac(position, lattice))
                for position in image.cartesian_positions
            ],
            parent_structure_id=document.project.host_structure_id
            if document is not None
            else None,
            properties={
                "pbc": [True, True, True],
                "source_image_label": image.label,
                "reaction_coordinate": image.reaction_coordinate,
                "energy_eV": image.energy,
            },
        )

    # --- definitions and rows ---

    def definition_and_key_for_job(
        self, job: Job
    ) -> tuple[CalculationDefinition, str] | tuple[None, None]:
        """Return the NEB definition and path key whose job mapping binds ``job``."""

        for definition in self.definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for key, value in mapping.items():
                if str(value) == str(job.id):
                    return definition, str(key)
        return None, None

    def ensure_definition(self, source_keys: Sequence[str]) -> CalculationDefinition | None:
        """Return the active NEB definition with the selected paths added, creating one if needed.

        Returns ``None`` when there is no document or none of ``source_keys``
        names a generated path.
        """

        document = self._context.document()
        if document is None:
            return None
        neb_candidates = {candidate.key: candidate for candidate in self.neb_candidates()}
        candidates = [neb_candidates[key] for key in source_keys if key in neb_candidates]
        if not candidates:
            return None
        structure_ids = _source_ids(candidates, ("reactant_structure_id", "product_structure_id"))
        site_ids = _source_ids(candidates, ("reactant_site_id", "product_site_id"))
        definition = self.active_definition()
        if definition is not None:
            for structure_id in structure_ids:
                if structure_id not in definition.input_structure_ids:
                    definition.input_structure_ids.append(structure_id)
            for site_id in site_ids:
                if site_id not in definition.input_site_ids:
                    definition.input_site_ids.append(site_id)
            config = definition.parameters.setdefault("configuration", {})
            if isinstance(config, dict):
                config.setdefault("method", "NEB")
                config["sources"] = merge_serialized_sources(
                    config.get("sources", []),
                    serialize_energy_barrier_neb_sources(document, source_keys),
                    key_field="key",
                )
            return definition

        return self._context.project_service.register_nudged_elastic_band_definition(
            document,
            input_structure_ids=structure_ids,
            input_site_ids=site_ids,
            parameters={
                "method": "NEB",
                "sources": serialize_energy_barrier_neb_sources(document, source_keys),
            },
            output_directory=document.project_root / "files" / "energy_barriers_neb",
        )

    def remove_labels_from_definition(
        self, definition: CalculationDefinition, labels: Collection[str]
    ) -> bool:
        document = self._context.document()
        if document is None:
            return False
        return remove_source_keys_from_definition(document, definition, labels)

    def remove_rows(self, labels: Iterable[str]) -> tuple[UUID, ...]:
        """Drop the paths ``labels`` name from the form state and every definition.

        Unlike the row-table workflows, paths live in the form state and in the
        legacy transition-state-search definitions as well as the NEB ones (see
        :meth:`path_definitions`). Job-file cleanup happens before this, in the
        window.
        """

        document = self._context.document()
        if document is None:
            return ()
        keys = set(labels)
        remove_generated_sources(document, keys)
        return tuple(
            definition.id
            for definition in self.path_definitions()
            if self.remove_labels_from_definition(definition, keys)
        )

    # --- configuration ---

    @staticmethod
    def parameter_specs() -> list[KeywordSpec]:
        """Return the NEB Configure dialog's parameter surface.

        ``TSSEARCH_FORCE_TOL`` is shown in eV/Ang (its catalog unit is H/BOHR)
        with CASTEP's own default pre-filled, so an untouched row reproduces
        CASTEP's behaviour. The physical editor forces a value's unit to the
        spec's, so the unit and the default change together.
        """

        specs = build_workflow_parameter_surface(
            workflow_stage="energy_barriers",
            basic_names=_KEYWORD_NAMES,
        )
        return [
            replace(spec, unit=_FORCE_TOL_UNIT, default_value=_DEFAULT_FORCE_TOL)
            if spec.name == "tssearch_force_tol"
            else spec
            for spec in specs
        ]

    def _project_parameter_values(
        self, *, target_lattice: list[list[float]] | None = None
    ) -> dict[str, object]:
        document = self._context.document()
        if document is None:
            return {}
        return project_convergence_parameter_values(document.project, target_lattice=target_lattice)

    def _project_input_defaults(self) -> InputTemplate | None:
        values = self._project_parameter_values()
        if not values:
            return None
        return build_workflow_settings_template(values)

    def row_configuration(
        self,
        definition: CalculationDefinition | None,
        key: str,
        *,
        target_lattice: list[list[float]] | None = None,
        source_candidate: EnergyBarrierSourceCandidate | None = None,
        source_defaults_cache: dict[UUID, dict[str, object]] | None = None,
    ) -> tuple[dict[str, object], tuple[str, ...], dict[str, object]]:
        """Return a path's effective ``(parameter values, advanced names, extras)``.

        Precedence: project defaults < the initial endpoint's relaxation settings
        < the row's stored configuration. Inheriting the endpoint's k-point grid
        and cut-off keeps the band on the same numerical footing as the relaxed
        structures it connects; when the relaxation's input files are missing,
        the project defaults stand in. The image count
        (``TSSEARCH_MAX_PATH_POINTS``) counts the intermediate images only, so
        the finished band holds that many plus the two endpoints. It defaults
        to an odd value, and an even count is rounded up: the workflow locks
        ``TSSEARCH_NEB_CLIMBING`` on, and with climbing on CASTEP itself adds
        one image to an even count (with a warning in the ``.castep``), so the
        stored count is the one CASTEP runs. With climbing off CASTEP keeps an
        even count as it is.

        A caller looping over many paths may pass the resolved
        ``source_candidate`` and a shared ``source_defaults_cache`` to avoid
        rebuilding the path list and re-reading an endpoint's files per path.
        """

        params, advanced, extras = workflow_row_configuration(definition, key)
        params = strip_legacy_energy_barrier_params(params)
        advanced = (
            *advanced,
            *(name for name in _ADD_PARAMETER_NEB_KEYWORDS if name in params),
        )
        values = self._project_parameter_values(target_lattice=target_lattice)
        candidate = (
            source_candidate
            if source_candidate is not None
            else next((candidate for candidate in self.candidates() if candidate.key == key), None)
        )
        if candidate is not None:
            reactant_job_id = candidate.reactant.job.id
            if source_defaults_cache is not None and reactant_job_id in source_defaults_cache:
                reactant_defaults = source_defaults_cache[reactant_job_id]
            else:
                reactant_defaults = source_relaxation_parameter_values(
                    candidate.reactant.job, self._context.job_output_directory
                )
                if source_defaults_cache is not None:
                    source_defaults_cache[reactant_job_id] = reactant_defaults
            values.update(reactant_defaults)
        values.setdefault("tssearch_max_path_points", _DEFAULT_IMAGES)
        values.update(params)
        try:
            image_count = int(values.get("tssearch_max_path_points"))
        except (TypeError, ValueError):
            image_count = None
        if image_count is not None and image_count % 2 == 0:
            values["tssearch_max_path_points"] = image_count + 1
        return values, advanced, extras

    def configure_request(self, keys: Sequence[str]) -> SourceConfigureRequest | None:
        """Return what the NEB Configure dialog opens with for the paths ``keys`` name.

        The dialog opens read-only when any selected path is not ready. It shows
        a farm count only when every selected path agrees, else 1; accepting it
        writes the shown value to every path.
        """

        document = self._context.document()
        if document is None:
            return None
        neb_candidates = {candidate.key: candidate for candidate in self.neb_candidates()}
        candidates = [neb_candidates[key] for key in keys if key in neb_candidates]
        if not candidates:
            return None
        definition = self.active_definition()
        source_candidates = {candidate.key: candidate for candidate in self.candidates()}
        source_defaults_cache: dict[UUID, dict[str, object]] = {}
        values_by_key: list[dict[str, object]] = []
        advanced_names: set[str] = set()
        reactant_structures: list[Structure | None] = []
        num_farms_values: list[int] = []
        for candidate in candidates:
            reactant_structure = document.structures.get(
                uuid_from_source(candidate.source, "reactant_structure_id")
            )
            reactant_structures.append(reactant_structure)
            values, row_advanced_names, extras = self.row_configuration(
                definition,
                candidate.key,
                target_lattice=reactant_structure.lattice_vectors
                if reactant_structure is not None
                else None,
                source_candidate=source_candidates.get(candidate.key),
                source_defaults_cache=source_defaults_cache,
            )
            values_by_key.append(values)
            advanced_names.update(row_advanced_names)
            num_farms_values.append(num_farms_from_extras(extras))
        states = self.build_states()
        read_only = any(
            not workflow_row_state_is_ready(states, candidate.key) for candidate in candidates
        )
        first = candidates[0]
        return SourceConfigureRequest(
            keys=tuple(candidate.key for candidate in candidates),
            row_values=tuple(values_by_key),
            advanced_names=frozenset(advanced_names),
            title="Inspect NEB Parameters" if read_only else "Configure NEB Parameters",
            intro_text=(
                _hop_heading(document, first, source_candidates.get(first.key))
                if len(candidates) == 1
                else f"Selected paths: {len(candidates)}"
            ),
            read_only=read_only,
            num_farms=(
                num_farms_values[0]
                if all(value == num_farms_values[0] for value in num_farms_values)
                else 1
            ),
            species_options=merge_species_options(reactant_structures),
        )

    def apply_configuration(
        self,
        keys: Sequence[str],
        result: SourceConfigureResult,
        extras: Mapping[str, object],
    ) -> None:
        """Store an accepted NEB Configure dialog on every path ``keys`` name.

        ``extras`` holds the dialog's farm count. A mixed parameter the user
        left alone keeps each path's own stored value.
        """

        payload = row_settings_payload(result.values, result.advanced_names, extras)
        definition = self.ensure_definition(keys)
        if definition is None:
            return
        for key in keys:
            store_workflow_row_configuration(
                definition, key, batch_row_payload(definition, key, payload, result.untouched_mixed)
            )
        self.refresh_view()
        self._context.persist_ui_state(
            failure_message="Energy Barriers NEB configuration was updated, but autosave failed."
        )

    # --- submission contract ---
    #
    # normalize=str and the base-default ``is_ready`` (workflow_row_state_is_ready)
    # apply. There is no submission_groups override: NEB jobs submit as a single
    # group.

    def normalize(self, value: object) -> str:
        return str(value)

    def pre_submit_check(self, keys: list[object]) -> bool:
        """Validate the selected paths' endpoint structures and sites against the host.

        NEB can be submitted before any NEB definition exists, so this checks
        the endpoints of the selected paths rather than a definition. A
        mismatch the user does not resolve by switching hosts refuses silently.
        """

        document = self._context.document()
        if document is None:
            return True
        candidates_by_key = {candidate.key: candidate for candidate in self.neb_candidates()}
        selected = [
            candidate
            for key in keys
            if (candidate := candidates_by_key.get(str(key).strip())) is not None
        ]
        return confirm_active_host(
            self._context.project_service,
            document,
            structure_ids=_source_ids(selected, ("reactant_structure_id", "product_structure_id")),
            site_ids=_source_ids(selected, ("reactant_site_id", "product_site_id")),
            handle_structure_mismatch=self._deps.handle_structure_host_mismatch,
            handle_site_mismatch=self._deps.handle_site_host_mismatch,
        )

    def build_states(self) -> dict[str, RelaxationRunState]:
        """Return each generated path's NEB job state; a path with no job is ready."""

        states: dict[str, RelaxationRunState] = {}
        definition = self.active_definition()
        for candidate in self.neb_candidates():
            job = self.job_for_neb_source_key(definition, candidate.key)
            if job is None:
                states[candidate.key] = RelaxationRunState(status="ready")
                continue
            states[candidate.key] = RelaxationRunState(
                status=effective_row_status(job),
                job_id=job.id,
                job_label=job.label,
                error_message=job.error_message,
            )
        return states

    def job_for_neb_source_key(
        self, definition: CalculationDefinition | None, key: str
    ) -> Job | None:
        """Return the NEB job ``definition`` binds to the path ``key``, if any."""

        return self.job_for_key(definition, key, calc_type=NUDGED_ELASTIC_BAND_CALC_TYPE)

    def jobs_for_removed_sources(
        self, keys: Collection[str]
    ) -> list[tuple[CalculationDefinition, Job]]:
        """Return ``(definition, job)`` for each job a removed source ``keys`` name is bound to.

        Walks every one of :meth:`path_definitions`, so paths an older
        configuration stored are cleaned up too; each job appears once, with
        the first definition that binds it.
        """

        document = self._context.document()
        if document is None:
            return []
        targets: list[tuple[CalculationDefinition, Job]] = []
        seen: set[UUID] = set()
        for definition in self.path_definitions():
            mapping = definition.parameters.get("job_ids_by_label", {})
            if not isinstance(mapping, dict):
                continue
            for key in keys:
                raw_job_id = mapping.get(key)
                if not isinstance(raw_job_id, str):
                    continue
                try:
                    job_id = UUID(raw_job_id)
                except ValueError:
                    continue
                job = document.jobs.get(job_id)
                if job is None or job.id in seen:
                    continue
                targets.append((definition, job))
                seen.add(job.id)
        return targets

    def jobs_for_keys(self, keys: list[object]) -> list[Job]:
        document = self._context.document()
        if document is None:
            return []
        selected: list[Job] = []
        seen: set[UUID] = set()
        str_keys = [str(key) for key in keys]
        for definition in self.definitions():
            for key in str_keys:
                job = self.job_for_neb_source_key(definition, key)
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
        definition = self.ensure_definition(str_keys)
        if definition is None:
            return []
        output_root = self._context.definition_output_directory(definition)
        host_id = document.project.host_structure_id
        host = document.structures.get(host_id) if host_id is not None else None
        if output_root is None or host is None:
            return []

        candidates_by_key = {candidate.key: candidate for candidate in self.neb_candidates()}
        source_candidates_by_key = {candidate.key: candidate for candidate in self.candidates()}
        source_defaults_cache: dict[UUID, dict[str, object]] = {}
        configs: list[NudgedElasticBandJobConfig] = []
        for key in str_keys:
            candidate = candidates_by_key.get(key)
            if candidate is None:
                continue
            reactant_id = uuid_from_source(candidate.source, "reactant_structure_id")
            product_id = uuid_from_source(candidate.source, "product_structure_id")
            reactant_site_id = uuid_from_source(candidate.source, "reactant_site_id")
            product_site_id = uuid_from_source(candidate.source, "product_site_id")
            reactant_run_id = uuid_from_source(candidate.source, "reactant_run_id")
            product_run_id = uuid_from_source(candidate.source, "product_run_id")
            reactant_job_id = uuid_from_source(candidate.source, "reactant_job_id")
            product_job_id = uuid_from_source(candidate.source, "product_job_id")
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
                    reactant_anchor_fractional=fractional_triplet_from_source(
                        candidate.source, "path_start_fractional"
                    ),
                    product_anchor_fractional=fractional_triplet_from_source(
                        candidate.source, "path_end_fractional"
                    ),
                )
                reactant_spec = structure_to_spec(prepared.reactant)
                product_block = _fractional_position_block(prepared.product)
                intermediate_structure = idpp_intermediate_structure(
                    prepared.reactant, prepared.product
                )
                intermediate_block = _fractional_position_block(intermediate_structure)
            except (EnergyBarrierError, StructureBridgeError, ValueError):
                continue
            values, _advanced_names, extras = self.row_configuration(
                definition,
                key,
                target_lattice=reactant_structure.lattice_vectors,
                source_candidate=source_candidates_by_key.get(key),
                source_defaults_cache=source_defaults_cache,
            )
            # ``num_farms`` is dropped alongside the workflow-locked ``task``: the
            # farm count comes from the row's extras (below), which is also what
            # decides the expected output filenames and the submission rank check.
            # A keyword-row copy would override the base template and silently
            # disagree with both. The configure dialog blocks the row, so this
            # only bites on configurations persisted by other paths.
            values = {
                name: value for name, value in values.items() if name not in {"task", "num_farms"}
            }
            try:
                overrides = build_workflow_settings_template(values)
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
                        path_length_ang=float_from_source(candidate.source, "path_length_ang"),
                        provenance={"source_key": key},
                    ),
                    num_farms=num_farms_from_extras(extras),
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
                global_defaults=self._project_input_defaults(),
            )
        except (ValueError, StructureBridgeError):
            return []

        mapping = definition.parameters.setdefault("job_ids_by_label", {})
        if not isinstance(mapping, dict):
            mapping = {}
            definition.parameters["job_ids_by_label"] = mapping

        spec_by_key = {str(spec.provenance.get("source_key", "")): spec for spec in specs}
        created: list[Job] = []
        for key in str_keys:
            spec = spec_by_key.get(key)
            existing = self.job_for_neb_source_key(definition, key)
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

        jobs = dedupe_jobs(created)
        if jobs:
            self.refresh_view()
            self._context.persist_ui_state(
                failure_message=("Energy Barriers NEB jobs were generated, but autosave failed.")
            )
        return jobs

    # --- result contract ---

    def validate_artifacts(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> str | None:
        """Fail an NEB or transition-state job whose ``.ts`` is missing or unparseable.

        Reads the carrier's ``ts_missing`` / ``ts_error``. The missing-file
        message names the file the collector looked for (``artifacts.ts_path``):
        under task farming that is ``<seed>_farm001.ts``, which CASTEP writes in
        place of ``<seed>.ts``.
        """

        if artifacts.ts_missing:
            expected_ts = (
                artifacts.ts_path.name
                if artifacts.ts_path is not None
                else f"{self._context.job_seed_name(job)}.ts"
            )
            return f"Expected CASTEP path output was not found: {expected_ts}"
        if artifacts.ts_error is not None:
            return f"Failed to parse CASTEP path output: {artifacts.ts_error}"
        return None

    def build_result(
        self,
        job: Job,
        parsed: CastepResult,
        run: CalculationRun,
        artifacts: ParsedJobArtifacts,
    ) -> ControllerResult:
        """Contribute the ``.ts`` transition-state payload for a completed NEB job.

        :meth:`validate_artifacts` has already verified the ``.ts`` file exists
        and parses (failing the job on a missing file or
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
        """Re-render the endpoint options, the path table and the NEB results.

        Does nothing while the lazily created view does not exist, which is what
        lets result dispatch and materialize run before the route is opened.
        """

        view = self._view_getter()
        if view is None:
            return
        document = self._context.document()
        if document is None:
            view.set_site_scope(show_every_site=False)
            view.set_endpoint_options([])
            view.set_path_symmetry_note("")
            view.set_sources([])
            view.set_neb_results([])
            return

        scope = self.site_scope()
        endpoint_rows = self.endpoint_rows(scope)
        reactant_key, product_key = selected_energy_barrier_endpoint_keys(
            document, [row.key for row in endpoint_rows]
        )
        # Keep the stored pair within the listed endpoints, so toggling "Show
        # every relaxed site" keeps the pair the pickers showed. An empty list
        # (nothing pinned) leaves the stored pair alone.
        state = energy_barrier_form_state(document)
        if endpoint_rows and (
            state.get("selected_reactant_key"),
            state.get("selected_product_key"),
        ) != (reactant_key, product_key):
            state["selected_reactant_key"] = reactant_key
            state["selected_product_key"] = product_key
            document.ui_state[ENERGY_BARRIERS_FORM_STATE_KEY] = state
        view.set_site_scope(show_every_site=scope.show_every_site, notice=scope.notice)
        view.set_endpoint_options(
            endpoint_rows,
            reactant_key=reactant_key,
            product_key=product_key,
            max_path_length_text=energy_barrier_max_path_length_text(document),
        )
        view.set_path_symmetry_note(self.path_symmetry_note(reactant_key))

        # One unified table: each generated path carries its NEB status + configuration.
        source_rows: list[EnergyBarrierSourceRow] = []
        neb_states = self.build_states()
        neb_definition = self.active_definition()
        for candidate in self.candidates():
            _params, _advanced, extras = workflow_row_configuration(neb_definition, candidate.key)
            source_rows.append(
                EnergyBarrierSourceRow(
                    key=candidate.key,
                    reactant=candidate.reactant.site.label or candidate.reactant.job.label,
                    product=candidate.product.site.label or candidate.product.job.label,
                    charge_state=charge_state_label_for_job(candidate.reactant.job),
                    path_length=f"{candidate.path.length_ang:.4g} Å",
                    status=getattr(neb_states.get(candidate.key), "status", "ready") or "ready",
                    farms=_farm_count_label(num_farms_from_extras(extras)),
                    configured=workflow_row_configuration_exists(neb_definition, candidate.key),
                )
            )
        view.set_sources(source_rows)

        neb_result_rows: list[EnergyBarrierNebResultRow] = []
        neb_sources_by_job_id = energy_barrier_neb_sources_by_job_id(document, self.definitions())
        for candidate in completed_energy_barrier_neb_candidates(document):
            result = candidate.result
            data = result.data if isinstance(result.data, dict) else {}
            forward, reverse, delta = _neb_barrier_summaries(data)
            source = neb_sources_by_job_id.get(candidate.job.id)
            path_length_ang = (
                float_from_source(source, "path_length_ang") if source is not None else None
            )
            raw_charge = str(source.get("charge_state", "")).strip() if source is not None else ""
            neb_result_rows.append(
                EnergyBarrierNebResultRow(
                    key=candidate.key,
                    label=energy_barrier_neb_result_path_label(document, candidate, source),
                    charge_state=muon_charge_state_label(raw_charge) if raw_charge else "",
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


def _endpoint_label(scope: DownstreamSiteScope, candidate: QuantumSourceCandidate) -> str:
    """Return the name the endpoint pickers give a relaxed site."""

    return (
        scope.pinned_label(candidate.key)
        or candidate.site.label
        or candidate.job.label
        or candidate.structure.label
    )


def _hop_heading(
    document: ProjectDocument,
    candidate: EnergyBarrierNebCandidate,
    source_candidate: EnergyBarrierSourceCandidate | None,
) -> str:
    """Return the hop as the endpoint pickers name it, with its length.

    A path whose endpoints are no longer eligible is named by its stored sites.
    """

    if source_candidate is not None:
        scope = downstream_site_scope(document, ENERGY_BARRIERS_SCOPE)
        names = (
            _endpoint_label(scope, source_candidate.reactant),
            _endpoint_label(scope, source_candidate.product),
        )
    else:
        names = energy_barrier_source_site_labels(document, candidate.source)
    hop = f"{names[0]} → {names[1]}" if names is not None else "Selected path"
    length_ang = float_from_source(candidate.source, "path_length_ang")
    return hop if length_ang is None else f"{hop} · {length_ang:.2f} Å"


def _source_ids(
    candidates: Sequence[EnergyBarrierNebCandidate], fields: Sequence[str]
) -> list[UUID]:
    """Return the ids ``fields`` hold across ``candidates``' sources, once each, in order."""

    return list(
        dict.fromkeys(
            value
            for candidate in candidates
            for field in fields
            if (value := uuid_from_source(candidate.source, field)) is not None
        )
    )


def _fractional_position_block(structure: Structure) -> PositionBlock:
    """Return a structure's positions as a fractional block.

    Fractional coordinates make the NEB product and intermediate blocks
    ``POSITIONS_FRAC_PRODUCT``, whose CASTEP reader accepts tagged species such
    as the muon's ``H:mu``.
    """

    rows = [
        AtomicPosition(str(species), tuple(float(value) for value in fractional))
        for species, fractional in zip(
            structure.species, structure.fractional_positions, strict=True
        )
    ]
    return PositionBlock(positions=tuple(rows), unit="frac")


def _parse_max_path_length(text: str) -> float | None:
    """Return the typed maximum path length; blank means none. Raise ``ValueError`` when not positive."""

    normalized = str(text).strip()
    if not normalized:
        return None
    value = float(normalized)
    if value <= 0.0:
        raise ValueError("Maximum path length must be greater than zero.")
    return value


def _transition_state_file_path(result: Result) -> Path | None:
    if not isinstance(result.data, dict):
        return None
    raw = result.data.get("transition_state_file")
    if not isinstance(raw, str) or not raw.strip():
        return None
    return Path(raw)


def _format_endpoint_relative_energy(value: float | None) -> str:
    """Format a site energy relative to its comparable minimum, as in the Muon Sites summary."""

    return "" if value is None else f"dE {float(value):.5f} eV"


def _neb_barrier_summaries(data: Mapping[str, object]) -> tuple[str, str, str]:
    """Format (forward barrier, reverse barrier, endpoint ΔE) from a NEB result payload.

    Both barriers and ΔE derive from the persisted per-image energies; a result
    without ``image_energies_eV`` falls back to the stored forward barrier with
    blank reverse and ΔE columns.
    """

    energies_payload = data.get("image_energies_eV")
    energies: list[float] = []
    if isinstance(energies_payload, (list, tuple)):
        try:
            energies = [float(value) for value in energies_payload]
        except (TypeError, ValueError):
            energies = []
    if len(energies) >= 2:
        peak = max(energies)
        return (
            format_energy_ev(peak - energies[0]),
            format_energy_ev(peak - energies[-1]),
            format_energy_ev(energies[-1] - energies[0]),
        )
    return format_energy_ev(data.get("barrier_height_eV")), "", ""


def _farm_count_label(num_farms: int) -> str:
    """Render a configured farm count for the sources table."""

    count = max(1, int(num_farms))
    return "Off" if count == 1 else str(count)


__all__ = ["EnergyBarriersController", "EnergyBarriersDeps", "PathPreview"]
