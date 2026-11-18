"""Magnetic Fields results stored as project ``Result`` records.

The Magnetic Fields page and each saved magnetic-field workspace keep their
latest dipolar-field, dipolar-tensor and nuclear-relaxation results as
``Result`` records in ``ProjectDocument.results``. The calculations run inside
MuFinder, so the records have no ``calculation_run_id``; ``data["owner"]``
names the page that holds them: :data:`MAGNETIC_FIELD_PAGE_OWNER` for the
Magnetic Fields page, or the workspace id for a saved workspace. Each owner has
at most one record of each result type.

The page's form state, kept in ``ui_state``, carries the same results inline
under ``magnetic_field_results``, ``dipolar_tensor_results`` and
``nuclear_relaxation_results`` while the page is open.
:func:`store_magnetic_field_results` moves them into records and returns the
form state without them, for ``ui_state``; :func:`attach_magnetic_field_results`
puts them back before the form is restored.

A dipolar-field record carries the source of the muon site it was computed at
in ``data["site_source"]``. :data:`UEP_SITE_SOURCE` marks an unrelaxed UEP
site, so the result is an estimate (:func:`is_unrelaxed_site_source`). A record
saved without it takes the source of the form's ``selected_muon_site``, because
choosing a different site clears the field results.

Dipolar-tensor and nuclear-relaxation records carry ``data["site_source"]`` the
same way. In the form state the source sits on each tensor row and on the
nuclear-relaxation result, as ``site_source``. These results survive a change
of muon site, so a record saved without a source has none: nothing is
inferred, and it is not shown as an estimate.

Dipolar-field, dipolar-tensor and nuclear-relaxation records carry the
calculation-method version of their numbers in ``data["calculation_version"]``,
written only when they are calculated (:data:`CALCULATION_METHOD_VERSION`). In
the form state it sits where ``site_source`` does. A result with an older
version, or none, keeps its numbers and shows :data:`OLDER_METHOD_RESULTS_NOTE`
(:func:`uses_older_method`); moving a result to a record never updates its
version.

A dipolar-field result calculated with a plate from a photo also records
:data:`PLATE_METHOD_VERSION` in ``data["plate_method_version"]``. Such a change
marks only the results it touches: a record without the key was calculated
without a plate or before the key existed, so
:func:`field_result_uses_older_method` decides from the sample in the owner's
saved form (:func:`sample_predates_plate_method`).

A commensurate result of an order with ``q != 0``, and every distribution, also records
:data:`TRANSLATION_METHOD_VERSION` in ``data["translation_method_version"]``.
A commensurate record without the key decides from its own rows: one that gave only ``T = 0``
is older (:func:`lists_only_one_translation`), while a ``q = 0`` result and the translated rows
of an approximant had nothing to change. A distribution without the key decides from the
magnetic structure its owner keeps, which calculating with a new one replaces: one with a
``q != 0`` term of period up to its recorded ``commensurate_max_period`` (64 when it records
none) held that term at ``T = 0`` and is older (:func:`distribution_held_a_short_period_at_zero`).

Reciprocal-equivalent long-period components share a signed phase, including q and 1 - q.
Version 3 records that correction. A version 2 distribution is older only when its owner's
model contains such a pair; unaffected version 2 fields keep their status. A keyless
aliased distribution records effective version 2 when copied to a simulation source.

A muon-polarisation record (:data:`MUON_POLARISATION_RESULT_TYPE`) holds a
:class:`~mufinder.science.polarisation.PolarisationResult` in
``data["result"]``, one record per owner. It is stored and read with
:func:`store_polarisation_result` and :func:`polarisation_result` and is not
part of the form state. A polarisation simulation
(:mod:`mufinder.services.polarisation_simulations`) is such a record whose
owner is the simulation, with its settings and sources beside the result.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from typing import Any

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import ProvenanceRecord, Result
from mufinder.persistence.codec import to_jsonable
from mufinder.science.demagnetization import FOURIER_PADDING
from mufinder.science.magnetic_fields import (
    MagneticFieldError,
    MAX_LISTED_TRANSLATION_PERIOD,
    has_short_period_modulation,
    has_reciprocal_phase_aliases,
)
from mufinder.science.magnetic_structure import MagneticStructureError, MagneticStructureModel
from mufinder.science.polarisation import PolarisationError, PolarisationResult
from mufinder.services.project_document import ProjectDocument
from mufinder.services.study_findings import (
    FINGERPRINT_KEY,
    SUBJECT_KEY,
    input_fingerprint,
    record_finding,
)


#: Result type of a dipolar-field calculation (commensurate field vectors or
#: incommensurate field distributions).
MAGNETIC_FIELD_RESULT_TYPE = "magnetic_field"
#: Result type of a dipolar-tensor calculation.
MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE = "magnetic_dipolar_tensor"
#: Result type of a nuclear dipolar relaxation calculation.
NUCLEAR_RELAXATION_RESULT_TYPE = "nuclear_relaxation"
#: Result type of a muon-spin polarisation calculation.
MUON_POLARISATION_RESULT_TYPE = "muon_polarisation"

#: ``data["owner"]`` of the Magnetic Fields page's own results. A saved
#: workspace's results use the workspace id instead.
MAGNETIC_FIELD_PAGE_OWNER = "magnetic_fields"

#: Prefix of a polarisation simulation's ``data["owner"]``; the simulation id follows it
#: (see :mod:`mufinder.services.polarisation_simulations`).
SIMULATION_OWNER_PREFIX = "polarisation:"

#: Site source of a muon site taken from UEP clustering: an unrelaxed UEP
#: minimum or batch cluster, so fields computed there are estimates.
UEP_SITE_SOURCE = "uep_clustering"

#: The note shown beside results computed at an unrelaxed UEP site.
UNRELAXED_RESULTS_NOTE = (
    "Computed at an unrelaxed UEP site, so these are estimates: "
    "the relaxed muon site can sit some distance from the UEP minimum."
)
#: Why a field at a UEP site is an estimate, shown where UEP sites are listed.
UNRELAXED_SITE_NOTE = (
    "UEP sites are unrelaxed, so fields computed at them are estimates: "
    "the relaxed muon site can sit some distance from the UEP minimum."
)
#: Appended to a status line about results calculated at a UEP site.
UNRELAXED_ESTIMATE_SUFFIX = " Computed at an unrelaxed UEP site, so these are estimates."

#: The method behind the dipolar-field, dipolar-tensor and nuclear-relaxation numbers
#: MuFinder calculates now. Add 1 whenever a change means the same inputs give different
#: numbers from one of those calculations, such as a corrected formula, position or
#: default. Leave it alone for a change that gives the same numbers. Stored results with
#: a lower version then show :data:`OLDER_METHOD_RESULTS_NOTE` until recalculated.
#:
#: This constant marks every stored result. For a change that touches only some of them, give
#: that kind of result its own version, as :data:`PLATE_METHOD_VERSION` does, so that the
#: others keep their real status.
CALCULATION_METHOD_VERSION = 1
#: The method behind the demagnetizing tensor of a plate from a photo. Add 1 whenever a change
#: means the same plate gives a different tensor, such as a new resolution or sum. A field
#: result calculated with a plate records it; results without a plate are never marked by it.
#: Version 1 is every plate result from before the version was recorded.
PLATE_METHOD_VERSION = 3
#: Where a stored field result keeps the :data:`PLATE_METHOD_VERSION` it was calculated with.
PLATE_METHOD_VERSION_KEY = "plate_method_version"
_UNRECORDED_PLATE_METHOD_VERSION = 1
#: What a simulation keeps for a source whose fields used no plate, so it never reads as older.
PLATE_METHOD_NOT_USED = 0
#: The method behind the translations of a ``q != 0`` order: the rows a commensurate result lists,
#: and the phases a distribution samples for a component of period up to 12. Add 1 whenever a
#: change means such an order gives different rows or samples. A commensurate result with
#: ``q != 0`` and every distribution record it; ``q = 0`` results are never marked by it.
#: Version 1 is every such result from before the version was recorded, which took a
#: commensurate component at ``T = 0`` alone. Version 2 resolves short periods; version 3
#: also gives reciprocal-equivalent long-period components their shared signed phase.
TRANSLATION_METHOD_VERSION = 3
#: Where a stored field result keeps the :data:`TRANSLATION_METHOD_VERSION` it was calculated with.
TRANSLATION_METHOD_VERSION_KEY = "translation_method_version"
_UNRECORDED_TRANSLATION_METHOD_VERSION = 1
#: The longest commensurate period of a distribution saved before ``commensurate_max_period`` was
#: recorded: components up to it were held at ``T = 0``.
_UNRECORDED_COMMENSURATE_MAX_PERIOD = 64
#: What a simulation keeps for a source this method does not apply to, so it never reads as older.
TRANSLATION_METHOD_NOT_USED = 0
#: Where a stored result keeps the :data:`CALCULATION_METHOD_VERSION` it was calculated with.
CALCULATION_VERSION_KEY = "calculation_version"

#: The note shown beside a stored result calculated with an older method.
OLDER_METHOD_RESULTS_NOTE = "Calculated with an older method. Calculate to update."

MAGNETIC_FIELD_RESULT_TYPES = (
    MAGNETIC_FIELD_RESULT_TYPE,
    MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
    NUCLEAR_RELAXATION_RESULT_TYPE,
    MUON_POLARISATION_RESULT_TYPE,
)


def is_unrelaxed_site_source(source: object) -> bool:
    """Return whether a muon site of ``source`` is unrelaxed, so its fields are estimates."""

    return source == UEP_SITE_SOURCE


def uses_older_method(result: Mapping[str, Any]) -> bool:
    """Return whether a stored result predates :data:`CALCULATION_METHOD_VERSION`.

    ``result`` is record data or a form-state result; one with no version
    recorded counts as older, and an empty or malformed one is no result at all.
    """

    if not isinstance(result, Mapping) or not result:
        return False
    version = result.get(CALCULATION_VERSION_KEY)
    return (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < CALCULATION_METHOD_VERSION
    )


def sample_predates_plate_method(form_state: object) -> bool:
    """Return whether a form's sample gave different numbers before the plate method changed.

    That is a plate from a photo, or a voxel grid saved with another padding or with smoothing.
    Read this from the saved form: loading a sample and saving it drops those voxel settings.
    """

    if not isinstance(form_state, Mapping):
        return False
    shape = str(form_state.get("demag_shape", "")).strip().lower()
    if shape == "image_plate":
        return True
    sample = form_state.get("demag_sample")
    if shape != "voxel_grid" or not isinstance(sample, Mapping):
        return False
    return (
        sample.get("padding", FOURIER_PADDING) != FOURIER_PADDING
        or sample.get("smoothing", 0.0) != 0.0
    )


def translation_method_applies(result: Mapping[str, Any]) -> bool:
    """Return whether a dipolar-field result is a distribution or a commensurate one at ``q != 0``."""

    rows = result.get("results")
    if not isinstance(rows, list):
        return False
    if result.get("kind") == "distribution":
        return True
    if result.get("kind") != "commensurate":
        return False
    for row in rows:
        parameters = row.get("calculation_parameters") if isinstance(row, Mapping) else None
        if isinstance(parameters, Mapping) and parameters.get("modulated"):
            return True
    return False


def lists_only_one_translation(result: Mapping[str, Any]) -> bool:
    """Return whether a commensurate ``q != 0`` result holds rows for ``T = 0`` alone.

    That is what such a result held before :data:`TRANSLATION_METHOD_VERSION` was recorded. Rows
    at ``T = 0`` lie in the home cell, so the list of an approximant, whose other rows lie beyond
    it, is not one.
    """

    return (
        result.get("kind") == "commensurate"
        and translation_method_applies(result)
        and not any(_lies_beyond_home_cell(row) for row in result["results"])
    )


def _lies_beyond_home_cell(row: object) -> bool:
    position = row.get("muon_position") if isinstance(row, Mapping) else None
    return isinstance(position, list) and any(
        isinstance(value, (int, float)) and value >= 1.0 for value in position
    )


def distribution_held_a_short_period_at_zero(
    result: Mapping[str, Any], magnetic_structure: object
) -> bool:
    """Return whether a distribution saved without the key took a short-period term at ``T = 0`` alone.

    That is what a distribution did for a ``q != 0`` term of period up to its recorded
    ``commensurate_max_period``, 64 when it records none, before :data:`TRANSLATION_METHOD_VERSION` was recorded.
    ``magnetic_structure`` is the model its owner keeps, as JSON or a model; it is the one the
    distribution was calculated from, since giving the owner another model clears its results.
    ``None`` or a model that does not read counts as not older.
    """

    return _distribution_model_matches(
        result,
        magnetic_structure,
        has_short_period_modulation,
        default_max_period=_UNRECORDED_COMMENSURATE_MAX_PERIOD,
    )


def distribution_used_aliased_phase_basis(
    result: Mapping[str, Any], magnetic_structure: object
) -> bool:
    """Return whether a distribution's model needs the reciprocal phase-basis correction."""

    return _distribution_model_matches(
        result,
        magnetic_structure,
        has_reciprocal_phase_aliases,
        default_max_period=MAX_LISTED_TRANSLATION_PERIOD,
    )


def _distribution_model_matches(
    result: Mapping[str, Any],
    magnetic_structure: object,
    predicate: Callable[..., bool],
    *,
    default_max_period: int,
) -> bool:
    rows = result.get("results")
    if result.get("kind") != "distribution" or not isinstance(rows, list) or not rows:
        return False
    try:
        model = (
            magnetic_structure
            if isinstance(magnetic_structure, MagneticStructureModel)
            else MagneticStructureModel.from_json(magnetic_structure)
        )
        first = rows[0] if isinstance(rows[0], Mapping) else {}
        parameters = first.get("calculation_parameters")
        parameters = parameters if isinstance(parameters, Mapping) else {}
        return predicate(
            model,
            tolerance=float(parameters.get("commensurate_tolerance", 1e-8)),
            max_period=int(parameters.get("commensurate_max_period", default_max_period)),
        )
    except (MagneticFieldError, MagneticStructureError, TypeError, ValueError):
        return False


def _predates_translation_method(result: Mapping[str, Any], magnetic_structure: object) -> bool:
    version = result.get(TRANSLATION_METHOD_VERSION_KEY)
    if version is None:
        return (
            lists_only_one_translation(result)
            or distribution_held_a_short_period_at_zero(result, magnetic_structure)
            or distribution_used_aliased_phase_basis(result, magnetic_structure)
        )
    if version == TRANSLATION_METHOD_NOT_USED and not isinstance(version, bool):
        return False
    if (
        version == 2
        and isinstance(version, int)
        and not isinstance(version, bool)
        and "kind" in result
    ):
        return distribution_used_aliased_phase_basis(result, magnetic_structure)
    return (
        not isinstance(version, int)
        or isinstance(version, bool)
        or version < TRANSLATION_METHOD_VERSION
    )


def field_result_uses_older_method(
    result: object, form_state: object = None, magnetic_structure: object = None
) -> bool:
    """Return whether a stored dipolar-field result predates the method that now applies to it.

    ``form_state`` is the saved form of the result's owner, or just its ``demag_shape`` and
    ``demag_sample``. It decides for a result that records no plate method version.
    ``magnetic_structure`` is the owner's model (:func:`magnetic_field_owner_model`); it decides
    for a distribution that records no translation method version.
    """

    if not isinstance(result, Mapping) or not result:
        return False
    if uses_older_method(result) or _predates_translation_method(result, magnetic_structure):
        return True
    version = result.get(PLATE_METHOD_VERSION_KEY)
    if version is None:
        return sample_predates_plate_method(form_state)
    if version == PLATE_METHOD_NOT_USED and not isinstance(version, bool):
        return False
    return (
        not isinstance(version, int) or isinstance(version, bool) or version < PLATE_METHOD_VERSION
    )


def magnetic_field_owner_form(document: ProjectDocument, owner: str) -> Mapping[str, Any] | None:
    """Return the saved form of the page or workspace ``owner``; ``None`` once it is gone."""

    if owner == MAGNETIC_FIELD_PAGE_OWNER:
        form = document.ui_state.get("magnetic_field_form")
        return form if isinstance(form, Mapping) else None
    workspaces = document.ui_state.get("magnetic_field_workspaces")
    for workspace in workspaces if isinstance(workspaces, list) else ():
        if isinstance(workspace, Mapping) and str(workspace.get("id", "")).strip() == owner:
            form = workspace.get("form")
            return form if isinstance(form, Mapping) else None
    return None


def magnetic_field_owner_model(document: ProjectDocument, owner: str) -> object:
    """Return the magnetic structure JSON the page or workspace ``owner`` keeps; ``None`` once it is gone."""

    if owner == MAGNETIC_FIELD_PAGE_OWNER:
        return document.project.magnetic_structure
    workspaces = document.ui_state.get("magnetic_field_workspaces")
    for workspace in workspaces if isinstance(workspaces, list) else ():
        if isinstance(workspace, Mapping) and str(workspace.get("id", "")).strip() == owner:
            return workspace.get("magnetic_structure")
    return None


def recorded_plate_method_version(data: Mapping[str, Any], form_state: object) -> object:
    """Return the plate method version of a field record, as a simulation of it should keep it."""

    version = data.get(PLATE_METHOD_VERSION_KEY)
    if version is not None:
        return version
    if sample_predates_plate_method(form_state):
        return _UNRECORDED_PLATE_METHOD_VERSION
    return PLATE_METHOD_NOT_USED


def recorded_translation_method_version(
    data: Mapping[str, Any], magnetic_structure: object = None
) -> object:
    """Return the translation method version of a field record, as a simulation should keep it.

    ``magnetic_structure`` is the record owner's model, which decides for a distribution.
    An unaffected version 2 record has the current method's effective version, so its
    simulation source keeps the same status without needing the owner's model later.
    """

    version = data.get(TRANSLATION_METHOD_VERSION_KEY)
    if version is not None:
        if (
            version == 2
            and not isinstance(version, bool)
            and not _predates_translation_method(data, magnetic_structure)
        ):
            return TRANSLATION_METHOD_VERSION
        return version
    if lists_only_one_translation(data) or distribution_held_a_short_period_at_zero(
        data, magnetic_structure
    ):
        return _UNRECORDED_TRANSLATION_METHOD_VERSION
    if distribution_used_aliased_phase_basis(data, magnetic_structure):
        return 2
    return TRANSLATION_METHOD_NOT_USED


def field_result_site_source(
    field_results: Mapping[str, Any],
    form_state: Mapping[str, object] | None = None,
) -> str | None:
    """Return the site source of a dipolar-field result.

    ``field_results`` is the record data or the form's ``magnetic_field_results``.
    A result saved without a source takes it from ``form_state``'s
    ``selected_muon_site``.
    """

    source = field_results.get("site_source")
    if not source and isinstance(form_state, Mapping):
        site = form_state.get("selected_muon_site")
        source = site.get("source") if isinstance(site, Mapping) else None
    return str(source) if source else None


#: The method versions a field result carries between its form state and its record.
_FIELD_VERSION_KEYS = (
    CALCULATION_VERSION_KEY,
    PLATE_METHOD_VERSION_KEY,
    TRANSLATION_METHOD_VERSION_KEY,
)


def _field_data(value: object) -> dict[str, Any] | None:
    if (
        not isinstance(value, Mapping)
        or not isinstance(value.get("results"), list)
        or not value["results"]
    ):
        return None
    data: dict[str, Any] = {"kind": str(value.get("kind", "")), "results": list(value["results"])}
    if value.get("site_source"):
        data["site_source"] = str(value["site_source"])
    for key in _FIELD_VERSION_KEYS:
        if key in value:
            data[key] = value[key]
    return data


def _field_value(data: Mapping[str, Any]) -> object:
    value: dict[str, Any] = {"kind": data.get("kind", ""), "results": list(data.get("results", []))}
    if data.get("site_source"):
        value["site_source"] = str(data["site_source"])
    for key in _FIELD_VERSION_KEYS:
        if key in data:
            value[key] = data[key]
    return value


def _tensor_data(value: object) -> dict[str, Any] | None:
    if not isinstance(value, list) or not value:
        return None
    data: dict[str, Any] = {"results": list(value)}
    for key in ("site_source", CALCULATION_VERSION_KEY):
        shared = [entry.get(key) if isinstance(entry, Mapping) else None for entry in value]
        if shared[0] and all(item == shared[0] for item in shared):
            data[key] = str(shared[0]) if key == "site_source" else shared[0]
            data["results"] = [
                {name: item for name, item in entry.items() if name != key}
                for entry in data["results"]
            ]
    return data


def _tensor_value(data: Mapping[str, Any]) -> object:
    rows = list(data.get("results", []))
    shared: dict[str, Any] = {}
    if data.get("site_source"):
        shared["site_source"] = str(data["site_source"])
    if CALCULATION_VERSION_KEY in data:
        shared[CALCULATION_VERSION_KEY] = data[CALCULATION_VERSION_KEY]
    if not shared:
        return rows
    return [{**row, **shared} if isinstance(row, Mapping) else row for row in rows]


def _nuclear_data(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    data: dict[str, Any] = {
        "result": {
            key: item
            for key, item in value.items()
            if key not in ("site_source", CALCULATION_VERSION_KEY)
        }
    }
    if value.get("site_source"):
        data["site_source"] = str(value["site_source"])
    if CALCULATION_VERSION_KEY in value:
        data[CALCULATION_VERSION_KEY] = value[CALCULATION_VERSION_KEY]
    return data


def _nuclear_value(data: Mapping[str, Any]) -> object:
    result = data.get("result")
    value = dict(result) if isinstance(result, Mapping) else {}
    if value and data.get("site_source"):
        value["site_source"] = str(data["site_source"])
    if value and CALCULATION_VERSION_KEY in data:
        value[CALCULATION_VERSION_KEY] = data[CALCULATION_VERSION_KEY]
    return value


#: Form-state key, result type, form value to record data, record data to form value.
_FORM_RESULTS: tuple[
    tuple[
        str, str, Callable[[object], dict[str, Any] | None], Callable[[Mapping[str, Any]], object]
    ],
    ...,
] = (
    ("magnetic_field_results", MAGNETIC_FIELD_RESULT_TYPE, _field_data, _field_value),
    ("dipolar_tensor_results", MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE, _tensor_data, _tensor_value),
    ("nuclear_relaxation_results", NUCLEAR_RELAXATION_RESULT_TYPE, _nuclear_data, _nuclear_value),
)

#: The form-state keys whose values live in ``Result`` records.
FORM_RESULT_KEYS = tuple(key for key, *_ in _FORM_RESULTS)

#: The form value that means "no result" for each key.
_EMPTY_FORM_RESULTS: dict[str, Callable[[], object]] = {
    "magnetic_field_results": lambda: None,
    "dipolar_tensor_results": list,
    "nuclear_relaxation_results": dict,
}


def magnetic_field_results(document: ProjectDocument, owner: str | None = None) -> list[Result]:
    """Return the Magnetic Fields result records, oldest first, optionally for one owner."""

    records = [
        result
        for result in document.results.values()
        if result.result_type in MAGNETIC_FIELD_RESULT_TYPES
        and (owner is None or _owner(result) == owner)
    ]
    return sorted(records, key=lambda result: result.created_at)


def magnetic_field_result(document: ProjectDocument, owner: str, result_type: str) -> Result | None:
    """Return ``owner``'s record of ``result_type``, or ``None``."""

    return next(
        (
            result
            for result in magnetic_field_results(document, owner)
            if result.result_type == result_type
        ),
        None,
    )


def store_magnetic_field_results(
    document: ProjectDocument,
    owner: str,
    form_state: Mapping[str, object],
    *,
    journal_title: str | None = None,
) -> dict[str, object]:
    """Record the results in ``form_state`` for ``owner`` and return the form without them.

    A result that changed replaces ``owner``'s record of that type with a new
    record; an unchanged result keeps its record; an empty or missing result
    removes it. With ``journal_title``, the title of the page the results were
    calculated on, new dipolar-field values are also recorded as a finding; a
    copy of results already calculated passes none.
    """

    for key, result_type, to_data, _ in _FORM_RESULTS:
        data = to_data(to_jsonable(form_state.get(key)))
        existing = magnetic_field_result(document, owner, result_type)
        if data is None:
            if existing is not None:
                del document.results[existing.id]
            continue
        data = {"owner": owner, **data}
        if existing is not None:
            if existing.data == data:
                continue
            del document.results[existing.id]
        record = Result(result_type=result_type, data=data)
        document.results[record.id] = record
        # Only new values count: a source added to an older record is not a calculation.
        if (
            journal_title is not None
            and result_type == MAGNETIC_FIELD_RESULT_TYPE
            and (existing is None or existing.data.get("results") != data.get("results"))
        ):
            record_finding(
                document, _field_calculated_record(owner, journal_title, data, form_state)
            )
    return {key: value for key, value in form_state.items() if key not in FORM_RESULT_KEYS}


#: The field terms a calculation reports, as the Journal names them.
_FIELD_TERMS = (
    ("dipolar", "dipolar"),
    ("lorentz", "Lorentz"),
    ("contact", "contact"),
    ("demag", "demagnetizing"),
    ("applied", "applied"),
)


def _field_calculated_record(
    owner: str, title: str, data: Mapping[str, Any], form_state: Mapping[str, object]
) -> ProvenanceRecord:
    """Return the finding of a dipolar-field calculation, without its sample arrays.

    The fingerprint holds the owner and the calculated values rather than the
    inputs: the magnetic structure is not part of the form, and the values
    change whenever any input does.
    """

    results = [result for result in data.get("results", []) if isinstance(result, Mapping)]
    site = form_state.get("selected_muon_site")
    site_label = str(site.get("label", "")).strip() if isinstance(site, Mapping) else ""
    if not site_label and results:
        site_label = str(results[0].get("site_label", "")).strip()
    kind = str(data.get("kind", ""))
    rows = [
        _commensurate_row(result) if kind == "commensurate" else _distribution_row(result)
        for result in results
    ]
    settings = _field_settings(results, site, form_state)
    return ProvenanceRecord(
        action="field_calculated",
        workflow_stage=WorkflowStage.MAGNETIC_FIELD,
        actor="user",
        description=f"Calculated the magnetic field at {site_label} on '{title}'",
        parameters={
            "owner": owner,
            "owner_title": title,
            "kind": kind,
            "site_label": site_label,
            "estimate": is_unrelaxed_site_source(data.get("site_source")),
            "terms": [
                name for key, name in _FIELD_TERMS if any(row["terms"].get(key) for row in rows)
            ],
            "results": rows,
            "applied_field_T": to_jsonable(form_state.get("applied_field_T")),
            "settings": settings,
            SUBJECT_KEY: owner,
            FINGERPRINT_KEY: input_fingerprint(owner, rows, settings),
        },
    )


def _field_settings(
    results: list[Mapping[str, Any]], site: object, form_state: Mapping[str, object]
) -> dict[str, object]:
    """The settings of a calculation: what its results say they used, else what the form held."""

    raw = results[0].get("calculation_parameters") if results else None
    used = raw if isinstance(raw, Mapping) else {}
    return {
        "contact_coupling_per_A3": used.get("contact_coupling"),
        "contact_neighbours": used.get("nnn"),
        "contact_radius_A": used.get("rcont"),
        "supercell": to_jsonable(used.get("supercell", form_state.get("supercell"))),
        "equivalent_sites": form_state.get("include_equivalent_sites"),
        "distortion_corrected": bool(
            used.get("distortion_corrected")
            or (isinstance(site, Mapping) and site.get("include_distortion"))
        ),
        "phase_samples": used.get("phase_sample_count"),
        "phase_sampling": used.get("resolved_phase_sampling"),
        "phase_sampling_seed": used.get("phase_sampling_seed"),
    }


def _is_nonzero(vector: object) -> bool:
    return isinstance(vector, list) and any(
        isinstance(value, (int, float)) and value != 0.0 for value in vector
    )


def _commensurate_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """One equivalent site's field: its size, its vector and the terms that contribute."""

    return {
        "site_label": result.get("site_label"),
        "muon_position": result.get("muon_position"),
        "magnitude_T": result.get("magnitude_T"),
        "total_T": result.get("total_T"),
        "terms": {
            key: result.get(f"{key}_T")
            for key, _name in _FIELD_TERMS
            if _is_nonzero(result.get(f"{key}_T"))
        },
    }


def _distribution_row(result: Mapping[str, Any]) -> dict[str, Any]:
    """One site's field distribution: its RMS, its mean and each term's RMS."""

    raw_terms = result.get("terms")
    terms = raw_terms if isinstance(raw_terms, Mapping) else {}
    checks = result.get("convergence")
    return {
        "site_label": result.get("site_label"),
        "muon_position": result.get("muon_position"),
        "total_rms_T": result.get("total_rms_T"),
        "mean_T": result.get("mean_T"),
        "terms": {
            key: term.get("rms_T")
            for key, term in terms.items()
            if isinstance(term, Mapping) and term.get("rms_T")
        },
        "not_converged": [
            str(check.get("message", ""))
            for check in (checks if isinstance(checks, list) else [])
            if isinstance(check, Mapping) and not check.get("converged", True)
        ],
    }


def attach_magnetic_field_results(
    document: ProjectDocument,
    owner: str,
    form_state: Mapping[str, object] | None,
) -> dict[str, object]:
    """Return ``form_state`` with ``owner``'s recorded results put back in.

    Every result key ``owner`` has no record of gets its empty value, so a page
    reused across projects or workspaces drops the results it showed before.
    An empty form with no records stays empty, which the page reads as "reset
    to defaults".
    """

    attached = dict(form_state) if isinstance(form_state, Mapping) else {}
    for key, result_type, _, to_value in _FORM_RESULTS:
        record = magnetic_field_result(document, owner, result_type)
        if record is not None and isinstance(record.data, Mapping):
            attached[key] = to_value(record.data)
    field_results = attached.get("magnetic_field_results")
    if isinstance(field_results, dict) and not field_results.get("site_source"):
        source = field_result_site_source(field_results, attached)
        if source is not None:
            attached["magnetic_field_results"] = {**field_results, "site_source": source}
    if not attached:
        return attached
    for key, empty in _EMPTY_FORM_RESULTS.items():
        attached.setdefault(key, empty())
    return attached


def store_polarisation_result(
    document: ProjectDocument,
    owner: str,
    result: PolarisationResult | None,
) -> Result | None:
    """Record ``result`` as ``owner``'s polarisation and return the record.

    The record replaces ``owner``'s previous polarisation record, unless that
    one holds the same result; ``None`` removes it.
    """

    existing = magnetic_field_result(document, owner, MUON_POLARISATION_RESULT_TYPE)
    if result is None:
        if existing is not None:
            del document.results[existing.id]
        return None
    data = {"owner": owner, "result": result.to_json()}
    if existing is not None:
        if existing.data == data:
            return existing
        del document.results[existing.id]
    record = Result(result_type=MUON_POLARISATION_RESULT_TYPE, data=data)
    document.results[record.id] = record
    return record


def polarisation_result(document: ProjectDocument, owner: str) -> PolarisationResult | None:
    """Return ``owner``'s recorded polarisation, or ``None`` if it has none or it is malformed."""

    record = magnetic_field_result(document, owner, MUON_POLARISATION_RESULT_TYPE)
    payload = record.data.get("result") if record is not None else None
    if not isinstance(payload, Mapping):
        return None
    try:
        return PolarisationResult.from_json(payload)
    except PolarisationError:
        return None


def remove_magnetic_field_results(document: ProjectDocument, owner: str) -> None:
    """Delete every result record ``owner`` holds."""

    for result in magnetic_field_results(document, owner):
        del document.results[result.id]


def commensurate_field_entries(result: Result) -> Iterator[Mapping[str, Any]]:
    """Yield the per-site entries of a commensurate field record."""

    if result.result_type != MAGNETIC_FIELD_RESULT_TYPE or not isinstance(result.data, Mapping):
        return
    if result.data.get("kind") != "commensurate":
        return
    entries = result.data.get("results")
    if isinstance(entries, list):
        yield from (entry for entry in entries if isinstance(entry, Mapping))


def _owner(result: Result) -> str | None:
    owner = result.data.get("owner") if isinstance(result.data, Mapping) else None
    return str(owner) if owner is not None else None
