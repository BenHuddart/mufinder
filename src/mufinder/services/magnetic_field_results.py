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

from mufinder.domain.models import Result
from mufinder.persistence.codec import to_jsonable
from mufinder.science.polarisation import PolarisationError, PolarisationResult
from mufinder.services.project_document import ProjectDocument


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

MAGNETIC_FIELD_RESULT_TYPES = (
    MAGNETIC_FIELD_RESULT_TYPE,
    MAGNETIC_DIPOLAR_TENSOR_RESULT_TYPE,
    NUCLEAR_RELAXATION_RESULT_TYPE,
    MUON_POLARISATION_RESULT_TYPE,
)


def is_unrelaxed_site_source(source: object) -> bool:
    """Return whether a muon site of ``source`` is unrelaxed, so its fields are estimates."""

    return source == UEP_SITE_SOURCE


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
    return data


def _field_value(data: Mapping[str, Any]) -> object:
    value: dict[str, Any] = {"kind": data.get("kind", ""), "results": list(data.get("results", []))}
    if data.get("site_source"):
        value["site_source"] = str(data["site_source"])
    return value


def _tensor_data(value: object) -> dict[str, Any] | None:
    if not isinstance(value, list) or not value:
        return None
    sources = {entry.get("site_source") if isinstance(entry, Mapping) else None for entry in value}
    source = sources.pop() if len(sources) == 1 else None
    if not source:
        return {"results": list(value)}
    rows = [{key: item for key, item in entry.items() if key != "site_source"} for entry in value]
    return {"results": rows, "site_source": str(source)}


def _tensor_value(data: Mapping[str, Any]) -> object:
    rows = list(data.get("results", []))
    source = data.get("site_source")
    if not source:
        return rows
    return [
        {**row, "site_source": str(source)} if isinstance(row, Mapping) else row for row in rows
    ]


def _nuclear_data(value: object) -> dict[str, Any] | None:
    if not isinstance(value, Mapping) or not value:
        return None
    data: dict[str, Any] = {
        "result": {key: item for key, item in value.items() if key != "site_source"}
    }
    if value.get("site_source"):
        data["site_source"] = str(value["site_source"])
    return data


def _nuclear_value(data: Mapping[str, Any]) -> object:
    result = data.get("result")
    value = dict(result) if isinstance(result, Mapping) else {}
    if value and data.get("site_source"):
        value["site_source"] = str(data["site_source"])
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
) -> dict[str, object]:
    """Record the results in ``form_state`` for ``owner`` and return the form without them.

    A result that changed replaces ``owner``'s record of that type with a new
    record; an unchanged result keeps its record; an empty or missing result
    removes it.
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
    return {key: value for key, value in form_state.items() if key not in FORM_RESULT_KEYS}


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
