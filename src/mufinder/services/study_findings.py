"""Findings: the provenance records of the results that conclude a study.

The Journal reads the provenance stream, so a result reaches it by leaving one
record there. Each record is built by the service that holds the result and
carries the numbers its expanded card shows, in the record's ``parameters``:

================================ ==============================================
``uep_minima_found``             :func:`record_uep_minima`
``sites_grouped``                ``services.muon_sites_summary``
``field_calculated``             ``services.magnetic_field_results``
``polarisation_simulated``       ``services.polarisation_simulations``
``quantum_corrected``            ``services.job_findings``
``barrier_calculated``           ``services.job_findings``
================================ ==============================================

Recalculating with identical inputs adds nothing: each record names its subject
(``SUBJECT_KEY``: the job, workspace, page, simulation or row it is about) and keeps
an :func:`input_fingerprint`, and :func:`record_finding` skips one that repeats the
latest finding about the same subject on the same local day. So the last card about
a subject is always current: ``A``, ``B``, ``A`` gives three. A later day records a
repeat too, because it is then part of the project's history.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from typing import Any

from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Job, ProvenanceRecord
from mufinder.persistence.codec import to_jsonable
from mufinder.services.project_document import ProjectDocument, append_provenance

LOGGER = logging.getLogger(__name__)

#: The ``parameters`` key of a finding's fingerprint.
FINGERPRINT_KEY = "input_fingerprint"
#: The ``parameters`` key of what a finding is about, within its action.
SUBJECT_KEY = "finding_subject"


def input_fingerprint(*inputs: object) -> str:
    """Return a stable digest of what a result was calculated from."""

    text = json.dumps(to_jsonable(inputs), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def record_finding(document: ProjectDocument, record: ProvenanceRecord) -> bool:
    """Append ``record`` unless it repeats the latest finding about its subject today.

    Returns whether it was appended.
    """

    latest = next(
        (
            earlier
            for earlier in reversed(document.provenance)
            if earlier.action == record.action
            and earlier.parameters.get(SUBJECT_KEY) == record.parameters.get(SUBJECT_KEY)
        ),
        None,
    )
    if (
        latest is not None
        and latest.parameters.get(FINGERPRINT_KEY) == record.parameters.get(FINGERPRINT_KEY)
        and latest.timestamp.astimezone().date() == record.timestamp.astimezone().date()
    ):
        return False
    try:
        append_provenance(document, record)
    except OSError:
        # The result is stored without its record: the Journal is not worth losing it for.
        LOGGER.exception("Could not write the finding %s", record.action)
        return False
    return True


def _uep_grouping_signature(state: object) -> list[object] | None:
    """What a UEP analysis state's search grouped into.

    The tolerance is not part of it, so a tolerance that gives the same groups is
    not a new finding.
    """

    summary = state.get("batch_summary_data") if isinstance(state, Mapping) else None
    clusters = summary.get("clusters") if isinstance(summary, Mapping) else None
    if not isinstance(clusters, list) or not clusters:
        return None
    found = [
        (
            cluster.get("size"),
            list(cluster.get("member_indexes") or []),
            cluster.get("minimum_energy_ev"),
        )
        for cluster in clusters
        if isinstance(cluster, Mapping)
    ]
    return found


def record_uep_minima(
    document: ProjectDocument,
    job: Job,
    previous_state: Mapping[str, Any] | None,
    state: Mapping[str, Any],
) -> bool:
    """Record the grouped minima of a UEP batch search when ``state`` holds new groups.

    A search is grouped as it ends, so the grouping is what concludes it. The state
    changes for many other reasons, such as a turned camera or a tolerance that
    gives the same groups; those record nothing, and the zero-point energy changes
    neither the groups nor their classical energies. The fingerprint is the job, the
    optimisation's start positions and what the grouping found, because the state
    does not keep the search settings that were used.
    """

    signature = _uep_grouping_signature(state)
    if signature is None or signature == _uep_grouping_signature(previous_state):
        return False
    summary = state["batch_summary_data"]
    raw_results = state.get("batch_results")
    results = raw_results if isinstance(raw_results, list) else []
    minima = sorted(
        (_uep_minimum(cluster) for cluster in summary["clusters"] if isinstance(cluster, Mapping)),
        key=lambda item: (
            item["minimum_energy_eV"] is None,
            item["minimum_energy_eV"] or 0.0,
            item["cluster_index"],
        ),
    )
    grouping = {
        key: summary.get(key)
        for key in ("clustering_method", "threshold_ang", "energy_tolerance_ev")
    }
    starts = [result.get("start_position_ang") for result in results if isinstance(result, Mapping)]
    return record_finding(
        document,
        ProvenanceRecord(
            action="uep_minima_found",
            workflow_stage=WorkflowStage.UEP,
            actor="user",
            description=f"Grouped the batch search of {job.label}",
            parameters={
                "job_label": job.label,
                "start_count": len(results),
                "minima": minima,
                "grouping": grouping,
                "excluded_count": len(results) - sum(item["size"] or 0 for item in minima),
                SUBJECT_KEY: str(job.id),
                FINGERPRINT_KEY: input_fingerprint(str(job.id), starts, signature),
            },
            job_id=job.id,
        ),
    )


def _uep_minimum(cluster: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "cluster_index": cluster.get("cluster_index"),
        "size": cluster.get("size"),
        "fractional_position": cluster.get("representative_fractional_position"),
        "minimum_energy_eV": cluster.get("minimum_energy_ev"),
    }


__all__ = [
    "FINGERPRINT_KEY",
    "SUBJECT_KEY",
    "input_fingerprint",
    "record_finding",
    "record_uep_minima",
]
