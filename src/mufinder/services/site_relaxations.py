"""Recording a completed muon relaxation on the site it started from.

A muon site links to one relaxation run (``MuonSite.relaxation_run_id``) and
keeps convenience copies of its energy and relaxed position. A start relaxed in
both charge states has a completed relaxation in each, so the link records the
charge state the site was first relaxed in and only a relaxation in that charge
state advances it. The other charge state's relaxation is found through the
muon-relaxation row bound to the site
(:func:`~mufinder.services.source_candidates.site_charge_state_relaxation_jobs`),
which keeps every downstream source key stable when the second charge state
finishes.
"""

from __future__ import annotations

from collections.abc import Iterable

from mufinder.domain.models import CalculationRun, Job, MuonSite
from mufinder.services.project_document import ProjectDocument
from mufinder.services.source_candidates import charge_state_for_job, site_linked_charge_state


def record_site_relaxation(
    document: ProjectDocument,
    site: MuonSite,
    run: CalculationRun,
    job: Job,
    *,
    total_energy_eV: float | None,
    fractional_position: Iterable[float] | None,
) -> bool:
    """Link ``site`` to ``run`` unless the site is linked to another charge state.

    The link moves when the site has none, its linked run or job is gone, or
    ``job`` relaxes the same charge state as the linked run (a continued or
    rerun relaxation). Returns whether the site was updated.
    """

    linked = site_linked_charge_state(document, site)
    if linked is not None and linked != charge_state_for_job(job):
        return False
    site.record_relaxation(
        run.id,
        total_energy_eV=total_energy_eV,
        fractional_position=fractional_position,
    )
    return True


__all__ = ["record_site_relaxation"]
