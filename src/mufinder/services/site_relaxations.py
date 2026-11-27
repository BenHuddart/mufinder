"""Recording a completed muon relaxation on the site it started from.

A muon site links to one relaxation run (``MuonSite.relaxation_run_id``) and
keeps convenience copies of its energy and relaxed position. A start relaxed in
both charge states has a completed relaxation in each. The charge state the
site is first relaxed in is recorded on the site
(:data:`~mufinder.services.source_candidates.SITE_KEY_CHARGE_STATE_PROPERTY`)
and owns both the link and the bare site-id source key; only a relaxation in
that charge state links the site. The other charge state's relaxation is found
through the muon-relaxation row bound to the site
(:func:`~mufinder.services.source_candidates.site_charge_state_relaxation_jobs`),
so every downstream source key stays with one charge state even when a
relaxation is reset and the other charge state finishes later.
"""

from __future__ import annotations

from collections.abc import Iterable

from mufinder.domain.models import CalculationRun, Job, MuonSite
from mufinder.services.project_document import ProjectDocument
from mufinder.services.source_candidates import (
    SITE_KEY_CHARGE_STATE_PROPERTY,
    charge_state_for_job,
    site_key_charge_state,
)


def record_site_relaxation(
    document: ProjectDocument,
    site: MuonSite,
    run: CalculationRun,
    job: Job,
    *,
    total_energy_eV: float | None,
    fractional_position: Iterable[float] | None,
) -> bool:
    """Link ``site`` to ``run`` unless the site belongs to another charge state.

    The link moves when the site has no recorded charge state yet (which
    ``job``'s then becomes) or ``job`` relaxes the recorded charge state (a
    continued or rerun relaxation, or a rerun after that relaxation was reset).
    A relaxation in the other charge state leaves the site unchanged, even
    while the site has no link. Returns whether the site was updated.
    """

    owner = site_key_charge_state(document, site)
    charge_state = charge_state_for_job(job)
    if owner is not None and charge_state is not None and owner != charge_state:
        return False
    if owner is None and charge_state is not None:
        owner = charge_state
    if owner is not None:
        site.properties[SITE_KEY_CHARGE_STATE_PROPERTY] = owner
    site.record_relaxation(
        run.id,
        total_energy_eV=total_energy_eV,
        fractional_position=fractional_position,
    )
    return True


__all__ = ["record_site_relaxation"]
