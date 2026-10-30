"""Submission-time validation of CASTEP task-farm counts against MPI ranks.

Task farming splits a calculation across independent CASTEP farms, each of which
gets ``floor(ranks / farms)`` MPI ranks. Two rules follow, and they differ in
severity:

* ``farms > ranks`` is fatal. CASTEP's ``comms_parallel_strategy`` calls
  ``comms_abort()`` outright ("Not enough nodes ... for requested number of
  farms"), so the job dies immediately rather than running slowly.
* ``ranks % farms != 0`` merely wastes the remainder: those ranks are assigned
  to no farm and sit idle for the whole run.

The farm count is chosen per row when a workflow is configured, while the rank
count is chosen at submission, so this is the first point at which the two are
known together. Both submission dialogs use these helpers so the rules read the
same wherever they surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mufinder.domain.text import plural


@dataclass(frozen=True, slots=True)
class FarmRankVerdict:
    """Outcome of checking configured farm counts against a rank count."""

    #: Set when at least one job requests more farms than there are ranks.
    #: Submission must be refused: CASTEP would abort.
    blocking_message: str = ""
    #: Set when farms divide unevenly into ranks. Advisory only.
    warning_message: str = ""

    @property
    def blocked(self) -> bool:
        """``True`` when submission must not proceed."""

        return bool(self.blocking_message)


def check_farm_counts(
    farm_counts: Mapping[str, int],
    *,
    ranks: int,
    rank_label: str = "processor cores",
    mpirun_configured: bool = True,
) -> FarmRankVerdict:
    """Validate configured task-farm counts against the requested rank count.

    ``farm_counts`` maps a job label to its configured farm count; entries of 1
    or less are ignored (farming disabled). ``rank_label`` names the rank control
    in the calling dialog so the message points at the right spin box.
    """

    farmed = {str(label): int(count) for label, count in farm_counts.items() if int(count) > 1}
    if not farmed:
        return FarmRankVerdict()

    largest = max(farmed.values())
    if not mpirun_configured:
        return FarmRankVerdict(
            blocking_message=(
                f"{_describe(farmed)} configured for task farming, which needs multiple "
                "MPI ranks. Set up mpirun in Setup > CASTEP, or set the farm count back "
                "to 1 in Configure."
            )
        )

    available = max(1, int(ranks))
    if available < largest:
        return FarmRankVerdict(
            blocking_message=(
                f"{_describe(farmed)} configured for more task farms than the {available} "
                f"{rank_label} requested. CASTEP stops immediately when farms outnumber MPI "
                f"ranks, so raise {rank_label} to at least {largest}, or lower the farm count "
                "in Configure."
            )
        )

    uneven = sorted({count for count in farmed.values() if available % count})
    if uneven:
        idle = min(available % count for count in uneven)
        return FarmRankVerdict(
            warning_message=(
                f"{available} {rank_label} do not divide evenly into "
                f"{_join(str(count) for count in uneven)} farms. At least {plural(idle, 'rank')} "
                "will sit idle for the whole run."
            )
        )
    return FarmRankVerdict()


def _describe(farmed: Mapping[str, int]) -> str:
    if len(farmed) == 1:
        label, count = next(iter(farmed.items()))
        return f"'{label}' is"
    return f"{len(farmed)} jobs are"


def _join(values) -> str:
    items = list(values)
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} or {items[-1]}"


__all__ = ["FarmRankVerdict", "check_farm_counts"]
