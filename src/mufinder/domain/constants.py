"""Physical constants for the muon and related particles."""

from __future__ import annotations

MUON_MASS_AMU: float = 0.11342892566
MUON_GAMMA_RAD_S_T: float = 851586494.1

#: Tag marking a job that records a calculation performed outside MuFinder.
#:
#: Carried on the ``Job`` rather than looked up through its definition row,
#: because every guard that must respect it — dashboard reset, resubmission,
#: continuation — already holds the job and may not hold the row. A job wearing
#: this tag must never be re-run or have its directory cleaned: the directory
#: holds the only copy of the imported results and their archived originals.
IMPORTED_JOB_TAG = "imported"
