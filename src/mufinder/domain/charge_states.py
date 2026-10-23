"""Canonical muon charge-state vocabulary shared across the workflow and app layers.

Both the muon-relaxation adapter (app layer) and the quantum-corrections workflow
need the same notion of "is this muonium?" to apply the open-shell defaults
(spin-polarised SCF, +1 muon seed) consistently. This module is the single source of
truth for the canonical charge-state strings, their classification, and the +1 spin seed
so the two paths cannot drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

POSITIVE_MUON_CHARGE_STATE = "positive_muon"
MUONIUM_CHARGE_STATE = "muonium"

# Neutral muonium is open-shell: mu+ binds one electron (S = 1/2, ~1 muB). Seeding +1 on the muon
# lets a spin-polarised SCF localise the unpaired electron there instead of converging to the wrong
# closed-shell solution; the diamagnetic mu+ has no bound electron and stays 0.
MUONIUM_MUON_SPIN = 1.0

_CHARGE_STATE_ALIASES = {
    "positive": POSITIVE_MUON_CHARGE_STATE,
    "positive_muon": POSITIVE_MUON_CHARGE_STATE,
    "positivemuon": POSITIVE_MUON_CHARGE_STATE,
    "mu+": POSITIVE_MUON_CHARGE_STATE,
    "muonium": MUONIUM_CHARGE_STATE,
    "mu": MUONIUM_CHARGE_STATE,
}


def canonical_charge_state(value: str | None) -> str | None:
    """Return the canonical charge-state key for ``value`` (or ``None`` for ``None``).

    Known aliases (case/separator-insensitive) map to the canonical strings; any other
    non-empty value passes through normalised, matching the historical behaviour of the
    relaxation adapter's ``_charge_state_key``.
    """

    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    return _CHARGE_STATE_ALIASES.get(normalized, normalized)


def is_muonium(value: str | None) -> bool:
    """Return true for the neutral, open-shell muonium charge state."""

    return canonical_charge_state(value) == MUONIUM_CHARGE_STATE


def seed_muon_spins(
    host_spins: Sequence[float] | None,
    *,
    atom_count: int,
    muon_indices: Iterable[int],
    muon_spin: float = MUONIUM_MUON_SPIN,
) -> list[float]:
    """Return an ``atom_count``-length per-atom spin list with the muon atoms seeded.

    Host moments in ``host_spins`` are preserved (copied into the leading positions); a missing
    (``None``) or shorter array is zero-filled, and each index in ``muon_indices`` is set to
    ``muon_spin``. Idempotent if the muon already carries the seed. This is the single source of
    truth for the seed policy so the relaxation and quantum paths cannot diverge on it; each caller
    adapts its own structure type and decides how it identifies the muon (positional vs species).
    """

    spins = [0.0] * atom_count
    if host_spins is not None:
        for index, value in enumerate(host_spins):
            if index < atom_count:
                spins[index] = float(value)
    for index in muon_indices:
        spins[index] = muon_spin
    return spins


#: Persisted ``calc_type`` for each muon charge state. These strings are written
#: into saved projects and are matched by the job-directory resolver, so they are
#: durable vocabulary rather than presentation detail — which is why they live in
#: ``domain`` rather than beside the UI adapter that first defined them.
MUON_RELAXATION_CALC_TYPES: dict[str, str] = {
    POSITIVE_MUON_CHARGE_STATE: "castep_positivemuon_relaxation",
    MUONIUM_CHARGE_STATE: "castep_muonium_relaxation",
}
#: Filesystem-safe directory prefix for each charge state.
MUON_RELAXATION_JOB_PREFIXES: dict[str, str] = {
    POSITIVE_MUON_CHARGE_STATE: "positivemuon",
    MUONIUM_CHARGE_STATE: "muonium",
}
#: CASTEP total ``CHARGE`` for each charge state.
MUON_RELAXATION_CHARGES: dict[str, int] = {
    POSITIVE_MUON_CHARGE_STATE: 1,
    MUONIUM_CHARGE_STATE: 0,
}


def _charge_state_key(value: str) -> str:
    return canonical_charge_state(value) or POSITIVE_MUON_CHARGE_STATE


def muon_charge_state_label(charge_state: str) -> str:
    """Return a user-facing charge-state label."""

    if _charge_state_key(charge_state) == MUONIUM_CHARGE_STATE:
        return "muonium"
    return "positive muon"


def muon_charge_state_prefix(charge_state: str) -> str:
    """Return the filesystem-safe prefix for one muon charge state."""

    key = _charge_state_key(charge_state)
    return MUON_RELAXATION_JOB_PREFIXES.get(
        key, MUON_RELAXATION_JOB_PREFIXES[POSITIVE_MUON_CHARGE_STATE]
    )


def muon_charge_for_state(charge_state: str) -> int:
    """Return the CASTEP total charge for one muon charge state."""

    key = _charge_state_key(charge_state)
    return MUON_RELAXATION_CHARGES.get(key, MUON_RELAXATION_CHARGES[POSITIVE_MUON_CHARGE_STATE])


def muon_relaxation_calc_type(charge_state: str) -> str:
    """Return the persisted calc type for one muon charge state."""

    key = _charge_state_key(charge_state)
    return MUON_RELAXATION_CALC_TYPES.get(
        key, MUON_RELAXATION_CALC_TYPES[POSITIVE_MUON_CHARGE_STATE]
    )


def is_muon_relaxation_calc_type(calc_type: str) -> bool:
    """Return true for any CASTEP muon structural-relaxation calc type."""

    return calc_type in set(MUON_RELAXATION_CALC_TYPES.values())
