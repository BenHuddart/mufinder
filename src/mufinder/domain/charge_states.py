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
