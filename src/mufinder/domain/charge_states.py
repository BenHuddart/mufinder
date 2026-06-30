"""Canonical muon charge-state vocabulary shared across the workflow and app layers.

Both the muon-relaxation adapter (app layer) and the quantum-corrections workflow
need the same notion of "is this muonium?" to apply the open-shell defaults
(spin-polarised SCF, +1 muon seed) consistently. This module is the single source of
truth for the canonical charge-state strings and their classification so the two paths
cannot drift apart.
"""

from __future__ import annotations

POSITIVE_MUON_CHARGE_STATE = "positive_muon"
MUONIUM_CHARGE_STATE = "muonium"

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


def is_positive_muon(value: str | None) -> bool:
    """Return true for the diamagnetic positive-muon charge state."""

    return canonical_charge_state(value) == POSITIVE_MUON_CHARGE_STATE
