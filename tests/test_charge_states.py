from __future__ import annotations

from mufinder.domain.charge_states import (
    MUONIUM_CHARGE_STATE,
    MUONIUM_MUON_SPIN,
    POSITIVE_MUON_CHARGE_STATE,
    canonical_charge_state,
    is_muonium,
    seed_muon_spins,
)


def test_canonical_charge_state_normalises_aliases() -> None:
    for alias in ("positive", "positive_muon", "PositiveMuon", "mu+", "positive-muon"):
        assert canonical_charge_state(alias) == POSITIVE_MUON_CHARGE_STATE
    for alias in ("muonium", "Mu", " MUONIUM "):
        assert canonical_charge_state(alias) == MUONIUM_CHARGE_STATE
    # None passes through; unknown non-empty values normalise but pass through unchanged.
    assert canonical_charge_state(None) is None
    assert canonical_charge_state("custom state") == "custom_state"


def test_is_muonium_only_true_for_muonium() -> None:
    assert is_muonium("muonium") is True
    assert is_muonium("Mu") is True
    assert is_muonium("positive_muon") is False
    assert is_muonium(None) is False


def test_seed_muon_spins_creates_zero_array_and_seeds_muon() -> None:
    # No host moments: a fresh zero array with only the muon (last atom) seeded.
    spins = seed_muon_spins(None, atom_count=3, muon_indices=(2,))
    assert spins == [0.0, 0.0, MUONIUM_MUON_SPIN]


def test_seed_muon_spins_preserves_host_moments() -> None:
    # Magnetic host: existing moments are kept, the muon index is overwritten with the seed.
    spins = seed_muon_spins([1.0, -1.0, 0.0], atom_count=3, muon_indices=(2,))
    assert spins == [1.0, -1.0, MUONIUM_MUON_SPIN]


def test_seed_muon_spins_zero_fills_when_host_shorter_than_atom_count() -> None:
    # Relaxation appends the muon, so host_spins is one shorter than the final atom count.
    spins = seed_muon_spins([1.0, -1.0], atom_count=3, muon_indices=(2,))
    assert spins == [1.0, -1.0, MUONIUM_MUON_SPIN]


def test_seed_muon_spins_is_idempotent() -> None:
    seeded = seed_muon_spins([0.0, 0.0, MUONIUM_MUON_SPIN], atom_count=3, muon_indices=(2,))
    assert seeded == seed_muon_spins(seeded, atom_count=3, muon_indices=(2,))


def test_seed_muon_spins_honours_custom_seed_for_diamagnetic_muon() -> None:
    # The diamagnetic mu+ appends a 0.0 alongside a magnetic host's moments.
    spins = seed_muon_spins([2.0, -2.0], atom_count=3, muon_indices=(2,), muon_spin=0.0)
    assert spins == [2.0, -2.0, 0.0]
