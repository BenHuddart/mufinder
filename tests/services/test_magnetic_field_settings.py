"""The Magnetic Fields calculation settings value: saved state, ranges, summary and options."""

from __future__ import annotations

from dataclasses import replace

import pytest

from mufinder.science.magnetic_fields import MagneticFieldOptions
from mufinder.services.magnetic_field_settings import (
    MAX_CALCULATION_CELLS,
    MAX_HISTOGRAM_BINS,
    MAX_INCOMMENSURATE_LOOKUP_ANGLES,
    MIN_INCOMMENSURATE_PHASE_SAMPLES,
    MagneticFieldCalculationSettings,
    normalize_phase_sampling_method,
    phase_sampling_label,
)

_CHANGED = MagneticFieldCalculationSettings(
    supercell=(24, 36, 48),
    nangles=8192,
    phase_sample_count=250_000,
    phase_sampling="grid",
    histogram_bins=1024,
    use_approximants=True,
    approximant_tolerance=0.002,
    approximant_max_period=8,
    contact_nnn=4,
    contact_rcont=3.5,
)


def test_defaults_match_the_calculation_options() -> None:
    options = MagneticFieldOptions(**MagneticFieldCalculationSettings().option_values())

    assert options == MagneticFieldOptions()
    assert MagneticFieldCalculationSettings().use_approximants is False


def test_saved_state_round_trips() -> None:
    state = _CHANGED.state_items()

    assert state == {
        "supercell": [24, 36, 48],
        "phase_lookup_angles": 8192,
        "incommensurate_samples": 250_000,
        "phase_sampling": "grid",
        "histogram_bins": 1024,
        "use_commensurate_approximants": True,
        "approximant_tolerance": 0.002,
        "approximant_max_period": 8,
        "contact_nnn": 4,
        "contact_rcont": 3.5,
    }
    assert MagneticFieldCalculationSettings().with_state(state) == _CHANGED


def test_option_values_fill_the_calculation_options() -> None:
    options = MagneticFieldOptions(**_CHANGED.option_values())

    assert options.supercell == (24, 36, 48)
    assert options.nangles == 8192
    assert options.phase_sample_count == 250_000
    assert options.phase_sampling == "grid"
    assert options.histogram_bins == 1024
    assert options.approximant_tolerance == 0.002
    assert options.approximant_max_period == 8
    assert options.nnn == 4
    assert options.rcont == 3.5


def test_counts_out_of_range_are_brought_into_range() -> None:
    settings = MagneticFieldCalculationSettings().with_state(
        {
            "supercell": [0, 600, 7],
            "phase_lookup_angles": 10**9,
            "incommensurate_samples": 3,
            "histogram_bins": 10**12,
        }
    )

    assert settings.supercell == (1, MAX_CALCULATION_CELLS, 7)
    assert settings.nangles == MAX_INCOMMENSURATE_LOOKUP_ANGLES
    assert settings.phase_sample_count == MIN_INCOMMENSURATE_PHASE_SAMPLES
    assert settings.histogram_bins == MAX_HISTOGRAM_BINS


def test_missing_or_unreadable_counts_keep_the_current_value() -> None:
    settings = _CHANGED.with_state(
        {"supercell": [1, 2], "phase_lookup_angles": "many", "histogram_bins": None}
    )

    assert settings.supercell == _CHANGED.supercell
    assert settings.nangles == _CHANGED.nangles
    assert settings.phase_sample_count == _CHANGED.phase_sample_count
    assert settings.phase_sampling == _CHANGED.phase_sampling
    assert settings.histogram_bins == _CHANGED.histogram_bins


def test_unreadable_approximant_and_contact_values_take_the_defaults() -> None:
    defaults = MagneticFieldCalculationSettings()

    settings = _CHANGED.with_state(
        {
            "approximant_tolerance": "wide",
            "approximant_max_period": 500,
            "contact_nnn": float("inf"),
            "contact_rcont": float("nan"),
        }
    )

    assert settings.use_approximants is False
    assert settings.approximant_tolerance == defaults.approximant_tolerance
    assert settings.approximant_max_period == defaults.approximant_max_period
    assert settings.contact_nnn == defaults.contact_nnn
    assert settings.contact_rcont == defaults.contact_rcont


def test_older_saved_keys_are_read() -> None:
    settings = MagneticFieldCalculationSettings().with_state(
        {"nangles": 2048, "phase_sampling": "Monte_Carlo", "contact_rcont": 3.456}
    )

    assert settings.nangles == 2048
    # Before the sample count had its own key it followed the lookup angles.
    assert settings.phase_sample_count == 2048
    assert settings.phase_sampling == "montecarlo"
    assert settings.contact_rcont == 3.46


@pytest.mark.parametrize(
    ("value", "key", "label"),
    [("grid", "grid", "Grid"), ("random", "montecarlo", "Monte Carlo"), ("?", "auto", "Auto")],
)
def test_phase_sampling_spellings(value: str, key: str, label: str) -> None:
    assert normalize_phase_sampling_method(value) == key
    assert phase_sampling_label(value) == label


def test_summary_names_approximants_and_the_contact_shell_only_when_used() -> None:
    plain = replace(_CHANGED, use_approximants=False)

    assert plain.summary(contact_shell=False) == (
        "Settings: cells 24 x 36 x 48; 8192 lookup angles per q; 250000 distribution "
        "samples; Grid sampling; 1024 histogram bins."
    )
    assert _CHANGED.summary(contact_shell=True).endswith(
        "; commensurate approximants within 0.002 r.l.u., period up to 8"
        "; contact shell of 4 nearest magnetic ions within 3.5 Å."
    )
    assert "of 1 nearest magnetic ion within" in replace(_CHANGED, contact_nnn=1).summary(
        contact_shell=True
    )


@pytest.mark.parametrize(
    "change",
    [
        {"supercell": (0, 1, 1)},
        {"nangles": 11},
        {"phase_sample_count": 10**8},
        {"phase_sampling": "random"},
        {"histogram_bins": 9},
        {"approximant_tolerance": 0.0},
        {"approximant_max_period": 65},
        {"contact_nnn": 0},
        {"contact_rcont": 100.5},
    ],
)
def test_settings_out_of_range_are_refused(change) -> None:
    with pytest.raises(ValueError, match="out of range"):
        replace(MagneticFieldCalculationSettings(), **change)


def test_infinite_saved_counts_keep_or_default_instead_of_failing() -> None:
    settings = _CHANGED.with_state(
        {
            "histogram_bins": float("inf"),
            "supercell": [float("inf"), 1, 1],
            "approximant_max_period": float("-inf"),
        }
    )

    assert settings.histogram_bins == _CHANGED.histogram_bins
    assert settings.supercell == _CHANGED.supercell
    assert (
        settings.approximant_max_period == MagneticFieldCalculationSettings().approximant_max_period
    )
