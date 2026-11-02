from __future__ import annotations

import pytest

from mufinder.domain.text import (
    KeptFolder,
    folder_outcome_status,
    folder_phrase,
    kept_folders_message,
    natural_sort_key,
    plural,
    space_group_text,
)


@pytest.mark.parametrize(
    ("n", "expected"),
    [(0, "0 jobs"), (1, "1 job"), (2, "2 jobs"), (12, "12 jobs"), (-1, "-1 jobs")],
)
def test_plural_adds_an_s_unless_the_count_is_one(n: int, expected: str) -> None:
    assert plural(n, "job") == expected


def test_plural_uses_the_given_plural_form() -> None:
    assert plural(1, "analysis", "analyses") == "1 analysis"
    assert plural(3, "analysis", "analyses") == "3 analyses"
    assert plural(0, "directory", "directories") == "0 directories"


def test_plural_makes_a_following_verb_agree() -> None:
    assert plural(1, "job was", "jobs were") == "1 job was"
    assert plural(4, "job was", "jobs were") == "4 jobs were"


def test_natural_sort_key_compares_numbers_as_numbers_and_ignores_case() -> None:
    labels = ["site_10", "Site_2", "site_1", "mu", "site_2b"]

    assert sorted(labels, key=natural_sort_key) == ["mu", "site_1", "Site_2", "site_2b", "site_10"]


def test_natural_sort_key_keeps_non_decimal_digits_as_text() -> None:
    assert natural_sort_key("Mu¹²_3") == ((1, "mu¹²_"), (0, 3))


@pytest.mark.parametrize(
    ("symbol", "expected"),
    [("P4_2/mnm", "P4₂/mnm"), ("P6_3mc", "P6₃mc"), ("Fm-3m", "Fm-3m"), ("P2_12_12_1", "P2₁2₁2₁")],
)
def test_space_group_text_writes_screw_axes_as_subscripts(symbol: str, expected: str) -> None:
    assert space_group_text(symbol) == expected


def test_plain_error_drops_the_errno_and_the_path() -> None:
    import errno

    from mufinder.domain.text import plain_error

    assert plain_error(PermissionError(errno.EACCES, "Permission denied", "/a/b")) == (
        "permission denied"
    )
    assert plain_error(FileNotFoundError(errno.ENOENT, "No such file", "/a/b")) == (
        "it is no longer there"
    )
    assert plain_error(OSError(errno.ENAMETOOLONG, "File name too long", "/a/b")) == (
        "the path is too long"
    )
    assert plain_error(OSError(errno.ENOSPC, "No space left on device", "/a/b")) == (
        "no space left on device"
    )
    assert plain_error(ValueError("Unknown block FOO.")) == "Unknown block FOO"


def test_folder_phrase_agrees_with_the_number_of_folders_and_sorts_them() -> None:
    assert folder_phrase("kept", ["a"]) == "kept folder 'a'"
    assert folder_phrase("deleted", ["b", "a"]) == "deleted folders 'a' and 'b'"
    assert folder_phrase("deleted", ["Mu_10", "Mu_2", "Mu_1"]) == (
        "deleted folders 'Mu_1', 'Mu_2' and 'Mu_10'"
    )


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        ({}, "Removed 1 job"),
        ({"deleted": 1}, "Removed 1 job; deleted 1 folder"),
        ({"kept": 1}, "Removed 1 job; kept 1 folder"),
        ({"deleted": 1, "kept": 1}, "Removed 1 job; deleted 1 folder; kept 1"),
        ({"deleted": 7, "kept": 3}, "Removed 1 job; deleted 7 folders; kept 3"),
        (
            {"deleted": 2, "archived": 1, "kept": 1},
            "Removed 1 job; deleted 2 folders; archived 1; kept 1",
        ),
        ({"cleaned": 1}, "Removed 1 job; cleaned 1 folder"),
    ],
)
def test_folder_outcome_status_gives_counts_with_one_noun(counts, expected) -> None:
    assert folder_outcome_status("Removed 1 job", **counts) == expected


def test_kept_folders_message_says_why_for_one_folder() -> None:
    reasons = ("another job uses it", "other jobs use them")

    assert kept_folders_message([KeptFolder("a", reasons)]) == (
        "MuFinder kept 1 folder because another job uses it.",
        "Open Details to see which one.",
        "- 'a': another job uses it.",
    )


def test_kept_folders_message_sorts_and_agrees_for_several() -> None:
    reasons = ("another job uses it", "other jobs use them")
    kept = [KeptFolder(name, reasons) for name in ("Mu_10", "Mu_2")]

    assert kept_folders_message(kept) == (
        "MuFinder kept 2 folders because other jobs use them.",
        "Open Details to see which ones.",
        "- 'Mu_2': another job uses it.\n- 'Mu_10': another job uses it.",
    )


def test_kept_folders_message_with_mixed_reasons_says_each_thing_once() -> None:
    kept = [KeptFolder("a", ("one", "many")), KeptFolder("b", ("other", "others"))]

    assert kept_folders_message(kept) == (
        "MuFinder kept 2 folders.",
        "Open Details to see why.",
        "- 'a': one.\n- 'b': other.",
    )


def test_kept_folders_message_with_a_reason_only_for_one_folder() -> None:
    kept = [KeptFolder("a", ("one", "")), KeptFolder("b", ("one", ""))]

    assert kept_folders_message(kept).summary == "MuFinder kept 2 folders."
