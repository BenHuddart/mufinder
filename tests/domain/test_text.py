from __future__ import annotations

import pytest

from mufinder.domain.text import folder_phrase, natural_sort_key, plural, space_group_text


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


def test_folder_phrase_agrees_with_the_number_of_folders() -> None:
    assert folder_phrase("kept", ["a"]) == "kept folder 'a'"
    assert folder_phrase("deleted", ["a", "b"]) == "deleted folders 'a' and 'b'"
    assert folder_phrase("deleted", ["a", "b", "c"]) == "deleted folders 'a', 'b' and 'c'"


def test_folder_phrase_counts_the_names_past_its_limit() -> None:
    assert (
        folder_phrase("kept", ["a", "b", "c", "d"], limit=2) == "kept folders 'a', 'b' and 2 more"
    )
    assert folder_phrase("kept", ["a", "b"], limit=2) == "kept folders 'a' and 'b'"
