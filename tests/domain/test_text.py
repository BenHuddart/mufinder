from __future__ import annotations

import pytest

from mufinder.domain.text import plural


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
