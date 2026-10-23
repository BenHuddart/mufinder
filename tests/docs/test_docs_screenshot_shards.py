"""How ``docs_screenshots.shards`` shares the screenshots out between workers and CI shards.

The module imports no Qt, so it is imported here directly.
"""

from __future__ import annotations

from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from docs_screenshots.shards import load_durations, parse_shard, partition, store_durations  # noqa: E402


def _names(count: int, prefix: str = "scene") -> list[str]:
    return [f"{prefix}{index}" for index in range(count)]


def test_every_scene_lands_in_exactly_one_share_in_registry_order() -> None:
    names = _names(12)
    projects = {name: index % 5 for index, name in enumerate(names)}
    durations = {name: float(index + 1) for index, name in enumerate(names)}

    shares = partition(names, projects, durations, 4)

    assert len(shares) == 4
    assert sorted(name for share in shares for name in share) == sorted(names)
    for share in shares:
        assert share == sorted(share, key=names.index)


def test_the_scenes_of_one_project_stay_together_when_they_fit() -> None:
    names = _names(8)
    projects = {name: index // 2 for index, name in enumerate(names)}
    durations = dict.fromkeys(names, 1.0)

    shares = partition(names, projects, durations, 4)

    assert sorted(shares) == sorted([names[index : index + 2] for index in range(0, 8, 2)])


def test_a_project_longer_than_an_even_share_is_split_into_consecutive_pieces() -> None:
    large = _names(6, "large")
    small = _names(2, "small")
    names = large + small
    projects = {name: name.rstrip("0123456789") for name in names}
    durations = {**dict.fromkeys(large, 3.0), **dict.fromkeys(small, 1.0)}

    shares = partition(names, projects, durations, 2)

    loads = [sum(durations[name] for name in share) for share in shares]
    # 18 s of one project over two 10 s shares: two 9 s pieces, then the rest.
    assert sorted(loads) == [9.0, 11.0]
    large_shares = [[name for name in share if name in large] for share in shares]
    assert sorted(large_shares) == [large[:3], large[3:]]


def test_the_shares_balance_measured_durations_longest_first() -> None:
    names = ["long", "medium", "short1", "short2", "short3"]
    projects = {name: name for name in names}
    durations = {"long": 6.0, "medium": 4.0, "short1": 2.0, "short2": 2.0, "short3": 2.0}

    shares = partition(names, projects, durations, 2)

    assert sorted(sum(durations[name] for name in share) for share in shares) == [8.0, 8.0]


def test_a_scene_without_a_recorded_duration_counts_as_the_median() -> None:
    names = ["a", "b", "c", "new"]
    projects = {name: name for name in names}
    durations = {"a": 1.0, "b": 5.0, "c": 9.0}

    shares = partition(names, projects, durations, 2)

    # The new scene weighs 5 s, the same as "b", so the two shares weigh 10 s each.
    assert sorted(shares) == [["a", "c"], ["b", "new"]]


def test_with_no_recorded_durations_every_scene_weighs_the_same() -> None:
    names = _names(6)
    projects = {name: name for name in names}

    shares = partition(names, projects, {}, 3)

    assert [len(share) for share in shares] == [2, 2, 2]


def test_scenes_recorded_as_zero_seconds_still_spread_across_the_shares() -> None:
    names = _names(8)
    projects = {name: name for name in names}

    shares = partition(names, projects, dict.fromkeys(names, 0.0), 4)

    assert [len(share) for share in shares] == [2, 2, 2, 2]


def test_more_shares_than_scenes_leaves_some_shares_empty() -> None:
    shares = partition(["a", "b"], {"a": 1, "b": 2}, {}, 4)

    assert sorted(len(share) for share in shares) == [0, 0, 1, 1]


def test_a_share_count_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        partition(["a"], {"a": 1}, {}, 0)


@pytest.mark.parametrize(("text", "expected"), [("1/4", (1, 4)), ("4/4", (4, 4)), ("1/1", (1, 1))])
def test_a_shard_is_written_k_of_n(text: str, expected: tuple[int, int]) -> None:
    assert parse_shard(text) == expected


@pytest.mark.parametrize(("text", "message"), [("0/4", "between"), ("5/4", "between"), ("2", "K/N"), ("a/b", "K/N")])
def test_a_malformed_or_out_of_range_shard_is_refused(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_shard(text)


def test_storing_durations_merges_rounds_and_drops_unknown_scenes(tmp_path: Path) -> None:
    path = tmp_path / "durations.json"
    path.write_text(json.dumps({"kept": 2.0, "removed": 3.0, "updated": 4.0}), encoding="utf-8")

    store_durations({"updated": 1.234, "added": 5.678}, ["added", "kept", "updated"], path)

    assert load_durations(path) == {"added": 5.7, "kept": 2.0, "updated": 1.2}
    assert list(json.loads(path.read_text(encoding="utf-8"))) == ["added", "kept", "updated"]


def test_a_missing_durations_file_reads_as_no_durations(tmp_path: Path) -> None:
    assert load_durations(tmp_path / "durations.json") == {}
