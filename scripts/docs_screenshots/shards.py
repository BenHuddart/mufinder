"""Share the screenshots out between render workers or CI shards by measured duration.

A scene's cost is its window, its drive and its grab; the first scene of a
project also builds that project, once per process. :func:`partition` keeps
the scenes of one project together, so each share builds a project once, and
splits a project's scenes only when they alone exceed an even share. The
pieces go to the least-loaded share, longest first.

``durations.json`` beside this module records each scene's seconds from a
single-process run, ``--store-durations``; only their ratios matter. A scene
missing from it counts as the median. The module imports no Qt.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
import json
import math
from pathlib import Path
import statistics

DURATIONS_PATH = Path(__file__).resolve().with_name("durations.json")
# The weight of every scene when no duration is known at all.
_DEFAULT_SECONDS = 1.0
# The least a scene weighs, the precision durations.json records: a scene
# recorded as 0.0 still counts, so the shares never collapse into one.
_MIN_SECONDS = 0.1


def load_durations(path: Path = DURATIONS_PATH) -> dict[str, float]:
    """The recorded seconds per scene, or an empty mapping if ``path`` does not exist."""

    if not path.exists():
        return {}
    return {str(name): float(seconds) for name, seconds in json.loads(path.read_text(encoding="utf-8")).items()}


def store_durations(measured: Mapping[str, float], known: Sequence[str], path: Path = DURATIONS_PATH) -> None:
    """Merge ``measured`` into ``path``, keeping only the ``known`` scene names, in their order."""

    durations = load_durations(path)
    durations.update(measured)
    kept = {name: round(durations[name], 1) for name in known if name in durations}
    path.write_text(json.dumps(kept, indent=2) + "\n", encoding="utf-8")


def parse_shard(text: str) -> tuple[int, int]:
    """``"K/N"`` as ``(K, N)`` with ``1 <= K <= N``; raises ``ValueError`` otherwise."""

    index, _, count = text.partition("/")
    try:
        shard, shards = int(index), int(count)
    except ValueError:
        raise ValueError(f"expected K/N, such as 2/4, not {text!r}") from None
    if not 1 <= shard <= shards:
        raise ValueError(f"shard {shard} is not between 1 and {shards}")
    return shard, shards


def partition(
    names: Sequence[str],
    projects: Mapping[str, Hashable],
    durations: Mapping[str, float],
    count: int,
) -> list[list[str]]:
    """Split ``names`` into ``count`` shares of about equal duration, each in ``names`` order.

    ``projects`` maps each name to its project, the key that keeps scenes
    together. A share can be empty when there are fewer pieces than shares.
    """

    if count < 1:
        raise ValueError("the share count must be at least 1")
    known = [durations[name] for name in names if name in durations]
    fallback = statistics.median(known) if known else _DEFAULT_SECONDS
    weight = {name: max(durations.get(name, fallback), _MIN_SECONDS) for name in names}
    target = sum(weight.values()) / count

    groups: dict[Hashable, list[str]] = {}
    for name in names:
        groups.setdefault(projects[name], []).append(name)
    pieces = [piece for group in groups.values() for piece in _split(group, weight, target)]

    order = {name: index for index, name in enumerate(names)}
    loads = [0.0] * count
    shares: list[list[str]] = [[] for _ in range(count)]
    # sorted is stable, so equal pieces keep their registry order.
    for piece in sorted(pieces, key=lambda piece: -sum(weight[name] for name in piece)):
        share = loads.index(min(loads))
        shares[share].extend(piece)
        loads[share] += sum(weight[name] for name in piece)
    return [sorted(share, key=order.__getitem__) for share in shares]


def _split(group: list[str], weight: Mapping[str, float], target: float) -> list[list[str]]:
    """``group`` as consecutive pieces no longer than about ``target`` each."""

    total = sum(weight[name] for name in group)
    pieces = max(1, math.ceil(total / target - 1e-9))
    if pieces == 1:
        return [group]
    size = total / pieces
    result: list[list[str]] = [[]]
    filled = 0.0
    for name in group:
        # Start the next piece once this one has its share, keeping the last
        # piece open for whatever remains.
        if result[-1] and filled + weight[name] / 2 > size * len(result) and len(result) < pieces:
            result.append([])
        result[-1].append(name)
        filled += weight[name]
    return result
