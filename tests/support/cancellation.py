"""A counting ``should_cancel`` for tests of calculations that stop between their stages."""

from __future__ import annotations


class CancelAfter:
    """A ``should_cancel`` that answers yes from its ``after``-th call; it counts every call."""

    def __init__(self, after: int | None) -> None:
        self.after = after
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.after is not None and self.calls >= self.after
