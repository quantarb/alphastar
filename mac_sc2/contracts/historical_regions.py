"""Shared coarse world-target labels for historical replay auxiliary tasks."""
from __future__ import annotations

REGION_BINS = 8
NO_TARGET_REGION = REGION_BINS * REGION_BINS
REGION_CLASSES = NO_TARGET_REGION + 1


def region(location: tuple[float, float] | None) -> int:
    if location is None:
        return NO_TARGET_REGION
    x, y = location
    return min(REGION_BINS - 1, max(0, int(x // 32))) + REGION_BINS * min(REGION_BINS - 1, max(0, int(y // 32)))
