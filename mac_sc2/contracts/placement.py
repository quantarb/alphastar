"""Shared contract for learned placement: replay labels select live legal tiles."""
from dataclasses import dataclass

from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS, ENTITY_FEATURES
CANDIDATE_OFFSETS = tuple((x, y) for radius in (0, 4, 8, 12, 16, 20)
                          for x in range(-radius, radius + 1, 4)
                          for y in range(-radius, radius + 1, 4)
                          if max(abs(x), abs(y)) == radius)

@dataclass(frozen=True)
class PlacementLabel:
    race: str
    ability: str
    point: tuple[float, float]
    frame: int
