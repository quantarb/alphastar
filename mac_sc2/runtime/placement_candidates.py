"""Live legal placement candidates; SC2, never a heuristic, validates each tile."""
from sc2.ids.ability_id import AbilityId
from sc2.position import Point2
from mac_sc2.contracts.placement import CANDIDATE_OFFSETS

async def candidates(bot, ability_id: int, anchor: Point2):
    ability = AbilityId(ability_id)
    points = [anchor.offset(offset) for offset in CANDIDATE_OFFSETS]
    valid = await bot.client._query_building_placement_fast(ability, points)
    return [point for point, ok in zip(points, valid) if ok]
