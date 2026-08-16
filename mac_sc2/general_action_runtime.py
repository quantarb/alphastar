"""Shared live-side helpers for the general action model.

No behaviour-specific branches: actor and target candidates are selected from
the model's role/type fields, then the generic SC2 executor validates ability
availability and target mode.
"""
from sc2.ids.unit_typeid import UnitTypeId


def actor_candidates(bot, role):
    all_units = bot.units | bot.structures
    if role == "worker": return bot.workers
    if role == "production": return bot.structures
    if role == "transport": return all_units.filter(lambda u: u.cargo_max > 0)
    if role == "combat": return bot.units.filter(lambda u: u.can_attack and not u.is_worker)
    return all_units


def target_candidates(bot, target_type):
    """Type-conditioned pointer candidates from actual current entities."""
    pool = bot.units | bot.structures | bot.enemy_units | bot.enemy_structures | bot.vespene_geyser
    compact = ''.join(ch.lower() for ch in target_type if ch.isalnum())
    return pool.filter(lambda u: compact in ''.join(ch.lower() for ch in u.type_id.name if ch.isalnum()))


def point_from_normalized(bot, point):
    area = bot.game_info.playable_area
    x = area.x + (float(point[0]) + 1) * .5 * area.width
    y = area.y + (float(point[1]) + 1) * .5 * area.height
    from sc2.position import Point2
    return Point2((x, y))
