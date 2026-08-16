"""Generic 4.9.2 executor for a factorised general-action policy.

There are deliberately no branches such as ``if repair`` or ``if upgrade``.
Legality is derived from the selected live units' available abilities and the
SC2 API's declared target mode.
"""
from sc2.ids.ability_id import AbilityId


NONE, POINT, UNIT, POINT_OR_UNIT, POINT_OR_NONE = 1, 2, 3, 4, 5


def expected_target_kind(bot, ability: AbilityId) -> str:
    target = bot.game_data.abilities[ability.value]._proto.target
    if target == NONE:
        return "none"
    if target == UNIT:
        return "unit"
    if target in (POINT, POINT_OR_NONE):
        return "point"
    if target == POINT_OR_UNIT:
        return "point_or_unit"
    raise ValueError(f"Unknown SC2 target mode {target} for {ability}")


async def execute(bot, actors, ability_id: int, target, queued: bool) -> int:
    """Issue an arbitrary live-legal raw ability to compatible selected units.

    ``actors`` and ``target`` are model-selected live entities/points.  The
    function rechecks availability per unit and target compatibility immediately
    before issuing commands; invalid choices become no-ops rather than hidden
    fallback behaviour.
    """
    ability = AbilityId(int(ability_id))
    actors = list(actors)
    if not actors:
        return 0
    mode = expected_target_kind(bot, ability)
    is_point = hasattr(target, "x") and hasattr(target, "y") and not hasattr(target, "tag")
    if mode == "unit" and not hasattr(target, "tag"):
        return 0
    if mode == "point" and not is_point:
        return 0
    if mode == "none":
        target = None
    if mode == "point_or_unit" and target is None:
        return 0
    available = await bot.get_available_abilities(actors)
    sent = 0
    for unit, abilities in zip(actors, available):
        if ability not in abilities:
            continue
        unit(ability, target=target, queue=queued)
        sent += 1
    return sent
