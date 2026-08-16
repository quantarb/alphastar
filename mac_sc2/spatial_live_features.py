"""Shared live SC2 entity encoding for spatial policy training and play."""
import torch

from spatial_tactical_policy import ENTITY_FEATURES, MAX_ENTITIES


def encode(bot, army_types):
    """Return own-first entity tokens and their matching live Unit objects.

    Own combat units, workers, structures, visible enemy units, and visible
    enemy structures share one positional schema.  The runner can therefore
    mask actor pointers to own entities while targets may point at enemies.
    """
    own = list(bot.units.of_type(army_types)) + list(bot.workers) + list(bot.structures)
    enemy = list(bot.enemy_units) + list(bot.enemy_structures)
    units = (own + enemy)[:MAX_ENTITIES]
    own_count = min(len(own), MAX_ENTITIES)
    tokens = torch.zeros(MAX_ENTITIES, ENTITY_FEATURES)
    home = bot.townhalls.first.position if bot.townhalls else bot.start_location
    for index, unit in enumerate(units):
        pos = unit.position
        tokens[index] = torch.tensor([
            1.0 if index < own_count else 0.0,
            min(int(unit.type_id.value), 2047) / 2047,
            min(max(pos.x / 200, 0), 1), min(max(pos.y / 200, 0), 1),
            float(unit.health_percentage), float(unit.shield_percentage),
            min(float(unit.weapon_cooldown) / 15, 1), float(unit.is_flying),
            float(unit.is_structure), float(unit.is_worker),
            float(unit.can_attack), min(unit.distance_to(home) / 200, 1),
            min(unit.radius / 4, 1), min((unit.health + unit.shield) / 500, 1),
            min(unit.health_max / 500, 1), float(unit.is_ready),
            float(unit.is_idle), min(len(own) / 64, 1), min(len(enemy) / 64, 1), 1.0,
        ])
    mask = torch.ones(MAX_ENTITIES, dtype=torch.bool); mask[:len(units)] = False
    actor_mask = torch.ones(MAX_ENTITIES, dtype=torch.bool); actor_mask[:own_count] = False
    target_mask = torch.ones(MAX_ENTITIES, dtype=torch.bool); target_mask[own_count:len(units)] = False
    return tokens, mask, actor_mask, target_mask, units, own_count
