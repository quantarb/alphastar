"""Live encoder for the rich transformer entity-token contract."""
from __future__ import annotations

import torch

from mac_sc2.contracts.rich_transformer_snapshot import ENTITY_SLOTS


def encode(bot):
    """Return cache-aligned rows and their live Unit identities for pointers."""
    units = sorted(list(bot.units | bot.structures | bot.enemy_units | bot.enemy_structures), key=lambda unit: unit.tag)[:ENTITY_SLOTS]
    worker_tags = {unit.tag for unit in bot.workers}
    tensor = torch.zeros(ENTITY_SLOTS, 13, dtype=torch.float32)
    padding = torch.ones(ENTITY_SLOTS, dtype=torch.bool)
    for index, unit in enumerate(units):
        # ``Unit.orders`` eagerly resolves ability IDs through python-sc2's
        # static game-data table.  Current clients can report an order (for
        # example ability 4135) which is absent from that table; resolution
        # raises KeyError and used to terminate the entire live match.  The
        # snapshot contract stores the raw numeric ID, so read it directly.
        raw_orders = unit._proto.orders
        order = int(raw_orders[0].ability_id) if raw_orders else 0
        tensor[index] = torch.tensor((unit.tag, int(unit.type_id.value), 4 if unit.is_mine else 1,
                                      unit.position.x, unit.position.y, unit.health, unit.health_max,
                                      unit.shield, unit.energy, unit.build_progress, 0,
                                      int(unit.is_flying), order), dtype=torch.float32)
        padding[index] = False
    return tensor, padding, units
