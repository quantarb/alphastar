"""Live legality mask and execution for the learned repair pointer policy."""
import torch
from sc2.ids.ability_id import AbilityId
from mac_sc2.contracts.repair import RepairAction

async def issue_learned_repair(bot, model, entities, mask, units):
    """Use model-ranked eligible SCVs and damaged friendly targets only."""
    workers={u.tag for u in bot.workers}; damaged=torch.tensor([u.health < u.health_max for u in units],dtype=torch.bool)
    if not workers or not damaged.any(): return False
    available=await bot.get_available_abilities([u for u in units if u.tag in workers])
    usable={u.tag for u,abilities in zip([u for u in units if u.tag in workers],available) if AbilityId(RepairAction().ability_id) in abilities}
    actor_mask=torch.tensor([u.tag not in usable for u in units],dtype=torch.bool)|mask[:len(units)]
    target_mask=(~damaged)|mask[:len(units)]
    if actor_mask.all() or target_mask.all(): return False
    with torch.no_grad(): actors,targets=model(entities[None],mask[None])
    actor=int(actors[0,:len(units)].masked_fill(actor_mask,-1e9).argmax());target=int(targets[0,:len(units)].masked_fill(target_mask,-1e9).argmax())
    if actor_mask[actor] or target_mask[target]: return False
    units[actor](AbilityId(RepairAction().ability_id),units[target]);print('repair',units[actor].tag,'->',units[target].tag,flush=True);return True
