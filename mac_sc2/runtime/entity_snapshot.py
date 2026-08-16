"""Live encoder for the shared entity-snapshot contract."""
import zlib, torch
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS

def encode(bot):
    home=bot.townhalls.first.position
    # Friendly and currently visible enemy entities use the exact same schema
    # as replay reconstruction.  Combat heads therefore see enemy composition.
    units=sorted(list(bot.units|bot.structures|bot.enemy_units|bot.enemy_structures),key=lambda u:u.tag)[:ENTITY_SLOTS]
    worker_tags={u.tag for u in bot.workers}; x=torch.zeros(ENTITY_SLOTS,8);mask=torch.ones(ENTITY_SLOTS,dtype=torch.bool)
    for i,u in enumerate(units):
        x[i]=torch.tensor([zlib.crc32(u.type_id.name.encode())%65535/65535,(u.position.x-home.x)/64,(u.position.y-home.y)/64,float(u.is_mine),u.health/max(u.health_max,1),u.build_progress,float(u.is_flying),float(u.tag in worker_tags)])
        mask[i]=False
    return x,mask,units
