"""~400k factorised policy with AlphaStar-sized (1,026-way) ability heads.

The full command remains compositional: ability + selected entity + target
entity/location + queue flag.  Thus 1,026 is the available ability vocabulary
per race, while the composed action space is much larger.
"""
import torch
from torch import nn

RACES = ("Terran", "Protoss", "Zerg")
ABILITY_VOCAB = 1026
MACRO_VOCAB = 7

class AlphaStarSizedCompactPolicy(nn.Module):
    def __init__(self, features=24, width=64):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(features,width),nn.LayerNorm(width),nn.GELU())
        block=nn.TransformerEncoderLayer(width,4,width*4,batch_first=True,norm_first=True)
        self.torso=nn.TransformerEncoder(block,2)
        self.race=nn.Embedding(3,width)
        self.history=nn.GRU(width,width,num_layers=2,batch_first=True)
        self.history_in=nn.Linear(features,width)
        self.fuse=nn.Sequential(nn.Linear(width*2,width),nn.GELU(),nn.LayerNorm(width))
        self.ability=nn.ModuleDict({r:nn.Linear(width,ABILITY_VOCAB) for r in RACES})
        self.macro=nn.ModuleDict({r:nn.Linear(width,MACRO_VOCAB) for r in RACES})
        self.select_query=nn.Linear(width,width); self.target_query=nn.Linear(width,width)
        self.location=nn.Linear(width,2); self.queue=nn.Linear(width,2); self.value=nn.Linear(width,1)
    def forward(self, entities, mask, history, race_ids):
        if race_ids.unique().numel()!=1: raise ValueError('one race per batch')
        race=RACES[int(race_ids[0])]; e=self.torso(self.entity(entities)+self.race(race_ids)[:,None,:],src_key_padding_mask=mask)
        valid=(~mask).unsqueeze(-1); pooled=(e*valid).sum(1)/valid.sum(1).clamp_min(1)
        h,_=self.history(self.history_in(history)); z=self.fuse(torch.cat((pooled,h[:,-1]),-1))
        return {'ability':self.ability[race](z),'macro':self.macro[race](z),'select_entity':torch.einsum('bd,bnd->bn',self.select_query(z),e).masked_fill(mask,-1e9),'target_entity':torch.einsum('bd,bnd->bn',self.target_query(z),e).masked_fill(mask,-1e9),'target_location':torch.tanh(self.location(z)),'queued':self.queue(z),'value':self.value(z).squeeze(-1)}
