"""One shared foundation with macro, placement, and repair task heads."""
import torch
from torch import nn
from mac_sc2.contracts.semantic import ACTOR_ROLES,FAMILIES,PAYLOAD_ROLES,RACES,TARGET_KINDS

class UnifiedPolicy(nn.Module):
    def __init__(self,width=224):
        super().__init__()
        # Exact name/layout of the semantic all-patch foundation: load it
        # directly, then add task heads without loading separate models.
        self.backbone=nn.Sequential(nn.Linear(17,width),nn.GELU(),nn.LayerNorm(width),nn.Linear(width,width),nn.GELU())
        self.heads=nn.ModuleDict({race:nn.ModuleDict({'actor':nn.Linear(width,len(ACTOR_ROLES)),'family':nn.Linear(width,len(FAMILIES)),'payload':nn.Linear(width,len(PAYLOAD_ROLES)),'target':nn.Linear(width,len(TARGET_KINDS)),'queued':nn.Linear(width,2)}) for race in RACES})
        self.entity=nn.Sequential(nn.Linear(8,width),nn.GELU(),nn.Linear(width,width),nn.GELU())
        self.placement=nn.Sequential(nn.Linear(width+2,width),nn.GELU(),nn.Linear(width,1))
        self.repair_actor=nn.Linear(width,width,bias=False);self.repair_target=nn.Linear(width,width,bias=False)
    def macro(self,state,race):
        h=self.backbone(state);out={}
        for key in ('actor','family','payload','target','queued'):
            all_logits=torch.stack([self.heads[name][key](h) for name in RACES],1);out[key]=all_logits[torch.arange(state.size(0),device=state.device),race]
        return h,out
    def entity_context(self,state,entities,mask):
        h,_=self.macro(state,torch.zeros(state.size(0),dtype=torch.long,device=state.device));e=self.entity(entities);z=e.masked_fill(mask.unsqueeze(-1),0).sum(1)/(~mask).sum(1,keepdim=True).clamp(min=1);return h+z,e
    def placement_scores(self,state,entities,mask,candidates):
        z,_=self.entity_context(state,entities,mask);z=z[:,None].expand(-1,candidates.size(1),-1);return self.placement(torch.cat((z,candidates),-1)).squeeze(-1)
    def repair_scores(self,state,entities,mask):
        z,e=self.entity_context(state,entities,mask);return (torch.einsum('bd,bnd->bn',self.repair_actor(z),e).masked_fill(mask,-1e9),torch.einsum('bd,bnd->bn',self.repair_target(z),e).masked_fill(mask,-1e9))
    def load_foundation(self,state_dict):
        missing,unexpected=self.load_state_dict(state_dict,strict=False)
        if unexpected or any(not key.startswith(('entity','placement','repair_')) for key in missing):raise ValueError('incompatible macro foundation')
