"""Patch/race-specific historical tactics heads over the shared V2 trunk."""
from __future__ import annotations

import re
import torch
from torch import nn

from mac_sc2.architectures.rich_transformer import RACES, RichEntityTransformerPolicy
from mac_sc2.architectures.historical_replay_adapter import HistoricalReplayInputAdapter
from mac_sc2.contracts.historical_tactics import REGION_CLASSES
from mac_sc2.data.historical_tactics import SCALAR_FIELDS


def _key(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", task)


class HistoricalTacticsMTL(nn.Module):
    """Offline heads, with V2 entity/temporal/decoder modules shared by reference."""
    research_only = True

    def __init__(self, live_policy: RichEntityTransformerPolicy, task_vocabs: dict[str, tuple[str, ...]],
                 input_adapter: HistoricalReplayInputAdapter | None = None):
        super().__init__()
        self.live_policy, self.task_vocabs = live_policy, {k: tuple(v) for k, v in task_vocabs.items()}
        self.tasks = tuple(sorted(self.task_vocabs)); self.task_index = {k:i for i,k in enumerate(self.tasks)}
        self.patches = tuple(sorted({task.split('/',1)[0] for task in self.tasks})); self.patch_index={k:i for i,k in enumerate(self.patches)}
        width=live_policy.width
        self.patch=nn.Embedding(len(self.patches),width); self.task=nn.Embedding(len(self.tasks),width)
        self.input_adapter=input_adapter or HistoricalReplayInputAdapter(width)
        self.history=nn.ModuleDict({_key(task):nn.Embedding(len(vocab)+1,width,padding_idx=0) for task,vocab in self.task_vocabs.items()})
        self.action_heads=nn.ModuleDict({_key(task):nn.Linear(width,len(vocab)) for task,vocab in self.task_vocabs.items()})
        self.region_heads=nn.ModuleDict({_key(task):nn.Linear(width,REGION_CLASSES) for task in self.task_vocabs})

    def logits(self, scalars: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor, history: torch.Tensor, task: str, mmr: torch.Tensor | None = None):
        if task not in self.task_index: raise KeyError(task)
        batch=scalars.size(0); patch,race,_=task.split('/',2)
        if mmr is None: mmr=torch.zeros(batch,1,dtype=scalars.dtype,device=scalars.device)
        cls=(self.live_policy.cls.expand(batch,-1,-1)+self.input_adapter.scalar(scalars).unsqueeze(1)+self.input_adapter.skill(mmr.view(batch,1)/7000).unsqueeze(1)+
             self.live_policy.race.weight[RACES.index(race)].view(1,1,-1)+self.patch.weight[self.patch_index[patch]].view(1,1,-1)+self.task.weight[self.task_index[task]].view(1,1,-1))
        memory=torch.cat((cls,self.input_adapter.entity_tokens(entities),self.live_policy.spatial.unsqueeze(0).expand(batch,-1,-1)),1)
        mask=torch.cat((torch.zeros(batch,1,dtype=torch.bool,device=padding.device),padding,torch.zeros(batch,self.live_policy.spatial_tokens,dtype=torch.bool,device=padding.device)),1)
        temporal=self.live_policy.temporal_encoder(self.history[_key(task)](history))
        memory=torch.cat((memory,temporal),1); mask=torch.cat((mask,torch.zeros(temporal.shape[:2],dtype=torch.bool,device=padding.device)),1)
        memory=self.live_policy.entity_encoder(memory,src_key_padding_mask=mask)
        hidden=self.live_policy._decode(memory,[self.live_policy.bos.expand(batch,-1,-1)])
        return self.action_heads[_key(task)](hidden),self.region_heads[_key(task)](hidden)
