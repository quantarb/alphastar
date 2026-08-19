"""Research-only build/tech heads sharing the V2 entity Transformers."""
from __future__ import annotations

import re

import torch
from torch import nn

from mac_sc2.architectures.rich_transformer import RACES, RichEntityTransformerPolicy
from mac_sc2.architectures.historical_replay_adapter import HistoricalReplayInputAdapter
from mac_sc2.contracts.historical_build_tech import validate_task_vocabs
from mac_sc2.contracts.historical_regions import REGION_CLASSES
from mac_sc2.data.historical_build_tech import SCALAR_FIELDS


def _key(task: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", task)


class HistoricalBuildTechMTL(nn.Module):
    """Historical adapters plus task heads around one shared live V2 trunk.

    ``live_policy.entity_encoder``, ``temporal_encoder``, and ``decoder`` are
    the exact modules used by the current-client policy.  Historical type and
    scalar adapters are deliberately separate because old replay IDs and state
    availability differ from the live V2 contract.
    """
    research_only = True

    def __init__(self, live_policy: RichEntityTransformerPolicy, task_vocabs: dict[str, tuple[str, ...]],
                 input_adapter: HistoricalReplayInputAdapter | None = None):
        super().__init__()
        validate_task_vocabs(task_vocabs)
        self.live_policy = live_policy
        self.task_vocabs = {task: tuple(vocab) for task, vocab in task_vocabs.items()}
        self.tasks = tuple(sorted(self.task_vocabs))
        self.task_index = {task: index for index, task in enumerate(self.tasks)}
        self.patches = tuple(sorted({task.split("/", 1)[0] for task in self.tasks}))
        self.patch_index = {patch: index for index, patch in enumerate(self.patches)}
        width = live_policy.width
        self.patch = nn.Embedding(len(self.patches), width)
        self.task = nn.Embedding(len(self.tasks), width)
        self.input_adapter = input_adapter or HistoricalReplayInputAdapter(width)
        self.history = nn.ModuleDict({_key(task): nn.Embedding(len(vocab) + 1, width, padding_idx=0)
                                      for task, vocab in self.task_vocabs.items()})
        # There is intentionally no global ability classifier: each head is
        # indexed by one exact (patch, race, build_tech_order) key.
        self.heads = nn.ModuleDict({_key(task): nn.Linear(width, len(vocab))
                                    for task, vocab in self.task_vocabs.items()})
        self.region_heads = nn.ModuleDict({_key(task): nn.Linear(width, REGION_CLASSES)
                                           for task in self.task_vocabs})

    def logits(self, scalars: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor,
               history: torch.Tensor, task: str, mmr: torch.Tensor | None = None) -> torch.Tensor:
        if task not in self.task_index:
            raise KeyError(f"unknown historical build/tech task: {task}")
        batch = scalars.size(0)
        patch, race, _ = task.split("/", 2)
        entity = self.input_adapter.entity_tokens(entities)
        if mmr is None: mmr = torch.zeros(batch, 1, dtype=scalars.dtype, device=scalars.device)
        cls = (self.live_policy.cls.expand(batch, -1, -1) + self.input_adapter.scalar(scalars).unsqueeze(1) + self.input_adapter.skill(mmr.view(batch, 1) / 7000).unsqueeze(1) +
               self.live_policy.race.weight[RACES.index(race)].view(1, 1, -1) +
               self.patch.weight[self.patch_index[patch]].view(1, 1, -1) +
               self.task.weight[self.task_index[task]].view(1, 1, -1))
        spatial = self.live_policy.spatial.unsqueeze(0).expand(batch, -1, -1)
        memory = torch.cat((cls, entity, spatial), 1)
        memory_padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=padding.device), padding,
                                    torch.zeros(batch, self.live_policy.spatial_tokens, dtype=torch.bool, device=padding.device)), 1)
        if history.numel():
            temporal = self.live_policy.temporal_encoder(self.history[_key(task)](history))
            memory = torch.cat((memory, temporal), 1)
            memory_padding = torch.cat((memory_padding, torch.zeros(temporal.shape[:2], dtype=torch.bool, device=padding.device)), 1)
        memory = self.live_policy.entity_encoder(memory, src_key_padding_mask=memory_padding)
        hidden = self.live_policy._decode(memory, [self.live_policy.bos.expand(batch, -1, -1)])
        return self.heads[_key(task)](hidden), self.region_heads[_key(task)](hidden)
