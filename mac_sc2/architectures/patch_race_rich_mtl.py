"""One-checkpoint MTL policy with a transferred shared trunk and task heads."""
from __future__ import annotations

import re
import torch
from torch import nn

STATE_SIZE = 17  # Live macro feature contract: see PatchRaceBot.feat().


def module_key(task: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", task)


class PatchRaceRichMTLPolicy(nn.Module):
    def __init__(self, task_specs: dict[str, list[dict]], width: int = 224):
        super().__init__()
        self.task_specs = {key: list(value) for key, value in task_specs.items()}
        # This layout exactly matches patch_race_recent_streaming_base.pt.
        self.shared = nn.Sequential(nn.Linear(STATE_SIZE, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.task_heads = nn.ModuleDict({module_key(task): nn.Linear(width, len(spec)) for task, spec in self.task_specs.items()})
        # Shared pointer/location heads consume the versioned 8-field snapshots.
        self.entity = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
        self.placement = nn.Sequential(nn.Linear(width + 2, width), nn.GELU(), nn.Linear(width, 1))
        self.actor_pointer = nn.Linear(width, width, bias=False)
        self.target_pointer = nn.Linear(width, width, bias=False)

    def hidden(self, state: torch.Tensor) -> torch.Tensor:
        return self.shared(state)

    def task_logits(self, state: torch.Tensor, task: str) -> torch.Tensor:
        return self.task_heads[module_key(task)](self.hidden(state))

    def pointers(self, state: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor):
        h = self.hidden(state); e = self.entity(entities)
        actor = torch.einsum("bd,bnd->bn", self.actor_pointer(h), e).masked_fill(padding, -1e9)
        target = torch.einsum("bd,bnd->bn", self.target_pointer(h), e).masked_fill(padding, -1e9)
        return actor, target

    def placement_scores(self, state: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        h = self.hidden(state)[:, None].expand(-1, candidates.size(1), -1)
        return self.placement(torch.cat((h, candidates), -1)).squeeze(-1)

    def load_streaming_backbone(self, state_dict: dict):
        source = {key.replace("shared.", "shared."): value for key, value in state_dict.items() if key.startswith("shared.")}
        missing, unexpected = self.load_state_dict(source, strict=False)
        if unexpected or any(key.startswith("shared.") for key in missing):
            raise ValueError("streaming initializer does not match the rich shared trunk")
