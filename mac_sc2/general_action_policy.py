"""Factorised policy for the common replay/live general-action grammar."""
import torch
from torch import nn

from general_action_spec import FIELDS
from semantic_action_schema import ACTOR_ROLES, TARGET_KINDS


class GeneralActionPolicy(nn.Module):
    """Predict all command fields; vocab masks are applied outside the model.

    Abilities and target types use global embedding tables while a task ID
    conditions the shared trunk.  A patch/race mask is required before loss or
    decoding, so labels from another patch never become executable actions.
    """
    def __init__(self, state_size: int, ability_count: int, target_type_count: int,
                 task_count: int, width: int = 224):
        super().__init__()
        self.task = nn.Embedding(task_count, width)
        self.trunk = nn.Sequential(
            nn.Linear(state_size, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.GELU(),
        )
        self.heads = nn.ModuleDict({
            "actor_role": nn.Linear(width, len(ACTOR_ROLES)),
            "ability": nn.Linear(width, ability_count),
            "target_kind": nn.Linear(width, len(TARGET_KINDS)),
            "target_type": nn.Linear(width, target_type_count + 1), # 0 = none
            "target_point": nn.Linear(width, 2),
            "queued": nn.Linear(width, 2),
            "delay_loops": nn.Linear(width, 8),
        })

    def forward(self, state: torch.Tensor, task_id: torch.Tensor):
        hidden = self.trunk(state) + self.task(task_id)
        out = {name: head(hidden) for name, head in self.heads.items()}
        out["target_point"] = torch.tanh(out["target_point"])
        return out

    @staticmethod
    def mask(logits: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """Mask impossible task/live choices before both loss and decode."""
        return logits.masked_fill(~allowed, -1e9)
