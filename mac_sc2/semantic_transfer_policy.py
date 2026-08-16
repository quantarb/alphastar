"""Patch-aware semantic heads transferred from the runnable macro backbone."""
import torch
from torch import nn

from multirace_general_policy import RACES
from semantic_action_schema import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, TARGET_KINDS


class SemanticTransferPolicy(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleDict({race: nn.ModuleDict({
            "actor": nn.Linear(224, len(ACTOR_ROLES)),
            "family": nn.Linear(224, len(FAMILIES)),
            "payload": nn.Linear(224, len(PAYLOAD_ROLES)),
            "target": nn.Linear(224, len(TARGET_KINDS)),
            "queued": nn.Linear(224, 2),
        }) for race in RACES})

    def forward(self, state, race):
        hidden = self.backbone(state)
        result = {}
        for key in ("actor", "family", "payload", "target", "queued"):
            logits = torch.stack([self.heads[name][key](hidden) for name in RACES], dim=1)
            result[key] = logits[torch.arange(state.shape[0], device=state.device), race]
        return result


def metadata():
    return {"architecture": "28.8k-game macro backbone + patch-aware semantic factor heads",
            "actor_roles": ACTOR_ROLES, "families": FAMILIES,
            "payload_roles": PAYLOAD_ROLES, "target_kinds": TARGET_KINDS}
