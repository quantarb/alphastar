"""Semantic macro MTL architecture matching the all-patch initializer."""
import torch
from torch import nn
from mac_sc2.contracts.semantic import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, RACES, TARGET_KINDS

STATE_SIZE = 17

class SemanticMacroPolicy(nn.Module):
    def __init__(self, width=224):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(STATE_SIZE, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.heads = nn.ModuleDict({race: nn.ModuleDict({
            "actor": nn.Linear(width, len(ACTOR_ROLES)), "family": nn.Linear(width, len(FAMILIES)),
            "payload": nn.Linear(width, len(PAYLOAD_ROLES)), "target": nn.Linear(width, len(TARGET_KINDS)),
            "queued": nn.Linear(width, 2),
        }) for race in RACES})
    def forward(self, state, race):
        hidden = self.backbone(state); result = {}
        for key in ("actor", "family", "payload", "target", "queued"):
            logits = torch.stack([self.heads[name][key](hidden) for name in RACES], dim=1)
            result[key] = logits[torch.arange(state.shape[0], device=state.device), race]
        return result
