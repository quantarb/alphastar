"""Checkpoint-compatible all-race semantic action MTL."""
from __future__ import annotations

import torch
from torch import nn

from mac_sc2.contracts.semantic_schema import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, TARGET_KINDS

RACES = ("Terran", "Protoss", "Zerg")


class SemanticActionMTL(nn.Module):
    def __init__(self, width: int = 224):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(17, width), nn.GELU(), nn.LayerNorm(width), nn.Linear(width, width), nn.GELU())
        self.heads = nn.ModuleDict({race: nn.ModuleDict({
            "actor": nn.Linear(width, len(ACTOR_ROLES)), "family": nn.Linear(width, len(FAMILIES)),
            "payload": nn.Linear(width, len(PAYLOAD_ROLES)), "target": nn.Linear(width, len(TARGET_KINDS)),
            "queued": nn.Linear(width, 2),
        }) for race in RACES})

    def forward(self, state: torch.Tensor, race: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.backbone(state)
        return {key: torch.stack([self.heads[name][key](hidden) for name in RACES], 1)[
            torch.arange(state.size(0), device=state.device), race]
            for key in ("actor", "family", "payload", "target", "queued")}
