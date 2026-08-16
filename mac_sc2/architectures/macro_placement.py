"""Factorised learned macro intent and legal-tile ranking heads."""
import torch
from torch import nn

class SnapshotEncoder(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(8, width), nn.GELU(), nn.Linear(width, width), nn.GELU())
    def forward(self, entities, mask):
        z = self.entity(entities)
        return z.masked_fill(mask.unsqueeze(-1), 0).sum(1) / (~mask).sum(1, keepdim=True).clamp(min=1)

class MacroIntentPolicy(nn.Module):
    """Chooses a build/land ability; it has no location output."""
    def __init__(self, abilities, width=128):
        super().__init__(); self.encoder=SnapshotEncoder(width); self.ability=nn.Linear(width, abilities)
    def forward(self, entities, mask): return self.ability(self.encoder(entities, mask))

class PlacementRanker(nn.Module):
    """Scores only candidates supplied by SC2's placement query."""
    def __init__(self, width=128):
        super().__init__(); self.encoder=SnapshotEncoder(width); self.score=nn.Sequential(nn.Linear(width+2,width),nn.GELU(),nn.Linear(width,1))
    def forward(self, entities, mask, candidates):
        z=self.encoder(entities,mask).unsqueeze(1).expand(-1,candidates.size(1),-1)
        return self.score(torch.cat((z,candidates),-1)).squeeze(-1)
