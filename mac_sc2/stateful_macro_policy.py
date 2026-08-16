"""Small race-conditioned policy for replay-derived, live-compatible macro state."""
import torch
from torch import nn

RACES = ("Terran", "Protoss", "Zerg")
MACRO_ACTIONS = ("worker", "supply", "production", "army", "expand", "attack", "wait")
STATE_SIZE = 14


class StatefulMacroPolicy(nn.Module):
    def __init__(self, width=192):
        super().__init__()
        self.race = nn.Embedding(3, 16)
        self.net = nn.Sequential(
            nn.Linear(STATE_SIZE + 16, width), nn.GELU(), nn.LayerNorm(width),
            nn.Linear(width, width), nn.GELU(), nn.Linear(width, len(MACRO_ACTIONS)),
        )

    def forward(self, state, race):
        return self.net(torch.cat((state, self.race(race)), dim=-1))
