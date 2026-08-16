"""Live-compatible multi-task SC2 behavioural-cloning policy.

Unlike the raw replay-ability model, this network consumes a fixed 16-value
state vector that can be produced both from a replay trace and from ``python-
sc2`` during a live game.  All races share the state encoder and Transformer;
only the final macro-action decoder is race-specific.
"""
import torch
from torch import nn

RACES = ("Terran", "Protoss", "Zerg")
RACE_TO_ID = {name: index for index, name in enumerate(RACES)}
MACRO_ACTIONS = ("worker", "supply", "production", "army", "expand", "attack", "wait")


class LiveMTLPolicy(nn.Module):
    def __init__(self, state_features: int = 16, width: int = 128, layers: int = 2):
        super().__init__()
        self.state_encoder = nn.Sequential(nn.Linear(state_features, width), nn.LayerNorm(width), nn.GELU())
        self.race_embedding = nn.Embedding(len(RACES), width)
        block = nn.TransformerEncoderLayer(width, 4, width * 4, batch_first=True, norm_first=True)
        self.shared_torso = nn.TransformerEncoder(block, layers)
        self.heads = nn.ModuleDict({race: nn.Linear(width, len(MACRO_ACTIONS)) for race in RACES})

    def forward(self, states: torch.Tensor, race_ids: torch.Tensor) -> torch.Tensor:
        # states: [batch, history, 16]; batches remain homogeneous by race.
        if race_ids.unique().numel() != 1:
            raise ValueError("batch one race at a time")
        race = RACES[int(race_ids[0])]
        hidden = self.state_encoder(states) + self.race_embedding(race_ids)[:, None, :]
        return self.heads[race](self.shared_torso(hidden)[:, -1])
