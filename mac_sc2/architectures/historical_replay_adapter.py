"""Shared adapter from coarse historical replay state into V2 token width."""
from __future__ import annotations

import torch
from torch import nn

from mac_sc2.data.historical_build_tech import SCALAR_FIELDS


class HistoricalReplayInputAdapter(nn.Module):
    """One patch-agnostic adapter shared by all historical auxiliary tasks."""
    def __init__(self, width: int):
        super().__init__()
        self.scalar = nn.Sequential(nn.Linear(len(SCALAR_FIELDS), width), nn.GELU(), nn.LayerNorm(width))
        self.skill = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.LayerNorm(width))
        self.unit_type = nn.Embedding(4096, width)
        self.alliance = nn.Embedding(8, width)
        self.numeric = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width))

    def entity_tokens(self, entities: torch.Tensor) -> torch.Tensor:
        values = entities[:, :, 1:]
        numeric = torch.stack((values[:, :, 2] / 256, values[:, :, 3] / 256,
                               values[:, :, 4] / 1000, values[:, :, 5] / 1000,
                               values[:, :, 6] / 1000, values[:, :, 7] / 500,
                               values[:, :, 8], values[:, :, 9], values[:, :, 10]), -1)
        return (self.unit_type(values[:, :, 0].long().remainder(4096)) +
                self.alliance(values[:, :, 1].long().clamp(0, 7)) + self.numeric(numeric))
