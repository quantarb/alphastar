"""A compact hierarchical multi-task policy for live SC2 control.

The entity encoder is shared by every race.  A recurrent temporal module
models recent decisions, while small race-specific heads decode macro actions.
It deliberately accepts fixed-size tensors so replay shards and ``BotAI`` can
produce exactly the same representation.
"""
import torch
from torch import nn

RACES = ("Terran", "Protoss", "Zerg")
MACRO_ACTIONS = ("worker", "supply", "production", "army", "expand", "attack", "wait")

class HierarchicalMTLPolicy(nn.Module):
    def __init__(self, entity_features=24, width=96, entity_layers=2, history_steps=16):
        super().__init__()
        self.history_steps = history_steps
        self.entity_encoder = nn.Sequential(nn.Linear(entity_features, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        block = nn.TransformerEncoderLayer(width, 8, width * 4, batch_first=True, norm_first=True)
        self.entity_transformer = nn.TransformerEncoder(block, entity_layers)
        self.race_embedding = nn.Embedding(3, width)
        self.history_encoder = nn.Sequential(nn.Linear(entity_features, width), nn.LayerNorm(width), nn.GELU())
        self.temporal = nn.GRU(width, width, num_layers=2, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(width * 2, width), nn.GELU(), nn.LayerNorm(width))
        self.macro_heads = nn.ModuleDict({race: nn.Linear(width, len(MACRO_ACTIONS)) for race in RACES})
        # These action-component heads share almost all parameters with macro
        # control.  They retain the AlphaStar-style factorisation without
        # ballooning the compact model into an untrainable local experiment.
        self.select_query = nn.Linear(width, width)
        self.target_query = nn.Linear(width, width)
        self.location_head = nn.Linear(width, 2)
        self.queue_head = nn.Linear(width, 2)
        self.value_head = nn.Linear(width, 1)

    def forward(self, entities, entity_mask, history, race_ids):
        if race_ids.unique().numel() != 1:
            raise ValueError("train or infer one race per batch")
        race = RACES[int(race_ids[0])]
        encoded = self.entity_encoder(entities) + self.race_embedding(race_ids)[:, None, :]
        encoded = self.entity_transformer(encoded, src_key_padding_mask=entity_mask)
        usable = (~entity_mask).unsqueeze(-1)
        pooled = (encoded * usable).sum(1) / usable.sum(1).clamp_min(1)
        temporal, _ = self.temporal(self.history_encoder(history))
        hidden = self.fusion(torch.cat((pooled, temporal[:, -1]), dim=-1))
        select = torch.einsum("bd,bnd->bn", self.select_query(hidden), encoded).masked_fill(entity_mask, -1e9)
        target = torch.einsum("bd,bnd->bn", self.target_query(hidden), encoded).masked_fill(entity_mask, -1e9)
        return {"macro": self.macro_heads[race](hidden), "select_entity": select,
                "target_entity": target, "target_location": torch.tanh(self.location_head(hidden)),
                "queued": self.queue_head(hidden), "value": self.value_head(hidden).squeeze(-1)}
