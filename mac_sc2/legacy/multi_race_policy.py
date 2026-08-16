"""Race-conditioned multi-task policy for data-efficient SC2 imitation.

The entity encoder and Transformer are shared among Terran, Protoss, and Zerg.
Race-specific heads keep mutually exclusive ability vocabularies separate.
"""
import torch
from torch import nn

RACES = ('Terran', 'Protoss', 'Zerg')
RACE_TO_ID = {race: index for index, race in enumerate(RACES)}


class MultiRaceSC2Policy(nn.Module):
    def __init__(self, entity_features=16, ability_vocab=2048, width=192, heads=6, layers=4):
        super().__init__()
        self.entity_encoder = nn.Sequential(
            nn.Linear(entity_features, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.race_embedding = nn.Embedding(len(RACES), width)
        block = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True, norm_first=True)
        self.shared_torso = nn.TransformerEncoder(block, layers)
        self.ability_heads = nn.ModuleDict({race: nn.Linear(width, ability_vocab) for race in RACES})
        self.select_query = nn.Linear(width, width)
        self.target_query = nn.Linear(width, width)
        self.location_head = nn.Linear(width, 2)
        self.value_head = nn.Linear(width, 1)

    def forward(self, entities, race_ids, padding_mask=None):
        """Predict an action for a homogeneous-race batch.

        `race_ids` is `[batch]`; training batches should contain one race so a
        matching head is applied.  The shared trunk still receives gradients
        from all three race tasks.
        """
        if race_ids.unique().numel() != 1:
            raise ValueError('Batch by race before calling MultiRaceSC2Policy.')
        race = RACES[int(race_ids[0])]
        hidden = self.entity_encoder(entities) + self.race_embedding(race_ids)[:, None, :]
        hidden = self.shared_torso(hidden, src_key_padding_mask=padding_mask)
        pooled = hidden[:, 0]
        select = torch.einsum('bd,bnd->bn', self.select_query(pooled), hidden)
        target = torch.einsum('bd,bnd->bn', self.target_query(pooled), hidden)
        if padding_mask is not None:
            select = select.masked_fill(padding_mask, -1e9)
            target = target.masked_fill(padding_mask, -1e9)
        return {
            'ability': self.ability_heads[race](pooled),
            'select_entity': select,
            'target_entity': target,
            'target_location': torch.tanh(self.location_head(pooled)),
            'value': self.value_head(pooled).squeeze(-1),
        }
