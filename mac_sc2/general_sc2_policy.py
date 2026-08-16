"""General structured policy interface for current SC2.

Unlike the earlier Marine prototype, this model has no build-order-specific
output head. An action is factorized into an ability, a selected controlled
entity, an optional target entity, and an optional map location. Legal masks
are supplied by the game at inference/training time.
"""
import torch
from torch import nn


class GeneralSC2Policy(nn.Module):
    def __init__(self, entity_features=16, ability_vocab=2048, width=192, heads=6, layers=4):
        super().__init__()
        self.entity = nn.Sequential(nn.Linear(entity_features, width), nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        encoder = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True, norm_first=True)
        self.torso = nn.TransformerEncoder(encoder, layers)
        self.ability = nn.Linear(width, ability_vocab)
        self.select_query = nn.Linear(width, width)
        self.target_query = nn.Linear(width, width)
        self.location = nn.Linear(width, 2)
        self.value = nn.Linear(width, 1)

    def forward(self, entities, padding_mask=None):
        """Return structured action logits.

        entities: [batch, max_entities, 16] current visible raw-unit features.
        padding_mask: True where an entity slot is padding.
        """
        h = self.torso(self.entity(entities), src_key_padding_mask=padding_mask)
        pooled = h[:, 0]
        select = torch.einsum('bd,bnd->bn', self.select_query(pooled), h)
        target = torch.einsum('bd,bnd->bn', self.target_query(pooled), h)
        if padding_mask is not None:
            select = select.masked_fill(padding_mask, -1e9)
            target = target.masked_fill(padding_mask, -1e9)
        return {'ability': self.ability(pooled), 'select_entity': select, 'target_entity': target,
                'target_location': torch.tanh(self.location(pooled)), 'value': self.value(pooled).squeeze(-1)}


def apply_legal_mask(logits, legal):
    """Mask unavailable SC2 abilities without narrowing the model vocabulary."""
    return logits.masked_fill(~legal.bool(), -1e9)
