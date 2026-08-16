"""Live-state tactical controller with entity-pointer action heads.

This is intentionally not a flat ability classifier.  Each decision names the
acting entity/squad and a visible target (or relative map offset), so the live
decoder never has to invent an enemy-start target for the model.
"""
import torch
from torch import nn
from spatial_action_contract import COMMANDS

ENTITY_FEATURES = 20
MAX_ENTITIES = 64
class SpatialTacticalPolicy(nn.Module):
    def __init__(self, macro_backbone, width=224, heads=4, layers=2):
        super().__init__()
        self.macro_backbone = macro_backbone
        self.entity = nn.Sequential(nn.Linear(ENTITY_FEATURES, width), nn.LayerNorm(width), nn.GELU())
        layer = nn.TransformerEncoderLayer(width, heads, width * 2, batch_first=True, norm_first=True)
        self.torso = nn.TransformerEncoder(layer, layers)
        self.command = nn.Linear(width, len(COMMANDS))
        self.actor_query = nn.Linear(width, width)
        self.target_query = nn.Linear(width, width)
        self.offset = nn.Linear(width, 2)
        self.queue = nn.Linear(width, 2)

    def forward(self, global_state, entities, padding_mask=None, actor_mask=None, target_mask=None):
        # Reuse the learned macro context but let local entity tokens decide
        # micro. The macro trunk is a context signal, not a hardcoded target.
        context = self.macro_backbone(global_state).unsqueeze(1)
        hidden = self.torso(self.entity(entities) + context, src_key_padding_mask=padding_mask)
        pooled = hidden[:, 0]
        actor = torch.einsum("bd,bnd->bn", self.actor_query(pooled), hidden)
        target = torch.einsum("bd,bnd->bn", self.target_query(pooled), hidden)
        if padding_mask is not None:
            actor = actor.masked_fill(padding_mask, -1e9)
            target = target.masked_fill(padding_mask, -1e9)
        # The model may score only entities the live decoder is permitted to
        # command/target.  The same masks are mandatory in BC training.
        if actor_mask is not None:
            actor = actor.masked_fill(actor_mask, -1e9)
        if target_mask is not None:
            target = target.masked_fill(target_mask, -1e9)
        return {"command": self.command(pooled), "actor": actor, "target": target,
                "offset": torch.tanh(self.offset(pooled)), "queue": self.queue(pooled)}
