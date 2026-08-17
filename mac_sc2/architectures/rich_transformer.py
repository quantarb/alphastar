"""Transformer-first, live-decodable entity policy.

Every command argument is generated causally.  Pointer logits address entity
tokens, while the semantic intent remains the only ability vocabulary because
the live executor owns the patch-valid concrete ability mapping.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mac_sc2.contracts.rich_transformer_snapshot import ENTITY_SLOTS, SCALAR_FIELDS
from mac_sc2.contracts.race_rich_actions import intents_for
from mac_sc2.contracts.terran_entity_ar import REGIONS

RACES = ("Terran", "Protoss", "Zerg")
MAX_INTENTS = max(len(intents_for(race)) for race in RACES)


@dataclass(frozen=True)
class RichActionOutput:
    intent: torch.Tensor
    actor: torch.Tensor
    target: torch.Tensor
    region: torch.Tensor
    queued: torch.Tensor


class RichEntityTransformerPolicy(nn.Module):
    """Entity, spatial, and temporal transformers with a causal action decoder."""
    def __init__(self, width: int = 192, layers: int = 4, heads: int = 6, history_size: int = 16,
                 spatial_tokens: int = 16):
        super().__init__()
        self.width, self.history_size, self.spatial_tokens = width, history_size, spatial_tokens
        self.race = nn.Embedding(len(RACES), width)
        # Tags are excluded: entities[:, :, 1:] starts with type and alliance.
        self.unit_type = nn.Embedding(4096, width)
        self.alliance = nn.Embedding(8, width)
        self.order = nn.Embedding(8192, width)
        self.entity_numeric = nn.Sequential(nn.Linear(9, width), nn.GELU(), nn.LayerNorm(width))
        self.scalar = nn.Sequential(nn.Linear(len(SCALAR_FIELDS), width), nn.GELU(), nn.LayerNorm(width))
        self.skill = nn.Sequential(nn.Linear(1, width), nn.GELU(), nn.LayerNorm(width))
        self.history = nn.Sequential(nn.Linear(len(SCALAR_FIELDS), width), nn.GELU(), nn.LayerNorm(width))
        self.history_action = nn.Embedding(MAX_INTENTS + 1, width, padding_idx=0)
        self.spatial = nn.Parameter(torch.randn(spatial_tokens, width) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        encoder = nn.TransformerEncoderLayer(width, heads, width * 4, batch_first=True,
                                             activation="gelu", norm_first=True)
        self.entity_encoder = nn.TransformerEncoder(encoder, num_layers=layers)
        self.temporal_encoder = nn.TransformerEncoder(encoder, num_layers=max(2, layers // 2))
        decoder = nn.TransformerDecoderLayer(width, heads, width * 4, batch_first=True,
                                             activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(decoder, num_layers=layers)
        self.bos = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        self.intent_embed = nn.Embedding(MAX_INTENTS, width)
        self.region_embed = nn.Embedding(len(REGIONS), width)
        self.queue_embed = nn.Embedding(2, width)
        # Separate task heads preserve race-specific vocabularies/legality while
        # all perception, temporal memory, and causal decoding are shared.
        self.task_heads = nn.ModuleDict({race: nn.ModuleDict({
            "intent": nn.Linear(width, MAX_INTENTS), "region": nn.Linear(width, len(REGIONS)),
            "queued": nn.Linear(width, 2),
        }) for race in RACES})
        self.actor_query = nn.Linear(width, width, bias=False)
        self.target_query = nn.Linear(width, width, bias=False)

    def _entity_tokens(self, entities: torch.Tensor) -> torch.Tensor:
        # [type, alliance, x, y, health, health_max, shield, energy, build, selected, flying, order]
        values = entities[:, :, 1:]
        type_id = values[:, :, 0].long().remainder(4096)
        alliance = values[:, :, 1].long().clamp(0, 7)
        order = values[:, :, 11].long().remainder(8192)
        numeric = torch.stack((values[:, :, 2] / 256, values[:, :, 3] / 256,
                               values[:, :, 4] / 1000, values[:, :, 5] / 1000,
                               values[:, :, 6] / 1000, values[:, :, 7] / 500,
                               values[:, :, 8], values[:, :, 9], values[:, :, 10]), -1)
        return self.unit_type(type_id) + self.alliance(alliance) + self.order(order) + self.entity_numeric(numeric)

    @staticmethod
    def _mask(logits: torch.Tensor, allowed: torch.Tensor | None) -> torch.Tensor:
        return logits if allowed is None else logits.masked_fill(~allowed, -1e9)

    def _decode(self, memory: torch.Tensor, prefix: list[torch.Tensor]) -> torch.Tensor:
        target = torch.cat(prefix, 1)
        size = target.size(1)
        causal = torch.ones(size, size, device=target.device, dtype=torch.bool).triu(1)
        return self.decoder(target, memory, tgt_mask=causal)[:, -1]

    def forward(self, scalars: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor,
                history_scalars: torch.Tensor | None = None, history_actions: torch.Tensor | None = None,
                race: torch.Tensor | None = None, target_mmr: torch.Tensor | None = None,
                intent: torch.Tensor | None = None, actor: torch.Tensor | None = None,
                target: torch.Tensor | None = None, region: torch.Tensor | None = None,
                intent_mask: torch.Tensor | None = None, actor_mask: torch.Tensor | None = None,
                target_mask: torch.Tensor | None = None) -> RichActionOutput:
        batch = scalars.size(0)
        if race is None:
            race = torch.zeros(batch, dtype=torch.long, device=scalars.device)
        if target_mmr is None:
            target_mmr = torch.zeros(batch, 1, device=scalars.device, dtype=scalars.dtype)
        entity = self._entity_tokens(entities)
        cls = (self.cls.expand(batch, -1, -1) + self.scalar(scalars).unsqueeze(1) +
               self.race(race).unsqueeze(1) + self.skill(target_mmr.view(batch, 1) / 7000).unsqueeze(1))
        spatial = self.spatial.unsqueeze(0).expand(batch, -1, -1)
        memory = torch.cat((cls, entity, spatial), 1)
        memory_padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=padding.device), padding,
                                    torch.zeros(batch, self.spatial_tokens, dtype=torch.bool, device=padding.device)), 1)
        if history_scalars is not None and history_scalars.numel():
            actions = history_actions if history_actions is not None else torch.zeros(
                history_scalars.shape[:2], dtype=torch.long, device=scalars.device)
            temporal = self.temporal_encoder(self.history(history_scalars) + self.history_action(actions))
            memory = torch.cat((memory, temporal), 1)
            memory_padding = torch.cat((memory_padding, torch.zeros(temporal.shape[:2], dtype=torch.bool, device=padding.device)), 1)
        memory = self.entity_encoder(memory, src_key_padding_mask=memory_padding)
        prefix = [self.bos.expand(batch, -1, -1)]
        hidden = self._decode(memory, prefix)
        intent_logits = torch.stack([self.task_heads[name]["intent"](hidden) for name in RACES], 1)[
            torch.arange(batch, device=scalars.device), race]
        intent_logits = self._mask(intent_logits, intent_mask)
        chosen_intent = intent_logits.argmax(-1) if intent is None else intent
        prefix.append(self.intent_embed(chosen_intent).unsqueeze(1))
        hidden = self._decode(memory, prefix); actor_logits = self._mask(
            torch.einsum("bd,bnd->bn", self.actor_query(hidden), memory[:, 1:1 + entities.size(1)]),
            actor_mask if actor_mask is not None else ~padding)
        chosen_actor = actor_logits.argmax(-1) if actor is None else actor
        prefix.append(memory[:, 1:1 + entities.size(1)][torch.arange(batch, device=scalars.device), chosen_actor].unsqueeze(1))
        hidden = self._decode(memory, prefix); target_logits = self._mask(
            torch.einsum("bd,bnd->bn", self.target_query(hidden), memory[:, 1:1 + entities.size(1)]),
            target_mask if target_mask is not None else ~padding)
        chosen_target = target_logits.argmax(-1) if target is None else target
        prefix.append(memory[:, 1:1 + entities.size(1)][torch.arange(batch, device=scalars.device), chosen_target].unsqueeze(1))
        hidden = self._decode(memory, prefix)
        region_logits = torch.stack([self.task_heads[name]["region"](hidden) for name in RACES], 1)[
            torch.arange(batch, device=scalars.device), race]
        chosen_region = region_logits.argmax(-1) if region is None else region
        prefix.append(self.region_embed(chosen_region).unsqueeze(1))
        hidden = self._decode(memory, prefix)
        queue_logits = torch.stack([self.task_heads[name]["queued"](hidden) for name in RACES], 1)[
            torch.arange(batch, device=scalars.device), race]
        return RichActionOutput(intent_logits, actor_logits, target_logits, region_logits, queue_logits)
