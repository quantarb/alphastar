"""Post-AlphaStar-style entity policy for the executable Terran contract.

The ``AutoregressiveArgumentHead`` below is a scoped port of the argument
embedding pattern in mini-AlphaStar's ``delay_head.py`` and ``queue_head.py``
(Ruo-Ze Liu, Apache-2.0; https://github.com/liuruoze/mini-AlphaStar).  It is
adapted to this repository's versioned, executable 5.0.16.97563 action contract.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from mac_sc2.contracts.entity_snapshot import ENTITY_FEATURES
from mac_sc2.contracts.terran_entity_ar import INTENTS, REGIONS

STATE_SIZE = 17


@dataclass(frozen=True)
class ActionOutput:
    intent: torch.Tensor
    actor: torch.Tensor
    target: torch.Tensor
    region: torch.Tensor
    queued: torch.Tensor


class AutoregressiveArgumentHead(nn.Module):
    """Predict a categorical argument then embed it into the next head.

    This is the central mini-AlphaStar decoder idea: subsequent argument heads
    see the sampled (or teacher-forced) earlier argument, rather than making
    independent predictions from the same core state.
    """
    def __init__(self, width: int, choices: int):
        super().__init__()
        self.predict = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(),
                                     nn.Linear(width, choices))
        self.embed = nn.Sequential(nn.Linear(choices, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())

    def forward(self, embedding: torch.Tensor, choice: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.predict(embedding)
        selected = logits.argmax(-1) if choice is None else choice
        one_hot = torch.nn.functional.one_hot(selected, logits.size(-1)).to(embedding.dtype)
        return logits, selected, embedding + self.embed(one_hot)


class TerranEntityARPolicy(nn.Module):
    """Encode scalars and visible entities, then decode action arguments.

    Heads are evaluated in the action order: intent -> actor -> target/region
    -> queue.  Conditioning embeddings make invalid independent head products
    less likely, while the live decoder remains the definitive legality gate.
    """
    def __init__(self, width: int = 192, layers: int = 3, history_size: int = 16):
        super().__init__()
        self.width = width
        self.history_size = history_size
        self.scalar = nn.Sequential(nn.Linear(STATE_SIZE, width), nn.GELU(), nn.LayerNorm(width))
        self.entity = nn.Sequential(nn.Linear(ENTITY_FEATURES, width), nn.GELU(), nn.LayerNorm(width))
        encoder_layer = nn.TransformerEncoderLayer(width, nhead=6, dim_feedforward=width * 3,
                                                   batch_first=True, activation="gelu", norm_first=True)
        self.entity_torso = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.temporal = nn.GRU(width, width, batch_first=True)
        self.history_embedding = nn.Embedding(len(INTENTS) + 1, width, padding_idx=0)
        self.initial_autoregressive = nn.Linear(width * 2, width)
        self.intent_head = AutoregressiveArgumentHead(width, len(INTENTS))
        self.actor_query = nn.Linear(width, width, bias=False)
        self.actor_update = nn.Linear(width, width)
        self.target_query = nn.Linear(width, width, bias=False)
        self.target_update = nn.Linear(width, width)
        self.region_head = AutoregressiveArgumentHead(width, len(REGIONS))
        self.queue_head = AutoregressiveArgumentHead(width, 2)

    def encode(self, state: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor,
               history: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        entity = self.entity_torso(self.entity(entities), src_key_padding_mask=padding)
        valid = (~padding).unsqueeze(-1)
        pooled = (entity * valid).sum(1) / valid.sum(1).clamp_min(1)
        temporal = torch.zeros_like(pooled)
        if history is not None and history.numel():
            _, hidden = self.temporal(self.history_embedding(history))
            temporal = hidden[-1]
        return self.scalar(state) + temporal, entity

    @staticmethod
    def _pointer(query: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,bnd->bn", query, entities).masked_fill(padding, -1e9)

    def forward(self, state: torch.Tensor, entities: torch.Tensor, padding: torch.Tensor,
                history: torch.Tensor | None = None, intent: torch.Tensor | None = None) -> ActionOutput:
        core, encoded = self.encode(state, entities, padding, history)
        pooled = encoded.masked_fill(padding.unsqueeze(-1), 0).mean(1)
        embedding = self.initial_autoregressive(torch.cat((core, pooled), -1))
        intent_logits, _, embedding = self.intent_head(embedding, intent)
        actor_logits = self._pointer(self.actor_query(embedding), encoded, padding)
        actor = actor_logits.argmax(-1)
        embedding = embedding + self.actor_update(encoded[torch.arange(encoded.size(0), device=encoded.device), actor] - pooled)
        target_logits = self._pointer(self.target_query(embedding), encoded, padding)
        target = target_logits.argmax(-1)
        embedding = embedding + self.target_update(encoded[torch.arange(encoded.size(0), device=encoded.device), target] - pooled)
        region_logits, _, embedding = self.region_head(embedding)
        queue_logits, _, _ = self.queue_head(embedding)
        return ActionOutput(intent_logits, actor_logits, target_logits, region_logits, queue_logits)
