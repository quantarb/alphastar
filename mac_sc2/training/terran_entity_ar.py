"""Train the mini-AlphaStar-style Terran policy directly from raw replays."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Process
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.terran_entity_ar import TerranEntityARPolicy
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.contracts.terran_entity_ar import INTENTS, contract_hash
from mac_sc2.data.terran_entity_ar import examples
from mac_sc2.evaluation.terran_entity_ar_match import run_match
from mac_sc2.runtime.terran_entity_ar_bot import validate_live_contract


@dataclass(frozen=True)
class TrainConfig:
    manifest: str
    output: str = "mac_sc2/artifacts/terran_entity_ar_first_pass.pt"
    games: int = 337
    checkpoint_every: int = 200
    batch_size: int = 128
    width: int = 96
    layers: int = 2
    learning_rate: float = 2e-4


def _save(path: Path, model: TerranEntityARPolicy, games: int, labels: Counter, discarded: Counter, config: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "games": games,
                "resumed_from": None, "initialization": "from_scratch_explicitly_authorized",
                "action_contract_hash": contract_hash(), "entity_snapshot_hash": snapshot_hash(),
                "architecture": {"width": config.width, "layers": config.layers}, "labels": dict(labels), "discarded": dict(discarded)}, path)


def _batch_loss(model, batch, device):
    state = torch.tensor([row["state"] for row in batch], dtype=torch.float32, device=device)
    entities = torch.zeros(len(batch), ENTITY_SLOTS, 8, device=device); padding = torch.ones(len(batch), ENTITY_SLOTS, dtype=torch.bool, device=device)
    for index, row in enumerate(batch):
        n = min(len(row["snapshot"]), ENTITY_SLOTS)
        entities[index, :n] = torch.tensor(row["snapshot"][:n], dtype=torch.float32, device=device); padding[index, :n] = False
    intent = torch.tensor([row["intent"] for row in batch], dtype=torch.long, device=device)
    output = model(state, entities, padding, intent=intent)
    loss = F.cross_entropy(output.intent, intent) + .2 * F.cross_entropy(output.queued, torch.tensor([row["queued"] for row in batch], device=device))
    for name, logits in (("actor", output.actor), ("target", output.target)):
        values = torch.tensor([row[name] for row in batch], dtype=torch.long, device=device); keep = values >= 0
        if keep.any(): loss = loss + .25 * F.cross_entropy(logits[keep], values[keep])
    return loss


def train(config: TrainConfig) -> dict:
    """First-pass BC; checkpoint 200 concurrently launches a literal SC2 match."""
    validate_live_contract()
    rows = [item for item in json.loads(Path(config.manifest).read_text())["valid"] if item["version"] == "5.0.16.97563"][:config.games]
    if len(rows) < config.checkpoint_every: raise ValueError("need at least 200 current-patch games for first live checkpoint")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TerranEntityARPolicy(width=config.width, layers=config.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    output, labels, discarded, evaluation = Path(config.output), Counter(), Counter(), None
    for game, item in enumerate(rows, 1):
        batch = list(examples(item["path"], item["version"], discarded))
        for start in range(0, len(batch), config.batch_size):
            chunk = batch[start:start + config.batch_size]
            if not chunk: continue
            loss = _batch_loss(model, chunk, device); optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            labels["commands"] += len(chunk)
        if game % config.checkpoint_every == 0:
            _save(output, model, game, labels, discarded, config)
            if evaluation is None:
                replay = str(output.with_suffix(".first_eval.SC2Replay"))
                evaluation = Process(target=run_match, args=(replay, "veryeasy", str(output)))
                evaluation.start()  # Training continues while the exact snapshot plays.
    _save(output, model, len(rows), labels, discarded, config)
    if evaluation is not None: evaluation.join()
    return {"checkpoint": str(output.resolve()), "games": len(rows), "labels": dict(labels), "discarded": dict(discarded)}
