"""Small, contract-checked V2 MTL training run for an end-to-end checkpoint."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.rich_transformer import RACES, RichEntityTransformerPolicy
from mac_sc2.contracts.rich_transformer_action import contract_hash
from mac_sc2.contracts.rich_transformer_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.data.rich_v2_labels import examples
from mac_sc2.runtime.race_rich_executor import validate_race_live_contract
from mac_sc2.runtime.terran_entity_ar_bot import validate_live_contract as validate_terran


def train(cache: str, race: str, output: str, max_labels: int = 256) -> dict:
    """Train one bounded raw-replay cache pass and save one overwrite checkpoint."""
    validate_terran(); validate_race_live_contract("Protoss"); validate_race_live_contract("Zerg")
    discarded: Counter[str] = Counter()
    rows = list(examples(cache, race, discarded))[:max_labels]
    rows = [row for row in rows if 0 <= row["actor"] < len(row["entities"]) and 0 <= row["target"] < len(row["entities"])]
    if not rows:
        raise RuntimeError(f"no usable {race} rich-V2 labels: {dict(discarded)}")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = RichEntityTransformerPolicy(width=96, layers=2, heads=6).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    race_id = RACES.index(race)
    for start in range(0, len(rows), 32):
        batch = rows[start:start + 32]; size = len(batch)
        scalars = torch.tensor([row["scalar"] for row in batch], dtype=torch.float32, device=device)
        entities = torch.zeros(size, ENTITY_SLOTS, 13, device=device); padding = torch.ones(size, ENTITY_SLOTS, dtype=torch.bool, device=device)
        for i, row in enumerate(batch):
            count = len(row["entities"]); entities[i, :count] = torch.tensor(row["entities"], dtype=torch.float32, device=device); padding[i, :count] = False
        intent = torch.tensor([row["intent"] for row in batch], device=device)
        actor = torch.tensor([row["actor"] for row in batch], device=device)
        target = torch.tensor([row["target"] for row in batch], device=device)
        region = torch.tensor([row["region"] for row in batch], device=device)
        queued = torch.tensor([row["queued"] for row in batch], device=device)
        result = model(scalars, entities, padding, race=torch.full((size,), race_id, device=device),
                       target_mmr=torch.tensor([[row["mmr"]] for row in batch], dtype=torch.float32, device=device),
                       intent=intent, actor=actor, target=target, region=region)
        loss = sum(F.cross_entropy(logits, values) for logits, values in ((result.intent, intent), (result.actor, actor),
                   (result.target, target), (result.region, region), (result.queued, queued)))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"architecture_name": "RichEntityTransformerPolicy", "architecture": {"width": 96, "layers": 2, "heads": 6},
                "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()}, "games": 1,
                "resumed_from": None, "initialization": "from_scratch_explicitly_authorized_v2_e2e",
                "action_contract_hash": contract_hash(), "entity_snapshot_hash": snapshot_hash(),
                "labels": len(rows), "discarded": dict(discarded), "source_cache": str(Path(cache).resolve())}, destination)
    return {"checkpoint": str(destination.resolve()), "labels": len(rows), "discarded": dict(discarded)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("cache"); parser.add_argument("--race", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--max-labels", type=int, default=256)
    args = parser.parse_args(); print(json.dumps(train(args.cache, args.race, args.output, args.max_labels), indent=2))
