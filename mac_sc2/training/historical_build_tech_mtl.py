"""Research-only historical build/tech auxiliary training for the V2 trunk.

This module never creates a checkpoint accepted by a live runner.  It is an
auxiliary-representation trainer: old replay events teach the existing V2
entity/temporal/decoder trunk about build and technology timing, while the
historical adapters and task vocabularies remain offline-only.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.historical_build_tech_mtl import HistoricalBuildTechMTL
from mac_sc2.architectures.rich_transformer import RichEntityTransformerPolicy
from mac_sc2.contracts.historical_build_tech import TASK, build_task_vocabs
from mac_sc2.contracts.rich_transformer_action import contract_hash
from mac_sc2.contracts.rich_transformer_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.data.historical_build_tech import HISTORY_SIZE, examples


def _load_live_trunk(checkpoint: str | Path) -> RichEntityTransformerPolicy:
    """Load only a contract-compatible V2 trunk as the shared initializer."""
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if (source.get("architecture_name") != "RichEntityTransformerPolicy" or
            source.get("action_contract_hash") != contract_hash() or
            source.get("entity_snapshot_hash") != snapshot_hash()):
        raise RuntimeError("historical auxiliary task needs a compatible V2 live initializer")
    model = RichEntityTransformerPolicy(**source["architecture"])
    model.load_state_dict(source["state_dict"])
    return model


def _batch(rows: list[dict], device: torch.device) -> tuple[torch.Tensor, ...]:
    count = len(rows)
    scalars = torch.tensor([row["scalar"] for row in rows], dtype=torch.float32, device=device)
    entities = torch.zeros(count, ENTITY_SLOTS, 13, dtype=torch.float32, device=device)
    padding = torch.ones(count, ENTITY_SLOTS, dtype=torch.bool, device=device)
    history = torch.zeros(count, HISTORY_SIZE, dtype=torch.long, device=device)
    labels = torch.tensor([row["goal"] for row in rows], dtype=torch.long, device=device)
    for index, row in enumerate(rows):
        present = len(row["entities"])
        if present:
            entities[index, :present] = torch.tensor(row["entities"], dtype=torch.float32, device=device)
            padding[index, :present] = False
        prior = row["history"]
        if prior:
            history[index, -len(prior):] = torch.tensor(prior, dtype=torch.long, device=device)
    return scalars, entities, padding, history, labels


def train_research_only(sources: list[tuple[str, str]], initializer: str | Path, output: str | Path,
                        registry: str | Path = "mac_sc2/artifacts/historical_action_registry.json",
                        epochs: int = 1, max_labels_per_replay: int = 0) -> dict:
    """Fine-tune a shared V2 trunk with historical build/tech labels only.

    ``sources`` contains ``(patch, replay_path)`` pairs.  The returned artifact
    is intentionally rejected by live runners; export/reintegration into a
    playable policy requires a separately validated, current-patch action path.
    """
    if epochs < 1:
        raise ValueError("epochs must be positive")
    vocabs = build_task_vocabs(registry)
    discarded: Counter[str] = Counter()
    prepared: dict[str, list[dict]] = {}
    source_count = Counter()
    for patch, replay in sources:
        rows = list(examples(replay, patch, vocabs, discarded))
        if max_labels_per_replay:
            rows = rows[:max_labels_per_replay]
        for row in rows:
            prepared.setdefault(row["task"], []).append(row)
        source_count[patch] += 1
    if not prepared:
        raise RuntimeError("no build/tech labels from the supplied historical replays")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HistoricalBuildTechMTL(_load_live_trunk(initializer), vocabs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    counts = {task: len(rows) for task, rows in prepared.items()}
    model.train()
    for _ in range(epochs):
        for task, rows in sorted(prepared.items()):
            for start in range(0, len(rows), 32):
                scalars, entities, padding, history, labels = _batch(rows[start:start + 32], device)
                batch_rows = rows[start:start + 32]
                mmr = torch.tensor([[row.get("mmr", 0)] for row in batch_rows], dtype=torch.float32, device=device)
                goal_logits, region_logits = model.logits(scalars, entities, padding, history, task, mmr)
                regions = torch.tensor([row["region"] for row in rows[start:start + 32]], dtype=torch.long, device=device)
                loss = F.cross_entropy(goal_logits, labels) + F.cross_entropy(region_logits, regions)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "historical_build_tech_mtl_research_v1",
        "research_only": True,
        "live_runner_compatible": False,
        "task": TASK,
        "resumed_from": str(Path(initializer).resolve()),
        "architecture": {"live": model.live_policy.__class__.__name__, "width": model.live_policy.width},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "task_vocabs": vocabs,
        "labels": counts,
        "source_replays": source_count,
        "discarded": dict(discarded),
        "epochs": epochs,
    }, output)
    return {"output": str(output.resolve()), "research_only": True, "labels": counts,
            "source_replays": dict(source_count), "discarded": dict(discarded)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train historical build/tech auxiliary task (research only)")
    parser.add_argument("--research-only", action="store_true", required=True)
    parser.add_argument("--source", action="append", required=True, metavar="PATCH=REPLAY")
    parser.add_argument("--initializer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry", default="mac_sc2/artifacts/historical_action_registry.json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-labels-per-replay", type=int, default=0)
    args = parser.parse_args()
    sources = [tuple(item.split("=", 1)) for item in args.source]
    print(json.dumps(train_research_only(sources, args.initializer, args.output, args.registry,
                                         args.epochs, args.max_labels_per_replay), indent=2))


if __name__ == "__main__":
    main()
