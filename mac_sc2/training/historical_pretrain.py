"""Research-only shared-backbone pretraining from all raw replay patches."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.historical_mtl import HistoricalReplayMTL
from mac_sc2.contracts.historical_mtl import build_specs, contract_hash
from mac_sc2.contracts.semantic_schema import FAMILIES
from mac_sc2.data.patch_race_exact import examples


@dataclass(frozen=True)
class HistoricalPretrainConfig:
    manifest: str
    registry: str
    output: str
    backbone: str = "mac_sc2/artifacts/patch_race_recent_streaming_base.pt"
    games: int | None = None
    learning_rate: float = 2e-4
    batch_size: int = 64
    checkpoint_every: int = 200


def _save(output: Path, model: HistoricalReplayMTL, config: HistoricalPretrainConfig,
          games: int, specs: dict[str, list[dict]], counts: Counter, discarded: Counter) -> None:
    """Persist one replace-in-place research snapshot, never a live checkpoint."""
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "research_only": True,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "games": games,
        "resumed_from": str(Path(config.backbone).resolve()),
        "historical_contract_hash": contract_hash(config.registry),
        "tasks": len(specs),
        "task_routes": {task: ("micro", "macro", *( ("build",) if model.build_ids[task] else ()))
                        for task in specs},
        "counts": dict(counts),
        "discarded": dict(discarded),
    }, output)


def _train_rows(model: HistoricalReplayMTL, rows: list[dict], specs: dict[str, list[dict]],
                family_ids: dict[str, int], optimizer: torch.optim.Optimizer,
                device: torch.device, batch_size: int, counts: Counter) -> None:
    """Train task-local heads in compact batches from one in-memory replay stream."""
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        by_task.setdefault(row["task"], []).append(row)
    for task, task_rows in by_task.items():
        for start in range(0, len(task_rows), batch_size):
            batch = task_rows[start:start + batch_size]
            state = torch.tensor([row["state"] for row in batch], dtype=torch.float32, device=device)
            tuple_ids = torch.tensor([row["tuple_id"] for row in batch], dtype=torch.long, device=device)
            loss = F.cross_entropy(model.micro_logits(state, task), tuple_ids)
            families = torch.tensor([family_ids[specs[task][row["tuple_id"]]["family"]] for row in batch], dtype=torch.long, device=device)
            loss = loss + .2 * F.cross_entropy(model.macro_logits(state, task), families)
            build_ids = model.build_ids[task]
            build_rows = [row for row in batch if row["tuple_id"] in build_ids]
            if build_rows:
                build_state = torch.tensor([row["state"] for row in build_rows], dtype=torch.float32, device=device)
                build_target = torch.tensor([build_ids.index(row["tuple_id"]) for row in build_rows], dtype=torch.long, device=device)
                loss = loss + .4 * F.cross_entropy(model.build_logits(build_state, task), build_target)
                counts["build"] += len(build_rows)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            counts["micro"] += len(batch)
            counts["macro"] += len(batch)


def pretrain(config: HistoricalPretrainConfig) -> dict:
    """Train offline heads only; this artifact is rejected by live runners."""
    import json
    items = json.loads(Path(config.manifest).read_text())["valid"]
    if config.games is not None:
        items = items[:config.games]
    specs = build_specs(config.registry)
    if not specs:
        raise ValueError("empty historical ActionSpec registry")
    model = HistoricalReplayMTL(specs)
    source = torch.load(config.backbone, map_location="cpu", weights_only=False)
    model.load_backbone(source["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    family_ids = {family: index for index, family in enumerate(FAMILIES)}
    counts, discarded = Counter(), Counter()
    output = Path(config.output)
    for game, item in enumerate(items, 1):
        try:
            _train_rows(model, list(examples(item["path"], item["version"], specs, discarded)), specs,
                        family_ids, optimizer, device, config.batch_size, counts)
        except Exception as exc:
            discarded[type(exc).__name__] += 1
        if game % config.checkpoint_every == 0:
            _save(output, model, config, game, specs, counts, discarded)
            print(f"research_games={game} labels={counts['micro']}", flush=True)
    _save(output, model, config, len(items), specs, counts, discarded)
    return {"checkpoint": str(output.resolve()), "tasks": len(specs), "counts": dict(counts), "discarded": dict(discarded)}
