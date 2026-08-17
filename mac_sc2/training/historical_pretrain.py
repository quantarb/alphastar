"""Unified playable MTL training from every raw replay patch."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.historical_mtl import UnifiedPlayableMTLPolicy
from mac_sc2.contracts.historical_mtl import build_specs as historical_specs, contract_hash as historical_hash
from mac_sc2.contracts.multitask import contract, contract_hash, task_routes, validate_checkpoint
from mac_sc2.contracts.patch_race_mtl import build_specs as live_specs, validate_live_contract
from mac_sc2.contracts.semantic_schema import FAMILIES
from mac_sc2.data.patch_race_exact import examples


@dataclass(frozen=True)
class HistoricalPretrainConfig:
    manifest: str
    registry: str
    output: str
    live_registry: str = "mac_sc2/artifacts/patch_race_4_9_2_action_registry_live.json"
    live_checkpoint: str = "mac_sc2/artifacts/patch_race_mtl_live.pt"
    games: int | None = None
    learning_rate: float = 2e-4
    batch_size: int = 64
    checkpoint_every: int = 200


def _load_live_initializer(model: UnifiedPlayableMTLPolicy, checkpoint: str, registry: str) -> dict:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    validate_checkpoint(payload, registry)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    invalid_missing = [key for key in missing if not key.startswith("historical_")]
    if invalid_missing or unexpected:
        raise ValueError(f"live initializer does not match playable core: missing={invalid_missing} unexpected={unexpected}")
    return payload


def _train_historical(model: UnifiedPlayableMTLPolicy, rows: list[dict], specs: dict[str, list[dict]],
                      family_ids: dict[str, int], optimizer: torch.optim.Optimizer,
                      device: torch.device, batch_size: int, counts: Counter) -> None:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["task"], []).append(row)
    for task, task_rows in grouped.items():
        for start in range(0, len(task_rows), batch_size):
            batch = task_rows[start:start + batch_size]
            state = torch.tensor([row["state"] for row in batch], dtype=torch.float32, device=device)
            labels = torch.tensor([row["tuple_id"] for row in batch], dtype=torch.long, device=device)
            loss = F.cross_entropy(model.historical_micro_logits(state, task), labels)
            families = torch.tensor([family_ids[specs[task][row["tuple_id"]]["family"]] for row in batch], dtype=torch.long, device=device)
            loss = loss + .2 * F.cross_entropy(model.historical_macro_logits(state, task), families)
            build_ids = model.historical_build_ids[task]
            build_rows = [row for row in batch if row["tuple_id"] in build_ids]
            if build_rows:
                build_state = torch.tensor([row["state"] for row in build_rows], dtype=torch.float32, device=device)
                build_labels = torch.tensor([build_ids.index(row["tuple_id"]) for row in build_rows], dtype=torch.long, device=device)
                loss = loss + .4 * F.cross_entropy(model.historical_build_logits(build_state, task), build_labels)
                counts["historical_build"] += len(build_rows)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            counts["historical_micro"] += len(batch); counts["historical_macro"] += len(batch)


def _train_live(model: UnifiedPlayableMTLPolicy, rows: list[dict], specs: dict[str, list[dict]],
                family_ids: dict[str, int], optimizer: torch.optim.Optimizer,
                device: torch.device, counts: Counter) -> None:
    """Keep the selected executable 4.9.2 heads trained in this checkpoint."""
    for row in rows:
        state = torch.tensor([row["state"]], dtype=torch.float32, device=device)
        patch, race = row["task"].split("/")
        label = torch.tensor([row["tuple_id"]], dtype=torch.long, device=device)
        family = specs[row["task"]][row["tuple_id"]]["family"]
        loss = F.cross_entropy(model.micro_logits(state, patch, race), label)
        loss = loss + .2 * F.cross_entropy(model.macro_logits(state, patch, race), torch.tensor([family_ids[family]], device=device))
        build_ids = model.build_action_ids[row["task"]]
        if row["tuple_id"] in build_ids:
            build_label = torch.tensor([build_ids.index(row["tuple_id"])], dtype=torch.long, device=device)
            loss = loss + .4 * F.cross_entropy(model.build_logits(state, patch, race), build_label)
            counts["live_build"] += 1
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        counts["live_micro"] += 1; counts["live_macro"] += 1


def _save(output: Path, model: UnifiedPlayableMTLPolicy, game: int, live_registry: str,
          historical_registry: str, counts: Counter, discarded: Counter, initializer: dict) -> None:
    torch.save({
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "games": game, "resumed_from": initializer,
        "tasks": task_routes(live_registry), "multitask_contract": contract(live_registry),
        "multitask_contract_hash": contract_hash(live_registry),
        "historical_contract_hash": historical_hash(historical_registry),
        "historical_tasks": len(model.historical_specs),
        "counts": dict(counts), "discarded": dict(discarded),
    }, output)


def pretrain(config: HistoricalPretrainConfig) -> dict:
    """Train all patch/race tasks while retaining the executable 4.9.2 head."""
    import json
    raw_items = json.loads(Path(config.manifest).read_text())["valid"]
    items = raw_items if config.games is None else raw_items[:config.games]
    historical = historical_specs(config.registry)
    live = live_specs(config.live_registry); routes = task_routes(config.live_registry)
    if not historical or not live:
        raise ValueError("empty historical or live ActionSpec")
    live_contract = validate_live_contract(config.live_registry)
    model = UnifiedPlayableMTLPolicy(live, routes, historical)
    source = _load_live_initializer(model, config.live_checkpoint, config.live_registry)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    family_ids = {family: index for index, family in enumerate(FAMILIES)}
    counts, discarded = Counter(), Counter()
    output = Path(config.output); output.parent.mkdir(parents=True, exist_ok=True)
    initializer = {"checkpoint": str(Path(config.live_checkpoint).resolve()), "games": source["games"],
                   "live_contract_hash": source["multitask_contract_hash"], "live_contract": live_contract}
    for game, item in enumerate(items, 1):
        try:
            _train_historical(model, list(examples(item["path"], item["version"], historical, discarded)), historical,
                              family_ids, optimizer, device, config.batch_size, counts)
            if ".".join(item["version"].split(".")[:3]) == "4.9.2":
                _train_live(model, list(examples(item["path"], item["version"], live, discarded)), live,
                            family_ids, optimizer, device, counts)
        except Exception as exc:
            discarded[type(exc).__name__] += 1
        if game % config.checkpoint_every == 0:
            _save(output, model, game, config.live_registry, config.registry, counts, discarded, initializer)
            print(f"trained_games={game} live_labels={counts['live_micro']} historical_labels={counts['historical_micro']}", flush=True)
    _save(output, model, len(items), config.live_registry, config.registry, counts, discarded, initializer)
    return {"checkpoint": str(output.resolve()), "games": len(items), "counts": dict(counts),
            "historical_tasks": len(historical), "live_tasks": routes}
