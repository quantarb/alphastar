"""Joint raw-replay fine tuning for primary micro and auxiliary task heads."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.multitask_policy import PlayableMultiTaskPolicy
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS
from mac_sc2.contracts.multitask import contract, contract_hash, task_routes
from mac_sc2.contracts.patch_race_mtl import build_specs, validate_live_contract
from mac_sc2.contracts.semantic_schema import FAMILIES
from mac_sc2.data.patch_race_exact import examples as macro_examples
from mac_sc2.data.validate_patch_race_rich_mtl import compatible_replays, validate_alignment
from mac_sc2.evaluation.patch_race_match import launch_easy_suite


@dataclass(frozen=True)
class MultiTaskConfig:
    manifest: str
    registry: str
    output: str
    macro_resume: str = "mac_sc2/artifacts/patch_race_recent_streaming_base.pt"
    games: int | None = None
    checkpoint_every: int = 200
    learning_rate: float = 2e-4
    macro_aux_weight: float = .2
    build_aux_weight: float = .4
    placement_aux_weight: float = .5
    evaluation_dir: str = "mac_sc2/artifacts"


def _pack(snapshot):
    entities = torch.zeros(ENTITY_SLOTS, 8); size = min(len(snapshot), ENTITY_SLOTS)
    if size: entities[:size] = torch.tensor(snapshot[:size])
    mask = torch.ones(ENTITY_SLOTS, dtype=torch.bool); mask[:size] = False
    return entities, mask


def _history(ids, size=16):
    values = [0] * (size - min(len(ids), size)) + [value + 1 for value in ids[-size:]]
    return torch.tensor([values], dtype=torch.long)


def fine_tune(config: MultiTaskConfig) -> dict:
    if config.checkpoint_every != 200:
        raise ValueError("playable checkpoints are overwrite-only and saved every 200 games")
    live_contract = validate_live_contract(config.registry)
    validation = validate_alignment(config.registry, config.manifest)
    specs = build_specs(config.registry); available = compatible_replays(config.manifest, specs)
    files = available if config.games is None else available[:config.games]
    if len(files) < config.checkpoint_every:
        raise ValueError(f"need at least {config.checkpoint_every} compatible raw replays, found {len(files)}")
    if config.games is not None and len(files) != config.games:
        raise ValueError(f"need {config.games} compatible raw replays")
    macro = torch.load(config.macro_resume, map_location="cpu", weights_only=False)
    routes = task_routes(config.registry)
    model = PlayableMultiTaskPolicy(specs, routes); model.load_initializers(macro["state_dict"])
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    counts, discarded, evaluations = Counter(), Counter(), []
    family_ids = {family: index for index, family in enumerate(FAMILIES)}
    output = Path(config.output); output.parent.mkdir(parents=True, exist_ok=True)
    for game_number, item in enumerate(files, start=1):
        try:
            for row in macro_examples(item["path"], item["version"], specs, discarded):
                state = torch.tensor([row["state"]], dtype=torch.float32, device=device)
                patch, race = row["task"].split("/")
                micro_loss = F.cross_entropy(model.micro_logits(state, patch, race, _history(row["history"]).to(device)), torch.tensor([row["tuple_id"]], device=device))
                family = specs[row["task"]][row["tuple_id"]]["family"]
                macro_loss = F.cross_entropy(model.macro_logits(state, patch, race), torch.tensor([family_ids[family]], device=device))
                loss = micro_loss + config.macro_aux_weight * macro_loss
                optimizer.zero_grad(); loss.backward(); optimizer.step(); counts["micro"] += 1; counts["macro"] += 1
                build_ids = model.build_action_ids[row["task"]]
                if row["tuple_id"] in build_ids:
                    build_label = build_ids.index(row["tuple_id"])
                    build_loss = config.build_aux_weight * F.cross_entropy(model.build_logits(state, patch, race, _history(row["history"]).to(device)), torch.tensor([build_label], device=device))
                    optimizer.zero_grad(); build_loss.backward(); optimizer.step(); counts["build"] += 1
                entities, mask = _pack(row["snapshot"])
                if (family == "build" or specs[row["task"]][row["tuple_id"]]["replay_ability"].lower().startswith("land")) and row["location"]:
                    if "build" in routes[f"{patch}/{race}"]:
                        positive = torch.tensor(row["location"])
                        candidates = torch.cat((positive[None], positive[None] + torch.tensor([[.1, 0], [-.1, 0], [0, .1], [0, -.1]])), 0)[None]
                        loss = config.placement_aux_weight * F.cross_entropy(model.build_placement_scores(entities[None].to(device), mask[None].to(device), candidates.to(device), patch, race), torch.tensor([0], device=device))
                        optimizer.zero_grad(); loss.backward(); optimizer.step(); counts["build_placement"] += 1
        except Exception as exc: discarded[type(exc).__name__] += 1
        if game_number % config.checkpoint_every == 0:
            torch.save({"state_dict": {key:value.detach().cpu() for key,value in model.state_dict().items()}, "games": game_number,
                        "resumed_from": {"micro_backbone": str(Path(config.macro_resume).resolve())},
                        "tasks": routes, "multitask_contract": contract(config.registry),
                        "multitask_contract_hash": contract_hash(config.registry), "validation": {"live_contract": live_contract, **validation},
                        "counts": dict(counts), "discarded": dict(discarded)}, output)
            # Easy games consume this immutable overwrite-only snapshot while training continues.
            evaluations.append(launch_easy_suite(str(output), config.registry, config.evaluation_dir))
    return {"checkpoint": str(output.resolve()), "counts": dict(counts), "validation": {"live_contract": live_contract, **validation},
            "evaluation_processes": evaluations}
