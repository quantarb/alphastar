"""Joint raw-replay lifecycle for macro, placement, and repair task heads."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from mac_sc2.architectures.multitask_policy import PlayableMultiTaskPolicy
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS
from mac_sc2.contracts.multitask import contract, contract_hash
from mac_sc2.contracts.patch_race_mtl import build_specs
from mac_sc2.data.patch_race_exact import examples as macro_examples
from mac_sc2.data.placement_replay import examples as placement_examples
from mac_sc2.data.repair_replay import examples as repair_examples
from mac_sc2.data.validate_patch_race_rich_mtl import compatible_replays, validate_alignment
from mac_sc2.evaluation.patch_race_match import launch_easy_suite


@dataclass(frozen=True)
class MultiTaskConfig:
    manifest: str
    registry: str
    output: str
    macro_resume: str = "mac_sc2/artifacts/patch_race_recent_streaming_base.pt"
    task_resume: str = "mac_sc2/artifacts/combined_policy_4_9_2.pt"
    games: int = 200
    learning_rate: float = 2e-4
    evaluation_dir: str = "mac_sc2/artifacts"


def _pack(snapshot):
    entities = torch.zeros(ENTITY_SLOTS, 8); size = min(len(snapshot), ENTITY_SLOTS)
    if size: entities[:size] = torch.tensor(snapshot[:size])
    mask = torch.ones(ENTITY_SLOTS, dtype=torch.bool); mask[:size] = False
    return entities, mask


def fine_tune(config: MultiTaskConfig) -> dict:
    if config.games != 200: raise ValueError("the first playable checkpoint is game 200")
    validation = validate_alignment(config.registry, config.manifest)
    specs = build_specs(config.registry); files = compatible_replays(config.manifest, specs)[:config.games]
    if len(files) != config.games: raise ValueError(f"need {config.games} compatible raw replays")
    macro = torch.load(config.macro_resume, map_location="cpu", weights_only=False)
    task = torch.load(config.task_resume, map_location="cpu", weights_only=False)
    model = PlayableMultiTaskPolicy(specs); model.load_initializers(macro["state_dict"], task)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=.01)
    counts, discarded = Counter(), Counter()
    for item in files:
        try:
            for row in macro_examples(item["path"], item["version"], specs, discarded):
                state = torch.tensor([row["state"]], dtype=torch.float32, device=device)
                loss = F.cross_entropy(model.task_logits(state, row["task"]), torch.tensor([row["tuple_id"]], device=device))
                optimizer.zero_grad(); loss.backward(); optimizer.step(); counts["macro"] += 1
            for snapshot, label, home in placement_examples(item["path"]):
                entities, mask = _pack(snapshot); positive = torch.tensor([(label.point[0]-home[0])/64, (label.point[1]-home[1])/64])
                candidates = torch.cat((positive[None], positive[None] + torch.tensor([[.1,0],[-.1,0],[0,.1],[0,-.1]])), 0)[None]
                loss = F.cross_entropy(model.placement(entities[None].to(device), mask[None].to(device), candidates.to(device)), torch.tensor([0], device=device))
                optimizer.zero_grad(); loss.backward(); optimizer.step(); counts["placement"] += 1
            for snapshot, actor, target in repair_examples(item["path"]):
                entities, mask = _pack(snapshot); actor_logits, target_logits = model.repair(entities[None].to(device), mask[None].to(device))
                loss = F.cross_entropy(actor_logits, torch.tensor([actor],device=device)) + F.cross_entropy(target_logits, torch.tensor([target],device=device))
                optimizer.zero_grad(); loss.backward(); optimizer.step(); counts["repair"] += 1
        except Exception as exc: discarded[type(exc).__name__] += 1
    output = Path(config.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {key:value.detach().cpu() for key,value in model.state_dict().items()}, "games": config.games,
                "resumed_from": {"macro": str(Path(config.macro_resume).resolve()), "placement_repair": str(Path(config.task_resume).resolve())},
                "tasks": ("macro", "placement", "repair"), "multitask_contract": contract(config.registry),
                "multitask_contract_hash": contract_hash(config.registry), "validation": validation,
                "counts": dict(counts), "discarded": dict(discarded)}, output)
    return {"checkpoint": str(output.resolve()), "counts": dict(counts), "validation": validation,
            "evaluation_processes": launch_easy_suite(str(output), config.registry, config.evaluation_dir)}
