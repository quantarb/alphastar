"""One-checkpoint metadata contract for composed playable MTL heads."""
from __future__ import annotations

import hashlib
import json

from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.patch_race_mtl import all_spec_hashes, build_specs
from mac_sc2.contracts.placement_spec import spec_hash as placement_hash


def contract(registry: str) -> dict:
    return {"micro_task_hashes": all_spec_hashes(registry), "build_placement_hash": placement_hash(registry),
            "snapshot_hash": snapshot_hash(), "task_routes": task_routes(registry)}


def contract_hash(registry: str) -> str:
    return hashlib.sha256(json.dumps(contract(registry), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def validate_checkpoint(data: dict, registry: str) -> None:
    if data.get("multitask_contract_hash") != contract_hash(registry):
        raise RuntimeError("multitask ActionSpec or snapshot contract mismatch")
    if not all(key in data for key in ("state_dict", "resumed_from", "tasks")):
        raise RuntimeError("incomplete multitask checkpoint")
    if data["tasks"] != task_routes(registry):
        raise RuntimeError("checkpoint task routing does not match the live patch/race contract")
TASKS = ("micro", "macro", "build")


def task_key(patch: str, race: str, task: str) -> str:
    if task not in TASKS: raise ValueError(f"unknown playable task: {task}")
    return f"{patch}/{race}/{task}"


def task_routes(registry: str) -> dict[str, tuple[str, ...]]:
    """Only tasks with a patch/race-valid decoder are exposed to the model."""
    routes = {}
    for base, vocab in build_specs(registry).items():
        patch, race = base.split("/", 1); enabled = ["micro", "macro"]
        if any(row["family"] == "build" or row["replay_ability"].lower().startswith("land") for row in vocab): enabled.append("build")
        routes[f"{patch}/{race}"] = tuple(enabled)
    return routes
