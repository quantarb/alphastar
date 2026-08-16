"""One-checkpoint metadata contract for composed playable MTL heads."""
from __future__ import annotations

import hashlib
import json

from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.patch_race_mtl import all_spec_hashes
from mac_sc2.contracts.placement_spec import spec_hash as placement_hash
from mac_sc2.contracts.repair import action_hash as repair_hash


def contract(registry: str) -> dict:
    return {"macro_task_hashes": all_spec_hashes(registry), "placement_hash": placement_hash(registry),
            "repair_hash": repair_hash(), "snapshot_hash": snapshot_hash()}


def contract_hash(registry: str) -> str:
    return hashlib.sha256(json.dumps(contract(registry), sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def validate_checkpoint(data: dict, registry: str) -> None:
    if data.get("multitask_contract_hash") != contract_hash(registry):
        raise RuntimeError("multitask ActionSpec or snapshot contract mismatch")
    if not all(key in data for key in ("state_dict", "resumed_from", "tasks")):
        raise RuntimeError("incomplete multitask checkpoint")
