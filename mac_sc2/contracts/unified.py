"""Checkpoint contract for the runnable macro, placement and repair policy."""
import hashlib
import json
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.placement_spec import spec_hash as placement_hash
from mac_sc2.contracts.repair import action_hash as repair_hash
from mac_sc2.contracts.semantic import action_hash as macro_hash

def policy_hash(registry_path):
    body = {"version": 1, "macro": macro_hash(), "placement": placement_hash(registry_path), "repair": repair_hash(), "snapshot": snapshot_hash()}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

def validate_checkpoint(data, registry_path):
    if data.get("unified_action_spec_hash") != policy_hash(registry_path):
        raise RuntimeError("unified ActionSpec mismatch")
    required = ("state_dict", "resumed_from")
    missing = [key for key in required if key not in data]
    if missing: raise RuntimeError(f"incomplete unified checkpoint: {missing}")
