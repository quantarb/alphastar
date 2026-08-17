"""Versioned rich entity-token contract used by the transformer V2 policy."""
from __future__ import annotations

import hashlib
import json

# The cache keeps tag for matching a pointer result back to a live unit.  Tags
# are deliberately excluded from learned features because they are arbitrary.
ENTITY_FIELDS = (
    "tag", "unit_type", "alliance", "x", "y", "health", "health_max", "shield",
    "energy", "build_progress", "selected", "flying", "first_order_ability",
)
ENTITY_SLOTS = 96
SCALAR_FIELDS = (
    "minerals", "vespene", "food_cap", "food_used", "food_army", "food_workers",
    "idle_worker_count", "army_count", "warp_gate_count", "larva_count",
)


def snapshot_hash() -> str:
    body = {"entity_fields": ENTITY_FIELDS, "entity_slots": ENTITY_SLOTS,
            "scalar_fields": SCALAR_FIELDS, "tag_is_model_input": False,
            "normalization": "absolute-map-v1"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
