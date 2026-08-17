"""Live-executable action contract for the rich transformer V2 policy."""
from __future__ import annotations

import hashlib
import json

from mac_sc2.contracts.rich_transformer_snapshot import snapshot_hash
from mac_sc2.contracts.race_rich_actions import intents_for
from mac_sc2.contracts.terran_entity_ar import PATCH, REGIONS


def contract_hash() -> str:
    body = {"name": "rich_transformer_v2_mtl", "patch": PATCH,
            "snapshot_hash": snapshot_hash(), "regions": REGIONS,
            "intents": {race: [intent.__dict__ for intent in intents_for(race)]
                        for race in ("Terran", "Protoss", "Zerg")}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
