"""Executable Terran entity-action contract for the 5.0.16.97563 live client.

This is deliberately a small *complete* action space.  Every intent has a
decoder in :mod:`mac_sc2.runtime.terran_entity_ar_bot`; model outputs are never
allowed to name an ability which that decoder cannot issue.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from mac_sc2.contracts.entity_snapshot import snapshot_hash

# This must be the literal ``GameVersion`` from both the installed client and
# replay metadata, not merely a broad balance-patch label.
PATCH = "5.0.16.97563"
RACE = "Terran"
REGIONS = ("home", "natural", "enemy_start", "enemy_army", "retreat")


@dataclass(frozen=True)
class Intent:
    name: str
    actor_role: str
    target_kind: str  # none, point, entity
    queued: bool = False


# The executor owns exact SC2 abilities, prerequisites, unit selection, and
# placement.  The policy picks an intent, actor entity, and a valid target.
INTENTS = (
    Intent("train_scv", "townhall", "none"),
    Intent("build_supply", "worker", "point"),
    Intent("build_refinery", "worker", "entity"),
    Intent("build_barracks", "worker", "point"),
    Intent("build_factory", "worker", "point"),
    Intent("build_command_center", "worker", "point"),
    Intent("train_marine", "barracks", "none"),
    Intent("train_hellion", "factory", "none"),
    Intent("morph_orbital", "townhall", "none"),
    Intent("call_mule", "townhall", "entity"),
    Intent("attack", "combat", "point"),
    Intent("retreat", "combat", "point"),
    Intent("scout", "combat", "point"),
    Intent("repair", "worker", "entity"),
)


def contract_hash() -> str:
    body = {"patch": PATCH, "race": RACE, "regions": REGIONS,
            "entity_snapshot_hash": snapshot_hash(),
            "intents": [asdict(intent) for intent in INTENTS]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]


def intent_id(name: str) -> int:
    for index, intent in enumerate(INTENTS):
        if intent.name == name:
            return index
    raise KeyError(name)
