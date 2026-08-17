"""Race-specific, live-decoder action specifications for transformer MTL V2.

These are semantic command roles, not replay ability ids.  A live runner must
provide the concrete 5.0.16.97563 ability and legality guard for every entry
before trajectories from that race are admitted to policy training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

from mac_sc2.contracts.terran_entity_ar import PATCH, REGIONS


@dataclass(frozen=True)
class RaceIntent:
    name: str
    actor_role: str
    target_kind: str  # none, point, entity
    queued: bool = False


SPECS: dict[str, tuple[RaceIntent, ...]] = {
    "Terran": tuple(),  # The existing Terran contract remains the source of truth.
    "Protoss": (
        RaceIntent("train_probe", "townhall", "none"),
        RaceIntent("build_pylon", "worker", "point"),
        RaceIntent("build_assimilator", "worker", "entity"),
        RaceIntent("build_gateway", "worker", "point"),
        RaceIntent("build_cybernetics", "worker", "point"),
        RaceIntent("build_nexus", "worker", "point"),
        RaceIntent("train_zealot", "gateway", "none"),
        RaceIntent("train_stalker", "gateway", "none"),
        RaceIntent("research_warpgate", "cybernetics", "none"),
        RaceIntent("chronoboost", "townhall", "entity"),
        RaceIntent("attack", "combat", "point"),
        RaceIntent("retreat", "combat", "point"),
        RaceIntent("scout", "combat", "point"),
    ),
    "Zerg": (
        RaceIntent("train_drone", "larva", "none"),
        RaceIntent("train_overlord", "larva", "none"),
        RaceIntent("build_extractor", "worker", "entity"),
        RaceIntent("build_spawning_pool", "worker", "point"),
        RaceIntent("build_roach_warren", "worker", "point"),
        RaceIntent("build_hatchery", "worker", "point"),
        RaceIntent("train_zergling", "larva", "none"),
        RaceIntent("train_roach", "larva", "none"),
        RaceIntent("inject_larva", "queen", "entity"),
        RaceIntent("spread_creep", "queen", "point"),
        RaceIntent("attack", "combat", "point"),
        RaceIntent("retreat", "combat", "point"),
        RaceIntent("scout", "combat", "point"),
        RaceIntent("morph_overseer", "overlord", "none"),
    ),
}


def intents_for(race: str) -> tuple[RaceIntent, ...]:
    if race == "Terran":
        from mac_sc2.contracts.terran_entity_ar import INTENTS
        return tuple(RaceIntent(**asdict(intent)) for intent in INTENTS)
    return SPECS[race]


def contract_hash() -> str:
    body = {"name": "rich_transformer_mtl_v2", "patch": PATCH, "regions": REGIONS,
            "races": {race: [asdict(intent) for intent in intents_for(race)] for race in ("Terran", "Protoss", "Zerg")}}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
