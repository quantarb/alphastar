"""Executable semantic macro vocabulary shared by extraction, model and runner."""
import hashlib
import json
from dataclasses import asdict, dataclass

RACES = ("Terran", "Protoss", "Zerg")
RACE_IDS = {race.lower(): index for index, race in enumerate(RACES)}
ACTOR_ROLES = ("worker", "combat", "transport", "production", "mixed", "unknown")
FAMILIES = ("move", "attack", "hold_stop", "patrol", "gather", "build", "train_morph", "research", "repair", "transport", "rally", "cancel", "cast", "other")
PAYLOAD_ROLES = ("none", "worker", "supply", "gas", "townhall", "production", "tech", "basic_army", "ranged_army", "advanced_army", "upgrade", "spell", "utility")
TARGET_KINDS = ("none", "self", "unit", "point")

@dataclass(frozen=True)
class MacroAction:
    actor: str
    family: str
    payload: str
    target: str = "none"

ACTIONS = (
    MacroAction("production", "train_morph", "worker"),
    MacroAction("worker", "build", "supply", "point"),
    MacroAction("worker", "build", "production", "point"),
    MacroAction("worker", "build", "gas", "unit"),
    MacroAction("worker", "build", "tech", "point"),
    MacroAction("production", "train_morph", "basic_army"),
    MacroAction("production", "train_morph", "ranged_army"),
    MacroAction("combat", "attack", "spell", "point"),
)

def supports(actor, family, payload, target):
    return MacroAction(actor, family, payload, target) in ACTIONS

def action_hash():
    return hashlib.sha256(json.dumps([asdict(x) for x in ACTIONS], sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
