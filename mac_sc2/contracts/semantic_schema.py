"""Patch-stable semantic representation of a StarCraft II player command.

Raw ability IDs deliberately never form a *cross-patch* label. IDs change
between patches and are legal only in the client that defined them. This
module retains both portable concepts and the replay ability name: the latter
is a task-local label for exact patch/race cloning, while the former can share
representations across tasks. Patch validation remains the live decoder's
responsibility.
"""
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

FAMILIES = (
    "move", "attack", "hold_stop", "patrol", "gather", "build",
    "train_morph", "research", "repair", "transport", "rally", "cancel",
    "cast", "other",
)
ACTOR_ROLES = ("worker", "combat", "transport", "production", "mixed", "unknown")
TARGET_KINDS = ("none", "self", "unit", "point")
PAYLOAD_ROLES = ("none", "worker", "supply", "gas", "townhall", "production", "tech", "basic_army", "ranged_army", "advanced_army", "upgrade", "spell", "utility")

WORKER = ("scv", "probe", "drone")
COMBAT = ("marine", "marauder", "reaper", "hellion", "tank", "thor", "ghost", "zealot", "stalker", "adept", "sentry", "immortal", "zergling", "roach", "hydralisk", "mutalisk", "queen")
TRANSPORT = ("medivac", "warpprism", "overlordtransport", "nydus")
PRODUCTION = ("barracks", "factory", "starport", "gateway", "robotics", "stargate", "hatchery", "lair", "hive")


@dataclass(frozen=True)
class SemanticAction:
    patch: str
    race: str
    actor_role: str
    family: str
    payload_role: str
    target_kind: str
    queued: bool
    ability_id: int
    ability_name: str
    target_name: str
    location: Optional[Tuple[float, float]]

    def record(self):
        return asdict(self)


def actor_role(selected_names):
    text = " ".join(selected_names).lower()
    matches = [
        ("worker", WORKER), ("combat", COMBAT), ("transport", TRANSPORT), ("production", PRODUCTION),
    ]
    found = [name for name, words in matches if any(word in text for word in words)]
    return found[0] if len(found) == 1 else ("mixed" if found else "unknown")


def family(ability_name):
    name = (ability_name or "").lower()
    if "repair" in name: return "repair"
    if name.startswith("move") or "move" in name: return "move"
    if "hold" in name or name == "stop" or "stop" in name: return "hold_stop"
    if "patrol" in name: return "patrol"
    if "harvest" in name or "gather" in name or "returncargo" in name: return "gather"
    if "cancel" in name: return "cancel"
    if "rally" in name: return "rally"
    if "load" in name or "unload" in name: return "transport"
    if "research" in name or "upgrade" in name: return "research"
    if "attack" in name: return "attack"
    if name.startswith("build") or "construct" in name: return "build"
    if name.startswith("train") or name.startswith("morph") or name.startswith("warpin"): return "train_morph"
    return "cast" if name else "other"


def target_kind(event):
    event_type = type(event).__name__
    if "TargetUnit" in event_type: return "unit"
    if "TargetPoint" in event_type: return "point"
    flags = getattr(event, "flag", {}) or {}
    return "self" if flags.get("target_self") else "none"


def payload_role(ability_name):
    """Stable purpose of a command's ability, without retaining its raw ID."""
    name = (ability_name or "").lower()
    if not name: return "none"
    if any(word in name for word in ("scv", "probe", "drone")): return "worker"
    if any(word in name for word in ("supplydepot", "pylon", "overlord")): return "supply"
    if any(word in name for word in ("refinery", "assimilator", "extractor")): return "gas"
    if any(word in name for word in ("commandcenter", "orbitalcommand", "planetaryfortress", "nexus", "hatchery", "lair", "hive")): return "townhall"
    if any(word in name for word in ("barracks", "factory", "starport", "gateway", "robotics", "stargate", "spawningpool", "roachwarren", "hydraliskden")): return "production"
    if any(word in name for word in ("cybernetics", "engineeringbay", "armory", "forge", "spire", "evolution")): return "tech"
    if any(word in name for word in ("marine", "zealot", "zergling")): return "basic_army"
    if any(word in name for word in ("hellion", "stalker", "adept", "roach", "hydralisk", "marauder")): return "ranged_army"
    if any(word in name for word in ("medivac", "immortal", "queen", "siegetank", "colossus", "mutalisk", "carrier")): return "advanced_army"
    if "research" in name or "upgrade" in name: return "upgrade"
    if any(word in name for word in ("scan", "chrono", "inject", "mule")): return "utility"
    return "spell"


def target_label(event):
    """Return a stable target-type label, never a replay-local unit id.

    sc2reader renders targets as e.g. ``VespeneGeyser [1980001]``.  The
    bracketed number is unique to one replay and would make an unusable target
    vocabulary.  Preserve the visible type/name only.
    """
    target = getattr(event, "target", None)
    if target is None:
        return ""
    name = getattr(target, "name", None) or getattr(target, "type_name", None)
    if name:
        return str(name)
    return str(target).split(" [", 1)[0]


def from_event(event, patch, race, selected_names):
    flags = getattr(event, "flag", {}) or {}
    location = getattr(event, "location", None)
    command_family = family(getattr(event, "ability_name", ""))
    selected_role = actor_role(selected_names)
    # When old replay UI data omits the selection, certain command families
    # still identify their executor unambiguously. This is semantic recovery,
    # not a game-specific behavior rule.
    if selected_role == "unknown":
        if command_family in ("repair", "gather", "build"):
            selected_role = "worker"
        elif command_family in ("train_morph", "research", "rally"):
            selected_role = "production"
        elif command_family == "transport":
            selected_role = "transport"
    return SemanticAction(
        patch=patch, race=race, actor_role=selected_role,
        family=command_family, payload_role=payload_role(getattr(event, "ability_name", "")), target_kind=target_kind(event),
        queued=bool(flags.get("queued", False)), ability_id=int(getattr(event, "ability_id", 0) or 0),
        ability_name=getattr(event, "ability_name", "") or "",
        target_name=target_label(event),
        location=tuple(location[:2]) if location else None,
    )
