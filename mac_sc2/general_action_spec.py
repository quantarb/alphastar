"""Versioned, declarative representation for every replay command.

This is deliberately a grammar, not a list of bespoke bot behaviours.  A
repair, an upgrade, a spell, and a movement order all share the same action
record.  Patch/race vocabularies restrict which *ability* labels exist; the
live runner further masks them to the abilities actually offered by selected
4.9.2 units.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Literal, Optional, Tuple


TargetKind = Literal["none", "self", "unit", "point"]


@dataclass(frozen=True)
class GeneralAction:
    """One complete command, in the format shared by data, policy and runner."""
    actor_role: str
    ability: str              # task-local replay ability label
    target_kind: TargetKind
    target_type: str          # empty for point/no-target commands
    target_point: Optional[Tuple[float, float]]
    queued: bool
    delay_loops: int


FIELDS = tuple(GeneralAction.__dataclass_fields__)
SCHEMA_VERSION = 1


def schema_hash() -> str:
    payload = json.dumps({"version": SCHEMA_VERSION, "fields": FIELDS}, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def action_record(**values) -> dict:
    """Validate and serialise an action without silently dropping fields."""
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise ValueError(f"Unknown action fields: {sorted(unknown)}")
    missing = set(FIELDS) - set(values)
    if missing:
        raise ValueError(f"Missing action fields: {sorted(missing)}")
    return asdict(GeneralAction(**values))
