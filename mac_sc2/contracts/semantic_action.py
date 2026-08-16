"""Executable semantic-action vocabulary for the all-race MTL checkpoint."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticAction:
    actor: str
    family: str
    payload: str
    target: str = "none"


ACTIONS = (
    SemanticAction("production", "train_morph", "worker"),
    SemanticAction("worker", "build", "supply", "point"),
    SemanticAction("worker", "build", "production", "point"),
    SemanticAction("worker", "build", "gas", "unit"),
    SemanticAction("worker", "build", "tech", "point"),
    SemanticAction("production", "train_morph", "basic_army"),
    SemanticAction("production", "train_morph", "ranged_army"),
    SemanticAction("combat", "attack", "spell", "point"),
    SemanticAction("worker", "build", "townhall", "point"),
    SemanticAction("production", "research", "upgrade"),
    SemanticAction("production", "train_morph", "advanced_army"),
    SemanticAction("combat", "move", "utility", "point"),
)

# The all-replay semantic baseline was trained under the original eight-action
# contract. It is accepted only as a fine-tuning initializer, never by live play.
BASELINE_SPEC_HASH = "8fcf0fdbd1b9d2a9"


def supports(actor: str, family: str, payload: str, target: str) -> bool:
    return (actor, family, payload, target) in {(x.actor, x.family, x.payload, x.target) for x in ACTIONS}


def spec_hash() -> str:
    body = json.dumps([x.__dict__ for x in ACTIONS], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]
