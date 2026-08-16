"""Single source of truth for trainable and executable semantic actions."""
import hashlib, json
from dataclasses import dataclass

@dataclass(frozen=True)
class Action:
    actor: str; family: str; payload: str; target: str = "none"

# Every tuple below has a concrete implementation in play_semantic_transfer.py.
# Add a behavior here *only* in the same change that adds its live decoder.
ACTIONS = (
    Action("production", "train_morph", "worker"),
    Action("worker", "build", "supply", "point"),
    Action("worker", "build", "production", "point"),
    Action("worker", "build", "gas", "unit"),
    Action("worker", "build", "tech", "point"),
    Action("production", "train_morph", "basic_army"),
    Action("production", "train_morph", "ranged_army"),
    Action("combat", "attack", "spell", "point"),
)

def key(actor, family, payload, target): return (actor, family, payload, target)
def supports(actor, family, payload, target): return key(actor, family, payload, target) in {key(a.actor,a.family,a.payload,a.target) for a in ACTIONS}
def spec_hash():
    blob=json.dumps([a.__dict__ for a in ACTIONS],sort_keys=True,separators=(",",":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]
