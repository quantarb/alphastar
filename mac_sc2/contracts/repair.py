"""Small executable repair contract for entity-pointer learning."""
import hashlib, json
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class RepairAction:
    race: str = "Terran"
    actor_role: str = "worker"
    ability: str = "SCVRepair"
    ability_id: int = 316
    target_kind: str = "friendly_entity"

def action_hash():
    return hashlib.sha256(json.dumps(asdict(RepairAction()),sort_keys=True,separators=(",", ":")).encode()).hexdigest()[:16]

def validate_checkpoint(data):
    if data.get("repair_action_spec_hash") != action_hash(): raise RuntimeError("Repair ActionSpec mismatch")
