"""Checkpoint contract for the general action policy."""
from general_action_spec import schema_hash


def metadata(vocab_hash, registry_path):
    return {
        "general_action_schema_hash": schema_hash(),
        "action_vocab_hash": vocab_hash,
        "registry": str(registry_path),
        "training_prediction_contract": "role + ability + target kind/type/point + queue + delay",
    }


def validate(checkpoint, vocab_hash):
    if checkpoint.get("general_action_schema_hash") != schema_hash():
        raise RuntimeError("General action schema mismatch")
    if checkpoint.get("action_vocab_hash") != vocab_hash:
        raise RuntimeError("Action vocabulary mismatch")
