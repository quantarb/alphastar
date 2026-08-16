"""The versioned entity representation shared by replay and live policies."""
import hashlib
import json

ENTITY_SLOTS = 64
# type, x/home, y/home, friendly, health fraction, build fraction, flying, worker
ENTITY_FEATURES = 8
FIELDS = ("type", "relative_x", "relative_y", "friendly", "health", "build", "flying", "worker")

def snapshot_hash():
    body = {"slots": ENTITY_SLOTS, "fields": FIELDS, "normalization": "home-relative-v1"}
    return hashlib.sha256(json.dumps(body, separators=(",", ":")).encode()).hexdigest()[:16]
