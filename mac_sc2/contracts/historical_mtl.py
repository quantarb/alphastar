"""Research-only ActionSpecs for replay patches without live runners."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SUPPORTED_RACES = frozenset(("Terran", "Protoss", "Zerg"))


def task_key(patch: str, race: str) -> str:
    return f"{patch}/{race}"


def build_specs(registry_path: str) -> dict[str, list[dict]]:
    """Keep replay-local ability IDs as labels; these specs are never live."""
    raw = json.loads(Path(registry_path).read_text())
    specs = {}
    for raw_task, rows in raw["tasks"].items():
        patch, race = raw_task.split(":", 1)
        # The source manifest also contains legacy/non-SC2 labels and localized
        # race strings.  They have no compatible live semantic role contract.
        if race not in SUPPORTED_RACES:
            continue
        seen, vocab = set(), []
        for row in rows:
            record = {"actor": row["actor"], "target_kind": row["target_kind"], "target_type": row["target_name"],
                      "queue": bool(row["queued"]), "payload": row["payload"], "family": row["family"],
                      "replay_ability": row["ability_name"], "replay_ability_id": int(row["ability_id"])}
            signature = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if signature not in seen:
                seen.add(signature); vocab.append(record)
        if vocab:
            specs[task_key(patch, race)] = vocab
    return specs


def contract_hash(registry_path: str) -> str:
    specs = build_specs(registry_path)
    body = json.dumps(specs, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()[:16]
