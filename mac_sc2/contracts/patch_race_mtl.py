"""Versioned task-local ActionSpecs for the runnable patch/race MTL policy."""
from __future__ import annotations

import hashlib
import json


def task_key(version: str, race: str) -> str:
    """Canonical key used by replay extraction, heads, and the live runner."""
    return f"{'.'.join(version.split('.')[:3])}/{race}"


def _task_from_registry_key(key: str) -> str:
    patch, race = key.split(":", 1)
    return f"{patch}/{race}"


def tuple_record(row: dict) -> dict:
    """The complete, executable label; replay ability ids are never reused."""
    live = row["live_4_9_2"]
    return {
        "actor": row["actor"], "ability": int(live["ability_id"]),
        "target_kind": row["target_kind"], "target_type": row.get("target_name", ""),
        "target_mode": live["target_mode"], "queue": bool(row["queued"]),
        "payload": row["payload"], "family": row["family"],
        "replay_ability": row["ability_name"],
    }


def build_specs(registry_path: str) -> dict[str, list[dict]]:
    """Keep precisely the registry entries resolved to an SC2 4.9.2 ability."""
    data = json.loads(open(registry_path).read())
    specs = {}
    for raw_task, entries in data["tasks"].items():
        seen, rows = set(), []
        for entry in entries:
            if entry.get("live_4_9_2", {}).get("status") != "resolved":
                continue
            record = tuple_record(entry)
            # The same tuple can occur with multiple historical replay ids.
            signature = json.dumps(record, sort_keys=True, separators=(",", ":"))
            if signature not in seen:
                seen.add(signature); rows.append(record)
        if rows:
            specs[_task_from_registry_key(raw_task)] = rows
    return specs


def spec_hash(vocab: list[dict]) -> str:
    blob = json.dumps(vocab, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def all_spec_hashes(registry_path: str) -> dict[str, str]:
    return {task: spec_hash(vocab) for task, vocab in build_specs(registry_path).items()}
