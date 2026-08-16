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


def is_build_or_land(record: dict) -> bool:
    return record["family"] == "build" or record["replay_ability"].lower().startswith("land")


def live_decodable(record: dict, race: str) -> bool:
    """Return whether the live runner has a complete target/actor decoder.

    ``cast`` and ``either`` commands are intentionally excluded until their
    specific target semantics are implemented and tested.  This prevents a
    broad replay vocabulary from becoming a shadow-only classifier.
    """
    actor, family, mode = record["actor"], record["family"], record["target_mode"]
    if actor not in {"worker", "combat", "production", "transport"}:
        return False
    if is_build_or_land(record):
        return actor in {"worker", "production"} and mode == "point"
    if family in {"train_morph", "research", "cancel", "hold_stop"}:
        return mode == "none"
    if family == "repair":
        # Repair is still a valid Terran micro tuple, but not an auxiliary
        # task.  Cross-race/captured-unit artifacts are never executable by a
        # normal worker in their nominal race.
        return race == "Terran" and actor == "worker" and mode == "unit" and record["ability"] in {78, 316}
    if family == "gather":
        return actor == "worker" and mode == "unit"
    if family in {"move", "patrol"}:
        return mode == "point"
    if family == "rally":
        return actor == "production" and mode == "point"
    if family == "attack":
        return actor == "combat" and mode in {"point", "unit"}
    return False


def build_specs(registry_path: str) -> dict[str, list[dict]]:
    """Keep precisely the registry entries resolved to an SC2 4.9.2 ability."""
    data = json.loads(open(registry_path).read())
    specs = {}
    for raw_task, entries in data["tasks"].items():
        seen, rows = set(), []
        race = raw_task.split(":", 1)[1]
        for entry in entries:
            if entry.get("live_4_9_2", {}).get("status") != "resolved":
                continue
            record = tuple_record(entry)
            if not live_decodable(record, race):
                continue
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


def validate_live_contract(registry_path: str) -> dict[str, int]:
    """Fail closed unless every retained tuple has the 4.9.2 decoder path."""
    specs = build_specs(registry_path)
    required = {"4.9.2/Terran", "4.9.2/Protoss", "4.9.2/Zerg"}
    if set(specs) != required:
        raise ValueError(f"expected exactly the playable 4.9.2 tasks, got {sorted(specs)}")
    for task, rows in specs.items():
        _, race = task.split("/", 1)
        if not rows or any(not live_decodable(row, race) for row in rows):
            raise ValueError(f"incomplete live decoder contract for {task}")
    return {task: len(rows) for task, rows in specs.items()}
