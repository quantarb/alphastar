"""Research-only, patch/race-specific coarse tactics task contract."""
from __future__ import annotations

import json
from pathlib import Path

from mac_sc2.contracts.historical_regions import REGION_CLASSES, region
from mac_sc2.contracts.semantic_schema import SemanticAction

TASK = "tactics"
RACES = ("Terran", "Protoss", "Zerg")
FAMILIES = ("attack", "move", "patrol", "hold_stop")


def task_key(patch: str, race: str, task: str = TASK) -> str:
    if race not in RACES or task != TASK:
        raise ValueError(f"unsupported historical tactics task: {patch}/{race}/{task}")
    return f"{patch}/{race}/{task}"


def is_tactics(action: SemanticAction) -> bool:
    return action.family in FAMILIES


def build_task_vocabs(registry_path: str | Path) -> dict[str, tuple[str, ...]]:
    registry = json.loads(Path(registry_path).read_text())
    vocabs = {}
    for raw_key, rows in registry["tasks"].items():
        patch, race = raw_key.split(":", 1)
        if race not in RACES:
            continue
        goals = {row["family"] for row in rows if row.get("family") in FAMILIES}
        if goals:
            vocabs[task_key(patch, race)] = tuple(sorted(goals))
    return vocabs
