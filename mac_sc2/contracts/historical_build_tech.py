"""Research-only build/tech task contract for historical SC2 replays.

These labels are semantic replay goals.  They are deliberately not a live
ActionSpec and must never be decoded straight into the current SC2 client.
"""
from __future__ import annotations

import json
from pathlib import Path

from mac_sc2.contracts.semantic_schema import SemanticAction

TASK = "build_tech_order"
RACES = ("Terran", "Protoss", "Zerg")


def task_key(patch: str, race: str, task: str = TASK) -> str:
    if race not in RACES or task != TASK:
        raise ValueError(f"unsupported historical build/tech task: {patch}/{race}/{task}")
    return f"{patch}/{race}/{task}"


def is_build_tech(action: SemanticAction) -> bool:
    """Keep structural construction, research, and tech morphs; exclude micro."""
    # Replay schemas call point-target tactical spells such as Auto-Turret and
    # Creep Tumor "build" commands.  Their payload role distinguishes them
    # from actual structures, and they are not part of a macro build order.
    if action.family == "build" and action.payload_role != "spell":
        return True
    if action.family == "research":
        return True
    # Lair/Hive and equivalent transformations are strategic tech decisions,
    # even though replay schemas call them morph commands.
    return action.family == "train_morph" and action.payload_role in {"townhall", "production", "tech"}


def goal_name(action: SemanticAction) -> str:
    """Task-local, human-readable goal identity without replay ability IDs."""
    return f"{action.family}:{action.ability_name}"


def build_task_vocabs(registry_path: str | Path) -> dict[str, tuple[str, ...]]:
    """Create one individual-command vocabulary for every patch/race task.

    This deliberately returns a mapping rather than one flattened vocabulary:
    ``5.0.7/Protoss/build_tech_order`` and
    ``5.0.7/Terran/build_tech_order`` have distinct output spaces.  A goal is
    admitted only when that exact patch/race registry observed it as a build,
    research, or strategic tech-morph command.
    """
    registry = json.loads(Path(registry_path).read_text())
    vocabs: dict[str, tuple[str, ...]] = {}
    for raw_key, rows in registry["tasks"].items():
        patch, race = raw_key.split(":", 1)
        if race not in RACES:
            continue
        goals = set()
        for row in rows:
            action = SemanticAction(
                patch=patch, race=race, actor_role=row.get("actor", "unknown"),
                family=row.get("family", "other"), payload_role=row.get("payload", "none"),
                target_kind=row.get("target_kind", "none"), queued=bool(row.get("queued", False)),
                ability_id=int(row.get("ability_id", 0)), ability_name=row.get("ability_name", ""),
                target_name=row.get("target_name", ""), location=None,
            )
            if is_build_tech(action):
                goals.add(goal_name(action))
        if goals:
            vocabs[task_key(patch, race)] = tuple(sorted(goals))
    validate_task_vocabs(vocabs)
    return vocabs


def validate_task_vocabs(task_vocabs: dict[str, tuple[str, ...]]) -> None:
    """Reject accidental flattening or malformed patch/race task vocabularies."""
    for key, goals in task_vocabs.items():
        patch, race, task = key.split("/", 2)
        if task != TASK:
            raise ValueError(f"unexpected historical task vocabulary: {key}")
        task_key(patch, race, task)
        if not goals or len(goals) != len(set(goals)):
            raise ValueError(f"task vocabulary must have unique build/tech goals: {key}")
        if any(not goal.startswith(("build:", "research:", "train_morph:")) for goal in goals):
            raise ValueError(f"non-build/tech goal in {key}")
