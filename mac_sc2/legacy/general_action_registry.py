"""The patch-valid ActionSpec used by general-action extraction, training and play."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class ActionTuple:
    task: str
    actor_role: str
    family: str
    ability: str
    ability_id: int
    target_kind: str
    target_type: str
    queued: bool

    @property
    def requires_placement(self) -> bool:
        """Whether SC2 must resolve a build/landing tile before execution.

        This is a command-category property, not an ability-specific bot
        branch: every replay action whose semantic family is construction or
        whose verb is a structure landing uses the same SC2 placement query.
        """
        return self.target_kind == "point" and (
            self.family == "build" or self.ability.lower().startswith("land")
        )

    @property
    def requires_flying_actor(self) -> bool:
        return self.ability.lower().startswith("land")

    @property
    def requires_grounded_actor(self) -> bool:
        return self.ability.lower().startswith("lift")


class ActionRegistry:
    """Read only actions that have an exact live 4.9.2 realization.

    A replay command with no target is *not* retained for an ability whose
    live API requires a unit target.  Keeping it would make the model emit an
    unexecutable tuple, even if the replay's UI record called it ``SCVRepair``.
    """
    def __init__(self, path: str | Path, patch: str = "4.9.2"):
        self.path = Path(path)
        self.patch = patch
        raw = json.loads(self.path.read_text())
        rows = []
        for task, entries in raw["tasks"].items():
            if not task.startswith(f"{patch}:"):
                continue
            for row in entries:
                live = row.get("live_4_9_2", {})
                if live.get("status") != "resolved" or not row.get("ability_name"):
                    continue
                mode = live.get("target_mode")
                if mode != row["target_kind"] and mode != "either":
                    continue
                rows.append(ActionTuple(
                    task, row["actor"], row["family"], row["ability_name"], int(live["ability_id"]),
                    row["target_kind"], row.get("target_name", ""), bool(row["queued"]),
                ))
        if not rows:
            raise ValueError(f"No executable {patch} actions in {self.path}")
        self.rows = tuple(sorted(set(rows), key=lambda x: (x.task, x.actor_role, x.family, x.ability, x.ability_id, x.target_kind, x.target_type, x.queued)))
        self.tasks = tuple(sorted({row.task for row in self.rows}))
        self.abilities = tuple(sorted({row.ability for row in self.rows}))
        self.target_types = tuple(sorted({row.target_type for row in self.rows if row.target_type}))
        self._by_signature = {(x.task, x.actor_role, x.ability, x.target_kind, x.target_type, x.queued): x for x in self.rows}
        body = [x.__dict__ for x in self.rows]
        self.hash = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]

    def task_id(self, race: str) -> int:
        return self.tasks.index(f"{self.patch}:{race}")

    def lookup(self, race: str, actor: str, ability: str, target_kind: str, target_type: str, queued: bool):
        return self._by_signature.get((f"{self.patch}:{race}", actor, ability, target_kind, target_type, queued))

    def candidates(self, race: str):
        return tuple(row for row in self.rows if row.task == f"{self.patch}:{race}")
