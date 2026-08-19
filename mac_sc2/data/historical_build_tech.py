"""Event-sourced, build/tech-only examples from historical raw replays."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path
import zlib

import sc2reader

from mac_sc2.contracts.historical_build_tech import goal_name, is_build_tech, task_key
from mac_sc2.contracts.historical_regions import region
from mac_sc2.contracts.semantic_schema import COMBAT, WORKER, from_event

HISTORY_SIZE = 16
# Source-specific strategic features.  The historical adapter maps these into
# the V2 Transformer width; they do not change the live snapshot contract.
SCALAR_FIELDS = (
    "minerals", "vespene", "food_cap", "food_used", "food_army", "food_workers",
    "idle_workers", "army_count", "warpgates", "larva", "mineral_rate", "vespene_rate",
    "resources_lost", "stats_age_seconds", "observed_construction", "recent_destruction",
    "minerals_active_forces", "minerals_current_commit", "minerals_in_progress",
    "vespene_active_forces", "vespene_current_commit", "vespene_in_progress",
    "minerals_lost", "vespene_lost", "minerals_killed", "vespene_killed",
    "known_enemy_units", "known_enemy_combat", "position_age_seconds",
)


def _type_id(name: str) -> int:
    return zlib.crc32(name.lower().encode()) % 4096


def _name(unit) -> str:
    return str(getattr(unit, "type_name", getattr(unit, "name", "")) or "").split(" [", 1)[0]


def _owner_pid(unit) -> int | None:
    return getattr(getattr(unit, "owner", None), "pid", None)


def _position(unit, fallback=(0.0, 0.0)) -> tuple[float, float]:
    point = getattr(unit, "location", None)
    return (float(point[0]), float(point[1])) if point else fallback


def _is_worker(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in WORKER)


def _is_combat(name: str) -> bool:
    lowered = name.lower()
    return any(word in lowered for word in COMBAT)


def _scalar(stats, units: dict[int, dict], pid: int, second: float, last_stats_second: float,
            recent_destruction: int, position_age: float = 0.0) -> list[float]:
    owned = [row for row in units.values() if row["owner"] == pid]
    names = [row["name"] for row in owned]
    workers = sum(_is_worker(name) for name in names)
    army = sum(_is_combat(name) for name in names)
    constructing = sum(not row["done"] for row in owned)
    def value(field: str, default=0):
        return float(getattr(stats, field, default) or default) if stats is not None else float(default)
    enemy = [row for row in units.values() if row["owner"] == 3 - pid]
    return [
        value("minerals_current"), value("vespene_current"), value("food_made"), value("food_used"),
        0.0, value("workers_active_count", workers), 0.0, float(army),
        float(sum("warpgate" in name.lower() for name in names)),
        float(sum("larva" in name.lower() for name in names)), value("minerals_collection_rate"),
        value("vespene_collection_rate"), value("resources_lost"), max(0.0, second - last_stats_second),
        float(constructing), float(recent_destruction),
        value("minerals_used_active_forces"), value("minerals_used_current"), value("minerals_used_in_progress"),
        value("vespene_used_active_forces"), value("vespene_used_current"), value("vespene_used_in_progress"),
        value("minerals_lost"), value("vespene_lost"), value("minerals_killed"), value("vespene_killed"),
        float(len(enemy)), float(sum(_is_combat(row["name"]) for row in enemy)), max(0.0, position_age),
    ]


def _entities(units: dict[int, dict], pid: int) -> list[list[float]]:
    rows = []
    for unit_id, row in units.items():
        if row["owner"] not in {pid, 3 - pid}:
            continue
        alliance = 1 if row["owner"] == pid else 4
        x, y = row["position"]
        # Historical streams do not reliably expose live health/order values;
        # zero marks unavailable fields while build_progress is event-derived.
        rows.append([unit_id, _type_id(row["name"]), alliance, x, y, 0.0, 0.0, 0.0,
                     0.0, 1.0 if row["done"] else 0.0, 0.0, 0.0, 0.0])
    return sorted(rows, key=lambda row: (row[2] != 1, row[0]))[:96]


def examples(path: str | Path, patch: str, task_vocabs: dict[str, tuple[str, ...]],
             discarded: Counter | None = None):
    """Yield individual ordered build/tech actions with their prior state.

    The replay is parsed headlessly.  Each example carries a 16-action history
    so parallel commands remain separate but are visible to the next action.
    """
    rejected = discarded if discarded is not None else Counter()
    replay = sc2reader.load_replay(str(path), load_level=4)
    races = {player.pid: player.play_race for player in replay.players}
    mmrs = {player.pid: int(getattr(player, "mmr", 0) or 0) for player in replay.players}
    latest_stats, stats_second = {}, defaultdict(float)
    position_second = defaultdict(float)
    units: dict[int, dict] = {}
    # A task head is shared across players, but action history is not: two
    # same-race players in one replay must produce independent trajectories.
    history = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    recent_destruction = defaultdict(int)
    vocab_index = {task: {goal: index for index, goal in enumerate(goals)}
                   for task, goals in task_vocabs.items()}

    for event in replay.events:
        kind = type(event).__name__
        if kind == "PlayerStatsEvent":
            latest_stats[event.pid] = event
            stats_second[event.pid] = float(event.second)
            continue
        if kind in {"UnitBornEvent", "UnitInitEvent", "UnitDoneEvent", "UnitTypeChangeEvent"}:
            unit = getattr(event, "unit", None)
            owner = _owner_pid(unit)
            if unit is not None and owner in races:
                unit_id = int(getattr(event, "unit_id", getattr(unit, "id", 0)) or 0)
                if unit_id:
                    prior = units.get(unit_id, {})
                    units[unit_id] = {
                        "owner": owner, "name": _name(unit) or prior.get("name", ""),
                        "position": _position(unit, prior.get("position", (0.0, 0.0))),
                        "done": kind != "UnitInitEvent",
                    }
            continue
        if kind == "UnitDiedEvent":
            unit_id = int(getattr(event, "unit_id", 0) or 0)
            prior = units.pop(unit_id, None)
            if prior is not None:
                recent_destruction[prior["owner"]] += 1
            continue
        if kind == "UnitPositionsEvent":
            for unit, position in getattr(event, "units", {}).items():
                unit_id = int(getattr(unit, "id", 0) or 0)
                if unit_id in units:
                    units[unit_id]["position"] = (float(position[0]), float(position[1]))
            for player_pid in races:
                position_second[player_pid] = float(event.second)
            continue
        if "CommandEvent" not in kind:
            continue
        player = getattr(event, "player", None)
        pid = getattr(player, "pid", None)
        if pid not in races:
            continue
        action = from_event(event, patch, races[pid], [])
        if not is_build_tech(action):
            continue
        task = task_key(patch, races[pid])
        goal = goal_name(action)
        if task not in vocab_index or goal not in vocab_index[task]:
            rejected["goal_outside_task_vocab"] += 1
            continue
        player_history = (task, pid)
        prior_history = list(history[player_history])
        yield {
            "task": task, "patch": patch, "race": races[pid], "player": pid, "mmr": mmrs[pid], "second": float(event.second),
            "scalar": _scalar(latest_stats.get(pid), units, pid, float(event.second), stats_second[pid], recent_destruction[pid],
                              float(event.second) - position_second[pid]),
            "entities": _entities(units, pid), "goal": vocab_index[task][goal],
            # Only actual structure construction has a placement target.
            # Research, production, and strategic morphs use no-target.
            "region": region(action.location) if action.family == "build" else region(None),
            "history": prior_history, "goal_name": goal,
        }
        # Reserve zero for padding in the shared temporal encoder.  The output
        # label remains zero-based; only prior-action history is shifted.
        history[player_history].append(vocab_index[task][goal] + 1)
        recent_destruction[pid] = 0
