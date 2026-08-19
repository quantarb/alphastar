"""Event-sourced coarse tactics labels from historical raw replays."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from pathlib import Path

import sc2reader

from mac_sc2.contracts.historical_tactics import region, task_key, is_tactics
from mac_sc2.contracts.semantic_schema import from_event
from mac_sc2.data.historical_build_tech import SCALAR_FIELDS, _entities, _name, _owner_pid, _position, _scalar

HISTORY_SIZE = 16
def examples(path: str | Path, patch: str, task_vocabs: dict[str, tuple[str, ...]],
             discarded: Counter | None = None):
    """Yield one tactical decision per player command, with independent history."""
    rejected = discarded if discarded is not None else Counter()
    replay = sc2reader.load_replay(str(path), load_level=4)
    races = {player.pid: player.play_race for player in replay.players}
    mmrs = {player.pid: int(getattr(player, "mmr", 0) or 0) for player in replay.players}
    latest_stats, stats_second = {}, defaultdict(float)
    units: dict[int, dict] = {}
    position_second = defaultdict(float)
    histories = defaultdict(lambda: deque(maxlen=HISTORY_SIZE))
    recent_deaths = defaultdict(int)
    indexes = {task: {action: index for index, action in enumerate(vocab)}
               for task, vocab in task_vocabs.items()}
    for event in replay.events:
        kind = type(event).__name__
        if kind == "PlayerStatsEvent":
            latest_stats[event.pid] = event; stats_second[event.pid] = float(event.second); continue
        if kind in {"UnitBornEvent", "UnitInitEvent", "UnitDoneEvent", "UnitTypeChangeEvent"}:
            unit = getattr(event, "unit", None); owner = _owner_pid(unit)
            if unit is not None and owner in races:
                unit_id = int(getattr(event, "unit_id", getattr(unit, "id", 0)) or 0)
                if unit_id:
                    prior = units.get(unit_id, {})
                    units[unit_id] = {"owner": owner, "name": _name(unit) or prior.get("name", ""),
                                      "position": _position(unit, prior.get("position", (0.0, 0.0))),
                                      "done": kind != "UnitInitEvent"}
            continue
        if kind == "UnitDiedEvent":
            unit_id = int(getattr(event, "unit_id", 0) or 0); prior = units.pop(unit_id, None)
            if prior is not None: recent_deaths[prior["owner"]] += 1
            continue
        if kind == "UnitPositionsEvent":
            for unit, position in getattr(event, "units", {}).items():
                unit_id = int(getattr(unit, "id", 0) or 0)
                if unit_id in units: units[unit_id]["position"] = (float(position[0]), float(position[1]))
            for pid in races: position_second[pid] = float(event.second)
            continue
        if "CommandEvent" not in kind or getattr(event, "player", None) is None: continue
        pid = event.player.pid
        if pid not in races: continue
        action = from_event(event, patch, races[pid], [])
        if not is_tactics(action): continue
        task = task_key(patch, races[pid])
        if task not in indexes or action.family not in indexes[task]:
            rejected["tactic_outside_task_vocab"] += 1; continue
        history_key = (task, pid)
        yield {"task": task, "patch": patch, "race": races[pid], "player": pid, "mmr": mmrs[pid], "second": float(event.second),
               "scalar": _scalar(latest_stats.get(pid), units, pid, float(event.second), stats_second[pid],
                                 recent_deaths[pid], float(event.second) - position_second[pid]), "entities": _entities(units, pid),
               "action": indexes[task][action.family], "region": region(action.location),
               "history": list(histories[history_key]), "action_name": action.family}
        histories[history_key].append(indexes[task][action.family] + 1)
        recent_deaths[pid] = 0
