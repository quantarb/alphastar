#!/usr/bin/env python3
"""Inspect real repair / defend-home evidence directly from replay commands.

This intentionally produces a compact report, not another cached training
dataset.  It is the validation gate for adding defensive heads to the live
agent: repair examples need an actual selected worker squad and concrete
target; defend-home examples need a selected combat squad moving to the
player's initial town-hall location.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import sc2reader

from train_general_macro_on_demand import event_pid

TOWN_HALL_WORDS = ("commandcenter", "nexus", "hatchery")
COMBAT_WORDS = ("marine", "marauder", "reaper", "hellion", "tank", "medivac", "zealot", "stalker", "adept", "immortal", "zergling", "roach", "hydralisk", "mutalisk")
WORKER_WORDS = ("scv", "probe", "drone")


def unit_name(unit):
    return str(getattr(unit, "type", unit)).lower()


def selection_kind(units):
    names = [unit_name(unit) for unit in units]
    if any(any(word in name for word in WORKER_WORDS) for name in names):
        return "worker_squad"
    if any(any(word in name for word in COMBAT_WORDS) for name in names):
        return "combat_squad"
    return "other_or_unknown"


def near(point, home, radius=24):
    if point is None or home is None:
        return False
    return ((point[0] - home[0]) ** 2 + (point[1] - home[1]) ** 2) ** .5 <= radius


def inspect(path):
    replay = sc2reader.load_replay(path, load_level=4)
    selected, homes = {}, {}
    repair, defend_home, examples = Counter(), Counter(), []
    for event in replay.events:
        pid = event_pid(event)
        if pid is None:
            continue
        event_type = type(event).__name__
        if event_type == "SelectionEvent":
            selected[pid] = list(getattr(event, "objects", []) or [])
            continue
        if event_type in ("UnitBornEvent", "UnitInitEvent"):
            name = (getattr(event, "unit_type_name", "") or "").lower()
            owner = getattr(getattr(event, "unit", None), "owner", None)
            owner_pid = getattr(owner, "pid", None)
            if owner_pid and owner_pid not in homes and any(word in name for word in TOWN_HALL_WORDS):
                unit = getattr(event, "unit", None)
                homes[owner_pid] = getattr(unit, "location", None)
            continue
        if "CommandEvent" not in event_type:
            continue
        ability = (getattr(event, "ability_name", "") or "").lower()
        kind = selection_kind(selected.get(pid, []))
        if "repair" in ability:
            repair[(kind, event_type)] += 1
            if len(examples) < 12:
                examples.append({"kind": "repair", "second": getattr(event, "second", 0), "selection": kind,
                                 "ability": getattr(event, "ability_name", None), "target": str(getattr(event, "target", None))})
        elif ("attack" in ability or ability == "move") and kind == "combat_squad" and near(getattr(event, "location", None), homes.get(pid)):
            defend_home[(kind, event_type)] += 1
            if len(examples) < 12:
                examples.append({"kind": "defend_home", "second": getattr(event, "second", 0), "selection": kind,
                                 "ability": getattr(event, "ability_name", None), "location": getattr(event, "location", None)})
    return repair, defend_home, examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--max-games", type=int, default=200)
    args = parser.parse_args()
    games = json.loads(Path(args.manifest).read_text())["valid"][:args.max_games]
    repair, defend, examples = Counter(), Counter(), []
    for index, item in enumerate(games, 1):
        try:
            r, d, e = inspect(item["path"])
        except Exception as exc:
            print(f"skip game={index} {type(exc).__name__}")
            continue
        repair.update(r); defend.update(d)
        examples.extend(e[: max(0, 12 - len(examples))])
    stringify = lambda counter: {" / ".join(key): value for key, value in counter.items()}
    print(json.dumps({"games_scanned": len(games), "repair_events": sum(repair.values()),
                      "repair_breakdown": stringify(repair), "defend_home_candidates": sum(defend.values()),
                      "defend_home_breakdown": stringify(defend), "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
