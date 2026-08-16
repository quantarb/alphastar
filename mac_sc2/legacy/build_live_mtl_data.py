#!/usr/bin/env python3
"""Create live-compatible, race-labelled macro examples from replay commands.

The vectors intentionally contain only quantities a BotAI can recreate at
inference time: elapsed time, compact selected-unit histogram, and preceding
macro actions.  This is a transitional BC dataset; it does not pretend the
replay parser provides hidden game observations such as fog-of-war state.
"""
import argparse, json
from collections import deque
from pathlib import Path
import sc2reader, torch

ROOT = Path(__file__).resolve().parents[1]
RACE_IDS = {"Terran": 0, "Protoss": 1, "Zerg": 2}

def macro(name):
    name = (name or "").lower()
    if "attack" in name or "scanmove" in name: return 5
    if any(x in name for x in ("commandcenter", "nexus", "hatchery", "expand")): return 4
    if any(x in name for x in ("supplydepot", "pylon", "overlord")): return 1
    if any(x in name for x in ("barracks", "gateway", "spawningpool", "factory", "cybernetics")): return 2
    if any(x in name for x in ("scv", "probe", "drone")): return 0
    if any(x in name for x in ("marine", "zealot", "zergling", "roach", "stalker", "marauder", "hellion")): return 3
    return 6

def vector(frame, selected, history):
    v = [min(frame, 40000) / 40000]
    buckets = [0] * 7
    for unit in selected: buckets[int(unit) % 7] += 1
    v += [min(x, 16) / 16 for x in buckets]
    v += [x / 6 for x in list(history)[-8:]]
    return v

def replay_rows(path):
    replay = sc2reader.load_replay(str(path), load_level=4)
    races = {p.pid: RACE_IDS.get(p.play_race) for p in replay.players}
    selected = {pid: [] for pid in races}; histories = {pid: deque([6] * 8, maxlen=8) for pid in races}; rows=[]
    for event in replay.events:
        pid = getattr(event, "pid", None)
        if pid not in races or races[pid] is None: continue
        if type(event).__name__ == "SelectionEvent":
            selected[pid] = [int(x[0]) for x in getattr(event, "new_unit_types", [])][:32]
        if "CommandEvent" not in type(event).__name__: continue
        label = macro(getattr(event, "ability_name", ""))
        rows.append((races[pid], vector(int(event.frame), selected[pid], histories[pid]), label))
        histories[pid].append(label)
    return rows

def main():
    p=argparse.ArgumentParser(); p.add_argument("--manifest", required=True); p.add_argument("--max-games", type=int); p.add_argument("--output", required=True); a=p.parse_args()
    manifest=json.loads(Path(a.manifest).read_text()); valid=manifest["valid"][:a.max_games]
    rows=[]
    for i, item in enumerate(valid, 1):
        try: rows += replay_rows(Path(item["path"]))
        except Exception as e: print(f"skip={Path(item['path']).name} {type(e).__name__}")
        if i % 100 == 0: print(f"games={i} rows={len(rows)}", flush=True)
    torch.save({"rows": rows, "games": len(valid), "state_features": 16}, a.output)
    print(f"saved={a.output} rows={len(rows)}")
if __name__ == "__main__": main()
