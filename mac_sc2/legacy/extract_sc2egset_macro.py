#!/usr/bin/env python3
"""Extract Terran macro behavior-cloning examples from SC2EGSet JSON replays.

Each training row is a compact game-state summary immediately before a human
player starts a key Terran macro item: SCV, Supply Depot, Barracks, or Marine.
This runs entirely offline and is portable to macOS.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import torch

LABELS = {'SCV': 0, 'SupplyDepot': 1, 'Barracks': 2, 'Marine': 3}


def encoded_state(loop, stats, counts):
    minerals = stats.get('scoreValueMineralsCurrent', 0)
    used = stats.get('scoreValueFoodUsed', 0)
    made = stats.get('scoreValueFoodMade', 0)
    workers = stats.get('scoreValueWorkersActiveCount', counts['SCV'])
    return [min(31, loop // 320), min(31, minerals // 50), min(15, max(0, made - used)),
            min(31, workers), min(15, counts['SupplyDepot']), min(15, counts['Barracks']),
            min(31, counts['Marine'])]


def examples_from_replay(data):
    events = sorted(data.get('trackerEvents', []), key=lambda e: e.get('loop', 0))
    counts = defaultdict(lambda: defaultdict(int))
    stats = defaultdict(dict)
    result = []
    terran = set()
    for event in events:
        typ = event.get('evtTypeName')
        player = event.get('upkeepPlayerId') or event.get('playerId')
        if typ == 'PlayerStats':
            stats[event.get('playerId')] = event.get('stats', {})
            continue
        name = event.get('unitTypeName')
        if typ in ('UnitBorn', 'UnitInit') and player and name:
            if name in LABELS:
                terran.add(player)
                result.append((encoded_state(event.get('loop', 0), stats[player], counts[player]), LABELS[name]))
                counts[player][name] += 1
            elif player in terran and name in ('CommandCenter', 'Factory', 'Starport', 'Refinery'):
                counts[player][name] += 1
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='local_data/sc2egset/processed')
    parser.add_argument('--output', default='mac_sc2/artifacts/sc2egset_macro_examples.pt')
    parser.add_argument('--limit', type=int, default=200000)
    args = parser.parse_args()
    rows = []
    for path in Path(args.input).rglob('*.SC2Replay.json'):
        try:
            rows.extend(examples_from_replay(json.loads(path.read_text())))
        except Exception as exc:
            print(f'Skipping {path.name}: {exc}')
        if len(rows) >= args.limit:
            break
    if not rows:
        raise SystemExit(f'No Terran macro examples found under {args.input}')
    rows = rows[:args.limit]
    x = torch.tensor([state for state, _ in rows], dtype=torch.long)
    y = torch.tensor([label for _, label in rows], dtype=torch.long)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'features': x, 'labels': y, 'class_labels': LABELS}, output)
    print(f'Wrote {len(rows):,} macro BC examples to {output.resolve()}')
    print({name: int((y == idx).sum()) for name, idx in LABELS.items()})


if __name__ == '__main__':
    main()
