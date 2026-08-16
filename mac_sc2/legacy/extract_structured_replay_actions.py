#!/usr/bin/env python3
"""Extract race-labelled structured action examples from SC2 replay events.

This preserves the components needed by an AlphaStar-style action decoder:
selected entity types, ability, target entity type or target location, and a
short action history.  Full client observations are added later when map
dependencies are available; this extractor intentionally never invents them.
"""
import argparse
import json
from collections import deque
from pathlib import Path

import sc2reader
import torch

ROOT = Path(__file__).resolve().parents[1]
RACES = {'Terran': 0, 'Protoss': 1, 'Zerg': 2}


def command_target(event):
    target = getattr(event, 'target', None)
    if hasattr(target, 'type_id'):
        return int(target.type_id), (-1.0, -1.0)
    if hasattr(target, 'x') and hasattr(target, 'y'):
        return -1, (float(target.x), float(target.y))
    return -1, (-1.0, -1.0)


def extract(path):
    replay = sc2reader.load_replay(str(path), load_level=4)
    races = {player.pid: RACES.get(player.play_race) for player in replay.players}
    selected = {pid: [] for pid in races}
    history = {pid: deque([0] * 8, maxlen=8) for pid in races}
    rows = []
    for event in replay.events:
        pid = getattr(event, 'pid', None)
        if pid not in races or races[pid] is None:
            continue
        if type(event).__name__ == 'SelectionEvent':
            # sc2reader exposes replay unit-type IDs in the selection event's
            # compact tuple, rather than on its Unit wrapper.
            selected[pid] = [int(entry[0]) for entry in getattr(event, 'new_unit_types', [])][:32]
        if 'CommandEvent' not in type(event).__name__:
            continue
        ability = getattr(event, 'ability_id', None)
        if ability is None:
            continue
        target_type, target_point = command_target(event)
        rows.append({'race': races[pid], 'frame': int(event.frame), 'history': list(history[pid]),
                     'selected_types': selected[pid], 'ability': int(ability),
                     'target_type': target_type, 'target_point': target_point})
        history[pid].append(int(ability))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='local_data/current_replays/manifest_spawningtool_pro_2026_all_versions.json')
    parser.add_argument('--max-games', type=int, default=1021)
    parser.add_argument('--output', default='mac_sc2/artifacts/structured_replay_actions.pt')
    args = parser.parse_args()
    manifest = json.loads((ROOT / args.manifest).read_text())
    rows = []
    for index, record in enumerate(manifest['valid'][:args.max_games], 1):
        try:
            rows.extend(extract(Path(record['path'])))
        except Exception as error:
            print(f'skip={Path(record["path"]).name} error={type(error).__name__}', flush=True)
        if index % 25 == 0: print(f'games={index} actions={len(rows)}', flush=True)
    out = ROOT / args.output; out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'rows': rows, 'races': RACES, 'source_manifest': args.manifest,
                'games': min(args.max_games, len(manifest['valid']))}, out)
    print(f'saved={out} rows={len(rows)}', flush=True)


if __name__ == '__main__': main()
