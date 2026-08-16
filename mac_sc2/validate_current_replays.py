#!/usr/bin/env python3
"""Create a reproducible manifest of replay files compatible with SC2 5.0.16."""
import argparse
import json
from pathlib import Path
import sc2reader

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--source', default='spawningtool_2026',
                    help='subdirectory below local_data/current_replays')
parser.add_argument('--patch-prefix', default='5.0.16')
parser.add_argument('--all-versions', action='store_true',
                    help='inventory every readable replay instead of filtering by patch')
arguments = parser.parse_args()
source = ROOT / 'local_data/current_replays' / arguments.source
valid, rejected = [], []
for replay_path in sorted(source.glob('*.SC2Replay')):
    try:
        replay = sc2reader.load_replay(str(replay_path), load_level=1)
        version = replay.release_string
        (valid if arguments.all_versions or version.startswith(arguments.patch_prefix) else rejected).append({'path': str(replay_path), 'version': version})
    except Exception as error:
        rejected.append({'path': str(replay_path), 'error': type(error).__name__})
suffix = 'all_versions' if arguments.all_versions else arguments.patch_prefix.replace('.', '_')
out = ROOT / 'local_data/current_replays' / f'manifest_{arguments.source}_{suffix}.json'
out.write_text(json.dumps({'source': str(source), 'patch_prefix': arguments.patch_prefix,
                           'valid': valid, 'rejected': rejected}, indent=2))
print(f'valid={len(valid)} rejected={len(rejected)} manifest={out}')
