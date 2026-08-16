#!/usr/bin/env python3
"""Build an exact, task-local action registry directly from raw replays.

The registry is deliberately an index rather than a materialized ML dataset.
Each patch/race task retains the replay ability name, observed target name, and
relative target location in addition to the portable semantic factors.  This
is the minimum evidence needed to construct a faithful task-local action
vocabulary without pretending that two differently named abilities are one
label.
"""
from __future__ import annotations

import json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

from mac_sc2.data.semantic_actions import actions_for_replay


def scan_replay(item):
    """Parse one replay in an isolated worker and return compact counts."""
    patch = '.'.join(item['version'].split('.')[:3])
    counter = Counter()
    try:
        for x in actions_for_replay(item['path'], patch):
            counter[(f'{patch}:{x.race}', x.actor_role, x.family,
                     x.payload_role, x.target_kind, int(x.queued),
                     x.ability_id, x.ability_name, x.target_name)] += 1
        return counter, None
    except Exception as exc:
        return Counter(), type(exc).__name__


def build_registry(manifest: str, patches: set[str] | None = None, workers: int = 4, max_games: int | None = None) -> dict:
    """Build raw replay ActionSpecs for any observed patch without shards.

    The result is research metadata.  It does not claim that historical
    tuples are executable by the installed 4.9.2 client.
    """
    items=json.loads(Path(manifest).read_text())['valid']
    if patches:
        items=[item for item in items if '.'.join(item['version'].split('.')[:3]) in patches]
    items=items[:max_games] if max_games else items; tasks=defaultdict(Counter)
    # sc2reader is CPU-heavy.  Workers return only Counters, never replay data.
    with mp.get_context('spawn').Pool(processes=max(1, workers)) as pool:
        for i, (counter, error) in enumerate(pool.imap_unordered(scan_replay, items, chunksize=8), 1):
            if error:
                print(f'skip game={i} {error}', flush=True)
                continue
            for (task, actor, family, payload, target, queued, ability_id, ability, target_name), count in counter.items():
                # Raw replay names are patch-local labels. They must remain
                # distinct even when their broad semantic family is the same.
                tasks[task][(actor, family, payload, target, queued, ability_id, ability, target_name)] += count
            if i % 200 == 0: print(f'games={i} tasks={len(tasks)}',flush=True)
    def row(key, count):
        return {
            'actor': key[0], 'family': key[1], 'payload': key[2],
            'target_kind': key[3], 'queued': bool(key[4]),
            'ability_id': key[5], 'ability_name': key[6],
            'target_name': key[7], 'count': count,
        }
    return {
        'schema_version': 2,
        'description': 'Research-only exact replay labels, partitioned by patch and player race.',
        'replays':len(items),
        'tasks':{task:[row(k,v) for k,v in counter.most_common()] for task,counter in tasks.items()},
    }


def write_registry(manifest: str, output: str, patches: set[str] | None = None, workers: int = 4, max_games: int | None = None) -> dict:
    registry = build_registry(manifest, patches, workers, max_games)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(registry))
    return registry
