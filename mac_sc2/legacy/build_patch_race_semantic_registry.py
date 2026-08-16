#!/usr/bin/env python3
"""Build an exact, task-local action registry directly from raw replays.

The registry is deliberately an index rather than a materialized ML dataset.
Each patch/race task retains the replay ability name, observed target name, and
relative target location in addition to the portable semantic factors.  This
is the minimum evidence needed to construct a faithful task-local action
vocabulary without pretending that two differently named abilities are one
label.
"""
import argparse, json
import multiprocessing as mp
from collections import Counter, defaultdict
from pathlib import Path

from extract_semantic_replay_actions import actions_for_replay


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


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4);p.add_argument('--max-games',type=int);p.add_argument('--patch');a=p.parse_args()
    items=json.loads(Path(a.manifest).read_text())['valid']
    if a.patch:
        items=[item for item in items if '.'.join(item['version'].split('.')[:3]) == a.patch]
    items=items[:a.max_games] if a.max_games else items; tasks=defaultdict(Counter)
    # sc2reader is CPU-heavy.  Workers return only Counters, never replay data.
    with mp.get_context('spawn').Pool(processes=max(1, a.workers)) as pool:
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
    out={
        'schema_version': 2,
        'description': 'Exact observed replay action labels, partitioned by patch and player race.',
        'replays':len(items),
        'tasks':{task:[row(k,v) for k,v in counter.most_common()] for task,counter in tasks.items()},
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out));print(f'saved={a.output} tasks={len(tasks)}')
if __name__=='__main__':main()
