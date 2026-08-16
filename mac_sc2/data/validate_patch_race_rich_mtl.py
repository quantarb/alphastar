#!/usr/bin/env python3
"""Contract and multi-race raw-replay smoke validation."""
import argparse, json
from collections import Counter
from pathlib import Path
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.patch_race_mtl import all_spec_hashes, build_specs
from mac_sc2.data.patch_race_exact import examples


def compatible_replays(manifest, specs):
    items=json.loads(Path(manifest).read_text())['valid']
    return [item for item in items if f"{'.'.join(item['version'].split('.')[:3])}/Terran" in specs]


def validate_extractor_versions(manifest, specs):
    """Smoke both the live patch and one older replay without cross-patch labels."""
    items = json.loads(Path(manifest).read_text())["valid"]
    live = next(item for item in items if f"{'.'.join(item['version'].split('.')[:3])}/Terran" in specs)
    older = next(item for item in items if f"{'.'.join(item['version'].split('.')[:3])}/Terran" not in specs)
    result = {}
    for name, item in (("live", live), ("older", older)):
        discarded = Counter()
        rows = sum(1 for _ in examples(item["path"], item["version"], specs, discarded))
        result[name] = {"version": item["version"], "rows": rows, "discarded": dict(discarded)}
    if not result["live"]["rows"] or result["older"]["discarded"].get("no_live_task", 0) == 0:
        raise ValueError("extractor did not preserve live/older-patch behavior")
    return result


def validate_alignment(registry, manifest, max_games=80):
    specs=build_specs(registry); hashes=all_spec_hashes(registry)
    if not specs or any(not vocab for vocab in specs.values()): raise ValueError('empty ActionSpec')
    rows=Counter(); discarded=Counter(); races=set()
    for item in compatible_replays(manifest,specs)[:max_games]:
        try:
            for row in examples(item['path'],item['version'],specs,discarded):
                rows[row['task']]+=1; races.add(row['task'].split('/')[-1])
        except Exception as exc: discarded[type(exc).__name__]+=1
    missing={'Terran','Protoss','Zerg'}-races
    if missing: raise ValueError(f'missing multi-race aligned labels: {missing}')
    return {'tasks':len(specs),'action_spec_hashes':hashes,'snapshot_contract_hash':snapshot_hash(),'aligned_rows':dict(rows),'discarded':dict(discarded),
            'extractor_versions': validate_extractor_versions(manifest, specs)}

def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--manifest',required=True);p.add_argument('--max-games',type=int,default=80);a=p.parse_args()
 specs=build_specs(a.registry); hashes=all_spec_hashes(a.registry)
 if not specs or any(not vocab for vocab in specs.values()): raise SystemExit('empty ActionSpec')
 rows=Counter(); discarded=Counter(); races=set()
 compatible=compatible_replays(a.manifest,specs)
 for item in compatible[:a.max_games]:
  try:
   stream=examples(item['path'],item['version'],specs,discarded)
   for row in stream: rows[row['task']]+=1; races.add(row['task'].split('/')[-1])
  except Exception as exc: discarded[type(exc).__name__]+=1
 required={'Terran','Protoss','Zerg'}
 if not required.issubset(races): raise SystemExit(f'missing multi-race aligned labels: {required-races}')
 print(json.dumps({'tasks':len(specs),'action_spec_hashes':hashes,'snapshot_contract_hash':snapshot_hash(),'aligned_rows':dict(rows),'discarded':dict(discarded)},indent=2))
if __name__=='__main__':main()
