#!/usr/bin/env python3
"""Contract and multi-race raw-replay smoke validation."""
import argparse, json
from collections import Counter
from pathlib import Path
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.patch_race_mtl import all_spec_hashes, build_specs
from mac_sc2.data.patch_race_exact import examples

def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--manifest',required=True);p.add_argument('--max-games',type=int,default=80);a=p.parse_args()
 specs=build_specs(a.registry); hashes=all_spec_hashes(a.registry)
 if not specs or any(not vocab for vocab in specs.values()): raise SystemExit('empty ActionSpec')
 rows=Counter(); discarded=Counter(); races=set()
 compatible=[item for item in json.loads(Path(a.manifest).read_text())['valid'] if '.'.join(item['version'].split('.')[:3])+'/Terran' in specs]
 for item in compatible[:a.max_games]:
  try:
   stream=examples(item['path'],item['version'],specs,discarded)
   for row in stream: rows[row['task']]+=1; races.add(row['task'].split('/')[-1])
  except Exception as exc: discarded[type(exc).__name__]+=1
 required={'Terran','Protoss','Zerg'}
 if not required.issubset(races): raise SystemExit(f'missing multi-race aligned labels: {required-races}')
 print(json.dumps({'tasks':len(specs),'action_spec_hashes':hashes,'snapshot_contract_hash':snapshot_hash(),'aligned_rows':dict(rows),'discarded':dict(discarded)},indent=2))
if __name__=='__main__':main()
