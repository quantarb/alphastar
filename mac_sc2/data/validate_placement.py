#!/usr/bin/env python3
"""Fail closed before learned-placement training starts."""
import argparse, json, sys
from pathlib import Path
from sc2.ids.ability_id import AbilityId
from mac_sc2.contracts.placement_spec import actions, spec_hash
def main():
 p=argparse.ArgumentParser();p.add_argument('--registry',required=True);p.add_argument('--catalog',required=True);a=p.parse_args()
 rows=actions(a.registry); catalog={int(x['id']) for x in json.loads(Path(a.catalog).read_text())}; known={x.value for x in AbilityId};bad=[]
 for row in rows:
  if row.ability_id not in known or row.ability_id not in catalog:bad.append(f'unknown live ability {row}')
  if row.actor_role not in ('worker','production'):bad.append(f'incompatible actor {row}')
  if row.family!='build' and not row.ability.lower().startswith('land'):bad.append(f'not placement command {row}')
 if bad:raise SystemExit('invalid placement contract:\n'+'\n'.join(bad[:20]))
 print(f'valid placement ActionSpec hash={spec_hash(a.registry)} actions={len(rows)} actors=worker,production candidates=SC2_query_building_placement')
if __name__=='__main__':main()
