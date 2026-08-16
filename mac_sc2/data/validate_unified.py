#!/usr/bin/env python3
"""Validate all contracts before a unified training run."""
import argparse
from mac_sc2.contracts.placement_spec import actions
from mac_sc2.contracts.semantic import ACTIONS
from mac_sc2.contracts.unified import policy_hash

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--registry',required=True);args=parser.parse_args()
    placement=actions(args.registry)
    if not placement: raise SystemExit('no live placement actions')
    if any(action.actor not in ('worker','production','combat') for action in ACTIONS): raise SystemExit('macro actor is not executable')
    print(f'unified ActionSpec={policy_hash(args.registry)} macro={len(ACTIONS)} placement={len(placement)} repair=SCVRepair candidates=SC2_query_building_placement')
if __name__=='__main__':main()
