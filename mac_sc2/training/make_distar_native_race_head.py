#!/usr/bin/env python3
"""Create a compact current-patch race action head from released DI-star weights."""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--distar-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--race', choices=('terran', 'protoss'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    import sys
    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.agent.default.lib.current_patch_actions import race_legacy_action_indices
    from distar.agent.default.model.model import Model
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import legacy_race_contract_hash
    source = torch.load(args.source, map_location='cpu', weights_only=False)
    cfg = read_config(str(args.distar_root / 'distar/bin/sl_user_config.yaml'))
    cfg.common.type = 'sl'; cfg.native_action_race = args.race
    target = Model(cfg).state_dict(); old = source['model']
    index = torch.tensor(race_legacy_action_indices(args.race), dtype=torch.long)
    changed = {'policy.action_type_head.action_fc.layer2.0.weight',
               'policy.action_type_head.action_fc.layer2.0.bias',
               'policy.action_type_head.action_map_fc1.0.weight'}
    for name, tensor in target.items():
        if name in changed: continue
        if name not in old or old[name].shape != tensor.shape: raise RuntimeError(f'incompatible tensor: {name}')
        target[name] = old[name]
    target['policy.action_type_head.action_fc.layer2.0.weight'] = old['policy.action_type_head.action_fc.layer2.0.weight'].index_select(0,index)
    target['policy.action_type_head.action_fc.layer2.0.bias'] = old['policy.action_type_head.action_fc.layer2.0.bias'].index_select(0,index)
    target['policy.action_type_head.action_map_fc1.0.weight'] = old['policy.action_type_head.action_map_fc1.0.weight'].index_select(1,index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model':target,'last_iter':0,'current_patch_contract_hash':legacy_race_contract_hash(args.race),
                'native_action_race':args.race,'resumed_from':str(args.source.resolve()),
                'native_action_count':len(index)},args.output)
    print({'race':args.race,'actions':len(index),'output':str(args.output)})
if __name__ == '__main__': main()
