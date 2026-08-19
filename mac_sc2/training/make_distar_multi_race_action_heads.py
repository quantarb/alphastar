#!/usr/bin/env python3
"""Create a three-race compact-head DI-star checkpoint from released weights."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--distar-root', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.agent.default.lib.current_patch_actions import race_legacy_action_indices
    from distar.agent.default.model.model import Model
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import multi_race_contract_hash

    source = torch.load(args.source, map_location='cpu', weights_only=False)['model']
    cfg = read_config(str(args.distar_root / 'distar/bin/sl_user_config.yaml'))
    cfg.common.type = 'sl'
    cfg.multi_race_action_heads = True
    cfg.current_patch_race = 'zerg'
    target = Model(cfg).state_dict()
    for name, tensor in list(target.items()):
        if name.startswith('policy.action_type_head.action_heads.') or name.startswith('policy.action_type_head.action_map_fc1_by_race.'):
            continue
        if name in source and source[name].shape == tensor.shape:
            target[name] = source[name]
    for race in ('zerg', 'terran', 'protoss'):
        index = torch.tensor(race_legacy_action_indices(race), dtype=torch.long)
        prefix = f'policy.action_type_head.action_heads.{race}.layer2.0.'
        target[prefix + 'weight'] = source['policy.action_type_head.action_fc.layer2.0.weight'].index_select(0, index)
        target[prefix + 'bias'] = source['policy.action_type_head.action_fc.layer2.0.bias'].index_select(0, index)
        target[f'policy.action_type_head.action_map_fc1_by_race.{race}.0.weight'] = source[
            'policy.action_type_head.action_map_fc1.0.weight'].index_select(1, index)
        # The bias is action-count invariant and comes directly from released weights.
        target[f'policy.action_type_head.action_map_fc1_by_race.{race}.0.bias'] = source[
            'policy.action_type_head.action_map_fc1.0.bias']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'model': target, 'last_iter': 0,
                'current_patch_contract_hash': multi_race_contract_hash(),
                'policy_action_encoding': 'multi_race_native_action_heads_v1',
                'multi_race_action_heads': True,
                'races': {'zerg': 113, 'terran': 138, 'protoss': 129},
                'resumed_from': str(args.source.resolve())}, args.output)
    print({'output': str(args.output.resolve()), 'contract': multi_race_contract_hash(),
           'heads': {'zerg': 113, 'terran': 138, 'protoss': 129}})


if __name__ == '__main__':
    main()
