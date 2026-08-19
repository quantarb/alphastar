#!/usr/bin/env python3
"""Run the released masked Zerg RL policy against the compact STL Zerg policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--distar-root', type=Path, required=True)
    parser.add_argument('--released-checkpoint', type=Path, required=True)
    parser.add_argument('--stl-checkpoint', type=Path, required=True)
    parser.add_argument('--replay-dir', type=Path, required=True)
    parser.add_argument('--result-json', type=Path, required=True)
    parser.add_argument('--visible', action='store_true')
    args = parser.parse_args()

    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.actor import Actor
    from distar.agent.default.lib.current_patch_actions import LEGACY_ACTION_INDICES
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import NATIVE_POLICY_ACTION_ENCODING, contract_hash

    released = torch.load(args.released_checkpoint, map_location='cpu', weights_only=False)
    stl = torch.load(args.stl_checkpoint, map_location='cpu', weights_only=False)
    masked_encoding = 'released_327_id_with_current_zerg_mask'
    if released.get('current_patch_contract_hash') != contract_hash(masked_encoding):
        raise RuntimeError('released checkpoint ActionSpec mismatch')
    if stl.get('current_patch_contract_hash') != contract_hash(NATIVE_POLICY_ACTION_ENCODING):
        raise RuntimeError('STL checkpoint ActionSpec mismatch')

    # Actor uses one model configuration for both players.  Expand the compact
    # Zerg output back into its exact released action IDs; the legality mask
    # prevents every non-Zerg row from being emitted.  All non-output tensors
    # come directly from the STL checkpoint.
    expanded = {name: value.clone() for name, value in released['model'].items()}
    compact = stl['model']
    for name, value in compact.items():
        if name not in expanded:
            continue
        if value.shape == expanded[name].shape:
            expanded[name] = value.clone()
    index = torch.tensor(LEGACY_ACTION_INDICES, dtype=torch.long)
    for name, dimension in (
        ('policy.action_type_head.action_fc.layer2.0.weight', 0),
        ('policy.action_type_head.action_fc.layer2.0.bias', 0),
        ('policy.action_type_head.action_map_fc1.0.weight', 1),
    ):
        expanded[name].index_copy_(dimension, index, compact[name])
    adapter_path = args.result_json.with_suffix('.stl_expanded_327.pt')
    torch.save({'model': expanded, 'last_iter': stl.get('last_iter'),
                'current_patch_contract_hash': contract_hash(masked_encoding),
                'policy_action_encoding': masked_encoding,
                'adapted_from': str(args.stl_checkpoint.resolve())}, adapter_path)

    args.replay_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_config(str(args.distar_root / 'distar/bin/user_config.yaml'))
    cfg.common.type = 'play'; cfg.common.experiment_name = 'distar_released_vs_stl_zerg'
    cfg.native_action_race = None; cfg.native_zerg_action_head = False
    cfg.current_patch_race = 'zerg'; cfg.policy_action_encoding = masked_encoding
    cfg.current_patch_contract_mode = 'masked_released_zerg'
    cfg.current_patch_contract_hash = contract_hash(masked_encoding)
    cfg.require_current_patch_contract = True
    cfg.agent.disable_z_strategy = False
    cfg.actor.job_type = 'eval_test'; cfg.actor.episode_num = 1; cfg.actor.use_cuda = False
    cfg.actor.player_ids = ['released_rl', 'stl_zerg']
    cfg.actor.agents = {'released_rl': 'default', 'stl_zerg': 'default'}
    cfg.actor.model_paths = {'released_rl': str(args.released_checkpoint.resolve()),
                             'stl_zerg': str(adapter_path.resolve())}
    cfg.env.player_ids = ['released_rl', 'stl_zerg']; cfg.env.races = ['zerg', 'zerg']
    cfg.env.visible = args.visible; cfg.env.realtime = False; cfg.env.map_name = 'NewRepugnancy'
    # DI-star's training environment injects a 0--3 loop latency at each
    # decision.  With two independent agents that can advance one controller
    # ahead of the other on the current client.  A deterministic zero-latency
    # scheduler is the correct evaluation setting for an agent-vs-agent game.
    cfg.env.random_delay_weights = [1]
    cfg.env.replay_dir = str(args.replay_dir.resolve()); cfg.env.match_result_path = str(args.result_json.resolve())
    cfg.env.game_steps_per_episode = 100000; cfg.env.version = 'latest'
    print({'released_rl': str(args.released_checkpoint.resolve()), 'stl_zerg': str(args.stl_checkpoint.resolve()),
           'stl_adapter': str(adapter_path.resolve()), 'contract': cfg.current_patch_contract_hash}, flush=True)
    Actor(cfg).run()
    # Actor records the literal result from player 0; retain the fixture identity.
    if not args.result_json.exists():
        raise RuntimeError('SC2 ended without a literal match result')
    result = json.loads(args.result_json.read_text())
    result.update({'player_0': 'released_rl_zerg_masked', 'player_1': 'stl_zerg_compact_expanded'})
    args.result_json.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
