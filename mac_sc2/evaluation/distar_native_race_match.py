#!/usr/bin/env python3
"""Run one completed current-patch DI-star race-head evaluation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--distar-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--race', choices=('terran', 'protoss', 'zerg'), required=True)
    parser.add_argument('--difficulty', choices=('veryeasy', 'easy', 'medium', 'hard', 'veryhard'), default='hard')
    parser.add_argument('--replay-dir', type=Path, required=True)
    parser.add_argument('--result-json', type=Path, required=True)
    parser.add_argument('--visible', action='store_true', help='show the SC2 game window')
    parser.add_argument('--realtime', action='store_true', help='advance at wall-clock game speed')
    parser.add_argument('--multi-race-action-heads', action='store_true')
    parser.add_argument('--released-zerg-mask', action='store_true',
                        help='run the un-fine-tuned 327-way released Zerg RL head behind the current-patch mask')
    args = parser.parse_args()

    sys.path.insert(0, str(args.distar_root.resolve()))
    from distar.actor import Actor
    from distar.ctools.utils import read_config
    from distar.current_patch_contract import (
        NATIVE_POLICY_ACTION_ENCODING,
        contract_hash,
        legacy_race_contract_hash,
    )

    args.replay_dir.mkdir(parents=True, exist_ok=True)
    cfg = read_config(str(args.distar_root / 'distar/bin/user_config.yaml'))
    player_id = 'candidate'
    cfg.common.type = 'play'
    cfg.common.experiment_name = f'distar_current_patch_{args.race}_hard_eval'
    if args.released_zerg_mask and args.race != 'zerg':
        parser.error('--released-zerg-mask is only valid for Zerg')
    cfg.native_action_race = None if args.released_zerg_mask else args.race
    cfg.multi_race_action_heads = args.multi_race_action_heads
    cfg.native_zerg_action_head = args.race == 'zerg' and not args.released_zerg_mask
    cfg.current_patch_race = args.race
    if args.released_zerg_mask:
        # The released 327-way RL action head remains intact.  The policy's
        # current-client Zerg legality mask is its executable ActionSpec.
        cfg.current_patch_contract_mode = 'masked_released_zerg'
        cfg.policy_action_encoding = 'released_327_id_with_current_zerg_mask'
        cfg.current_patch_contract_hash = contract_hash(cfg.policy_action_encoding)
    elif args.multi_race_action_heads:
        from distar.current_patch_contract import multi_race_contract_hash
        cfg.current_patch_contract_mode = 'multi_race'
        cfg.current_patch_contract_hash = multi_race_contract_hash()
    elif args.race == 'zerg' and not args.multi_race_action_heads:
        # The original compact Zerg checkpoint has a 113-way local output
        # (no-op + 112 legal actions), rather than the later 327-way
        # race-masked contract used by the Terran/Protoss adapters.
        cfg.current_patch_contract_mode = 'native_zerg'
        cfg.current_patch_contract_hash = contract_hash(NATIVE_POLICY_ACTION_ENCODING)
    else:
        cfg.current_patch_contract_mode = 'legacy_race'
        cfg.current_patch_contract_hash = legacy_race_contract_hash(args.race)
    cfg.require_current_patch_contract = True
    cfg.agent.disable_z_strategy = args.race != 'zerg'
    cfg.actor.job_type = 'eval_test'
    cfg.actor.episode_num = 1
    cfg.actor.use_cuda = False
    cfg.actor.player_ids = [player_id]
    cfg.actor.agents = {player_id: 'default'}
    cfg.actor.model_paths = {player_id: str(args.checkpoint.resolve())}
    cfg.env.player_ids = [player_id, 'bot5']
    cfg.env.races = [args.race, 'zerg']
    cfg.env.visible = args.visible
    cfg.env.realtime = args.realtime
    # Evaluation should measure the policy, not DI-star's training-time
    # simulated network/action latency.  This also keeps SC2 controllers in
    # lockstep for deterministic non-realtime matches.
    cfg.env.random_delay_weights = [1]
    cfg.env.map_name = 'NewRepugnancy'
    cfg.env.replay_dir = str(args.replay_dir.resolve())
    cfg.env.match_result_path = str(args.result_json.resolve())
    cfg.env.game_steps_per_episode = 100000
    cfg.env.version = 'latest'
    print({'checkpoint': str(args.checkpoint.resolve()), 'race': args.race,
           'difficulty': args.difficulty, 'contract': cfg.current_patch_contract_hash}, flush=True)
    # DI-star names builtin levels bot0..bot10; bot5 maps to Hard.
    cfg.env.player_ids[1] = 'bot5'
    Actor(cfg).run()


if __name__ == '__main__':
    main()
