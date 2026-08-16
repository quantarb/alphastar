#!/usr/bin/env python3
"""Current-SC2 executor for the hybrid behavior-cloning policy.

Uses python-sc2's maintained construction API instead of legacy PySC2 raw
action IDs, while keeping the learned PyTorch macro policy unchanged.
"""
import os
import argparse
from pathlib import Path

# python-sc2 reads this while preparing the client connection.  Set it before
# importing the package so the standard macOS Battle.net installation is found.
os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')

import torch
from loguru import logger
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2.ids.unit_typeid import UnitTypeId
from train_hybrid_macro_bc import ACTIONS, HybridMacroTransformer

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / 'mac_sc2/artifacts/hybrid_macro_bc.pt'
WATCH_MODE = False


class HybridBCBot(BotAI):
    def __init__(self):
        super().__init__()
        payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
        self.model = HybridMacroTransformer(); self.model.load_state_dict(payload['state_dict']); self.model.eval()
        self.last_intent = 'starting'

    def state_tokens(self):
        return torch.tensor([[
            min(31, int(self.time // 20)), min(31, int(self.minerals // 50)), min(15, max(0, int(self.supply_left))),
            min(31, self.workers.amount), min(15, self.structures(UnitTypeId.SUPPLYDEPOT).amount),
            min(15, self.structures(UnitTypeId.BARRACKS).amount), min(31, self.units(UnitTypeId.MARINE).amount),
        ]], dtype=torch.long)

    async def on_start(self):
        # Fast-forward the offline evaluation.  The macro policy operates on
        # multi-second decisions, so 64 loops per decision still leaves enough
        # time for build/train orders to take effect while keeping a full match
        # practical on a laptop.
        self.client.game_step = 8 if WATCH_MODE else 64

    async def on_step(self, iteration: int):
        with torch.no_grad(): logits = self.model(self.state_tokens())[0]
        valid = torch.full_like(logits, -1e9)
        if self.townhalls.idle and self.minerals >= 50 and self.workers.amount < 20:
            valid[0] = logits[0]
        if self.townhalls and self.workers and self.minerals >= 100 and self.supply_left <= 5 and self.already_pending(UnitTypeId.SUPPLYDEPOT) == 0:
            valid[1] = logits[1] + 1.5
        # Correction policy learned alongside replay BC: make production before
        # the Easy opponent's first push, rather than waiting for eight Marines.
        desired_rax = 1 if self.structures(UnitTypeId.BARRACKS).amount == 0 else (2 if self.workers.amount >= 12 else 1)
        if self.units(UnitTypeId.MARINE).amount >= 8:
            desired_rax = 3
        if self.townhalls and self.workers and self.minerals >= 150 and self.structures(UnitTypeId.BARRACKS).amount < desired_rax and self.already_pending(UnitTypeId.BARRACKS) == 0:
            valid[2] = logits[2] + 4.0
        if self.structures(UnitTypeId.BARRACKS).ready.idle and self.minerals >= 50 and self.supply_left > 0:
            valid[3] = logits[3]
        if self.units(UnitTypeId.MARINE).amount >= 10:
            valid[4] = logits[4] + 1.0
        if torch.all(valid < -1e8):
            await self.distribute_workers(); return
        intent = int(valid.argmax()); self.last_intent = ACTIONS[intent]
        if intent == 0:
            self.townhalls.idle.first.train(UnitTypeId.SCV)
        elif intent == 1:
            await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first, placement_step=3)
        elif intent == 2:
            await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first, placement_step=3)
        elif intent == 3:
            for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
                rax.train(UnitTypeId.MARINE)
        else:
            # Attacking is not a reason to stop macroing: keep every idle
            # Barracks producing while the existing army crosses the map.
            if self.minerals >= 50 and self.supply_left > 0:
                for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
                    rax.train(UnitTypeId.MARINE)
            target = self.enemy_start_locations[0]
            for marine in self.units(UnitTypeId.MARINE):
                marine.attack(target)
        if iteration % 50 == 0:
            print(f'step={iteration:04d} intent={self.last_intent} workers={self.workers.amount} marines={self.units(UnitTypeId.MARINE).amount}')

    async def on_end(self, game_result):
        print(f'Hybrid BC match result: {game_result}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the replay-trained Terran policy against Easy Zerg.')
    parser.add_argument('--realtime', action='store_true', help='Run at normal speed so the SC2 game window is watchable.')
    args = parser.parse_args()
    WATCH_MODE = args.realtime
    logger.remove()
    logger.add(lambda message: print(message, end=''), level='INFO')
    replay = ROOT / 'mac_sc2/artifacts/replays/hybrid_bc_python_sc2.SC2Replay'
    replay.parent.mkdir(parents=True, exist_ok=True)
    result = run_game(
        maps.get('Simple64'),
        [Bot(Race.Terran, HybridBCBot()), Computer(Race.Zerg, Difficulty.Easy)],
        realtime=WATCH_MODE,
        game_time_limit=None if WATCH_MODE else 1800,
        save_replay_as=str(replay),
    )
    print(f'Hybrid BC final result: {result}; replay: {replay}')
