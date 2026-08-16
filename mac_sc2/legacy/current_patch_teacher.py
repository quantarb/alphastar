#!/usr/bin/env python3
"""Collect winning 5.0.16 Terran demonstrations against the built-in Easy bot.

This is deliberately a transparent curriculum teacher.  It is not presented as
a learned policy: its only job is to generate *current-patch* trajectories for
the Transformer student, avoiding the 4.9.2-to-5.0.16 replay mismatch.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')

import torch
from loguru import logger
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'mac_sc2/artifacts/current_patch_teacher_trajectories.pt'
ACTIONS = ('scv', 'supply', 'barracks', 'marine', 'attack', 'wait')


class CurrentPatchTeacher(BotAI):
    def __init__(self, difficulty, verbose=False):
        super().__init__()
        self.rows = []
        self.verbose = verbose
        self.hard = difficulty == Difficulty.Hard

    def features(self):
        return [
            min(63, int(self.time // 15)),
            min(63, int(self.minerals // 25)),
            min(31, max(0, int(self.supply_left))),
            min(31, self.workers.amount),
            min(15, self.structures(UnitTypeId.SUPPLYDEPOT).amount),
            min(15, self.structures(UnitTypeId.BARRACKS).amount),
            min(63, self.units(UnitTypeId.MARINE).amount),
        ]

    async def on_start(self):
        self.client.game_step = 32

    async def on_step(self, iteration):
        workers = self.workers.amount
        depots = self.structures(UnitTypeId.SUPPLYDEPOT).amount
        raxes = self.structures(UnitTypeId.BARRACKS).amount
        marines = self.units(UnitTypeId.MARINE).amount
        pending_depot = self.already_pending(UnitTypeId.SUPPLYDEPOT)
        pending_rax = self.already_pending(UnitTypeId.BARRACKS)
        label = 5

        # The 5.0.16 opening begins with eight workers. Establish production
        # first, then make an early three-Barracks Marine timing.
        desired_rax = 1 if workers < 12 else (3 if self.hard and workers >= 16 else (2 if marines < 8 else 3))
        if self.townhalls and self.supply_left <= 4 and self.minerals >= 100 and pending_depot == 0:
            label = 1
            await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first, placement_step=3)
        elif self.townhalls and self.workers and self.minerals >= 150 and raxes < desired_rax and pending_rax == 0:
            label = 2
            await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first, placement_step=3)
        elif self.structures(UnitTypeId.BARRACKS).ready.idle and self.minerals >= 50 and self.supply_left > 0:
            label = 3
            for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
                rax.train(UnitTypeId.MARINE)
        elif self.townhalls.idle and self.minerals >= 50 and workers < (16 if self.hard else 20):
            label = 0
            self.townhalls.idle.first.train(UnitTypeId.SCV)

        # Attack and production are concurrent. The label captures the strategic
        # attack transition while the executor will continue idle production.
        if marines >= (16 if self.hard else 12):
            label = 4
            if self.minerals >= 50 and self.supply_left > 0:
                for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
                    rax.train(UnitTypeId.MARINE)
            for marine in self.units(UnitTypeId.MARINE):
                marine.attack(self.enemy_start_locations[0])

        self.rows.append((self.features(), label))
        if self.verbose and iteration % 50 == 0:
            print(f'step={iteration:04d} action={ACTIONS[label]} scv={workers} depots={depots} rax={raxes} marines={marines}')

    async def on_end(self, game_result):
        self.result = str(game_result)
        print(f'Teacher result: {game_result}; rows={len(self.rows)}')


def play_episode(difficulty, verbose=False):
    bot = CurrentPatchTeacher(difficulty, verbose=verbose)
    result = run_game(
        maps.get('Simple64'),
        [Bot(Race.Terran, bot), Computer(Race.Zerg, difficulty)],
        realtime=False,
        game_time_limit=1800,
    )
    return {'features': torch.tensor([r[0] for r in bot.rows], dtype=torch.long),
            'labels': torch.tensor([r[1] for r in bot.rows], dtype=torch.long),
            'result': str(result)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--difficulty', choices=('easy', 'medium', 'hard'), default='easy')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    logger.remove(); logger.add(lambda m: print(m, end=''), level='INFO')
    difficulty = getattr(Difficulty, args.difficulty.title())
    episodes = []
    for number in range(args.episodes):
        print(f'=== Teacher episode {number + 1}/{args.episodes} ===')
        episodes.append(play_episode(difficulty, args.verbose))
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'actions': ACTIONS, 'episodes': episodes, 'patch': '5.0.16', 'difficulty': args.difficulty}, DATA_PATH)
    wins = sum(e['result'].endswith('Victory') for e in episodes)
    print(f'Saved {len(episodes)} current-patch episodes ({wins} wins): {DATA_PATH}')


if __name__ == '__main__':
    main()
