#!/usr/bin/env python3
"""Generate a stronger current-patch Terran curriculum demonstration."""
import os
import json
from pathlib import Path
os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer


class HardCurriculumTeacher(BotAI):
    async def on_start(self):
        self.client.game_step = 24

    async def on_step(self, iteration):
        # Economic and production priorities are intentionally explicit here:
        # this is a data-collection teacher, not the deployed learned policy.
        if self.townhalls.idle and self.minerals >= 50 and self.workers.amount < 38:
            self.townhalls.idle.first.train(UnitTypeId.SCV)
        if self.supply_left < 6 and self.minerals >= 100 and not self.already_pending(UnitTypeId.SUPPLYDEPOT):
            await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first)
        if self.gas_buildings.amount < 2 and self.workers and self.minerals >= 75:
            for geyser in self.vespene_geyser.closer_than(12, self.townhalls.first):
                if not self.gas_buildings.closer_than(1, geyser):
                    await self.build(UnitTypeId.REFINERY, geyser); break
        if self.structures(UnitTypeId.BARRACKS).amount < 3 and self.minerals >= 150 and not self.already_pending(UnitTypeId.BARRACKS):
            await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first)
        if self.structures(UnitTypeId.FACTORY).amount < 2 and self.minerals >= 150 and self.vespene >= 100 and not self.already_pending(UnitTypeId.FACTORY):
            await self.build(UnitTypeId.FACTORY, near=self.townhalls.first)
        for rax in self.structures(UnitTypeId.BARRACKS).ready.idle:
            if self.minerals >= 50 and self.supply_left: rax.train(UnitTypeId.MARINE)
        for factory in self.structures(UnitTypeId.FACTORY).ready.idle:
            if self.minerals >= 100 and self.vespene >= 0 and self.supply_left: factory.train(UnitTypeId.HELLION)
        army = self.units(UnitTypeId.MARINE) | self.units(UnitTypeId.HELLION)
        if army.amount >= 28:
            for unit in army: unit.attack(self.enemy_start_locations[0])
        if iteration % 100 == 0:
            print(f'loop={iteration} scv={self.workers.amount} marine={self.units(UnitTypeId.MARINE).amount} hellion={self.units(UnitTypeId.HELLION).amount}', flush=True)

    async def on_end(self, result):
        Path(__file__).resolve().parent.joinpath('artifacts', 'hard_curriculum_teacher_result.json').write_text(
            json.dumps({'result': str(result)}))
        print(f'Hard curriculum teacher: {result}', flush=True)


if __name__ == '__main__':
    result = run_game(maps.get('Simple64'), [Bot(Race.Terran, HardCurriculumTeacher()), Computer(Race.Zerg, Difficulty.Hard)], realtime=False, game_time_limit=1800)
    Path(__file__).resolve().parent.joinpath('artifacts', 'hard_curriculum_teacher_result.json').write_text(
        json.dumps({'result': str(result)}))
    print(result)
