#!/usr/bin/env python3
"""Evaluate the Transformer student against Easy on SC2 5.0.16."""
import os
import json
import argparse
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
from current_patch_transformer import ACTIONS, CurrentPatchTransformer
from micro_policy import MicroTransformer

ROOT = Path(__file__).resolve().parents[1]
WATCH_MODE = False

class Student(BotAI):
    def __init__(self):
        super().__init__(); p=torch.load(ROOT/'mac_sc2/artifacts/current_patch_transformer.pt', map_location='cpu', weights_only=False)
        self.model=CurrentPatchTransformer(); self.model.load_state_dict(p['state_dict']); self.model.eval(); self.history=[]
        micro=torch.load(ROOT/'mac_sc2/artifacts/micro_transformer.pt',map_location='cpu',weights_only=False)
        self.micro=MicroTransformer(); self.micro.load_state_dict(micro['state_dict']); self.micro.eval()
    def features(self):
        return [min(63,int(self.time//15)), min(63,int(self.minerals//25)), min(31,max(0,int(self.supply_left))), min(31,self.workers.amount), min(15,self.structures(UnitTypeId.SUPPLYDEPOT).amount), min(15,self.structures(UnitTypeId.BARRACKS).amount), min(63,self.units(UnitTypeId.MARINE).amount)]
    async def on_start(self): self.client.game_step = 1
    async def micro_step(self):
        enemies = self.enemy_units
        if not enemies: return
        for marine in self.units(UnitTypeId.MARINE):
            nearby = enemies.closer_than(18, marine)
            if not nearby: continue
            target = min(nearby, key=lambda u: (u.health + u.shield, marine.distance_to(u)))
            banelings = nearby.of_type({UnitTypeId.BANELING})
            bane_dist = min((marine.distance_to(b) for b in banelings), default=31)
            state=torch.tensor([[
                min(31,int(marine.health_percentage*31)), min(4,int(marine.weapon_cooldown)),
                min(31,int(marine.distance_to(target))), min(31,int(bane_dist)),
                min(15,self.units(UnitTypeId.MARINE).closer_than(10,marine).amount),
                min(19,nearby.amount), min(31,int((target.health+target.shield)/5)),
            ]],dtype=torch.long)
            with torch.no_grad(): mode=int(self.micro(state).argmax())
            if mode == 0: marine.attack(target)
            elif mode == 1: marine.move(marine.position.towards(target.position, -4))
            elif self.townhalls: marine.move(self.townhalls.first.position)
    async def on_step(self, it):
        await self.micro_step()
        # Tactical control has no artificial rate limit; macro decisions are
        # deliberately slower because buildings and production queues do not
        # change meaningfully every game frame.
        if it % 32:
            return
        self.history.append(self.features()); self.history=self.history[-8:]
        states=torch.tensor([[ [0]*7 ]*(8-len(self.history))+self.history],dtype=torch.long)
        with torch.no_grad(): score=self.model(states)[0]
        valid=torch.full_like(score,-1e9); workers=self.workers.amount; marines=self.units(UnitTypeId.MARINE).amount
        desired=1 if workers<12 else (2 if marines<8 else 3)
        if self.townhalls.idle and self.minerals>=50 and workers<20: valid[0]=score[0]
        if self.townhalls and self.supply_left<=4 and self.minerals>=100 and self.already_pending(UnitTypeId.SUPPLYDEPOT)==0: valid[1]=score[1]
        if self.townhalls and self.workers and self.minerals>=150 and self.structures(UnitTypeId.BARRACKS).amount<desired and self.already_pending(UnitTypeId.BARRACKS)==0: valid[2]=score[2]
        if self.structures(UnitTypeId.BARRACKS).ready.idle and self.minerals>=50 and self.supply_left>0: valid[3]=score[3]
        if marines>=12: valid[4]=score[4]
        # Waiting is a fallback, never a competing macro choice. This prevents
        # the common idle label from overwhelming rare build/train decisions.
        if torch.all(valid[:5] < -1e8): valid[5]=score[5]
        # The Transformer remains the action policy.  A replay-trained action
        # model can nevertheless predict `wait` at a state that differs by a
        # few frames from its demonstrations.  Decode that token safely: do
        # not let an unavailable/no-op action starve the economy or leave an
        # idle production building unused.  This is an action-validity guard,
        # not a replacement build order.
        a = int(valid.argmax())
        critical = None
        if marines >= 12:
            critical = 4
        elif self.townhalls and self.supply_left <= 4 and self.minerals >= 100 and self.already_pending(UnitTypeId.SUPPLYDEPOT) == 0:
            critical = 1
        elif self.townhalls and self.workers and self.minerals >= 150 and self.structures(UnitTypeId.BARRACKS).amount < desired and self.already_pending(UnitTypeId.BARRACKS) == 0:
            critical = 2
        elif self.structures(UnitTypeId.BARRACKS).ready.idle and self.minerals >= 50 and self.supply_left > 0:
            critical = 3
        elif self.townhalls.idle and self.minerals >= 50 and workers < 20:
            critical = 0
        if a == 5 or valid[a] < -1e8:
            a = critical if critical is not None else 5
        if a==0: self.townhalls.idle.first.train(UnitTypeId.SCV)
        elif a==1: await self.build(UnitTypeId.SUPPLYDEPOT, near=self.townhalls.first, placement_step=3)
        elif a==2: await self.build(UnitTypeId.BARRACKS, near=self.townhalls.first, placement_step=3)
        elif a==3:
            for r in self.structures(UnitTypeId.BARRACKS).ready.idle: r.train(UnitTypeId.MARINE)
        elif a==4:
            for r in self.structures(UnitTypeId.BARRACKS).ready.idle:
                if self.minerals>=50 and self.supply_left>0: r.train(UnitTypeId.MARINE)
            for m in self.units(UnitTypeId.MARINE): m.attack(self.enemy_start_locations[0])
        if it%50==0: print(f'step={it} action={ACTIONS[a]} marines={marines}')
    async def on_end(self,result):
        (ROOT / 'mac_sc2/artifacts/current_patch_student_last_result.json').write_text(json.dumps({'result': str(result)}))
        print(f'Student result: {result}')

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--realtime', action='store_true', help='Run at normal speed in a visible SC2 window.')
    parser.add_argument('--difficulty', choices=('easy', 'medium', 'hard'), default='easy')
    args = parser.parse_args()
    WATCH_MODE = args.realtime
    logger.remove(); logger.add(lambda m: print(m,end=''),level='INFO')
    difficulty = getattr(Difficulty, args.difficulty.title())
    print('Student final:',run_game(maps.get('Simple64'),[Bot(Race.Terran,Student()),Computer(Race.Zerg,difficulty)],realtime=WATCH_MODE,game_time_limit=None if WATCH_MODE else 1800))
