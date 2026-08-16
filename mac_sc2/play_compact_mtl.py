#!/usr/bin/env python3
"""Run the compact MTL checkpoint as Protoss against the built-in SC2 AI."""
import argparse, json, os
from collections import deque
from pathlib import Path
os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')
import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from hierarchical_mtl_policy import HierarchicalMTLPolicy

ROOT=Path(__file__).resolve().parents[1]

class ProtossMTL(BotAI):
    def __init__(self, checkpoint):
        super().__init__(); data=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=HierarchicalMTLPolicy(); self.model.load_state_dict(data['state_dict']); self.model.eval(); self.history=deque(maxlen=16); self.macro_history=deque([6]*8,maxlen=8)
    def tensors(self):
        own=list(self.units)+list(self.structures); own=own[:32]; ent=torch.zeros(1,64,24); mask=torch.ones(1,64,dtype=torch.bool)
        for i,u in enumerate(own):
            ent[0,i,0]=min(u.type_id.value,65535)/65535; ent[0,i,1]=i/31; ent[0,i,2]=min(self.time*22.4,65535)/65535; ent[0,i,3]=u.health_percentage
            mask[0,i]=False
        mask[0,0]=False
        state=torch.zeros(24); state[0]=min(self.time*22.4,65535)/65535
        for i,a in enumerate(self.macro_history): state[1+i]=a/6
        self.history.append(state); hist=list(self.history)
        hist=[torch.zeros(24)]*(16-len(hist))+hist
        return ent,mask,torch.stack(hist).unsqueeze(0)
    async def on_step(self, iteration):
        if iteration%32:return
        ent,mask,hist=self.tensors()
        with torch.no_grad(): action=int(self.model(ent,mask,hist,torch.tensor([1]))['macro'].argmax())
        self.macro_history.append(action)
        probes=self.workers.amount; gateways=self.structures(UnitTypeId.GATEWAY).amount; zealots=self.units(UnitTypeId.ZEALOT).amount+self.units(UnitTypeId.STALKER).amount
        # Validity guards make a predicted macro intent executable; they do
        # not provide a build order or replace the shared learned policy.
        if self.townhalls.idle and self.minerals>=50 and probes<24: action=0
        elif self.supply_left<=4 and self.minerals>=100 and not self.already_pending(UnitTypeId.PYLON): action=1
        elif gateways<2 and self.minerals>=150 and not self.already_pending(UnitTypeId.GATEWAY): action=2
        elif self.structures(UnitTypeId.GATEWAY).ready.idle and self.minerals>=100 and self.supply_left>0: action=3
        elif zealots>=12: action=5
        if action==0 and self.townhalls.idle: self.townhalls.idle.first.train(UnitTypeId.PROBE)
        elif action==1 and self.townhalls: await self.build(UnitTypeId.PYLON,near=self.townhalls.first,placement_step=3)
        elif action==2 and self.townhalls: await self.build(UnitTypeId.GATEWAY,near=self.townhalls.first,placement_step=3)
        elif action==3:
            for g in self.structures(UnitTypeId.GATEWAY).ready.idle:
                if self.minerals>=100 and self.supply_left>0:g.train(UnitTypeId.ZEALOT)
        elif action==4 and self.minerals>=400: await self.expand_now()
        elif action==5:
            for u in self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}):u.attack(self.enemy_start_locations[0])
        print(f't={self.time:.0f} macro={action} probes={probes} army={zealots}',flush=True)
    async def on_end(self,result):
        (ROOT/'mac_sc2/artifacts/compact_mtl_protoss_result.json').write_text(json.dumps({'result':str(result)})); print(f'result={result}',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--difficulty',choices=('easy','medium','hard'),default='easy');p.add_argument('--realtime',action='store_true');a=p.parse_args()
    print(run_game(maps.get('Simple64'),[Bot(Race.Protoss,ProtossMTL(a.checkpoint)),Computer(Race.Zerg,getattr(Difficulty,a.difficulty.title()))],realtime=a.realtime,game_time_limit=None if a.realtime else 1800))
