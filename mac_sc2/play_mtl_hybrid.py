#!/usr/bin/env python3
"""Macro-first live runner for the finished MTL checkpoint."""
import argparse,json,os
from collections import deque
from pathlib import Path
os.environ.setdefault('SC2PATH','/Applications/StarCraft II')
import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty,Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot,Computer
from alphastar_sized_compact_policy import AlphaStarSizedCompactPolicy
ROOT=Path(__file__).resolve().parents[1]
class Hybrid(BotAI):
 def __init__(self,p):
  super().__init__();d=torch.load(p,map_location='cpu',weights_only=False);self.m=AlphaStarSizedCompactPolicy();self.m.load_state_dict(d['state_dict']);self.m.eval();self.h=deque([1025]*8,maxlen=8)
 async def on_step(self,it):
  if it%32:return
  own=(list(self.units)+list(self.structures))[:32];n=len(own)
  ent=torch.zeros(1,64,24);mask=torch.ones(1,64,dtype=torch.bool)
  for i,u in enumerate(own):ent[0,i,0]=u.type_id.value/65535;ent[0,i,1]=i/31;mask[0,i]=False
  mask[0,0]=False;hist=torch.zeros(1,16,24);hist[:,:,1:9]=torch.tensor(list(self.h),dtype=torch.float32).view(1,1,8)/1025
  with torch.no_grad():o=self.m(ent,mask,hist,torch.tensor([1]));a=int(o['macro'][0].argmax());self.h.append(int(o['ability'][0].argmax()))
  probes=self.workers.amount; pylons=self.structures(UnitTypeId.PYLON).amount; gates=self.structures(UnitTypeId.GATEWAY).amount;army=self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}).amount
  # Macro intent is learned; these guards merely prevent illegal/no-op orders.
  # Supply must take priority over worker production: the previous ordering
  # attempted Probes at the cap forever, never reaching the Pylon branch.
  if not self.townhalls:
   return
  if self.supply_left<=4 and not self.already_pending(UnitTypeId.PYLON):a=1
  elif gates<2 and pylons>=1 and not self.already_pending(UnitTypeId.GATEWAY):a=2
  elif probes<20:a=0
  # A `wait` prediction must not turn ready production buildings into no-ops.
  # This is action masking (not a scripted build order).
  elif a==6 and self.structures(UnitTypeId.GATEWAY).ready.idle and self.supply_left>0:a=3
  elif army>=12:a=5
  if a==0 and self.townhalls.idle:self.townhalls.idle.first.train(UnitTypeId.PROBE)
  elif a==1 and self.can_afford(UnitTypeId.PYLON):await self.build(UnitTypeId.PYLON,near=self.townhalls.first,placement_step=3)
  elif a==2 and self.can_afford(UnitTypeId.GATEWAY):await self.build(UnitTypeId.GATEWAY,near=self.townhalls.first,placement_step=3)
  elif a==3:
   for g in self.structures(UnitTypeId.GATEWAY).ready.idle:
    if self.minerals>=100 and self.supply_left>0:g.train(UnitTypeId.ZEALOT)
  elif a==4 and self.minerals>=400:await self.expand_now()
  elif a==5:
   for u in self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}):u.attack(self.enemy_start_locations[0])
  print(f't={self.time:.0f} macro={a} probes={probes} gates={gates} army={army}',flush=True)
 async def on_end(self,result):
  (ROOT/'mac_sc2/artifacts/mtl_hybrid_hard_result.json').write_text(json.dumps({'result':str(result)}));print(result,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='mac_sc2/artifacts/mtl_ondemand_1000_model.pt');p.add_argument('--difficulty',default='hard');p.add_argument('--realtime',action='store_true');a=p.parse_args();print(run_game(maps.get('Simple64'),[Bot(Race.Protoss,Hybrid(a.checkpoint)),Computer(Race.Zerg,getattr(Difficulty,a.difficulty.title()))],realtime=a.realtime,game_time_limit=None if a.realtime else 1800))
