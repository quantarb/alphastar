#!/usr/bin/env python3
"""Live evaluator for the replay-trained general macro policy (Protoss)."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('SC2PATH','/Applications/StarCraft II')
import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty,Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot,Computer
from general_macro_policy import GeneralMacroPolicy
ROOT=Path(__file__).resolve().parents[1]
class GeneralProtoss(BotAI):
 def __init__(self,p):
  super().__init__();d=torch.load(p,map_location='cpu',weights_only=False);self.m=GeneralMacroPolicy();self.m.load_state_dict(d['state_dict']);self.m.eval();self.advanced_option=False
 async def on_start(self):
  # SC2 4.9.2 exposes a live order ability absent from python-sc2's static
  # table.  Treat it as the generic Smart command so all helper paths can
  # inspect the live worker order safely.
  self.game_data.abilities.setdefault(4135,self.game_data.abilities[1])
  self.prevent_double_actions=lambda action:True
 def features(self):
  n=lambda u:self.structures(u).amount
  army=lambda us:self.units.of_type(us).amount
  c=[self.workers.amount,n(UnitTypeId.PYLON),n(UnitTypeId.GATEWAY),n(UnitTypeId.ASSIMILATOR),n(UnitTypeId.CYBERNETICSCORE)+n(UnitTypeId.ROBOTICSFACILITY)+n(UnitTypeId.STARGATE),army({UnitTypeId.ZEALOT}),army({UnitTypeId.STALKER,UnitTypeId.ADEPT,UnitTypeId.SENTRY}),army({UnitTypeId.IMMORTAL,UnitTypeId.VOIDRAY,UnitTypeId.PHOENIX})]
  return torch.tensor([[min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(self.supply_cap/200,1),min(max(self.supply_left,0)/30,1),min(c[0]/80,1),* [min(q/20,1) for q in c[1:]],0,0,0]],dtype=torch.float32)
 async def on_step(self,it):
  if it%16 or not self.townhalls:return
  # python-sc2 does not automatically saturate newly built Assimilators.
  # Allocate two workers to each finished gas building so learned gas/tech
  # intents can actually pay for ranged and advanced units.
  if it%64==0:
   for gas in self.gas_buildings.ready:
    for worker in self.workers.sorted_by_distance_to(gas)[:2]:worker.gather(gas)
  with torch.no_grad():logits=self.m(self.features(),torch.tensor([1]))[0]
  v=torch.full_like(logits,-1e9);gates=self.structures(UnitTypeId.GATEWAY).ready.idle;core=self.structures(UnitTypeId.CYBERNETICSCORE).ready;army=lambda us:self.units.of_type(us).amount
  if self.townhalls.idle and self.can_afford(UnitTypeId.PROBE) and self.workers.amount<70:v[0]=logits[0]
  if self.can_afford(UnitTypeId.PYLON) and self.supply_left<=5 and not self.already_pending(UnitTypeId.PYLON):v[1]=logits[1]
  if self.can_afford(UnitTypeId.GATEWAY) and self.structures(UnitTypeId.PYLON).ready and self.structures(UnitTypeId.GATEWAY).amount<5 and not self.already_pending(UnitTypeId.GATEWAY):v[2]=logits[2]
  if self.can_afford(UnitTypeId.ASSIMILATOR) and self.structures(UnitTypeId.PYLON).ready and self.structures(UnitTypeId.ASSIMILATOR).amount<2:v[3]=logits[3]
  if self.can_afford(UnitTypeId.CYBERNETICSCORE) and self.structures(UnitTypeId.GATEWAY).ready and not core and not self.already_pending(UnitTypeId.CYBERNETICSCORE):v[4]=logits[4]
  if gates and self.can_afford(UnitTypeId.ZEALOT) and self.supply_left>0:v[5]=logits[5]
  if gates and core and self.can_afford(UnitTypeId.STALKER) and self.supply_left>0:v[6]=logits[6]
  robo=self.structures(UnitTypeId.ROBOTICSFACILITY).ready
  if (self.can_afford(UnitTypeId.ROBOTICSFACILITY) and core and not self.structures(UnitTypeId.ROBOTICSFACILITY).amount) or (robo.idle and self.can_afford(UnitTypeId.IMMORTAL) and self.supply_left>0):v[7]=logits[7]
  if self.can_afford(UnitTypeId.NEXUS) and self.townhalls.amount<3:v[8]=logits[8]
  if army({UnitTypeId.ZEALOT,UnitTypeId.STALKER,UnitTypeId.IMMORTAL})>=10:v[9]=logits[9]
  if v.max().item() <= -1e8:return
  a=int(v.argmax())
  # Advanced-unit production is a multi-step macro option: the learned intent
  # must survive the short construction delay for its prerequisite facility.
  if a==7:self.advanced_option=True
  if self.advanced_option and self.structures(UnitTypeId.ROBOTICSFACILITY).amount<1:a=7
  if self.advanced_option and robo.idle:
   if self.can_afford(UnitTypeId.IMMORTAL) and self.supply_left>0:a=7
   else:return  # Reserve minerals and gas until this learned macro completes.
  if a==0 and self.townhalls.idle:self.townhalls.idle.first.train(UnitTypeId.PROBE)
  elif a==1:await self.build(UnitTypeId.PYLON,near=self.townhalls.first,placement_step=3)
  elif a==2:await self.build(UnitTypeId.GATEWAY,near=self.townhalls.first,placement_step=3)
  elif a==3:
   for geyser in self.vespene_geyser.closer_than(12,self.townhalls.first):
    if self.can_afford(UnitTypeId.ASSIMILATOR):
     self.workers.closest_to(geyser).build(UnitTypeId.ASSIMILATOR,geyser);break
  elif a==4:await self.build(UnitTypeId.CYBERNETICSCORE,near=self.townhalls.first,placement_step=3)
  elif a==5:
   for x in gates:
    if self.can_afford(UnitTypeId.ZEALOT):x.train(UnitTypeId.ZEALOT)
  elif a==6:
   for x in gates:
    if self.can_afford(UnitTypeId.STALKER):x.train(UnitTypeId.STALKER)
  elif a==7:
   if robo.idle and self.can_afford(UnitTypeId.IMMORTAL) and self.supply_left>0:
    robo.idle.first.train(UnitTypeId.IMMORTAL);self.advanced_option=False
   elif not self.structures(UnitTypeId.ROBOTICSFACILITY).amount:await self.build(UnitTypeId.ROBOTICSFACILITY,near=self.townhalls.first,placement_step=3)
  elif a==8:await self.expand_now()
  elif a==9:
   for x in self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER,UnitTypeId.IMMORTAL}):x.attack(self.enemy_start_locations[0])
  print(f't={self.time:.0f} action={a} minerals={self.minerals} gas={self.vespene} probes={self.workers.amount} zealots={army({UnitTypeId.ZEALOT})} stalkers={army({UnitTypeId.STALKER})} immortals={army({UnitTypeId.IMMORTAL})} robo={self.structures(UnitTypeId.ROBOTICSFACILITY).amount}',flush=True)
 async def on_end(self,r):
  (ROOT/'mac_sc2/artifacts/general_macro_hard_result.json').write_text(json.dumps({'result':str(r)}));print(r,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--difficulty',default='hard');p.add_argument('--realtime',action='store_true');p.add_argument('--replay',default='mac_sc2/artifacts/general_macro_latest.SC2Replay');a=p.parse_args();print(run_game(maps.get('Simple64'),[Bot(Race.Protoss,GeneralProtoss(a.checkpoint)),Computer(Race.Zerg,getattr(Difficulty,a.difficulty.title()))],realtime=a.realtime,save_replay_as=a.replay,game_time_limit=None if a.realtime else 1800))
