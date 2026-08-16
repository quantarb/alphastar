#!/usr/bin/env python3
"""Evaluate the stateful replay macro policy in a live Protoss game."""
import argparse, json, os
from pathlib import Path
os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')
import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from stateful_macro_policy import StatefulMacroPolicy

ROOT=Path(__file__).resolve().parents[1]
class StatefulProtoss(BotAI):
 def __init__(self, checkpoint):
  super().__init__(); data=torch.load(checkpoint,map_location='cpu',weights_only=False); self.model=StatefulMacroPolicy(); self.model.load_state_dict(data['state_dict']); self.model.eval()
 def state(self):
  probes=self.workers.amount; pylons=self.structures(UnitTypeId.PYLON).amount; gates=self.structures(UnitTypeId.GATEWAY).amount; army=self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}).amount; nexuses=self.townhalls.amount
  return torch.tensor([[min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(self.supply_cap/200,1),min(max(self.supply_left,0)/30,1),min(probes/80,1),min(pylons/12,1),min(gates/12,1),min(army/80,1),min(nexuses/8,1),0,0,0]],dtype=torch.float32)
 async def on_step(self, iteration):
  if iteration%16 or not self.townhalls:return
  with torch.no_grad(): logits=self.model(self.state(),torch.tensor([1]))[0]
  valid=torch.full_like(logits,-1e9)
  # Mask impossible actions; scores still select among legal learned intents.
  if self.townhalls.idle and self.can_afford(UnitTypeId.PROBE) and self.workers.amount<70: valid[0]=logits[0]
  if self.can_afford(UnitTypeId.PYLON) and self.supply_left<=5 and not self.already_pending(UnitTypeId.PYLON): valid[1]=logits[1]
  if self.can_afford(UnitTypeId.GATEWAY) and self.structures(UnitTypeId.PYLON).ready and self.structures(UnitTypeId.GATEWAY).amount<6 and not self.already_pending(UnitTypeId.GATEWAY): valid[2]=logits[2]
  if self.structures(UnitTypeId.GATEWAY).ready.idle and self.can_afford(UnitTypeId.ZEALOT) and self.supply_left>0: valid[3]=logits[3]
  if self.can_afford(UnitTypeId.NEXUS) and self.townhalls.amount<3: valid[4]=logits[4]
  if self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}).amount>=8: valid[5]=logits[5]
  valid[6]=logits[6]
  action=int(valid.argmax())
  if action==0:self.townhalls.idle.first.train(UnitTypeId.PROBE)
  elif action==1:await self.build(UnitTypeId.PYLON,near=self.townhalls.first,placement_step=3)
  elif action==2:await self.build(UnitTypeId.GATEWAY,near=self.townhalls.first,placement_step=3)
  elif action==3:
   for gate in self.structures(UnitTypeId.GATEWAY).ready.idle:
    if self.can_afford(UnitTypeId.ZEALOT) and self.supply_left>0:gate.train(UnitTypeId.ZEALOT)
  elif action==4:await self.expand_now()
  elif action==5:
   for unit in self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}):unit.attack(self.enemy_start_locations[0])
  print(f't={self.time:.0f} action={action} workers={self.workers.amount} gates={self.structures(UnitTypeId.GATEWAY).amount} army={self.units.of_type({UnitTypeId.ZEALOT,UnitTypeId.STALKER}).amount}',flush=True)
 async def on_end(self,result):
  path=ROOT/'mac_sc2/artifacts/stateful_macro_hard_result.json';path.write_text(json.dumps({'result':str(result)}));print(result,flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',required=True);ap.add_argument('--difficulty',default='hard');ap.add_argument('--realtime',action='store_true');args=ap.parse_args()
 print(run_game(maps.get('Simple64'),[Bot(Race.Protoss,StatefulProtoss(args.checkpoint)),Computer(Race.Zerg,getattr(Difficulty,args.difficulty.title()))],realtime=args.realtime,game_time_limit=None if args.realtime else 1800))
