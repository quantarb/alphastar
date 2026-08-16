#!/usr/bin/env python3
"""Run semantic-transfer checkpoints with only valid 4.9.2 command decodes."""
import argparse, os
os.environ.setdefault('SC2PATH', '/Applications/StarCraft II')
import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from macro_decoder_config import RACE_CONFIG, RACE_IDS
from multirace_general_policy import MultiRaceGeneralMacroPolicy
from semantic_action_schema import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES
from semantic_transfer_policy import SemanticTransferPolicy
from semantic_action_contract import spec_hash, supports


class SemanticBot(BotAI):
 def __init__(self, checkpoint, race_name):
  super().__init__(); self.race_name=race_name;self.rid=RACE_IDS[race_name];self.c=RACE_CONFIG[race_name]
  base=MultiRaceGeneralMacroPolicy();self.model=SemanticTransferPolicy(base.shared)
  data=torch.load(checkpoint,map_location='cpu',weights_only=False)
  if data.get('action_contract_hash') != spec_hash(): raise RuntimeError('Checkpoint action contract does not match this live decoder')
  self.model.load_state_dict(data['state_dict']);self.model.eval()
 async def on_start(self):self.game_data.abilities.setdefault(4135,self.game_data.abilities[1]);self.prevent_double_actions=lambda a:True
 def amount(self,t):return self.structures(t).amount if t else 0
 def feat(self):
  c=self.c; a=lambda t:self.units.of_type({t}).amount
  return torch.tensor([[min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(self.supply_cap/200,1),min(max(self.supply_left,0)/30,1),min(self.workers.amount/80,1),min(self.amount(c['supply'])/20,1),min(self.amount(c['prod'])/20,1),min(self.amount(c['gas'])/20,1),min(self.amount(c['tech'])/20,1),min(a(c['basic'])/20,1),min(a(c['ranged'])/20,1),min(a(c['advanced'])/20,1),0,0,0]],dtype=torch.float32)
 async def near(self,t):
  if self.townhalls: await self.build(t,near=self.townhalls.first,placement_step=3)
 async def on_step(self,it):
  if it%16 or not self.townhalls:return
  c=self.c
  with torch.no_grad():o=self.model(self.feat(),torch.tensor([self.rid]))
  # Score only semantic combinations with an executable 4.9.2 realization.
  fi={x:i for i,x in enumerate(FAMILIES)};pi={x:i for i,x in enumerate(PAYLOAD_ROLES)};ai={x:i for i,x in enumerate(ACTOR_ROLES)}
  choices=[]
  def add(actor,family,payload,code,ok):
   target='unit' if code=='gas' else ('point' if code in ('supply','prod','tech','attack') else 'none')
   if ok and supports(actor,family,payload,target):choices.append((float(o['actor'][0,ai[actor]]+o['family'][0,fi[family]]+o['payload'][0,pi[payload]]),code))
  add('production','train_morph','worker','worker',bool(self.townhalls.idle and self.can_afford(c['worker']) and self.workers.amount<70))
  add('worker','build','supply','supply',self.can_afford(c['supply']) and self.supply_left<=5 and not self.already_pending(c['supply']))
  add('worker','build','production','prod',self.can_afford(c['prod']) and self.amount(c['prod'])<4)
  add('worker','build','gas','gas',self.can_afford(c['gas']) and self.amount(c['gas'])<2)
  add('worker','build','tech','tech',self.can_afford(c['tech']) and not self.amount(c['tech']))
  add('production','train_morph','basic_army','basic',bool(self.structures(c['prod']).ready.idle and self.can_afford(c['basic']) and self.supply_left>0))
  add('production','train_morph','ranged_army','ranged',bool(self.structures(c['ranged_prod']).ready.idle and self.can_afford(c['ranged']) and self.supply_left>0))
  army=self.units.of_type({c['basic'],c['ranged'],c['advanced']})
  add('combat','attack','spell','attack',army.amount>=8)
  if not choices:return
  code=max(choices)[1]
  if code=='worker':self.townhalls.idle.first.train(c['worker'])
  elif code=='supply':await self.near(c['supply'])
  elif code=='prod':await self.near(c['prod'])
  elif code=='gas':
   for g in self.vespene_geyser.closer_than(12,self.townhalls.first):self.workers.closest_to(g).build(c['gas'],g);break
  elif code=='tech':await self.near(c['tech'])
  elif code in ('basic','ranged'):
   unit=c['basic'] if code=='basic' else c['ranged'];buildings=self.structures(c['prod'] if code=='basic' else c['ranged_prod']).ready.idle
   if buildings:buildings.first.train(unit)
  elif code=='attack':
   for u in army:u.attack(self.enemy_start_locations[0])
  print(f't={self.time:.0f} semantic={code}',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--race',default='terran',choices=tuple(RACE_IDS));p.add_argument('--difficulty',default='hard');p.add_argument('--realtime',action='store_true');p.add_argument('--replay');a=p.parse_args()
 print(run_game(maps.get('Simple64'),[Bot(getattr(Race,a.race.title()),SemanticBot(a.checkpoint,a.race)),Computer(Race.Zerg,getattr(Difficulty,a.difficulty.title()))],realtime=a.realtime,save_replay_as=a.replay,game_time_limit=1800))
if __name__=='__main__':main()
