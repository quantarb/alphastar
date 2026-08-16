#!/usr/bin/env python3
"""Run any race from the shared-backbone, race-head MTL macro checkpoint."""
import argparse, json, os
from pathlib import Path
os.environ.setdefault('SC2PATH','/Applications/StarCraft II')
import torch
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer
from multirace_general_policy import MultiRaceGeneralMacroPolicy
from macro_decoder_config import RACE_IDS, RACE_CONFIG
from factorized_micro_policy import FactorizedMicroPolicy

ROOT=Path(__file__).resolve().parents[1]

class MultiRaceBot(BotAI):
 def __init__(self,checkpoint,race_name,attack_min,tactical_checkpoint=None):
  super().__init__(); data=torch.load(checkpoint,map_location='cpu',weights_only=False); self.model=MultiRaceGeneralMacroPolicy();self.model.load_state_dict(data['state_dict']);self.model.eval();self.race_name=race_name;self.rid=RACE_IDS[race_name];self.c=RACE_CONFIG[race_name];self.advanced_option=False;self.attack_min=attack_min;self.tactical=None
  if tactical_checkpoint:
   tactical_data=torch.load(tactical_checkpoint,map_location='cpu',weights_only=False);self.tactical=FactorizedMicroPolicy(self.model.shared,freeze_backbone=False);self.tactical.load_state_dict(tactical_data['state_dict']);self.tactical.eval()
 async def on_start(self):
  self.game_data.abilities.setdefault(4135,self.game_data.abilities[1]);self.prevent_double_actions=lambda action:True
 def amount(self,t):return (self.structures(t).amount if t else 0)
 def army(self,types):return self.units.of_type(types).amount
 def features(self):
  c=self.c;workers=self.workers.amount;supply=self.amount(c['supply']);prod=self.amount(c['prod']);gas=self.amount(c['gas']);tech=self.amount(c['tech']);basic=self.army({c['basic']});ranged=self.army({c['ranged']});advanced=self.army({c['advanced']})
  return torch.tensor([[min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(self.supply_cap/200,1),min(max(self.supply_left,0)/30,1),min(workers/80,1),min(supply/20,1),min(prod/20,1),min(gas/20,1),min(tech/20,1),min(basic/20,1),min(ranged/20,1),min(advanced/20,1),0,0,0]],dtype=torch.float32)
 def tactical_features(self,army,enemies):
  """Live combat state for the factorized heads; all values are bounded."""
  nearby=enemies.closer_than(18,army.center)
  closest=min((army.center.distance_to(e) for e in nearby),default=40)
  avg_distance=sum(army.center.distance_to(e) for e in nearby)/max(nearby.amount,1)
  low=sum(u.health_percentage<.55 for u in army)
  reloading=sum(u.weapon_cooldown>0 for u in army)
  friendly_health=sum(u.health_percentage for u in army)/max(army.amount,1)
  enemy_health=sum(u.health_percentage for u in nearby)/max(nearby.amount,1)
  basic=self.units.of_type({self.c['basic']}).amount;ranged=self.units.of_type({self.c['ranged']}).amount;advanced=self.units.of_type({self.c['advanced']}).amount
  home_distance=army.center.distance_to(self.townhalls.first) if self.townhalls else 40
  return torch.tensor([[self.rid/2,min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(army.amount/60,1),min(nearby.amount/60,1),low/max(army.amount,1),reloading/max(army.amount,1),min(avg_distance/40,1),min(closest/40,1),friendly_health,enemy_health,min(basic/40,1),min(ranged/40,1),min(advanced/40,1),min(home_distance/80,1),min(self.supply_left/30,1),min(self.workers.amount/80,1),min(self.townhalls.amount/4,1)]],dtype=torch.float32)
 async def tactical_step(self,it):
  if not self.tactical:return
  army=self.units.of_type({self.c['basic'],self.c['ranged'],self.c['advanced']}).ready
  enemies=self.enemy_units
  if army.amount<2 or not enemies:return
  nearby=enemies.closer_than(24,army.center)
  if not nearby:return
  with torch.no_grad():out=self.tactical(self.tactical_features(army,enemies),torch.tensor([self.rid]))
  group=int(out['group'][0].argmax());intent=int(out['intent'][0].argmax());target_mode=int(out['target'][0].argmax());direction=int(out['direction'][0].argmax())
  squads=(army,army.filter(lambda u:u.health_percentage<.55),army.closer_than(18,nearby.center));squad=squads[group] if squads[group] else army
  if target_mode==0:target=nearby.closest_to(squad.center)
  elif target_mode==1:target=min(nearby,key=lambda u:(u.health+u.shield,u.distance_to(squad.center)))
  else:target=max(nearby,key=lambda u:(u.health_max+u.shield_max,-u.distance_to(squad.center)))
  # Every factor has a live, patch-valid interpretation. The tactical policy
  # only commands real own units and visible enemy targets / real positions.
  if intent==3 or direction==3:
   for unit in squad:unit.hold_position()
  elif intent==2 or direction==2:
   if self.townhalls:
    for unit in squad:unit.move(self.townhalls.closest_to(unit).position)
  elif intent==1 or direction==1:
   for unit in squad:unit.move(unit.position.towards(target.position,-4))
  else:
   for unit in squad:unit.attack(target)
  if it%50==0:print(f't={self.time:.0f} micro group={group} intent={intent} target={target_mode} direction={direction} units={squad.amount}',flush=True)
 async def build_near_home(self,t):
  if self.townhalls:await self.build(t,near=self.townhalls.first,placement_step=3)
 async def on_step(self,it):
  await self.tactical_step(it)
  if it%16 or not self.townhalls:return
  c=self.c
  if it%64==0:
   for gas in self.gas_buildings.ready:
    for worker in self.workers.sorted_by_distance_to(gas)[:2]:worker.gather(gas)
  with torch.no_grad(): logits=self.model(self.features(),torch.tensor([self.rid]))[0]
  v=torch.full_like(logits,-1e9);prod_ready=self.larva if self.race_name=='zerg' else self.structures(c['prod']).ready.idle;ranged_ready=self.larva if self.race_name=='zerg' else self.structures(c['ranged_prod']).ready.idle;tech_ready=self.structures(c['tech']).ready
  advanced_ready=self.structures(c['advanced_build']).ready.idle if c['advanced_build'] else self.townhalls.ready.idle
  next_expansion=await self.get_next_expansion() if self.townhalls.amount<3 else None
  if (self.larva if self.race_name=='zerg' else self.townhalls.idle) and self.can_afford(c['worker']) and self.workers.amount<70:v[0]=logits[0]
  if self.can_afford(c['supply']) and self.supply_left<=5 and not self.already_pending(c['supply']):v[1]=logits[1]
  if self.can_afford(c['prod']) and self.amount(c['prod'])<4 and not self.already_pending(c['prod']):v[2]=logits[2]
  if self.can_afford(c['gas']) and self.amount(c['gas'])<2:v[3]=logits[3]
  if self.can_afford(c['tech']) and self.structures(c['prod']).ready and not self.amount(c['tech']):v[4]=logits[4]
  if prod_ready and self.structures(c['prod']).ready and self.can_afford(c['basic']) and self.supply_left>0:v[5]=logits[5]
  if (tech_ready if self.race_name!='zerg' else self.structures(c['tech']).ready) and ranged_ready and self.can_afford(c['ranged']) and self.supply_left>0:v[6]=logits[6]
  if c['advanced_build']:
   if (self.can_afford(c['advanced_build']) and tech_ready and not self.amount(c['advanced_build'])) or (advanced_ready and self.can_afford(c['advanced']) and self.supply_left>0):v[7]=logits[7]
  elif self.townhalls.ready.idle and self.can_afford(c['advanced']) and self.supply_left>0:v[7]=logits[7]
  # A policy cannot learn that a map has no remaining buildable expansion from
  # our compact state alone. Treat an unavailable location as an illegal action
  # instead of repeatedly turning "expand" into a no-op.
  if self.can_afford(c['townhall']) and next_expansion is not None:v[8]=logits[8]
  army_types={c['basic'],c['ranged'],c['advanced']}
  # Preserve the model's attack preference, but reject obvious sacrificial
  # attacks. This is an inference-time legality/safety guard, exposed as a
  # CLI setting rather than hidden in a race-specific build order.
  if self.army(army_types)>=self.attack_min:v[9]=logits[9]
  if v.max().item()<=-1e8:return
  action=int(v.argmax())
  if action==7:self.advanced_option=True
  if self.advanced_option and c['advanced_build'] and not self.amount(c['advanced_build']):action=7
  if self.advanced_option and advanced_ready:
   if self.can_afford(c['advanced']) and self.supply_left>0:action=7
   else:return
  if action==0:
   if self.race_name=='zerg' and self.larva:self.larva.first.train(c['worker'])
   elif self.townhalls.idle:self.townhalls.idle.first.train(c['worker'])
  elif action==1:
   if self.race_name=='zerg' and self.larva:self.larva.first.train(c['supply'])
   else:await self.build_near_home(c['supply'])
  elif action==2:await self.build_near_home(c['prod'])
  elif action==3:
   for geyser in self.vespene_geyser.closer_than(12,self.townhalls.first):
    if self.can_afford(c['gas']):self.workers.closest_to(geyser).build(c['gas'],geyser);break
  elif action==4:await self.build_near_home(c['tech'])
  elif action==5:
   if self.race_name=='zerg':
    if self.larva and self.can_afford(c['basic']) and self.supply_left>0:self.larva.first.train(c['basic'])
   else:
    for building in prod_ready:
     if self.can_afford(c['basic']) and self.supply_left>0:building.train(c['basic'])
  elif action==6:
   if self.race_name=='zerg':
    if self.larva and self.can_afford(c['ranged']) and self.supply_left>0:self.larva.first.train(c['ranged'])
   else:
    for building in ranged_ready:
     if self.can_afford(c['ranged']) and self.supply_left>0:building.train(c['ranged'])
  elif action==7:
   if c['advanced_build']:
    if advanced_ready and self.can_afford(c['advanced']) and self.supply_left>0:advanced_ready.first.train(c['advanced']);self.advanced_option=False
    elif not self.amount(c['advanced_build']):await self.build_near_home(c['advanced_build'])
   elif self.townhalls.ready.idle:self.townhalls.ready.idle.first.train(c['advanced']);self.advanced_option=False
  elif action==8:await self.expand_now(location=next_expansion)
  elif action==9:
   for unit in self.units.of_type(army_types):unit.attack(self.enemy_start_locations[0])
  print(f't={self.time:.0f} race={self.race_name} action={action} basic={self.army({c["basic"]})} ranged={self.army({c["ranged"]})} advanced={self.army({c["advanced"]})}',flush=True)
 async def on_end(self,result):
  out=ROOT/f'mac_sc2/artifacts/mtl_{self.race_name}_result.json';out.write_text(json.dumps({'result':str(result)}));print(result,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='mac_sc2/artifacts/multirace_general_1000.pt');p.add_argument('--tactical-checkpoint',help='Transfer-trained factorized tactical checkpoint');p.add_argument('--race',choices=tuple(RACE_IDS),default='protoss');p.add_argument('--difficulty',default='easy');p.add_argument('--attack-min',type=int,default=16);p.add_argument('--realtime',action='store_true');p.add_argument('--replay');a=p.parse_args();race=getattr(Race,a.race.title());print(run_game(maps.get('Simple64'),[Bot(race,MultiRaceBot(a.checkpoint,a.race,a.attack_min,a.tactical_checkpoint)),Computer(Race.Zerg,getattr(Difficulty,a.difficulty.title()))],realtime=a.realtime,save_replay_as=a.replay,game_time_limit=None if a.realtime else 1800))
