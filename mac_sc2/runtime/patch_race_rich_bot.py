#!/usr/bin/env python3
"""Live 4.9.2 decoder: every selected tuple is checked before it is issued."""
import json, os
from pathlib import Path
import torch
os.environ.setdefault('SC2PATH','/Applications/StarCraft II')
from sc2.bot_ai import BotAI
from sc2.ids.ability_id import AbilityId
from mac_sc2.architectures.multitask_policy import PlayableMultiTaskPolicy
from mac_sc2.architectures.patch_race_rich_mtl import PatchRaceRichMTLPolicy
from mac_sc2.contracts.multitask import validate_checkpoint as validate_multitask
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.patch_race_mtl import all_spec_hashes, build_specs, task_key
from mac_sc2.runtime.macro_decoder_config import RACE_CONFIG
from mac_sc2.runtime.entity_snapshot import encode
from mac_sc2.runtime.placement_candidates import candidates

class PatchRaceBot(BotAI):
 def __init__(self,checkpoint,registry,race,smoke_steps=None):
  super().__init__();self.race_name=race;self.smoke_steps=smoke_steps;self.task=task_key('4.9.2',race.title());self.specs=build_specs(registry);data=torch.load(checkpoint,map_location='cpu',weights_only=False)
  self.multitask='multitask_contract_hash' in data
  if self.multitask:
   validate_multitask(data,registry);self.model=PlayableMultiTaskPolicy(self.specs)
  else:
   if data.get('task_action_spec_hashes')!=all_spec_hashes(registry) or data.get('snapshot_contract_hash')!=snapshot_hash():raise RuntimeError('checkpoint ActionSpec or snapshot contract mismatch')
   self.model=PatchRaceRichMTLPolicy(self.specs)
  self.model.load_state_dict(data['state_dict']);self.model.eval();self.c=RACE_CONFIG[race]
 def feat(self):
  c=self.c;amount=lambda x:self.structures(x).amount if x else 0;unit=lambda x:self.units.of_type({x}).amount
  return torch.tensor([[min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(self.supply_cap/200,1),min(max(self.supply_left,0)/30,1),min(self.workers.amount/80,1),min(amount(c['supply'])/20,1),min(amount(c['prod'])/20,1),min(amount(c['gas'])/20,1),min(amount(c['tech'])/20,1),min(unit(c['basic'])/20,1),min(unit(c['ranged'])/20,1),min(unit(c['advanced'])/20,1),0,0,0]],dtype=torch.float32)
 def actors(self,role):
  if role=='worker':return self.workers
  if role=='production':return self.structures.ready
  if role=='combat':return self.units.exclude_type({self.c['worker']})
  return self.units|self.structures
 async def issue(self,row):
  actors=self.actors(row['actor']);
  if not actors:return False
  ability=AbilityId(row['ability']); entities,padding,owned=encode(self)
  # The shared pointer ranks only concrete, role-legal live entities.
  with torch.no_grad(): actor_scores,target_scores=(self.model.macro.pointers(self.feat(),entities[None],padding[None]) if self.multitask else self.model.pointers(self.feat(),entities[None],padding[None]))
  if row['family']=='repair' and self.multitask:
   with torch.no_grad(): actor_scores,target_scores=self.model.repair(entities[None],padding[None])
  by_tag={u.tag:u for u in actors}; ranked=[owned[i] for i in actor_scores[0].argsort(descending=True).tolist() if i<len(owned) and owned[i].tag in by_tag]
  available=await self.get_available_abilities(ranked); actor=next((u for u,a in zip(ranked,available) if ability in a),None)
  if actor is None:return False
  target=None
  if row['target_mode']=='point':
   # Building/landing actions may receive locations only from SC2 placement queries.
   valid=await candidates(self,ability.value,self.townhalls.first.position) if self.townhalls else []
   if not valid:return False
   home=self.townhalls.first.position;coords=torch.tensor([((p.x-home.x)/64,(p.y-home.y)/64) for p in valid])
   with torch.no_grad(): scores=(self.model.placement(entities[None],padding[None],coords[None])[0] if self.multitask else self.model.placement_scores(self.feat(),coords[None])[0])
   target=valid[int(scores.argmax())]
  elif row['target_mode']=='unit':
   targets=self.enemy_units if row['actor']=='combat' else (self.units|self.structures)
   if not targets:return False
   # Pointers are used for own repair/micro targets; enemy targets are visible-only.
   if row['family']=='repair' and self.multitask:
    own={u.tag:u for u in owned}; target=next((own[owned[i].tag] for i in target_scores[0].argsort(descending=True).tolist() if i<len(owned) and owned[i].tag in own),None)
   elif row['actor']!='combat':
    own={u.tag:u for u in owned}; target=next((own[owned[i].tag] for i in target_scores[0].argsort(descending=True).tolist() if i<len(owned) and owned[i].tag in own),None)
   else: target=targets.closest_to(actor)
   if target is None:return False
  try: actor(ability,target=target,queue=row['queue']);return True
  except Exception:return False
 async def on_step(self,it):
  if it%16 or not self.townhalls:return
  with torch.no_grad():logits=self.model.task_logits(self.feat(),self.task)[0]
  for index in logits.argsort(descending=True).tolist():
   if await self.issue(self.specs[self.task][index]):print(f't={self.time:.0f} tuple={index}',flush=True);break
  if self.smoke_steps is not None and it>=self.smoke_steps: await self.client.leave()
 async def on_end(self,result):
  if getattr(self,'result_path',None):Path(self.result_path).write_text(json.dumps({'result':str(result)}))
  print(result,flush=True)
