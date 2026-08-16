#!/usr/bin/env python3
import argparse,asyncio,json,os,torch,sys
from pathlib import Path
os.environ.setdefault('SC2PATH','/Applications/StarCraft II')
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Race,Difficulty
from sc2.main import run_game
from sc2.player import Bot,Computer
from sc2.ids.ability_id import AbilityId
from mac_sc2.architectures.macro_placement import MacroIntentPolicy,PlacementRanker
from mac_sc2.contracts.placement_spec import actions,validate_checkpoint
from mac_sc2.runtime.placement_candidates import candidates
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.runtime.entity_snapshot import encode
from mac_sc2.contracts.repair import validate_checkpoint as validate_repair
from mac_sc2.architectures.repair import RepairPolicy
from mac_sc2.runtime.repair_runner import issue_learned_repair
class B(BotAI):
 def __init__(self,ck,reg,loaded_marker=None,repair_checkpoint=None):
  super().__init__();d=torch.load(ck,map_location='cpu',weights_only=False);validate_checkpoint(d,reg)
  if d.get('entity_snapshot_hash') != snapshot_hash(): raise RuntimeError('entity snapshot contract mismatch')
  self.spec=actions(reg);self.names=d['abilities'];self.macro=MacroIntentPolicy(len(self.names));self.ranker=PlacementRanker();self.macro.load_state_dict(d['macro_state_dict']);self.ranker.load_state_dict(d['placement_state_dict']);self.macro.eval();self.ranker.eval();self.out=None
  self.repair=None
  rd=d if 'repair_state_dict' in d else (torch.load(repair_checkpoint,map_location='cpu',weights_only=False) if repair_checkpoint else None)
  if rd:
   validate_repair(rd)
   if rd.get('entity_snapshot_hash')!=snapshot_hash():raise RuntimeError('repair entity snapshot contract mismatch')
   self.repair=RepairPolicy();self.repair.load_state_dict(rd['repair_state_dict'] if 'repair_state_dict' in rd else rd['state_dict']);self.repair.eval()
  # The trainer may overwrite its one checkpoint only after every component
  # of this exact snapshot has been deserialized successfully.
  if loaded_marker: Path(loaded_marker).write_text(json.dumps({'checkpoint':str(Path(ck).resolve()),'games':d['games'],'decisions':d['decisions'],'placement_spec_hash':d['placement_spec_hash']}))
 async def on_step(self,i):
  if i%32 or not self.townhalls:return
  home=self.townhalls.first.position;x,mask,u=encode(self)
  if self.repair and await issue_learned_repair(self,self.repair,x,mask,u): return
  with torch.no_grad(): al=self.macro(x[None],mask[None])
  for k in al[0].argsort(descending=True).tolist():
   name=self.names[k];opts=[s for s in self.spec if s.race=='Terran' and s.ability==name]
   for s in opts:
    actors=self.workers if s.actor_role=='worker' else self.structures
    avail=await self.get_available_abilities(actors)
    usable=[q for q,ab in zip(actors,avail) if any(a.value==s.ability_id for a in ab)]
    if not usable:continue
    cs=await candidates(self,s.ability_id,home)
    if not cs:continue
    c=torch.tensor([((p.x-home.x)/20,(p.y-home.y)/20) for p in cs]);score=self.ranker(x[None],mask[None],c[None]);target=cs[int(score[0].argmax())];usable[0](AbilityId(s.ability_id),target);print('placement',name,target);return
 async def on_end(self,r):Path(self.out).write_text(json.dumps({'result':str(r)}))
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--registry',required=True);p.add_argument('--replay',required=True);p.add_argument('--loaded-marker');p.add_argument('--repair-checkpoint');p.add_argument('--difficulty',choices=('easy','medium','hard'),default='easy');a=p.parse_args();b=B(a.checkpoint,a.registry,a.loaded_marker,a.repair_checkpoint);b.out=str(Path(a.replay).with_suffix('.json'));print(run_game(maps.get('Simple64'),[Bot(Race.Terran,b),Computer(Race.Zerg,getattr(Difficulty,a.difficulty.title()))],realtime=False,save_replay_as=a.replay))
if __name__=='__main__':main()
