#!/usr/bin/env python3
"""Run the unified semantic macro + legal placement + repair checkpoint."""
import argparse, json, os
from pathlib import Path
import torch
os.environ.setdefault("SC2PATH", "/Applications/StarCraft II")
from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from mac_sc2.architectures.macro_placement import PlacementRanker
from mac_sc2.architectures.repair import RepairPolicy
from mac_sc2.architectures.semantic_macro import SemanticMacroPolicy
from mac_sc2.contracts.semantic import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, RACE_IDS, supports
from mac_sc2.contracts.unified import validate_checkpoint
from mac_sc2.runtime.macro_decoder_config import RACE_CONFIG
from mac_sc2.runtime.entity_snapshot import encode
from mac_sc2.runtime.placement_candidates import candidates
from mac_sc2.runtime.repair_runner import issue_learned_repair

class UnifiedBot(BotAI):
    def __init__(self, checkpoint, registry, race_name, loaded_marker=None):
        super().__init__()
        self.data=torch.load(checkpoint,map_location="cpu",weights_only=False); validate_checkpoint(self.data,registry)
        self.race_name=race_name; self.race_id=RACE_IDS[race_name]; self.config=RACE_CONFIG[race_name]
        self.macro=SemanticMacroPolicy(); self.macro.load_state_dict(self.data["macro_state_dict"]); self.macro.eval()
        self.placement=PlacementRanker(); self.placement.load_state_dict(self.data["placement_state_dict"]); self.placement.eval()
        self.repair=RepairPolicy(); self.repair.load_state_dict(self.data["repair_state_dict"]); self.repair.eval()
        self.result_path=None
        if loaded_marker: Path(loaded_marker).write_text(json.dumps({"checkpoint":str(Path(checkpoint).resolve()),"games":self.data["games"],"unified_action_spec_hash":self.data["unified_action_spec_hash"]}))

    def amount(self, kind): return self.structures(kind).amount if kind else 0
    def macro_features(self):
        c=self.config; units=lambda kind:self.units.of_type({kind}).amount
        return torch.tensor([[min(self.time/900,1),min(self.minerals/1500,1),min(self.vespene/1000,1),min(self.supply_used/200,1),min(self.supply_cap/200,1),min(max(self.supply_left,0)/30,1),min(self.workers.amount/80,1),min(self.amount(c['supply'])/20,1),min(self.amount(c['prod'])/20,1),min(self.amount(c['gas'])/20,1),min(self.amount(c['tech'])/20,1),min(units(c['basic'])/20,1),min(units(c['ranged'])/20,1),min(units(c['advanced'])/20,1),0,0,0]],dtype=torch.float32)

    async def learned_build(self, building):
        """SC2 provides candidates; the model chooses only among those tiles."""
        if not self.townhalls or not self.workers: return False
        ability=self.game_data.units[building.value].creation_ability
        if ability is None: return False
        home=self.townhalls.first.position; valid=await candidates(self,ability.id.value,home)
        if not valid: return False
        entities,mask,_=encode(self); coords=torch.tensor([((p.x-home.x)/64,(p.y-home.y)/64) for p in valid])
        with torch.no_grad(): scores=self.placement(entities[None],mask[None],coords[None])[0]
        target=valid[int(scores.argmax())]; worker=self.workers.closest_to(target)
        available=(await self.get_available_abilities([worker]))[0]
        if ability.id not in available: return False
        worker.build(building,target); print(f"placement {building.name} {target}",flush=True); return True

    async def macro_step(self):
        c=self.config
        with torch.no_grad(): output=self.macro(self.macro_features(),torch.tensor([self.race_id]))
        actor={value:index for index,value in enumerate(ACTOR_ROLES)}; family={value:index for index,value in enumerate(FAMILIES)}; payload={value:index for index,value in enumerate(PAYLOAD_ROLES)}
        choices=[]
        def add(role,kind,content,code,legal):
            target='unit' if code=='gas' else ('point' if code in ('supply','production','tech','attack') else 'none')
            if legal and supports(role,kind,content,target): choices.append((float(output['actor'][0,actor[role]]+output['family'][0,family[kind]]+output['payload'][0,payload[content]]),code))
        add('production','train_morph','worker','worker',bool(self.townhalls.idle and self.can_afford(c['worker']) and self.workers.amount<70))
        add('worker','build','supply','supply',self.can_afford(c['supply']) and self.supply_left<=5 and not self.already_pending(c['supply']))
        add('worker','build','production','production',self.can_afford(c['prod']) and self.amount(c['prod'])<4)
        add('worker','build','gas','gas',self.can_afford(c['gas']) and self.amount(c['gas'])<2)
        add('worker','build','tech','tech',self.can_afford(c['tech']) and not self.amount(c['tech']))
        add('production','train_morph','basic_army','basic',bool(self.structures(c['prod']).ready.idle and self.can_afford(c['basic']) and self.supply_left>0))
        add('production','train_morph','ranged_army','ranged',bool(self.structures(c['ranged_prod']).ready.idle and self.can_afford(c['ranged']) and self.supply_left>0))
        army=self.units.of_type({c['basic'],c['ranged'],c['advanced']}); add('combat','attack','spell','attack',army.amount>=8)
        if not choices: return
        code=max(choices,key=lambda item:item[0])[1]
        if code=='worker': self.townhalls.idle.first.train(c['worker'])
        elif code=='supply': await self.learned_build(c['supply'])
        elif code=='production': await self.learned_build(c['prod'])
        elif code=='tech': await self.learned_build(c['tech'])
        elif code=='gas':
            for geyser in self.vespene_geyser.closer_than(12,self.townhalls.first):
                worker=self.workers.closest_to(geyser)
                ability=self.game_data.units[c['gas'].value].creation_ability
                available=(await self.get_available_abilities([worker]))[0]
                if ability and ability.id in available: worker.build(c['gas'],geyser); break
        elif code in ('basic','ranged'):
            unit=c['basic'] if code=='basic' else c['ranged']; buildings=self.structures(c['prod'] if code=='basic' else c['ranged_prod']).ready.idle
            if buildings: buildings.first.train(unit)
        elif code=='attack':
            for unit in army: unit.attack(self.enemy_start_locations[0])
        print(f"t={self.time:.0f} macro={code}",flush=True)

    async def on_step(self, iteration):
        if not self.townhalls: return
        # Macro has a reserved cadence; repair cannot permanently starve it.
        if iteration % 16 == 0: await self.macro_step()
        elif iteration % 16 == 8:
            entities,mask,units=encode(self); await issue_learned_repair(self,self.repair,entities,mask,units)
    async def on_end(self,result):
        if self.result_path: Path(self.result_path).write_text(json.dumps({"result":str(result)}))
        print(result,flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--registry',required=True);parser.add_argument('--race',choices=tuple(RACE_IDS),default='terran');parser.add_argument('--difficulty',choices=('easy','medium','hard'),default='easy');parser.add_argument('--replay',required=True);parser.add_argument('--loaded-marker');args=parser.parse_args()
    bot=UnifiedBot(args.checkpoint,args.registry,args.race,args.loaded_marker);bot.result_path=str(Path(args.replay).with_suffix('.json'))
    print(run_game(maps.get('Simple64'),[Bot(getattr(Race,args.race.title()),bot),Computer(Race.Zerg,getattr(Difficulty,args.difficulty.title()))],realtime=False,save_replay_as=args.replay))
if __name__=='__main__': main()
