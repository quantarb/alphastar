#!/usr/bin/env python3
"""Fine-tune the runnable unified policy directly from raw 4.9.2 replays."""
import argparse, json, subprocess, sys, time
from pathlib import Path
import torch
from torch.nn import functional as F
from mac_sc2.architectures.macro_placement import PlacementRanker
from mac_sc2.architectures.repair import RepairPolicy
from mac_sc2.architectures.semantic_macro import SemanticMacroPolicy
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.contracts.placement import CANDIDATE_OFFSETS
from mac_sc2.contracts.placement_spec import spec_hash as placement_hash
from mac_sc2.contracts.repair import action_hash as repair_hash
from mac_sc2.contracts.semantic import ACTOR_ROLES, FAMILIES, PAYLOAD_ROLES, RACE_IDS, TARGET_KINDS, action_hash as macro_hash
from mac_sc2.contracts.unified import policy_hash
from mac_sc2.data.placement_replay import examples as placement_examples
from mac_sc2.data.repair_replay import examples as repair_examples
from mac_sc2.data.semantic_replay import examples as macro_examples

def pack(snapshot):
    entities=torch.zeros(ENTITY_SLOTS,8); size=min(len(snapshot),ENTITY_SLOTS)
    if size: entities[:size]=torch.tensor(snapshot[:size],dtype=torch.float32)
    mask=torch.ones(ENTITY_SLOTS,dtype=torch.bool);mask[:size]=False;return entities,mask

def load_components(macro_path, component_path, registry, device):
    macro_data=torch.load(macro_path,map_location='cpu',weights_only=False)
    if macro_data.get('action_contract_hash') != macro_hash(): raise ValueError('macro initializer has incompatible semantic ActionSpec')
    component_data=torch.load(component_path,map_location='cpu',weights_only=False)
    if component_data.get('placement_spec_hash') != placement_hash(registry) or component_data.get('repair_action_spec_hash') != repair_hash() or component_data.get('entity_snapshot_hash') != snapshot_hash(): raise ValueError('placement/repair initializer contracts are incompatible')
    macro=SemanticMacroPolicy();macro.load_state_dict(macro_data['state_dict'])
    placement=PlacementRanker();placement.load_state_dict(component_data['placement_state_dict'])
    repair=RepairPolicy();repair.load_state_dict(component_data['repair_state_dict'])
    return macro.to(device),placement.to(device),repair.to(device)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--registry',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--macro-resume',default='mac_sc2/artifacts/semantic_contract_all_replays.pt');parser.add_argument('--component-resume',default='mac_sc2/artifacts/combined_policy_4_9_2.pt')
    parser.add_argument('--checkpoint-every',type=int,default=200);parser.add_argument('--max-games',type=int);args=parser.parse_args()
    if not 0<args.checkpoint_every<=200: raise ValueError('checkpoint-every must be 1..200')
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');macro,placement,repair=load_components(args.macro_resume,args.component_resume,args.registry,device)
    optimizer=torch.optim.AdamW([*macro.parameters(),*placement.parameters(),*repair.parameters()],lr=2e-4,weight_decay=.01)
    files=[item for item in json.loads(Path(args.manifest).read_text())['valid'] if item['version'].startswith('4.9.2')]; files=files[:args.max_games] if args.max_games else files
    indices={'actor':{x:i for i,x in enumerate(ACTOR_ROLES)},'family':{x:i for i,x in enumerate(FAMILIES)},'payload':{x:i for i,x in enumerate(PAYLOAD_ROLES)},'target':{x:i for i,x in enumerate(TARGET_KINDS)}}
    counts={'macro':0,'placement':0,'repair':0,'discarded':0}; launched=False
    def state(model): return {key:value.detach().cpu().clone() for key,value in model.state_dict().items()}
    def save(game):
        torch.save({'macro_state_dict':state(macro),'placement_state_dict':state(placement),'repair_state_dict':state(repair),'games':game,'counts':counts.copy(),'resumed_from':{'macro':str(Path(args.macro_resume).resolve()),'placement_repair':str(Path(args.component_resume).resolve())},'macro_action_spec_hash':macro_hash(),'placement_spec_hash':placement_hash(args.registry),'repair_action_spec_hash':repair_hash(),'entity_snapshot_hash':snapshot_hash(),'unified_action_spec_hash':policy_hash(args.registry),'registry':str(Path(args.registry).resolve()),'architecture':'semantic macro MTL + SC2-legal placement ranker + repair pointers'},args.output)
    for game,item in enumerate(files,1):
        try: macro_rows=list(macro_examples(item['path']))
        except Exception as exc: print(f'macro_skip game={game} {type(exc).__name__}',flush=True);macro_rows=[]
        for race,features,label in macro_rows:
            output=macro(torch.tensor([features],dtype=torch.float32,device=device),torch.tensor([race],device=device))
            values={'actor':label.actor_role,'family':label.family,'payload':label.payload_role,'target':label.target_kind}
            loss=sum(F.cross_entropy(output[key],torch.tensor([indices[key][values[key]]],device=device)) for key in values)+F.cross_entropy(output['queued'],torch.tensor([int(label.queued)],device=device))
            optimizer.zero_grad();loss.backward();optimizer.step();counts['macro']+=1
        try: placement_rows=list(placement_examples(item['path']))
        except Exception as exc: print(f'placement_skip game={game} {type(exc).__name__}',flush=True);placement_rows=[]
        for snapshot,label,home in placement_rows:
            entities,mask=pack(snapshot); positive=torch.tensor([(label.point[0]-home[0])/64,(label.point[1]-home[1])/64])
            # Replay coordinates supervise location preference; SC2 supplies
            # legality-filtered candidates only at live execution.
            negative=torch.tensor(CANDIDATE_OFFSETS,dtype=torch.float32)/16+positive; candidate=torch.cat((positive[None],negative),0)
            scores=placement(entities[None].to(device),mask[None].to(device),candidate[None].to(device));loss=F.cross_entropy(scores,torch.tensor([0],device=device))
            optimizer.zero_grad();loss.backward();optimizer.step();counts['placement']+=1
        try: repair_rows=list(repair_examples(item['path']))
        except Exception as exc: print(f'repair_skip game={game} {type(exc).__name__}',flush=True);repair_rows=[]
        for snapshot,actor,target in repair_rows:
            entities,mask=pack(snapshot);actor_logits,target_logits=repair(entities[None].to(device),mask[None].to(device));loss=F.cross_entropy(actor_logits,torch.tensor([actor],device=device))+F.cross_entropy(target_logits,torch.tensor([target],device=device))
            optimizer.zero_grad();loss.backward();optimizer.step();counts['repair']+=1
        if game%25==0: print(f'games={game} {counts}',flush=True)
        if game%args.checkpoint_every==0:
            save(game)
            if not launched:
                marker=Path(f'{args.output}.game_{game}.loaded');marker.unlink(missing_ok=True)
                subprocess.Popen([sys.executable,'-m','mac_sc2.scripts.play_unified','--checkpoint',args.output,'--registry',args.registry,'--replay',str(Path(args.output).with_suffix('.first_eval.SC2Replay')),'--loaded-marker',str(marker)])
                deadline=time.monotonic()+30
                while not marker.exists() and time.monotonic()<deadline: time.sleep(.1)
                if not marker.exists(): raise RuntimeError('evaluator did not load exact checkpoint')
                print(f'evaluation_loaded game={game}; training_continues',flush=True);launched=True
    save(len(files));print(f'saved={args.output} counts={counts}',flush=True)
if __name__=='__main__':main()
