#!/usr/bin/env python3
"""Fine-tune one executable patch/race MTL checkpoint from raw replays."""
import argparse, json, subprocess, sys, time
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from mac_sc2.architectures.patch_race_rich_mtl import PatchRaceRichMTLPolicy
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.contracts.patch_race_mtl import all_spec_hashes, build_specs
from mac_sc2.data.patch_race_exact import examples

def pack(rows):
 x=torch.zeros(ENTITY_SLOTS,8); n=min(len(rows),ENTITY_SLOTS)
 if n:x[:n]=torch.tensor(rows[:n]); mask=torch.ones(ENTITY_SLOTS,dtype=torch.bool);mask[:n]=False
 else:mask=torch.ones(ENTITY_SLOTS,dtype=torch.bool)
 return x,mask

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--registry',required=True);p.add_argument('--output',required=True);p.add_argument('--resume',default='mac_sc2/artifacts/patch_race_recent_streaming_base.pt');p.add_argument('--max-games',type=int,default=200);p.add_argument('--checkpoint-every',type=int,default=200);p.add_argument('--lr',type=float,default=2e-4);p.add_argument('--eval-replay-dir',default='mac_sc2/artifacts');a=p.parse_args()
 if a.checkpoint_every!=200: raise ValueError('one overwrite-only checkpoint must be saved at game 200')
 specs=build_specs(a.registry); hashes=all_spec_hashes(a.registry); init=torch.load(a.resume,map_location='cpu',weights_only=False)
 model=PatchRaceRichMTLPolicy(specs);model.load_streaming_backbone(init['state_dict'])
 device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model.to(device);opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01)
 files=[item for item in json.loads(Path(a.manifest).read_text())['valid'] if '.'.join(item['version'].split('.')[:3])+'/Terran' in specs][:a.max_games]; counts=Counter(); discarded=Counter()
 if len(files)<200: raise ValueError(f'need 200 compatible raw replays, found {len(files)}')
 def save(game):
  state={k:v.detach().cpu() for k,v in model.state_dict().items()};Path(a.output).parent.mkdir(parents=True,exist_ok=True)
  torch.save({'state_dict':state,'games':game,'resumed_from':str(Path(a.resume).resolve()),'registry':str(Path(a.registry).resolve()),'task_action_spec_hashes':hashes,'snapshot_contract_hash':snapshot_hash(),'architecture':'shared_backbone + patch/race tuple heads + placement + actor/target pointers','counts':dict(counts),'discarded':dict(discarded)},a.output)
 for game,item in enumerate(files,1):
  try: stream=examples(item['path'],item['version'],specs,discarded)
  except Exception as exc: discarded[type(exc).__name__]+=1;continue
  try:
   for row in stream:
    state=torch.tensor([row['state']],dtype=torch.float32,device=device); logit=model.task_logits(state,row['task']);loss=F.cross_entropy(logit,torch.tensor([row['tuple_id']],device=device))
    entities,mask=pack(row['snapshot']); entities=entities[None].to(device);mask=mask[None].to(device)
    if row['actor']>=0:
     actor,_=model.pointers(state,entities,mask);loss=loss+F.cross_entropy(actor,torch.tensor([row['actor']],device=device));counts['actor_pointer']+=1
    if row['target']>=0:
     _,target=model.pointers(state,entities,mask);loss=loss+F.cross_entropy(target,torch.tensor([row['target']],device=device));counts['target_pointer']+=1
    if row['location'] is not None:
     point=torch.tensor(row['location'],device=device); candidates=torch.cat((point[None],point[None]+torch.tensor([[.1,0],[-.1,0],[0,.1],[0,-.1]],device=device)),0)[None];scores=model.placement_scores(state,candidates);loss=loss+F.cross_entropy(scores,torch.tensor([0],device=device));counts['placement']+=1
    opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();counts['tuples']+=1
  except Exception as exc: discarded[type(exc).__name__]+=1
  if game%25==0: print(f'games={game} counts={dict(counts)} discarded={dict(discarded)}',flush=True)
  if game==200:
   save(game)
   # Exact saved bytes are independently loaded by all three Easy runners while this process continues.
   for race in ('terran','protoss','zerg'):
    replay=str(Path(a.eval_replay_dir)/f'patch_race_rich_200_{race}_easy.SC2Replay')
    subprocess.Popen([sys.executable,'-m','mac_sc2.scripts.play_patch_race_rich_mtl','--checkpoint',a.output,'--registry',a.registry,'--race',race,'--difficulty','easy','--replay',replay])
 print(f'saved={a.output} games={len(files)} counts={dict(counts)}',flush=True)
if __name__=='__main__':main()
