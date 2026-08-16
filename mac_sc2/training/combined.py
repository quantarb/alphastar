#!/usr/bin/env python3
import argparse,json,subprocess,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from mac_sc2.contracts.placement import ENTITY_SLOTS,CANDIDATE_OFFSETS
from mac_sc2.architectures.macro_placement import MacroIntentPolicy,PlacementRanker
from mac_sc2.data.placement_replay import examples
from mac_sc2.contracts.placement_spec import actions,spec_hash
from mac_sc2.contracts.entity_snapshot import snapshot_hash
from mac_sc2.contracts.repair import action_hash as repair_action_hash
from mac_sc2.architectures.repair import RepairPolicy
from mac_sc2.data.repair_replay import examples as repair_examples
def pack(snapshot):
 x=torch.zeros(ENTITY_SLOTS,8); x[:min(len(snapshot),ENTITY_SLOTS)]=torch.tensor(snapshot[:ENTITY_SLOTS]); m=torch.ones(ENTITY_SLOTS,dtype=torch.bool);m[:min(len(snapshot),ENTITY_SLOTS)]=False;return x,m
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--registry',required=True);p.add_argument('--output',required=True);p.add_argument('--checkpoint-every',type=int,default=200);p.add_argument('--max-games',type=int);a=p.parse_args()
 if not 0<a.checkpoint_every<=200:raise ValueError('checkpoint-every must be 1..200')
 spec=actions(a.registry); names=sorted({x.ability for x in spec});idx={x:i for i,x in enumerate(names)};dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');macro=MacroIntentPolicy(len(names)).to(dev);ranker=PlacementRanker().to(dev);repair=RepairPolicy().to(dev);opt=torch.optim.AdamW(list(macro.parameters())+list(ranker.parameters())+list(repair.parameters()),lr=3e-4)
 files=[x for x in json.load(open(a.manifest))['valid'] if x['version'].startswith('4.9.2')];files=files[:a.max_games] if a.max_games else files;seen=0;launched=False
 def state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 repair_seen=0
 def save(game):torch.save({'macro_state_dict':state(macro),'placement_state_dict':state(ranker),'repair_state_dict':state(repair),'games':game,'decisions':seen,'repair_decisions':repair_seen,'abilities':names,'placement_spec_hash':spec_hash(a.registry),'repair_action_spec_hash':repair_action_hash(),'entity_snapshot_hash':snapshot_hash(),'registry':str(Path(a.registry).resolve()),'architecture':'macro-intent + SC2-legal placement ranker + repair actor/target pointers'},a.output)
 for game,item in enumerate(files,1):
  try: rows=list(examples(item['path']))
  except Exception as e:print('skip',game,type(e).__name__);continue
  allowed={(x.race,x.ability) for x in spec}
  rows=[r for r in rows if (r[1].race,r[1].ability) in allowed]
  for snap,label,home in rows:
   ent,mask=pack(snap);cand=torch.tensor(CANDIDATE_OFFSETS,dtype=torch.float32)/20; target=torch.tensor(((label.point[0]-home[0])/20,(label.point[1]-home[1])/20));ci=((cand-target).square().sum(1)).argmin()
   ability=macro(ent[None].to(dev),mask[None].to(dev));score=ranker(ent[None].to(dev),mask[None].to(dev),cand[None].to(dev));loss=F.cross_entropy(ability,torch.tensor([idx[label.ability]],device=dev))+F.cross_entropy(score,ci[None].to(dev));opt.zero_grad();loss.backward();opt.step();seen+=1
  try: repair_rows=list(repair_examples(item['path']))
  except Exception as e: print('repair_skip',game,type(e).__name__);repair_rows=[]
  for snap,actor,target in repair_rows:
   ent,mask=pack(snap);actor_logits,target_logits=repair(ent[None].to(dev),mask[None].to(dev));loss=F.cross_entropy(actor_logits,torch.tensor([actor],device=dev))+F.cross_entropy(target_logits,torch.tensor([target],device=dev));opt.zero_grad();loss.backward();opt.step();repair_seen+=1
  if game%25==0:print(f'games={game} placement_decisions={seen} repair_decisions={repair_seen}',flush=True)
  if game%a.checkpoint_every==0:
   save(game)
   if not launched:
    marker=Path(str(a.output)+'.game_%d.loaded'%game);marker.unlink(missing_ok=True)
    subprocess.Popen([sys.executable,'-m','mac_sc2.scripts.play_combined','--checkpoint',a.output,'--registry',a.registry,'--replay',str(Path(a.output).with_suffix('.first_eval.SC2Replay')),'--loaded-marker',str(marker)])
    # Do not overwrite the sole checkpoint until the evaluator confirms that
    # it has deserialized this exact game-N snapshot.  The match then runs in
    # its own process while this raw-replay loop proceeds.
    deadline=time.monotonic()+30
    while not marker.exists() and time.monotonic()<deadline: time.sleep(.1)
    if not marker.exists(): raise RuntimeError('game-%d evaluator did not load checkpoint within 30 seconds'%game)
    print('evaluation_loaded game=%d; training_continues'%game,flush=True);launched=True
 save(len(files))
if __name__=='__main__':main()
