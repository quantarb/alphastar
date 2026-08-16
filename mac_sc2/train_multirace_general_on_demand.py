#!/usr/bin/env python3
"""True MTL macro training: shared replay-state trunk, per-race action heads."""
import argparse,json
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from multirace_general_policy import MultiRaceGeneralMacroPolicy,RACES
from train_general_macro_on_demand import rows

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);ap.add_argument('--resume');ap.add_argument('--start-game',type=int,default=0);ap.add_argument('--max-games',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=512);ap.add_argument('--window',type=int,default=8);ap.add_argument('--lr',type=float,default=8e-4);ap.add_argument('--patch',help='Restrict to one replay patch family, e.g. 4.9.2');ap.add_argument('--checkpoint-every',type=int,default=0);ap.add_argument('--winner-only',action='store_true');a=ap.parse_args()
 files=json.loads(Path(a.manifest).read_text())['valid']
 if a.patch:files=[item for item in files if '.'.join(item['version'].split('.')[:3])==a.patch]
 files=files[a.start_game:a.start_game+a.max_games];dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=MultiRaceGeneralMacroPolicy().to(dev)
 if a.resume:model.load_state_dict(torch.load(a.resume,map_location='cpu',weights_only=False)['state_dict'])
 opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.01);labels=[Counter() for _ in RACES];seen=[0]*3;correct=[0]*3
 def save_checkpoint(games_completed):
  state={key:value.detach().cpu() for key,value in model.state_dict().items()}
  Path(a.output).parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':state,'games':games_completed,'start_game':a.start_game,'patch':a.patch,'resumed_from':a.resume,'winner_only':a.winner_only,'decisions_by_race':dict(zip(RACES,seen)),'labels_by_race':dict(zip(RACES,map(dict,labels))),'architecture':'shared state backbone + Terran/Protoss/Zerg macro heads'},a.output)
 for gi,item in enumerate(files,1):
  try:data=rows(item['path'],a.window,winner_only=a.winner_only)
  except Exception as e:print(f'skip game={gi} {type(e).__name__}',flush=True);continue
  for race in range(3):
   d=[z for z in data if z[0]==race]
   if not d:continue
   labels[race].update(y for _,_,y in d)
   for start in range(0,len(d),a.batch_size):
    b=d[start:start+a.batch_size];x=torch.tensor([q for _,q,_ in b],dtype=torch.float32,device=dev);r=torch.full((len(b),),race,device=dev,dtype=torch.long);y=torch.tensor([q for _,_,q in b],device=dev)
    w=torch.tensor([max(seen[race],1)/(10*max(labels[race][j],1)) for j in range(10)],device=dev).clamp(max=15);o=model(x,r);loss=F.cross_entropy(o,y,weight=w);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();seen[race]+=len(b);correct[race]+=o.argmax(-1).eq(y).sum().item()
  if gi%25==0:print(f'games={gi} '+ ' | '.join(f'{RACES[r]}={seen[r]} decisions, acc={correct[r]/max(seen[r],1):.3f}, labels={dict(labels[r])}' for r in range(3)),flush=True)
  if a.checkpoint_every and gi%a.checkpoint_every==0:save_checkpoint(gi);print(f'checkpoint={a.output} games={gi}',flush=True)
 save_checkpoint(len(files));print(f'saved={a.output} decisions_by_race={dict(zip(RACES,seen))}',flush=True)
if __name__=='__main__':main()
