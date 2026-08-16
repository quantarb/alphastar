#!/usr/bin/env python3
"""Single-pass MTL training: parse one raw replay, train, discard it."""
import argparse,json
from collections import deque
from pathlib import Path
import sc2reader,torch
from torch.nn import functional as F
from alphastar_sized_compact_policy import AlphaStarSizedCompactPolicy
R={'Terran':0,'Protoss':1,'Zerg':2}
def macro(n):
 n=(n or '').lower()
 if 'attack' in n or 'scanmove' in n:return 5
 if any(x in n for x in ('commandcenter','nexus','hatchery','expand')):return 4
 if any(x in n for x in ('supplydepot','pylon','overlord')):return 1
 if any(x in n for x in ('barracks','gateway','spawningpool','factory','cybernetics')):return 2
 if any(x in n for x in ('scv','probe','drone')):return 0
 if any(x in n for x in ('marine','zealot','zergling','roach','stalker','marauder','hellion')):return 3
 return 6
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--vocab',required=True);p.add_argument('--output',required=True);p.add_argument('--max-games',type=int,default=2000);p.add_argument('--batch-size',type=int,default=256);a=p.parse_args()
 v=json.loads(Path(a.vocab).read_text());lookup={int(k):x for k,x in v['ability_to_index'].items()};unk=v['unknown_index'];files=json.loads(Path(a.manifest).read_text())['valid'][:a.max_games];dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');m=AlphaStarSizedCompactPolicy().to(dev);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.02);seen=ok=0
 for gi,item in enumerate(files,1):
  try:
   r=sc2reader.load_replay(item['path'],load_level=4);races={x.pid:R.get(x.play_race) for x in r.players};sel={x:[] for x in races};hist={x:deque([unk]*8,maxlen=8) for x in races};rows=[]
   for e in r.events:
    pid=getattr(e,'pid',None)
    if pid not in races or races[pid] is None:continue
    if type(e).__name__=='SelectionEvent':sel[pid]=[int(z[0]) for z in getattr(e,'new_unit_types',[])][:32]
    if 'CommandEvent' not in type(e).__name__ or getattr(e,'ability_id',None) is None:continue
    if getattr(e,'ability_name','') == 'RightClick':continue
    rows.append((races[pid],sel[pid][:],list(hist[pid]),lookup.get(int(e.ability_id),unk),macro(getattr(e,'ability_name',''))));hist[pid].append(rows[-1][3])
   for race in range(3):
    rr=[z for z in rows if z[0]==race]
    for s in range(0,len(rr),a.batch_size):
     b=rr[s:s+a.batch_size];n=len(b);ent=torch.zeros(n,64,24);mask=torch.ones(n,64,dtype=torch.bool);h=torch.zeros(n,16,24)
     units=torch.tensor([z[1]+[0]*(32-len(z[1])) for z in b],dtype=torch.float32)
     histories=torch.tensor([z[2] for z in b],dtype=torch.float32)
     ent[:,:32,0]=units/65535;ent[:,:32,1]=torch.arange(32,dtype=torch.float32)[None,:]/31
     mask[:,:32]=units.eq(0);mask[:,0]=False;h[:,:,1:9]=histories[:,None,:]/1025
     y=torch.tensor([z[3] for z in b]);ym=torch.tensor([z[4] for z in b]);out=m(ent.to(dev),mask.to(dev),h.to(dev),torch.full((n,),race,device=dev,dtype=torch.long));loss=F.cross_entropy(out['ability'],y.to(dev))+.5*F.cross_entropy(out['macro'],ym.to(dev));o.zero_grad();loss.backward();o.step();seen+=n;ok+=out['ability'].argmax(-1).eq(y.to(dev)).sum().item()
  except Exception as e:print(f'skip={gi} {type(e).__name__}',flush=True)
  if gi%25==0:print(f'games={gi} examples={seen} ability_accuracy={ok/max(seen,1):.3f}',flush=True)
 torch.save({'state_dict':m.cpu().state_dict(),'games':len(files),'examples':seen,'architecture':'371k on-demand MTL macro+1026 ability'},a.output)
if __name__=='__main__':main()
