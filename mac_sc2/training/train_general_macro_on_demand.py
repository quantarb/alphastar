#!/usr/bin/env python3
"""Stateful, on-demand replay BC for economic, tech, and three army tiers."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import sc2reader,torch
from torch.nn import functional as F
from mac_sc2.legacy.general_macro_policy import GeneralMacroPolicy,RACES
RID={r:i for i,r in enumerate(RACES)}
WORDS=[('scv','probe','drone'),('supplydepot','pylon','overlord'),('barracks','gateway','spawningpool'),('refinery','assimilator','extractor'),('cybernetics','robotics','stargate','forge','engineeringbay','armory','factory','spire','hydraliskden','roachwarren'),('marine','zealot','zergling'),('stalker','adept','sentry','roach','hydralisk','marauder','hellion'),('immortal','colossus','disruptor','templar','carrier','voidray','phoenix','siegetank','medivac','mutalisk','lurker'),('commandcenter','nexus','hatchery','lair','hive')]
def has(n,words):return any(x in n for x in words)
def label(name):
 n=(name or '').lower()
 if 'attack' in n or 'scanmove' in n:return 9
 if has(n,WORDS[8]) or 'expand' in n:return 8
 if has(n,WORDS[1]):return 1
 if has(n,WORDS[3]):return 3
 if has(n,WORDS[4]):return 4
 if has(n,WORDS[2]):return 2
 if has(n,WORDS[0]):return 0
 if has(n,WORDS[7]):return 7
 if has(n,WORDS[6]):return 6
 if has(n,WORDS[5]):return 5
 return None
def cat(name):
 n=(name or '').lower();return [int(has(n,w)) for w in WORDS]
def vec(s,c,t):
 m=getattr(s,'minerals_current',0);g=getattr(s,'vespene_current',0);used=getattr(s,'food_used',0);made=getattr(s,'food_made',0);workers=max(getattr(s,'workers_active_count',0),c[0])
 return [min(t/900,1),min(m/1500,1),min(g/1000,1),min(used/200,1),min(made/200,1),min(max(made-used,0)/30,1),min(workers/80,1),* [min(x/20,1) for x in c[1:]],min(getattr(s,'minerals_collection_rate',0)/2500,1),min(getattr(s,'vespene_collection_rate',0)/1500,1),min(getattr(s,'resources_lost',0)/10000,1)]
def event_pid(event):
 """Return the replay player's one-based id across sc2reader event classes.

 Command events expose a zero-based ``pid`` but also their resolved ``player``;
 stats and unit events use the regular one-based id.  Always prefer the
 resolved player to avoid assigning an opponent's commands to this player's
 race or winner flag.
 """
 player=getattr(event,'player',None)
 return getattr(player,'pid',None) if player is not None else getattr(event,'pid',getattr(event,'control_pid',None))
def rows(path,window,winner_only=False):
 r=sc2reader.load_replay(path,load_level=4);race={p.pid:RID.get(p.play_race) for p in r.players};winners={p.pid for p in r.players if getattr(p,'result',None)=='Win'};latest={};counts=defaultdict(lambda:[0]*8);emitted=defaultdict(lambda:-1);out=[]
 for e in r.events:
  pid=event_pid(e)
  if pid not in race or race[pid] is None:continue
  typ=type(e).__name__
  if typ=='PlayerStatsEvent':latest[pid]=e;continue
  if typ in ('UnitBornEvent','UnitInitEvent'):
   q=cat(getattr(e,'unit_type_name',''));counts[pid]=[a+b for a,b in zip(counts[pid],q)];continue
  if 'CommandEvent' not in typ:continue
  y=label(getattr(e,'ability_name',''))
  if y is None or pid not in latest or (winner_only and pid not in winners):continue
  b=int(getattr(e,'second',0)//window)
  if b==emitted[pid]:continue
  emitted[pid]=b;out.append((race[pid],vec(latest[pid],counts[pid],getattr(e,'second',0)),y))
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--manifest',required=True);ap.add_argument('--output',required=True);ap.add_argument('--max-games',type=int,default=1000);ap.add_argument('--batch-size',type=int,default=512);ap.add_argument('--window',type=int,default=8);a=ap.parse_args();files=json.loads(Path(a.manifest).read_text())['valid'][:a.max_games];dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=GeneralMacroPolicy().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=8e-4,weight_decay=.01);labels=Counter();seen=ok=0
 for gi,item in enumerate(files,1):
  try:data=rows(item['path'],a.window)
  except Exception as e:print(f'skip game={gi} {type(e).__name__}',flush=True);continue
  labels.update(y for _,_,y in data)
  for i in range(0,len(data),a.batch_size):
   b=data[i:i+a.batch_size];x=torch.tensor([q for _,q,_ in b],dtype=torch.float32,device=dev);r=torch.tensor([q for q,_,_ in b],device=dev);y=torch.tensor([q for _,_,q in b],device=dev);w=torch.tensor([max(seen,1)/(10*max(labels[j],1)) for j in range(10)],device=dev).clamp(max=15);o=model(x,r);loss=F.cross_entropy(o,y,weight=w);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();seen+=len(b);ok+=o.argmax(-1).eq(y).sum().item()
  if gi%25==0:print(f'games={gi} decisions={seen} accuracy={ok/max(seen,1):.3f} labels={dict(labels)}',flush=True)
 Path(a.output).parent.mkdir(parents=True,exist_ok=True);torch.save({'state_dict':model.cpu().state_dict(),'games':a.max_games,'decisions':seen,'labels':dict(labels),'architecture':'stateful 10-action general macro BC'},a.output);print(f'saved={a.output} decisions={seen} labels={dict(labels)}')
if __name__=='__main__':main()
