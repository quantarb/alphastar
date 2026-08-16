#!/usr/bin/env python3
"""Canonical replay shards for the 1,026-way ability policy (no macro labels)."""
import argparse,json
from collections import deque
from pathlib import Path
import numpy as np,sc2reader
R={'Terran':0,'Protoss':1,'Zerg':2}
def macro(name):
 n=(name or '').lower()
 if 'attack' in n or 'scanmove' in n:return 5
 if any(x in n for x in ('commandcenter','nexus','hatchery','expand')):return 4
 if any(x in n for x in ('supplydepot','pylon','overlord')):return 1
 if any(x in n for x in ('barracks','gateway','spawningpool','factory','cybernetics')):return 2
 if any(x in n for x in ('scv','probe','drone')):return 0
 if any(x in n for x in ('marine','zealot','zergling','roach','stalker','marauder','hellion')):return 3
 return 6
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--vocab',required=True);p.add_argument('--out',required=True);p.add_argument('--max-games',type=int);p.add_argument('--shard-size',type=int,default=100000);a=p.parse_args()
 vocab=json.loads(Path(a.vocab).read_text()); lookup={int(k):v for k,v in vocab['ability_to_index'].items()}; unknown=vocab['unknown_index']; files=json.loads(Path(a.manifest).read_text())['valid'][:a.max_games];out=Path(a.out);out.mkdir(parents=True,exist_ok=True);buf=[];shard=total=0
 def flush():
  nonlocal buf,shard,total
  if not buf:return
  np.savez_compressed(out/f'shard_{shard:05d}.npz',race=np.asarray([x[0] for x in buf],np.uint8),frame=np.asarray([x[1] for x in buf],np.uint16),units=np.stack([x[2] for x in buf]),history=np.stack([x[3] for x in buf]),ability=np.asarray([x[4] for x in buf],np.uint16),target=np.stack([x[5] for x in buf]),macro=np.asarray([x[6] for x in buf],np.uint8))
  total+=len(buf);shard+=1;buf=[];print(f'shards={shard} examples={total}',flush=True)
 for i,item in enumerate(files,1):
  try:
   replay=sc2reader.load_replay(item['path'],load_level=4);races={x.pid:R.get(x.play_race) for x in replay.players};selected={pid:[] for pid in races};hist={pid:deque([unknown]*8,maxlen=8) for pid in races}
   for e in replay.events:
    pid=getattr(e,'pid',None)
    if pid not in races or races[pid] is None:continue
    if type(e).__name__=='SelectionEvent':selected[pid]=[int(x[0]) for x in getattr(e,'new_unit_types',[])][:32]
    if 'CommandEvent' not in type(e).__name__ or getattr(e,'ability_id',None) is None:continue
    y=lookup.get(int(e.ability_id),unknown);u=np.zeros(32,np.uint16);u[:len(selected[pid])]=selected[pid];t=getattr(e,'target',None);point=np.array([getattr(t,'x',-1.),getattr(t,'y',-1.)],np.float16)
    buf.append((races[pid],min(int(e.frame),65535),u,np.asarray(hist[pid],np.uint16),y,point,macro(getattr(e,'ability_name',''))));hist[pid].append(y)
    if len(buf)>=a.shard_size:flush()
  except Exception as e:print(f'skip={Path(item["path"]).name} {type(e).__name__}',flush=True)
  if i%100==0:print(f'games={i}',flush=True)
 flush();(out/'manifest.json').write_text(json.dumps({'games':len(files),'examples':total,'shards':shard,'vocab':str(a.vocab),'task':'1026-way ability'}))
if __name__=='__main__':main()
