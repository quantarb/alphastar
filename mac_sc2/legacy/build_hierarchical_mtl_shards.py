#!/usr/bin/env python3
"""Stream replay actions to compact MTL shards (never retain all rows in RAM)."""
import argparse, json
from collections import deque
from pathlib import Path
import numpy as np
import sc2reader

RACE = {"Terran": 0, "Protoss": 1, "Zerg": 2}
def label(name):
    n=(name or "").lower()
    if "attack" in n or "scanmove" in n: return 5
    if any(x in n for x in ("commandcenter","nexus","hatchery","expand")): return 4
    if any(x in n for x in ("supplydepot","pylon","overlord")): return 1
    if any(x in n for x in ("barracks","gateway","spawningpool","factory","cybernetics")): return 2
    if any(x in n for x in ("scv","probe","drone")): return 0
    if any(x in n for x in ("marine","zealot","zergling","roach","stalker","marauder","hellion")): return 3
    return 6

def rows(path):
    r=sc2reader.load_replay(str(path), load_level=4); races={p.pid:RACE.get(p.play_race) for p in r.players}
    selected={p:[] for p in races}; history={p:deque([6]*8,maxlen=8) for p in races}
    for e in r.events:
        pid=getattr(e,"pid",None)
        if pid not in races or races[pid] is None: continue
        if type(e).__name__=="SelectionEvent": selected[pid]=[int(x[0]) for x in getattr(e,"new_unit_types",[])][:32]
        if "CommandEvent" not in type(e).__name__: continue
        y=label(getattr(e,"ability_name", "")); units=np.zeros(32,np.uint16); units[:len(selected[pid])]=np.asarray(selected[pid],np.uint16)
        yield races[pid], min(int(e.frame),65535), units, np.asarray(history[pid],np.uint8), y
        history[pid].append(y)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--out',required=True); p.add_argument('--max-games',type=int); p.add_argument('--shard-size',type=int,default=100000); a=p.parse_args()
    valid=json.loads(Path(a.manifest).read_text())["valid"][:a.max_games]; out=Path(a.out); out.mkdir(parents=True,exist_ok=True); buf=[]; shard=0; total=0
    def flush():
        nonlocal shard, total, buf
        if not buf:return
        np.savez_compressed(out/f'shard_{shard:05d}.npz', race=np.asarray([x[0] for x in buf],np.uint8), frame=np.asarray([x[1] for x in buf],np.uint16), units=np.stack([x[2] for x in buf]), history=np.stack([x[3] for x in buf]), label=np.asarray([x[4] for x in buf],np.uint8))
        total+=len(buf); shard+=1; buf=[]; print(f'shards={shard} examples={total}',flush=True)
    for i,item in enumerate(valid,1):
        try:
            for row in rows(Path(item['path'])):
                buf.append(row)
                if len(buf)>=a.shard_size: flush()
        except Exception as e: print(f'skip={Path(item["path"]).name} {type(e).__name__}',flush=True)
        if i%100==0: print(f'games={i}',flush=True)
    flush(); (out/'manifest.json').write_text(json.dumps({'games':len(valid),'examples':total,'shards':shard,'format':'race:uint8 frame:uint16 units:[32]uint16 history:[8]uint8 label:uint8'}))
if __name__=='__main__': main()
