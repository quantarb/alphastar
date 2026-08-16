#!/usr/bin/env python3
"""Train the executable SCV repair actor/target pointers directly from replays."""
import argparse,json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS,snapshot_hash
from mac_sc2.contracts.repair import action_hash
from mac_sc2.architectures.repair import RepairPolicy
from mac_sc2.data.repair_replay import examples

def pack(rows):
    x=torch.zeros(ENTITY_SLOTS,8);n=min(len(rows),ENTITY_SLOTS);x[:n]=torch.tensor(rows[:n]);m=torch.ones(ENTITY_SLOTS,dtype=torch.bool);m[:n]=False;return x,m
def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--max-games',type=int);p.add_argument('--checkpoint-every',type=int,default=200);a=p.parse_args()
    if not 0<a.checkpoint_every<=200: raise ValueError('checkpoint-every must be 1..200')
    files=[x for x in json.load(open(a.manifest))['valid'] if x['version'].startswith('4.9.2')];files=files[:a.max_games] if a.max_games else files
    dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');model=RepairPolicy().to(dev);opt=torch.optim.AdamW(model.parameters(),lr=3e-4);seen=0
    def save(game): torch.save({'state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'games':game,'decisions':seen,'repair_action_spec_hash':action_hash(),'entity_snapshot_hash':snapshot_hash(),'architecture':'raw replay SCVRepair actor + friendly-target pointers'},a.output)
    for game,item in enumerate(files,1):
        try: rows=list(examples(item['path']))
        except Exception as e: print('skip',game,type(e).__name__);continue
        for snap,actor,target in rows:
            x,m=pack(snap);al,tl=model(x[None].to(dev),m[None].to(dev));loss=F.cross_entropy(al,torch.tensor([actor],device=dev))+F.cross_entropy(tl,torch.tensor([target],device=dev));opt.zero_grad();loss.backward();opt.step();seen+=1
        if game%25==0: print(f'games={game} repair_decisions={seen}',flush=True)
        if game%a.checkpoint_every==0: save(game)
    save(len(files));print(f'saved={a.output} games={len(files)} repair_decisions={seen}')
if __name__=='__main__':main()
