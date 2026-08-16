#!/usr/bin/env python3
"""Train the 1,026-way factorised ability head from canonical replay shards."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from torch.nn import functional as F
from mac_sc2.legacy.alphastar_sized_compact_policy import AlphaStarSizedCompactPolicy
def main():
 p=argparse.ArgumentParser();p.add_argument('--shards',required=True);p.add_argument('--output',required=True);p.add_argument('--batch-size',type=int,default=128);a=p.parse_args();root=Path(a.shards);meta=json.loads((root/'manifest.json').read_text());device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');m=AlphaStarSizedCompactPolicy().to(device);o=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=.02);seen=ok=0
 for f in sorted(root.glob('shard_*.npz')):
  x=np.load(f)
  for race in range(3):
   ids=np.flatnonzero(x['race']==race)
   for s in range(0,len(ids),a.batch_size):
    z=ids[s:s+a.batch_size];n=len(z);u=torch.from_numpy(x['units'][z].astype(np.int64));ent=torch.zeros(n,64,24);ent[:,:32,0]=u.float()/65535;ent[:,:32,1]=torch.arange(32)[None,:]/31;ent[:,:32,2]=torch.from_numpy(x['frame'][z].astype(np.float32)/65535)[:,None];mask=torch.ones(n,64,dtype=torch.bool);mask[:,:32]=u.eq(0);mask[:,0]=False;h=torch.zeros(n,16,24);h[:,:,1:9]=torch.from_numpy(x['history'][z].astype(np.float32)/1025)[:,None,:];y=torch.from_numpy(x['ability'][z].astype(np.int64));ym=torch.from_numpy(x['macro'][z].astype(np.int64));out=m(ent.to(device),mask.to(device),h.to(device),torch.full((n,),race,device=device,dtype=torch.long));loss=F.cross_entropy(out['ability'],y.to(device))+0.5*F.cross_entropy(out['macro'],ym.to(device));o.zero_grad();loss.backward();o.step();seen+=n;ok+=out['ability'].argmax(-1).eq(y.to(device)).sum().item()
  print(f'{f.name} examples={seen} accuracy={ok/seen:.3f}',flush=True)
 torch.save({'state_dict':m.cpu().state_dict(),**meta,'architecture':'370k factorised 1026-way ability policy'},a.output)
if __name__=='__main__':main()
