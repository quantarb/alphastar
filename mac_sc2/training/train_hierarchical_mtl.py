#!/usr/bin/env python3
"""Memory-bounded trainer for compact hierarchical MTL replay shards."""
import argparse, json
from pathlib import Path
import numpy as np, torch
from torch.nn import functional as F
from mac_sc2.legacy.hierarchical_mtl_policy import HierarchicalMTLPolicy

def tensors(part, device):
    n=len(part['label']); units=torch.from_numpy(part['units'].astype(np.int64)); frame=torch.from_numpy(part['frame'].astype(np.float32)/65535); hist=torch.from_numpy(part['history'].astype(np.float32)/6)
    entities=torch.zeros(n,64,24); entities[:,:32,0]=units.float()/65535; entities[:,:32,1]=torch.arange(32).view(1,-1)/31; entities[:,:32,2]=frame[:,None]
    entities[:,:32,3:11]=hist[:,None,:]; mask=torch.ones(n,64,dtype=torch.bool); mask[:,:32]=units.eq(0); mask[:,0]=False
    state=torch.zeros(n,16,24); state[:,:,0]=frame[:,None]; state[:,:,1:9]=hist[:,None,:]
    return entities.to(device),mask.to(device),state.to(device),torch.from_numpy(part['label'].astype(np.int64)).to(device)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--shards',required=True); p.add_argument('--output',required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch-size',type=int,default=64); p.add_argument('--max-shards',type=int); a=p.parse_args()
    root=Path(a.shards); manifest_path=root/'manifest.json'; manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'games': 'partial', 'examples': 'partial', 'shards': 'partial'}; device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu'); model=HierarchicalMTLPolicy().to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=.02)
    for epoch in range(a.epochs):
        seen=correct=0
        paths=sorted(root.glob('shard_*.npz'))[:a.max_shards]
        for path in paths:
            raw=np.load(path)
            for race in range(3):
                idx=np.flatnonzero(raw['race']==race)
                if not len(idx): continue
                order=np.random.permutation(idx)
                for start in range(0,len(order),a.batch_size):
                    part={k:raw[k][order[start:start+a.batch_size]] for k in raw.files}; ent,mask,hist,y=tensors(part,device); ids=torch.full((len(y),),race,device=device,dtype=torch.long)
                    logits=model(ent,mask,hist,ids)['macro']; loss=F.cross_entropy(logits,y); opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
                    seen+=len(y);correct+=logits.argmax(-1).eq(y).sum().item()
            print(f'epoch={epoch+1} shard={path.name} examples={seen} accuracy={correct/seen:.3f}',flush=True)
    Path(a.output).parent.mkdir(parents=True,exist_ok=True); torch.save({'state_dict':model.cpu().state_dict(), **manifest, 'architecture':'390k shared entity Transformer + temporal GRU with component action heads'},a.output)
if __name__=='__main__': main()
