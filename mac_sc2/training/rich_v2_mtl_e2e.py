"""Bounded all-three-race V2 MTL smoke trainer."""
from __future__ import annotations
import argparse, json
from multiprocessing import Process
from collections import Counter
from pathlib import Path
import torch
from torch.nn import functional as F
from mac_sc2.architectures.rich_transformer import RACES, RichEntityTransformerPolicy
from mac_sc2.contracts.rich_transformer_action import contract_hash
from mac_sc2.contracts.rich_transformer_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.data.rich_v2_labels import examples
from mac_sc2.runtime.race_rich_executor import validate_race_live_contract
from mac_sc2.runtime.terran_entity_ar_bot import validate_live_contract

def sources_from_cache_dir(cache_dir: str, manifests: list[str]) -> list[tuple[str,str]]:
    """Resolve completed player caches to replay-declared race names."""
    import gzip
    import sc2reader
    replay_paths={}
    for manifest in manifests:
        body=json.loads(Path(manifest).read_text())
        for row in body.get('rows',body.get('valid',[])):
            replay_paths[Path(row['path']).name]=row['path']
    sources=[]
    for cache in sorted(Path(cache_dir).glob('*/player_*.compact.jsonl.gz')):
        with gzip.open(cache,'rt') as stream: header=json.loads(next(stream))
        replay=sc2reader.load_replay(replay_paths[header['replay']],load_level=1)
        sources.append((replay.attributes[int(header['player'])]['Race'],str(cache)))
    return sources

def train(sources: list[tuple[str,str]], output: str, max_labels: int = 64, resume: str | None = None,
          epochs: int = 1):
    validate_live_contract()
    for race in ('Protoss','Zerg'): validate_race_live_contract(race)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    architecture={'width':96,'layers':2,'heads':6}
    if resume:
        prior=torch.load(resume,map_location='cpu',weights_only=False)
        if prior.get('architecture_name')!='RichEntityTransformerPolicy' or prior.get('action_contract_hash')!=contract_hash() or prior.get('entity_snapshot_hash')!=snapshot_hash() or prior.get('architecture')!=architecture:
            raise RuntimeError('incompatible V2 MTL initializer')
        model=RichEntityTransformerPolicy(**architecture); model.load_state_dict(prior['state_dict'])
    else:
        model=RichEntityTransformerPolicy(**architecture)
    model=model.to(device); opt=torch.optim.AdamW(model.parameters(),lr=2e-4)
    prepared=[]; discarded=Counter(); counts=Counter()
    for race,cache in sources:
        rows=list(examples(cache,race,discarded)); rows=rows[:max_labels] if max_labels else rows
        if not rows: raise RuntimeError(f'no usable labels for {race}')
        prepared.append((race,rows)); counts[race]+=len(rows)
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True); evaluation=None
    def save(epoch):
        torch.save({'architecture_name':'RichEntityTransformerPolicy','architecture':architecture,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'games':len(sources),'epochs':epoch,'resumed_from':str(Path(resume).resolve()) if resume else None,'initialization':'fine_tune_v2_mtl' if resume else 'from_scratch_explicitly_authorized_v2_mtl_e2e','action_contract_hash':contract_hash(),'entity_snapshot_hash':snapshot_hash(),'labels':dict(counts),'discarded':dict(discarded),'source_caches':dict(sources)},p)
    for epoch in range(1,epochs+1):
      for race,rows in prepared:
        for start in range(0,len(rows),32):
            b=rows[start:start+32]; n=len(b); scalar=torch.tensor([x['scalar'] for x in b],dtype=torch.float32,device=device)
            entity=torch.zeros(n,ENTITY_SLOTS,13,device=device); pad=torch.ones(n,ENTITY_SLOTS,dtype=torch.bool,device=device)
            for i,x in enumerate(b): entity[i,:len(x['entities'])]=torch.tensor(x['entities'],dtype=torch.float32,device=device);pad[i,:len(x['entities'])]=False
            values={k:torch.tensor([x[k] for x in b],device=device) for k in ('intent','actor','target','region','queued')}
            out=model(scalar,entity,pad,race=torch.full((n,),RACES.index(race),device=device),target_mmr=torch.tensor([[x['mmr']] for x in b],dtype=torch.float32,device=device),intent=values['intent'],actor=values['actor'],target=values['target'],region=values['region'])
            loss=sum(F.cross_entropy(getattr(out,k),values[k]) for k in values);opt.zero_grad();loss.backward();opt.step()
      save(epoch)
      if epoch==1:
          from mac_sc2.evaluation.rich_transformer_match import run_match
          evaluation=Process(target=run_match,args=(str(p.with_suffix('.epoch1_eval.SC2Replay')),'veryeasy',str(p),128)); evaluation.start()
    if evaluation: evaluation.join()
    return {'checkpoint':str(p.resolve()),'labels':dict(counts),'discarded':dict(discarded),'epochs':epochs}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',action='append');p.add_argument('--cache-dir');p.add_argument('--manifest',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--max-labels',type=int,default=0,help='0 uses every usable label');p.add_argument('--resume');p.add_argument('--epochs',type=int,default=1);a=p.parse_args();sources=[tuple(x.split('=',1)) for x in a.source] if a.source else sources_from_cache_dir(a.cache_dir,a.manifest);print(json.dumps(train(sources,a.output,a.max_labels,a.resume,a.epochs),indent=2))
