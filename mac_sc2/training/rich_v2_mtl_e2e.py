"""Bounded all-three-race V2 MTL smoke trainer."""
from __future__ import annotations
import argparse, json
from multiprocessing import Process
from collections import Counter, defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from mac_sc2.architectures.historical_build_tech_mtl import HistoricalBuildTechMTL
from mac_sc2.architectures.historical_tactics_mtl import HistoricalTacticsMTL
from mac_sc2.architectures.historical_replay_adapter import HistoricalReplayInputAdapter
from mac_sc2.architectures.rich_transformer import RACES, RichEntityTransformerPolicy
from mac_sc2.contracts.historical_build_tech import build_task_vocabs
from mac_sc2.contracts.historical_tactics import build_task_vocabs as build_tactics_vocabs
from mac_sc2.contracts.rich_transformer_action import contract_hash
from mac_sc2.contracts.rich_transformer_snapshot import ENTITY_SLOTS, snapshot_hash
from mac_sc2.data.rich_v2_labels import examples
from mac_sc2.data.historical_build_tech import examples as historical_examples
from mac_sc2.data.historical_tactics import examples as historical_tactics_examples
from mac_sc2.training.historical_build_tech_mtl import _batch as historical_batch
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
        with gzip.open(cache,'rt') as stream:
            header=json.loads(next(stream))
            # Empty player caches carry no trainable labels.  In particular,
            # some one-player replays still have an empty player_2 header;
            # do not attempt to infer a race for a non-existent trajectory.
            if next(stream, None) is None:
                continue
        replay=sc2reader.load_replay(replay_paths[header['replay']],load_level=2)
        player=next((item for item in replay.players if item.pid == int(header['player'])),None)
        race = player.play_race if player is not None else None
        # Resolve Random from the actual selected replay race.  If that is not
        # present, exclude it rather than assigning an invalid legal decoder.
        if race in RACES:
            sources.append((race,str(cache)))
    return sources

def historical_sources_from_manifests(manifests: list[str]) -> list[tuple[str, str]]:
    """Return every raw replay with its recorded game version for offline MTL."""
    sources=[]
    for manifest in manifests:
        body=json.loads(Path(manifest).read_text())
        for row in body.get('rows',body.get('valid',[])):
            path=row.get('path')
            version=row.get('version')
            if path and version:
                # The historical registry is indexed by major.minor.patch,
                # while manifests retain the client build suffix.
                patch='.'.join(str(version).split('.')[:3])
                sources.append((patch,str(path)))
    return sources

def train(sources: list[tuple[str,str]], output: str, max_labels: int = 64, resume: str | None = None,
          epochs: int = 1, historical_sources: list[tuple[str, str]] = (),
          historical_registry: str = 'mac_sc2/artifacts/historical_action_registry.json',
          historical_max_labels: int = 0, historical_tactics_sources: list[tuple[str, str]] = (),
          checkpoint_every_games: int = 200):
    if checkpoint_every_games <= 0 or checkpoint_every_games > 200:
        raise ValueError('checkpoint_every_games must be in 1..200 for playable-agent recovery')
    validate_live_contract()
    for race in ('Protoss','Zerg'): validate_race_live_contract(race)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    architecture={'width':96,'layers':2,'heads':6}
    prior_epochs=0
    if resume:
        prior=torch.load(resume,map_location='cpu',weights_only=False)
        if prior.get('architecture_name')!='RichEntityTransformerPolicy' or prior.get('action_contract_hash')!=contract_hash() or prior.get('entity_snapshot_hash')!=snapshot_hash() or prior.get('architecture')!=architecture:
            raise RuntimeError('incompatible V2 MTL initializer')
        model=RichEntityTransformerPolicy(**architecture); model.load_state_dict(prior['state_dict'])
        prior_epochs=int(prior.get('epochs',0))
    else:
        model=RichEntityTransformerPolicy(**architecture)
    model=model.to(device)
    prepared=[]; discarded=Counter(); counts=Counter()
    for race,cache in sources:
        rows=list(examples(cache,race,discarded)); rows=rows[:max_labels] if max_labels else rows
        if not rows:
            discarded['source_no_usable_labels']+=1
            continue
        prepared.append((race,rows)); counts[race]+=len(rows)
    if any(race not in counts for race in RACES):
        raise RuntimeError(f'MTL needs usable tick labels for all races, got {dict(counts)}')
    # Historical build/tech order is an auxiliary representation task.  It
    # shares the exact live V2 trunk but owns source adapters/vocab heads; those
    # heads are intentionally omitted from the runnable checkpoint below.
    historical_counts=Counter(); historical_discarded=Counter(); historical_vocabs={}; auxiliary=None
    historical_input_adapter = (HistoricalReplayInputAdapter(model.width).to(device)
                                if historical_sources or historical_tactics_sources else None)
    if historical_sources:
        historical_vocabs=build_task_vocabs(historical_registry)
        auxiliary=HistoricalBuildTechMTL(model,historical_vocabs,historical_input_adapter).to(device)
    def historical_batches():
        """Stream raw replay labels so all historical patches fit in memory."""
        buffers=defaultdict(list)
        for source_index,(patch,replay_path) in enumerate(historical_sources,1):
            if source_index == 1 or source_index % 100 == 0:
                print(json.dumps({'historical_replays_processed':source_index-1,
                                  'historical_replays_total':len(historical_sources),
                                  'historical_build_tech_labels':sum(historical_counts.values())}),flush=True)
            emitted=0
            try:
                for row in historical_examples(replay_path,patch,historical_vocabs,historical_discarded):
                    if historical_max_labels and emitted >= historical_max_labels: break
                    emitted+=1; historical_counts[row['task']]+=1
                    buffers[row['task']].append(row)
                    if len(buffers[row['task']]) == 32:
                        yield row['task'],buffers[row['task']]; buffers[row['task']]=[]
            except Exception:
                historical_discarded['replay_parse_error']+=1
            if source_index % checkpoint_every_games == 0:
                # This sentinel is emitted only after all labels from the replay
                # have been consumed, making the overwrite checkpoint a valid
                # playable policy snapshot at a known recovery boundary.
                yield '__checkpoint__', []
        for task,rows in buffers.items():
            if rows: yield task,rows
    tactics_counts=Counter(); tactics_discarded=Counter(); tactics_vocabs={}; tactics_auxiliary=None
    if historical_tactics_sources:
        tactics_vocabs=build_tactics_vocabs(historical_registry)
        tactics_auxiliary=HistoricalTacticsMTL(model,tactics_vocabs,historical_input_adapter).to(device)
    def tactics_batches():
        buffers=defaultdict(list)
        for source_index,(patch,replay_path) in enumerate(historical_tactics_sources,1):
            if source_index == 1 or source_index % 100 == 0:
                print(json.dumps({'historical_tactics_replays_processed':source_index-1,
                                  'historical_tactics_replays_total':len(historical_tactics_sources),
                                  'historical_tactics_labels':sum(tactics_counts.values())}),flush=True)
            try:
                for row in historical_tactics_examples(replay_path,patch,tactics_vocabs,tactics_discarded):
                    tactics_counts[row['task']]+=1; buffers[row['task']].append(row)
                    if len(buffers[row['task']]) == 32:
                        yield row['task'],buffers[row['task']]; buffers[row['task']]=[]
            except Exception:
                tactics_discarded['replay_parse_error']+=1
            if source_index % checkpoint_every_games == 0:
                yield '__checkpoint__', []
        for task,rows in buffers.items():
            if rows: yield task,rows
    parameters=[]; seen=set()
    for module in (model,auxiliary,tactics_auxiliary):
        if module is not None:
            for parameter in module.parameters():
                if id(parameter) not in seen: parameters.append(parameter);seen.add(id(parameter))
    opt=torch.optim.AdamW(parameters,lr=2e-4)
    p=Path(output);p.parent.mkdir(parents=True,exist_ok=True); evaluation=None
    def save(epoch, *, completed: bool, recovery_phase: str | None = None):
        torch.save({'architecture_name':'RichEntityTransformerPolicy','architecture':architecture,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'games':len({Path(cache).parent.name for _,cache in sources}),'trajectories':len(sources),'epochs':prior_epochs+epoch if completed else prior_epochs,'epochs_this_run':epoch if completed else 0,'training_status':'complete_epoch' if completed else 'in_progress','recovery_phase':recovery_phase,'checkpoint_every_games':checkpoint_every_games,'resumed_from':str(Path(resume).resolve()) if resume else None,'initialization':'fine_tune_v2_mtl' if resume else 'from_scratch_explicitly_authorized_v2_mtl_e2e','action_contract_hash':contract_hash(),'entity_snapshot_hash':snapshot_hash(),'labels':dict(counts),'discarded':dict(discarded),'source_caches':sources,'historical_build_tech_auxiliary':{'research_only':True,'task_labels':dict(historical_counts),'discarded':dict(historical_discarded),'heads_saved':False} if auxiliary else None,'historical_tactics_auxiliary':{'research_only':True,'task_labels':dict(tactics_counts),'discarded':dict(tactics_discarded),'heads_saved':False} if tactics_auxiliary else None},p)
    for epoch in range(1,epochs+1):
      first_recovery_evaluation_started = False
      for race,rows in prepared:
        for start in range(0,len(rows),32):
            b=rows[start:start+32]; n=len(b); scalar=torch.tensor([x['scalar'] for x in b],dtype=torch.float32,device=device)
            entity=torch.zeros(n,ENTITY_SLOTS,13,device=device); pad=torch.ones(n,ENTITY_SLOTS,dtype=torch.bool,device=device)
            for i,x in enumerate(b): entity[i,:len(x['entities'])]=torch.tensor(x['entities'],dtype=torch.float32,device=device);pad[i,:len(x['entities'])]=False
            values={k:torch.tensor([x[k] for x in b],device=device) for k in ('intent','actor','target','region','queued')}
            out=model(scalar,entity,pad,race=torch.full((n,),RACES.index(race),device=device),target_mmr=torch.tensor([[x['mmr']] for x in b],dtype=torch.float32,device=device),intent=values['intent'],actor=values['actor'],target=values['target'],region=values['region'])
            loss=sum(F.cross_entropy(getattr(out,k),values[k]) for k in values);opt.zero_grad();loss.backward();opt.step()
      # Alternating auxiliary batches update the shared V2 entity, temporal,
      # and decoder transformers.  Build/tech heads remain offline-only.
      if auxiliary:
        for task,rows in historical_batches():
          if task == '__checkpoint__':
            save(epoch,completed=False,recovery_phase='historical_build_tech')
            print(json.dumps({'checkpoint':str(p.resolve()),'checkpoint_status':'in_progress','recovery_phase':'historical_build_tech'}),flush=True)
            if not first_recovery_evaluation_started:
              from mac_sc2.evaluation.rich_transformer_match import run_match
              evaluation=Process(target=run_match,args=(str(p.with_suffix('.first_checkpoint_eval.SC2Replay')),'veryeasy',str(p),128)); evaluation.start()
              first_recovery_evaluation_started=True
            continue
          scalar,entity,pad,history,label=historical_batch(rows,device)
          mmr=torch.tensor([[row.get('mmr',0)] for row in rows],dtype=torch.float32,device=device)
          goal_logits,region_logits=auxiliary.logits(scalar,entity,pad,history,task,mmr)
          regions=torch.tensor([row['region'] for row in rows],device=device)
          loss=F.cross_entropy(goal_logits,label)+F.cross_entropy(region_logits,regions)
          opt.zero_grad();loss.backward();opt.step()
        if not historical_counts: raise RuntimeError('no historical build/tech labels from supplied replay sources')
      if tactics_auxiliary:
        for task,rows in tactics_batches():
          if task == '__checkpoint__':
            save(epoch,completed=False,recovery_phase='historical_tactics')
            if not first_recovery_evaluation_started:
              from mac_sc2.evaluation.rich_transformer_match import run_match
              evaluation=Process(target=run_match,args=(str(p.with_suffix('.first_checkpoint_eval.SC2Replay')),'veryeasy',str(p),128)); evaluation.start()
              first_recovery_evaluation_started=True
            continue
          converted=[{**row,'goal':row['action']} for row in rows]
          scalar,entity,pad,history,action=historical_batch(converted,device)
          mmr=torch.tensor([[row.get('mmr',0)] for row in rows],dtype=torch.float32,device=device)
          action_logits,region_logits=tactics_auxiliary.logits(scalar,entity,pad,history,task,mmr)
          regions=torch.tensor([row['region'] for row in rows],device=device)
          loss=F.cross_entropy(action_logits,action)+F.cross_entropy(region_logits,regions)
          opt.zero_grad();loss.backward();opt.step()
        if not tactics_counts: raise RuntimeError('no historical tactics labels from supplied replay sources')
      save(epoch,completed=True)
      if epoch==1 and not first_recovery_evaluation_started:
          from mac_sc2.evaluation.rich_transformer_match import run_match
          evaluation=Process(target=run_match,args=(str(p.with_suffix('.epoch1_eval.SC2Replay')),'veryeasy',str(p),128)); evaluation.start()
    if evaluation: evaluation.join()
    return {'checkpoint':str(p.resolve()),'labels':dict(counts),'discarded':dict(discarded),'epochs':prior_epochs+epochs,'epochs_this_run':epochs,
            'historical_build_tech_labels':dict(historical_counts),'historical_tactics_labels':dict(tactics_counts)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',action='append');p.add_argument('--cache-dir');p.add_argument('--manifest',action='append',default=[]);p.add_argument('--output',required=True);p.add_argument('--max-labels',type=int,default=0,help='0 uses every usable label');p.add_argument('--resume');p.add_argument('--epochs',type=int,default=1);p.add_argument('--historical-source',action='append',default=[],metavar='PATCH=REPLAY');p.add_argument('--historical-manifest',action='append',default=[]);p.add_argument('--historical-tactics-source',action='append',default=[],metavar='PATCH=REPLAY');p.add_argument('--historical-tactics-manifest',action='append',default=[]);p.add_argument('--historical-registry',default='mac_sc2/artifacts/historical_action_registry.json');p.add_argument('--historical-max-labels',type=int,default=0);p.add_argument('--checkpoint-every-games',type=int,default=200);a=p.parse_args();sources=[tuple(x.split('=',1)) for x in a.source] if a.source else sources_from_cache_dir(a.cache_dir,a.manifest);historical=[tuple(x.split('=',1)) for x in a.historical_source]+historical_sources_from_manifests(a.historical_manifest);tactics=[tuple(x.split('=',1)) for x in a.historical_tactics_source]+historical_sources_from_manifests(a.historical_tactics_manifest);print(json.dumps(train(sources,a.output,a.max_labels,a.resume,a.epochs,historical,a.historical_registry,a.historical_max_labels,tactics,a.checkpoint_every_games),indent=2))
