"""Fine-tune the expanded semantic MTL from its all-replay baseline."""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
import sc2reader, torch
from torch.nn import functional as F
from mac_sc2.architectures.semantic_action_mtl import SemanticActionMTL
from mac_sc2.contracts.semantic_action import BASELINE_SPEC_HASH, spec_hash, supports
from mac_sc2.contracts.semantic_schema import ACTOR_ROLES,FAMILIES,PAYLOAD_ROLES,TARGET_KINDS,from_event
from mac_sc2.data.patch_race_exact import cat,vec
RID={"Terran":0,"Protoss":1,"Zerg":2}; IDX={"actor":{x:i for i,x in enumerate(ACTOR_ROLES)},"family":{x:i for i,x in enumerate(FAMILIES)},"payload":{x:i for i,x in enumerate(PAYLOAD_ROLES)},"target":{x:i for i,x in enumerate(TARGET_KINDS)}}
def rows(path,version):
 r=sc2reader.load_replay(path,load_level=4); races={p.pid:p.play_race for p in r.players}; latest={}; counts=defaultdict(lambda:[0]*8); selected=defaultdict(list)
 for e in r.events:
  pid=getattr(getattr(e,"player",None),"pid",None); race=races.get(pid); typ=type(e).__name__
  if race not in RID: continue
  if typ=="PlayerStatsEvent":latest[pid]=e;continue
  if typ in ("UnitBornEvent","UnitInitEvent"):counts[pid]=[a+b for a,b in zip(counts[pid],cat(getattr(e,"unit_type_name","")))];continue
  if typ=="SelectionEvent":selected[pid]=[str(x) for x in getattr(e,"objects",[]) or []];continue
  if "CommandEvent" not in typ or pid not in latest:continue
  a=from_event(e,".".join(version.split(".")[:3]),race,selected[pid])
  if supports(a.actor_role,a.family,a.payload_role,a.target_kind):yield RID[race],vec(latest[pid],counts[pid],getattr(e,"second",0)),a
def fine_tune(manifest,baseline,output,games=None):
 d=torch.load(baseline,map_location="cpu",weights_only=False)
 source_hash=d.get("action_contract_hash"); start=d.get("games",0) if source_hash==spec_hash() else 0
 if source_hash not in (BASELINE_SPEC_HASH,spec_hash()):raise ValueError("semantic baseline contract mismatch")
 m=SemanticActionMTL();m.load_state_dict(d["state_dict"]);dev=torch.device("mps" if torch.backends.mps.is_available() else "cpu");m.to(dev);opt=torch.optim.AdamW(m.parameters(),lr=1e-4);n=Counter()
 items=json.loads(Path(manifest).read_text())["valid"]; total=len(items) if games is None else games
 if not start < total:raise ValueError(f"nothing to train: {start}/{total}")
 for game,item in enumerate(items[start:total],start+1):
  data=list(rows(item["path"],item["version"]))
  for start in range(0,len(data),512):
   batch=data[start:start+512];race=torch.tensor([x[0] for x in batch],device=dev);state=torch.tensor([x[1] for x in batch],dtype=torch.float32,device=dev);out=m(state,race)
   fields={"actor":"actor_role","family":"family","payload":"payload_role","target":"target_kind"};loss=sum(F.cross_entropy(out[k],torch.tensor([IDX[k][getattr(x[2],fields[k])] for x in batch],device=dev)) for k in fields)+F.cross_entropy(out["queued"],torch.tensor([int(x[2].queued) for x in batch],device=dev));opt.zero_grad();loss.backward();opt.step();n["labels"]+=len(batch)
  if game%200==0:Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":{k:v.detach().cpu() for k,v in m.state_dict().items()},"games":game,"resumed_from":str(Path(baseline).resolve()),"action_contract_hash":spec_hash(),"counts":dict(n)},output);print(f"games={game} labels={n['labels']}",flush=True)
 Path(output).parent.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":{k:v.detach().cpu() for k,v in m.state_dict().items()},"games":total,"resumed_from":str(Path(baseline).resolve()),"action_contract_hash":spec_hash(),"counts":dict(n)},output);return {"checkpoint":str(Path(output).resolve()),"counts":dict(n),"games":total}
