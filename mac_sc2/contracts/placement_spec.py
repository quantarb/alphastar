"""Versioned contract for learned, legal SC2 placement actions."""
import hashlib, json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass(frozen=True)
class PlacementAction:
    race: str; ability: str; ability_id: int; actor_role: str; family: str

def actions(registry_path):
    raw=json.loads(Path(registry_path).read_text()); out=[]
    for task, rows in raw['tasks'].items():
        if not task.startswith('4.9.2:'): continue
        race=task.split(':',1)[1]
        for row in rows:
            live=row.get('live_4_9_2',{})
            if (live.get('status')=='resolved' and row['target_kind']=='point' and row['actor'] in ('worker','production')
                    and (row['family']=='build' or row['ability_name'].lower().startswith('land'))):
                out.append(PlacementAction(race,row['ability_name'],int(live['ability_id']),row['actor'],row['family']))
    return tuple(sorted(set(out),key=lambda x:(x.race,x.ability,x.ability_id,x.actor_role)))

def spec_hash(registry_path):
    body=[asdict(x) for x in actions(registry_path)]
    return hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]

def validate_checkpoint(data, registry_path):
    if data.get('placement_spec_hash') != spec_hash(registry_path): raise RuntimeError('Placement ActionSpec mismatch')
