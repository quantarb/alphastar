"""On-demand exact-tuple extraction with the same entity snapshot as live play."""
from __future__ import annotations

from collections import Counter, defaultdict
import zlib
import sc2reader

from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS
from mac_sc2.contracts.patch_race_mtl import task_key, tuple_record
from mac_sc2.data.semantic_replay import cat, vec
from mac_sc2.contracts.semantic_schema import from_event

TOWN_HALL = ("commandcenter", "orbitalcommand", "planetaryfortress", "nexus", "hatchery", "lair", "hive")

def event_owner(event):
    """Command ownership is the replay player pid, never a zero-based pid."""
    player = getattr(event, "player", None)
    return getattr(player, "pid", None)

def _id(unit): return int(getattr(unit, "id", getattr(unit, "unit_id", 0)) or 0)
def _name(unit): return str(getattr(unit, "type_name", getattr(unit, "name", unit))).split(" [", 1)[0]
def _point(unit):
    point = getattr(unit, "location", None)
    return (float(point[0]), float(point[1])) if point else None

def _snapshot(units, pid, home):
    rows, ids = [], []
    for key, (unit, owner, position) in sorted(units.items())[:ENTITY_SLOTS]:
        if owner != pid: continue
        name = _name(unit).lower(); health=float(getattr(unit, "health", 0) or 0); maximum=float(getattr(unit, "health_max", 0) or 0)
        rows.append((zlib.crc32(_name(unit).encode()) % 65535 / 65535, (position[0]-home[0])/64, (position[1]-home[1])/64, 1., health/max(maximum,1), float(getattr(unit,"build_progress",1) or 1), float(getattr(unit,"is_flying",False)), float(any(word in name for word in ("scv","probe","drone")))))
        ids.append(key)
        if len(rows) == ENTITY_SLOTS: break
    return rows, ids

def registry_indices(specs):
    result = {}
    for task, vocab in specs.items():
        result[task] = {tuple(sorted(row.items())): index for index, row in enumerate(vocab)}
    return result

def examples(path, version, specs, stats=None):
    replay=sc2reader.load_replay(path,load_level=4); races={p.pid:p.play_race for p in replay.players}
    latest={}; counts=defaultdict(lambda:[0]*8); selected=defaultdict(list); groups=defaultdict(dict); units={}; homes={}; indices=registry_indices(specs); discarded=Counter()
    for event in replay.events:
        pid=event_owner(event); typ=type(event).__name__
        if typ in ("UnitBornEvent","UnitInitEvent","UnitDoneEvent","UnitTypeChangeEvent"):
            unit=getattr(event,"unit",None); owner=getattr(getattr(unit,"owner",None),"pid",None); position=_point(unit)
            if unit and owner and position:
                units[_id(unit)]=(unit,owner,position)
                if owner not in homes and any(word in _name(unit).lower() for word in TOWN_HALL): homes[owner]=position
            if owner in races: counts[owner]=[a+b for a,b in zip(counts[owner],cat(getattr(event,"unit_type_name",_name(unit) if unit else "")))]
            continue
        if typ=="UnitDiedEvent": units.pop(int(getattr(event,"unit_id",0) or 0),None); continue
        if typ=="UnitPositionsEvent":
            for unit,position in getattr(event,"units",{}).items():
                if _id(unit) in units: units[_id(unit)]=(units[_id(unit)][0],units[_id(unit)][1],(float(position[0]),float(position[1])))
            continue
        if pid not in races: continue
        if typ=="PlayerStatsEvent": latest[pid]=event; continue
        if typ=="SelectionEvent": selected[pid]=[_id(unit) for unit in (getattr(event,"objects",[]) or []) if _id(unit)]; continue
        if "ControlGroupEvent" in typ:
            group=getattr(event,"control_group",0)
            if typ=="SetControlGroupEvent": groups[pid][group]=list(selected[pid])
            elif typ=="AddToControlGroupEvent": groups[pid][group]=list(dict.fromkeys(groups[pid].get(group,[])+selected[pid]))
            elif typ=="GetControlGroupEvent" and groups[pid].get(group): selected[pid]=groups[pid][group]
            continue
        if "CommandEvent" not in typ or pid not in latest or pid not in homes:
            continue
        task=task_key(version,races[pid])
        if task not in indices: discarded["no_live_task"]+=1; continue
        action=from_event(event,".".join(version.split(".")[:3]),races[pid],[str(units[x][0]) for x in selected[pid] if x in units])
        candidate={"actor":action.actor_role,"target_kind":action.target_kind,"target_type":action.target_name,"queue":action.queued,"payload":action.payload_role,"family":action.family,"replay_ability":action.ability_name}
        # Match the registry entry including target type and queue; ability is resolved only by registry.
        label=next((i for i,row in enumerate(specs[task]) if all(row[k]==v for k,v in candidate.items())),None)
        if label is None: discarded["ambiguous_or_unexecutable"]+=1; continue
        snapshot, ids=_snapshot(units,pid,homes[pid])
        if not snapshot: discarded["empty_snapshot"]+=1; continue
        target=_id(getattr(event,"target",None)); actor=next((ids.index(x) for x in selected[pid] if x in ids),None)
        target_index=ids.index(target) if target in ids else -1
        location=getattr(event,"location",None)
        yield {"task":task,"state":vec(latest[pid],counts[pid],getattr(event,"second",0)),"snapshot":snapshot,"tuple_id":label,"actor":-1 if actor is None else actor,"target":target_index,"location":None if not location else ((float(location[0])-homes[pid][0])/64,(float(location[1])-homes[pid][1])/64)}
    if stats is not None: stats.update(discarded)
    return discarded
