"""Raw-replay repair pointer labels on the shared last-known entity snapshot."""
import zlib
import sc2reader
from mac_sc2.contracts.entity_snapshot import ENTITY_SLOTS
from mac_sc2.data.events import event_pid

def _id(unit): return int(getattr(unit, 'id', getattr(unit, 'unit_id', 0)) or 0)
def _name(unit): return str(getattr(unit, 'type_name', getattr(unit, 'name', unit))).split(' [',1)[0]
def _point(unit):
    p=getattr(unit,'location',None)
    return (float(p[0]),float(p[1])) if p else None
def _unit_row(unit, owner, pos, home):
    health=float(getattr(unit,'health',0) or 0); maximum=float(getattr(unit,'health_max',0) or 0)
    build=float(getattr(unit,'build_progress',1) or 1)
    name=_name(unit).lower()
    return (zlib.crc32(_name(unit).encode())%65535/65535,(pos[0]-home[0])/64,(pos[1]-home[1])/64,1.0,health/maximum if maximum else 1.0,build,float(getattr(unit,'is_flying',False)),float(any(x in name for x in ('scv','probe','drone'))))

def examples(path):
    replay=sc2reader.load_replay(path,load_level=4); races={p.pid:p.play_race for p in replay.players}
    units={}; homes={}; selected={}
    for event in replay.events:
        pid=event_pid(event); typ=type(event).__name__
        if typ in ('UnitBornEvent','UnitInitEvent','UnitDoneEvent','UnitTypeChangeEvent'):
            u=getattr(event,'unit',None); owner=getattr(getattr(u,'owner',None),'pid',None); pos=_point(u)
            if u and owner and pos:
                units[_id(u)]=(u,owner,pos)
                if owner not in homes and any(x in _name(u).lower() for x in ('commandcenter','orbitalcommand','planetaryfortress','nexus','hatchery','lair','hive')): homes[owner]=pos
            continue
        if typ=='UnitDiedEvent': units.pop(int(getattr(event,'unit_id',0) or 0),None); continue
        if typ=='UnitPositionsEvent':
            for u,pos in getattr(event,'units',{}).items():
                key=_id(u)
                if key in units: units[key]=(units[key][0],units[key][1],(float(pos[0]),float(pos[1])))
            continue
        if typ=='SelectionEvent' and pid in races:
            selected[pid]=[_id(u) for u in (getattr(event,'objects',[]) or []) if _id(u)]
            continue
        if 'CommandEvent' not in typ or pid not in races or races[pid]!='Terran' or getattr(event,'ability_name','')!='SCVRepair': continue
        target=getattr(event,'target',None); target_id=_id(target); home=homes.get(pid)
        if not home or not target_id or target_id not in units: continue
        own=[(k,v) for k,v in sorted(units.items()) if v[1]==pid][:ENTITY_SLOTS]; ids=[k for k,_ in own]
        target_index=ids.index(target_id) if target_id in ids else None
        actor_index=next((ids.index(k) for k in selected.get(pid,[]) if k in ids),None)
        if actor_index is None or target_index is None: continue
        yield [_unit_row(u,owner,pos,home) for _,(u,owner,pos) in own], actor_index, target_index
